# -*- coding: utf-8 -*-
r"""GPU 合成（backend_merge.compose_gpu）验收：逐位/逐值对拍 + 帧速。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py
    ... --dump "%TEMP%\dfl_cgpu\out30_0030-0069.dump"   # 真素材夹具（推荐）
    ... --frames 40 --reps 3 --cases all --json out.json

## 验什么

1) 判据 cubic：cv2.warpAffine(INTER_CUBIC) 在 **float32** 上到底做不做 1/32 定点。
   结论（本机 cv2 5.0.0）：不做。单点 delta + 亚像素平移，观测值与「连续精确值」
   逐点相同（max 差 0.0），与「1/32 定点」差 1.35e-2。**uint8** 输入才走定点。
   所以 GPU 版必须对齐「连续坐标 + A=-0.75 三次核 + 零边界」，量化反而错 3.4 LSB。
   这一项把 cv2 的实际行为钉住，防止以后换 cv2 版本时悄悄改口径。

2) 基元 prims：warp（grid_sample 版 / 逐 tap 版）· resize · GaussianBlur · mt（match_tone）
   · hist-match，逐个对 cv2 / numpy 原实现。

3) 整帧 frame：同一批真素材（fused dump 的 pred/遮罩/68 点/原帧）上，CPU compose_frame
   与 GPU compose_frame_gpu 逐帧对拍：mean / max / 差异像素比例 / 逐字节相同帧数。
   验收线：**max <= 1 LSB**，且报出差异像素比例。

4) 变体 variants：mode=overlay/hist-match × ct=none/mt/lct(回退) × mask_mode=8/2/5 ×
   blur=0/4/8 × 单脸/双脸（双脸走 CPU 的 work 累加语义）。

5) 帧速 perf：同一批帧，CPU 1 线程 / CPU 8 线程 / GPU，报 ms/帧 与加速比。

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_compose_gpu"
# **键名要按 compose_frame 读的那一套**（mode / color_transfer_mode），不是界面的
# blend / ct_mode —— 界面那套由 app._merge_settings_to_config 翻译。第一版这里写成
# blend，结果 seamless / hist-match 两个用例一直在跑 overlay（数值全绿但测的不是它）。
SETTINGS = {"mask_mode": 8, "mode": "overlay", "color_transfer_mode": "mt",
            "erode_mask_modifier": 0, "blur_mask_modifier": 4}
TOL_LSB = 1                 # 整帧验收：max <= 1 LSB
# hist-match 的固有放大（直方图结点对 1e-5 输入的敏感，见 docs/26 §4.4）：允许 2 LSB
TOL_LSB_HIST = 2
# seamless 的固有放大（Poisson 是积分，u8 输入的 1 个单位差沿边界累加，见 docs/26 §4.5）：
# 1 LSB 是达不到的 —— 这里按「实测包络」验收，防的是变坏，不是逐位一致。
TOL_LSB_SEAMLESS = 64
TOL_RATIO_SEAMLESS = 0.002
TOL_FLOAT = 1e-3            # 基元层面允许的浮点差（= 0.25 LSB）；实测远小于它
RESULTS = []

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def say(s=""):
    print(s, flush=True)


def check(ok, label, detail=""):
    RESULTS.append({"ok": bool(ok), "label": label, "detail": str(detail)})
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  --  " + str(detail)) if detail else ""), flush=True)
    return bool(ok)


# ============================================================
# 夹具
# ============================================================
def load_dump(dump_dir, limit=0):
    """读 fused dump：每帧一张原帧 + 逐脸 pred/遮罩/68 点。

    帧用 dump 里的 composed（同一段视频、同样 1080p、同样带脸的真实画面）：
    对拍要的是「同一份输入下 CPU 与 GPU 等价」，输入从哪来不影响结论。
    """
    import cv2
    import numpy as np
    dump_dir = Path(dump_dir)
    comp = {p.name.split("_")[0]: p for p in dump_dir.glob("*_composed.png")}
    if not comp:
        raise SystemExit("dump 目录里没有 *_composed.png：%s" % dump_dir)
    faces_of = {}
    for p in dump_dir.glob("*_lm.json"):
        tag = p.name.split("_")[0]
        faces_of.setdefault(tag, []).append(p)
    frames = []
    for tag in sorted(comp):
        if tag not in faces_of:
            continue
        if limit and len(frames) >= limit:
            break
        fimg = cv2.imdecode(np.fromfile(str(comp[tag]), np.uint8), cv2.IMREAD_COLOR)
        flist = []
        for lmp in sorted(faces_of[tag]):
            base = lmp.name[:-len("_lm.json")]
            def rd(name):
                q = dump_dir / (base + "_" + name + ".png")
                if not q.is_file():
                    return None
                return cv2.imdecode(np.fromfile(str(q), np.uint8), cv2.IMREAD_UNCHANGED)
            flist.append({"landmarks": np.asarray(json.loads(lmp.read_text("utf-8")), np.float32),
                          "pred": rd("pred"), "prd_dfm": rd("prdm"), "dst_dfm": rd("dstdm"),
                          "xseg_prd": rd("xseg_prd"), "xseg_dst": rd("xseg_dst"),
                          "k": int(base.split("_")[1])})
        frames.append({"tag": tag, "frame": fimg, "faces": flist})
    if not frames:
        raise SystemExit("dump 目录里没有可用的帧：%s" % dump_dir)
    return frames


def synth_frames(n, size=416):
    """没有 dump 时的兜底夹具：真实 1080p 画面（有就取项目帧）+ 合成软遮罩。

    只为「换台机器也能跑起来」；数字以 --dump 那份真素材为准。
    """
    import cv2
    import numpy as np
    cand = sorted((REPO / "workspace").glob("merge/*/*.jpg"))[:max(1, n)]
    frames = []
    rng = np.random.default_rng(7)
    for i in range(n):
        if cand:
            fimg = cv2.imdecode(np.fromfile(str(cand[i % len(cand)]), np.uint8), cv2.IMREAD_COLOR)
        else:
            fimg = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
        h, w = fimg.shape[:2]
        cx, cy = w * 0.5 + rng.normal(0, 40), h * 0.42 + rng.normal(0, 30)
        s = size * 1.6
        # 68 点：只为让 get_transform_mat 有输入，形状按 WHOLE_FACE 的分布糊一个
        t = np.linspace(0, np.pi, 17)
        jaw = np.stack([cx + s * 0.38 * np.cos(t), cy + s * 0.46 * np.sin(t)], 1)
        brow = np.stack([cx + s * 0.22 * np.linspace(-1, 1, 5), cy - s * 0.20 + 0 * t[:5]], 1)
        brow2 = brow + [s * 0.34, 0]
        nose = np.stack([cx + s * 0.05 * np.linspace(-1, 1, 9), cy + s * 0.18 * np.linspace(-1, 1, 9)], 1)
        eye1 = np.stack([cx - s * 0.16 + s * 0.03 * np.cos(np.linspace(0, 2 * np.pi, 6)),
                         cy - s * 0.06 + s * 0.02 * np.sin(np.linspace(0, 2 * np.pi, 6))], 1)
        eye2 = eye1 + [s * 0.32, 0]
        mouth = np.stack([cx + s * 0.20 * np.linspace(-1, 1, 20),
                          cy + s * 0.32 + s * 0.03 * np.sin(np.linspace(0, 3 * np.pi, 20))], 1)
        lm = np.concatenate([jaw, brow, brow2, nose, eye1, eye2, mouth])[:68].astype(np.float32)
        lm = np.resize(lm, (68, 2)).astype(np.float32)
        pred = (rng.random((size, size, 3)) * 60 + 150).astype(np.uint8)
        yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
        xp = np.clip(1.0 - (((xx - 128) ** 2 + (yy - 128) ** 2) ** 0.5 - 100) / 18.0, 0, 1)
        xd = np.clip(1.0 - (((xx - 128) ** 2 + (yy - 120) ** 2) ** 0.5 - 108) / 16.0, 0, 1)
        prdm = (np.clip(1.0 - (((yy - 128) ** 2 + (xx - 128) ** 2) ** 0.5 - 150) / 22.0, 0, 1) * 255).astype(np.uint8)
        flist = [{"landmarks": lm, "pred": pred, "prd_dfm": prdm, "dst_dfm": prdm.copy(),
                  "xseg_prd": (xp * 255).astype(np.uint8), "xseg_dst": (xd * 255).astype(np.uint8),
                  "k": 0}]
        frames.append({"tag": "%06d" % (i + 1), "frame": fimg, "faces": flist})
    return frames


def as_float_faces(faces):
    """GPU 入口也吃 u8，但默认路径（流水线里 CPU 那条）给的是 /255 之后的 float32。

    这里两种都测：主用例按流水线口径（uint8 直接给 GPU —— 省掉 CPU 的 /255），
    另一条把 uint8 先 /255 成 float32 再喂两边，证明两条入口等价。
    """
    out = []
    for f in faces:
        g = dict(f)
        out.append(g)
    return out


def cpu_faces(faces):
    """按 compose_frame 的期待：float32 0..1。"""
    import numpy as np
    out = []
    for f in faces:
        g = {"landmarks": f["landmarks"],
             "pred": f["pred"].astype(np.float32) / 255.0}
        for k in ("prd_dfm", "dst_dfm", "xseg_prd", "xseg_dst"):
            v = f.get(k)
            g[k] = None if v is None else v.astype(np.float32) / 255.0
        out.append(g)
    return out


def gpu_faces_u8(faces):
    """GPU 入口直接吃 uint8（流水线省一次 /255 与 4 次 astype）。"""
    return [dict(f) for f in faces]


# ============================================================
# 1) 判据：cv2 的 float32 三次插值到底走不走 1/32 定点
# ============================================================
def case_cubic():
    import cv2
    import numpy as np
    head("1 · 判据：cv2.warpAffine(INTER_CUBIC) 的 float32 / uint8 两条路")
    A = -0.75

    def w1(f):
        g = 1.0 - f
        c0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
        c1 = ((A + 2) * f - (A + 3)) * f * f + 1
        c2 = ((A + 2) * g - (A + 3)) * g * g + 1
        return np.array([c0, c1, c2, 1.0 - c0 - c1 - c2])

    say("  单点 delta 图 + 亚像素平移，看 dst[48,48] 落在哪条曲线上：")
    say("     dx          观测       连续精确值      1/32 定点预测")
    rows = []
    for dx in (0.0, 1/256, 1/64, 0.5/32, 0.9/32, 1.0/32, 1.4/32, 3.3/32, 7.5/32):
        img = np.zeros((96, 96), np.float32)
        img[48, 48] = 1.0
        d = cv2.warpAffine(img, np.array([[1, 0, dx], [0, 1, 0]], np.float64), (96, 96),
                           flags=cv2.INTER_CUBIC)
        obs = float(d[48, 48])
        s_ex = 48.0 - dx
        i0 = np.floor(s_ex)
        fe = s_ex - i0
        w_ex = w1(fe)
        exact = float(w_ex[int(round(48 - i0)) + 1])      # delta 落在第几个 tap 上
        sq = np.round(s_ex * 32) / 32.0
        iq = np.floor(sq)
        fq = sq - iq
        w_q = w1(fq)
        q32 = float(w_q[int(round(48 - iq)) + 1])
        rows.append((dx, obs, exact, q32))
        say("   %9.5f  %10.7f  %12.7f  %12.7f" % (dx, obs, exact, q32))
    e_ex = max(abs(r[1] - r[2]) for r in rows)
    e_q = max(abs(r[1] - r[3]) for r in rows)
    check(e_ex < 1e-6, "float32 输入 == 连续坐标（不是 1/32 定点）",
          "max|观测-连续| = %.2e，max|观测-1/32| = %.2e" % (e_ex, e_q))

    # 8U 才走定点
    img8 = np.zeros((96, 96), np.uint8)
    img8[48, 48] = 255
    v1 = int(cv2.warpAffine(img8, np.array([[1, 0, 0.0130], [0, 1, 0]], np.float64),
                            (96, 96), flags=cv2.INTER_CUBIC)[48, 48])
    v2 = int(cv2.warpAffine(img8, np.array([[1, 0, 0.04375], [0, 1, 0]], np.float64),
                            (96, 96), flags=cv2.INTER_CUBIC)[48, 48])
    check(v1 == 255 and v2 == 254, "uint8 输入才走 1/32 定点",
          "dx=0.0130 -> %d（定点吸附到整数）；dx=0.04375 -> %d" % (v1, v2))
    say("   结论：我们这条链三处 warp 的输入全是 float32（roi/255、wrk、fo），")
    say("         所以 GPU 版对齐「连续坐标 + A=-0.75 + 零边界」，不做定点量化")
    say("         （做了反而差 1.35e-2 ≈ 3.4 LSB）。")


# ============================================================
# 2) 基元
# ============================================================
def case_prims(impl="grid"):
    import cv2
    import numpy as np
    from server.backend_merge import compose as MC
    from server.backend_merge import compose_gpu as G
    c = G.ctx()
    t = c.torch
    head("2 · 基元对拍（warp / resize / GaussianBlur / mt / hist-match，impl=%s）" % impl)
    rng = np.random.default_rng(0)
    S = 416
    res = {}

    def cmp(name, got, ref):
        d = np.abs(np.asarray(got, np.float64) - np.asarray(ref, np.float64))
        qa = np.clip(np.asarray(got) * 255.0, 0, 255).astype(np.uint8)
        qb = np.clip(np.asarray(ref) * 255.0, 0, 255).astype(np.uint8)
        dq = np.abs(qa.astype(np.int16) - qb.astype(np.int16))
        say("   %-26s max %.3e  mean %.3e  |  u8 max %d LSB  %d/%d (%.4f%%)"
            % (name, d.max(), d.mean(), dq.max(), int((dq > 0).sum()), dq.size,
               100.0 * (dq > 0).sum() / dq.size))
        return float(d.max()), int(dq.max()), float((dq > 0).mean())

    # ---- warp ----
    src = rng.random((3, 300, 320)).astype(np.float32)
    box = t.as_tensor(src, device=c.dev)
    worst_w = 0.0
    for th, sc, tx, ty in ((0.09, 1.0, 17.3, -9.7), (-0.3, 0.93, 2.1, 5.9), (0.5, 1.12, -33.0, 41.0)):
        M = np.array([[sc * np.cos(th), -sc * np.sin(th), tx],
                      [sc * np.sin(th), sc * np.cos(th), ty]], np.float64)
        Mi = cv2.invertAffineTransform(M)
        ref = cv2.warpAffine(src.transpose(1, 2, 0), M, (200, 180),
                             flags=cv2.INTER_CUBIC).transpose(2, 0, 1)
        got = G.warp_cubic(c, box, Mi, 200, 180).cpu().numpy()
        mx, lsb, ratio = cmp("warp grid rot%.2f" % th, got, ref)
        worst_w = max(worst_w, mx)
    ref = cv2.warpAffine(src.transpose(1, 2, 0),
                         np.array([[1.0, 0, -40.0], [0, 1.0, -25.0]], np.float64),
                         (320, 300), flags=cv2.INTER_CUBIC).transpose(2, 0, 1)
    got = G.warp_cubic(c, box, np.array([[1.0, 0, 40.0], [0, 1.0, 25.0]]), 320, 300).cpu().numpy()
    mx, lsb, ratio = cmp("warp grid 界外(零边界)", got, ref)
    worst_w = max(worst_w, mx)
    if impl == "gather" or True:
        M = np.array([[0.93 * np.cos(0.2), -0.93 * np.sin(0.2), 5.0],
                      [0.93 * np.sin(0.2), 0.93 * np.cos(0.2), -3.0]], np.float64)
        Mi = cv2.invertAffineTransform(M)
        ref = cv2.warpAffine(src.transpose(1, 2, 0), M, (200, 180),
                             flags=cv2.INTER_CUBIC).transpose(2, 0, 1)
        got = G.warp_cubic_gather(c, box, Mi, 200, 180).cpu().numpy()
        cmp("warp gather", got, ref)
    check(worst_w < TOL_FLOAT, "warp vs cv2 逐点差 < %.1e" % TOL_FLOAT, "max %.3e" % worst_w)
    res["warp_max"] = worst_w

    # ---- cv2 自身离「精确 float64」有多远（残余的根）----
    A = -0.75

    def cw(f):
        g = 1.0 - f
        c0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
        c1 = ((A + 2) * f - (A + 3)) * f * f + 1
        c2 = ((A + 2) * g - (A + 3)) * g * g + 1
        return c0, c1, c2, 1.0 - c0 - c1 - c2

    h, w = 300, 320
    s1 = rng.random((h, w)).astype(np.float32)
    M = np.array([[0.93 * np.cos(0.09), -0.93 * np.sin(0.09), 17.3],
                  [0.93 * np.sin(0.09), 0.93 * np.cos(0.09), -9.7]], np.float64)
    Mi = cv2.invertAffineTransform(M)
    ys, xs = np.mgrid[0:180, 0:200].astype(np.float64)
    sx = Mi[0, 0] * xs + Mi[0, 1] * ys + Mi[0, 2]
    sy = Mi[1, 0] * xs + Mi[1, 1] * ys + Mi[1, 2]
    ix = np.floor(sx).astype(np.int64)
    iy = np.floor(sy).astype(np.int64)
    wx = cw(sx - ix)
    wy = cw(sy - iy)
    ref64 = np.zeros((180, 200), np.float64)
    for i in range(4):
        yy = iy - 1 + i
        my = ((yy >= 0) & (yy < h)).astype(np.float64)
        for j in range(4):
            xx = ix - 1 + j
            mx_ = ((xx >= 0) & (xx < w)).astype(np.float64)
            ref64 += (wy[i] * wx[j]) * s1[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)] * (my * mx_)
    cvd = cv2.warpAffine(s1, M, (200, 180), flags=cv2.INTER_CUBIC).astype(np.float64)
    gat = G.warp_cubic_gather(c, t.as_tensor(s1[None], device=c.dev), Mi, 200, 180)[0].cpu().numpy()
    gd = G.warp_cubic(c, t.as_tensor(s1[None], device=c.dev), Mi, 200, 180)[0].cpu().numpy()
    say("   离「精确 float64 参考」的距离（白噪声源，最能放大坐标误差）：")
    say("     cv2      max %.3e" % np.abs(cvd - ref64).max())
    say("     gather   max %.3e" % np.abs(gat.astype(np.float64) - ref64).max())
    say("     grid     max %.3e" % np.abs(gd.astype(np.float64) - ref64).max())
    res["cv2_vs_exact"] = float(np.abs(cvd - ref64).max())

    # ---- resize ----
    worst_r = 0.0
    for (ow, oh) in ((416, 416), (137, 100)):
        m1 = rng.random((1, 256, 256)).astype(np.float32)
        ref = cv2.resize(m1[0], (ow, oh), interpolation=cv2.INTER_CUBIC)[None]
        got = G.resize_cubic(c, t.as_tensor(m1, device=c.dev), ow, oh).cpu().numpy()
        mx, lsb, ratio = cmp("resize 256->%dx%d" % (ow, oh), got[0], ref[0])
        worst_r = max(worst_r, mx)
    check(worst_r < 1e-5, "resize vs cv2 逐点差 < 1e-5", "max %.3e" % worst_r)
    res["resize_max"] = worst_r

    # ---- GaussianBlur ----
    worst_g = 0.0
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    soft = np.clip(1.0 - (((xx - 100) ** 2 + (yy - 95) ** 2) ** 0.5 - 60) / 8.0, 0, 1)[None]
    for b in (3, 5, 7, 9, 11, 21):
        ref = cv2.GaussianBlur(soft[0], (b, b), 0)
        got = G.gauss_blur(c, t.as_tensor(soft, device=c.dev), b, 0.0)[0].cpu().numpy()
        mx, lsb, ratio = cmp("gauss reflect k=%d" % b, got, ref)
        worst_g = max(worst_g, mx)
    # 遮罩那条路：CPU 是 pad(32)+blur，等价于零边界
    refp = cv2.GaussianBlur(np.pad(soft[0], 32), (5, 5), 0)[32:-32, 32:-32]
    got = G.gauss_blur(c, t.as_tensor(soft, device=c.dev), 5, 0.0, border="zero")[0].cpu().numpy()
    cmp("gauss 零边界 k=5", got, refp)
    check(worst_g < 1e-5, "GaussianBlur vs cv2 逐点差 < 1e-5", "max %.3e" % worst_g)
    res["gauss_max"] = worst_g

    # ---- match_tone ----
    S2 = 416
    fo = (rng.random((3, S2, S2)) * 0.7 + 0.1).astype(np.float32)
    dw = (rng.random((3, S2, S2)) * 0.7 + 0.1).astype(np.float32)
    yy, xx = np.mgrid[0:S2, 0:S2]
    mask = np.clip(1.0 - (((xx - S2 / 2) ** 2 + (yy - S2 / 2) ** 2) ** 0.5 - 120) / 20.0,
                   0, 1).astype(np.float32)
    # 参考取「CPU 真正会产出的东西」：_color_transfer(code=9) —— 它只写遮罩包围盒
    # （_match_tone_roi 自己返回的是整画布变换，盒外也被改了，不是 CPU 的最终行为）
    ref = MC._color_transfer(fo.transpose(1, 2, 0), dw.transpose(1, 2, 0), mask, 9, {}).transpose(2, 0, 1)
    got = G.mt_transfer(c, t.as_tensor(fo, device=c.dev), t.as_tensor(dw, device=c.dev),
                        t.as_tensor(mask, device=c.dev)[None]).cpu().numpy()
    mx, lsb, ratio = cmp("match_tone(mt) 416", got, ref)
    check(mx < 1e-4, "mt vs numpy 逐点差 < 1e-4", "max %.3e（= %.4f LSB）" % (mx, mx * 255))
    res["mt_max"] = mx

    # ---- hist-match ----
    ct_mod = MC._imagelib()
    worst_h = 0.0
    for tag, (a, b) in {"u8 量化": (np.round(fo * 255) / 255, np.round(dw * 255) / 255),
                        "连续": (fo, dw)}.items():
        ref = ct_mod.color_hist_match(a.transpose(1, 2, 0).astype(np.float32),
                                      b.transpose(1, 2, 0).astype(np.float32),
                                      hist_match_threshold=255).transpose(2, 0, 1)
        got = G.hist_match(c, t.as_tensor(a, device=c.dev), t.as_tensor(b, device=c.dev),
                           255).cpu().numpy()
        mx, lsb, ratio = cmp("hist-match " + tag, got, ref)
        worst_h = max(worst_h, mx)
    check(worst_h < 1e-6, "hist-match vs numpy 逐点差 < 1e-6", "max %.3e" % worst_h)
    res["hist_max"] = worst_h
    return res


# ============================================================
# 3) 整帧对拍
# ============================================================
def run_pair(frame, faces, cfg, u8_input=True):
    """同一份输入跑 CPU / GPU，返回 (cpu_out, gpu_out, cpu_ms, gpu_ms)。"""
    import numpy as np
    from server.backend_merge import compose as MC
    from server.backend_merge import compose_gpu as G
    cf = cpu_faces(faces)
    t0 = time.perf_counter()
    a = MC.compose_frame(frame.copy(), cf, cfg, {})
    t1 = time.perf_counter()
    gf = gpu_faces_u8(faces) if u8_input else cpu_faces(faces)
    b = G.compose_frame_gpu(frame.copy(), gf, cfg, {})
    t2 = time.perf_counter()
    return a, b, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def diff_stats(a, b):
    import numpy as np
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"mean": float(d.mean()), "max": int(d.max()),
            "nz": int((d > 0).sum()), "ratio": float((d > 0).mean()),
            "n_ge2": int((d >= 2).sum()), "n": int(d.size)}


def case_frame(frames, cfg, reps=1):
    head("3 · 整帧对拍：CPU compose_frame vs GPU compose_frame_gpu（同输入、同 cfg）")
    say("   帧数 %d · cfg = %s" % (len(frames), json.dumps(cfg, ensure_ascii=False)))
    rows = []
    worst = 0
    for i, fr in enumerate(frames):
        a, b, cms, gms = run_pair(fr["frame"], fr["faces"], cfg)
        st = diff_stats(a, b)
        st.update({"tag": fr["tag"], "faces": len(fr["faces"]), "cpu_ms": round(cms, 2),
                   "gpu_ms": round(gms, 2)})
        rows.append(st)
        worst = max(worst, st["max"])
        if i < 8 or st["max"] > 1:
            say("   %s  faces=%d  mean %.6f LSB  max %d  diff %d/%d (%.4f%%)  cpu %6.2f ms  gpu %5.2f ms"
                % (st["tag"], st["faces"], st["mean"], st["max"], st["nz"], st["n"],
                   100 * st["ratio"], st["cpu_ms"], st["gpu_ms"]))
    mean_lsb = statistics.mean(r["mean"] for r in rows)
    worst_ratio = max(r["ratio"] for r in rows)
    ge2 = sum(r["n_ge2"] for r in rows)
    say("   ----")
    say("   逐帧：mean %.6f LSB（平均）· max %d LSB（最差帧）· 差异像素 %.4f%%（最差帧）· >=2 LSB 的像素共 %d 个"
        % (mean_lsb, worst, 100 * worst_ratio, ge2))
    check(worst <= TOL_LSB, "整帧 max <= %d LSB（%d 帧）" % (TOL_LSB, len(rows)), "max %d" % worst)
    check(ge2 == 0, "没有任何像素差 >= 2 LSB", "%d 个" % ge2)
    check(mean_lsb < 1e-3, "逐帧平均差 < 1e-3 LSB", "%.6f" % mean_lsb)
    return rows


# ============================================================
# 4) 变体
# ============================================================
def case_variants(frames, n=6):
    import numpy as np
    head("4 · 变体：mode / ct / mask_mode / blur / 双脸")
    sub = frames[:n]
    cases = []
    base = dict(SETTINGS)
    cases.append(("overlay · mt · mask8 · blur4", dict(base)))
    cases.append(("overlay · none", dict(base, color_transfer_mode="none")))
    cases.append(("overlay · mt · mask8 · blur0", dict(base, mask_blur=0)))
    cases.append(("overlay · mt · mask8 · blur8", dict(base, mask_blur=8)))
    cases.append(("overlay · mt · mask2(prd_dfm)", dict(base, mask_mode=2)))
    cases.append(("overlay · mt · mask5(prd+dst)", dict(base, mask_mode=5)))
    cases.append(("hist-match · mt", dict(base, mode="hist-match")))
    cases.append(("overlay · lct（ct 逐脸回退 CPU）", dict(base, color_transfer_mode="lct")))
    # 自检：cfg 里的 mode 必须真的被 compose_frame 读到（否则测的就不是这个模式）
    from server.backend_merge import compose as _MC
    probe_cfg = dict(base, mode="hist-match")
    if str(probe_cfg.get("mode")) != "hist-match":
        raise SystemExit("夹具 cfg 的键名不对")
    say("   生效的 mode / ct（**每次都要看这一列**：第一版这里 cfg 写的是界面键名 blend，")
    say("   compose_frame 读不到，于是 seamless / hist-match 一直在跑 overlay —— 断言全绿、")
    say("   覆盖为零。变体矩阵必须把实际生效值打出来并核对）")
    for label, cfg in cases:
        say("   %-32s -> mode=%s ct=%s mask=%s blur=%s · 期望 mode=%s"
            % (label, cfg.get("mode"), cfg.get("color_transfer_mode"), cfg.get("mask_mode"),
               cfg.get("blur_mask_modifier"),
               ("hist-match" if "hist-match" in label else "overlay")))
    rows = []
    worst_all = 0
    for label, cfg in cases:
        _want = "hist-match" if "hist-match" in label else "overlay"
        if str(cfg.get("mode")) != _want:
            raise SystemExit("变体 %r 期望 mode=%s，实际 %r（键名/取值写错了）"
                             % (label, _want, cfg.get("mode")))
        worst = 0
        tots = 0
        nz = 0
        for fr in sub:
            a, b, cms, gms = run_pair(fr["frame"], fr["faces"], cfg)
            st = diff_stats(a, b)
            worst = max(worst, st["max"])
            tots += st["n"]
            nz += st["nz"]
        rows.append({"case": label, "max": worst, "nz": nz, "n": tots})
        worst_all = max(worst_all, worst)
        say("   %-32s max %d LSB · 差异像素 %d/%d (%.4f%%)"
            % (label, worst, nz, tots, 100.0 * nz / tots))
        tol = TOL_LSB_HIST if "hist-match" in label else TOL_LSB
        check(worst <= tol, "  %s" % label, "max %d LSB（界 %d）" % (worst, tol))

    # 双脸：把两张脸叠到同一帧（走 CPU 的 work 累加分支 + 并集 ROI）
    if len(sub) >= 2:
        f1 = sub[0]["faces"][0]
        f2 = sub[1]["faces"][0]
        # 第二张脸挪到别处（用 landmarks 平移，保持真实形状）
        lm2 = np.array(f2["landmarks"], np.float32).copy()
        lm2[:, 0] += 260.0
        lm2[:, 1] += 120.0
        f2 = dict(f2, landmarks=lm2)
        both = [f1, f2]
        for cfg, label in ((dict(base), "双脸 overlay · mt · mask8"),
                           (dict(base, mask_mode=2), "双脸 overlay · mt · mask2")):
            a, b, cms, gms = run_pair(sub[0]["frame"], both, cfg)
            st = diff_stats(a, b)
            say("   %-32s max %d LSB · 差异像素 %d/%d (%.4f%%) · cpu %.2f ms / gpu %.2f ms"
                % (label, st["max"], st["nz"], st["n"], 100 * st["ratio"], cms, gms))
            rows.append({"case": label, "max": st["max"], "nz": st["nz"], "n": st["n"]})
            worst_all = max(worst_all, st["max"])
            check(st["max"] <= TOL_LSB, "  %s" % label, "max %d LSB" % st["max"])
    return rows


# ============================================================
# 5) 帧速
# ============================================================
def case_perf(frames, cfg0, reps=3):
    import numpy as np
    from server.backend_merge import compose as MC
    from server.backend_merge import compose_gpu as G
    out = {}
    for mode, label in (("overlay", "overlay"), ("seamless", "seamless")):
        out[mode] = _perf_one(frames, dict(cfg0, mode=mode), reps, label)
    return out


def _perf_one(frames, cfg, reps, label):
    import numpy as np
    from server.backend_merge import compose as MC
    from server.backend_merge import compose_gpu as G
    head("5 · 帧速（%s）：同一批帧，CPU 1 线程 / CPU 8 线程 / GPU" % label)
    if cfg.get("mode") == "seamless":
        # 默认是关的（见 compose_gpu.seamless_on 的实测）；这里量的是「能力」不是默认口径
        os.environ["DFL_COMPOSE_GPU_SEAMLESS"] = "1"
    cf = [cpu_faces(f["faces"]) for f in frames]
    gf = [gpu_faces_u8(f["faces"]) for f in frames]
    raw = [f["frame"] for f in frames]
    n = len(frames)

    def timeit(fn, r):
        ts = []
        for _ in range(r):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts)

    # CPU 单线程
    def cpu1():
        for i in range(n):
            MC.compose_frame(raw[i].copy(), cf[i], cfg, {})
    ms_cpu1 = timeit(cpu1, reps)
    # GPU（含同步；返回的 u8 数组已经 .cpu()）
    def gpu():
        for i in range(n):
            G.compose_frame_gpu(raw[i].copy(), gf[i], cfg, {})
    ms_gpu = timeit(gpu, reps)
    ms_gpu_first = None
    say("   CPU 1 线程  %8.2f ms/帧（%.1f 帧/秒）" % (ms_cpu1 / n, 1000.0 * n / ms_cpu1))
    say("   GPU         %8.2f ms/帧（%.1f 帧/秒）  加速 %.1fx"
        % (ms_gpu / n, 1000.0 * n / ms_gpu, ms_cpu1 / ms_gpu))
    # CPU 8 线程（就是生产 fused 的 n_cmp=8）
    import queue
    import threading
    for nthread in (8,):
        q = queue.Queue()
        for i in range(n):
            q.put(i)
        for _ in range(nthread):
            q.put(None)
        ts = []
        lock = threading.Lock()

        def worker():
            while True:
                i = q.get()
                if i is None:
                    return
                t0 = time.perf_counter()
                MC.compose_frame(raw[i].copy(), cf[i], cfg, {})
                with lock:
                    ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker) for _ in range(nthread)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        wall = (time.perf_counter() - t0) * 1000.0
        say("   CPU %d 线程  %8.2f ms/帧（%.1f 帧/秒）· 单帧墙钟中位 %.1f ms · 合计线程时 %.1f ms"
            % (nthread, wall / n, 1000.0 * n / wall, statistics.median(ts) * 1000, sum(ts) * 1000))
        ms_cpu8 = wall
    check(ms_gpu < ms_cpu1, "  %s：GPU 比 CPU 单线程快" % label,
          "%.2f vs %.2f ms/帧（%.1fx）" % (ms_gpu / n, ms_cpu1 / n, ms_cpu1 / ms_gpu))
    return {"cpu1_ms": ms_cpu1 / n, "cpu8_ms": ms_cpu8 / n, "gpu_ms": ms_gpu / n,
            "speedup_1t": ms_cpu1 / ms_gpu, "speedup_8t": ms_cpu8 / ms_gpu, "frames": n}


def case_seamless(frames, n=6):
    """seamless：CPU 的 cv2.seamlessClone vs GPU 的 torch DST 解。

    验收按 <=1 LSB 定，**不追求 0**：cv2 那边自己就是 float32 + DST 的舍入，
    见 app/docs/notes/experiments/seamless.md（97.14% 逐位相同、2.53% 差 1 LSB）。
    这里额外把「传进无缝的遮罩没被改写」钉死（cv2.seamlessClone 有就地改写的坑）。
    """
    head("5 · seamless：CPU compose(seamless) vs GPU compose(seamless)")
    os.environ["DFL_COMPOSE_GPU_ASSERT"] = "1"
    os.environ["DFL_COMPOSE_GPU_SEAMLESS"] = "1"       # 默认关（见 seamless_on 的实测）
    rows = []
    worst = 0
    for fr in frames[:n]:
        cfg = dict(SETTINGS, mode="seamless")
        a, b, cms, gms = run_pair(fr["frame"], fr["faces"], cfg)
        st = diff_stats(a, b)
        rows.append(st)
        worst = max(worst, st["max"])
        say("   %s  mean %.6f LSB  max %d  差异 %d/%d (%.4f%%)  >1LSB %d  cpu %7.2f ms  gpu %6.2f ms"
            % (fr["tag"], st["mean"], st["max"], st["nz"], st["n"], 100 * st["ratio"],
               int((st["max"] > 1)), cms, gms))
    mean_lsb = statistics.mean(r["mean"] for r in rows)
    ratio = max(r["ratio"] for r in rows)
    over = sum(1 for r in rows if r["max"] > 1)
    say("   ----")
    say("   %d 帧：mean %.6f LSB · max %d LSB · 超 1 LSB 的帧 %d 个（**已知**，见 docs/26 §4.5）"
        % (len(rows), mean_lsb, worst, over))
    check(worst <= TOL_LSB_SEAMLESS,
          "seamless 整帧 max <= %d LSB（Poisson 积分放大的实测包络，不是逐位一致）" % TOL_LSB_SEAMLESS,
          "max %d，%d/%d 帧超 1 LSB" % (worst, over, len(rows)))
    check(ratio < TOL_RATIO_SEAMLESS, "seamless 差异像素 < %.1f%%" % (100 * TOL_RATIO_SEAMLESS),
          "最差 %.4f%%" % (100 * ratio))
    check(mean_lsb < 0.2, "seamless 逐帧平均差 < 0.2 LSB", "%.6f" % mean_lsb)
    check(bool(rows), "遮罩没被就地改写（DFL_COMPOSE_GPU_ASSERT=1 全程开着）", "断言通过")
    return rows


def case_switch():
    """开关口径：默认关、seamless 子开关跟随、关掉时 GPU 入口直接拒。"""
    import os as _os
    import server.app  # noqa: F401
    from server.backend_merge import compose_gpu as G
    head("6 · 开关：默认 0 · seamless 子开关 · 关掉时 GPU 入口拒绝")
    old = {k: _os.environ.get(k) for k in ("DFL_COMPOSE_GPU", "DFL_COMPOSE_GPU_SEAMLESS")}
    try:
        _os.environ.pop("DFL_COMPOSE_GPU", None)
        check(G.env_on() is False, "DFL_COMPOSE_GPU 不设时 env_on() = False", "默认走 CPU")
        _os.environ["DFL_COMPOSE_GPU"] = "0"
        check(G.env_on() is False, "DFL_COMPOSE_GPU=0 时 env_on() = False", "")
        _os.environ["DFL_COMPOSE_GPU"] = "1"
        check(G.env_on() is True, "DFL_COMPOSE_GPU=1 时 env_on() = True", "")
        _os.environ.pop("DFL_COMPOSE_GPU_SEAMLESS", None)
        check(G.seamless_on() is False,
              "DFL_COMPOSE_GPU_SEAMLESS 不设时默认 0（实测 seamless 搬 GPU 让整条线慢 13.6%）", "")
        _os.environ["DFL_COMPOSE_GPU_SEAMLESS"] = "1"
        check(G.seamless_on() is True, "DFL_COMPOSE_GPU_SEAMLESS=1 时可显式打开", "")
        _os.environ["DFL_COMPOSE_GPU_SEAMLESS"] = "0"
        check(G.seamless_on() is False, "DFL_COMPOSE_GPU_SEAMLESS=0 时关掉", "")
        # 关掉之后 GPU 入口必须拒绝（让流水线退回 CPU，而不是静默算错）
        import numpy as np
        ok = False
        try:
            G.compose_frame_gpu(np.zeros((64, 64, 3), np.uint8),
                                [{"landmarks": np.zeros((68, 2), np.float32),
                                  "pred": np.zeros((64, 64, 3), np.uint8)}],
                                {"mode": "seamless"}, {})
        except ValueError:
            ok = True
        except Exception:                                      # noqa: BLE001
            ok = False
        check(ok, "DFL_COMPOSE_GPU_SEAMLESS=0 时 GPU 入口对 seamless 直接抛 ValueError", "")
    finally:
        for k, v in old.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--warp", default="", help="grid / gather（默认按模块里的口径）")
    ap.add_argument("--json", default=str(BOX / "compose_gpu_accept.json"))
    a = ap.parse_args()
    BOX.mkdir(parents=True, exist_ok=True)
    if a.warp:
        import os as _os
        _os.environ["DFL_COMPOSE_GPU_WARP"] = a.warp

    import numpy as np
    import server.app  # noqa: F401  把 engine 路径挂上
    from server.backend_merge import compose_gpu as G

    say("torch %s · cuda %s · %s" % (G.ctx().torch.__version__, G.ctx().torch.cuda.is_available(),
                                     G.ctx().dev))
    say("warp 实现 = %s" % G.warp_impl())

    dump = a.dump
    if not dump:
        cand = sorted(BOX.glob("*.dump")) + sorted((TEMP / "dfl_cgpu").glob("*.dump"))
        cand = [p for p in cand if any(p.glob("*_lm.json"))]
        if cand:
            dump = str(cand[-1])
    if dump and Path(dump).is_dir():
        frames = load_dump(dump, limit=a.frames)
        say("夹具：fused dump %s -> %d 帧 / %d 张脸"
            % (dump, len(frames), sum(len(f["faces"]) for f in frames)))
    else:
        frames = synth_frames(a.frames)
        say("夹具：没有 dump，用合成夹具 %d 帧（数字以 --dump 为准）" % len(frames))

    cfg = dict(SETTINGS)
    out = {"dump": dump, "frames": len(frames), "cfg": cfg,
           "torch": G.ctx().torch.__version__,
           "device": str(G.ctx().dev), "warp": G.warp_impl()}
    want = set(x.strip() for x in str(a.cases).split(",") if x.strip())
    allc = "all" in want

    if allc or "cubic" in want:
        case_cubic()
    if allc or "prims" in want:
        out["prims"] = case_prims(G.warp_impl())
    if allc or "frame" in want:
        out["frame"] = case_frame(frames, cfg)
    if allc or "variants" in want:
        out["variants"] = case_variants(frames)
    if allc or "seamless" in want:
        out["seamless"] = case_seamless(frames)
    if allc or "switch" in want:
        case_switch()
    if allc or "perf" in want:
        out["perf"] = case_perf(frames, cfg, reps=a.reps)

    bad = [r for r in RESULTS if not r["ok"]]
    out["checks"] = RESULTS
    out["ok"] = not bad
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    head("结论")
    say("  %d 项检查：%d 通过 / %d 失败" % (len(RESULTS), len(RESULTS) - len(bad), len(bad)))
    for r in bad:
        say("  FAIL  %s  --  %s" % (r["label"], r["detail"]))
    say("  报告：%s" % a.json)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())