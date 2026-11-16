# -*- coding: utf-8 -*-
r"""aligned 原生合成 vs facelib 画布合成：A/B 对拍、位置校验、帧速表。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\proj_compose_ab.py
    (只跑某几段 --cases pixel,fps,pos；改帧数 --frames 60；重复取中位 --reps 2)

验的东西：
  1) pos   几何：aligned 原生合成反贴回原帧的**亚像素**精度（应 <1px）——画布上一个
           已知坐标的色斑，贴回原帧再按同一张矩阵取回画布，量重心偏移；同一张画布
           若用 facelib 的 WHOLE_FACE 矩阵贴，偏移就是两张画布的差。
           另外量真实对齐脸上两张画布的四角+中心在原帧里的距离。
           **2026-12 口径翻转**：`_align.align_mat_from_68` 只拟合 33 点之后，aligned
           芯片与 facelib 是同一张画布，所以这两项从「必须错位（>3px / 19px）」改成
           「必须重合（<1px）」。改前的数字（13.06 px / mean 19.25 px）留在
           docs/23-ALIGNED-COMPOSE.md 里当历史。
  2) pixel 两条路在同一项目上逐像素对拍：mean LSB / max / 差异像素比例 / 逐字节相同帧数；
           差异是否只在脸边缘（相位相关位移 = 0 + 差异像素落在梯度上的比例），
           差异图落到 %TEMP%\dfl_compose_ab\out\。
  3) fps   384 / 416 两档、facelib vs aligned，各 reps 次取中位，给帧速与分段耗时。

退出码 0 = 全过（位置 <1px、差异确实存在且不是整体错位、帧速不倒退）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_compose_ab"
SRC = TEMP / "dfl_pool_accept" / "proj"          # proj_pool_accept 建的真项目夹具
PROJ = BOX / "proj"
OUT = BOX / "out"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
TOL_POS_PX = 1.0            # 位置验收：误差 < 1px
TOL_MEAN = 6.0              # 两条路允许的像素差上限（LSB）：它本该有差（少一次重采样），
                            # 但不该大到「换了一张脸」
RESULTS = []


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def say(s=""):
    print(s)


def check(ok, label, detail=""):
    RESULTS.append({"ok": bool(ok), "label": label, "detail": str(detail)})
    print(("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                            ("  —  " + str(detail)) if detail else "")))
    return bool(ok)


# ------------------------------------------------------------
# 夹具：从 proj_pool_accept 的真项目里切 N 帧（帧 + 对应 aligned DFLJPG）
# ------------------------------------------------------------
def build_fixture(nframes):
    if not SRC.is_dir():
        raise SystemExit("找不到夹具项目 %s（先跑一次 proj_pool_accept.py）" % SRC)
    shutil.rmtree(PROJ, ignore_errors=True)
    (PROJ / "aligned").mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".png", ".jpg"))[:nframes]
    keep = set()
    for f in frames:
        shutil.copy2(str(f), str(PROJ / f.name))
        keep.add(f.stem)
    n = 0
    for a in sorted((SRC / "aligned").glob("*.jpg")):
        if a.stem.split("_")[0] in keep:
            shutil.copy2(str(a), str(PROJ / "aligned" / a.name))
            n += 1
    print("夹具：%d 帧 · %d 张对齐脸 → %s" % (len(frames), n, PROJ))
    return len(frames), n


def pick_dfm(which):
    w = TEMP / "dfl_pool_accept" / "ws" / "model"
    cand = sorted(w.glob("*.dfm"), key=lambda x: x.stat().st_size)
    if not cand:
        raise SystemExit("workspace 里没有 .dfm")
    # 小的那个是 416（阿証），大的那个是 384（张元英）
    return str(cand[0] if which == "416" else cand[-1])


def run_export(mode, dfm, nframes, tag):
    """在**本进程**里跑一次 _proj_pipe.run（不启服务），返回 status 快照。"""
    import server.app as A
    from server import routes_merge as rm
    os.environ["DFL_PROJ_COMPOSE"] = mode
    box = BOX / ("run_" + tag)
    shutil.rmtree(box, ignore_errors=True)
    xseg = str(rm._xseg_of({"xseg": str(Path(A.WORKSPACE) / "model" / "XSegLite" / "xseglite.onnx")}))
    cfg = {"project": str(PROJ), "frames_dir": str(PROJ), "aligned": str(PROJ / "aligned"),
           "merged": str(box / "merged"),
           "video": str(TEMP / "dfl_pool_accept" / "media" / "clip.mp4"),
           "out_video": str(box / "out.mp4"), "fps": 30.0, "video_frames": nframes,
           "models": [], "dfm": dfm, "model": Path(dfm).stem,
           "config": A._merge_settings_to_config(SETTINGS), "settings": SETTINGS,
           "xseg": xseg, "detector": "tinymog", "thresh": 0.5, "quality": 92,
           "force": True, "workers": 1, "batch": 4, "adaptive": {}, "vram_plan": {}}
    t0 = time.time()
    rm._proj_run(cfg)
    sn = rm._proj_snap()
    sn = dict(sn)
    sn["_wall"] = round(time.time() - t0, 2)
    sn["_dir"] = str(box / "merged")
    (box / "status.json").write_text(json.dumps(sn, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return sn


def cmp_dirs(a, b, dfm_name=""):
    """两个 merged 目录逐像素对拍 + 「差异是不是只在边缘」的量化。"""
    import cv2
    import numpy as np
    na = sorted(x.name for x in Path(a).iterdir() if x.is_file())
    nb = sorted(x.name for x in Path(b).iterdir() if x.is_file())
    if na != nb:
        return {"error": "帧集合不同：%d vs %d" % (len(na), len(nb))}
    tot = 0.0
    npix = 0
    mx = 0
    diffpx = 0
    byte_same = 0
    worst = None
    worst_pair = None
    edge_hit = 0
    edge_tot = 0
    big_edge = 0
    big_tot = 0
    edge_mag = []
    flat_mag = []
    shifts = []
    inside = []
    for n in na:
        fa, fb = Path(a) / n, Path(b) / n
        if fa.read_bytes() == fb.read_bytes():
            byte_same += 1
        ia = cv2.imread(str(fa), cv2.IMREAD_COLOR)
        ib = cv2.imread(str(fb), cv2.IMREAD_COLOR)
        if ia is None or ib is None or ia.shape != ib.shape:
            return {"error": "读不出/形状不同：%s" % n}
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        tot += float(d.sum())
        npix += int(d.size)
        m = int(d.max())
        mx = max(mx, m)
        nd = int((d.max(axis=2) > 0).sum())
        diffpx += nd
        if nd:
            # 「差异像素是不是落在原帧的边缘上」：灰度 Sobel 幅度 > 24 视作边缘
            g = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            dm = (d.max(axis=2) > 0)
            ed = (mag > 24)
            big = (d.max(axis=2) > 16)
            big_n = int(big.sum())
            if big_n:
                big_edge += int((big & ed).sum())
                big_tot += big_n
            # 差异幅度在「边缘 / 平坦」两组上的均值：重采样差异的特征是
            # 边缘上大、平坦处几乎是 0（<1 LSB）。
            if ed.any() and (~ed).any():
                edge_mag.append(float(d.max(axis=2)[ed].mean()))
                flat_mag.append(float(d.max(axis=2)[~ed].mean()))
            edge_hit += int((dm & ed).sum())
            edge_tot += nd
            # 全局错位会留下一个「整体平移」：两条产物之间做相位相关，位移应 ≈ 0
            ga = np.float32(g)
            gb = np.float32(cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY))
            (sx, sy), _r = cv2.phaseCorrelate(ga, gb)
            shifts.append((float(sx), float(sy)))
            # 差异像素里落在「两张图都不變」的区域的比例（背景为 0 → 差异只在脸上）
            inside.append(float(nd) / float(g.size))
        if m and (worst is None or m > worst[2]):
            worst = (n, round(float(d.mean()), 4), m)
            worst_pair = (ia, ib, d)
    smax = max((max(abs(s[0]), abs(s[1])) for s in shifts), default=0.0)
    out = {"frames": len(na), "mean_lsb": round(tot / max(1, npix), 6),
           "max_lsb": mx, "diff_px_pct": round(100.0 * diffpx / max(1, npix), 4),
           "byte_identical": byte_same, "worst": worst,
           "shift_px_max": round(smax, 4),
           "edge_hit_pct": (round(100.0 * edge_hit / edge_tot, 2) if edge_tot else None),
           "big_diff_px": big_tot,
           "big_on_edge_pct": (round(100.0 * big_edge / big_tot, 2) if big_tot else None),
           "mean_abs_on_edge": (round(float(np.mean(edge_mag)), 4) if edge_mag else None),
           "mean_abs_off_edge": (round(float(np.mean(flat_mag)), 4) if flat_mag else None),
           "face_area_pct_max": (round(100.0 * max(inside), 2) if inside else None)}
    if worst_pair is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        ia, ib, d = worst_pair
        amp = np.clip(d.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
        plate = np.hstack([ia, ib, amp])
        p = OUT / ("diff_%s.png" % (dfm_name or "x"))
        cv2.imwrite(str(p), plate)
        y, x = None, None
        ys, xs = np.nonzero(d.max(axis=2))
        if len(xs):
            x = int(xs.mean())
            y = int(ys.mean())
        p2 = OUT / ("diff_%s_crop.png" % (dfm_name or "x"))
        pad = 220
        cx = x if x is not None else ia.shape[1] // 2
        cy = y if y is not None else ia.shape[0] // 2
        x0 = max(0, min(ia.shape[1] - 2 * pad, cx - pad))
        y0 = max(0, min(ia.shape[0] - 2 * pad, cy - pad))
        if ia.shape[0] > 2 * pad and ia.shape[1] > 2 * pad:
            cv2.imwrite(str(p2), np.hstack([ia[y0:y0 + 2 * pad, x0:x0 + 2 * pad],
                                            ib[y0:y0 + 2 * pad, x0:x0 + 2 * pad],
                                            amp[y0:y0 + 2 * pad, x0:x0 + 2 * pad]]))
            out["diff_png"] = str(p)
            out["crop_png"] = str(p2)
    return out


# ------------------------------------------------------------
# 位置：aligned 原生合成的亚像素精度 + 两张画布的真实差距
# ------------------------------------------------------------
def canvas_delta(aligned_dir, limit=40):
    """真实对齐脸：image_to_face_mat vs facelib WHOLE_FACE 在**原帧**里的差距。"""
    import cv2
    import numpy as np
    import server.app as A                                  # noqa: F401
    from server import routes_merge as rm
    from server._merge_compose import ensure_engine_path
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    rows = []
    for p in sorted(Path(aligned_dir).glob("*.jpg"))[:limit]:
        m = rm._proj_head_meta(p)
        mat = m.get("image_to_face_mat")
        lm = m.get("source_landmarks")
        if not mat or not lm:
            continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        S = int(im.shape[0])
        A_ = np.asarray(get_transform_mat(np.asarray(lm, np.float32), S,
                                          FaceType.WHOLE_FACE, scale=1.0), np.float64)
        B_ = np.asarray(mat, np.float64)
        corners = np.float64([[0, 0], [S, 0], [S, S], [0, S], [S / 2.0, S / 2.0]])
        pts = np.hstack([corners, np.ones((5, 1))])
        fa = pts @ cv2.invertAffineTransform(A_).T
        fb = pts @ cv2.invertAffineTransform(B_).T
        d = np.linalg.norm(fa - fb, axis=1)
        rows.append({"file": p.name, "S": S, "mean": float(d.mean()),
                     "max": float(d.max())})
    if not rows:
        return None
    return {"n": len(rows),
            "mean_px": round(float(np.mean([r["mean"] for r in rows])), 2),
            "max_px": round(float(np.max([r["max"] for r in rows])), 2),
            "rows": rows[:8]}


def dot_centroid(img, thr=0.55):
    """品红斑点（R、B 高、G 低）的亚像素重心；没有就返回 None。"""
    import numpy as np
    a = img.astype(np.float32) / 255.0
    m = (a[..., 2] > thr) & (a[..., 0] > thr) & (a[..., 1] < thr * 0.5)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    wgt = (a[..., 0] * m + a[..., 2] * m)[ys, xs]
    return (float((xs * wgt).sum() / wgt.sum()), float((ys * wgt).sum() / wgt.sum()),
            int(m.sum()))


def pos_check(aligned_dir):
    """画布上画一个已知位置的斑点 → 贴回原帧 → 再取回画布 → 量重心偏移。"""
    import cv2
    import numpy as np
    import server.app as A                                  # noqa: F401
    from server import _merge_compose as MC
    from server import routes_merge as rm
    MC.ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat

    cfg = {"mask_mode": 0, "mode": "overlay", "color_transfer_mode": 0,
           "erode_mask_modifier": 0, "blur_mask_modifier": 0}
    S = 416
    # 挑一张「画布整个落在帧内」的脸：脸贴着画面边缘时贴回会被裁掉，斑点就找不着了
    # （那不是错位，是物理上没地方贴）。
    pick = None
    for p in sorted(Path(aligned_dir).glob("*.jpg")):
        meta = rm._proj_head_meta(p)
        mat = meta.get("image_to_face_mat")
        if not mat:
            continue
        s_al = int(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR).shape[0])
        m = np.asarray(mat, np.float64) * (float(S) / float(s_al))
        frame = cv2.imdecode(np.fromfile(str(Path(aligned_dir).parent /
                                           meta["original_name"]), np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            continue
        hh, ww = frame.shape[:2]
        box = MC._roi_of_face(m, S, frame.shape, margin=0)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if x0 <= 2 or y0 <= 2 or x1 >= ww - 2 or y1 >= hh - 2:
            continue                     # 画布被画面边缘裁到，换一张
        pick = (p, meta, frame, m, s_al)
        break
    if pick is None:
        return {"error": "夹具里没有一张「画布完整落在帧内」的对齐脸"}
    p, meta, frame, mat_b, s_al = pick
    lm = np.asarray(meta["source_landmarks"], np.float32)
    mat_a = np.asarray(get_transform_mat(lm, S, FaceType.WHOLE_FACE, scale=1.0), np.float64)
    full = np.ones((S, S), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    # 画布上 4 个「远离中心」的位置（越靠边，两张画布的差越大 —— 中心几乎不动）
    spots = [(0.35 * S, 0.32 * S), (0.62 * S, 0.32 * S),
             (0.35 * S, 0.62 * S), (0.62 * S, 0.62 * S)]

    def canvas_with(cx, cy):
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 3.0 ** 2)))
        cv_ = np.zeros((S, S, 3), np.float32)
        cv_[..., 0] = blob
        cv_[..., 2] = blob
        return cv_

    def measure(out, fwd_mat):
        # 取回画布：cv2 默认把 M 当 src→dst（帧→画布就是 fwd_mat 本身），所以这里
        # **不能**再加 WARP_INVERSE_MAP（那是「M 是 dst→src」，要配 inv(mat) 用）。
        back = cv2.warpAffine(out, fwd_mat, (S, S), flags=cv2.INTER_CUBIC)
        c = dot_centroid(back)
        if c is None:
            return None
        return (c[0], c[1])

    errs_b, errs_c, errs_s, rows = [], [], [], []
    for (cx, cy) in spots:
        canvas = canvas_with(cx, cy)
        pred = np.clip(canvas, 0, 1)
        # ① aligned 原生：画布上的 (cx,cy) 贴回原帧再取回画布，应还在 (cx,cy)
        face = {"mat": mat_b, "pred": pred, "dw": None, "landmarks": lm,
                "prd_dfm": None, "dst_dfm": None,
                "xseg_prd": None, "xseg_dst": None}
        out_b = MC.compose_aligned_frame(frame.copy(), [face], cfg, {})
        got = measure(out_b, mat_b)
        eb = (float("inf") if got is None
              else float(np.hypot(got[0] - cx, got[1] - cy)))
        # ② 交叉：同一张画布用 facelib 的 WHOLE_FACE 矩阵贴（= 两张画布混用）
        out_a = MC.compose_frame(frame.copy(), [
            {"landmarks": lm, "pred": pred,
             "prd_dfm": full, "dst_dfm": None, "xseg_prd": None, "xseg_dst": None}],
            dict(cfg, mask_mode=2), {})
        got_c = measure(out_a, mat_b)
        ec = (float("inf") if got_c is None
              else float(np.hypot(got_c[0] - cx, got_c[1] - cy)))
        # ③ 自检：facelib 画的图用 facelib 的矩阵取回（应 ≈ 0，证明仪器本身是对的）
        got_s = measure(out_a, mat_a)
        es = (float("inf") if got_s is None
              else float(np.hypot(got_s[0] - cx, got_s[1] - cy)))
        errs_b.append(eb)
        errs_c.append(ec)
        errs_s.append(es)
        rows.append({"c": [round(cx, 1), round(cy, 1)], "aligned": round(eb, 3),
                     "cross": round(ec, 3), "selfcheck": round(es, 3)})
    err_b = max(errs_b)
    err_c = max(errs_c)
    err_s = max(errs_s)
    say()
    say("  画布上 %d 个已知位置的斑点，贴回原帧后再取回画布，量重心偏移：" % len(spots))
    for r in rows:
        say("    画布 %-16s  aligned 原生 %6.3f px  ·  混用 facelib 矩阵 %7.3f px  ·  自检 %6.3f px"
            % (str(tuple(r["c"])), r["aligned"], r["cross"], r["selfcheck"]))
    say("    ── 取最大：aligned 原生 %.3f px · 混用 %.3f px · 自检 %.3f px"
        % (err_b, err_c, err_s))
    return {"S": S, "spots": rows, "err_aligned": err_b, "err_cross": err_c,
            "err_selfcheck": err_s}


def _fmt(v):
    return "n/a" if v is None else "%.3f" % v


def case_pos():
    head("1 · 位置：aligned 原生合成贴回的亚像素精度 + 两张画布的重合度（2026-12 起应当重合）")
    res = pos_check(PROJ / "aligned")
    if res.get("error"):
        check(False, "位置校验跑起来", res["error"])
        return {"pos": res}
    check(res["err_aligned"] is not None and res["err_aligned"] < TOL_POS_PX,
          "aligned 原生合成贴回 <%.1fpx（画布坐标 → 原帧 → 画布，闭合成环）" % TOL_POS_PX,
          "%s px" % _fmt(res["err_aligned"]))
    check(res["err_selfcheck"] is not None and res["err_selfcheck"] < TOL_POS_PX,
          "仪器自检：facelib 画布 + facelib 矩阵同样闭合（<%.1fpx）" % TOL_POS_PX,
          "%s px" % _fmt(res["err_selfcheck"]))
    # 2026-12 起 aligned 芯片与 facelib 是同一张画布（_align.align_mat_from_68 只拟合
    # 33 点，见 docs/24-ALIGN-WHOLEFACE.md），所以「混用」不再错位 —— 这条从「必须错位」
    # 翻成「必须重合」。改前这里是 13.06 px，现在应当 <1px。
    check(res["err_cross"] is not None and res["err_cross"] < TOL_POS_PX,
          "两张画布已经合并：混用 facelib 矩阵也闭合（改前 13.06px，现在应 <1px）",
          "%s px" % _fmt(res["err_cross"]))
    d = canvas_delta(PROJ / "aligned")
    if d:
        say()
        say("  真实对齐脸（%d 张）：image_to_face_mat vs facelib WHOLE_FACE，" % d["n"])
        say("  画布四角+中心映回原帧的距离  mean %.2f px / max %.2f px" % (d["mean_px"], d["max_px"]))
        # 同一条口径翻转：改前 mean 19.25 / max 37.68 px，重切后应当 ≈ 0。
        check(d["mean_px"] < TOL_POS_PX, "真实人脸上两张画布已重合（改前 19.25px）",
              "mean %.2f px / max %.2f px" % (d["mean_px"], d["max_px"]))
    return {"pos": res, "canvas": d}


# ------------------------------------------------------------
# 像素：两条路在同一项目上对拍
# ------------------------------------------------------------
def case_pixel(nframes):
    head("2 · 像素：aligned 原生 vs facelib 画布（同一项目、同一模型、force=true）")
    out = {}
    for which in ("416", "384"):
        dfm = pick_dfm(which)
        sa = run_export("aligned", dfm, nframes, "al_%s" % which)
        sf = run_export("facelib", dfm, nframes, "fc_%s" % which)
        st = cmp_dirs(sf["_dir"], sa["_dir"], which)
        out[which] = {"aligned": sa, "facelib": sf, "cmp": st}
        say()
        say("  [%s] %s" % (which, Path(dfm).name))
        if st.get("error"):
            check(False, "像素对拍跑完", st["error"])
            continue
        say("    帧数 %d · mean %.4f LSB · max %d · 差异像素 %.2f%% · 逐字节相同 %d 帧"
            % (st["frames"], st["mean_lsb"], st["max_lsb"], st["diff_px_pct"],
               st["byte_identical"]))
        say("    最差帧 %s（mean %.4f / max %d）" % (st["worst"][0], st["worst"][1],
                                                   st["worst"][2]))
        say("    相位相关位移 max %.4f px · 单帧差异面积上限 %s%%"
            % (st["shift_px_max"], st["face_area_pct_max"]))
        say("    差异幅度：强边缘上 mean %s LSB · 平坦处 mean %s LSB"
            % (st["mean_abs_on_edge"], st["mean_abs_off_edge"]))
        say("    >16 LSB 的差异像素 %d 个，其中 %s%% 落在原帧的强边缘上"
            % (st["big_diff_px"], st["big_on_edge_pct"]))
        if st.get("diff_png"):
            say("    差异图：%s" % st["diff_png"])
            say("    局部放大：%s" % st["crop_png"])
        check(st["mean_lsb"] <= TOL_MEAN,
              "两条路有差异（换了一张画布），但均值在 %.0f LSB 内" % TOL_MEAN,
              "mean %.4f LSB" % st["mean_lsb"])
        check(st["shift_px_max"] < 0.5, "差异不是整体错位（相位相关位移 < 0.5px）",
              "max %.4f px" % st["shift_px_max"])
        check(st["diff_px_pct"] < 5.0, "差异像素占比 < 5%（只有脸上一小片，不是整张脸）",
              "%.2f%%" % st["diff_px_pct"])
        check((st["big_on_edge_pct"] or 0) > 80.0,
              ">16 LSB 的大差异几乎全在原帧强边缘上（重采样/遮罩边界的指纹，不是整体错位）",
              "%s%% 落在梯度 >24 的像素上" % st["big_on_edge_pct"])
        check(st["byte_identical"] < st["frames"],
              "确实不是逐字节相同（预期：画布换了，重采样的落点就不一样）",
              "%d/%d 逐字节相同" % (st["byte_identical"], st["frames"]))
    (OUT / "pixel.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "pixel.json").write_text(json.dumps(
        {k: {"cmp": v["cmp"], "aligned_fps": v["aligned"].get("frame_fps"),
             "facelib_fps": v["facelib"].get("frame_fps")} for k, v in out.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------
# 帧速
# ------------------------------------------------------------
def case_fps(nframes, reps):
    head("3 · 帧速：384 / 416，facelib 画布 vs aligned 原生（各 %d 次取中位）" % reps)
    table = {}
    for which in ("416", "384"):
        dfm = pick_dfm(which)
        for mode in ("facelib", "aligned"):
            fps, seg = [], []
            for k in range(reps):
                s = run_export(mode, dfm, nframes, "fps_%s_%s_%d" % (which, mode, k))
                fps.append(float(s.get("frame_fps") or 0.0))
                seg.append(s.get("profile") or {})
            fps.sort()
            med = fps[len(fps) // 2] if fps else 0.0
            table.setdefault(which, {})[mode] = {"fps": fps, "median": med,
                                                "profile": seg[-1] if seg else {},
                                                "n": nframes}
            grp = seg[-1] if seg else {}
            say("  [%s] %-8s 帧/秒 %s（中位 %.2f）  合成 %.1fms/帧 · 读帧 %.1f · 编码 %.1f"
                % (which, mode, "/".join("%.2f" % v for v in fps), med,
                   float(grp.get("compose") or 0.0), float(grp.get("prep") or 0.0),
                   float(grp.get("encode") or 0.0)))
    say()
    say("  %-6s %-10s %-10s %-8s" % ("模型", "facelib", "aligned", "变化"))
    all_ok = True
    for which, row in table.items():
        f = row["facelib"]["median"]
        a = row["aligned"]["median"]
        delta = (100.0 * (a - f) / f) if f else 0.0
        say("  %-6s %-10.2f %-10.2f %+.1f%%" % (which, f, a, delta))
        all_ok = all_ok and (a >= f * 0.97)
    check(all_ok, "aligned 原生不比 facelib 慢（≥ 97%）", "")
    (OUT / "fps.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "fps.json").write_text(json.dumps(table, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="pos,pixel,fps")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    cases = [c.strip() for c in str(a.cases).split(",") if c.strip()]
    BOX.mkdir(parents=True, exist_ok=True)
    build_fixture(a.frames)
    sys.path.insert(0, str(APP))
    t0 = time.time()
    if "pos" in cases:
        case_pos()
    if "pixel" in cases:
        case_pixel(a.frames)
    if "fps" in cases:
        case_fps(a.frames, max(1, a.reps))
    ok = sum(1 for r in RESULTS if r["ok"])
    head("结果：%d PASS / %d FAIL（%.1fs）" % (ok, len(RESULTS) - ok, time.time() - t0))
    for r in RESULTS:
        if not r["ok"]:
            print("  FAIL  %s  —  %s" % (r["label"], r["detail"]))
    (BOX / "result.json").write_text(json.dumps({"results": RESULTS}, ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    print("报告 →", BOX / "result.json")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
