# -*- coding: utf-8 -*-
r"""对齐质量验收（**最终那张对齐脸**，不是检测框、也不是 68 点像素差）。

为什么再造一个脚本
────────────────────────────────────────────────────────────────────────
之前的验收全在量中间量：5 点位移、68 点位移、对齐脸 max LSB、成片 md5。
它们能证明"换 EP 没改坏"，回答不了"**最终对齐出来的那张脸好不好**"：

  68 点差 1.5 px 到底意味着脸歪了还是只是点抖了？抖动 0.42° 算飘还是不算？
  粗裁 INTER_LINEAR 把画面下采样 2.7 倍，aliasing 到底有没有糊到 chip 上？
  我们的终裁矩阵和 iperov 的差 4e-4 px，那这张脸算不算"和参考实现一模一样"？

所以这里只量四件事，全部在**对齐脸坐标系**里量（单位：芯片像素 / 度）：

  A 几何自洽 —— 眼线相对水平线夹角、鼻尖到画布中心偏移（÷画布边）、眼距÷画布宽。
                三者都拿 facelib 模板（landmarks_2D_new 经 unit→chip 映射）当理想值。
                注意：相似变换拟合会把"整体的滚转"吃干净，所以眼线角反映的是
                **这张脸的眼眶相对它自己的下颌/鼻子的形状**与平均脸的差（含模型本身的
                系统性偏置），不是"头歪了几度"。抖动才是噪声；绝对值是偏置。
  B 时间稳定 —— A 的三项在**相邻帧**之间的一阶差分（抖动）。这里的关键是：
                这段素材里人在动，一阶差分 = 真实运动 + 噪声，真实运动会把三组
                之间的差**淹掉**。所以同时算二阶差分 |x[i+2]-2x[i+1]+x[i]|（曲率），
                再除以 √6 —— 那是**逐帧白噪声的估计值**，真实运动在二阶差分里基本抵消。
                判"哪组更稳"看噪声底，不看一阶差分。
  C 清晰度   —— 对齐脸 Laplacian 方差；以及它和"直接拿 facelib 矩阵从**原始全分辨率**
                重裁一次"的比值（这个问题本身就是答案：chip 从来不是从 192 粗裁图放大的，
                两条路只差插值舍入）。另外量粗裁图自己有多 aliased：
                与"8× 超采样 + INTER_AREA"的带限参考比 PSNR。
  D 参考一致性 —— 同一组 68 点走 facelib/iperov 的 get_transform_mat(WHOLE_FACE)，
                比 zoom / rot / 画布四角位移 / 两张芯片的像素差。

三组对照（全部离线，**不改生产代码**）
  ① 现状      ：kps round(0.1) + 粗裁 INTER_LINEAR
  ② 去 round  ：kps 保留 float（round 只在上层存盘时做）+ 粗裁 INTER_LINEAR
  ③ ② + AREA  ：kps float + 粗裁改成「4× 超采样 → INTER_AREA」（真·面积平均）

  **③ 为什么不是直接把 warpAffine 的 flags 换成 INTER_AREA**：本机 cv2 5.0.0 上
  warpAffine(INTER_AREA) 与 INTER_LINEAR **逐位相同**（--selftest 会验），直接换
  flag 等于什么都没改，还以为是"面积平均"。真面积平均只能超采样再 cv2.resize(AREA)。

整条链路（一个字都没改，只是把它拆开量）：
    检测器 5 点 → coarse_align_from_5(192, ×1.3) → 2D106 → landmark106to68
    → landmarks68_in_source → align_face_68（33 点拟合 facelib landmarks_2D_new，INTER_CUBIC）

跑法（仓库根，CPU 即可；不接 dfm / 不合成 / 不导视频）：
    runtime\server\Scripts\python.exe app\tools\accept\align_chip_quality.py ^
        --start 29 --end 135 --min-eyed5 0.25 --json %TEMP%\alignq.json
退出码：0 = 量出来了；2 = 一帧合格的都没取到 / 依赖缺失。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
for _p in (str(APP), str(APP / "engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DFL_WORKSPACE", str(REPO / "workspace"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                             # noqa: E402

SS = 4          # ③ 粗裁的超采样倍数（4×4 块做面积平均）
SS_REF = 8      # 粗裁的带限参考倍数（C：粗裁图本身有多 aliased）


# ---------------------------------------------------------------- 分布工具
def _dist(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    s = sorted(v)
    def q(p):
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return {"n": len(v), "med": statistics.median(s), "p95": q(0.95),
            "max": s[-1], "min": s[0], "mean": sum(s) / len(s)}


def _f(d, nd=3):
    if not d or d.get("n", 0) == 0:
        return "-"
    return "%.*f / %.*f / %.*f" % (nd, d["med"], nd, d["p95"], nd, d["max"])


# ---------------------------------------------------------------- 取帧
def _decode(video):
    """和抽取同一条 ffmpeg rawvideo/bgr24 口径（不经 cv2 解码，颜色/锐度不掺私货）。"""
    import cv2
    import imageio_ffmpeg
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-loglevel", "quiet", "-i", str(video), "-vsync", "0",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    fsz, shape = W * H * 3, (H, W, 3)
    i = 0
    while True:
        b = proc.stdout.read(fsz)
        if len(b) < fsz:
            break
        yield i, np.frombuffer(b, np.uint8).reshape(shape).copy()
        i += 1
    try:
        proc.stdout.close()
        proc.wait(timeout=1)
    except Exception:                                          # noqa: BLE001
        proc.kill()


# ---------------------------------------------------------------- 检测（可关量化）
def faces_from_fwd(det, fwd, ctx, quant, min_face=16):
    """detect._faces_from_fwd 的**逐行副本**，只把 round(1) 换成可关。

    quant=1 时与生产逐位一致（--selftest 会核对），quant=None 时保留 float。
    """
    if len(fwd) == 3:
        scores, boxes, kpss = fwd
    else:
        scores, boxes = fwd
        kpss = np.zeros((scores.shape[0], 5, 2), np.float32)
    if scores.size == 0:
        return []
    scale = float(ctx["scale"])
    W, H = int(ctx["w"]), int(ctx["h"])
    order = scores.argsort()[::-1]
    dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
    kpss = kpss[order]
    keep = det._nms(dets, det.nms_thresh)
    out = []
    for i in keep:
        d = dets[i]
        x1 = max(0.0, d[0] / scale)
        y1 = max(0.0, d[1] / scale)
        x2 = min(float(W), d[2] / scale)
        y2 = min(float(H), d[3] / scale)
        if min(x2 - x1, y2 - y1) < min_face:
            continue
        kp = kpss[i] / scale
        if quant:
            box = [round(float(x1), 1), round(float(y1), 1),
                   round(float(x2), 1), round(float(y2), 1)]
            k5 = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp]
        else:
            box = [float(x1), float(y1), float(x2), float(y2)]
            k5 = [[float(p[0]), float(p[1])] for p in kp]
        out.append({"box": box, "kps": k5, "score": round(float(d[4]), 4)})
    out.sort(key=lambda x: -x["score"])
    return out


# ---------------------------------------------------------------- 几何小工具
def _lapvar(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F, ksize=3).var())


def _aff3(mat2):
    return np.vstack([np.asarray(mat2, np.float64), [0.0, 0.0, 1.0]])


def _corner_disp(m1, m2, size):
    """两张芯片矩阵在画布四角上的位移（源图像素）：芯片四角反变回原图再比。"""
    c = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64)
    p1 = (np.linalg.inv(_aff3(m1)) @ np.c_[c, np.ones(4)].T).T[:, :2]
    p2 = (np.linalg.inv(_aff3(m2)) @ np.c_[c, np.ones(4)].T).T[:, :2]
    return float(np.linalg.norm(p1 - p2, axis=1).max())


def _ordered5(pts5):
    """检测器 5 点左右反的时候修回来（对齐内核 _ordered_5 的同款逻辑）。"""
    k = np.asarray(pts5, np.float64).copy()
    if k[1, 0] < k[0, 0]:
        k[[0, 1]] = k[[1, 0]]
    if k[4, 0] < k[3, 0]:
        k[[3, 4]] = k[[4, 3]]
    return k


def _psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------- 一帧 × 一组
def run_variant(img, kps, box, side, cmode, AL, lm, cv2, UNIT):
    cmat = AL.coarse_align_from_5(kps)
    if cmode == "linear":
        coarse = cv2.warpAffine(img, cmat, (AL.COARSE_SIZE, AL.COARSE_SIZE),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:                                          # area_ss：超采样 + INTER_AREA
        M = np.asarray(cmat, np.float64).copy()
        M[:, :2] *= SS
        M[:, 2] *= SS
        big = cv2.warpAffine(img, M, (AL.COARSE_SIZE * SS, AL.COARSE_SIZE * SS),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        coarse = cv2.resize(big, (AL.COARSE_SIZE, AL.COARSE_SIZE),
                            interpolation=cv2.INTER_AREA)
    # 粗裁的带限参考（8× 超采样 → AREA）：只看粗裁图 aliased 到什么程度
    M = np.asarray(cmat, np.float64).copy()
    M[:, :2] *= SS_REF
    M[:, 2] *= SS_REF
    big = cv2.warpAffine(img, M, (AL.COARSE_SIZE * SS_REF, AL.COARSE_SIZE * SS_REF),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cref = cv2.resize(big, (AL.COARSE_SIZE, AL.COARSE_SIZE), interpolation=cv2.INTER_AREA)

    l68 = lm.landmarks68(coarse)
    pts = AL.landmarks68_in_source(l68, cmat)
    chip, meta = AL.align_face_68(img, pts, box=box, size=side)
    face_frac = float(max(l68[:, 0].max() - l68[:, 0].min(),
                          l68[:, 1].max() - l68[:, 1].min()) / float(AL.COARSE_SIZE))
    return {"cmat": cmat, "coarse": coarse, "cref": cref, "coarse_psnr": _psnr(coarse, cref),
            "coarse_lapvar": _lapvar(coarse), "l68": l68, "pts": pts, "chip": chip,
            "meta": meta, "face_frac": face_frac}


def measure(v, img, side, AL, fx_gtm, FaceType, cv2, UNIT):
    chip, meta = v["chip"], v["meta"]
    mat = np.asarray(meta["image_to_face_mat"], np.float64)
    _m, T, resid, ppu = AL.align_fit(v["pts"], side)
    u2c = (_aff3(mat) @ np.linalg.inv(T))[0:2]          # 单位方形 → 芯片（纯相似）
    lc = AL.transform_points(v["pts"], mat)             # 芯片里的 68 点（全精度）
    el, er = lc[36:42].mean(0), lc[42:48].mean(0)
    d = er - el
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    iod = float(np.hypot(*d) / float(side))
    nose = float(np.hypot(lc[30][0] - side / 2.0, lc[30][1] - side / 2.0) / float(side))
    # 模板理想值：UNIT_68 经同一条 unit→chip 相似变换
    tlc = AL.transform_points(UNIT, u2c)
    tel, ter = tlc[36:42].mean(0), tlc[42:48].mean(0)
    td = ter - tel
    t_ang = float(np.degrees(np.arctan2(td[1], td[0])))
    t_iod = float(np.hypot(*td) / float(side))
    t_nose = float(np.hypot(tlc[30][0] - side / 2.0, tlc[30][1] - side / 2.0) / float(side))

    # C：芯片锐度，以及与「facelib 矩阵 + 原始全分辨率重裁」的对照
    rmat = np.asarray(fx_gtm(np.asarray(v["pts"], np.float64), side, FaceType.WHOLE_FACE),
                      np.float64)
    refchip = cv2.warpAffine(img, rmat, (side, side), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    lv, lvr = _lapvar(chip), _lapvar(refchip)
    pd = np.abs(chip.astype(np.int16) - refchip.astype(np.int16))
    s_o = float(np.hypot(mat[0, 0], mat[1, 0]))
    s_r = float(np.hypot(rmat[0, 0], rmat[1, 0]))
    return {
        "ang": ang, "iod": iod, "nose": nose,
        "d_ang": ang - t_ang, "d_iod": iod - t_iod, "d_nose": nose - t_nose,
        "t_ang": t_ang, "t_iod": t_iod, "t_nose": t_nose,
        "lapvar": lv, "lapvar_ref": lvr,
        "sharp_ratio": (lv / lvr) if lvr > 0 else float("nan"),
        "refchip_max": int(pd.max()), "refchip_mean": float(pd.mean()),
        "refchip_gt2_pct": float(np.count_nonzero(pd > 2)) / pd.size * 100.0,
        "zoom_pct": (s_r / s_o - 1.0) * 100.0,
        "rot_deg": float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                         - np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))),
        "corner_px": _corner_disp(mat, rmat, side),
        "resid_max": float(resid.max()), "resid_med": float(np.median(resid)),
        "resid_rel": float(resid.max() / max(1e-6, np.median(resid))),
        "face_frac": v["face_frac"], "coarse_psnr": v["coarse_psnr"],
        "coarse_lapvar": v["coarse_lapvar"],
        "px_per_src": float(np.sqrt(abs(np.linalg.det(mat[:, :2])))),
        "side": side,
    }


def jitter(rs, keys):
    """一阶差分（含真实运动）与二阶差分/√6（≈ 逐帧白噪声底）。"""
    out = {}
    for k in keys:
        d1, d2 = [], []
        for a, b in zip(rs, rs[1:]):
            if b["i"] - a["i"] == 1:
                d1.append(abs(b[k] - a[k]))
        for a, b, c in zip(rs, rs[1:], rs[2:]):
            if b["i"] - a["i"] == 1 and c["i"] - b["i"] == 1:
                d2.append(abs(c[k] - 2.0 * b[k] + a[k]))
        out[k] = {"d1": _dist(d1), "d2": _dist(d2),
                  "noise": _dist([x / math.sqrt(6.0) for x in d2])}
    out["pairs"] = sum(1 for a, b in zip(rs, rs[1:]) if b["i"] - a["i"] == 1)
    return out


def paired(ra, rb, key):
    """成对比较：逐帧对 |Δ| 谁更小。n=50 上下时比 p95 可靠得多。"""
    wa = wb = tie = 0
    diff = []
    for (a1, a2), (b1, b2) in zip(zip(ra, ra[1:]), zip(rb, rb[1:])):
        if a2["i"] - a1["i"] != 1 or b2["i"] - b1["i"] != 1:
            continue
        if a1["i"] != b1["i"]:
            continue
        da, db = abs(a2[key] - a1[key]), abs(b2[key] - b1[key])
        if abs(da - db) < 1e-12:
            tie += 1
        elif db < da:
            wb += 1
        else:
            wa += 1
        diff.append(db - da)
    n = wa + wb
    return {"a_better": wa, "b_better": wb, "tie": tie, "n": n,
            "b_win_pct": (100.0 * wb / n) if n else float("nan"),
            "d_med": statistics.median(diff) if diff else float("nan")}


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=int, default=29)
    ap.add_argument("--end", type=int, default=135)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--min-eyed5", type=float, default=0.25,
                    help="检测器 5 点「眼距/框宽」下限：低于它说明这帧其实是后脑/侧脸，"
                         "5 点是退化的，粗裁视野是错的（whole_face 对齐在那种帧上没有意义）")
    ap.add_argument("--min-boxh", type=float, default=180.0)
    ap.add_argument("--det", default="retinaface_10g")
    ap.add_argument("--json", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    import cv2
    from server._vision import align as AL
    from facelib.LandmarksProcessor import get_transform_mat as fx_gtm
    from facelib.FaceType import FaceType
    from server._landmarks106 import InsightFace106
    from server._vision.detect import RetinaFaceDet

    video = Path(a.video) if a.video else sorted((REPO / "workspace").glob("*.mp4"))[0]
    onnx = REPO / "weights" / ("%s.onnx" % a.det)
    lm106_path = REPO / "weights" / "insightface_2d106.onnx"
    for p in (video, onnx, lm106_path):
        if not p.is_file():
            print("退出码 2：缺文件 %s" % p)
            return 2
    CPU = ["CPUExecutionProvider"]
    det = RetinaFaceDet(str(onnx), 640, providers=CPU)      # 不建 TRT/CUDA 会话
    lm = InsightFace106(str(lm106_path), providers=CPU)

    if a.selftest:
        return _selftest(cv2, det, faces_from_fwd)

    UNIT = AL.UNIT_68
    VARIANTS = (("① 现状", 1, "linear"), ("② 去round", None, "linear"),
                ("③ +AREA", None, "area_ss"))

    pick = []          # 合格帧：(idx, img, faces_rounded, faces_raw, stats)
    diag = []          # 不合格帧（后脑/侧脸）：只留诊断量
    t0 = time.time()
    scanned = 0
    for idx, img in _decode(video):
        scanned = idx + 1
        if idx < a.start:
            continue
        if idx >= a.end or len(pick) >= a.frames:
            break
        ctx = det.prep(img)
        fwd = det._forward(ctx["canvas"])
        fq = faces_from_fwd(det, fwd, ctx, quant=1)
        fr = faces_from_fwd(det, fwd, ctx, quant=None)
        area = [((f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), k)
                for k, f in enumerate(fq)
                if min(f["box"][2] - f["box"][0], f["box"][3] - f["box"][1]) >= 32]
        if not area:
            continue
        j = max(area)[1]
        bx, kp = fq[j]["box"], np.asarray(fq[j]["kps"], np.float64)
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        ed = float(np.linalg.norm(kp[1] - kp[0]) / max(1e-6, bw))
        okframe = (ed >= a.min_eyed5 and bh >= a.min_boxh)
        (pick if okframe else diag).append((idx, img, fq[j], fr[j], ed, bh))
        if (len(pick) + len(diag)) % 20 == 0:
            print("  扫到第 %d 帧：合格 %d / 退化 %d（%.1fs）"
                  % (idx, len(pick), len(diag), time.time() - t0), flush=True)
    if not pick:
        print("退出码 2：%d..%d 里没有一帧 5 点合格的（眼距/框宽 ≥ %.2f）"
              % (a.start, a.end, a.min_eyed5))
        return 2
    print("[取帧] %s：%d..%d 扫 %d 帧 → 合格 %d 帧（眼距/框宽≥%.2f, 框高≥%.0f）·"
          " 退化 %d 帧（后脑/侧脸，单独看）· %.1fs"
          % (video.name, a.start, a.end, scanned, len(pick), a.min_eyed5, a.min_boxh,
             len(diag), time.time() - t0), flush=True)

    per, diag_rows = {}, []
    d68 = {"round_vs_float": [], "linear_vs_area": []}
    for name, _, _ in VARIANTS:
        per[name] = []

    def chain_one(img, fq, fr):
        bx = fq["box"]
        bw = max(1.0, bx[2] - bx[0])
        bh = max(1.0, bx[3] - bx[1])
        side = int(round(max(bw, bh) * 2.0))      # 生产口径：2×框，偶数化
        side -= side % 2
        return side

    for n, (idx, img, fq, fr, ed, bh) in enumerate(pick):
        side = chain_one(img, fq, fr)
        got = {}
        for name, quant, cmode in VARIANTS:
            kps = _ordered5(fq["kps"] if quant else fr["kps"])
            got[name] = run_variant(img, kps, fq["box"], side, cmode, AL, lm, cv2, UNIT)
        d68["round_vs_float"].append(
            float(np.linalg.norm(got["② 去round"]["pts"] - got["① 现状"]["pts"], axis=1).max()))
        d68["linear_vs_area"].append(
            float(np.linalg.norm(got["③ +AREA"]["pts"] - got["② 去round"]["pts"], axis=1).max()))
        for name, _, _ in VARIANTS:
            v = got[name]
            if v["chip"] is None:
                continue
            r = measure(v, img, side, AL, fx_gtm, FaceType, cv2, UNIT)
            r["i"] = idx
            per[name].append(r)
        if (n + 1) % 10 == 0:
            print("  量完 %d/%d 帧（%d）%.1fs" % (n + 1, len(pick), idx, time.time() - t0),
                  flush=True)

    # 退化帧：只留「粗裁视野被 5 点拖成什么样」+ 68 点残差（诊断用，不进 A~D 结论）
    for idx, img, fq, fr, ed, bh in diag[:80]:
        side = chain_one(img, fq, fr)
        try:
            v = run_variant(img, _ordered5(fr["kps"]), fq["box"], side, "linear",
                            AL, lm, cv2, UNIT)
            if v["chip"] is None:
                continue
            r = measure(v, img, side, AL, fx_gtm, FaceType, cv2, UNIT)
            r.update({"i": idx, "eyed5": ed})
            diag_rows.append(r)
        except Exception:                                       # noqa: BLE001
            continue

    KEYS = ("ang", "iod", "nose", "d_ang", "d_iod", "d_nose", "lapvar", "lapvar_ref",
            "sharp_ratio", "refchip_max", "refchip_mean", "refchip_gt2_pct", "zoom_pct",
            "rot_deg", "corner_px", "resid_max", "resid_med", "resid_rel", "face_frac",
            "coarse_psnr", "coarse_lapvar", "px_per_src", "side")
    agg = {k: {m: _dist([r[m] for r in per[k]]) for m in KEYS} for k, _, _ in VARIANTS}
    jit = {k: jitter(per[k], ("ang", "iod", "nose", "lapvar")) for k, _, _ in VARIANTS}

    r0 = per["① 现状"][0]
    print("")
    print("=" * 104)
    print("对齐质量（最终那张脸，芯片坐标）：%s · %d 帧 · %d..%d · 芯片 %d~%d px · 平均放大 %.2f×"
          % (video.name, len(per["① 现状"]), pick[0][0], pick[-1][0],
             agg["① 现状"]["side"]["min"], agg["① 现状"]["side"]["max"],
             agg["① 现状"]["px_per_src"]["med"]))
    print("模板理想值：眼线 %.4f° · 眼距/画布 %.4f · 鼻尖偏心/画布 %.4f"
          % (r0["t_ang"], r0["t_iod"], r0["t_nose"]))
    print("")
    print("A 几何自洽 —— 中位 / p95 / max")
    for name, _, _ in VARIANTS:
        A = agg[name]
        print("  %-9s 眼线 %s °（vs 模板 %s °）· 鼻尖偏心 %s（vs 模板 %s）· 眼距/画布 %s（vs 模板 %s）"
              % (name, _f(A["ang"]), _f(A["d_ang"]), _f(A["nose"], 4),
                 _f(A["d_nose"], 4), _f(A["iod"], 4), _f(A["d_iod"], 4)))
    print("")
    print("B 时间稳定 —— 相邻帧一阶差分（中位/p95/max）｜二阶差分/√6 = 逐帧噪声底")
    for name, _, _ in VARIANTS:
        J = jit[name]
        print("  %-9s 眼线 %s ° ｜噪声 %s °" % (name, _f(J["ang"]["d1"], 3),
                                               _f(J["ang"]["noise"], 3)))
        print("  %-9s 鼻尖   %s   ｜噪声 %s  " % ("", _f(J["nose"]["d1"], 5),
                                                 _f(J["nose"]["noise"], 5)))
        print("  %-9s 眼距   %s   ｜噪声 %s  " % ("", _f(J["iod"]["d1"], 5),
                                                 _f(J["iod"]["noise"], 5)))
    print("")
    print("C 清晰度 —— 芯片 lapvar / 与「原始全分辨率重裁」的对照 / 粗裁图自身 aliasing")
    for name, _, _ in VARIANTS:
        A = agg[name]
        print("  %-9s 芯片 lapvar %s ｜重裁参考 %s ｜比值 %s  ｜像素差 max %s LSB"
              % (name, _f(A["lapvar"], 1), _f(A["lapvar_ref"], 1), _f(A["sharp_ratio"], 5),
                 _f(A["refchip_max"], 0)))
        print("  %-9s 粗裁 lapvar %s ｜与 8× 带限参考 PSNR %s dB"
              % ("", _f(A["coarse_lapvar"], 1), _f(A["coarse_psnr"], 2)))
    print("")
    print("D 与 facelib get_transform_mat(WHOLE_FACE) 的一致性 —— 中位 / p95 / max")
    for name, _, _ in VARIANTS:
        A = agg[name]
        print("  %-9s zoom %s %% ｜rot %s ° ｜四角 %s px ｜参考芯片 >2LSB %s%%"
              % (name, _f(A["zoom_pct"], 6), _f(A["rot_deg"], 6), _f(A["corner_px"], 6),
                 _f(A["refchip_gt2_pct"], 4)))
    print("")
    print("")
    print("B' 成对比较：逐帧对 |Δ| 谁更小（①vs② / ①vs③）—— n 只有 50 上下，p95 不可靠，看这个")
    for k, lab in (("ang", "眼线°"), ("nose", "鼻尖"), ("iod", "眼距")):
        p12 = paired(per["① 现状"], per["② 去round"], k)
        p13 = paired(per["① 现状"], per["③ +AREA"], k)
        print("  %-6s ①vs② ②赢 %5.1f%%（%d/%d，Δ中位 %+.5f）｜①vs③ ③赢 %5.1f%%（%d/%d，Δ中位 %+.5f）"
              % (lab, p12["b_win_pct"], p12["b_better"], p12["n"], p12["d_med"],
                 p13["b_win_pct"], p13["b_better"], p13["n"], p13["d_med"]))
    print("中间量（只解释 A~D，不当结论）：①→② 68 点 |Δ| %s px ｜②→③ %s px（源图）"
          % (_f(_dist(d68["round_vs_float"]), 3), _f(_dist(d68["linear_vs_area"]), 3)))
    print("脸颊占粗裁画布：%s（%.2f = 106 的理想输入，实测 %.2f）"
          % (_f(agg["① 现状"]["face_frac"], 3), 0.6, agg["① 现状"]["face_frac"]["med"]))
    diag_jit = None
    if diag_rows:
        drs = [{"i": r["i"], "ang": r["ang"], "nose": r["nose"], "iod": r["iod"]}
               for r in diag_rows]
        diag_jit = jitter(drs, ("ang", "nose", "iod"))
        print("退化帧（5 点眼距/框宽 < %.2f，共 %d 帧，%d..%d）：脸颊占粗裁画布 %s ｜"
              "眼线一阶差分 %s ° ｜噪声底 %s °"
              % (a.min_eyed5, len(diag), diag_rows[0]["i"], diag_rows[-1]["i"],
                 _f(_dist([r["face_frac"] for r in diag_rows]), 3),
                 _f(diag_jit["ang"]["d1"], 3), _f(diag_jit["ang"]["noise"], 3)))
    print("=" * 104)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"video": video.name, "frames": len(per["① 现状"]),
             "range": [pick[0][0], pick[-1][0]],
             "gate": {"min_eyed5": a.min_eyed5, "min_boxh": a.min_boxh},
             "template": {"ang": r0["t_ang"], "iod": r0["t_iod"], "nose": r0["t_nose"]},
             "agg": agg, "jitter": jit,
             "d68": {k: _dist(v) for k, v in d68.items()},
             "paired": {("%s|%s|%s" % (x, y, k)): paired(per[x], per[y], k)
                        for x, y in (("① 现状", "② 去round"), ("① 现状", "③ +AREA"),
                                     ("② 去round", "③ +AREA"))
                        for k in ("ang", "nose", "iod", "lapvar")},
             "diag_jitter": diag_jit,
             "diag": {"n": len(diag_rows),
                      "face_frac": _dist([r["face_frac"] for r in diag_rows]),
                      "resid_med": _dist([r["resid_med"] for r in diag_rows]),
                      "ang": _dist([r["ang"] for r in diag_rows])},
             "per": per}, ensure_ascii=False, indent=1, default=float), "utf-8")
        print("JSON：%s" % a.json)
    return 0


def _selftest(cv2, det, faces_from_fwd):
    """两条自证：① 我的 decoded 副本 quant=1 必须等于生产 detect()；② AREA flag 是空的。"""
    src = (np.random.RandomState(0).rand(600, 800, 3) * 255).astype(np.uint8)
    M = np.array([[0.32, 0.0, 10.0], [0.0, 0.32, 5.0]], np.float64)
    L = cv2.warpAffine(src, M, (192, 192), flags=cv2.INTER_LINEAR)
    A = cv2.warpAffine(src, M, (192, 192), flags=cv2.INTER_AREA)
    print("自证① warpAffine(INTER_AREA)==INTER_LINEAR ? %s（True = 直接换 flag 等于没改）"
          % bool(np.array_equal(A, L)))
    v = sorted((REPO / "workspace").glob("*.mp4"))[0]
    same, n = 0, 0
    for idx, img in _decode(v):
        if idx < 29:
            continue
        if idx > 35:
            break
        ctx = det.prep(img)
        same += int(det.detect_ctx(ctx) == faces_from_fwd(det, det._forward(ctx["canvas"]),
                                                          ctx, quant=1))
        n += 1
    print("自证② decoded 副本(quant=1) 与生产 detect() 逐位相同：%d/%d 帧" % (same, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
