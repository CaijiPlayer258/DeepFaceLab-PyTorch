# -*- coding: utf-8 -*-
r"""bf16 精度到底改了多少像素：pred 与成片两张表 + 肉眼对比图。

**这是「记录 + 回滚依据」，不是「能不能做」的门**（用户 2026-11 已拍板 bf16 可用）。
所以这里的口径要能被以后的人复查：
  · 基准 = onnxruntime CUDA fp32（现状后端）的同一张图、同一个 batch 的结果；
  · 对拍两个 bf16 档：autocast（权重 fp32、算子 bf16）与 bf16w（权重也压 bf16）；
  · 每个输出报 max|diff| / mean / p99 / >2LSB 与 >4LSB 的像素占比；
  · 成片：把 pred + dfm 自带两张遮罩喂给**真的** compose_frame（backend_merge.compose），
    再对拍合成结果 —— 差分经过遮罩混合之后是不是还被放大，这一栏才算数；
  · 肉眼：把「脸中心」「脸边缘」两块裁图并排存 PNG（原图 / onnx / bf16 / 差图×8）。

跑法（server venv；一进程一个 (model,dtype)，避免 ORT 与 torch 抢显存）：
    python app\tools\bench\dfm_bf16_quality.py --dump 416:bf16
    python app\tools\bench\dfm_bf16_quality.py --report        # 只读 npz，不碰 GPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfm_bf16_quality"
MODELS = {
    "416": REPO / "workspace" / "model" / "阿証証証証-416.dfm",
    "384": REPO / "workspace" / "model" / "张元英.dfm",
}
SIZES = {"416": 416, "384": 384}
ALIGNED = REPO / "workspace" / "merge" / "9月29日 (5)" / "aligned"
FRAMES = REPO / "workspace" / "merge" / "9月29日 (5)"
BATCH = 8


# ------------------------------------------------------------------ 输入
def real_faces(tag, n=BATCH):
    """真对齐脸（缩到模型输入尺寸）。用 imdecode：cv2.imread 打不开非 ASCII 路径。"""
    import cv2
    S = SIZES[tag]
    files = sorted(ALIGNED.glob("*.jpg"))[:max(n, 8)]
    out = np.empty((n, S, S, 3), np.float32)
    for i in range(n):
        im = cv2.imdecode(np.fromfile(str(files[i % len(files)]), np.uint8),
                          cv2.IMREAD_COLOR)
        im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
        out[i] = im.astype(np.float32) / 255.0
    return out


# ------------------------------------------------------------------ 单个后端跑一遍
def dump(tag, dtype, n=BATCH):
    path = str(MODELS[tag])
    x = real_faces(tag, n)
    out_p = BOX / ("%s_%s.npz" % (tag, dtype))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "app" / "server"))
    if dtype in ("onnx", "ortfp32"):
        # onnx = ORT CUDA fp32（**基线**）；fp32 = torch 严格 fp32（关 TF32），两回事
        from _ortcuda import add_cuda_dll_dirs
        add_cuda_dll_dirs()
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        s = ort.InferenceSession(path, sess_options=so,
                                 providers=["CUDAExecutionProvider"])
        nm = s.get_inputs()[0].name
        s.run(None, {nm: x})                                    # warm
        outs = s.run(None, {nm: x})
        ys = [np.asarray(o) for o in outs]
        del s
    else:
        from backend_models import dfm_torch as DT
        s = DT.load(path, dtype=dtype, max_batch=n)
        feed = {s.input_name: x}
        s.run(None, feed)
        ys = s.run(None, feed)
        print("[dump] %s %s · 单张 %.2f ms · %s" % (tag, dtype, s.per_face_ms(), s.vram_mb()))
        s.close()
    np.savez(out_p, **{"o%d" % i: y for i, y in enumerate(ys)})
    print("[dump] %s %s -> %s  %s" % (tag, dtype, out_p.name,
                                      [list(y.shape) for y in ys]))
    return 0


# ------------------------------------------------------------------ 统计
def stats(a, b):
    """a/b: 同一形状 → 差分的 max/mean/p99/%>2LSB/%>4LSB。

    单位：float 输出（pred / mask，0..1）乘 255 换成 LSB；**uint8 成片本身就已经是 LSB**，
    不能再乘一次（第一版就是这么把 5 LSB 写成 1275 的）。
    """
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    lsb = d if a.dtype == np.uint8 else d * 255.0
    return {"max": float(d.max()), "mean": float(d.mean()),
            "p99": float(np.percentile(d, 99.0)),
            "max_lsb": float(lsb.max()), "mean_lsb": float(lsb.mean()),
            "p99_lsb": float(np.percentile(lsb, 99.0)),
            "gt2": float((lsb > 2.0).mean() * 100.0),
            "gt4": float((lsb > 4.0).mean() * 100.0),
            "gt8": float((lsb > 8.0).mean() * 100.0)}


# ------------------------------------------------------------------ 成片对拍
def compose_pair(tag):
    """同一张真帧 + 同一组 landmarks，用两套 pred 各合成一次，对拍成片像素。

    走的是生产那一条 compose_frame（backend_merge.compose），不是自己手搓的混合 ——
    不然「bf16 在成片上会不会被放大」这个结论就没有意义。
    """
    import cv2
    sys.path.insert(0, str(REPO / "app" / "server"))
    sys.path.insert(0, str(REPO))
    from app.server.backend_merge.compose import compose_frame
    from app.server._vision.align import read_dfljpg_meta

    ok = []
    files = sorted(ALIGNED.glob("*.jpg"))
    for f in files[:40]:
        key = f.stem.split("_")[0]
        fr = FRAMES / (key + ".jpg")
        if not fr.is_file():
            continue
        meta = read_dfljpg_meta(str(f)) or {}
        lm = meta.get("source_landmarks") or meta.get("landmarks")
        if lm is None or len(np.asarray(lm).reshape(-1)) != 136:
            continue
        ok.append((f, fr, np.asarray(lm, np.float32).reshape(68, 2)))
        if len(ok) >= BATCH:
            break
    if not ok:
        print("[compose] 找不到带 source_landmarks 的对齐脸，跳过成片对拍")
        return {}

    frame = cv2.imdecode(np.fromfile(str(ok[0][1]), np.uint8), cv2.IMREAD_COLOR)
    lm = ok[0][2]
    cfg = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
           "mask_erode": 0, "mask_blur": 4, "mode": "seamless"}

    ref = np.load(BOX / ("%s_onnx.npz" % tag))
    res = {}
    for dt in ("bf16", "bf16w"):
        p = BOX / ("%s_%s.npz" % (tag, dt))
        if not p.is_file():
            continue
        cur = np.load(p)
        out = {}
        for i in range(2):                                  # 前两张脸各合成一次
            faces_r, faces_c = [], []
            for src, dst in ((ref, faces_r), (cur, faces_c)):
                dst.append({"landmarks": lm,
                            "pred": np.clip(np.asarray(src["o1"])[i], 0, 1),
                            "prd_dfm": np.squeeze(np.asarray(src["o0"])[i]),
                            "dst_dfm": np.squeeze(np.asarray(src["o2"])[i])})
            a = compose_frame(frame.copy(), faces_r, dict(cfg))
            b = compose_frame(frame.copy(), faces_c, dict(cfg))
            out["face%d" % i] = stats(a, b)
        res[dt] = out
        # 肉眼图 ①：整帧（原帧 / onnx 成片 / bf16 成片 / 差图×8）
        # 肉眼图 ②：上面那张的**脸框放大**（原帧 / onnx / bf16 / bf16w / 差图×16）——
        #   这一张才是「用户看得见的那块像素」，整帧缩到 1600 宽之后细节全没了。
        img = np.concatenate([frame, a, b,
                              np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16))
                                      * 8, 0, 255).astype(np.uint8)], axis=1)
        h, w = img.shape[:2]
        sc = 1600.0 / max(1, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imencode(".png", img)[1].tofile(str(BOX / ("compose_%s_%s.png" % (tag, dt))))
        # 脸框：按 compose 用的变换矩阵把 S×S 画布的四角投回帧坐标，再留 12px 边
        try:
            from facelib import FaceType
            from facelib.LandmarksProcessor import get_transform_mat
            m = get_transform_mat(lm, SIZES[tag], FaceType.WHOLE_FACE, scale=1.0)
            Sx = SIZES[tag]
            pts = np.array([[0, 0, 1], [Sx, 0, 1], [0, Sx, 1], [Sx, Sx, 1]], np.float64)
            q = pts @ m.T
            q = q / np.maximum(q[:, 2:3], 1e-9)
            x0 = int(max(0, q[:, 0].min() - 12)); x1 = int(min(frame.shape[1], q[:, 0].max() + 12))
            y0 = int(max(0, q[:, 1].min() - 12)); y1 = int(min(frame.shape[0], q[:, 1].max() + 12))
            tiles2 = [frame[y0:y1, x0:x1], a[y0:y1, x0:x1], b[y0:y1, x0:x1]]
            d2 = np.clip(np.abs(a[y0:y1, x0:x1].astype(np.int16)
                                - b[y0:y1, x0:x1].astype(np.int16)) * 16,
                         0, 255).astype(np.uint8)
            tiles2.append(d2)
            zoom = np.concatenate(tiles2, axis=1)
            k = max(1.0, min(3.0, 1600.0 / max(1, zoom.shape[1])))
            zoom = cv2.resize(zoom, (int(zoom.shape[1] * k), int(zoom.shape[0] * k)),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imencode(".png", zoom)[1].tofile(
                str(BOX / ("face_%s_%s.png" % (tag, dt))))
            res.setdefault("zoom", {})[dt] = os.path.basename(
                str(BOX / ("face_%s_%s.png" % (tag, dt))))
        except Exception as _e:                          # noqa: BLE001
            print("[compose] zoom 失败：%r" % (_e,))
    return res


# ------------------------------------------------------------------ 肉眼裁图
def crops(tag):
    """脸中心 + 脸边缘两块，按 原图/onnx/bf16/bf16w/差图 竖着拼。"""
    import cv2
    ref = np.load(BOX / ("%s_onnx.npz" % tag))
    S = SIZES[tag]
    center = (slice(S // 2 - 64, S // 2 + 64), slice(S // 2 - 64, S // 2 + 64))
    edge = (slice(S // 8, S // 8 + 128), slice(S // 8, S // 8 + 128))
    tiles = []
    for name, box in (("center", center), ("edge", edge)):
        row = []
        for k in ("onnx", "bf16", "bf16w"):
            p = BOX / ("%s_%s.npz" % (tag, k))
            if not p.is_file():
                continue
            y = np.clip(np.asarray(np.load(p)["o1"])[0], 0, 1)
            row.append((y[box] * 255.0).astype(np.uint8)[:, :, ::-1])
        d = None
        if len(row) >= 2:
            d = np.clip(np.abs(row[0].astype(np.int16) - row[1].astype(np.int16)) * 12,
                        0, 255).astype(np.uint8)
        if d is not None:
            row.append(d)
        tiles.append(np.concatenate(row, axis=1))
    img = np.concatenate(tiles, axis=0)
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3),
                     interpolation=cv2.INTER_NEAREST)
    out = BOX / ("crops_%s.png" % tag)
    cv2.imencode(".png", img)[1].tofile(str(out))
    return str(out)


# ------------------------------------------------------------------ 报告
def report():
    rows = []
    res = {}
    for tag in MODELS:
        refp = BOX / ("%s_onnx.npz" % tag)
        if not refp.is_file():
            continue
        ref = np.load(refp)
        o = {k: np.asarray(ref[k]) for k in ref.files}
        base = np.load(BOX / ("%s_ortfp32.npz" % tag)) if (BOX / ("%s_ortfp32.npz" % tag)).is_file() else None
        for dt in ("ortfp32", "tf32", "bf16", "bf16w"):
            p = BOX / ("%s_%s.npz" % (tag, dt))
            if not p.is_file():
                continue
            cur = np.load(p)
            st = {k: stats(np.asarray(cur[k]), o[k]) for k in sorted(o)}
            res.setdefault(tag, {})[dt] = st
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="model:dtype，例如 416:bf16")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--crops", action="store_true")
    ap.add_argument("--compose", action="store_true")
    a = ap.parse_args()
    if a.dump:
        tag, dt = a.dump.split(":")
        return dump(tag, dt)
    if a.report:
        report()
        return 0
    if a.crops:
        for tag in MODELS:
            if (BOX / ("%s_onnx.npz" % tag)).is_file():
                print("[crops]", crops(tag))
        return 0
    if a.compose:
        for tag in MODELS:
            if (BOX / ("%s_onnx.npz" % tag)).is_file():
                r = compose_pair(tag)
                print("[compose] %s %s" % (tag, json.dumps(r, ensure_ascii=False)))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
