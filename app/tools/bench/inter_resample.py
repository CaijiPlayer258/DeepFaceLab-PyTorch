# -*- coding: utf-8 -*-
"""Inter 权重换分辨率（SAEHD / df, liae）。

为什么单独一个工具：DFL 的 *_inter.pth 不是 state_dict，而是**一个 zip，
里面按顺序放 param_N.npy**（numpy 数组，无 pickle）。所以不能 torch.load，
要 zipfile + np.lib.format.read_array 读，写回也要保持同样的 zip-of-npy 结构。

形状（实测 416 / ae=320 / e_dims=80 / df-udt）：
  dense1 weight  (ldr*ldr*640, ae)   ldr = res/32 → 416 时 13×13×640 = 108160
  dense1 bias    (ae,)
  dense2 weight  (ae, ldr*ldr*C)     df: C=ae → 54080；liae: C=ae*2
  dense2 bias    (ldr*ldr*C,)

所以「换分辨率」= 把 dense1 的**行**按 (ldr, ldr, C) 重排后做 2D 重采样，
dense2 的**列**同理；bias 跟着 dense2 的列一起重采样（bias 与列一一对应）。

用法：
  python app/tools/bench/inter_resample.py --src <inter.pth> --res 416 --to 256 \
      --method bilinear --out <inter_256.pth>
方法：nearest / bilinear / bilinear_ac / bicubic / area / area_up
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

AE_FALLBACK = 320


def read_pth(path) -> list:
    """zip 里的 param_N.npy 按序号返回（保持原顺序，写回时原样）。"""
    out = []
    with zipfile.ZipFile(str(path)) as zf:
        names = sorted(zf.namelist(), key=lambda n: int("".join(ch for ch in n if ch.isdigit()) or 0))
        for nm in names:
            with zf.open(nm) as f:
                out.append(np.lib.format.read_array(f, allow_pickle=False))
    return out


def write_pth(path, arrays) -> None:
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_STORED) as zf:
        for i, a in enumerate(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a), allow_pickle=False)
            zf.writestr("param_%d.npy" % i, buf.getvalue())


def _grid(n_old: int, n_new: int, method: str) -> np.ndarray:
    """一维重采样矩阵（n_new × n_old）。全部方法都用矩阵乘，方便验证与批量。"""
    if method == "nearest":
        idx = np.clip(((np.arange(n_new) + 0.5) * n_old / n_new - 0.5).round().astype(int), 0, n_old - 1)
        M = np.zeros((n_new, n_old), np.float64)
        M[np.arange(n_new), idx] = 1.0
        return M
    if method in ("bilinear", "bilinear_ac"):
        ac = method.endswith("_ac")
        def src(i):
            if ac:
                return i * (n_old - 1) / max(1, n_new - 1)
            return (i + 0.5) * n_old / n_new - 0.5
        M = np.zeros((n_new, n_old), np.float64)
        for i in range(n_new):
            x = src(i)
            x = min(max(x, 0.0), n_old - 1.0)
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, n_old - 1)
            w = x - x0
            M[i, x0] += 1.0 - w
            M[i, x1] += w
        return M
    if method == "bicubic":
        # Catmull-Rom（a=-0.5），align_corners=False 的坐标映射
        def wfun(t):
            t = abs(t)
            if t <= 1:
                return 1.5 * t ** 3 - 2.5 * t ** 2 + 1
            if t < 2:
                return -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2
            return 0.0
        M = np.zeros((n_new, n_old), np.float64)
        for i in range(n_new):
            x = (i + 0.5) * n_old / n_new - 0.5
            x0 = int(np.floor(x))
            for k in range(-1, 3):
                j = min(max(x0 + k, 0), n_old - 1)
                M[i, j] += wfun(x - (x0 + k))
        return M
    if method == "area":
        M = np.zeros((n_new, n_old), np.float64)
        for i in range(n_new):
            a = i * n_old / n_new
            b = (i + 1) * n_old / n_new
            j0, j1 = int(np.floor(a)), int(np.ceil(b))
            for j in range(j0, min(j1, n_old)):
                lo, hi = max(a, j), min(b, j + 1)
                if hi > lo:
                    M[i, j] += (hi - lo)
            s = M[i].sum()
            if s:
                M[i] /= s
        return M
    raise SystemExit("未知方法：%s" % method)


CV2_METHODS = {"nearest": 0, "bilinear": 1, "bicubic": 2, "area": 3}


def _interp_plane(plane, n_new, method, cv2):
    import numpy as _np
    code = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}[method]
    return cv2.resize(_np.ascontiguousarray(plane, dtype=_np.float32), (n_new, n_new),
                      interpolation=code)


def convert(src, res_from: int, res_to: int, method: str, out) -> dict:
    """按 **通道优先 (c,h,w)** 解析后做空间插值（这是引擎真实的展平次序）。

    4.0 的老实现把 bias/dense2 按 (h,w,c) 解释 —— 等于把同一段内存换个形状读，
    轴序错位，出来就是花屏。这里统一走 (C,h,w)。
    """
    import cv2
    arrs = read_pth(src)
    if len(arrs) < 4:
        raise SystemExit("inter 里只有 %d 个数组，不是期望的 4 个" % len(arrs))
    d1w, d1b, d2w, d2b = arrs[0], arrs[1], arrs[2], arrs[3]
    ae = int(d1w.shape[1])
    # 有 t 时 enc = res/32，否则 res/16；本模型是 df-udt → /32
    enc_hw = max(1, res_from // 32)
    low_hw = max(1, res_from // 32)
    enc_new = max(1, res_to // 32)
    low_new = max(1, res_to // 32)

    # dense1: (enc²*C, ae) → (C, enc, enc, ae)
    rows = int(d1w.shape[0])
    c1 = rows // (enc_hw * enc_hw)
    b1 = d1w.reshape(c1, enc_hw, enc_hw, ae)
    out1 = np.zeros((c1, enc_new, enc_new, ae), np.float32)
    for c in range(c1):
        for k in range(ae):
            out1[c, :, :, k] = _interp_plane(b1[c, :, :, k], enc_new, method, cv2)
    d1w_new = out1.reshape(-1, ae)

    # dense2: (ae, low²*C) → (ae, C, low, low)
    cols = int(d2w.shape[1])
    c2 = cols // (low_hw * low_hw)
    b2 = d2w.reshape(ae, c2, low_hw, low_hw)
    out2 = np.zeros((ae, c2, low_new, low_new), np.float32)
    for a in range(ae):
        for c in range(c2):
            out2[a, c] = _interp_plane(b2[a, c], low_new, method, cv2)
    d2w_new = out2.reshape(ae, -1)

    # bias: (low²*C,) → (C, low, low)
    b3 = d2b.reshape(c2, low_hw, low_hw)
    out3 = np.zeros((c2, low_new, low_new), np.float32)
    for c in range(c2):
        out3[c] = _interp_plane(b3[c], low_new, method, cv2)
    d2b_new = out3.reshape(-1)

    write_pth(out, [d1w_new.astype(np.float32), d1b.astype(np.float32),
                    d2w_new.astype(np.float32), d2b_new.astype(np.float32)])
    return {"dense1": (d1w.shape, d1w_new.shape),
            "dense2": (d2w.shape, d2w_new.shape),
            "bias": (d2b.shape, d2b_new.shape), "grid": (enc_hw, enc_new)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--res", type=int, required=True, help="源模型分辨率")
    ap.add_argument("--to", type=int, required=True, help="目标分辨率")
    ap.add_argument("--method", default="bilinear")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = convert(a.src, a.res, a.to, a.method, a.out)
    print("method=%s  %s" % (a.method, info))


if __name__ == "__main__":
    main()
