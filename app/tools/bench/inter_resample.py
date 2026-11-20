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


def resample_2d(block: np.ndarray, n_new: int, method: str) -> np.ndarray:
    """block: (n, n, C) → (n_new, n_new, C)。按轴分别左乘/右乘插值矩阵。"""
    n = block.shape[0]
    M = _grid(n, n_new, method)                      # (n_new, n)
    tmp = np.tensordot(M, block, axes=(1, 0))        # (n_new, n, C)
    out = np.tensordot(tmp, M, axes=(1, 1))          # (n_new, n_new, C)? 需要转置处理
    return np.transpose(out, (0, 2, 1))


def convert(src, res_from: int, res_to: int, method: str, out) -> dict:
    arrs = read_pth(src)
    if len(arrs) < 4:
        raise SystemExit("inter 里只有 %d 个数组，不是期望的 4 个（dense1/dense2 各 w+b）"
                         % len(arrs))
    d1w, d1b, d2w, d2b = arrs[0], arrs[1], arrs[2], arrs[3]
    ae = int(d1w.shape[1])
    ldr_from = int(round((d1w.shape[0] / ae) ** 0.5)) if False else None
    # dense1 行数 = ldr² × C1；C1 由 dense2 的输出通道数推出（df: ae，liae: 2*ae）
    rows = int(d1w.shape[0])
    cols2 = int(d2w.shape[1])
    # 猜测 C1：rows / ldr² 必须是整数，且 ldr 与 res 的关系 = res/32
    ldr_from = max(1, res_from // 32)
    ldr_to = max(1, res_to // 32)
    if rows % (ldr_from * ldr_from) != 0:
        raise SystemExit("dense1 行数 %d 不能被 ldr²=%d 整除，模型或许不是 32 倍下采样"
                         % (rows, ldr_from * ldr_from))
    c1 = rows // (ldr_from * ldr_from)
    if cols2 % (ldr_from * ldr_from) != 0:
        raise SystemExit("dense2 列数 %d 不能被 ldr²=%d 整除" % (cols2, ldr_from * ldr_from))
    c2 = cols2 // (ldr_from * ldr_from)

    b1 = d1w.reshape(ldr_from, ldr_from, c1, ae)
    b2 = d2w.reshape(ae, ldr_from, ldr_from, c2)

    def _rs4(b, n_old, n_new):
        """(h,w,c,ae) 或 (ae,h,w,c) 统一处理：把 h,w 两个轴重采样。"""
        M = _grid(n_old, n_new, method)
        if b.shape[0] == n_old and b.shape[1] == n_old:
            t = np.tensordot(M, b, axes=(1, 0))
            t = np.tensordot(t, M, axes=(1, 0))
            return np.transpose(t, (0, 1, 2, 3))
        if b.shape[1] == n_old and b.shape[2] == n_old:
            t = np.tensordot(b, M.T, axes=(1, 0))     # (ae, n_new, w, c)? 保守写法见下
            raise RuntimeError("dense2 走专门分支")
        raise RuntimeError("未知布局")

    # dense1: (ldr, ldr, c1, ae)
    M = _grid(ldr_from, ldr_to, method)
    t = np.tensordot(M, b1, axes=(1, 0))              # (n_new, ldr, c1, ae)
    t = np.einsum("xy,yzca->xzca", M, t.transpose(1, 0, 2, 3))
    d1w_new = np.ascontiguousarray(t).reshape(-1, ae)

    # dense2: (ae, ldr, ldr, c2)
    M2 = _grid(ldr_from, ldr_to, method)
    t2 = np.einsum("ab,bhwd->ahwd", np.eye(ae), b2)   # 占位，保持形状
    t2 = np.einsum("x h w d -> h x w d", b2)          # (ldr, ae, ldr, c2) 不必要，简化：
    # 直接对 h 轴：sum_h M[x,h] * b[:, h, w, :]
    th = np.einsum("xh,ahwd->axwd", M2, b2)           # (ae, n_new, ldr, c2)
    tw = np.einsum("yw,axwd->axyd", M2, th)           # (ae, n_new, n_new, c2)
    d2w_new = np.ascontiguousarray(tw).reshape(ae, -1)
    # bias 与 dense2 的列一一对应 → 同样按 (ldr, ldr, c2) 重采样后展平
    b2b = d2b.reshape(ldr_from, ldr_from, c2)
    tb = np.einsum("xh,hwd->xwd", M2, b2b)
    tb = np.einsum("yw,xwd->xyd", M2, tb)
    d2b_new = np.ascontiguousarray(tb).reshape(-1)

    write_pth(out, [d1w_new.astype(np.float32), d1b.astype(np.float32),
                    d2w_new.astype(np.float32), d2b_new.astype(np.float32)])
    return {"from": (ldr_from, c1, c2, ae), "to": (ldr_to,),
            "dense1": (d1w.shape, d1w_new.shape),
            "dense2": (d2w.shape, d2w_new.shape),
            "bias": (d2b.shape, d2b_new.shape)}


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
