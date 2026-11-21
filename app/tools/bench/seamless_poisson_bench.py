# -*- coding: utf-8 -*-
"""实验：cv2.seamlessClone 与 FFT/泊松解 的等价性 + 耗时。

两种实现：
  A  cv2.seamlessClone(NORMAL_CLONE)                      —— 现状（CPU）
  B  矩形 ROI 上的 DST 泊松解（Dirichlet 边界取 dst 边框） —— 频域，O(N log N)
  C  只在遮罩内求解的共轭梯度（Dirichlet 取遮罩边缘）      —— 参考实现，最接近 A 的边界条件
输出：逐像素差、PSNR、单次耗时（中位数）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

REPO = Path("D:/MyProject/DeepFaceLab-PyTorch-4.1")


def pick_frame():
    vs = sorted((REPO / "workspace").glob("*.mp4"))
    if not vs:
        raise SystemExit("no video")
    cap = cv2.VideoCapture(str(vs[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("read frame failed")
    return f


def make_problem(dst, size):
    """从 dst 里裁一块 ROI 当目标，另造一张内容相近但梯度不同的 src。"""
    h, w = dst.shape[:2]
    cx, cy = w // 2, h // 2
    x0, y0 = cx - size // 2, cy - size // 2
    roi = dst[y0:y0 + size, x0:x0 + size].copy()
    src = cv2.GaussianBlur(roi, (0, 0), 2.0)
    src = np.clip(src.astype(np.int16) + 24, 0, 255).astype(np.uint8)     # 亮度偏移，保证梯度不同
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, size // 2), (int(size * 0.34), int(size * 0.42)), 0, 0, 360, 255, -1)
    return roi, src, mask, (x0, y0)


def laplacian(img):
    f = img.astype(np.float64)
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4 * f[1:-1, 1:-1])
    return out


def poisson_fft(src, dst, mask):
    """在 bbox 矩形上解 Dirichlet 泊松，只在 mask 内替换像素。"""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = src[y0:y1, x0:x1].astype(np.float64)
    d = dst[y0:y1, x0:x1].astype(np.float64)
    H, W = s.shape[:2]
    b = laplacian(s)                                    # 目标梯度场
    # 未知量域 = 内部 (1..H-2, 1..W-2)，边界取 dst
    rhs = b[1:-1, 1:-1].copy()
    rhs[0, :] -= d[0, 1:-1]
    rhs[-1, :] -= d[-1, 1:-1]
    rhs[:, 0] -= d[1:-1, 0]
    rhs[:, -1] -= d[1:-1, -1]
    # DST-I 对角化（零 Dirichlet）
    n_h, n_w = H - 2, W - 2
    kh = np.arange(1, n_h + 1)
    kw = np.arange(1, n_w + 1)
    mu_h = 2 * np.cos(np.pi * kh / (n_h + 1)) - 2
    mu_w = 2 * np.cos(np.pi * kw / (n_w + 1)) - 2
    denom = mu_h[:, None] + mu_w[None, :]
    f = d.copy()
    for ch in range(s.shape[2] if s.ndim == 3 else 1):
        r_ch = rhs[..., ch] if s.ndim == 3 else rhs
        u = idstn(dstn(r_ch, type=1, norm="ortho") / denom, type=1, norm="ortho")
        if s.ndim == 3:
            f[1:-1, 1:-1, ch] = u
        else:
            f[1:-1, 1:-1] = u
    out = dst.copy()
    sub = mask[y0:y1, x0:x1] > 0
    out[y0:y1, x0:x1][sub] = np.clip(f, 0, 255)[sub]
    return out


def poisson_cg(src, dst, mask):
    """只在遮罩内解，边界值取 dst 的遮罩边缘（最接近 cv2 的边界条件）。逐通道求解。"""
    if src.ndim == 3:
        chans = [poisson_cg(src[..., ch], dst[..., ch], mask) for ch in range(src.shape[2])]
        return np.stack(chans, axis=-1)
    return _poisson_cg_2d(src, dst, mask)


def _poisson_cg_2d(src, dst, mask):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1] > 0
    s = src[y0:y1, x0:x1].astype(np.float64)
    d = dst[y0:y1, x0:x1].astype(np.float64)
    b = laplacian(s)
    idx = -np.ones(m.shape, np.int32)
    n = int(m.sum())
    idx[m] = np.arange(n)
    A = lil_matrix((n, n))
    rhs = np.zeros(n)
    H, W = m.shape
    for y in range(H):
        for x in range(W):
            if not m[y, x]:
                continue
            i = idx[y, x]
            A[i, i] = 4.0
            rhs[i] = -b[y, x]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx]:
                    A[i, idx[yy, xx]] -= 1.0
                else:
                    rhs[i] += d[yy, xx] if (0 <= yy < H and 0 <= xx < W) else d[y, x]
    A = A.tocsr()
    u, _info = cg(A, rhs, rtol=1e-8, maxiter=2000)
    sol = d.copy()
    sol[m] = u
    out = dst.copy()
    sub = mask[y0:y1, x0:x1] > 0
    out[y0:y1, x0:x1][sub] = np.clip(sol, 0, 255)[sub]
    return out


def bench(fn, n=7):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def metrics_inner(a, b, roi_mask, erode=3):
    k = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
    inner = cv2.erode(roi_mask, k)
    return metrics(a, b, inner)


def metrics(a, b, roi_mask):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    m = roi_mask > 0
    md = d.max()
    mean = float(d[m].mean()) if m.any() else float(d.mean())
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m].mean())
    psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse)
    return md, mean, psnr


def main():
    frame = pick_frame()
    print("frame", frame.shape)
    for size in (256, 512):
        roi3, src3, mask, (x0, y0) = make_problem(frame, size)
        roi = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
        src = cv2.cvtColor(src3, cv2.COLOR_BGR2GRAY)
        # A: cv2
        center = (size // 2, size // 2)
        a3 = cv2.seamlessClone(src3, roi3, mask, center, cv2.NORMAL_CLONE)
        a = cv2.cvtColor(a3, cv2.COLOR_BGR2GRAY)
        # B: FFT
        b = poisson_fft(src, roi, mask)
        # C: CG
        c = poisson_cg(src, roi, mask)
        for name, img in (("fft", b), ("cg", c)):
            md, mean, psnr = metrics(a, img, mask)
            md2, mean2, psnr2 = metrics_inner(a, img, mask, 4)
            print("size=%-4d %-3s vs cv2   full: max=%-3d mean=%.2f psnr=%.2f | inner(-4px): max=%-3d mean=%.2f psnr=%.2f"
                  % (size, name, md, mean, psnr, md2, mean2, psnr2))
        md, mean, psnr = metrics(b, c, mask)
        print("size=%-4d fft vs cg    max=%-3d mean=%.2f psnr=%.2f dB" % (size, md, mean, psnr))
        m = mask > 0
        off = float(a[m].mean() - b[m].mean())
        b2 = np.clip(b.astype(np.float64) + off, 0, 255)
        md3, mean3, psnr3 = metrics(a, b2, mask)
        print("size=%-4d fft+offset(%+.2f) max=%-3d mean=%.2f psnr=%.2f dB" % (size, off, md3, mean3, psnr3))
        print("size=%-4d signed mean diff (cv2-fft) = %+.3f ; cv2 mean=%.2f fft mean=%.2f"
              % (size, off, float(a[m].mean()), float(b[m].mean())))
        t_a = bench(lambda: cv2.seamlessClone(src3, roi3, mask, center, cv2.NORMAL_CLONE))
        t_b = bench(lambda: poisson_fft(src, roi, mask))
        t_c = bench(lambda: poisson_cg(src, roi, mask), n=3)
        print("size=%-4d time  cv2=%.2f ms  fft=%.2f ms  cg=%.1f ms" % (size, t_a, t_b, t_c))
        out = REPO / "app" / "docs" / "notes" / ("seamless_%d.png" % size)
        cv2.imwrite(str(out), np.hstack([roi, src, a, b, c]))


if __name__ == "__main__":
    main()
