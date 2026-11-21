# -*- coding: utf-8 -*-
"""遮罩（alpha）上采样：双线性 / 双三次 / lanczos / 引导滤波 / 联合双边。

XSeg 在 256² 出遮罩，合成时被放大到 1K~4K。这里用可解析的 alpha 做受控实验：
  GT       1024² 解析 alpha（椭圆软边 + 细发丝）
  降采样    x4 → 256²
  上采样    各种方法 → 1024²
  指标      PSNR / IoU@0.5 / 边界定位误差（距离变换）/ 细丝保留
"""
from __future__ import annotations

import time

import cv2
import numpy as np

N = 1024


def analytic_alpha(n=N):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx, cy = n * 0.5, n * 0.52
    ax, ay = n * 0.26, n * 0.33
    r = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
    a = np.clip((1.0 - r) * 12.0 + 0.5, 0, 1)                     # 约 0.08 像素的软边
    # 细发丝：几根正弦曲线，宽度 2~3 px
    for k, (amp, freq, off) in enumerate([(0.05, 5.0, 0.18), (0.04, 7.0, 0.30), (0.06, 4.0, 0.42)]):
        y0 = cy - ay + n * off
        y = y0 + amp * n * np.sin(xx / n * np.pi * freq)
        d = np.abs(yy - y)
        w = 1.6 + k * 0.4
        a = np.maximum(a, np.clip((w - d) * 0.9 + 0.5, 0, 1))
    return a.astype(np.float32)


def box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r=8, eps=1e-3):
    """He et al. 引导滤波：guide 为高分辨率图（灰度），src 为待上采样的低分辨率 alpha。"""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI, mp = box(I, r), box(p, r)
    cIp = box(I * p, r) - mI * mp
    cII = box(I * I, r) - mI * mI
    a = cIp / (cII + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)


def joint_bilateral(low, guide, scale, r=3, sigma_s=2.0, sigma_r=0.12):
    """联合双边上采样：在低分辨率网格上按空间+值域权重插值，guide 提供值域参考。"""
    h, w = guide.shape[:2]
    lh, lw = low.shape[:2]
    out = np.zeros((h, w), np.float32)
    ys = (np.arange(h) + 0.5) / scale - 0.5
    xs = (np.arange(w) + 0.5) / scale - 0.5
    for j in range(h):
        yl = ys[j]
        y0 = int(np.floor(yl))
        for i in range(w):
            xl = xs[i]
            x0 = int(np.floor(xl))
            num = 0.0
            den = 0.0
            g0 = guide[j, i]
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = y0 + dy, x0 + dx
                    if 0 <= yy < lh and 0 <= xx < lw:
                        ds = (yy + 0.5 - yl) ** 2 + (xx + 0.5 - xl) ** 2
                        dr = (low[yy, xx] - g0) ** 2
                        wgt = np.exp(-ds / (2 * sigma_s ** 2) - dr / (2 * sigma_r ** 2))
                        num += wgt * low[yy, xx]
                        den += wgt
            out[j, i] = num / den if den > 0 else g0
    return out


def metrics(gt, pred):
    mse = float(((gt - pred) ** 2).mean())
    psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)
    g = gt > 0.5
    p = pred > 0.5
    iou = float((g & p).sum()) / max(1, int((g | p).sum()))
    # 边界定位误差：预测边界到 GT 边界的平均距离
    gb = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    pb = cv2.morphologyEx(p.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(1 - gb, cv2.DIST_L2, 5)
    err = float(dt[pb > 0].mean()) if pb.any() else 0.0
    return psnr, iou, err


def main():
    gt = analytic_alpha()
    guide = np.clip(gt * 200 + np.random.RandomState(0).normal(0, 20, gt.shape), 0, 255).astype(np.uint8)
    low = cv2.resize(gt, (N // 4, N // 4), interpolation=cv2.INTER_AREA)
    big_guide = cv2.resize(guide, (N, N), interpolation=cv2.INTER_CUBIC)

    out = {}
    t = time.perf_counter()
    out["bilinear"] = cv2.resize(low, (N, N), interpolation=cv2.INTER_LINEAR)
    t_b = (time.perf_counter() - t) * 1000
    out["bicubic"] = cv2.resize(low, (N, N), interpolation=cv2.INTER_CUBIC)
    out["lanczos"] = cv2.resize(low, (N, N), interpolation=cv2.INTER_LANCZOS4)

    up = out["bilinear"]
    t = time.perf_counter()
    out["guided"] = np.clip(guided_filter(big_guide, up, r=8, eps=1e-3), 0, 1)
    t_g = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    out["jbf"] = np.clip(joint_bilateral(low, big_guide.astype(np.float32) / 255.0, 4, r=2), 0, 1)
    t_j = (time.perf_counter() - t) * 1000

    print("%-9s %8s %8s %10s" % ("method", "PSNR", "IoU", "edge err"))
    for k, v in out.items():
        psnr, iou, err = metrics(gt, v.astype(np.float32))
        print("%-9s %8.2f %8.4f %10.3f" % (k, psnr, iou, err))
    print("time  bilinear=%.1f ms  guided=%.1f ms  jbf=%.0f ms (1024²)" % (t_b, t_g, t_j))

    strip = np.hstack([(gt * 255).astype(np.uint8)] + [(out[k] * 255).astype(np.uint8) for k in ("bilinear", "bicubic", "lanczos", "guided")])
    cv2.imwrite("D:/MyProject/DeepFaceLab-PyTorch-4.1/app/docs/notes/alpha_upsample.png", strip)


if __name__ == "__main__":
    main()
