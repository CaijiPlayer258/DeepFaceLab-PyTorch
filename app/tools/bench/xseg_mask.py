# -*- coding: utf-8 -*-
"""XSegLite 遮罩（ONNX）+ 遮罩内指标。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
XSEG = REPO / "workspace" / "model" / "XSegLite" / "xseglite.onnx"
_SESS = None


def _sess():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        _SESS = ort.InferenceSession(str(XSEG), providers=["CPUExecutionProvider"])
    return _SESS


def mask_of(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """输入 0..255 BGR（任意分辨率），返回同尺寸 0..1 遮罩。"""
    h, w = bgr.shape[:2]
    x = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]                  # NCHW
    s = _sess()
    y = s.run(None, {s.get_inputs()[0].name: x})[0]
    m = np.asarray(y).reshape(size, size)
    m = np.clip(m, 0.0, 1.0)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def masked(a: np.ndarray, b: np.ndarray, m: np.ndarray):
    """a,b: 0..1 float RGB/BGR 同形状；m: 0..1。返回遮罩内指标。"""
    w = m[..., None]
    diff = (a - b) ** 2 * w
    mse = float(diff.sum() / (w.sum() * a.shape[2] + 1e-9))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)
    mae = float((np.abs(a - b) * w).sum() / (w.sum() * a.shape[2] + 1e-9))
    # SSIM 在遮罩内取均值
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gb, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu1 * mu2
    smap = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    ssim = float((smap * m).sum() / (m.sum() + 1e-9))
    return dict(psnr=psnr, ssim=ssim, mae=mae * 255.0)


def structure_metrics(a: np.ndarray, b: np.ndarray, m: np.ndarray, sigma: float = 4.0):
    """把「结构」和「锐度」分开量。

    structure: 双方都做 sigma=4 低通后的 NCC（对模糊不敏感，只看几何/明暗分布是否一致）
    hf_ratio : 高频能量比 E_hf(b)/E_hf(a)（Laplacian 方差），>1 表示比基准更锐
    """
    w = m[..., None]
    msum = float(w.sum() * a.shape[2]) + 1e-9
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    la = cv2.GaussianBlur(ga, (0, 0), sigma)
    lb = cv2.GaussianBlur(gb, (0, 0), sigma)
    ma = float((la * m).sum() / (m.sum() + 1e-9))
    mb = float((lb * m).sum() / (m.sum() + 1e-9))
    da = (la - ma) * m
    db = (lb - mb) * m
    ncc = float((da * db).sum() / (np.sqrt((da * da).sum() * (db * db).sum()) + 1e-9))
    hfa = float((cv2.Laplacian(ga, cv2.CV_32F) ** 2 * m).sum() / (msum))
    hfb = float((cv2.Laplacian(gb, cv2.CV_32F) ** 2 * m).sum() / (msum))
    return dict(ncc=ncc, hf_ratio=(hfb / (hfa + 1e-9)))
