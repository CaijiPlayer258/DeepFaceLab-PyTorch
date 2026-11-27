# -*- coding: utf-8 -*-
"""按人脸框裁一个「正方形 ROI」，越界补黑 —— 供 ROI 跟踪检测用（尚未接入流程）。

**为什么补黑而不是夹到画面内**：
  · 夹（clamp）会让 ROI 变成非正方形 —— 宽高比一变，检测器看到的尺度和角度就变了，
    反变换也要处理两个不同的 scale；
  · 补黑则保持**尺寸和角度都不变**，反变换只是一次线性换算；
  · 而且和提取链路的约定一致：align.py:792 的对齐就是 BORDER_CONSTANT(0,0,0)，
    注释写着「越界部分填纯黑（DFL 的做法）」。DFL 从 iperov 时代起训练集里就有
    大量带黑边的人脸，模型认得这种输入。

**真正的换脸失败原因是「对齐不准确」，不是「有黑边」** —— 补黑不会让对齐变差。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def pad_value() -> int:
    """越界填什么。默认 0（纯黑，和提取对齐一致）。

    DFL_ROI_PAD=128 可以试「灰边」—— 检测器的 letterbox pad 就是 128
    （TinyMog/SCRFD 的约定），但**对齐/裁剪**这条链一直是 0，
    所以这里的默认值也必须是 0，否则同一个 ROI 在检测和对齐里看到两种边。
    """
    try:
        return int(os.environ.get("DFL_ROI_PAD") or 0)
    except (TypeError, ValueError):
        return 0


def square_side(w: float, h: float, scale: float = 2.5,
                lo: int = 256, hi: Optional[int] = None,
                frame_size: Optional[Tuple[int, int]] = None) -> int:
    """按脸的框算 ROI 边长：脸的长边 × scale，夹到 [lo, hi]。

    scale=2.5 → 脸占输出画布的 40%（640/2.5 = 256 px），落在检测器训练分布的中段。
    太小（1.0）脸会占满画布、超出分布；太大（>6）就退化成整帧了。
    """
    # 上界不能写死 1280：4K 里一张 1500 px 的脸按 2.5 倍要 3750，写死的上界会让
    # 脸直接超出 ROI（补黑的就是脸本身了）。给了 frame_size 就按画面短边兜住。
    if hi is None:
        hi = 1280
        if frame_size:
            try:
                hi = max(int(lo), int(min(int(frame_size[0]), int(frame_size[1]))))
            except (TypeError, ValueError, IndexError):
                hi = 1280
    try:
        side = int(round(max(float(w), float(h)) * float(scale)))
    except (TypeError, ValueError):
        return int(lo)
    return int(max(int(lo), min(int(side), int(hi))))


def crop_square_padded(img: np.ndarray, cx: float, cy: float, side: int,
                       out_size: int = 640, pad: Optional[int] = None,
                       interpolation: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """以 (cx, cy) 为中心裁一个 side×side 的正方形，缩放到 out_size，越界补黑。

    **一次 warpAffine 完成裁剪 + 缩放 + 补黑**：不用先切片（切片在越界时会抛/要
    自己算 pad），也不用先 np.full 一张画布。warpAffine 只采样目标区域，
    源图多大都一样。

    返回 (画布, 反变换用的 meta)：
      canvas : (out_size, out_size, 3) uint8
      meta   : {"x1","y1","side","s","out_size"} —— 把画布坐标换回原图：
               x_src = x_canvas / s + x1
               y_src = y_canvas / s + y1
    """
    import cv2

    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("crop_square_padded: 空图")
    side = int(max(1, int(side)))
    out_size = int(out_size)
    pv = int(pad_value() if pad is None else pad)
    interp = cv2.INTER_LINEAR if interpolation is None else interpolation

    # 左上角用浮点（允许为负 —— 那正是需要补黑的情况）
    x1 = float(cx) - side / 2.0
    y1 = float(cy) - side / 2.0
    s = float(out_size) / float(side)          # 各向同性：正方形进、正方形出

    M = np.array([[s, 0.0, -x1 * s],
                  [0.0, s, -y1 * s]], dtype=np.float64)
    canvas = cv2.warpAffine(img, M, (out_size, out_size), flags=interp,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(pv, pv, pv))
    return canvas, {"x1": x1, "y1": y1, "side": side, "s": s,
                    "out_size": out_size, "pad": pv}


def map_box_back(box, meta: dict):
    """把画布坐标的框换回原图坐标。box = (x1,y1,x2,y2) 或 None。"""
    if box is None:
        return None
    try:
        s = float(meta["s"])
        ox, oy = float(meta["x1"]), float(meta["y1"])
        bx1, by1, bx2, by2 = [float(v) for v in box]
    except (KeyError, TypeError, ValueError):
        return None
    return (bx1 / s + ox, by1 / s + oy, bx2 / s + ox, by2 / s + oy)
