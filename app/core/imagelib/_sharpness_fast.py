"""Marziliano 边缘宽度 —— numba 加速版。

原实现在 core/imagelib/estimate_sharpness.py 里是**逐像素 Python 双循环**：
512x512 = 262144 次迭代做 atan2，然后又是一轮边缘宽度游走。实测这一项独占
estimate_sharpness 的 85%（383 ms / 453 ms）。

numba 的 njit 把它编译成机器码：同样的算术语义，速度差两个数量级。
许可证：numba 是 BSD-2-Clause，与原项目兼容，且**不需要 C 编译器**（Cython 需要
MSVC 工具链，分发给用户时会更麻烦）。
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                      # 没装 numba 时自动退回原实现
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or callable(a[0]) else deco


@njit(cache=True, fastmath=False)
def _angles(gx, gy):
    """逐像素算梯度方向（度）。语义与原始 Python 循环完全一致。"""
    h, w = gx.shape
    out = np.zeros((h, w), np.float64)
    for r in range(h):
        for c in range(w):
            x = gx[r, c]
            y = gy[r, c]
            if x != 0.0:
                out[r, c] = np.arctan2(y, x) * (180.0 / np.pi)
            elif y == 0.0:
                out[r, c] = 0.0
            elif y == np.pi / 2.0:      # 原实现里这个分支几乎不可能命中，保留以对齐行为
                out[r, c] = 90.0
    return out


@njit(cache=True, fastmath=False)
def _widths(edges, image, quantized):
    """边缘宽度游走。逐像素判断 + 双向扫描，最多 100 步。"""
    h, w = image.shape
    out = np.zeros((h, w), np.float64)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if edges[r, c] != 1.0:
                continue
            q = quantized[r, c]

            if q == 180.0 or q == -180.0:
                # 左侧：强度递增时停
                margin = 0
                for margin in range(101):
                    inner = (c - 1) - margin
                    outer = (c - 2) - margin
                    if outer < 0 or (image[r, outer] - image[r, inner]) <= 0.0:
                        break
                width_left = margin + 1
                for margin in range(101):
                    inner = (c + 1) + margin
                    outer = (c + 2) + margin
                    if outer >= w or (image[r, outer] - image[r, inner]) >= 0.0:
                        break
                width_right = margin + 1
                out[r, c] = width_left + width_right

            elif q == 0.0:
                margin = 0
                for margin in range(101):
                    inner = (c - 1) - margin
                    outer = (c - 2) - margin
                    if outer < 0 or (image[r, outer] - image[r, inner]) >= 0.0:
                        break
                width_left = margin + 1
                for margin in range(101):
                    inner = (c + 1) + margin
                    outer = (c + 2) + margin
                    if outer >= w or (image[r, outer] - image[r, inner]) <= 0.0:
                        break
                width_right = margin + 1
                out[r, c] = width_left + width_right

    return out


@njit(cache=True, fastmath=False, nogil=True)
def _fused(edges, image, gx, gy):
    """把"算角度"和"量边缘宽度"合成一趟：省一次 262k 像素的遍历和整个 angles 数组。"""
    h, w = image.shape
    out = np.zeros((h, w), np.float64)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if edges[r, c] != 1.0:
                continue
            x = gx[r, c]
            y = gy[r, c]
            if x != 0.0:
                ang = np.arctan2(y, x) * (180.0 / np.pi)
            elif y == 0.0:
                ang = 0.0
            elif y == np.pi / 2.0:
                ang = 90.0
            else:
                ang = 0.0     # 原实现里 edge_angles 初值为 0，这个分支不能跳过
            q = 45.0 * np.round(ang / 45.0)
            if q != 180.0 and q != -180.0 and q != 0.0:
                continue

            margin = 0
            if q == 180.0 or q == -180.0:
                for margin in range(101):
                    inner = (c - 1) - margin
                    outer = (c - 2) - margin
                    if outer < 0 or (image[r, outer] - image[r, inner]) <= 0.0:
                        break
                width_left = margin + 1
                for margin in range(101):
                    inner = (c + 1) + margin
                    outer = (c + 2) + margin
                    if outer >= w or (image[r, outer] - image[r, inner]) >= 0.0:
                        break
                out[r, c] = width_left + (margin + 1)
            else:
                for margin in range(101):
                    inner = (c - 1) - margin
                    outer = (c - 2) - margin
                    if outer < 0 or (image[r, outer] - image[r, inner]) >= 0.0:
                        break
                width_left = margin + 1
                for margin in range(101):
                    inner = (c + 1) + margin
                    outer = (c + 2) + margin
                    if outer >= w or (image[r, outer] - image[r, inner]) <= 0.0:
                        break
                out[r, c] = width_left + (margin + 1)
    return out


def marziliano_fast(edges, image):
    """与原始 marziliano_method 等价，只是把两个 Python 循环换成编译版本。"""
    gradient_y, gradient_x = np.gradient(image)
    gy = np.ascontiguousarray(gradient_y, dtype=np.float64)
    gx = np.ascontiguousarray(gradient_x, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    image = np.ascontiguousarray(image, dtype=np.float64)
    return _fused(edges, image, gx, gy)
