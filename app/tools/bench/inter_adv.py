# -*- coding: utf-8 -*-
"""比 bicubic 更好的 inter 重采样：从「保留线性泛函」出发。

记号：dense1 的权重 W ∈ R^{(C·n²)×A}，行索引 = c·n² + i·n + j。
分辨率变化时，encoder 输出的特征图网格从 n 变成 m（n = res0/32，m = res/32）。
设重采样算子是 1-D 的 L ∈ R^{m×n}（bicubic/bilinear 都是一种 L），2-D 就是 L⊗L。

**为什么插值不是最优**
插值做的是 W' = (L⊗L) W：把权重当成一张"图"去插值。
但真正要保的是泛函 ⟨W, f⟩（f 是特征图）：换分辨率后 f' 与 f 的关系是 f' ≈ (L⊗L) f，
于是要求 ⟨W', f'⟩ = ⟨W, f⟩ 对所有 f 成立，即

    (L⊗L)ᵀ W' = W                        … (1)

- 升分辨率（m > n）：(1) 欠定（方程 n² 个、未知 m² 个）→ 最小范数解
      W' = (L⊗L) · [(L⊗L)ᵀ(L⊗L)]^{-1} W        … (2)
  也就是「先做一次逆滤波 (LᵀL)^{-1}，再插值」。插值缺的正是这个逆滤波，
  所以它把权重抹平了（低频骨架在，高频没了）。
- 降分辨率（m < n）：(1) 超定 → 最小二乘解
      W' = [(L⊗L)(L⊗L)ᵀ]^{-1} (L⊗L) W        … (3)
  也就是「先降采样，再做逆滤波」。

(2)(3) 里的逆滤波在高频会放大量化噪声，所以加 Tikhonov：A = (LᵀL + λI) 之类。

**再往前一步：用真实激活做最小二乘（数据驱动）**
(2)(3) 假设 f' = (L⊗L) f，但真实 encoder 是非线性的，这个假设只是近似。
可以直接取真实特征：同一张脸在 416 上跑 encoder 得 f，在目标分辨率上跑同一个
encoder 得 f'（卷积与分辨率无关），然后解

    min_{W'} || F' W' - F W ||²  + λ || W' - W'_bicubic ||²

F' ∈ R^{N×(C·m²)}、F ∈ R^{N×(C·n²)}、N 是图片数。C·m² 很大（16000~163840），
所以用对偶形式：W' = W'_bic + F'ᵀ α，α = (F'F'ᵀ + λI)^{-1} (F W - F' W'_bic)，
只需解 N×N —— 精确且便宜。这一项把「bicubic 的偏差」按真实数据回归掉。
"""
from __future__ import annotations

import numpy as np


def interp_matrix(n: int, m: int, kind: str = "bicubic") -> np.ndarray:
    """1-D 重采样矩阵 L ∈ R^{m×n}（align_corners=False 的坐标映射）。"""
    if kind == "nearest":
        idx = np.clip(((np.arange(m) + 0.5) * n / m - 0.5).round().astype(int), 0, n - 1)
        L = np.zeros((m, n))
        L[np.arange(m), idx] = 1.0
        return L
    if kind == "area":
        L = np.zeros((m, n))
        for i in range(m):
            a, b = i * n / m, (i + 1) * n / m
            j0, j1 = int(np.floor(a)), min(int(np.ceil(b)), n)
            for j in range(j0, j1):
                lo, hi = max(a, j), min(b, j + 1)
                if hi > lo:
                    L[i, j] += hi - lo
            s = L[i].sum()
            if s:
                L[i] /= s
        return L
    def w_lin(t):
        return max(0.0, 1.0 - abs(t))
    def w_cub(t):
        t = abs(t)
        if t <= 1:
            return 1.5 * t ** 3 - 2.5 * t ** 2 + 1
        if t < 2:
            return -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2
        return 0.0
    def w_lan(t, a=3.0):
        t = abs(t)
        if t < 1e-8:
            return 1.0
        if t >= a:
            return 0.0
        return a * np.sin(np.pi * t) * np.sin(np.pi * t / a) / (np.pi ** 2 * t ** 2)

    if kind == "lanczos":
        wf, R = w_lan, 3
    elif kind == "bicubic":
        wf, R = w_cub, 2
    else:
        wf, R = w_lin, 1
    L = np.zeros((m, n))
    for i in range(m):
        x = (i + 0.5) * n / m - 0.5
        x0 = int(np.floor(x))
        for k in range(-R + 1, R + 1):
            j = x0 + k
            wgt = wf(x - j)
            if 0 <= j < n and wgt:
                L[i, j] += wgt
        s = L[i].sum()
        if s:                       # 边界处归一，避免整体增益偏移
            L[i] /= s
    return L


def _resample_2d(T: np.ndarray, n: int, m: int, kind: str, mode: str, lam: float) -> np.ndarray:
    """T: (C, n, n, A) → (C, m, m, A)。

    mode='interp'  插值（bicubic/bilinear/nearest/area 的原样做法）
    mode='pinv'    泛函保持：升采样走最小范数解、降采样走最小二乘解，加 λ 正则
    """
    L = interp_matrix(n, m, kind)
    if mode == "interp":
        T1 = np.einsum("ij,cjka->cika", L, T)
        return np.einsum("ij,ckja->ckia", L, T1)
    def ax1(M, T):                    # 沿第 1 个空间轴
        return np.einsum("ij,cjka->cika", M, T)

    def ax2(M, T):                    # 沿第 2 个空间轴
        return np.einsum("ij,ckja->ckia", M, T)

    if m >= n:                        # 升：W' = L (LᵀL+λI)⁻¹ ⊗ 同
        Ai = np.linalg.inv(L.T @ L + lam * np.eye(n))
        T = ax1(Ai, T)
        T = ax2(Ai, T)
        T = ax1(L, T)
        return ax2(L, T)
    B = np.linalg.inv(L @ L.T + lam * np.eye(m))   # 降：(LLᵀ+λI)⁻¹ L ⊗ 同
    T = ax1(L, T)
    T = ax1(B, T)
    T = ax2(L, T)
    return ax2(B, T)


def resample_inter(arrays, res_from: int, res_to: int, method: str = "bicubic",
                   mode: str = "interp", lam: float = 1e-2):
    """arrays: [dense1_w, dense1_b, dense2_w, dense2_b]。返回同样 4 个数组（float32）。"""
    d1w, d1b, d2w, d2b = [np.asarray(a) for a in arrays]
    ae = d1w.shape[1]
    n = max(1, res_from // 32)
    m = max(1, res_to // 32)
    C1 = d1w.shape[0] // (n * n)
    T = d1w.reshape(C1, n, n, ae)
    T2 = _resample_2d(T, n, m, method, mode, lam)
    d1w_new = T2.reshape(-1, ae)

    C2 = d2w.shape[1] // (n * n)
    U = d2w.reshape(ae, C2, n, n).transpose(1, 2, 3, 0)       # (C2, n, n, ae)
    U2 = _resample_2d(U, n, m, method, mode, lam)             # (C2, m, m, ae)
    d2w_new = U2.transpose(3, 0, 1, 2).reshape(ae, -1)
    B2 = d2b.reshape(C2, n, n, 1)
    B3 = _resample_2d(B2, n, m, method, mode, lam).reshape(-1)
    return [d1w_new.astype(np.float32), d1b.astype(np.float32),
            d2w_new.astype(np.float32), B3.astype(np.float32)]


def ls_correct(d1w_bic: np.ndarray, d1w_old: np.ndarray, F: np.ndarray, Fp: np.ndarray, lam: float):
    """对偶岭回归修正 dense1：W' = W_bic + Fpᵀ α，α=(FpFpᵀ+λI)^{-1}(FW - FpW_bic)。

    F : (N, C·n²) 416 上的真实 encoder 特征
    Fp: (N, C·m²) 目标分辨率上的真实 encoder 特征
    """
    Z = F @ d1w_old                                  # (N, A) 目标潜码（原模型）
    R = Z - Fp @ d1w_bic                             # 残差
    K = Fp @ Fp.T + lam * np.eye(Fp.shape[0])
    alpha = np.linalg.solve(K, R)                    # (N, A)
    return (d1w_bic + Fp.T @ alpha).astype(np.float32)
