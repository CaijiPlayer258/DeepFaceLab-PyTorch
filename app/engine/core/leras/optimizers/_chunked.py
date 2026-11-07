# -*- coding: utf-8 -*-
"""优化器的大张量分块工具。

为什么需要：1024 分辨率下 inter 会出现**单个 10 亿参数**的权重（fp32 就是 4 GiB）。
优化器 step 里任何 "先 .float() / g * scale / (g-m).pow(2)" 都会为它再 materialize
一份同样大的临时量 —— 20 GB 卡上直接 OOM。

这里把这类逐元素运算按 CHUNK 切块：临时量从 GB 级降到 16 MB 级，
代价只是多几次 kernel 启动（实测对 it/s 影响在噪声范围内）。
"""
from __future__ import annotations

import torch

# 4M 个元素：fp32 临时量 = 16 MB。太大省不了显存，太小 kernel 启动次数暴涨。
CHUNK = 1 << 22


def flat(t: torch.Tensor) -> torch.Tensor:
    """reshape 成一维视图；非连续张量会退化成拷贝（只发生在极小概率的非常规权重上）。"""
    return t.reshape(-1)


def grad_norm_sq(params, scale: float = 1.0):
    """全局梯度平方和（给 clipnorm 用），分块累加，避免整块 fp32 临时量。"""
    total = None
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        gp = flat(g)
        n = gp.numel()
        for s in range(0, n, CHUNK):
            term = gp[s:s + CHUNK].float()
            if scale != 1.0:
                term = term * scale
            v = term.pow(2).sum()
            total = v if total is None else (total + v)
    return total


def chunks(n: int):
    """[(start, end), ...]"""
    for s in range(0, n, CHUNK):
        yield s, min(s + CHUNK, n)
