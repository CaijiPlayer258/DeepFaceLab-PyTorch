# -*- coding: utf-8 -*-
"""分辨率课程的成本模型：逐分辨率的训练步耗时（真机测量）。

不是完整训练，而是同一个 SAEHD(df-udt) 主干、同一 batch 下的
前向 + 反向 + 一次优化器步进，用来估算各档分辨率的 it/s 与显存占用。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path("D:/MyProject/DeepFaceLab-PyTorch-4.1")
sys.path.insert(0, str(REPO / "app" / "engine"))

from core.leras import nn as leras          # noqa: E402
leras.initialize_main_env()
leras.initialize(data_format="NCHW")

from facelib.DeepFakeArchi import DeepFakeArchi   # noqa: E402


def bench(res, batch=8, iters=12):
    opts = "udt"
    m = DeepFakeArchi(res, opts=opts).cuda().train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    x = torch.rand(batch, 3, res, res, device="cuda")
    # 前几次不计时（kernel 预热 / cuDNN autotune）
    for _ in range(3):
        y = m(x)
        loss = sum(t.float().pow(2).mean() for t in y if torch.is_tensor(t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = m(x)
        loss = sum(t.float().pow(2).mean() for t in y if torch.is_tensor(t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    del m, opt, x
    torch.cuda.empty_cache()
    return dt * 1000, 1.0 / dt, peak


def main():
    print("%-5s %10s %8s %10s" % ("res", "ms/it", "it/s", "peak MB"))
    for res in (128, 192, 256, 320, 416):
        try:
            ms, ips, peak = bench(res)
            print("%-5d %10.1f %8.2f %10.0f" % (res, ms, ips, peak))
        except RuntimeError as e:
            print("%-5d OOM: %s" % (res, str(e)[:60]))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
