# -*- coding: utf-8 -*-
r"""单配置的 torch 前向 + CUDA Graph 实测（不掺 onnxruntime，一个进程只跑一个配置）。

为什么要单独一支：dfm_torch_bench.py 一个进程里会把 ORT / eager / compile / CUDA Graph / CPU
全跑一遍，8GB 的 5070 Laptop 到后面显存吃满，驱动开始往系统内存回退，量出来的
CUDA Graph 会变成 7 秒一张（不是图的问题，是显存的问题）。这里一个进程只做一件事。

    runtime\engine\python.exe app\tools\bench\dfm_cudagraph_bench.py --model 416 --batch 8 --reps 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dfm_torch_graph import build_from_onnx     # noqa: E402

MODELS = {
    "384": r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\model\张元英.dfm",
    "416": r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\model\阿証証証証-416.dfm",
}


def med(xs):
    return float(statistics.median(xs))


def timed(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="416")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--channels-last", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    a = ap.parse_args()

    path = MODELS.get(a.model, a.model)
    res = 416 if "416" in path else 384
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    x = torch.rand(a.batch, res, res, 3, device="cuda") * 0.5 + 0.25

    net = build_from_onnx(path, batch=a.batch)
    net = net.cuda()
    if a.channels_last:
        net = net.to(memory_format=torch.channels_last)

    out = {"model": a.model, "batch": a.batch, "tf32": a.tf32,
           "channels_last": a.channels_last, "nodes": net.n_runtime_nodes}

    with torch.no_grad():
        ts = timed(lambda: net(x), a.reps)
    out["eager_ms"] = round(med(ts), 2)
    out["eager_p10_p90"] = [round(sorted(ts)[len(ts) // 10], 2),
                            round(sorted(ts)[-max(1, len(ts) // 10)], 2)]
    out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)

    if a.cpu_threads:
        torch.set_num_threads(a.cpu_threads)
        xc = x.cpu()
        netc = net.cpu()
        with torch.no_grad():
            for _ in range(2):
                netc(xc)
            t0 = time.perf_counter()
            for _ in range(max(3, a.reps // 4)):
                netc(xc)
            out["cpu_ms_thread%d" % a.cpu_threads] = round(
                (time.perf_counter() - t0) * 1000.0 / max(3, a.reps // 4), 2)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return

    # ---- CUDA Graph：固定 batch、固定输入地址
    try:
        static_in = torch.empty_like(x)
        static_in.copy_(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    net(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                _ = net(static_in)
        torch.cuda.synchronize()
        out["capture_s"] = round(time.perf_counter() - t0, 2)
        ts = timed(lambda: g.replay(), a.reps, warm=2)
        out["graph_ms"] = round(med(ts), 2)
        out["graph_ok"] = True
        out["vram_peak_mb_graph"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)
    except Exception as e:                                # noqa: BLE001
        out["graph_ok"] = False
        out["graph_err"] = "%s: %s" % (type(e).__name__, str(e)[:300])

    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
