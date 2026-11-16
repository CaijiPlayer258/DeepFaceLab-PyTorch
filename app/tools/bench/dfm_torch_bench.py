# -*- coding: utf-8 -*-
r"""dfm 的四条推理路径实测（同一份权重，两个模型，batch 1/8）。

背景：dfm 现在是 ONNX（.dfm = ONNX protobuf，权重只以 initializer 形式存在，**没有**
torch state_dict）。本脚本用 app/tools/bench/dfm_torch_graph.py 把同一张图静态折叠后
代码生成成 nn.Module，于是「同一份权重」可以走四条路：

    A  onnxruntime CUDA EP（现状基线）
    A' onnxruntime CPU EP（对照：同一个图，ONNX 的 CPU 实现）
    B  torch CUDA eager（channels_last 开/关 × TF32 开/关）
    C  torch.compile(CUDA)     —— 这台机器没有 triton，inductor 起不来，
                                  用 backend="cudagraphs"（aot_eager + CUDA Graph）代替，
                                  两条都记：失败原文 + 替代方案的数字
    D  torch CPU（intra-op 线程 4/8/16）

跑法（必须 engine venv：它同时有 onnx / onnxruntime / torch）：
    runtime\engine\python.exe app\tools\bench\dfm_torch_bench.py
    runtime\engine\python.exe app\tools\bench\dfm_torch_bench.py --reps 20 --batch 1 8
    runtime\engine\python.exe app\tools\bench\dfm_torch_bench.py --models 384=... --out r.json

口径：
  · 每次计时都是「一次前向」的墙钟，CUDA 侧用 cuda event + synchronize；
  · 报中位数（--reps 次），warmup 单独跑不计；
  · 输入 0..1 的 float32 NHWC（dfm 的原生输入，和 _dfm_run_batch 一致）；
  · 对拍：torch 输出 vs ORT CUDA 输出的 max|diff|（不接近 1e-4 就等于没跑同一个模型）。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dfm_torch_graph import build_from_onnx   # noqa: E402

DEFAULT_MODELS = {
    "384": r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\model\张元英.dfm",
    "416": r"D:\MyProject\DeepFaceLab-PyTorch-4.1\workspace\model\阿証証証証-416.dfm",
}


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _mk_input(res, batch):
    rng = np.random.default_rng(1234)
    x = rng.random((batch, res, res, 3), dtype=np.float32)
    return x


def _med(xs):
    return float(statistics.median(xs))


# ------------------------------------------------------------------ ORT

def ort_session(path, provider):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    kw = {}
    if provider == "CUDAExecutionProvider":
        kw["provider_options"] = [{"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC",
                                   "arena_extend_strategy": "kSameAsRequested",
                                   "do_copy_in_default_stream": "1"}]
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=[provider], **kw)
    return sess, time.perf_counter() - t0


def bench_ort(sess, x, reps, warmup=3):
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return _med(ts), [o.copy() for o in outs]


# ------------------------------------------------------------------ torch

def bench_torch_gpu(net, x, reps, warmup=3):
    xd = torch.from_numpy(x).cuda()
    with torch.no_grad():
        for _ in range(warmup):
            net(xd)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(reps):
            ev0.record()
            net(xd)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
    return _med(ts), xd


def bench_torch_cpu(net, x, reps, threads, warmup=2):
    xc = torch.from_numpy(x)
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(warmup):
            net(xc)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            net(xc)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return _med(ts)


def try_compile(net, x_np, reps):
    """先试 inductor（这台机器没有 triton），失败再把原文记下来并退到 cudagraphs。"""
    out = {"inductor": None, "inductor_err": None, "cudagraphs": None,
           "cudagraphs_err": None, "compile_s": None}
    xd = torch.from_numpy(x_np).cuda()
    t0 = time.perf_counter()
    try:
        cm = torch.compile(net, mode="default")
        with torch.no_grad():
            cm(xd)
        torch.cuda.synchronize()
        out["compile_s"] = time.perf_counter() - t0
        ms, _ = bench_torch_gpu(cm, x_np, reps, warmup=2)
        out["inductor"] = ms
        return out
    except Exception as e:
        out["inductor_err"] = f"{type(e).__name__}: {str(e)[:300]}"
    t0 = time.perf_counter()
    try:
        cm = torch.compile(net, backend="cudagraphs")
        with torch.no_grad():
            cm(xd)
        torch.cuda.synchronize()
        out["compile_s"] = time.perf_counter() - t0
        ms, _ = bench_torch_gpu(cm, x_np, reps, warmup=2)
        out["cudagraphs"] = ms
    except Exception as e:
        out["cudagraphs_err"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def try_cuda_graph(net, x_np, reps):
    """固定 batch、固定输入地址，手工捕获 CUDA Graph（不依赖 triton）。"""
    res = {"ok": False, "ms": None, "err": None, "warm_s": None}
    try:
        xd = torch.from_numpy(x_np).cuda()
        static_in = torch.empty_like(xd)
        static_in.copy_(xd)
        torch.cuda.synchronize()
        # 预热到稳定（避免捕获时的 lazy 分配）
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    net(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                out = net(static_in)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        res["warm_s"] = time.perf_counter() - t0
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(reps):
            ev0.record()
            g.replay()
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        res["ok"] = True
        res["ms"] = _med(ts)
    except Exception as e:
        res["err"] = f"{type(e).__name__}: {str(e)[:300]}"
    return res


# ------------------------------------------------------------------ main

def run_model(tag, path, batches, reps, cpu_threads, do_ort=True, skip_ort_cpu=False, modes=()):
    import onnxruntime as ort
    res = {"tag": tag, "path": str(path), "onnx": {}, "torch": {}, "rows": []}
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    shp = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    res["res"] = shp[1] if len(shp) > 1 else None
    res["n_nodes"] = len(m.graph.node)
    res["n_init"] = len(m.graph.initializer)
    res["producer"] = f"{m.producer_name} {m.producer_version}"

    for B in batches:
        x = _mk_input(res["res"], B)
        row = {"batch": B}

        # ---- A onnxruntime
        if do_ort and (not modes or "ort" in modes):
            _provs = [("CUDAExecutionProvider", "ort_cuda")]
            if not skip_ort_cpu:
                _provs.append(("CPUExecutionProvider", "ort_cpu"))
            for prov, key in _provs:
                try:
                    sess, load_s = ort_session(path, prov)
                    try:
                        ms, outs = bench_ort(sess, x, reps)
                    except Exception as e2:
                        row[key] = None
                        row[key + "_err"] = "run %s: %s" % (type(e2).__name__, str(e2)[:160])
                        del sess
                        _free()
                        continue
                    row[key] = ms
                    row[key + "_load_s"] = load_s
                    if key == "ort_cuda":
                        row["_ref"] = outs
                    del sess
                    _free()
                except Exception as e:
                    row[key] = None
                    row[key + "_err"] = f"{type(e).__name__}: {str(e)[:200]}"

        # ---- 构建 torch 图
        net = build_from_onnx(path, batch=B)
        row["torch_nodes"] = net.n_runtime_nodes
        net = net.cuda()

        # 对拍（batch=1 且 ORT 结果在手上时）
        ref = row.get("_ref")
        if ref is not None:
            with torch.no_grad():
                y = net(torch.from_numpy(x).cuda())
                y = [t.detach().cpu().numpy() for t in y]
            row["maxdiff"] = float(max(float(np.max(np.abs(a - b))) for a, b in zip(y, ref)))
        row.pop("_ref", None)

        # ---- B  torch CUDA eager：channels_last × TF32
        for cl in ((False, True) if (not modes or "torch" in modes) else ()):
            for tf32 in (False, True):
                torch.backends.cudnn.allow_tf32 = tf32
                torch.backends.cuda.matmul.allow_tf32 = tf32
                n2 = net
                xx = x
                if cl:
                    n2 = net.to(memory_format=torch.channels_last)
                ms, _ = bench_torch_gpu(n2, xx, reps)
                row[f"torch_cuda_cl{int(cl)}_tf32{int(tf32)}"] = ms
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # ---- C  torch.compile
        row["compile"] = try_compile(net, x, reps) if (not modes or "compile" in modes) else None

        # ---- Q4 手工 CUDA Graph
        row["cudagraph"] = try_cuda_graph(net, x, reps) if (not modes or "graph" in modes) else None

        # ---- D  torch CPU
        cpu = {}
        for th in (cpu_threads if (not modes or "cpu" in modes) else ()):
            cpu[f"threads{th}"] = bench_torch_cpu(net.cpu(), x, max(5, reps // 2), th)
        row["torch_cpu"] = cpu
        net = net.cuda()
        res["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("compile")},
                         default=str, ensure_ascii=False), flush=True)
        del net
        _free()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="tag=path，缺省用 384/416 两个样本")
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 8])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cpu-threads", nargs="*", type=int, default=[4, 8, 16])
    ap.add_argument("--no-ort", action="store_true")
    ap.add_argument("--only", default=None, help="只跑某个 tag")
    ap.add_argument("--skip-ort-cpu", action="store_true", help="大模型别开 ORT CPU EP（建 session 要好几分钟）")
    ap.add_argument("--modes", default="", help="逗号分隔：ort,torch,compile,graph,cpu（空=全跑）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    models = DEFAULT_MODELS
    if a.models:
        models = {}
        for s in a.models:
            tag, p = s.split("=", 1)
            models[tag] = p
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    all_res = []
    for tag, p in models.items():
        if a.only and a.only != tag:
            continue
        print(f"\n=== {tag}: {p}")
        all_res.append(run_model(tag, p, a.batch, a.reps, a.cpu_threads,
                                 do_ort=not a.no_ort, skip_ort_cpu=a.skip_ort_cpu,
                                 modes=tuple(x.strip() for x in a.modes.split(",") if x.strip())))
    out = Path(a.out) if a.out else HERE / "dfm_torch_bench.json"
    out.write_text(json.dumps(all_res, indent=2, default=str), encoding="utf-8")
    print("\n[ok]", out)


if __name__ == "__main__":
    main()
