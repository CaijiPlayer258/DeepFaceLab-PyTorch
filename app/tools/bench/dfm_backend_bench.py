# -*- coding: utf-8 -*-
r"""dfm 后端的统一基准：onnxruntime vs torch（fp32/tf32/bf16），带逐值对拍。

口径（每一条都是量的，报告里的表就是这个脚本产的）：

  · **一格一个进程**。同一进程里先跑 ORT 再跑 torch，8GB 卡会被 arena + 碎片吃满，
    驱动回退到系统内存，量出 7420 ms/张 这种假数（DFM_TORCH_REPORT.md `4 的坑）。
  · 每格两份耗时：
        run_ms   调 run() 的墙钟，**含** H2D 入参 + D2H 出参 —— 生产就是这个代价
                 （_dfm_run_batch 喂的是 CPU 上的 numpy，拿回来的也是 numpy）；
        gpu_ms   只算 GPU 上那一段（CUDA event、输入常数已在卡上、不出参）。
    ORT 拿不到「只有 GPU」的数（sess.run 内部就含两次拷贝），所以对表时看 run_ms。
  · 显存：采样 torch.cuda.mem_get_info() 的 (total − free)，减掉建会话之前的基线 ——
    整卡口径，和 DFM_TORCH_REPORT 里那两个数（661 / 1611 MB）同源。
  · 对拍：同一份输入，基准 = ORT CUDA fp32 的输出（存 npz），
    报 max|diff| / mean|diff|，以及**换成 0-255 像素尺度**的那个 max 和「>2 LSB 的比例」。

跑法（必须 server venv：它才是生产进程；engine venv 没有 onnx 也不影响这里）：

    runtime\server\Scripts\python.exe app\tools\bench\dfm_backend_bench.py --sweep
    ... --sweep --models 416 --dtypes fp32,tf32,bf16 --batches 1,8
    ... --cell <base64 json>                 # 驱动内部用，单格
    ... --refs                               # 只生成 ORT fp32 基准输出

产物：%TEMP%\dfm_backend_bench\<run>\cells.jsonl + refs\*.npz
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfm_backend_bench"

MODELS = {
    "416": REPO / "workspace" / "model" / "阿証証証証-416.dfm",
    "384": REPO / "workspace" / "model" / "张元英.dfm",
}
SIZES = {"416": 416, "384": 384}
ALIGNED = REPO / "workspace" / "merge" / "9月29日 (5)" / "aligned"


# ------------------------------------------------------------------ 输入
def make_input(tag, batch, source="noise"):
    """固定种子的输入。noise = 均匀噪声（对数值最狠）；real = 真对齐脸（最能代表肉眼）。"""
    S = SIZES[tag]
    if source == "real":
        import cv2
        files = sorted(ALIGNED.glob("*.jpg"))[:max(8, batch)]
        out = np.empty((batch, S, S, 3), np.float32)
        for i in range(batch):
            # cv2.imread 打不开非 ASCII 路径（"9月29日 (5)"），走 imdecode
            im = cv2.imdecode(np.fromfile(str(files[i % len(files)]), np.uint8),
                              cv2.IMREAD_COLOR)
            im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
            out[i] = im.astype(np.float32) / 255.0      # BGR：和 warpAffine 出来的一致
        return out
    rng = np.random.default_rng(1234 + S)
    return rng.random((batch, S, S, 3), dtype=np.float32)


def ref_path(tag, batch, source):
    d = BOX / "refs"
    d.mkdir(parents=True, exist_ok=True)
    return d / ("%s_b%d_%s.npz" % (tag, batch, source))


# ------------------------------------------------------------------ 显存采样
class MemSampler(threading.Thread):
    """整卡 used 的峰值：torch.cuda.mem_get_info 是直接问驱动，对 ORT 一样有效。"""

    def __init__(self, period=0.02):
        super().__init__(daemon=True)
        self.period = period
        self.stop_flag = False
        self.peak_used = 0.0
        self.total = 0.0
        self.base = 0.0

    def snapshot(self):
        try:
            import torch
            free, total = torch.cuda.mem_get_info(0)
            used = (total - free) / 1048576.0
            self.total = total / 1048576.0
            return used
        except Exception:                       # noqa: BLE001
            return 0.0

    def run(self):
        while not self.stop_flag:
            self.peak_used = max(self.peak_used, self.snapshot())
            time.sleep(self.period)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=2.0)
        self.peak_used = max(self.peak_used, self.snapshot())


def _med(xs):
    import statistics
    return float(statistics.median(xs))


def setup_dll():
    """ORT 的 CUDA EP 要 torch/lib 里的 cuDNN9 + cuBLASLt12；必须在建 session 之前加。"""
    sys.path.insert(0, str(REPO / "app" / "server"))
    try:
        from _ortcuda import add_cuda_dll_dirs
        d = add_cuda_dll_dirs()
        print("[dll] %s" % (d,))
    except Exception as e:                          # noqa: BLE001
        print("[dll] 失败：%r" % (e,))


# ------------------------------------------------------------------ 单格
def run_cell(cell):
    tag = cell["model"]
    path = str(MODELS[tag])
    backend = cell["backend"]
    dtype = cell["dtype"]
    batch = int(cell["batch"])
    source = cell.get("source", "noise")
    reps = int(cell.get("reps", 20))
    cl = bool(int(cell.get("cl", 0)))
    x = make_input(tag, batch, source)
    row = dict(cell)
    row["res"] = SIZES[tag]
    row["input_mb"] = round(x.nbytes / 1048576.0, 2)

    # 先把 CUDA context 建出来再取基线：context（~400MB/进程）是 _vram_plan 里单独一项
    # （_VRAM_CTX_MB），不该记到任何一个后端头上。两家共用同一个 primary context。
    import torch as _torch
    _torch.cuda.init()
    _torch.zeros(1, device="cuda")
    _torch.cuda.synchronize()
    ms = MemSampler()
    base = ms.snapshot()
    ms.base = base
    ms.peak_used = base
    ms.start()
    try:
        if backend == "ort":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            t0 = time.perf_counter()
            sess = ort.InferenceSession(path, sess_options=so,
                                        providers=["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
            row["load_s"] = round(time.perf_counter() - t0, 3)
            row["providers"] = list(sess.get_providers())
            nm = sess.get_inputs()[0].name
            for _ in range(3):
                sess.run(None, {nm: x})
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                outs = sess.run(None, {nm: x})
                ts.append((time.perf_counter() - t0) * 1000.0)
            row["run_ms"] = round(_med(ts), 2)
            row["run_ms_min"] = round(min(ts), 2)
            row["run_ms_p90"] = round(float(np.percentile(ts, 90)), 2)
            row["gpu_ms"] = None
            y = [np.asarray(o) for o in outs]
            del sess
        else:
            sys.path.insert(0, str(REPO / "app" / "server"))
            from backend_models import dfm_torch as DT
            t0 = time.perf_counter()
            s = DT.load(path, dtype=dtype, max_batch=max(8, batch), channels_last=cl)
            row["load_s"] = round(time.perf_counter() - t0, 3)
            row["describe"] = s.describe()
            feed = {s.input_name: x}
            for _ in range(3):
                s.run(None, feed)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                outs = s.run(None, feed)
                ts.append((time.perf_counter() - t0) * 1000.0)
            row["run_ms"] = round(_med(ts), 2)
            row["run_ms_min"] = round(min(ts), 2)
            row["run_ms_p90"] = round(float(np.percentile(ts, 90)), 2)
            # 只算 GPU 那一段（输入在卡上、不出参）：看「算力」而不是「搬运」
            net = s._net(batch)
            xd = s._to_device(x)
            import torch

            def _fwd():
                # 精度分支**必须**在预热的第一次也走：bf16w 的权重是 bf16，
                # 直接喂 float32 会 Input type (float) and bias type (BFloat16)。
                if s.dtype == "bf16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        net(xd)
                elif s.dtype == "bf16w":
                    net(xd.to(torch.bfloat16))
                else:
                    net(xd)

            with torch.no_grad():
                for _ in range(3):
                    _fwd()
                torch.cuda.synchronize()
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                gs = []
                for _ in range(reps):
                    ev0.record()
                    _fwd()
                    ev1.record()
                    torch.cuda.synchronize()
                    gs.append(ev0.elapsed_time(ev1))
            row["gpu_ms"] = round(_med(gs), 2)
            row["vram"] = s.vram_mb()
            row["stats"] = {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in s.stats.items()}
            y = [np.asarray(o) for o in outs]
    finally:
        ms.stop()

    row["peak_mb"] = round(max(0.0, ms.peak_used - base), 1)
    row["card_used_mb"] = round(ms.peak_used, 1)
    row["per_face_ms"] = round(row["run_ms"] / batch, 2)
    row["fps"] = round(1000.0 * batch / row["run_ms"], 2)
    row["out_shapes"] = [list(a.shape) for a in y]

    # ---- 对拍（基准 = ORT CUDA fp32）
    rp = ref_path(tag, batch, source)
    if rp.is_file():
        z = np.load(rp)
        ref = [z["o%d" % i] for i in range(len(z.files))]
        d = [np.abs(a.astype(np.float64) - b.astype(np.float64)) for a, b in zip(y, ref)]
        row["diff"] = {
            "max": [float(v.max()) for v in d],
            "mean": [float(v.mean()) for v in d],
            "max_lsb": [float(v.max()) * 255.0 for v in d],
            "pct_gt2lsb": [float((v * 255.0 > 2.0).mean() * 100.0) for v in d],
        }
        row["diff_max"] = max(row["diff"]["max"])
        row["diff_max_lsb"] = max(row["diff"]["max_lsb"])
        row["diff_mean"] = max(row["diff"]["mean"])
    return row


# ------------------------------------------------------------------ 基准输出
def gen_refs(tags, batches, sources):
    """基准输出 = ORT CUDA fp32（一个模型一个 session，扫完所有 batch 再放）。"""
    import onnxruntime as ort
    for tag in tags:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(MODELS[tag]), sess_options=so,
                                    providers=["CUDAExecutionProvider"])
        nm = sess.get_inputs()[0].name
        for b in batches:
            for src in sources:
                x = make_input(tag, b, src)
                t0 = time.perf_counter()
                outs = sess.run(None, {nm: x})
                dt = (time.perf_counter() - t0) * 1000.0
                rp = ref_path(tag, b, src)
                np.savez(rp, **{"o%d" % i: np.asarray(o) for i, o in enumerate(outs)})
                print("[ref] %s B=%d %-5s -> %s (%.1f ms)" % (tag, b, src, rp.name, dt))
        del sess


# ------------------------------------------------------------------ 驱动
def sweep(args):
    cells = []
    for tag in args.models:
        for b in args.batches:
            for dt in args.dtypes:
                cells.append({"model": tag, "backend": "torch", "dtype": dt,
                              "batch": int(b), "reps": args.reps,
                              "source": args.source})
                if dt == "fp32":            # ONNX 只有 fp32 一档：每个 batch 只跑一次
                    cells.append({"model": tag, "backend": "ort", "dtype": "fp32",
                                  "batch": int(b), "reps": args.reps,
                                  "source": args.source})
    if args.cl:
        for tag in args.models:
            for b in args.batches:
                for dt in args.dtypes:
                    if dt == "fp32":
                        continue
                    cells.append({"model": tag, "backend": "torch", "dtype": dt,
                                  "batch": int(b), "reps": args.reps,
                                  "source": args.source, "cl": 1})
    run = args.run or time.strftime("%m%d_%H%M%S")
    outdir = BOX / run
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "cells.jsonl"
    py = sys.executable
    print("[sweep] %d 格 -> %s" % (len(cells), out), flush=True)
    for i, c in enumerate(cells):
        blob = base64.b64encode(json.dumps(c).encode()).decode()
        t0 = time.time()
        # encoding 必须写死：子进程打的是 UTF-8 中文，不写就按 GBK 解，
        # 在 cp936 控制台上直接 UnicodeDecodeError，整个 sweep 一格都跑不出来。
        p = subprocess.run([py, str(Path(__file__)), "--cell", blob],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        line = ""
        for ln in (p.stdout or "").splitlines():
            if ln.startswith("[cell]"):
                line = ln[6:]
        if not line:
            print("[sweep] %d/%d 失败 %s\n%s\n%s"
                  % (i + 1, len(cells), c, (p.stdout or "")[-800:], (p.stderr or "")[-800:]))
            continue
        row = json.loads(line)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[sweep] %d/%d %-4s %-5s %-5s b%d%s run %6.2f ms  每张 %5.2f  %5.0f MB"
              % (i + 1, len(cells), c["model"], c["backend"], c["dtype"], c["batch"],
                 " CL" if c.get("cl") else "", row["run_ms"], row["per_face_ms"],
                 row["peak_mb"]), flush=True)
    print("[sweep] done -> %s" % out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None, help="base64(json) 单格")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--models", nargs="*", default=["416", "384"])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 4, 8])
    ap.add_argument("--dtypes", nargs="*", default=["fp32", "tf32", "bf16"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--source", default="noise", choices=["noise", "real"])
    ap.add_argument("--cl", action="store_true", help="额外跑一组 channels_last")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    setup_dll()
    if a.cell:
        cell = json.loads(base64.b64decode(a.cell).decode())
        row = run_cell(cell)
        print("[cell]" + json.dumps(row, ensure_ascii=False, default=str))
        return 0
    if a.refs:
        gen_refs(a.models, a.batches, ["noise", "real"])
        return 0
    sweep(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
