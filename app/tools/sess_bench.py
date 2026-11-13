# -*- coding: utf-8 -*-
r"""按模型量「GPU EP vs CPU EP」：单张 ms、线程扩展、池吞吐、每份 session 的内存。

为什么不用跑整条 fused 来量：整条链的端到端帧率是**瓶颈段**决定的（本机 detect 98%），
把 xseg 搬到 CPU 这种改动在端到端上看不出差别。要回答「加 CPU session 值不值」，
必须先有这张按模型的底表 —— 端到端只是它的推论。

量什么：
  ① 每个模型：GPU EP 单张 ms（中位）/ CPU EP 单张 ms（中位）
  ② CPU 的 intra_op 线程扩展：1 / 2 / 4 / 8 / 16 → 单张 ms
  ③ 池吞吐：m 份 CPU session × T 线程并行 → 张/秒（这才是「多开几份」的真实收益）
  ④ 每份 session 的内存占用（MB）：derive() 用它推 CPU session 上限

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\sess_bench.py
    runtime\server\Scripts\python.exe app\tools\sess_bench.py --quick
    runtime\server\Scripts\python.exe app\tools\sess_bench.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WS = REPO / "workspace"
WEIGHTS = REPO / "weights"


def mb(v):
    return round(float(v) / 1048576.0, 1)


def _setup_dll_dirs():
    """把 torch/lib、nvidia/*/bin、onnxruntime/capi 塞进 DLL 搜索路径。

    为什么必须自己做：这里是**独立进程**（不是后端），没有 app 的 _ensure_ort_dll_dirs。
    少了这一步 ORT 只会打一行 warning 就退到 CPU EP —— 于是「GPU 单张 ms」全是 None，
    「CPU 值不值」这种结论就建立在一个空列上。
    """
    import glob
    sp = None
    try:
        import site
        for cand in (site.getsitepackages() or []):
            if (Path(cand) / "onnxruntime").is_dir():
                sp = Path(cand)
                break
    except Exception:                                       # noqa: BLE001
        sp = None
    if sp is None:
        sp = REPO / "runtime" / "server" / "Lib" / "site-packages"
    dirs = []
    for pat in ("torch/lib", "nvidia/*/bin", "nvidia/*/lib", "cudnn/bin", "cublas/bin",
                "cudnn/lib", "cublas/lib", "onnxruntime/capi"):
        for x in glob.glob(str(Path(sp) / pat)):
            if Path(x).is_dir():
                dirs.append(x)
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except Exception:                                   # noqa: BLE001
            pass
        os.environ["PATH"] = str(d) + os.pathsep + str(os.environ.get("PATH") or "")
    return dirs


def rss_mb():
    try:
        import psutil
        return mb(psutil.Process().memory_info().rss)
    except Exception:                                       # noqa: BLE001
        return 0.0


def find(names):
    """按候选名在 workspace/ 和 weights/ 里找文件（平铺 + 老的分目录布局都认）。"""
    cands = []
    for base in (WEIGHTS, WS / "model", REPO / "weights"):
        for n in names:
            cands.append(Path(base) / n)
    for c in cands:
        if Path(c).is_file():
            return Path(c)
    return None


MODELS = [
    ("detect.retinaface_10g", ["retinaface_10g.onnx", "onnx/RetinaFace/RetinaFace.onnx",
                               "RetinaFace.onnx"]),
    ("detect.tinymog", ["tinymog.onnx", "onnx/TinyMog/TinyMog.onnx", "TinyMog.onnx"]),
    ("landmarks.2d106", ["insightface_2d106.onnx",
                         "onnx/InsightFace2d106/InsightFace2D106.onnx"]),
    ("xseg.xseglite", ["xseglite.onnx", "XSegLite/xseglite.onnx",
                       "model/XSegLite/xseglite.onnx", "xseglite_256.onnx"]),
    ("dfm.416", ["model/../model/阿証証証証-416.dfm", "阿証証証証-416.dfm"]),
    ("dfm.384", ["model/../model/张元英.dfm", "张元英.dfm"]),
]


def make_session(path, ep, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    provs = (["CUDAExecutionProvider", "CPUExecutionProvider"] if ep == "gpu"
             else ["CPUExecutionProvider"])
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=provs)
    load_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    got = list(sess.get_providers())
    if ep == "gpu" and got[:1] == ["CPUExecutionProvider"]:
        return None, load_ms, got
    return sess, load_ms, got


SIZE_HINT = {"detect.retinaface_10g": 640, "detect.tinymog": 640,
             "landmarks.2d106": 192, "xseg.xseglite": 256, "dfm.384": 384,
             "dfm.416": 416}


def shape_of(sess, hint=0):
    """输入形状。**动态维必须补成真尺寸**：retinaface 的 H/W 是动态的，全填 1 会量出
    「3.5 ms/张」这种假数（1x3x1x1 的前向当然快），整个「CPU 值不值」的结论就废了。
    """
    import numpy as np
    inp = sess.get_inputs()[0]
    shp = []
    for i, d in enumerate(list(inp.shape)):
        if isinstance(d, int):
            shp.append(int(d))
        else:
            shp.append(int(hint) if (i >= 2 and hint) else 1)
    if len(shp) == 4 and shp[1] in (1, 3):
        arr = np.random.rand(*shp).astype(np.float32)
    elif len(shp) == 4:
        arr = np.random.rand(*shp).astype(np.float32)
    else:
        arr = np.random.rand(*shp).astype(np.float32)
    return inp.name, arr, shp


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(ts), 2)


def measure_one(tag, path, n=5):
    out = {"name": tag, "path": str(path), "gpu": None, "cpu": None,
           "cpu_threads": {}, "mem_cpu_mb": None, "gpu_providers": None,
           "cpu_providers": None, "load_gpu_ms": None, "load_cpu_ms": None}
    try:
        import numpy as np
    except Exception as e:                                  # noqa: BLE001
        raise SystemExit("缺 numpy：%s" % e)
    # GPU
    r0 = rss_mb()
    sess, load_ms, provs = make_session(path, "gpu")
    out["load_gpu_ms"] = load_ms
    out["gpu_providers"] = provs
    hint = SIZE_HINT.get(tag) or 0
    if sess is not None:
        name, arr, shp = shape_of(sess, hint)
        out["input"] = shp
        try:
            out["gpu"] = bench(lambda: sess.run(None, {name: arr}), n=n)
        except Exception as e:                              # noqa: BLE001
            out["gpu_err"] = str(e)[:200]
        del sess
    # CPU（默认线程数）
    r1 = rss_mb()
    sess, load_ms, provs = make_session(path, "cpu")
    out["load_cpu_ms"] = load_ms
    out["cpu_providers"] = provs
    if sess is not None:
        name, arr, shp = shape_of(sess, hint)
        out["input"] = shp
        try:
            out["cpu"] = bench(lambda: sess.run(None, {name: arr}), n=max(3, n // 2))
        except Exception as e:                              # noqa: BLE001
            out["cpu_err"] = str(e)[:200]
        out["mem_cpu_mb"] = round(max(0.0, rss_mb() - r0), 1)
        del sess
    out["rss_after_mb"] = rss_mb()
    return out, shp if out.get("input") else None


def full_detect(tag, path, n=3):
    """整帧检测（1080p 零图）在 GPU / CPU 两条 EP 上的 ms —— 端到端只有这个数算数。"""
    import numpy as np
    out = {}
    try:
        sys.path.insert(0, str(REPO / "app"))
        from server._detect import RetinaFaceDet, TinyMogDet
    except Exception as e:                                  # noqa: BLE001
        return {"err": str(e)[:160]}
    frame = np.zeros((1920, 1080, 3), np.uint8)
    for ep, provs in (("gpu", None), ("cpu", ["CPUExecutionProvider"])):
        try:
            if "tinymog" in tag:
                det = TinyMogDet(str(path), workspace=str(WS), providers=provs)
            elif provs is None:
                det = RetinaFaceDet(path, 640)
            else:
                # RetinaFaceDet 没有 providers 通道：造好后换成 CPU EP 那份
                import onnxruntime as _ort
                det = RetinaFaceDet(path, 640)
                det.sess = _ort.InferenceSession(str(path), providers=list(provs))
                det.input_name = det.sess.get_inputs()[0].name
        except Exception as e:                              # noqa: BLE001
            out[ep] = None
            out[ep + "_err"] = str(e)[:160]
            continue
        if provs is None and list(getattr(det, "providers", []) or []):
            pass
        try:
            det.thresh = 0.5
            out[ep] = bench(lambda: det.detect(frame), n=n, warmup=1)
        except Exception as e:                              # noqa: BLE001
            out[ep] = None
            out[ep + "_err"] = str(e)[:160]
    return out


def cpu_thread_sweep(tag, path, threads_list, n=3):
    """CPU EP 的 intra_op 线程扩展：单份 session，改线程数，看单张 ms。"""
    rows = []
    for t in threads_list:
        sess, load_ms, provs = make_session(path, "cpu", threads=t)
        if sess is None:
            continue
        name, arr, shp = shape_of(sess, SIZE_HINT.get(tag) or 0)
        try:
            ms = bench(lambda: sess.run(None, {name: arr}), n=n, warmup=1)
        except Exception as e:                              # noqa: BLE001
            ms = None
        rows.append({"threads": int(t), "ms": ms})
        del sess
    return rows


def pool_scaling(path, tid, m_list, threads, n_each=3, batch=1):
    """m 份独立 session（各自 T 线程）并行跑 → 张/秒。这是「多开几份」的真实收益。"""
    out = []
    for m in m_list:
        sesss = []
        for _ in range(m):
            s, _l, _p = make_session(path, "cpu", threads=threads)
            if s is None:
                break
            sesss.append(s)
        if len(sesss) < m:
            out.append({"sessions": m, "ips": None, "note": "起不来"})
            continue
        name, arr, shp = shape_of(sesss[0], SIZE_HINT.get(tag) or 0)
        per = [[] for _ in range(m)]
        lock = threading.Lock()

        def work(k):
            s = sesss[k]
            ts = []
            for _ in range(n_each):
                t0 = time.perf_counter()
                try:
                    s.run(None, {name: arr})
                except Exception:                           # noqa: BLE001
                    return
                ts.append(time.perf_counter() - t0)
            with lock:
                per[k] = ts

        ths = [threading.Thread(target=work, args=(k,)) for k in range(m)]
        t_all = time.perf_counter()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        wall = time.perf_counter() - t_all
        total = sum(len(x) for x in per)
        out.append({"sessions": m, "threads": threads,
                    "ips": round(total / max(1e-6, wall), 2),
                    "per_ms": round(1000.0 * wall / max(1, total), 2)})
        del sesss
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="按模型量 GPU/CPU EP 的单张 ms 与池吞吐")
    ap.add_argument("--quick", action="store_true", help="3 次取中位、少扫几档")
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    ap.add_argument("--models", default="", help="只量这几个（逗号分隔的 tag）")
    args = ap.parse_args(argv)
    n = 3 if args.quick else 5
    threads_list = [1, 2, 4, 8] if args.quick else [1, 2, 4, 8, 16]
    want = [x.strip() for x in str(args.models).split(",") if x.strip()]

    dirs = _setup_dll_dirs()
    print("DLL 目录 %d 个（torch/lib、nvidia/*/bin …）" % len(dirs))
    try:
        import onnxruntime as ort
    except Exception as e:                                  # noqa: BLE001
        raise SystemExit("缺 onnxruntime：%s" % e)
    print("onnxruntime %s · providers %s" % (ort.__version__, ort.get_available_providers()))
    print("物理核 %d · 逻辑核 %d · RSS 起点 %.0f MB"
          % (_cores(), int(os.cpu_count() or 0), rss_mb()))
    res = []
    for tag, names in MODELS:
        if want and tag not in want:
            continue
        path = find(names)
        if path is None:
            print("[skip] %-22s 找不到 onnx（试过 %s）" % (tag, names[0]))
            continue
        try:
            one, shp = measure_one(tag, path, n=n)
        except Exception as e:                              # noqa: BLE001
            print("[fail] %-22s %r" % (tag, e))
            continue
        print("[ok]   %-22s gpu=%s ms · cpu=%s ms · ratio=%s · cpu内存≈%s MB · %s"
              % (tag, one.get("gpu"), one.get("cpu"),
                 (round(float(one["cpu"]) / float(one["gpu"]), 1)
                  if one.get("gpu") and one.get("cpu") else "-"),
                 one.get("mem_cpu_mb"), one.get("input")))
        if tag.startswith("detect."):
            # 真正该比的是**整帧 detect()**：模型只是其中一小块，1080p→640 的 letterbox
            # resize + NMS + python 开销都在这一步里（实测整段 48.9 ms，模型只占 7.9 ms）。
            mode = full_detect(tag, path, n=max(2, n // 2))
            for k, v in (mode or {}).items():
                one["full_" + k] = v
            print("       整帧 detect(1080p)：GPU=%s ms · CPU=%s ms" % (mode.get("gpu"),
                                                                    mode.get("cpu")))
        one["cpu_threads"] = cpu_thread_sweep(tag, path, threads_list, n=max(2, n // 2))
        print("       CPU 线程扩展：%s"
              % " ".join("T%d=%sms" % (r["threads"], r["ms"]) for r in one["cpu_threads"]))
        res.append(one)
    # 池扩展：只对「搬去 CPU 有意义」的那几个量
    scaling = {}
    for tag, names in MODELS:
        if want and tag not in want:
            continue
        if tag not in ("detect.retinaface_10g", "detect.tinymog", "xseg.xseglite",
                       "dfm.384", "dfm.416"):
            continue
        path = find(names)
        if path is None:
            continue
        t = 4 if tag.startswith("dfm") or "retinaface" in tag else 2
        m_list = [1, 2] if args.quick else [1, 2, 4]
        rows = pool_scaling(path, tag, m_list, t, n_each=2 if args.quick else 3)
        scaling[tag] = rows
        print("[pool] %-22s T%d → %s"
              % (tag, t, " · ".join("%d 份=%.2f 张/秒" % (r["sessions"], r["ips"] or 0)
                                    for r in rows)))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"models": res, "pool": scaling,
                        "cores": _cores(), "logical": int(os.cpu_count() or 0)},
                       ensure_ascii=False, indent=1), "utf-8")
        print("JSON：%s" % args.json)
    return 0


def _cores():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from server import _sess
        return int(_sess.probe().get("cores") or 0)
    except Exception:                                       # noqa: BLE001
        return max(1, int(os.cpu_count() or 1) // 2)


if __name__ == "__main__":
    raise SystemExit(main())
