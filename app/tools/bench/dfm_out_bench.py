# -*- coding: utf-8 -*-
r"""dfm 出参路径（D2H + 流同步）的拆解 —— 只量这一件事，不碰像素。

为什么单独一支：_forward_one 里 t_out 实测 9.22 ms/批（8 张），占 GPU 空转的 42.5%，
但那是**一个数字**，不是**一个原因**。这个脚本把它拆成：

    detach / dtype 转换(float) / pinned 分配 / copy_ 入队 / 流同步 / numpy 包装

并给出五种候选实现的实测对比（含「合并成一次拷贝 + 一次同步 + 常驻 buffer」），
以及 PCIe D2H 的**下限**（纯 memcpy，无 forward、无 pinned 分配）。

跑法（仓库根，必须 server venv —— 生产进程就是它）：
    runtime\server\Scripts\python.exe app\tools\bench\dfm_out_bench.py
    ... --batch 8 --reps 30 --dtype bf16w --source real
    ... --json out.json

口径：
  · 「隔离态」= 先 torch.cuda.synchronize() 再做那一步：量的是这一步自己的钱；
  · 「真实态」= forward 刚入队就做出参：量的是生产里 t_out 真正等的那段（含前向尾巴）；
  · 两列都报，因为优化只该砍「隔离态」里属于多余开销的部分。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
APP = REPO / "app"
for _p in (APP, APP / "server"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

MODELS = {"416": REPO / "workspace" / "model" / "阿証証証証-416.dfm",
          "384": REPO / "workspace" / "model" / "张元英.dfm"}
SIZES = {"416": 416, "384": 384}
ALIGNED = REPO / "workspace" / "merge" / "9月29日 (5)" / "aligned"
MB = 1048576.0


def make_input(tag, batch, source="noise"):
    S = SIZES[tag]
    if source == "real":
        import cv2
        files = sorted(ALIGNED.glob("*.jpg"))[:max(8, batch)]
        out = np.empty((batch, S, S, 3), np.float32)
        for i in range(batch):
            im = cv2.imdecode(np.fromfile(str(files[i % len(files)]), np.uint8),
                              cv2.IMREAD_COLOR)
            im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
            out[i] = im.astype(np.float32) / 255.0
        return out
    rng = np.random.default_rng(1234 + S)
    return rng.random((batch, S, S, 3), dtype=np.float32)


def med(xs):
    return float(statistics.median(xs)) if len(xs) else float("nan")


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, np.float64), q)) if len(xs) else float("nan")


def say(*a):
    print(*a, flush=True)


def table(rows, cols, title=None):
    if title:
        say("")
        say(title)
    line = "  ".join((str(h).ljust(w) if a == "l" else str(h).rjust(w)) for h, w, a in cols)
    say(line)
    say("-" * len(line))
    for r in rows:
        cells = []
        for (h, w, a), v in zip(cols, r):
            s = "" if v is None else str(v)
            cells.append(s.ljust(w) if a == "l" else s.rjust(w))
        say("  ".join(cells))


# ------------------------------------------------------------------ 各家出参实现
_BUF = {}


def _keep(key, n, dtype):
    b = _BUF.get(key)
    if b is None or b.numel() != n or b.dtype != dtype:
        b = torch.empty(n, dtype=dtype, pin_memory=True)
        _BUF[key] = b
    return b


def out_cur(sess, ys):
    """现状：每个张量各自 float() + 分配 pinned + copy_ + 一次流同步。"""
    res = []
    for t in ys:
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        buf = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
        buf.copy_(t, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        res.append(buf.numpy())
    return res


def out_cur_break(ys, br):
    """现状 + 每一步打点（隔离态用：调用前已 drain）。"""
    res = []
    for t in ys:
        t0 = time.perf_counter()
        t = t.detach()
        t1 = time.perf_counter()
        if t.dtype != torch.float32:
            t = t.float()
        t2 = time.perf_counter()
        buf = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
        t3 = time.perf_counter()
        buf.copy_(t, non_blocking=True)
        t4 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        t5 = time.perf_counter()
        a = buf.numpy()
        t6 = time.perf_counter()
        br["detach"].append((t1 - t0) * 1000)
        br["float"].append((t2 - t1) * 1000)
        br["alloc"].append((t3 - t2) * 1000)
        br["copy_enq"].append((t4 - t3) * 1000)
        br["sync"].append((t5 - t4) * 1000)
        br["numpy"].append((t6 - t5) * 1000)
        res.append(a)
    return res


def out_nofloat(sess, ys):
    """去掉 float()：bf16 原样拷到 pinned，再在 CPU 上**精确**拓宽成 fp32
    （位模式完全保留，像素不变），PCIe 只走一半字节。"""
    res = []
    for t in ys:
        t = t.detach()
        buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        buf.copy_(t, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        a = buf.numpy()
        res.append((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32))
    return res


def out_merged(sess, ys):
    """一次分配 + 每个张量一次 copy_ 进大 buffer 的切片 + **只有一次**流同步。"""
    tot = sum(int(t.numel()) for t in ys)
    buf = torch.empty(tot, dtype=torch.float32, pin_memory=True)
    off = 0
    views = []
    for t in ys:
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        n = int(t.numel())
        v = buf[off:off + n].view(t.shape)
        v.copy_(t, non_blocking=True)
        off += n
        views.append(v)
    torch.cuda.current_stream().synchronize()
    return [v.numpy() for v in views]


def out_merged_reuse(sess, ys):
    """合并 + 常驻 pinned（按 batch/dtype 分桶，不每帧 cudaHostAlloc）。"""
    tot = sum(int(t.numel()) for t in ys)
    buf = _keep(("f32", tot), tot, torch.float32)
    off = 0
    views = []
    for t in ys:
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        n = int(t.numel())
        v = buf[off:off + n].view(t.shape)
        v.copy_(t, non_blocking=True)
        off += n
        views.append(v)
    torch.cuda.current_stream().synchronize()
    return [v.numpy() for v in views]


def out_merged_reuse_event(sess, ys):
    """合并 + 常驻 + 用 CUDA event 而不是循环流同步（对照组：WDDM 提交开销）。"""
    tot = sum(int(t.numel()) for t in ys)
    buf = _keep(("f32", tot), tot, torch.float32)
    off = 0
    views = []
    for t in ys:
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        n = int(t.numel())
        v = buf[off:off + n].view(t.shape)
        v.copy_(t, non_blocking=True)
        off += n
        views.append(v)
    ev = torch.cuda.Event()
    ev.record()
    ev.synchronize()
    return [v.numpy() for v in views]


def out_pageable(sess, ys):
    """pin=False 的现状（对照：pageable 那条路有多贵）。"""
    res = []
    for t in ys:
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        res.append(t.cpu().numpy())
    return res


VARIANTS = [("cur 现状（每张量 pinned+sync）", out_cur),
            ("nofloat（bf16 直拷+CPU 拓宽）", out_nofloat),
            ("merged（合并 buffer+一次 sync）", out_merged),
            ("merged+reuse（常驻 pinned）", out_merged_reuse),
            ("merged+reuse+event", out_merged_reuse_event),
            ("pageable（pin=False 对照）", out_pageable)]


# ------------------------------------------------------------------ PCIe 下限
def pcie_floor(nbytes, reps=30):
    """纯 D2H：一块 dev 上的连续 buffer -> pinned，一次 copy_ + 一次 sync。"""
    n = int(nbytes // 4)
    d = torch.empty(n, dtype=torch.float32, device="cuda")
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    hp = torch.empty(n, dtype=torch.float32)                       # pageable 对照
    for _ in range(3):
        h.copy_(d, non_blocking=True)
    torch.cuda.synchronize()
    ts, gs, tp = [], [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        h.copy_(d, non_blocking=True)
        ev1.record()
        ev1.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
        gs.append(ev0.elapsed_time(ev1))
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hp.copy_(d)
        tp.append((time.perf_counter() - t0) * 1000)
    return {"bytes": int(n * 4), "pinned_ms": round(med(ts), 3),
            "gpu_ms": round(med(gs), 3), "pageable_ms": round(med(tp), 3),
            "gbps_pinned": round((n * 4) / (med(ts) / 1000.0) / 1e9, 2),
            "gbps_gpu": round((n * 4) / (med(gs) / 1000.0) / 1e9, 2)}


# ------------------------------------------------------------------ 同进程 A/B
def ab_mode(s, tag, batches, reps, rounds):
    """同进程内交替 A/B：legacy（每张量分配+同步）vs merged（一条 pinned+一次同步）。

    方法学：**交替**而不是两段独立跑 —— 这台机器上 fp32 前向在 16~26ms 之间摆
    （桌面 Wallpaper Engine / 浏览器 GPU 加速会抢），两次独立启动的差不可信。
    每轮都换实现、每档 ≥3 轮，报中位数和原始值。
    """
    from backend_models import dfm_torch as _DT
    say("")
    say("同进程交替 A/B（%d 轮 × 每轮 %d 次 · 量的是 _forward_one 整段墙钟）"
        % (rounds, reps))
    rows = []
    for b in batches:
        x = make_input(tag, b, "noise")
        xd = s._to_device(x)
        for mode in ("legacy", "merged"):
            s.out_mode = mode
            s.forward(x)                 # 每档先真跑一次：cuDNN 形状/算法选完再计时
        torch.cuda.synchronize()
        res = {"legacy": [], "merged": []}
        sub = {m: {"t_out": [], "t_dsync": [], "t_copy": [], "t_gpu": []}
               for m in ("legacy", "merged")}
        for _r in range(rounds):
            for mode in ("legacy", "merged"):
                s.out_mode = mode
                st0 = dict(s.stats)
                for _ in range(reps):
                    t0 = time.perf_counter()
                    s._forward_one(x)
                    res[mode].append((time.perf_counter() - t0) * 1000)
                for k in sub[mode]:
                    sub[mode][k].append((s.stats[k] - st0[k]) / max(1, reps))
        for mode in ("legacy", "merged"):
            rows.append([mode, b, round(med(res[mode]), 2),
                         round(med(res[mode]) / b, 2),
                         round(med(sub[mode]["t_out"]), 2),
                         round(med(sub[mode]["t_dsync"]), 2),
                         round(med(sub[mode]["t_copy"]), 3),
                         round(med(sub[mode]["t_gpu"]), 2),
                         "%.1f~%.1f" % (min(res[mode]), max(res[mode]))])
    table(rows, [("实现", 7, "l"), ("B", 2, "r"), ("整段ms", 8, "r"),
                 ("ms/张", 7, "r"), ("t_out", 8, "r"), ("t_dsync", 8, "r"),
                 ("t_copy", 7, "r"), ("t_gpu", 8, "r"), ("原始区间", 13, "r")],
          title="同进程交替 A/B：出参搬法对 t_out 的影响（前向本身不动，t_gpu 是控制变量）")
    return rows


def ab_pixels(s, tag, batch):
    """同一批 device 输出，用两种搬法各搬一次，逐位比对（必须 max=0）。"""
    from backend_models import dfm_torch as _DT
    x = make_input(tag, batch, "real")
    xd = s._to_device(x)
    with torch.no_grad():
        if s.dtype == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ys = s._net(batch)(xd)
        elif s.dtype == "bf16w":
            ys = s._net(batch)(xd.to(torch.bfloat16))
        else:
            ys = s._net(batch)(xd)
    s.out_mode = "legacy"
    a = [np.array(v, copy=True) for v in s._outs(ys)]
    s.out_mode = "merged"
    b = [np.array(v, copy=True) for v in s._outs(ys)]
    st = []
    for i, (u, v) in enumerate(zip(a, b)):
        d = np.abs(u.astype(np.float64) - v.astype(np.float64))
        st.append({"i": i, "shape": list(u.shape), "max": float(d.max()),
                   "nz": int(np.count_nonzero(d))})
    say("")
    say("逐位对拍（legacy vs merged · 同一份 device 输出 · u8 尺度 = ×255）")
    table([[r["i"], str(r["shape"]), "%.3g" % r["max"], "%.3g" % (r["max"] * 255),
            r["nz"]] for r in st],
          [("输出", 4, "r"), ("形状", 22, "l"), ("max|diff|", 12, "r"),
           ("max LSB", 9, "r"), ("非零像素", 9, "r")])
    return st


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="416")
    ap.add_argument("--dtype", default="bf16w")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--source", default="noise", choices=["noise", "real"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--ab", action="store_true",
                    help="同进程交替 A/B：legacy vs merged（出参搬法）")
    ap.add_argument("--ab-batches", nargs="*", type=int, default=[1, 4, 8])
    ap.add_argument("--ab-rounds", type=int, default=3)
    ap.add_argument("--ab-reps", type=int, default=12)
    a = ap.parse_args()

    from backend_models import dfm_torch as DT
    path = str(MODELS[a.model])
    b = int(a.batch)
    say("[load] %s dtype=%s batch=%d" % (Path(path).name, a.dtype, b))
    s = DT.load(path, dtype=a.dtype, max_batch=max(8, b))
    net = s._net(b)
    x = make_input(a.model, b, a.source)
    xd = s._to_device(x)

    def fwd():
        with torch.no_grad():
            if s.dtype == "bf16":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return net(xd)
            if s.dtype == "bf16w":
                return net(xd.to(torch.bfloat16))
            return net(xd)

    for _ in range(5):
        ys = fwd()
    torch.cuda.synchronize()
    dtypes = [str(t.dtype).replace("torch.", "") for t in ys]
    total_b = sum(int(t.numel()) * t.element_size() for t in ys)
    fp32_b = sum(int(t.numel()) * 4 for t in ys)
    say("[out] %d 个输出 · dtype=%s · 实际 %.2f MB/批 · 按 fp32 出参 %.2f MB/批"
        % (len(ys), dtypes, total_b / MB, fp32_b / MB))
    for i, t in enumerate(ys):
        say("      o%d %-20s %-8s contiguous=%s %.2f MB"
            % (i, str(list(t.shape)), str(t.dtype).replace("torch.", ""),
               bool(t.is_contiguous()), t.numel() * t.element_size() / MB))

    if a.ab:
        ab_pixels(s, a.model, b)
        ab_mode(s, a.model, a.ab_batches, a.ab_reps, a.ab_rounds)
        return 0

    out = {"model": a.model, "dtype": a.dtype, "batch": b, "source": a.source,
           "out_shapes": [list(t.shape) for t in ys], "out_dtypes": dtypes,
           "bytes_fp32": fp32_b}

    # ---- ① 隔离态逐步拆解
    br = {k: [] for k in ("detach", "float", "alloc", "copy_enq", "sync", "numpy")}
    for _ in range(a.reps):
        torch.cuda.synchronize()
        out_cur_break(ys, br)
    rows = [[k, round(med(v) * len(ys), 3), round(med(v), 3), round(sum(v), 1)]
            for k, v in br.items()]
    tot = sum(sum(v) for v in br.values())
    rows = [r + [("%.1f%%" % (100.0 * r[3] / tot if tot else 0))] for r in rows]
    table(rows, [("步骤", 12, "l"), ("批合计ms", 10, "r"), ("单次ms", 9, "r"),
                 ("总量ms", 8, "r"), ("占比", 7, "r")],
          title="① 现状出参的逐步拆解（隔离态：每步之前已 drain · %d 批 × %d 输出）"
                % (a.reps, len(ys)))
    say("    合计每批 %.2f ms（隔离态 = 只算出参自身；生产 t_out 还要含前向尾巴）"
        % (tot / a.reps))
    out["breakdown_isolated"] = {k: round(med(v), 3) for k, v in br.items()}
    out["breakdown_isolated_total"] = round(tot / a.reps, 3)

    # ---- ② 六种实现
    table_rows = []
    for name, fn in VARIANTS:
        t_iso = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(s, ys)
            t_iso.append((time.perf_counter() - t0) * 1000)
        t_real = []
        for _ in range(a.reps):
            ys2 = fwd()
            t0 = time.perf_counter()
            fn(s, ys2)
            t_real.append((time.perf_counter() - t0) * 1000)
        table_rows.append([name, round(med(t_iso), 2), round(med(t_real), 2),
                           round(med(t_real) / b, 3)])
    table(table_rows, [("出参实现", 33, "l"), ("隔离ms", 9, "r"), ("真实ms", 9, "r"),
                       ("真实ms/张", 10, "r")],
          title="② 六种出参实现（同一份 device 输出；真实态 = 前向刚入队就要出参）")
    out["variants"] = [{"name": r[0], "iso_ms": r[1], "real_ms": r[2]} for r in table_rows]

    # ---- ③ PCIe 下限
    say("")
    floors = []
    for nb in (total_b, fp32_b):
        nb = int(nb // 1048576 * 1048576)
        if nb <= 0:
            continue
        f = pcie_floor(nb)
        floors.append(f)
        say("[pcie] %.2f MB 连续 D2H：pinned %.3f ms (%.2f GB/s) · 纯 GPU 段 %.3f ms "
            "(%.2f GB/s) · pageable %.3f ms"
            % (nb / MB, f["pinned_ms"], f["gbps_pinned"], f["gpu_ms"], f["gbps_gpu"],
               f["pageable_ms"]))
    out["pcie"] = floors

    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
        say("")
        say("[json] %s" % a.json)
    say("[env] torch %s · cuda %s · %s"
        % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
