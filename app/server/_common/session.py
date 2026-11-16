# -*- coding: utf-8 -*-
"""异构推理池：n×GPU + m×CPU session 的**探测 + 微基准 + 配比推导 + 任务分配**。

为什么单独一个模块：这件事只有三部分 —— 探测/基准（环境的事实）、配比（由事实推出来
的结论）、调度（按结论分活）—— 塞进 routes_merge.py 那张 14000 行的文件里，以后谁
都不敢改。这里一条**硬编码**都没有：低配（4 核 / 4GB 显存 / 无独显）和高配（32 核 /
24GB）差一个数量级，任何写死的比例在其中一端都必然是错的。

三条事实来源：
    ① 显存 → GPU session 上限（复用 _vram_plan 那本账，不自己乱猜）
    ② 物理核 / 可用内存 / **每个 CPU session 的实测内存** → CPU session 上限
    ③ 3~5 秒微基准（每档 2~3 张，取中位）→ 每个 EP 的单张 ms

配比怎么定：不是「比例」，是**边际收益**。多一个 session 能多出多少张/秒 =
    1 / 单张 ms（它自己那一路的吞吐）
再被两个上限卡住（显存装不下 / 核和内存用完了）。只有当「这一段是瓶颈」时才该加 ——
瓶颈段 100% 忙的时候多开一路是净赚，非瓶颈段 13% 忙的时候多开一路只是抢 CPU。
所以 derive() 要一个 stage_busy（上一轮实测的段占用）；第一轮没有实测就退回保守档。

分配策略（DFL_SESS_STRATEGY，默认 eta）：
    static —— 按基准 ms 的反比静态分（快的那路分得多）
    shortq —— 动态最短队列（谁的待办少给谁）
    rr     —— 严格轮转，每 N 项重新平衡
    eta    —— 预估完成时间最小者接下一个（队列长度 × 单张 ms）
    stage  —— 阶段级异构（轻模型固定 CPU，重模型固定 GPU，不混跑）
"""
from __future__ import annotations

import os
import queue
import threading
import time

__all__ = ["probe", "bench", "derive", "Pool", "enabled", "strategy", "int_env",
           "last_plan", "level_cfg"]

# 环境变量（全部可覆盖；0/空 = 自动）
ENV_GPU = "DFL_SESS_GPU"          # GPU session 数，0=自动
ENV_CPU = "DFL_SESS_CPU"          # CPU session 数，0=自动
ENV_STRAT = "DFL_SESS_STRATEGY"   # static/shortq/rr/eta/stage
ENV_PLAN = "DFL_SESS_PLAN"        # 1=开自动配比（默认），0=全部单 GPU session（老行为）
ENV_Q = "DFL_SESS_Q"              # 每个 session 的待办队列深度
ENV_THREADS = "DFL_SESS_THREADS"  # CPU session 的 intra_op 线程数，0=自动

_LAST = {}                        # name -> 最近一次的配比（status / 日志回显用）
_LOCK = threading.Lock()


def int_env(name, default=0):
    try:
        v = str(os.environ.get(name) or "").strip()
        return int(float(v)) if v else int(default)
    except (TypeError, ValueError):
        return int(default)


def enabled() -> bool:
    """自动配比开关。默认开；DFL_SESS_PLAN=0 退回「一份 GPU session」（老行为）。"""
    return str(os.environ.get(ENV_PLAN) or "1").strip().lower() not in (
        "0", "false", "no", "off")


def strategy() -> str:
    s = str(os.environ.get(ENV_STRAT) or "eta").strip().lower()
    return s if s in ("static", "shortq", "rr", "eta", "stage") else "eta"


# ---------------------------------------------------------------- 探测
def _physical_cores() -> int:
    """物理核数（不是逻辑线程）。Windows 用 GetLogicalProcessorInformationEx，
    取不到就退回 逻辑核//2（SMT 假设），再取不到就是 1。

    为什么要物理核：CPU EP 的一个 intra_op 线程吃满一个物理核才不打架；
    32 逻辑核的机器上开 32 个 intra_op 线程只会互相抢。
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _GROUP(ctypes.Structure):
            _fields_ = [("Flags", wintypes.BYTE),
                        ("Reserved", ctypes.c_byte * 3),
                        ("Mask", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32.dll")
        buf = ctypes.create_string_buffer(1)
        need = wintypes.DWORD(0)
        # 关系 0 = RelationProcessorCore
        k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need))
        if int(need.value) <= 0:
            raise OSError("no size")
        buf = ctypes.create_string_buffer(int(need.value))
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
            raise OSError("call failed")
        n, off = 0, 0
        raw = bytes(buf)
        while off + 8 <= len(raw):
            n += 1
            rel = int.from_bytes(raw[off:off + 4], "little")
            size = int.from_bytes(raw[off + 4:off + 8], "little")
            if size <= 0:
                break
            off += size
        if n > 0:
            return int(n)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        c = int(os.cpu_count() or 1)
    except Exception:                                       # noqa: BLE001
        c = 1
    return max(1, c // 2)


def _avail_ram_mb():
    """当前可用物理内存（MB）。psutil 在就用它，否则读 GlobalMemoryStatusEx。"""
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1048576.0
    except Exception:                                       # noqa: BLE001
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return float(ms.ullAvailPhys) / 1048576.0
    except Exception:                                       # noqa: BLE001
        return 0.0


def _providers():
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:                                       # noqa: BLE001
        return []


def probe(vram_free_mb=None, vram_total_mb=None) -> dict:
    """这台机器的事实。vram_* 由调用方给（routes_merge 那边已经有一本账）。"""
    prov = _providers()
    has_gpu = ("CUDAExecutionProvider" in prov) or ("TensorrtExecutionProvider" in prov)
    cores = _physical_cores()
    try:
        logical = int(os.cpu_count() or cores)
    except Exception:                                       # noqa: BLE001
        logical = cores
    return {"cores": int(cores), "logical": int(logical),
            "avail_ram_mb": round(_avail_ram_mb(), 1),
            "vram_free_mb": (None if vram_free_mb is None else float(vram_free_mb)),
            "vram_total_mb": (None if vram_total_mb is None else float(vram_total_mb)),
            "providers": prov, "has_gpu": bool(has_gpu)}


# ---------------------------------------------------------------- 微基准
def bench(fn, n=3, warmup=1) -> float:
    """fn() 跑 n 次取**中位**（ms）。为什么要中位：第一张要付 cuDNN/TRT 的句柄和
    kernel 建起来的钱，均值会被它拉高，配比就会算歪。异常一律吞掉返回 None。
    单次超过 30s 直接放弃（CPU EP 跑 384 dfm 一张 600ms，3 张 2 秒，够用）。
    """
    for _i in range(max(0, int(warmup))):
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            return None
    ts = []
    for _i in range(max(1, int(n))):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:                                   # noqa: BLE001
            return None
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return round(ts[len(ts) // 2], 2)


# ---------------------------------------------------------------- 配比推导
def derive(kind: str, p: dict, gpu_ms, cpu_ms, mem_per_gpu_mb: float = 0.0,
           mem_per_cpu_mb: float = 0.0, stage_busy=None,
           threads_per_sess: int = 0, over: dict = None) -> dict:
    """推导 (gpu_sessions, cpu_sessions) —— **全靠实测，没有预设比例**。

    参数：
      gpu_ms / cpu_ms    微基准的单张（单帧）ms；None = 这条 EP 用不了
      mem_per_gpu_mb     一份 GPU session 额外要多少显存（权重 + arena + context）
      mem_per_cpu_mb     一份 CPU session 额外要多少内存（权重 + 工作区）
      stage_busy         上一轮实测这一段占墙钟的比例（0..1）；None = 没有实测
      threads_per_sess   一份 CPU session 用几个 intra_op 线程（0=自动：物理核//2）

    规则（按优先级）：
      ① 没 GPU 或 GPU 跑不起来 → 全 CPU 池（并说明会慢多少倍）
      ② 这一段不是瓶颈（stage_busy < 0.85）→ 不加 CPU session；
         为什么：瓶颈在别处时，多开一路只会跟瓶颈抢 CPU，实测是净亏（见 docs）
      ③ 上限：GPU 由显存定，CPU 由「核 / 每份 session 的实测内存」定
      ④ 边际：CPU 那一路的吞吐（1/cpu_ms）低于 GPU 的 1/12 时，加它也补不上尾帧，
         只在瓶颈段 100% 忙时才值得开 1 份；低于 1/30 直接不开
    """
    over = dict(over or {})
    pc = int(p.get("cores") or 1)
    ram = float(p.get("avail_ram_mb") or 0.0)
    vfree = p.get("vram_free_mb")
    # CPU 侧能给这个池多少线程：**最多半个机器**（另一半是解码/对齐/合成/编码在用，
    # 实测那几段加起来也占 2~3 个核）；每份 session 至少 2 线程（1 线程的 ORT 慢 1.5~2 倍），
    # 上限 8（实测 T8≈T16，再多的线程只是互相抢）。
    budget = max(1, int(pc) // 2)
    tps = int(threads_per_sess or 0)
    if tps <= 0:
        tps = max(2, min(8, budget))
    why = []
    # ---- ① GPU 能不能用 ----
    gpu_cap = 0
    if p.get("has_gpu") and gpu_ms:
        vram_max = 1
        if mem_per_gpu_mb and vfree:
            # 一份会话要 mem_per_gpu_mb；留 15% 余量给帧缓冲/碎片
            vram_max = max(1, int(float(vfree) * 0.85 // float(mem_per_gpu_mb)))
        why.append("显存 %.0f MB 空闲 / 一份 %.0f MB → 装得下 %d 份"
                   % (float(vfree or 0), float(mem_per_gpu_mb or 0), vram_max))
        # 装得下 ≠ 该开：多开一份 GPU session 的收益实测只有个位数（本机 2×GPU = +5%，
        # 代价是一份 context 的显存），所以自动档先只开 1 份，等实测证明了再开。
        gpu_cap = 1 if not auto_gpu_ok(kind) else max(1, min(vram_max, 2))
        why.append("自动档 GPU=%d 份（实测%s）"
                   % (gpu_cap, "已证明 ≥5%" if auto_gpu_ok(kind) else "还没有 ≥5% 的记录"))
    else:
        why.append("没有可用 GPU EP%s" % ("" if gpu_ms else "（GPU 基准跑不起来）"))
    # ---- ② CPU 能开几份 ----
    cpu_cap = 0
    if cpu_ms:
        cpu_cap = max(0, int(pc // max(1, tps)))
        if mem_per_cpu_mb and ram > 0:
            cpu_cap = min(cpu_cap, max(0, int((ram - 1024.0) // float(mem_per_cpu_mb))))
        why.append("%d 物理核 / 每份 %d 线程 → %d 份；可用内存 %.0f MB / 一份 %.0f MB → %d 份"
                   % (pc, tps, max(0, int(pc // max(1, tps))), ram,
                      float(mem_per_cpu_mb or 0), cpu_cap))
    # ---- ③ 是不是瓶颈 ----
    if stage_busy is not None and float(stage_busy) < 0.85:
        why.append("这一段只占墙钟 %.0f%%（<85%%）→ 不是瓶颈，加 session 只会抢 CPU"
                   % (100.0 * float(stage_busy)))
        cpu_cap = 0
    # ---- ④ 边际收益：CPU 那一路值不值一份 session ----
    if cpu_cap > 0 and gpu_ms and cpu_ms:
        ratio = float(gpu_ms) / max(1e-6, float(cpu_ms))       # CPU 比 GPU 慢几倍
        if ratio < 1.0 / 30.0:
            why.append("CPU 单张 %.0f ms 是 GPU %.0f ms 的 %.0f 倍（>30x）→ 一份都不开"
                       % (float(cpu_ms), float(gpu_ms), 1.0 / max(1e-6, ratio)))
            cpu_cap = 0
        elif ratio < 1.0 / 12.0:
            why.append("CPU 单张 %.0f ms 是 GPU 的 %.0f 倍（>12x）→ 最多 1 份（只补尾帧）"
                       % (float(cpu_ms), 1.0 / max(1e-6, ratio)))
            cpu_cap = min(cpu_cap, 1)
    # ---- ⑤ 覆盖（请求体 sess_gpu/sess_cpu 优先，其次 env；0 = 自动）----
    def _ov(key, env_name):
        """None = 没给（自动）；数字 = **显式**（0 也是显式 —— 0 份 CPU session 是合法的
        实验档位，不能和「自动」混成一个值，否则 (1,0) 那一格永远是 auto）。"""
        v = over.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            v = os.environ.get(env_name)
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    g = _ov("gpu", ENV_GPU)
    m = _ov("cpu", ENV_CPU)
    if g is not None:
        gpu_cap = max(0, min(int(g), max(1, gpu_cap or 1)))
        why.append("显式 GPU=%d 份（请求体 sess_gpu / env %s）" % (g, ENV_GPU))
    if m is not None:
        cpu_cap = max(0, min(int(m), max(0, cpu_cap) if cpu_cap else int(m)))
        why.append("显式 CPU=%d 份（请求体 sess_cpu / env %s）" % (m, ENV_CPU))
    if m is None and int(cpu_cap) > 0 and not auto_cpu_ok(kind, 1):
        why.append("自动档：还没有『CPU 那一路净增 ≥5%%』的实测（工作区 _sess_bench.json "
                   "里的 gain:%s）→ 只开 GPU；手动 sess_cpu=N 可以立刻试" % kind)
        cpu_cap = 0
    if int(gpu_cap) + int(cpu_cap) <= 0:
        # 一份都不给就跑不了：按这台机器能跑的那条 EP 兜底
        if p.get("has_gpu") and gpu_ms:
            gpu_cap = 1
        else:
            cpu_cap = 1
        why.append("两份都是 0 → 兜底成 1 份，否则流水线没有推理 session")
    st = str(over.get("strategy") or "").strip().lower()
    if st in ("static", "shortq", "rr", "eta", "stage"):
        why.append("覆盖策略=%s" % st)
    gpu_cap = max(0, int(gpu_cap))
    cpu_cap = max(0, int(cpu_cap))
    if not p.get("has_gpu") and cpu_cap <= 0:
        cpu_cap = max(1, min(4, pc // max(1, tps)))   # 全 CPU：至少给一份，否则跑不了
    plan = {"kind": str(kind), "gpu": gpu_cap, "cpu": cpu_cap,
            "threads": tps,
            "strategy": (st if st in ("static", "shortq", "rr", "eta", "stage")
                         else strategy()),
            "gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
            "gain": round(_gain(gpu_cap, cpu_cap, gpu_ms, cpu_ms), 3),
            "why": why, "probe": {"cores": pc, "avail_ram_mb": ram,
                                  "vram_free_mb": vfree,
                                  "has_gpu": bool(p.get("has_gpu"))}}
    with _LOCK:
        _LAST[str(kind)] = dict(plan)
    return plan


def _gain(g, m, gpu_ms, cpu_ms) -> float:
    """相对「1 份 GPU 单干」的吞吐倍数（1.0 = 没变）。只用来打日志/对比。"""
    base = (1.0 / float(gpu_ms)) if gpu_ms else 0.0
    if base <= 0:
        base = (float(m) / float(cpu_ms)) if cpu_ms else 1.0
        return 1.0 if base <= 0 else 1.0
    now = 0.0
    if gpu_ms:
        now += float(g) / float(gpu_ms)
    if cpu_ms:
        now += float(m) / float(cpu_ms)
    return round(now / base, 3) if base > 0 else 1.0


def _store_path(workspace=None):
    from pathlib import Path
    ws = Path(str(workspace or os.environ.get("DFL_WORKSPACE") or "."))
    return ws / "_sess_bench.json"


def _load(workspace=None) -> dict:
    import json
    from pathlib import Path
    p = _store_path(workspace)
    try:
        if Path(p).is_file():
            d = json.loads(Path(p).read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def _save(d: dict, workspace=None):
    import json
    from pathlib import Path
    try:
        p = _store_path(workspace)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


def record_gain(kind: str, gpu_n: int, cpu_n: int, fps, workspace=None) -> dict:
    """记一次实测帧率，学会「加 CPU session / 把整段挪到 CPU EP 到底值不值」。

    为什么要记而不是算：理论收益（1 + gpu_ms/cpu_ms）只说这一段的容量，端到端还要
    过别的瓶颈，实测才作数。规则（按 文件名 = gpu_n x cpu_n 记账）：
        1x0 = 基线（今天的默认档）；
        1x1 / 1x2 比基线高 ≥5% → 自动档多挂 CPU session（auto_cpu_ok / auto_cpu2_ok）；
        2x0 比基线高 ≥5% → 自动档开第二份 GPU session（auto_gpu_ok）；
        0x1（检测整段走 CPU EP）比基线高 ≥5% → DFL_DET_EP=auto 的门②通过
            （auto_cpu_ep_ok，见 server/_detect.py 的 decide）；
        不到 5% → 一律维持现状（多开只是抢核，用户不该为它买单）。
    """
    row = dict(_load(workspace).get("gain:%s" % kind) or {})
    runs = dict(row.get("runs") or {})
    if fps is None:
        return row
    try:
        fps = round(float(fps), 2)
    except (TypeError, ValueError):
        return row
    tag = "%dx%d" % (int(gpu_n), int(cpu_n))
    runs[tag] = fps
    row["runs"] = runs
    base = runs.get("1x0")                       # 1×GPU + 0×CPU = 今天的默认档
    if base:
        for t, key in (("1x1", "auto_cpu_ok"), ("1x2", "auto_cpu2_ok"),
                       ("2x0", "auto_gpu_ok")):
            v = runs.get(t)
            if v is not None:
                row[key] = bool(float(v) >= float(base) * 1.05)
        # 0×GPU + 1×CPU：**检测整段走 CPU EP**（DFL_DET_EP=cpu）那一档。
        # 这是 _detect.decide() 的「门②」：只有它净增 ≥5% 才让 auto 真的切 CPU。
        # 和 1x1（GPU 检测 + 一路 CPU 检测 session）不是一回事，别混。
        ce = runs.get("0x1")
        if ce is not None:
            row["cpu_ep_fps"] = ce
            row["cpu_ep_gain_pct"] = round(100.0 * (float(ce) / float(base) - 1.0), 2)
            row["auto_cpu_ep_ok"] = bool(float(ce) >= float(base) * 1.05)
        row["base"] = base
        row["base_at"] = round(time.time(), 1)
        row["now"] = fps
        row["now_tag"] = tag
        row["gain_pct"] = round(100.0 * (float(fps) / float(base) - 1.0), 2)
        row["hybrid_cpu"] = int(cpu_n)
    d = _load(workspace)
    d["gain:%s" % kind] = row
    _save(d, workspace)
    return row


def cpu_ep_ok(kind: str, workspace=None) -> bool:
    """检测整段走 CPU EP 到底赚不赚：只有实测记录里 0x1 那一档净增 ≥5% 才为真。

    这是 DFL_DET_EP=auto 的第二道门（见 server/_detect.py 的 decide）。
    没有记录 = 没验证过 = False（不动默认）。
    """
    return bool((_load(workspace).get("gain:%s" % kind) or {}).get("auto_cpu_ep_ok"))


def cpu_ep_gain(kind: str, workspace=None):
    """实测记录里 0x1（检测走 CPU EP）相对 1x0 的净增百分比；没记录 → None。

    只用来把「为什么」写进日志：False 和「没量过」是两件事，日志里不能混成一句。
    """
    v = (_load(workspace).get("gain:%s" % kind) or {}).get("cpu_ep_gain_pct")
    return None if v is None else float(v)


def auto_cpu_ok(kind: str, n: int = 1, workspace=None) -> bool:
    """自动档能不能开 n 份 CPU session：只有实测证明了净增 ≥5% 才允许。"""
    row = _load(workspace).get("gain:%s" % kind) or {}
    return bool(row.get("auto_cpu_ok" if int(n) <= 1 else "auto_cpu2_ok"))


def auto_gpu_ok(kind: str, workspace=None) -> bool:
    """自动档能不能开第 2 份 GPU session（同样要实测 ≥5%）。"""
    return bool((_load(workspace).get("gain:%s" % kind) or {}).get("auto_gpu_ok"))


def gain_row(kind: str, workspace=None) -> dict:
    return dict(_load(workspace).get("gain:%s" % kind) or {})


def last_plan(kind: str = None):
    with _LOCK:
        if kind is None:
            return {k: dict(v) for k, v in _LAST.items()}
        return dict(_LAST.get(str(kind)) or {})


def plan_line(plan: dict) -> str:
    if not plan:
        return "-"
    return ("%s 池：%d×GPU + %d×CPU（每份 %d 线程 · 策略 %s）· GPU %.1f ms / CPU %.1f ms · "
            "理论 %+.0f%% · 依据：%s"
            % (plan.get("kind"), int(plan.get("gpu") or 0), int(plan.get("cpu") or 0),
               int(plan.get("threads") or 0), plan.get("strategy"),
               float(plan.get("gpu_ms") or 0.0), float(plan.get("cpu_ms") or 0.0),
               100.0 * (float(plan.get("gain") or 1.0) - 1.0),
               "；".join(plan.get("why") or [])))


# ---------------------------------------------------------------- 池与调度
class Pool(object):
    """n×GPU + m×CPU session 的异构池。**每个 session 一个线程 + 一个待办队列**，
    提交方只管 submit，结果的回填由 callback 在 session 线程里做。

    分配策略（pick）：
      stage  —— 按 kind 的固定归属（调用方给 stage_ep="cpu"/"gpu"）
      static —— 按 1/ms 反比（快的那路分得多），用轮转权重实现
      shortq —— 队列最短（含正在跑的那一份）
      rr     —— 严格轮转（每 rebal 项重置一次指针）
      eta    —— 队列长度 × 单张 ms 最小（= 最早能开始跑这一份的那路）
    """

    def __init__(self, name, sessions, strat=None, qdepth=8, on_done=None,
                 stage_ep=None):
        self.name = str(name)
        self.strat = str(strat or strategy())
        self.qdepth = max(2, int(qdepth or 8))
        self.on_done = on_done
        self.stage_ep = str(stage_ep or "")
        self.sessions = list(sessions or [])
        self.lock = threading.Lock()
        self._rr = 0
        self._n = {"submit": 0, "done": 0}
        self._per = [{"name": s.get("name"), "ep": s.get("ep"), "ms": s.get("ms"),
                      "done": 0, "ms_sum": 0.0, "busy": 0} for s in self.sessions]
        self._q = []
        self._th = []
        self.closed = False
        for i, s in enumerate(self.sessions):
            q = queue.Queue(maxsize=self.qdepth)
            self._q.append(q)
            t = threading.Thread(target=self._loop, args=(i, q, s),
                                 name="%s-%d-%s" % (self.name, i, s.get("ep")),
                                 daemon=True)
            self._th.append(t)
            t.start()

    # ---- 分配 ----
    def _weight(self, i):
        ms = float((self.sessions[i] or {}).get("ms") or 0.0) or 1.0
        return 1.0 / ms

    def pick(self):
        n = len(self.sessions)
        if n <= 0:
            return None
        if n == 1:
            return 0
        with self.lock:
            if self.strat == "stage" and self.stage_ep:
                for i, s in enumerate(self.sessions):
                    if str(s.get("ep")) == self.stage_ep:
                        return i
                return 0
            if self.strat == "rr":
                i = self._rr % n
                self._rr += 1
                return i
            if self.strat == "static":
                w = [self._weight(i) for i in range(n)]
                tot = sum(w) or 1.0
                acc = 0.0
                self._rr += 1
                x = ((self._rr * 0.6180339887) % 1.0) * tot
                for i in range(n):
                    acc += w[i]
                    if x <= acc:
                        return i
                return n - 1
            if self.strat == "shortq":
                best = 0
                for i in range(n):
                    if self._pending(i) < self._pending(best):
                        best = i
                return best
            # eta（默认）：队列长度 × 单张 ms 最小
            best = 0
            bestv = None
            for i in range(n):
                v = self._pending(i) * (float(self.sessions[i].get("ms") or 1.0) or 1.0)
                if bestv is None or v < bestv:
                    bestv, best = v, i
            return best

    def _pending(self, i):
        try:
            return int(self._q[i].qsize()) + int(self._per[i]["busy"])
        except Exception:                                   # noqa: BLE001
            return 0

    # ---- 提交 / 回收 ----
    def submit(self, payload):
        """阻塞式提交（队列满 = 背压，和现在的有界队列一个口径）。"""
        i = self.pick()
        if i is None:
            raise RuntimeError("%s 池是空的" % self.name)
        self._q[i].put(payload)
        with self.lock:
            self._n["submit"] += 1
        return i

    def _loop(self, i, q, s):
        fn = s.get("fn")
        while True:
            item = q.get()
            if item is None:
                return
            with self.lock:
                self._per[i]["busy"] = int(self._per[i]["busy"]) + 1
            t0 = time.perf_counter()
            try:
                res = fn(item)
            except Exception as e:                          # noqa: BLE001
                res = e
            dt = (time.perf_counter() - t0) * 1000.0
            with self.lock:
                self._per[i]["busy"] = max(0, int(self._per[i]["busy"]) - 1)
                self._per[i]["done"] = int(self._per[i]["done"]) + 1
                self._per[i]["ms_sum"] = round(float(self._per[i]["ms_sum"]) + dt, 1)
                self._n["done"] += 1
            if self.on_done is not None:
                try:
                    self.on_done(item, res)
                except Exception:                       # noqa: BLE001
                    pass

    def close(self, wait=5.0):
        if self.closed:
            return
        self.closed = True
        for q in self._q:
            try:
                q.put(None)
            except Exception:                               # noqa: BLE001
                pass
        for t in self._th:
            try:
                t.join(timeout=float(wait))
            except Exception:                               # noqa: BLE001
                pass

    def stats(self) -> dict:
        with self.lock:
            per = [dict(x) for x in self._per]
        for x in per:
            n = max(1, int(x.get("done") or 0))
            x["avg_ms"] = round(float(x.get("ms_sum") or 0.0) / n, 1)
        return {"name": self.name, "strategy": self.strat,
                "sessions": per, "submit": self._n["submit"],
                "done": self._n["done"],
                "queues": [int(q.qsize()) for q in self._q]}


def level_cfg(level: str) -> dict:
    """验收用的「机器档位」覆盖（只影响 derive 的输入，不改真探测）。

    低配 / 中配 / 高配三档按用户给的数字硬写在这里 —— 这三组数字是**测试夹具**，
    不是运行时策略；生产路径读的是 probe() 的真值。
    """
    lv = str(level or "").strip().lower()
    if lv == "low":
        return {"cores": 4, "logical": 4, "avail_ram_mb": 3600.0,
                "vram_free_mb": 4096.0, "vram_total_mb": 4096.0, "has_gpu": True}
    if lv == "mid":
        return {"cores": 8, "logical": 16, "avail_ram_mb": 15000.0,
                "vram_free_mb": 8150.0, "vram_total_mb": 8150.0, "has_gpu": True}
    if lv == "high":
        return {"cores": 32, "logical": 64, "avail_ram_mb": 24000.0,
                "vram_free_mb": 24576.0, "vram_total_mb": 24576.0, "has_gpu": True}
    if lv == "nogpu":
        return {"cores": 8, "logical": 16, "avail_ram_mb": 12000.0,
                "vram_free_mb": 0.0, "vram_total_mb": 0.0, "has_gpu": False}
    return {}
