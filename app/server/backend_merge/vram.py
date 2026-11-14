# -*- coding: utf-8 -*-
"""显存预算 / 归属 / 回收，ONNX 会话与 EP 的账本，检测器排队与微基准。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from .._common.http import (
    _BadInput,
)
from ._shared import (
    _PRED_BATCH_CAP, _fmt_mb,
)


# ============================================================
# onnxruntime 的 CUDA provider：先把 cublasLt / cuDNN 的目录找出来
# ------------------------------------------------------------
# ORT 的 CUDAExecutionProvider 要 cublasLt64_12.dll（还有 cuDNN 9）。这两份都在
# torch/lib 里，但那个目录默认不在进程的 DLL 搜索路径上 —— 找不到时 ORT 只留一行
# warning（Error 126）然后**静默**退回 CPU。实测差 7 倍：同一个 416 的 DFM、同样
# 1920x1706 的帧，单张合成 CUDA 78ms / CPU 648ms（前向 36ms vs 524ms）。
# 所以 import 这个模块时就把路径补上：检测器、2D106、预览 worker 都是这之后才建
# session 的。补不上也不影响可用性 —— provider 列表里本来就带着 CPU 兜底。
# ============================================================
_ORT_DLL_DIRS = []
_ORT_DLL_COOKIES = []      # add_dll_directory 的句柄：不持有的话目录会被回收掉


def _ort_dll_candidates() -> list:
    """可能要补进 DLL 搜索路径的目录（只返回真的存在的，去重、保序）。"""
    import sys as _sys
    out, seen = [], set()

    def _push(*parts):
        if not parts or not parts[0]:
            return
        try:
            p = os.path.join(*[str(x) for x in parts if str(x)])
        except Exception:
            return
        try:
            if not p or not os.path.isdir(p):
                return
            k = os.path.normcase(p)
        except OSError:
            return
        if k not in seen:
            seen.add(k)
            out.append(p)

    prefix = str(getattr(_sys, "prefix", "") or "")
    # 1) 本进程自己装的 torch（服务器 python 是 cu128 版，cublasLt64_12.dll 就在里面）
    _push(prefix, "Lib", "site-packages", "torch", "lib")
    # 2) 4.0 引擎 python 的 torch（预览 worker 跑在那个解释器里）
    try:
        _push(Path(str(_engine_root())), "python", "Lib", "site-packages", "torch", "lib")
    except Exception:
        pass
    # 3) pip 的 nvidia-* 布局（有的话，版本和 ORT 更贴）
    _push(prefix, "Lib", "site-packages", "nvidia", "cublas", "bin")
    _push(prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin")
    return out


def _ensure_ort_dll_dirs() -> list:
    """把上面那些目录塞进 DLL 搜索路径；返回真加进去的目录（幂等，只做一次）。"""
    global _ORT_DLL_DIRS
    if _ORT_DLL_DIRS:
        return list(_ORT_DLL_DIRS)
    # 仓库里已经有一份现成的（_ortcuda 会挑带 cudnn64_9.dll 的那个 torch/lib）
    try:
        from .._ortcuda import add_cuda_dll_dirs as _add
        _add()
    except Exception:
        pass
    added = []
    for d in _ort_dll_candidates():
        try:
            if hasattr(os, "add_dll_directory"):
                _ORT_DLL_COOKIES.append(os.add_dll_directory(d))   # 句柄要留着
        except Exception:
            pass
        added.append(d)
    if added:
        try:
            os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
        except Exception:
            pass
    _ORT_DLL_DIRS = list(added)
    if added:
        print("[merge-preview] ORT DLL 目录：%s" % "；".join(added), flush=True)
    return list(_ORT_DLL_DIRS)


try:                                    # 别为了补 DLL 把整个路由 import 弄挂
    _ensure_ort_dll_dirs()
except Exception:                       # pragma: no cover
    _ORT_DLL_DIRS = []


def _sess_batch_cap(sess) -> int:
    """这个 session 一次前向最多能塞几张（1 = 只能逐张）。"""
    try:
        mb = getattr(sess, "max_batch", None)
        if callable(mb):
            return max(1, int(mb() or 1))            # 原生 TRT：按优化档位
    except Exception:                                   # noqa: BLE001
        return 1
    try:
        shape = list(sess.get_inputs()[0].shape)
        if shape and isinstance(shape[0], int) and shape[0] > 0:
            return int(shape[0])
    except Exception:                                   # noqa: BLE001
        pass
    return 8                                            # 动态 batch（ort）


# ---------------- ③ 换脸（dfm 推理）：batch / 显存 / 流水线 ----------------
def _vram_free_torch():
    """torch 看到的 (free, total) MB —— 和推理同一次驱动查询，但**偏乐观**（见下）。"""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return int(free // (1 << 20)), int(total // (1 << 20))
    except Exception:                                       # noqa: BLE001
        pass
    return None, None


_SMI_MEM = {"at": 0.0, "free": None, "total": None, "used": None}


def _vram_free_smi(fresh: bool = False):
    """nvidia-smi 的 (free, total, used) MB。

    '--query-gpu' 一行一卡，**只取第 0 行**：torch 的 'cuda:0' 就是 nvidia-smi 的
    GPU 0（这台机器上另一块是 AMD，nvidia-smi 根本看不见它，所以不存在「把 AMD 的
    显存算进来」这回事）。一次约 80ms，2 秒小缓存 —— 预览每拖一帧都要问。
    """
    now = time.time()
    if not fresh and _SMI_MEM["at"] and (now - _SMI_MEM["at"]) < 2.0:
        return _SMI_MEM["free"], _SMI_MEM["total"], _SMI_MEM["used"]
    free = total = used = None
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=memory.free,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10,
                             creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        row = [x for x in (out.stdout or "").strip().splitlines() if x.strip()]
        f, t, u = [int(float(x)) for x in row[0].split(",")[:3]]
        free, total, used = f, t, u
    except Exception:                                       # noqa: BLE001
        pass
    if free is not None:
        _SMI_MEM.update({"at": now, "free": free, "total": total, "used": used})
    return free, total, used


def _vram_free_mb(fresh: bool = False):
    """(free, total) MB —— torch 与 nvidia-smi 取**更小的那个空闲**。

    **为什么两个都问**（这一条是量的，不是拍的；本机 RTX 5070 Laptop 8GB / WDDM）：
    空载时 torch.cuda.mem_get_info 说空闲 6999 MB，同一刻 nvidia-smi 说 5693 MB。
    差的 1.3GB 不是谁算错了 —— WDDM 下 mem_get_info 报的是「驱动还愿意答应多少」
    （不够就换页/回落共享内存），真分配下去立刻变成 nvidia-smi 的 used。**拿它当
    预算就是乐观到 OOM**；反过来 nvidia-smi 的 free 是「这张卡上所有进程之外还剩
    多少」，那才是敢用的数。所以两个都探、取小。

    两个都探不到的机器（没有 torch、也没有 nvidia-smi）返回 (None, None) ——
    调用方按「不猜」处理（不降档、不拒绝）。
    """
    tf, tt = _vram_free_torch()
    sf, st, _su = _vram_free_smi(fresh=fresh)
    cands = [(v, t) for v, t in ((tf, tt), (sf, st)) if v is not None]
    if not cands:
        return None, None
    free = min(v for v, _t in cands)
    total = next((t for _v, t in cands if t), None)
    return int(free), (None if total is None else int(total))


# ============================================================
# 显存预算（三条路共用）：模型体量 → (计划 batch, 帧预算, 预计峰值, 可行?)
# ============================================================
# 为什么要有这一块：以前只有 _fused_budget（只按 30/16/8 降帧）和 _pred_pick_batch
# （OOM 之后才降 batch）—— 两条都是「先跑，崩了再退」，**没有把模型体量算进总需求**，
# 也没在「连 batch=1 都装不下」时停下来。结果是 3GB 空闲的机器上点导出会一路 OOM 到
# 崩进程（fused 那条尤其难看：线程全炸、日志只剩 traceback）。
#
# 现在的口径是「按预算跑，跑不了就明说」：
#   ① 探测可用显存（torch → nvidia-smi；DFL_VRAM_FREE_MB 可伪造，给验收/排障用）；
#   ② 预算 = 空闲 × (1 - 20%) - 512MB（20% 是驱动/碎片余量，512MB 给桌面合成器）；
#   ③ 需求 = 会话（权重 + ORT arena）+ CUDA context + 检测/遮罩 + 帧缓冲 + dfm 批；
#   ④ 沿阶往下试，第一个装得下的就用（降档要记账）；连最小档都装不下：
#        · 默认 → 明确拒绝（HTTP 400，消息里给「需要 X GB / 可用 Y GB / 建议」）；
#        · DFL_ALLOW_CPU_FALLBACK=1 → 回落 CPU EP 照跑（慢 ~17 倍），日志和 status
#          都写「当前在 CPU 上跑」。
#
# 这里的常数都是**量的**，不是拍的（RTX 5070 Laptop 8GB + 416 dfm + ort-CUDA）：
#   会话建好（含这个进程的 CUDA context）          1161 MB   ← 权重 514 + context 512 + arena 135
#   batch 1 / 2 / 4 / 8 / 16 的整卡增量             1173 / 1240 / 2199 / 2256 / 6274 MB
#     → 减去会话本身：+12 / +79 / +1036 / +1091 / +5117（batch 4 起 ORT 的 arena
#       换到另一个池子，所以是台阶不是直线；16 会把 8GB 卡顶到墙 —— 与 _pred_pick_batch
#       里「16 撑爆 arena 之后整个进程变慢」的实测一致）；
#   384 模型（文件 1567MB）会话 ≈ 2.07GB（用户口径 + 本仓库实测）→ 权重 + context 就够，
#   arena 是小头，所以 _VRAM_ARENA_MB 取 140（416 实测 135，上取整）。
_VRAM_ENV_FREE = "DFL_VRAM_FREE_MB"        # 验收/排障：假装可用显存只有这么多（三条路共用）
_VRAM_ENV_CPU = "DFL_ALLOW_CPU_FALLBACK"   # 1 = 显存不够时回落 CPU EP（默认关：宁可拒绝）
# ---- 预留（2026-05 改小：老的「空闲 20% + 512MB」没有任何实测依据，在 8GB 卡上
#      等于给用户凭空造了 1.37GB 的死重 —— 见 _vram_reserve_mb 的注释）----
_VRAM_SAFETY = 0.05                        # 预留 = 空闲的 5%（随空闲缩放）
_VRAM_RESERVE_MB = 256                     # 底线：探测抖动（torch 与 smi 差 ≤256MB 的容差）
_VRAM_RESERVE_MAX_MB = 768                 # 上限：大卡上不许把预留变成 1GB+
_VRAM_CTX_MB = 512                         # 每进程一份 CUDA context
_VRAM_ARENA_MB = 140                       # ORT CUDA arena：会话建好后比权重多出来的那点
_VRAM_AUX_MB = 300                         # 检测器 + 2D106 关键点 + XSeg（用户口径 0.3GB）
_VRAM_CPU_SLOW = 17.0                       # 回落 CPU 的代价（416 实测 527ms vs 31ms/张）
_VRAM_BASE_PX = 416                         # 下面两张表都是按 416 量的，其它尺寸按 (s/416)² 缩放
# dfm 批的成本**按模型各量一张表**（用户口径：同一个 batch，大网络和小网络多占的不是
# 一个数）。点集 = ((batch, 相对同一份会话 batch=1 时多占的 MB), ...)，台阶不是直线（ORT 的
# arena 在 batch 4 换池子），所以查表 + 分段线性。
# 实测（本机 RTX 5070 Laptop 8GB · 1080x1920 真项目 120 帧 · 1 个 worker · 整卡 nvidia-smi
# 峰值，空载基线 450MB；见 %TEMP%/dfl_projplan/bench_steps.json）：
#   384（权重 1567MB · 输入 384）：b1 3184 / b2 4858（+1674）/ b4 6160（+2976）/ b8 7736（+4552）
#   416（权重  514MB · 输入 416）：b1 1990 / b2 3469（+1479）/ b4 4040（+2050）/ b8 7322（+5332）
# 注意两件事：
#   · 增量主要由**帧尺寸**（合成那一路的整帧张量）决定，不是由 dfm 大小 —— 所以两张表
#     差不多，但 384 那份在 batch 4 明显更贵（+2976 vs +2050）；
#   · batch 8 两张表都超过 +4500MB，这时候已经在换页（实测 swap p50 1906~2266ms）。
_VRAM_BATCH_PTS_416 = ((1, 0.0), (2, 1479.0), (4, 2050.0), (8, 5332.0))
_VRAM_BATCH_PTS_384 = ((1, 0.0), (2, 1674.0), (4, 2976.0), (8, 4552.0))
# (权重上限 MB, 量的那个输入边长, 点集)：权重 ≤900MB 用 416 那张，更大的用 384 那张
_VRAM_BATCH_TABLES = ((900.0, 416, _VRAM_BATCH_PTS_416),
                      (1e9, 384, _VRAM_BATCH_PTS_384))
# fused（帧预算那条路）的 dfm 批台阶是**另一个量**：那条路一次只让 dfm 前向 k 张，
# 不背「整帧合成 + 落盘」的 k 份缓冲，所以它比项目导出便宜得多。老表就是在那条路上量的
# （416 实测：+80/+160/+1100/+1150/+5200），保留给 frames / none 两条路；项目导出用上面
# 那两张按模型各量的表。混用会把 fused 的降档判据整体抬高一倍（实测：30 帧/批 1 从
# 1644MB 变成 3232MB，fused_export_accept 的 C 段当场从「降到 16 帧」变成「照跑 30 帧」）。
_VRAM_BATCH_PTS = ((1, 80.0), (2, 160.0), (4, 1100.0), (8, 1150.0), (16, 5200.0))
# 项目导出 worker 的**运行时工作集**：会话建好、再真跑起来之后 ORT/CuDNN 还会多占一块，
# 而这一块**不在**「权重 + context」里。实测（本机 8GB，1080x1920 真项目 120 帧，
# 1 个 worker、整卡 nvidia-smi 峰值、空载基线 450MB，见 %TEMP%/dfl_projplan/bench_steps.json）：
#   · 384（权重 1567MB · 输入 384）batch=1 整卡 3184MB → 一份会话
#       = 3184 − 450(桌面) − 300(检测/遮罩) − 0(batch 1) = **2434MB**
#       = 1567 权重 + 512 context + **355 运行时**；
#   · 416（权重  514MB · 输入 416）batch=1 整卡 1990MB → 一份会话 = **1240MB**
#       = 514 + 512 + **214 运行时**。
# 取 320MB（两次实测的中间偏上）。**batch 的成本不在这里**：它按模型各有一张表
# （_VRAM_BATCH_TABLES），因为 384 那份 batch 4 要多 2976MB、416 只要 2050MB。
# 最早那版把这个常数写成 1300，是把 dfm 批那一项重复算进了「会话」里，已修正。
_VRAM_SESS_RUNTIME_MB = 320
_VRAM_PROBE_TTL = 4.0                       # 预览每拖一次都要问：nvidia-smi 80ms 不能每次都跑
_VRAM_PROBE = {"at": 0.0, "free": None, "total": None, "src": ""}
_VRAM_MODEL_CACHE = {}
_VRAM_SIZE_CACHE = {}
_VRAM_LAST = {}                             # label -> 最近一次的计划（status 端点要能回显）
_VRAM_LOGGED = {}                           # label -> 上一次打过的签名（同样的决定只打一次）
_VRAM_LOCK = threading.RLock()
_CPU_EP = {"on": False, "why": ""}          # 全局「强制 CPU EP」：_dfm_session / _stage_xseg 读它
_CPU_EP_LOCK = threading.Lock()


# ============================================================
# 显存归属：这块卡现在被谁占着（本程序 / 其它程序 / 空闲）
# ------------------------------------------------------------
# 用户那一版截图最糟的地方不是被拒，而是**被拒了不知道是谁占的**：界面写「空闲
# 1387MB」，而他自己什么都没跑 —— 因为占着的就是我们（预览 worker 里常驻的 dfm
# 会话、本进程里加载的 dfm / 检测器，以及 ORT 那个会长大的 CUDA arena）。
#
# 这里只记**我们自己**的账，而且尽量是**实测**：
#   · 本进程：每个 session 加载前后各探一次空闲，差值就是它真的吃掉的那一块
#     （含 CUDA context / ORT arena —— 估不出来，只能量）；
#   · 预览 worker：每次 call 前后探一次，差值记在 worker 上（它可能是 1 个也可能是
#     3 个模型，还带着自己的 arena）；
#   · 剩下的 = 总显存 - 空闲 - 本程序 → 归「其它程序」（浏览器/游戏/另一个导出）。
# ============================================================
_VRAM_OWN = {}          # tag -> MB（本进程自己加载的，实测增量）
_VRAM_OWN_LOCK = threading.RLock()
_VRAM_OWN_PENDING = {}  # tag -> 起始空闲（量到一半的，防并发重复计）


def _vram_own_note(tag: str, mb: float):
    """记一笔「本进程里这个东西吃掉了多少显存」（MB，实测增量）。"""
    tag = str(tag or "")
    mb = max(0.0, float(mb or 0.0))
    if not tag or mb <= 0:
        return
    with _VRAM_OWN_LOCK:
        _VRAM_OWN[tag] = round(max(mb, float(_VRAM_OWN.get(tag) or 0.0)), 1)


def _vram_own_drop(tag: str):
    """这个东西被放掉了（session 关了/切了 CPU）—— 从本进程的账上划掉。"""
    with _VRAM_OWN_LOCK:
        _VRAM_OWN.pop(str(tag or ""), None)


def _vram_own_detail() -> dict:
    with _VRAM_OWN_LOCK:
        return {str(k): round(float(v), 1) for k, v in sorted(_VRAM_OWN.items())}


def _vram_own_inproc_mb() -> float:
    """本进程自己占着多少（MB）。"""
    with _VRAM_OWN_LOCK:
        return round(float(sum(_VRAM_OWN.values())), 1)


def _vram_measure_begin():
    """开始量：返回起始空闲（MB，探不到就是 None）。"""
    f0, _t = _vram_free_mb(fresh=True)
    return f0


def _vram_measure_end(tag: str, f0):
    """量完：把「空闲掉了多少」记到 tag 上（探不到就不记，绝不用 0 冒充）。"""
    if f0 is None:
        return None
    f1, _t = _vram_free_mb(fresh=True)
    if f1 is None:
        return None
    used = max(0.0, float(f0) - float(f1))
    _vram_own_note(tag, used)
    return round(used, 1)


def _vram_preview_info() -> dict:
    """常驻预览 worker 现在占着多少、都常驻了什么、能不能回收。

    优先用**实测**（worker 每次调用前后探到的空闲差，见 _PreviewWorker.call）；
    没量过才退回会话口径估（模型体量 + 一份 CUDA context + ORT arena）。实测优先
    是因为 ORT 的 CUDA arena 会长：384 模型实测「建会话 +2.3GB」，之后每一批 8 张
    的前向还会再涨 ~1GB（本机 3 次之后稳定在 6.4GB/8151MB）—— 估出来的数会偏小。
    """
    try:
        w = _WORKER
        alive = bool(w.alive())
        models = list((w._models or {}).keys()) if alive else []
        measured = float(getattr(w, "_used_mb", 0.0) or 0.0)
    except Exception:                                       # noqa: BLE001
        alive, models, measured = False, [], 0.0
    est = 0.0
    if alive and models:
        est = (float(_vram_models_mb(models)) + float(_VRAM_CTX_MB)
               + float(_VRAM_ARENA_MB) * max(1, len(models)))
    # **没有常驻模型 = 那块显存已经还回去了**（unload / LRU 淘汰都走 worker 里的
    # _vram_flush：掐引用 → gc → empty_cache）。这时 `_used_mb` 是上一次加载的旧账，
    # 不能再算进"本程序占用"—— 否则账上永远挂着 1.6GB，谁会去回收一个不存在的东西。
    mb = max(measured, est) if models else 0.0
    return {"alive": bool(alive), "models": [os.path.basename(str(p)) for p in models],
            "paths": [str(p) for p in models], "measured_mb": round(measured, 1),
            "estimated_mb": round(est, 1), "mb": round(mb, 1),
            "measured": bool(measured > 0)}


def _vram_owners(free_mb=None, total_mb=None, source: str = "") -> dict:
    """谁占着显存：本程序（分项 + 可回收多少）/ 其它程序 / 空闲。

    拒绝消息、status、日志都读这一份，所以「空闲 1387MB」不再是一个孤零零的数字 ——
    用户能看见「其中本程序 2.0GB 是可回收的预览模型，其它程序 3.4GB」。
    """
    if free_mb is None or total_mb is None:
        pfree, ptotal, _p = _vram_probe()
        free_mb = pfree if free_mb is None else free_mb
        total_mb = ptotal if total_mb is None else total_mb
    pv = _vram_preview_info()
    inproc = _vram_own_inproc_mb()
    own = float(inproc) + float(pv.get("mb") or 0.0)
    other = None
    src = str(source or "")
    real = not (src.startswith("env:") or src.startswith("file:") or src == "override")
    if free_mb is not None and total_mb is not None and real:
        # 只有真探到的 free 才能反推「其它程序」；env/file/override 是伪造值，
        # 拿它减出来的"其它程序"是假账（宁可显示 ?，也不给用户一个错数）。
        other = round(max(0.0, float(total_mb) - float(free_mb) - own), 1)
    torch_res = None
    try:
        import torch
        if torch.cuda.is_available():
            torch_res = round(torch.cuda.memory_reserved() / 1048576.0, 1)
    except Exception:                                       # noqa: BLE001
        pass
    return {"free_mb": (None if free_mb is None else int(free_mb)),
            "total_mb": (None if total_mb is None else int(total_mb)),
            "own_mb": round(own, 1), "own_inproc_mb": round(float(inproc), 1),
            "own_preview_mb": round(float(pv.get("mb") or 0.0), 1),
            "own_parts": _vram_own_detail(),
            "preview_models": list(pv.get("models") or []),
            "preview_measured": bool(pv.get("measured")),
            "reclaimable_mb": round(float(pv.get("mb") or 0.0), 1),
            "other_mb": other, "torch_reserved_mb": torch_res,
            "free_torch_mb": _vram_free_torch()[0],
            "free_smi_mb": _vram_free_smi()[0]}


def _vram_reclaim(why: str = "导出") -> dict:
    """把**我们自己**占的显存收回来：请常驻预览 worker 卸掉它的 DFM 会话。

    为什么导出前必须做一遍：用户先预览过一帧（worker 里 1.6GB 的 dfm + CUDA context
    常驻），再点导出 —— 计划看到的是「空闲 1.4GB」，于是把一单本来完全跑得动的活拒
    了，而占着的正是我们自己的子进程。这里先让它放掉、等驱动真回收（轮询空闲涨上来，
    最多 4s），再拿新的空闲去算预算。

    导出结束后**不重载**：下次预览再懒加载（实测一次重载 ~4.1s，比来回抖一晚上强）。
    """
    t0 = time.time()
    try:
        w = _WORKER
        alive = bool(w.alive())
        models = list((w._models or {}).keys()) if alive else []
        used = float(getattr(w, "_used_mb", 0.0) or 0.0)
    except Exception:                                       # noqa: BLE001
        alive, models, used = False, [], 0.0
    if not alive or not models:
        return {"ok": True, "did": False, "freed_mb": 0.0, "ms": 0.0,
                "models": [], "before_mb": None, "after_mb": None,
                "why": "预览 worker 里没有常驻模型，不用回收"}
    before, _t = _vram_free_mb(fresh=True)
    want = max(float(used),
               float(_vram_models_mb(models)) + float(_VRAM_CTX_MB)
               + float(_VRAM_ARENA_MB) * max(1, len(models)))
    names = [os.path.basename(str(p)) for p in models]
    reply = {}
    try:
        reply = dict(w.unload("") or {})
    except Exception as e:                                  # noqa: BLE001
        print("[vram] 回收预览模型失败（%s）：unload 抛异常：%s"
              % (why, str(e)[:200]), flush=True)
        return {"ok": False, "did": False, "freed_mb": 0.0,
                "ms": round((time.time() - t0) * 1000.0, 1), "models": names,
                "before_mb": before, "after_mb": before,
                "why": "回收失败（unload 抛异常）：%s" % str(e)[:160]}
    # ★ 「回应说没放掉」同样算失败。worker 的 unload 是在**子进程**里执行的，它自己
    #   失败时（模型被那一帧占着 / _vram_flush 抛错 / 不认这个命令）不会抛异常，只回
    #   ok=false 或**原样带着 models 清单**回来。以前这里只 catch 异常 —— 于是失败被
    #   记成「成功、腾出 0MB」，日志上只剩一个 0，原因永远看不见（用户口径：失败要写
    #   清原因，不能只打 0）。
    try:
        left = [str(p) for p in (w._models or {}).keys()]
    except Exception:                                       # noqa: BLE001
        left = []
    if left or reply.get("ok") is False:
        _r_why = str(reply.get("error") or reply.get("why") or reply.get("msg")
                     or reply.get("stage") or "").strip()
        if not _r_why:
            _r_why = ("worker 回了 ok=false" if not left else
                      "worker 说放掉了，但 %d 份模型还在：%s"
                      % (len(left), "/".join(os.path.basename(p) for p in left[:3])))
        print("[vram] 回收预览模型失败（%s）：%s（worker 里仍常驻 %d 份：%s）"
              % (why, _r_why[:200], len(left),
                 "/".join(os.path.basename(p) for p in left[:3]) or "-"), flush=True)
        return {"ok": False, "did": False, "freed_mb": 0.0,
                "ms": round((time.time() - t0) * 1000.0, 1), "models": names,
                "left_models": [os.path.basename(p) for p in left],
                "before_mb": before, "after_mb": before,
                "why": "回收失败：%s" % _r_why[:160]}
    after = before
    deadline = time.time() + 4.0
    while time.time() < deadline:
        after, _t = _vram_free_mb(fresh=True)
        if before is None or after is None:
            break
        if (float(after) - float(before)) >= 0.85 * max(1.0, want):
            break
        time.sleep(0.15)
    freed = (0.0 if (before is None or after is None)
             else max(0.0, float(after) - float(before)))
    # ---- 卸了模型、显存却没回来？把 worker **进程**整个回收掉 ----
    # WDDM 上释放不等于还给驱动：本机实测预览 worker 卸掉一份 384 模型之后，进程仍握着
    # 2347MB（Windows「GPU Process Memory · Dedicated」实测），而 nvidia-smi 的 free 只
    # 涨了 135MB。这块残留就是用户那句「我什么都没跑，却被自己的后端堵死」的另一半 ——
    # 只要进程不退，那 2.3GB 就一直在账上，谁也算不出「其实是自己」。
    # 只在「卸完几乎没腾出东西」时动手（正常情况这里是 2GB+，一行都不用走），
    # 代价是下次预览要重开进程（实测 boot 1.6s + 冷加载 3.5~4.2s），比被拒强得多。
    _rc_switch = str(os.environ.get("DFL_VRAM_RECYCLE_WORKER") or "").strip().lower()
    recycled = False
    if _rc_switch not in ("0", "false", "no", "off") and freed < 0.5 * max(1.0, want):
        print("[vram] 卸掉预览模型只腾出 %.0f MB（应约 %.0f MB）—— 进程仍握着这些显存，"
              "直接回收 worker（下次预览懒加载）" % (freed, want), flush=True)
        try:
            recycled = bool(w.close("回收显存：卸模型后只腾出 %.0f MB" % freed))
        except Exception as e:                              # noqa: BLE001
            print("[vram] 回收预览 worker 失败：%s" % str(e)[:160], flush=True)
        if recycled:
            deadline = time.time() + 8.0
            while time.time() < deadline:
                a2, _t = _vram_free_mb(fresh=True)
                if a2 is not None:
                    after = a2
                if (before is None or after is None
                        or (float(after) - float(before)) >= 0.85 * max(1.0, want)):
                    break
                time.sleep(0.2)
            freed = (0.0 if (before is None or after is None)
                     else max(0.0, float(after) - float(before)))
    try:
        w._used_mb = 0.0            # 账已经清了：下一次 call 重新量
    except Exception:                                       # noqa: BLE001
        pass
    ms = (time.time() - t0) * 1000.0
    # 卸掉了模型、空闲却没涨够（WDDM 上「释放」不等于还给驱动）：这同样是「回收没成」，
    # 必须有个原因挂在 why 上 —— 否则预算里那一栏是 0，日志上也只有一个 0。
    # 口径（按「腾出来多少」分两档，别把 84% 的火气也写成失败）：
    #   < 50%  → 真失败（ok=False，why 进日志/summary/拒绝消息）
    #   50~85% → 只是比「模型体量 + context」估的少（note，不判失败）
    _short = ""
    _note = ""
    if before is None or after is None:
        _short = "显存探不到（torch / nvidia-smi 都没答），只能按模型体量估 %.0f MB" % want
    elif freed < 0.5 * max(1.0, want):
        _short = ("卸了模型但空闲只涨了 %.0f MB（应约 %.0f MB）%s"
                  % (freed, want,
                     "；worker 进程已回收" if recycled else
                     "；进程没回收（DFL_VRAM_RECYCLE_WORKER=0 或回收失败）"))
    elif freed < 0.85 * max(1.0, want):
        _note = ("比按模型体量估的 %.0f MB 少（进程%s）"
                 % (want, "已回收" if recycled else "仍常驻，WDDM 上释放不全正常"))
    print("[vram] 回收预览模型 %.0f MB（%s · %s · %.0fms%s）：空闲 %s → %s MB%s%s"
          % (freed, why, "/".join(names[:3]) or "-", ms,
             " · 进程已回收" if recycled else "",
             "?" if before is None else int(before),
             "?" if after is None else int(after),
             (" · " + _short) if _short else "",
             (" · " + _note) if _note else ""), flush=True)
    return {"ok": not bool(_short), "did": True, "freed_mb": round(freed, 1),
            "ms": round(ms, 1),
            "models": names, "before_mb": before, "after_mb": after,
            "want_mb": round(want, 1), "recycled": bool(recycled),
            "why": _short, "note": _note}


def _vram_src_real(src) -> bool:
    """这个 free 是不是**真探到的**。env / file / override 都是伪造或单测注入的值，
    拿它们再加「可回收」就是假账（一个文本文件就能把预算抬高 2.7GB）。"""
    s = str(src or "")
    return not (s.startswith("env:") or s.startswith("file:") or s in ("override", ""))


_VRAM_RECLAIM_MARK = {"at": 0.0, "rec": {}}


def _vram_reclaim_once(why: str = "导出", gap: float = 6.0) -> dict:
    """回收预览 worker 的模型，但**同一单里只真收一次**。

    为什么要节流：导出起手有两拍预算 —— 路由闸门（HTTP 400 那一拍）和 worker 线程
    里 _fused_run 的第一拍。两拍都开 reclaim_first，就会连着卸两次；第二拍其实什么
    都没得卸（worker 里已经空了），却会再跑 2~3 次 nvidia-smi（WDDM 上每次 ~80ms）。
    gap 内的重复调用直接返回上一次的结果（带 throttled=True 留痕）。
    """
    # 回滚 / A-B 对照开关：1 = 完全不卸预览模型（日志里会写明是它，不装成「没得卸」）
    if str(os.environ.get("DFL_VRAM_NO_RECLAIM") or "").strip().lower() in ("1", "true", "yes", "on"):
        return {"ok": False, "did": False, "freed_mb": 0.0, "ms": 0.0, "models": [],
                "before_mb": None, "after_mb": None,
                "why": "DFL_VRAM_NO_RECLAIM=1（回滚/对照用：这一轮不卸预览模型）"}
    now = time.time()
    with _VRAM_LOCK:
        at = float(_VRAM_RECLAIM_MARK.get("at") or 0.0)
        last = dict(_VRAM_RECLAIM_MARK.get("rec") or {})
    if at and (now - at) < float(gap):
        out = dict(last or {})
        out.update({"throttled": True, "age_ms": round((now - at) * 1000.0, 1)})
        return out
    try:
        rec = _vram_reclaim(why)
    except Exception as e:                                  # noqa: BLE001
        rec = {"ok": False, "did": False, "freed_mb": 0.0, "ms": 0.0, "models": [],
               "before_mb": None, "after_mb": None,
               "why": "回收抛异常：%s: %s" % (e.__class__.__name__, str(e)[:140])}
        print("[vram] 回收预览模型失败（%s）：%s" % (why, str(rec["why"])[:200]), flush=True)
    with _VRAM_LOCK:
        _VRAM_RECLAIM_MARK.update({"at": time.time(), "rec": dict(rec or {})})
    return dict(rec or {})


def _mem_delta(m0: dict, m1: dict) -> str:
    """「导出前后」那一行：RSS 差 / 提交差 + 峰值 RSS。取不到的项写 ?，不抛。"""
    def _n(d, k):
        v = (d or {}).get(k)
        return None if v is None else float(v)

    a, b = _n(m0, "rss"), _n(m1, "rss")
    c, d = _n(m0, "private"), _n(m1, "private")
    out = "RSS %s → %s MB" % (_fmt_mb(a), _fmt_mb(b))
    if a is not None and b is not None:
        out += "（差 %+.1f）" % (b - a)
    out += " · 提交 %s → %s MB" % (_fmt_mb(c), _fmt_mb(d))
    if c is not None and d is not None:
        out += "（差 %+.1f）" % (d - c)
    pk = _n(m1, "peak_rss") or _n(m0, "peak_rss")
    if pk is not None:
        out += " · 峰值 RSS %.1f MB" % pk
    return out


def _free_preview_sessions(why: str = "") -> dict:
    """让**预览 worker** 把它常驻的 dfm / XSeg 卸掉（进程留着，下次预览懒加载）。

    为什么导出收尾必须做这一下：导出用的是**后端进程自己**的会话，预览 worker
    那份是另开的一份 —— 用户实测「导出完了还占着」，占的就是它（一张 384 dfm：
    worker 侧实测 2.3 GB 显存 / 1.3 GB 工作集 / 5.7 GB 提交）。
    _vram_release_inproc 只放本进程，管不到子进程；这里补上。

    worker 没起过（alive()=False）就什么都不做 —— **绝不能**为了「卸一下」把一个
    已经退出的 worker 叫醒（call() 会 _spawn）。
    """
    try:
        if not _WORKER.alive():
            return {"ok": True, "did": False, "why": "worker 没起（不需要卸）"}
    except Exception:                                       # noqa: BLE001
        return {"ok": True, "did": False, "why": "worker 状态读不到"}
    t0 = time.time()
    try:
        r = _WORKER.unload("")
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "did": False,
                "why": "%s: %s" % (e.__class__.__name__, str(e)[:160])}
    return {"ok": bool((r or {}).get("ok", True)), "did": True,
            "unloaded": int((r or {}).get("unloaded") or 0),
            "ms": round((time.time() - t0) * 1000.0, 1), "why": why}


def _vram_release_inproc(reason: str = "任务收尾") -> dict:
    """把**本进程**常驻的 dfm / XSeg 会话放掉（导出收尾时用，成功/失败/取消都放）。

    为什么不能在导出结束时留着：一份 384 会话建好 2.3GB，跑完几十帧之后 ORT 的
    CUDA arena 还会继续涨（本机实测：12 帧之后整卡 used 7049/8151，空闲只剩 843MB）。
    留着它，用户下一次预览（要新建一份 2.4GB 会话）就会被**自己的后端**堵死 ——
    那正是这一轮用户报的「我什么都没跑，却说我显存不够」。放掉之后：下一次预览/导出
    各自懒加载（实测冷加载 3.5~4.2s），中间这段时间卡是干净的。

    只动本进程的缓存，**不动预览 worker**（那一路有自己的懒加载节奏）。
    """
    freed = []
    try:
        with _PRED_LOCK:
            for k in list(_DFM_SESS):
                _DFM_SESS.pop(k, None)
                freed.append("dfm:" + Path(str(k).split("|")[0]).name)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        with _XSEG_SESS_LOCK:
            for _k in list(_XSEG_SESS):
                _XSEG_SESS.pop(_k, None)
                freed.append("xseg")
    except Exception:                                       # noqa: BLE001
        pass
    for tag in freed:
        _vram_own_drop(tag)
    if not freed:
        return {"ok": True, "did": False, "released": [], "reason": reason}
    # 引用断了之后真的把缓存还给驱动（ORT 每个 session 的 arena 跟着 session 一起没）
    try:
        import gc
        gc.collect()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
    f1, _t = _vram_free_mb(fresh=True)
    print("[vram] 收尾放掉本进程的会话（%s）：%s · 空闲现在 %s MB"
          % (reason, "、".join(freed[:4]), "?" if f1 is None else int(f1)), flush=True)
    return {"ok": True, "did": True, "released": list(freed), "free_mb": f1,
            "reason": reason}


def _vram_probe(ttl: float = None) -> tuple:
    """(free_mb, total_mb, source)。

    source 用来在日志里说清「这个数是哪来的」：env / file / torch / nvidia-smi / none。
    env 伪造的值不走缓存（验收脚本会在同一个进程里换档，缓存会骗人）。

    还有一个**只在验收里用**的伪造口：<WORKSPACE>/_vram_free_mb.txt（一行整数，MB）。
    比 env 强的地方是它能在**同一个后端进程**里改 —— 用户那条「关掉占显存的程序后
    再点一次导出，不重启后端也能出片」就是靠它复现的（重启后端来换 env 等于把
    "缓存有没有坏"这件事测没了）。文件不存在就当没有，生产路径上零成本。
    """
    fake = str(os.environ.get(_VRAM_ENV_FREE) or "").strip()
    if fake:
        try:
            return int(float(fake)), None, "env:" + _VRAM_ENV_FREE
        except (TypeError, ValueError):
            pass
    _hook_on = str(os.environ.get("DFL_ALLOW_TEST_HOOK") or "").strip() in ("1", "true", "yes", "on")
    ff = Path(WORKSPACE) / "_vram_free_mb.txt"
    if not _hook_on:                                     # 生产环境不读这个文件（否则工作区里一个文本就能改显存判断）
        ff = None
    try:
        if ff is not None and ff.is_file():
            txt = ff.read_text("utf-8", "ignore").strip().splitlines()[0].strip()
            return int(float(txt)), None, "file:" + ff.name
    except Exception:                                       # noqa: BLE001
        # 文件在、但这一拍读不出来（正好被改写/内容不是数字）：**不许退回旧缓存**
        # —— 那会让"刚刚改成 1400"读成上一次的 8000。这里改成强制重新探测一次。
        if ff is not None and ff.is_file():
            ttl = 0.0
    old = str(os.environ.get(_FUSED_ENV_VRAM) or "").strip()   # 老开关继续认（既有验收脚本在用）
    if old:
        try:
            return int(float(old)), None, "env:" + _FUSED_ENV_VRAM
        except (TypeError, ValueError):
            pass
    ttl = _VRAM_PROBE_TTL if ttl is None else float(ttl)
    now = time.time()
    with _VRAM_LOCK:
        if _VRAM_PROBE["at"] and (now - _VRAM_PROBE["at"]) < ttl:
            return _VRAM_PROBE["free"], _VRAM_PROBE["total"], _VRAM_PROBE["src"]
    free, total = _vram_free_mb(fresh=(ttl <= 0.0))
    # 来源写清楚：torch 与 nvidia-smi 取小（WDDM 上 torch 偏乐观 1.3GB，见 _vram_free_mb）
    tf, sf = _vram_free_torch()[0], _vram_free_smi()[0]
    src = ("none" if free is None else
           "min(torch %s / smi %s)" % ("?" if tf is None else int(tf),
                                       "?" if sf is None else int(sf)))
    with _VRAM_LOCK:
        _VRAM_PROBE.update({"at": now, "free": free, "total": total, "src": src,
                            "free_torch": tf, "free_smi": sf})
    return free, total, src


# ------------------------------------------------------------
# .dfm 到底是什么：它就是一份 **ONNX ModelProto**（_dfm_session 直接
# ort.InferenceSession(dfm)）。所以「模型体量」要问图里要，**别看文件名、也别看分辨率**：
# 用户那一单里「416」那份权重只有 0.5GB、「384」那份 1.5GB —— 名字里的数字只是输入边长，
# 参数量由网络宽度决定，两者没有可比性。
#
# 这里按 protobuf 线格式扫一遍（mmap + 只读头，raw_data 只跳不读）：
#   ModelProto.graph = 7 · GraphProto.initializer = 5 (TensorProto)
#   TensorProto.dims = 1 / data_type = 2 / raw_data = 9
#   GraphProto.input = 11 → ValueInfoProto.type(2).tensor_type(1).shape(2).dim(1).dim_value(1)
# 不依赖 onnx 包（这台机器没装），也不建 session、不加载权重 —— 实测两份模型合计 < 0.2s。
# ------------------------------------------------------------
_DFM_INFO = {}
_DFM_INFO_LOCK = threading.RLock()
_PB_DT_SIZE = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 8: 8, 9: 1, 10: 2, 11: 8, 12: 4,
               13: 8, 14: 4, 15: 8, 16: 2}


def _pb_varint(buf, i, end):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def _pb_fields(buf, i, end):
    """遍历 protobuf 字段：(fno, wire, value)；wire=0 → int、wire=2 → (start, n)。"""
    while i < end:
        key, i = _pb_varint(buf, i, end)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _pb_varint(buf, i, end)
            yield fno, wt, v
        elif wt == 2:
            n, i = _pb_varint(buf, i, end)
            yield fno, wt, (i, n)
            i += n
        elif wt == 5:
            yield fno, wt, int.from_bytes(buf[i:i + 4], "little")
            i += 4
        elif wt == 1:
            yield fno, wt, int.from_bytes(buf[i:i + 8], "little")
            i += 8
        else:
            raise ValueError("wire type %d" % wt)


def _pb_tensor(buf, s, e):
    """一个 TensorProto → (字节数, 元素数, data_type)。"""
    dims, dt, raw = [], None, None
    for fno, wt, v in _pb_fields(buf, s, e):
        if fno == 1:
            if wt == 0:
                dims.append(int(v))
            elif wt == 2:
                i, end = v[0], v[0] + v[1]
                while i < end:
                    d, i = _pb_varint(buf, i, end)
                    dims.append(int(d))
        elif fno == 2 and wt == 0:
            dt = int(v)
        elif fno == 9 and wt == 2:
            raw = int(v[1])                    # 只记长度，不读内容（1.5GB 权重在这儿）
    n = 1
    for d in dims:
        n *= int(d)
    nbytes = raw if raw is not None else n * int(_PB_DT_SIZE.get(dt or 0, 0))
    return int(nbytes), int(n), dt


def _pb_input_px(buf, s, e):
    """ValueInfoProto → 输入边长（取 shape 里最大的那个空间维，动态维跳过）。"""
    for fno, wt, v in _pb_fields(buf, s, e):
        if fno != 2 or wt != 2:                # type
            continue
        for f2, w2, v2 in _pb_fields(buf, v[0], v[0] + v[1]):
            if f2 != 1 or w2 != 2:             # tensor_type
                continue
            for f3, w3, v3 in _pb_fields(buf, v2[0], v2[0] + v2[1]):
                if f3 != 2 or w3 != 2:         # shape
                    continue
                dims = []
                for f4, w4, v4 in _pb_fields(buf, v3[0], v3[0] + v3[1]):
                    if f4 != 1 or w4 != 2:     # dim
                        continue
                    for f5, w5, v5 in _pb_fields(buf, v4[0], v4[0] + v4[1]):
                        if f5 == 1 and w5 == 0:
                            dims.append(int(v5))
                if dims:
                    return int(max(dims[1:])) if len(dims) > 1 else int(max(dims))
    return 0


def _dfm_info(path) -> dict:
    """一份 .dfm 的**真实**体量：权重字节（initializer 之和）/ 参数量 / 输入边长 / 非权重字节。

    返回 {weights_mb, params, params_m, size_px, file_mb, non_weight_mb, init, src}；
    读不出来（不是 ONNX / 用了 external data）→ weights_mb=0 + src 写明原因，调用方回退文件体积。
    """
    key = str(path or "")
    if not key:
        return {"weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
                "file_mb": 0.0, "non_weight_mb": 0.0, "init": 0, "src": "空路径"}
    try:
        st = os.stat(key)
        sig = (int(st.st_size), int(st.st_mtime))
    except OSError:
        return {"weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
                "file_mb": 0.0, "non_weight_mb": 0.0, "init": 0, "src": "文件读不到"}
    with _DFM_INFO_LOCK:
        hit = _DFM_INFO.get(key)
    if hit and hit.get("sig") == sig:
        return dict(hit)
    info = {"sig": sig, "weights_mb": 0.0, "params": 0, "params_m": 0.0, "size_px": 0,
            "file_mb": round(st.st_size / 1048576.0, 1), "non_weight_mb": 0.0,
            "init": 0, "src": ""}
    try:
        import mmap
        with open(key, "rb") as fh:
            buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                graph = None
                for fno, wt, v in _pb_fields(buf, 0, len(buf)):
                    if fno == 7 and wt == 2:          # ModelProto.graph
                        graph = v
                        break
                if graph is None:
                    info["src"] = "不是 ONNX（找不到 graph）→ 退回文件体积"
                else:
                    nbytes = params = init = 0
                    px = 0
                    for fno, wt, v in _pb_fields(buf, graph[0], graph[0] + graph[1]):
                        if fno == 5 and wt == 2:      # initializer
                            b, n, _dt = _pb_tensor(buf, v[0], v[0] + v[1])
                            nbytes += b
                            params += n
                            init += 1
                        elif fno == 11 and wt == 2 and not px:
                            px = _pb_input_px(buf, v[0], v[0] + v[1])
                    if nbytes <= 0:
                        info["src"] = "initializer 是 external data → 退回文件体积"
                    else:
                        info.update({"weights_mb": round(nbytes / 1048576.0, 1),
                                     "params": int(params),
                                     "params_m": round(params / 1e6, 2),
                                     "size_px": int(px), "init": int(init),
                                     "non_weight_mb":
                                         round((st.st_size - nbytes) / 1048576.0, 1),
                                     "src": "ONNX initializer 实测"})
            finally:
                buf.close()
    except Exception as e:                        # noqa: BLE001
        info["src"] = "ONNX 解析失败（%s）→ 退回文件体积" % str(e)[:80]
    with _DFM_INFO_LOCK:
        _DFM_INFO[key] = dict(info)
    return dict(info)


def _vram_model_mb(path) -> float:
    """一份 .dfm 的**权重**体量（MB）—— 取 ONNX initializer 之和，不是文件体积、更不是名字。

    实测（本机两份真模型，_dfm_info 直接扫图）：
        张元英.dfm        输入 384 · 权重 **1566.9MB**（410.76M 参数 · 94 个 initializer）
        阿証証証証-416.dfm 输入 416 · 权重 **514.2MB**（134.79M 参数 · 78 个 initializer）
    两份的「文件体积 − 权重」分别是 0.0 / 0.1MB（dfm 里没有别的数据），所以文件体积在这两
    份上是准的 —— 但**口径**必须是权重：外部权重、带训练残留、或将来换成 fp16 时，体积就
    不再等于权重了。读不出图（不是 ONNX / external data）才退回文件体积，并在日志里写明。
    """
    key = str(path or "")
    if not key:
        return 0.0
    info = _dfm_info(key)
    mb = float(info.get("weights_mb") or 0.0)
    if mb > 0:
        return mb
    try:                                        # 兜底：文件体积（src 里已写明原因）
        return max(0.0, os.stat(key).st_size / 1048576.0)
    except OSError:
        return 0.0


def _vram_models_mb(paths) -> float:
    """一组 dfm 里最大的那份。

    多模型导出时它们常驻在同一个进程里（上限 _MAX_MODELS=3），但同一帧只会用到其中
    一路、而且 A 卸 B 载是按 LRU 走的 —— 按**最大那份**算比按和算更接近峰值，
    又不至于把 3 份 0.5GB 的模型误判成装不下。
    """
    best = 0.0
    for p in (paths or []):
        best = max(best, _vram_model_mb(p))
    return best


# ------------------------------------------------------------
# 项目导出：**一份 worker 会话**真跑起来占多少（实测优先，标定值兜底）
# ------------------------------------------------------------
# 为什么要单独记这一笔：同一份 .dfm，「会话建好」和「跑了几十帧」不是一个数 ——
# 前者是权重 + context（384：2079MB），后者还要加 ORT/CuDNN 那截会长大的工作集
# （实测 384 一份 ≈ 3470MB、416 一份 ≈ 2220MB，见 _VRAM_SESS_RUNTIME_MB 的注释）。
# 预算按前者算就会放行装不下的 worker 数 → WDDM 换页 → 帧速崩到 3.61。
#
# 口径：跑完一单就把「整卡峰值 − 开跑前的整卡占用」摊到每个 worker 上，记进
# <WORKSPACE>/_vram_sess_peak.json（和 _sess_bench.json 一个套路）；下一次算账就用
# max(标定值, 实测值) —— 实测只能把预估**抬高**（更保守），不会把已经放行的单子
# 悄悄放宽。峰值贴到 88% 以上的那一单（已经在换页了）不记：那种数是被墙压扁的。
_VRAM_SESS_PEAK = {}
_VRAM_SESS_LOCK = threading.RLock()
_VRAM_SESS_PAGE_PCT = 0.88
# 记账口径的版本号：口径一变（比如 dfm 批改成「按模型查表 + 乘进程数」），旧文件里那些
# 按旧口径反推出来的会话体量就是错的（它把批成本吞进了会话里），必须整批判掉。
_VRAM_SESS_VER = 2


def _vram_sess_key(dfm_path) -> str:
    """模型身份 = 名字 + 体积 + mtime：换模型 / 重训导出之后实测值自动失效。"""
    key = str(dfm_path or "")
    if not key:
        return ""
    try:
        st = os.stat(key)
        return "%s|%d|%d" % (Path(key).name, int(st.st_size), int(st.st_mtime))
    except OSError:
        return "%s|0|0" % Path(key).name if key else ""


def _vram_sess_path() -> Path:
    return Path(WORKSPACE) / "_vram_sess_peak.json"


def _vram_sess_load() -> dict:
    try:
        p = _vram_sess_path()
        if p.is_file():
            d = json.loads(p.read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def _vram_sess_put(key: str, row: dict):
    """记一笔实测（内存 + 工作区 json；写不进去只影响下一次的预估，不影响这一单）。"""
    if not key:
        return
    with _VRAM_SESS_LOCK:
        _VRAM_SESS_PEAK[str(key)] = dict(row)
    try:
        d = _vram_sess_load()
        d[str(key)] = dict(row)
        _vram_sess_path().write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


def _vram_sess_get(key: str) -> dict:
    if not key:
        return {}
    with _VRAM_SESS_LOCK:
        hit = _VRAM_SESS_PEAK.get(str(key))
    if hit is None:
        hit = _vram_sess_load().get(str(key))
    if not isinstance(hit, dict):
        return {}
    if int(hit.get("v") or 0) != int(_VRAM_SESS_VER):   # 旧口径记的：整批判掉
        return {}
    return dict(hit)


def _vram_sess_peak_mb(paths) -> dict:
    """一份项目导出 worker 会话的**峰值**（MB）—— 多模型时取最大的那一份。

    返回 {"sess_mb", "src", "model_mb", "measured_mb", "estimate_mb", "key", "row"}：
      · estimate = 权重 + context + _VRAM_SESS_RUNTIME_MB（实测标定的运行时工作集）；
      · measured = 上一单真跑出来的「整卡峰值增量 / worker 数」（没有就空）；
      · 采用 max(estimate, measured) —— 实测只抬高、不放松（见上面那一段）。
    """
    paths = [str(p) for p in (paths or []) if str(p or "")]
    model_mb = _vram_models_mb(paths)
    est = float(model_mb) + float(_VRAM_CTX_MB) + float(_VRAM_SESS_RUNTIME_MB)
    best_mb, best_row = 0.0, {}
    for p in paths:
        row = _vram_sess_get(_vram_sess_key(p))
        try:
            mb = float(row.get("sess_mb") or 0.0)
        except (TypeError, ValueError):
            mb = 0.0
        if mb > best_mb:
            best_mb, best_row = mb, row
    sess = max(est, best_mb) if best_mb > 0 else est
    # 兜底上限：实测值再离谱也不许超过整卡的 60%（那是「读数坏了」而不是这台机器）
    try:
        _free, total, _src = _vram_probe()
        if total:
            sess = min(sess, float(total) * 0.6)
    except Exception:                                       # noqa: BLE001
        pass
    return {"sess_mb": round(float(sess), 1),
            "src": ("实测 max(标定 %.0f, 量到 %.0f)" % (est, best_mb)) if best_mb > 0
                   else ("标定 %.0f = 权重 %.0f + context %d + 运行时 %d"
                         % (est, model_mb, int(_VRAM_CTX_MB), int(_VRAM_SESS_RUNTIME_MB))),
            "model_mb": round(float(model_mb), 1),
            "measured_mb": round(float(best_mb), 1), "estimate_mb": round(est, 1),
            "key": _vram_sess_key(paths[0]) if paths else "", "row": dict(best_row)}

def _proj_record_sess_peak(dfms, n_workers: int, batch: int, base_used_mb,
                           peak_used_mb, total_mb, size_px=None,
                           ok: bool = True) -> dict:
    """把这一单**实测**的「一份 worker 会话峰值」记下来（下一单的预算直接用它）。

    口径：ours = 整卡峰值 − 开跑前的整卡占用（含预热那几秒）；一份 = ours / worker 数。
    不采纳的四种情形（宁可不记，也不许把预算带偏）：
      · 这一单不是正常跑完的（取消/有失败帧）—— 峰值没跑到；
      · 峰值贴到整卡 88% 以上 —— 已经在换页，读数是墙压出来的（正是要修的病症）；
      · 量出来的比权重还小 —— 读数坏了（探测到的是别人的进程在退）；
      · 多模型项目（LRU 换进换出，摊不到单份上）。
    """
    model_mb = _vram_models_mb(dfms)
    try:
        base = float(base_used_mb or 0.0)
        peak = float(peak_used_mb or 0.0)
        total = float(total_mb or 0.0)
    except (TypeError, ValueError):
        return {}
    if not ok or len([p for p in (dfms or []) if str(p or "")]) != 1:
        return {}
    if peak <= 0 or base <= 0 or peak <= base:
        return {}
    if total and peak >= total * float(_VRAM_SESS_PAGE_PCT):
        print("[project] 峰值 %.0f/%.0f MB = %.0f%% 已经贴墙（换页），这一次的会话体量不记账"
              % (peak, total, 100.0 * peak / total), flush=True)
        return {}
    per = (peak - base) / float(max(1, int(n_workers)))
    # 把预算里**单列的那两项**扣回去，否则同一笔钱记两遍：
    #   · dfm 批（这一档线程数）—— 下一单按它自己的 batch 再加；
    #   · 检测/遮罩（_VRAM_AUX_MB）—— 多 worker 时它只算一份，摊到每份头上就是白记。
    # 剩下的才是「一份会话」：权重 + context + 运行时工作集 + 本进程那点 torch context。
    try:
        per -= float(_vram_batch_mb(int(size_px or _VRAM_BASE_PX), int(batch),
                                    model_mb, "procs"))
        per -= float(_VRAM_AUX_MB)
    except Exception:                                   # noqa: BLE001
        pass
    # 实测只用来**纠偏**：抬到标定值的 1.5 倍以上就不像话了（多半是本进程那份 torch
    # context / 别人的进程在那一拍抖了一下），按 1.5 倍记。
    est = float(model_mb or 0.0) + float(_VRAM_CTX_MB) + float(_VRAM_SESS_RUNTIME_MB)
    per = min(float(per), est * 1.5)
    if per < float(model_mb or 0.0):
        return {}
    row = {"v": int(_VRAM_SESS_VER), "sess_mb": round(float(per), 1),
           "workers": int(n_workers), "batch": int(batch), "base_used_mb": round(base, 1),
           "peak_used_mb": round(peak, 1), "total_mb": round(total, 1),
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    key = _vram_sess_key(dfms[0])
    _vram_sess_put(key, row)
    print("[project] 实测一份 worker 会话 %.2fGB（整卡峰值 %.0f − 开跑前 %.0f，%d 个 worker）"
          "→ 记进 _vram_sess_peak.json，下一单的预算按它算"
          % (per / 1024.0, peak, base, int(n_workers)), flush=True)
    return row


def _dfm_size_hint(dfm) -> int:
    """这个 .dfm 的输入边长（384/416/…）——**不加载模型**就要知道，因为预算要在加载前算。

    口径：**问图要**（ONNX 的 graph.input 形状，_dfm_info），不问文件名 —— 文件名里的数字
    会骗人（本仓库两份真模型：名字里带 416 的那份输入确实是 416，不带数字的 张元英.dfm
    其实是 **384**，老代码按名字取不到就退回 416，预算里的帧/批两项因此一直算偏）。
    名字只作为最后的兜底，再不行才 416。
    真正喂给 session 的尺寸仍由 session 自己决定：_fused_run 加载后会拿实测尺寸
    把这个预算**再算一遍**（不一样就按新尺寸重新判一次）。
    """
    key = str(dfm or "")
    with _VRAM_LOCK:
        hit = _VRAM_SIZE_CACHE.get(key)
    if hit:
        return int(hit)
    size = int((_dfm_info(key) or {}).get("size_px") or 0)
    if size:
        with _VRAM_LOCK:
            _VRAM_SIZE_CACHE[key] = int(size)
        return int(size)
    try:
        stem = Path(key).stem
        cand = [int(m) for m in re.findall(
            r"(?<![0-9])(256|288|320|384|416|448|480|512|640)(?![0-9])", stem)]
        if cand:
            size = max(cand)
    except Exception:                                       # noqa: BLE001
        size = 0
    if not size:
        size = _VRAM_BASE_PX
    with _VRAM_LOCK:
        _VRAM_SIZE_CACHE[key] = int(size)
    return int(size)


def _vram_frame_mb(size_px: int) -> float:
    """一帧在飞时压在 GPU 上大约多少 MB：dfm 输入 + 输出脸 + prd + dst 四份 float32。

    416 → 7.92MB（用户口径 30×7.9 ≈ 237MB）；512 → 12.0MB。
    """
    size_px = max(64, int(size_px or _VRAM_BASE_PX))
    return max(4.0, (size_px * size_px * 3 * 4 * 4) / 1048576.0)


def _vram_batch_table(model_mb=None, mode: str = "frames"):
    """挑那张批台阶表 → (量的输入边长, 点集)。

    · `mode == "procs"`（项目导出）：按**权重体量**挑 —— ≤900MB 用 416 那张
      （阿証…-416.dfm，514MB），更大的用 384 那张（张元英.dfm，1567MB）；
    · 其它（fused 帧预算 / 预览）：老表 `_VRAM_BATCH_PTS`（量的是「dfm 前向 k 张」，
      不是项目那条路的「k 帧合成 + 落盘」）；基准边长 416。
    """
    if str(mode) != "procs":
        return int(_VRAM_BASE_PX), _VRAM_BATCH_PTS
    mb = float(model_mb or 0.0)
    for lim, base_px, pts in _VRAM_BATCH_TABLES:
        if mb <= float(lim):
            return int(base_px), pts
    return int(_VRAM_BATCH_TABLES[-1][1]), _VRAM_BATCH_TABLES[-1][2]


def _vram_batch_mb(size_px: int, batch, model_mb=None, mode: str = "frames") -> float:
    """dfm 一次前向喂 batch 张时**多占**的显存（相对同一份会话 batch=1）。

    点集按模型体量挑（_vram_batch_table），再按 (size/基准边长)² 缩放：基准边长就是
    那张表量的时候那个模型的输入（384 表在 384 上量的，416 表在 416 上量的）。
    """
    b = int(batch or 0)
    if b <= 0:
        return 0.0
    base_px, pts = _vram_batch_table(model_mb, mode)
    k = (max(64, int(size_px or base_px)) / float(base_px)) ** 2
    if b <= pts[0][0]:
        return pts[0][1] * k
    for i in range(1, len(pts)):
        b0, c0 = pts[i - 1]
        b1, c1 = pts[i]
        if b <= b1:
            return (c0 + (c1 - c0) * (b - b0) / float(b1 - b0)) * k
    b_last, c_last = pts[-1]
    b_prev, c_prev = pts[-2]
    slope = (c_last - c_prev) / float(b_last - b_prev)
    return (c_last + slope * (b - b_last)) * k


def _vram_reserve_mb(free_mb) -> float:
    """这次该给驱动/碎片/别人插队留多少 MB —— **可解释，而且不给 8GB 卡留死重**。

    口径：reserve = min(768, max(256, free × 5%))。

    为什么不是老的「free × 20% + 512MB」（那在 8GB 卡空闲 3.4GB 时是 1.19GB，
    再叠上「一份会话 / 一帧缓冲」之后正好变成用户看到的「需要 2.46GB / 可用 2.03GB」）：

      · **20% 是拍的，不是量的。** 它出在 ceaf4f4 第一版预算里（提交信息只有
        「avail=free×0.8-512MB」，没有任何测量支撑）。而真正会被预留吸收的那两项
        早就有实测，而且都已经落在 need 里：
          - CUDA context：每份会话 512MB，need 里明着算（_VRAM_CTX_MB）；
          - ORT 的 CUDA arena：416 实测 +135MB、384 实测 +221MB（会话建好那一刻），
            而「batch 4 起 arena 换池子」那一步的 +1036MB **已经包含在
            _VRAM_BATCH_PTS 的实测台阶里**（那张表量的是「整卡增量 − 会话本身」），
            再拿一个 20% 去覆盖它 = 同一笔钱记两遍。
      · 真正需要预留的是**探测滞后**和**别人插队**，量级都在几百 MB：
          - P0 校准：我们的 free 与 nvidia-smi 原始行差 ≤256MB（两条路取小之后）；
          - 探测有 2~4s 缓存（_VRAM_PROBE_TTL / _SMI_MEM），这期间浏览器开个视频
            硬件加速就能抢走几百 MB；
          - 用户改一次参数会连打几次 dry_run，中间那点抖动不该被判成「不够」。
        所以底线取 256MB 正好覆盖第一条，5% 让大空闲时自动放宽一点，
        上限 768MB 防止 24GB 卡上又长出一个 1.2GB 的死重。

    实测佐证（本机 RTX 5070 Laptop 8GB）：按新口径算，「空闲 3.17GB / 需要 2.46GB」
    的单子直接放行，真跑 12 帧 0 OOM（app/tools/accept/vram_force_accept.py 的 A 段）；
    而按老口径它必被拒 —— 那 1.37GB 里没有一个字节是实测需要的。
    """
    free = max(0.0, float(free_mb or 0.0))
    # 对照/回滚用：DFL_VRAM_RESERVE_MODE=legacy20 复原老口径（空闲 20% + 512MB）。
    # （下面的 _vram_reserve_rule_text 把同一件事说成一句人话，文案里照抄它。）
    # 只给验收脚本做「新旧两套数字并排」用 —— 生产默认 auto（上面的新口径）。
    if str(os.environ.get("DFL_VRAM_RESERVE_MODE") or "").strip().lower() in ("legacy", "legacy20", "old"):
        return free * 0.20 + 512.0
    return min(float(_VRAM_RESERVE_MAX_MB),
               max(float(_VRAM_RESERVE_MB), free * float(_VRAM_SAFETY)))

def _vram_cpu_fallback_env() -> bool:
    return str(os.environ.get(_VRAM_ENV_CPU) or "").strip().lower() in ("1", "true", "yes", "on")


def _vram_cpu_on() -> bool:
    """当前是不是「强制 CPU EP」（显存不够 + 用户开了 DFL_ALLOW_CPU_FALLBACK）。

    _dfm_session / _stage_xseg 都读它 —— 一处开关，全链路一致，不会出现
    「dfm 在 CPU 上、XSeg 还在抢显存」这种半吊子状态。
    """
    with _CPU_EP_LOCK:
        return bool(_CPU_EP["on"])


def _vram_set_cpu(on: bool, why: str = "") -> bool:
    on = bool(on)
    with _CPU_EP_LOCK:
        was = bool(_CPU_EP["on"])
        _CPU_EP["on"] = on
        _CPU_EP["why"] = str(why or "") if on else ""
    if on != was:
        print("[vram] %s CPU EP（%s）"
              % ("★ 切到" if on else "解除 CPU（回到 GPU）", why or "显存预算恢复"), flush=True)
    return on


def _vram_cands(mode: str, want_frames: int, want_batch: int,
                want_procs: int, procs: int) -> list:
    """候选计划：从「最想要」到「最省」。

    · frames —— fused 视频导出：先保帧预算（流水线的重叠靠它），再降 dfm 批
      （416 上 B1/B4/B8 每张耗时几乎一样，所以降批几乎不花钱）；
    · procs  —— 项目导出：先降 worker 进程数（每份 session + 一个 CUDA context），
      减到 1 个之后再降 worker 内的线程数（batch=几帧并行）；两样都到最小才拒绝；
    · none   —— 预览：只问「一份会话装不装得下」。
    """
    if mode == "none":
        return [{"frames": 0, "batch": 1, "procs": max(1, int(procs))}]
    if mode == "procs":
        # 降档顺序是用户定的：**先减 worker（每份 = 一整套会话 + 一个 CUDA context），
        # 减到 1 个之后才降 worker 内的线程数（batch），两样都到最小还装不下才拒绝**。
        # 以前只列 procs、batch 永远按 1 算 —— 于是「1 个 worker 也装不下」时
        # 直接 400，而其实把 2 线程降到 1 线程就够（dfm 批那一项 160 → 80MB 也是钱）。
        procs_ladder = [{"frames": 0, "batch": int(want_batch), "procs": n}
                        for n in range(max(1, int(want_procs)), 0, -1)]
        batches = sorted({int(b) for b in (want_batch, 4, 2, 1)
                          if 1 <= int(b) < int(want_batch)}, reverse=True)
        return procs_ladder + [{"frames": 0, "batch": int(b), "procs": 1} for b in batches]
    frames_ladder = sorted({int(f) for f in (want_frames, int(_FUSED_BUDGETS[1]),
                                             int(_FUSED_BUDGETS[2]), int(_FUSED_BUDGETS[-1]))
                            if 1 <= int(f) <= int(want_frames)}, reverse=True) \
        or [int(want_frames)]
    batch_ladder = sorted({int(b) for b in (want_batch, 4, 2, 1)
                           if 1 <= int(b) <= int(want_batch)}, reverse=True) \
        or [int(want_batch)]
    out = []
    for n in [max(1, int(procs))]:               # fused：单进程，只有帧/批两个旋钮
        for f in frames_ladder:
            for b in batch_ladder:
                out.append({"frames": int(f), "batch": int(b), "procs": int(n)})
    return out


def _vram_need_mb(cand: dict) -> float:
    """一个候选/计划的总需求（MB）：会话×进程 + 检测遮罩 + 帧缓冲 + dfm 批。

    resident = 这个进程**已经**持有几份会话（fused 加载完 dfm 后重算那一拍 = 1）：
    那几份的「权重 + context」已经在卡上了，不再算进需求 —— 再算一遍就是重复计数，
    会把一个本来跑得动的单子判死（用户截图那一单就是这么被拒的）。
    """
    procs = max(1, int(cand.get("procs") or 1))
    new_sess = max(0, procs - max(0, int(cand.get("resident") or 0)))
    # dfm 批是**每个进程各自**的（worker 里 k 个线程同时前向），所以要乘进程数 ——
    # 只在外面加一次会漏掉「2 个 worker × 2 线程」的那份（实测差 1.4GB）。
    return (float(cand.get("sess_mb") or 0.0) * new_sess
            + float(cand.get("batch_mb") or 0.0) * procs
            + float(cand.get("aux_mb") or 0.0)
            + int(cand.get("frames") or 0) * float(cand.get("frame_mb") or 0.0))


def _vram_plan(model_mb, opts: dict = None) -> dict:
    """统一的显存预算：模型体量 + 选项 → 计划 batch / 帧预算 / 预计峰值 / 可行?

    opts:
      mode       frames（fused 视频导出，默认）| procs（项目导出 worker）| none（预览）
      size_px    dfm 输入边长（默认 416；用 _dfm_size_hint 拿，别加载模型）
      frames     想要的帧缓冲（默认 30 = _FUSED_BUDGETS[0]）
      batch      想要的 dfm 批（默认 _FUSED_BATCH_TARGET）
      procs      固定要几份会话（fused/preview = 1）
      max_procs  mode=procs 时的上限（默认 = procs）
      aux        要不要算检测器/关键点/XSeg（默认 True）
      label      用途名（进日志与 status）
      free_mb / total_mb  覆盖探测量（单测用；不传就走 _vram_probe）
      reclaim    起算之前先**回收**我们自己占的那一块（常驻预览 worker），默认关；
                 导出闸门开它 —— 用户「预览过一帧再点导出」被拒的病根就在这里
      resident   本进程**已经**持有这份会话了（fused 加载后重算那一拍）：会话那一项
                 不再计入需求，否则等于把一个已经吃掉的 2GB 又算一遍 → 必被拒
      sess_mb / sess_src  **一份会话的峰值**直接给（项目导出用）：给了就不再用
                 「权重 + context」推 —— 那条路一份会话真跑起来还要多 1.3GB，
                 见 _vram_sess_peak_mb / _VRAM_SESS_RUNTIME_MB 的实测注释

    返回 dict（关键字见下面 plan = {...}）。**ok=False 的调用方必须拒绝**：
    _vram_plan 只算账，不抛异常（拒绝的样子由 _vram_guard / 调用点决定）。
    开了 DFL_ALLOW_CPU_FALLBACK 且装不下时，返回 ok=True + cpu=True。
    """
    opts = dict(opts or {})
    mode = str(opts.get("mode") or "frames")
    label = str(opts.get("label") or "显存预算")
    size_px = int(opts.get("size_px") or _VRAM_BASE_PX)
    procs = max(1, int(opts.get("procs") or 1))
    want_frames = int(opts.get("frames") or 0) or int(_FUSED_BUDGETS[0])
    want_frames = max(int(_FUSED_BUDGETS[-1]), min(64, int(want_frames)))
    want_batch = int(opts.get("batch") or 0) or int(_FUSED_BATCH_TARGET)
    want_batch = max(1, min(int(_PRED_BATCH_CAP), int(want_batch)))
    max_procs = max(1, min(16, int(opts.get("max_procs") or procs)))
    aux = float(_VRAM_AUX_MB) if opts.get("aux", True) else 0.0
    # resident：这份会话（含它自己的 CUDA context）**已经在当前进程里**了。
    # 只对「同一进程、同一份会话」成立，而且只在重算那一拍用（见 _fused_run）。
    resident = 1 if opts.get("resident") else 0
    model_mb = max(0.0, float(model_mb or 0.0))
    # 模型身份（参数量 / 权重 / 输入边长）：由调用方从 .dfm 的图里读出来（_dfm_info），
    # 这里只负责挂到 plan 上，让 status/日志/拒绝消息都能说清「凭什么按这一档」。
    minfo = dict(opts.get("model_info") or {})
    # 一份会话 = 权重 + 它自己的 CUDA context。ORT 的 arena（416 建好时 +135MB）不单列
    # —— 它由「空闲的 5% + 256MB」那份预留吸收。但**项目导出**那条路上会话跑起来还要
    # 再长一块（实测 +1190 / +1390MB，见 _VRAM_SESS_RUNTIME_MB）：那一块必须明着算，
    # 否则 384 模型 2 个 worker 会被算成 4538MB（≤ 可用）放行，真跑起来顶到 94.9%
    # → WDDM 换页 → 3.61 帧/秒。调用方给了 sess_mb（实测值）就照给的算。
    sess_override = opts.get("sess_mb")
    if sess_override is not None:
        sess_mb = max(float(sess_override or 0.0), model_mb)
        sess_src = str(opts.get("sess_src") or "调用方给的实测值")
    else:
        sess_mb = model_mb + float(_VRAM_CTX_MB)
        sess_src = "权重 + context"
        if mode == "procs":
            sess_mb += float(_VRAM_SESS_RUNTIME_MB)
            sess_src = ("权重 + context + 运行时工作集 %dMB（标定值）"
                        % int(_VRAM_SESS_RUNTIME_MB))
    frame_mb = _vram_frame_mb(size_px)
    # ---- ★ 顺序（用户口径，别改回去）：先回收我们自己占的那块 → 再探测 → 再算预算 ----
    # 以前是「先按现状算，**装不下才**回收」，而阶梯里永远有 8 帧 / 批 1 这一档 ——
    # 于是「装得下（降档）」成立、回收那一步一次都没走到：日志里「可回收 2733 MB」进了
    # 「显存去向」那一栏，公式里却是 0（reclaim_left 初值写死 0），帧预算 30 → 8、
    # dfm 批 8 → 1，用户实测 22 → 15.8 帧/秒。
    # reclaim_first=True = 导出真起手那一拍（路由闸门 + worker 线程里的那一次）：
    #   卸掉常驻预览 worker 的模型，**用回收后的空闲**算账；dry_run 预检不传它
    #   （预检不该为了算账去卸模型：拖一次滑杆就重载 4 秒），预检靠下面
    #   「可回收计入可用」这一条拿到同样的结论，真起手时再真回收。
    rec = {}
    if opts.get("free_mb") is not None:
        free, total, src = int(opts["free_mb"]), opts.get("total_mb"), "override"
    else:
        if opts.get("reclaim") and opts.get("reclaim_first"):
            rec = _vram_reclaim_once(label)
        # 开过回收就**重新探**（ttl=0）：拿回收前的旧数算账等于白回收一遍
        free, total, src = _vram_probe(0.0 if rec else None)

    def _mk(frames, batch, nproc):
        # resident=1 时，第一份会话的「权重 + context」已经在这块卡上了（本进程持有），
        # 需求里只剩**增量**：检测/遮罩 + 帧缓冲 + dfm 批。
        cand = {"sess_mb": round(sess_mb, 1), "aux_mb": round(aux, 1),
                "frame_mb": round(frame_mb, 2), "frames": int(frames),
                "batch": int(batch),
                "batch_mb": round(_vram_batch_mb(size_px, batch, model_mb, mode), 1),
                "procs": int(nproc), "resident": int(resident),
                "resident_mb": round(float(sess_mb) * min(int(nproc), int(resident)), 1)}
        cand["need_mb"] = round(_vram_need_mb(cand), 1)
        return cand

    ladder = [_mk(c["frames"], c["batch"], c["procs"])
              for c in _vram_cands(mode, want_frames, want_batch, max_procs, procs)]
    floor = ladder[-1]
    plan = {
        "ok": False, "cpu": False, "mode": mode, "label": label, "size_px": int(size_px),
        "model_mb": round(model_mb, 1), "sess_mb": round(sess_mb, 1),
        # 模型体量的来路：参数量 / 权重 MB / 输入边长（全部来自 ONNX 图，不看文件名）
        "params_m": (round(float(minfo.get("params_m") or 0.0), 2)
                     if minfo.get("params_m") else None),
        "params": (int(minfo["params"]) if minfo.get("params") else None),
        "model_mb_src": str(minfo.get("src") or "文件体积（没读到图）"),
        "file_mb": float(minfo.get("file_mb") or 0.0) or None,
        "non_weight_mb": round(float(minfo.get("non_weight_mb") or 0.0), 1),
        "size_px_src": ("ONNX 图输入" if minfo.get("size_px") else "文件名/默认"),
        # 一份会话里「权重 + context」之外的那截（项目导出的运行时工作集 / 实测值）
        "sess_src": sess_src,
        "sess_extra_mb": round(max(0.0, float(sess_mb) - model_mb - float(_VRAM_CTX_MB)), 1),
        "ctx_mb": int(_VRAM_CTX_MB), "arena_mb": int(_VRAM_ARENA_MB),   # arena 只在日志里提一句
        "aux_mb": round(aux, 1), "frame_mb": round(frame_mb, 2),
        # 预留：随空闲缩放（free×5%，底线 256MB、上限 768MB），见 _vram_reserve_mb
        "safety": float(_VRAM_SAFETY),
        "reserve_mb": round(_vram_reserve_mb(free), 1),
        "reserve_min_mb": int(_VRAM_RESERVE_MB), "reserve_cap_mb": int(_VRAM_RESERVE_MAX_MB),
        "reserve_pct_mb": round(float(free or 0.0) * float(_VRAM_SAFETY), 1),
        "forced": False,
        "free_mb": (None if free is None else int(free)),
        "total_mb": (None if total is None else int(total)),
        "source": src, "avail_mb": None, "need_mb": round(floor["need_mb"], 1),
        "peak_mb": None, "procs": int(max_procs if mode == "procs" else procs),
        # 谁占着显存 + 我们回收掉了多少（拒绝消息/status/日志都读这几个键）
        "reclaim": dict(rec or {}),
        "reclaimed_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1),
        "reclaim_ms": round(float((rec or {}).get("ms") or 0.0), 1),
        "reclaim_note": str((rec or {}).get("why") or ""),
        "reclaim_left_mb": 0.0,
        "own_mb": None, "own_inproc_mb": None, "own_preview_mb": None,
        "reclaimable_mb": None, "other_mb": None, "preview_models": [],
        "free_torch_mb": None, "free_smi_mb": None, "resident_mb": round(floor["resident_mb"], 1),
        "frames": int(want_frames if mode != "procs" else 0),
        # procs 模式下 batch = worker 内的线程数（几帧并行），已经是预算的一部分了
        "batch": int(want_batch),
        "batch_mb": round(_vram_batch_mb(size_px, want_batch, model_mb, mode), 1),
        "requested_frames": int(want_frames), "requested_batch": int(want_batch),
        "requested_procs": int(max_procs if mode == "procs" else procs),
        "downgraded": False, "steps": [], "reason": "", "msg": "", "suggest": [],
        # 老字段（_fused_budget 的调用方/验收脚本还在读）
        "frame_budget": int(want_frames if mode != "procs" else 0),
        "requested": int(want_frames), "per_frame_mb": round(frame_mb, 2),
    }
    if free is None:
        plan.update({"ok": True,
                     "reason": "拿不到显存信息（torch / nvidia-smi 都没答）→ 不猜、不降档",
                     "msg": "%s：拿不到显存信息 → 按请求值跑（帧预算 %d / 批 %d）"
                            % (label, plan["frames"], plan["batch"])})
        return plan
    # ---- 谁占着这块卡（本程序 / 其它程序 / 空闲）+ 可回收多少 ----
    def _apply_own(o):
        plan.update({"own_mb": o.get("own_mb"),
                     "own_inproc_mb": o.get("own_inproc_mb"),
                     "own_preview_mb": o.get("own_preview_mb"),
                     "own_parts": dict(o.get("own_parts") or {}),
                     "reclaimable_mb": o.get("reclaimable_mb"),
                     "other_mb": o.get("other_mb"),
                     "preview_models": list(o.get("preview_models") or []),
                     "free_torch_mb": o.get("free_torch_mb"),
                     "free_smi_mb": o.get("free_smi_mb")})

    def _avail_for(extra_mb):
        # 口径：可用 = 空闲 − 预留(free) + 可回收。
        #   · 「+ 可回收」：我们自己占着 2GB 但随时能收回来，不该算成「这台机器不够跑」；
        #   · 「− 预留」：预留**随空闲缩放**（5%，底线 256 / 上限 768），
        #     不是老的 20%+512MB —— 见 _vram_reserve_mb 那一大段实测依据。
        reserve = float(_vram_reserve_mb(free))
        plan["reserve_mb"] = round(reserve, 1)
        plan["reserve_pct_mb"] = round(float(free) * float(_VRAM_SAFETY), 1)
        return max(0.0, float(free) - reserve + float(extra_mb))

    def _pick(avail_value):
        plan["avail_mb"] = round(avail_value, 1)
        plan["steps"] = []
        first = None
        for cand in ladder:
            fit = bool(cand["need_mb"] <= avail_value)
            plan["steps"].append({"frames": int(cand["frames"]),
                                  "batch": int(cand["batch"]),
                                  "procs": int(cand["procs"]),
                                  "need_mb": cand["need_mb"],
                                  "batch_mb": cand["batch_mb"],
                                  "avail_mb": round(avail_value, 1), "fit": fit})
            if fit and first is None:
                first = cand
        return first

    own = _vram_owners(free, total, src)
    _apply_own(own)
    # ---- 可回收**真的加进公式** ----
    # 这一条以前是写死的 0：日志第二行把 reclaim_left_mb 当「+ 可回收」打出来，
    # 而它只在「装不下才回收」那个分支里赋值 —— 那个分支永远走不到，于是
    # 「可回收 2733 MB」只是个装饰数字，预算里加的是 0。
    # 口径：回收**没成**（这回没卸 / 卸了没腾出来）时，那 N MB 仍按「可回收」计入
    # 可用 —— 因为导出起手时还会再收一次；回收**成了**就不再计（那 N MB 已经算在
    # 新的 free 里，重复加等于骗自己）。
    # 只对真探测的 free 生效：env/file/override 是伪造/单测值，拿它加可回收就是假账。
    reclaim_left = 0.0
    if opts.get("reclaim") and not (rec or {}).get("did") and _vram_src_real(src):
        reclaim_left = float(own.get("reclaimable_mb") or 0.0)
    plan["reclaim_left_mb"] = round(reclaim_left, 1)
    avail = _avail_for(reclaim_left)
    hit = _pick(avail)

    # ---- 装不下才回收我们自己占的那块（预览 worker 里常驻的 dfm），然后算第二遍 ----
    # 回收成功 → 空闲里已经含那 2GB，**不再**把它加进 avail（重复算等于骗自己）；
    # 没收成（worker 正忙/超时）→ 那 2GB 仍按「可回收」计入 avail，消息里注明
    # 「导出前会再收一次」（_vram_reclaim 在起 job 前还会被调一次）。
    # 三档模式（frames / procs / none）都适用：阶段链的闸门也是 mode=none。
    # 预览自己不会走到这里 —— _preview_guard 明确传 reclaim=False（它要用的正是
    # worker 里那份常驻模型）。
    # 补一次回收：只在「这一拍还没收过」且真的降到底才走（dry_run 预检的第一拍就会
    # 命中这里 —— 预检不该先卸模型，但真降到底时宁可收一次也别把一个能跑的单子拒了）。
    if hit is None and opts.get("reclaim") and opts.get("free_mb") is None and not rec:
        rec = _vram_reclaim_once(label)
        plan.update({"reclaim": dict(rec or {}),
                     "reclaimed_mb": round(float((rec or {}).get("freed_mb") or 0.0), 1),
                     "reclaim_ms": round(float((rec or {}).get("ms") or 0.0), 1),
                     "reclaim_note": str((rec or {}).get("why") or "")})
        free2, total2, src2 = _vram_probe(0.0)
        if free2 is not None and (rec.get("did") or int(free2) > int(free)):
            free = free2
            if total2 is not None:
                total = total2
            src = src2
            plan["free_mb"] = int(free)
            plan["total_mb"] = (None if total is None else int(total))
            plan["source"] = src
        own = _vram_owners(free, total, src)
        _apply_own(own)
        # 没收成 → 那 N MB 仍按「可回收」计入（起 job 前 _fused_run/_stage_worker 会再收
        # 一次；这一拍先别把它判死）。收成了 → 已经在 free 里，计 0，绝不重复加。
        reclaim_left = (float(own.get("reclaimable_mb") or 0.0)
                        if (not rec.get("did") and _vram_src_real(src)) else 0.0)
        plan["reclaim_left_mb"] = round(reclaim_left, 1)
        avail = _avail_for(reclaim_left)
        hit = _pick(avail)

    # ---- 还装不下？那就放掉**本进程**常驻的会话（预览/下一次导出再懒加载）----
    # 只在「没有任何任务在跑」时做：有任务在跑时放掉 session 等于把它的腿锯了。
    if (hit is None and opts.get("reclaim_inproc") and opts.get("free_mb") is None
            and not _merge_job.get("running") and not _STAGE_JOB.get("running")):
        rel = _vram_release_inproc(label)
        if rel.get("did"):
            plan["reclaim_inproc"] = dict(rel)
            free2, total2, src2 = _vram_probe(0.0)
            if free2 is not None:
                free = free2
                if total2 is not None:
                    total = total2
                src = src2
                plan["free_mb"] = int(free)
                plan["total_mb"] = (None if total is None else int(total))
                plan["source"] = src
                own = _vram_owners(free, total, src)
                _apply_own(own)
                avail = _avail_for(0.0)
                hit = _pick(avail)
    if hit is not None:
        # 降档只看**这条路真的有旋钮**的那几维：procs 模式帧预算恒为 0（requested_frames
        # 却有个默认 30），照老写法比一下永远「降了档」，status 里那句「已降档」就成了噪音。
        _down = bool(hit["batch"] < plan["requested_batch"]
                     or hit["procs"] < plan["requested_procs"])
        if mode != "procs":
            _down = _down or bool(hit["frames"] < plan["requested_frames"])
        plan.update({"ok": True, "frames": int(hit["frames"]), "batch": int(hit["batch"]),
                     "procs": int(hit["procs"]), "batch_mb": hit["batch_mb"],
                     "peak_mb": hit["need_mb"], "downgraded": _down})
        plan["frame_budget"] = int(plan["frames"])
        plan["reason"] = ("装得下：请求 (帧 %d / 批 %d / 进程 %d) → 实际 (帧 %d / 批 %d / 进程 %d)"
                          % (plan["requested_frames"], plan["requested_batch"],
                             plan["requested_procs"], plan["frames"], plan["batch"],
                             plan["procs"])
                          if plan["downgraded"] else
                          "装得下：帧 %d / 批 %d / 进程 %d（不用降档）"
                          % (plan["frames"], plan["batch"], plan["procs"]))
        if rec.get("did") and float(plan.get("reclaimed_mb") or 0.0) > 0:
            plan["reason"] += ("（回收预览模型 %.1f GB 之后才装得下）"
                               % (float(plan["reclaimed_mb"]) / 1024.0))
        elif reclaim_left > 0:
            plan["reason"] += ("（其中 %.1f GB 要靠回收预览 worker 才腾得出来%s）"
                               % (reclaim_left / 1024.0,
                                  ("；这一拍没回收成：" + str(rec.get("why") or "没有原因")[:80])
                                  if rec and not rec.get("did") else ""))
        plan["msg"] = "%s：%s · 预计峰值 %.0f MB / 可用 %.0f MB" % (
            label, plan["reason"], hit["need_mb"], avail)
        # 放行的这一份也带 summary：status 页 / 排障日志里的「本程序 / 其它程序 /
        # 空闲」和拒绝态是同一句话，用户不用在两个地方看两套口径。
        plan["summary"] = _vram_summary_line(plan)
        return plan

    # 连最小档都装不下：把「最小档」的数字落到 plan 上（拒绝消息里的拆项必须和
    # need_mb 对得上，否则用户照着算会发现少了一截）
    plan.update({"peak_mb": None, "frames": int(floor["frames"]),
                 "batch": int(floor["batch"]), "batch_mb": floor["batch_mb"],
                 "procs": int(floor["procs"])})
    plan["frame_budget"] = int(plan["frames"])
    plan["msg"] = _vram_refuse_msg(plan, label)
    plan["summary"] = _vram_summary_line(plan)
    # 注意：这段文字会进后端日志，而验收脚本靠「日志里没有 out of memory / OOM」
    # 证明「拒绝发生在加载模型之前」—— 所以这里不写 OOM 字样（按钮标签里有，够了）。
    plan["suggest"] = ["关掉占 GPU 的程序后重试（界面里有「本程序/其它程序/空闲」三栏）",
                       "点「仍然尝试（可能爆显存）」强制放行一次，真跑不动会自动回滚",
                       "换更小的 .dfm / 调低帧预算",
                       "设 %s=1 用 CPU 跑" % _VRAM_ENV_CPU]
    if _vram_cpu_fallback_env():
        plan.update({"ok": True, "cpu": True, "procs": 1,
                     "frames": int(want_frames if mode != "procs" else 0), "batch": 1,
                     "downgraded": True,
                     "reason": "显存不够但开了 %s → 回落 CPU EP" % _VRAM_ENV_CPU})
        plan["frame_budget"] = int(plan["frames"])
        plan["msg"] = ("%s：显存不够（需要 %.2f GB / 可用 %.2f GB）→ ★ 当前在 CPU 上跑 ★"
                       "（慢约 %.0f 倍；不设 %s 时这里会明确拒绝）"
                       % (label, plan["need_mb"] / 1024.0, avail / 1024.0,
                          _VRAM_CPU_SLOW, _VRAM_ENV_CPU))
        return plan
    if opts.get("force"):
        # 用户明确要试（拒绝面板上的「仍然尝试（可能 OOM）」）：按**最小档**放行。
        # 这不是把闸门拆了 —— 只是把「我替你决定」换成「你自己决定」：
        # 该记的账全都留在 plan 上（forced=True、差多少 MB），真 OOM 了由
        # _vram_oom_rollback 把现场收干净并回到拒绝态（见 routes 里的 OOM 分支）。
        plan.update({"ok": True, "cpu": False, "forced": True, "downgraded": True,
                     "peak_mb": round(float(floor["need_mb"]), 1),
                     "reason": "用户选择强制尝试：按最小档（帧 %d / 批 %d / 进程 %d）放行，"
                               "可能跑不动" % (int(floor["frames"]), int(floor["batch"]),
                                               int(floor["procs"]))})
        plan["msg"] = ("%s：强制尝试 —— 需要 %.2f GB / 可用 %.2f GB（差 %.0f MB），"
                       "按最小档放行（帧 %d / 批 %d）；真跑不动会自动回滚到拒绝态"
                       % (label, float(plan["need_mb"]) / 1024.0, avail / 1024.0,
                          max(0.0, float(plan["need_mb"]) - float(avail)),
                          int(plan["frames"]), int(plan["batch"])))
        plan["summary"] = _vram_summary_line(plan)
        print("[vram] 强制尝试 %s：需要 %.0f MB / 可用 %.0f MB（预留 %.0f MB）→ 放行最小档"
              % (label, float(plan["need_mb"]), float(avail),
                 float(plan.get("reserve_mb") or 0.0)), flush=True)
        return plan
    plan["reason"] = (("连 1 个 worker 都要 %.0f MB > 可用 %.0f MB"
                       % (floor["need_mb"], avail)) if mode == "procs" else
                      ("连最小档（帧 %d / 批 %d）都要 %.0f MB > 可用 %.0f MB"
                       % (floor["frames"], floor["batch"], floor["need_mb"], avail)))
    return plan


def _vram_refuse_extra(plan: dict, what: str = "") -> dict:
    """拒绝时的**结构化字段**（HTTP 响应体顶层）。

    前端只认这几个数就能写出「显存不足：需要 2.44 GB，可用 0.60 GB」这一行，
    不用去正则抠中文；vram 里那份完整计划留给排障和 status 回显。

    口径：
      need_mb   最小档真跑起来的预计峰值（和拒绝消息里的「需要 X GB」同一个数）
      avail_mb  预算（= 空闲 × (1-安全系数) - 预留），也就是「可用 Y GB」
      free_mb   探测到的空闲显存（伪造时是 DFL_VRAM_FREE_MB 那个值）
      suggest   建议列表（第一条就是「关掉占 GPU 的程序后重试」）
    """
    label = str(what or plan.get("label") or "")
    suggest = [str(s) for s in (plan.get("suggest") or [])
               if str(s or "").strip()]
    if not suggest:
        suggest = ["关掉占 GPU 的程序后重试"]
    def _num(key):
        v = plan.get(key)
        return None if v is None else round(float(v or 0.0), 1)
    return {
        "kind": "vram",
        "label": label,
        "need_mb": round(float(plan.get("need_mb") or 0.0), 1),
        "avail_mb": (None if plan.get("avail_mb") is None
                     else round(float(plan.get("avail_mb") or 0.0), 1)),
        "free_mb": (None if plan.get("free_mb") is None else int(plan.get("free_mb") or 0)),
        "total_mb": (None if plan.get("total_mb") is None else int(plan.get("total_mb") or 0)),
        # 显存去向（前端可以直接把「谁占的」写进提示；本程序那部分是可回收的）
        "own_mb": _num("own_mb"), "own_inproc_mb": _num("own_inproc_mb"),
        "own_preview_mb": _num("own_preview_mb"),
        "reclaimable_mb": _num("reclaimable_mb"), "reclaimed_mb": _num("reclaimed_mb"),
        "other_mb": _num("other_mb"), "preview_models": list(plan.get("preview_models") or []),
        # 预留是**显式**的一个数（不再是「20% + 512MB」那种让用户猜的口径）
        "reserve_mb": _num("reserve_mb"), "reserve_min_mb": int(plan.get("reserve_min_mb") or 0),
        "reserve_pct_mb": _num("reserve_pct_mb"),
        "safety": _num("safety"),
        # 一行话（前端原样显示：需要 / 空闲 − 预留 = 可用 / 本程序·其它程序·空闲）
        "summary": _vram_summary_line(plan),
        # 拒绝面板上那颗「仍然尝试（可能 OOM）」：前端看到它才亮按钮
        "can_force": True, "forced": bool(plan.get("forced")),
        "procs": int(plan.get("procs") or 1),
        "frames": int(plan.get("frames") or 0),
        "batch": int(plan.get("batch") or 0),
        "suggest": suggest,
        "vram": dict(plan),
    }


def _vram_summary_line(plan: dict) -> str:
    """**一行**说清这笔账（前端 toast、拒绝面板、日志都读它，不再自己拼中文）。

    形状（用户口径，这一轮定的）：
        需要 2.46 GB · 空闲 3.40 GB 里预留 0.25 GB 给驱动/碎片（max(256MB, 空闲×5%)）
        = 实际可用 3.15 GB · 本程序 0.01 GB / 其它程序 4.78 GB / 空闲 3.40 GB

    为什么非要挤在一行：老的文案把「显存去向」另起了一句，而前端 mgVramLine 只认
    need_mb/avail_mb 两个数 —— 结构化字段**根本没被显示出来**，用户看到的永远只有
    「需要 X，可用 Y」，当然不知道那 4.6GB 是谁占的、也不知道 1.37GB 是预留。
    """
    plan = dict(plan or {})
    free = plan.get("free_mb")
    avail = float(plan.get("avail_mb") or 0.0) / 1024.0
    need = float(plan.get("need_mb") or 0.0) / 1024.0
    reserve = float(plan.get("reserve_mb") or 0.0) / 1024.0
    pct = int(round(float(plan.get("safety") or 0.0) * 100))
    floor_mb = int(plan.get("reserve_min_mb") or _VRAM_RESERVE_MB)
    bits = ["需要 %.2f GB" % need]
    if free is not None:
        bits.append("空闲 %.2f GB 里预留 %.2f GB 给驱动/碎片（规则 %s）"
                    " = 实际可用 %.2f GB" % (float(free) / 1024.0, reserve,
                                             _vram_reserve_rule_text(), avail))
    else:
        bits.append("实际可用 %.2f GB（显存探不到）" % avail)
    own = plan.get("own_mb")
    other = plan.get("other_mb")
    if own is not None or other is not None:
        bits.append("本程序 %.2f GB / 其它程序 %s / 空闲 %s"
                    % (float(own or 0.0) / 1024.0,
                       ("?" if other is None else "%.2f GB" % (float(other or 0.0) / 1024.0)),
                       ("?" if free is None else "%.2f GB" % (float(free) / 1024.0))))
    # 回收预览模型的结果也进这一行：用户看 toast 就得知道「那 2.7GB 是我们自己的」，
    # 以及**为什么没腾出来**（只说 0 = 什么也没说）。
    _rec = dict(plan.get("reclaim") or {})
    _got = float(plan.get("reclaimed_mb") or 0.0)
    _why2 = str((_rec or {}).get("why") or "").strip()
    if _rec and not _rec.get("did") and float(plan.get("reclaimable_mb") or 0.0) <= 0.0:
        _rec = {}                       # 没有常驻模型可收 = 正常，不进 summary
    if _got > 0 and not _why2:
        bits.append("（已回收预览模型 %.2f GB）" % (_got / 1024.0))
    elif _rec and not _rec.get("did"):
        bits.append("（预览模型 %.2f GB 回收失败：%s）"
                    % (float(plan.get("reclaimable_mb") or 0.0) / 1024.0,
                       (_why2 or "没有原因")[:80]))
    elif _rec and _why2:
        bits.append("（预览模型只腾出 %.2f GB：%s）" % (_got / 1024.0, _why2[:80]))
    if plan.get("forced"):
        bits.append("（强制尝试）")
    return " · ".join(bits)


def _vram_refuse_msg(plan: dict, what: str = "这个任务") -> str:
    """拒绝消息：**必须带数字**（需要多少 / 空闲多少 / 预留多少 / 可用多少 / 怎么改）。

    第一句就是 _vram_summary_line 那一行（前端原样显示），后面才是拆项与建议 ——
    用户口径：「把预留写出来，别让我猜」。
    """
    what = str(what or plan.get("label") or "这个任务")
    need = float(plan.get("need_mb") or 0.0) / 1024.0
    avail = float(plan.get("avail_mb") or 0.0) / 1024.0
    free = plan.get("free_mb")
    nproc = int(plan.get("requested_procs") or 1)
    parts = ["模型 %.2f GB × %d 份" % (float(plan.get("model_mb") or 0.0) / 1024.0,
                                      max(1, min(nproc, int(plan.get("procs") or 1)))),
             "CUDA context %.2f GB × %d"
             % (int(plan.get("ctx_mb") or _VRAM_CTX_MB) / 1024.0,
                max(1, min(nproc, int(plan.get("procs") or 1)))),
             "检测/遮罩 %.2f GB" % (float(plan.get("aux_mb") or 0.0) / 1024.0)]
    if float(plan.get("sess_extra_mb") or 0.0) > 0:
        parts.append("运行时工作集 %.2f GB × %d（%s）"
                     % (float(plan.get("sess_extra_mb") or 0.0) / 1024.0,
                        max(1, min(nproc, int(plan.get("procs") or 1))),
                        plan.get("sess_src") or "标定值"))
    if int(plan.get("frames") or 0):
        parts.append("帧缓冲 %d×%.2f MB"
                     % (int(plan.get("frames") or 0),
                        float(plan.get("per_frame_mb") or 0.0)))
    parts.append("dfm 批（最小档）%.2f GB" % (float(plan.get("batch_mb") or 0.0) / 1024.0))
    # ---- 显存去向的补注（归属那一行在 _vram_summary_line 里，前端/toast 都显示它）----
    rec_left = float(plan.get("reclaimable_mb") or 0.0) / 1024.0
    got_back = float(plan.get("reclaimed_mb") or 0.0) / 1024.0
    src = ""
    if got_back > 0:
        src = "（本轮已经回收预览模型 %.2f GB）" % got_back
    elif rec_left > 0:
        src = "（预览 worker 忙，这 %.2f GB 还没回收掉 —— 导出前会再收一次）" % rec_left
    return ("%s：显存不够 —— %s。"
            "拆项：%s。%s"
            "建议：① 关掉占 GPU 的程序（浏览器硬件加速/游戏/另一个导出）后重试；"
            "② 点「仍然尝试（可能爆显存）」强制放行一次（真跑不动会自动回滚，不会把后端搞坏）；"
            "③ 换更小的 .dfm，或把「帧预算」调低（请求的是 %d 帧）；"
            "④ 设 %s=1 改用 CPU 跑（慢约 %.0f 倍，但能出片）。"
            % (what, _vram_summary_line(plan), " + ".join(parts), src,
               int(plan.get("requested_frames") or 0),
               _VRAM_ENV_CPU, _VRAM_CPU_SLOW))


def _vram_log_lines(plan: dict, tag: str = "") -> list:
    """把一份计划摊成几行日志（降档逐档记账，拒绝能看出差多少）。"""
    tag = str(tag or plan.get("label") or "显存")
    src = str(plan.get("source") or "?")
    if plan.get("free_mb") is None:
        return ["%s：显存探不到（%s）→ %s" % (tag, src, plan.get("reason") or "不降档")]
    lines = ["%s：模型 参数量 %s · 权重 %.0f MB（%s）· 输入 %s px（%s）· 非权重 %.1f MB"
             % (tag, ("%.1fM" % float(plan.get("params_m")))
                if plan.get("params_m") else "?",
                float(plan.get("model_mb") or 0), str(plan.get("model_mb_src") or "?"),
                plan.get("size_px") or "?", str(plan.get("size_px_src") or "?"),
                float(plan.get("non_weight_mb") or 0.0)),
             "%s：显存 空闲 %s / 共 %s MB（%s）· 预算 %.0f MB = 空闲 − 预留 %.0f MB"
             "（max(%d MB, 空闲×%d%%)）+ 可回收 %.0f MB"
             % (tag, plan.get("free_mb"), plan.get("total_mb") or "?", src,
                float(plan.get("avail_mb") or 0.0),
                float(plan.get("reserve_mb") or 0.0),
                int(plan.get("reserve_min_mb") or _VRAM_RESERVE_MB),
                int(round(float(plan.get("safety") or 0.0) * 100)),
                float(plan.get("reclaim_left_mb") or 0.0)),
             # 显存去向：谁占的（本程序/其它程序）。用户那句「我什么都没跑」就是靠这一行回答
             "%s：显存去向 本程序 %.0f MB（本进程 %.0f + 预览 worker %.0f）· 其它程序 %s MB"
             " · 可回收 %.0f MB%s"
             % (tag, float(plan.get("own_mb") or 0.0),
                float(plan.get("own_inproc_mb") or 0.0),
                float(plan.get("own_preview_mb") or 0.0),
                ("?" if plan.get("other_mb") is None else int(plan.get("other_mb"))),
                float(plan.get("reclaimable_mb") or 0.0),
                ("" if not plan.get("reclaimed_mb")
                 else "（已回收 %.0f MB / %.0fms）"
                      % (float(plan.get("reclaimed_mb") or 0.0),
                         float(plan.get("reclaim_ms") or 0.0)))),
             "%s：需求拆项 模型 %.0f + context %d%s%s + 检测/遮罩 %.0f MB/份"
             " + 帧 %d×%.2f + dfm 批 %s MB（ORT arena %d MB 含在预留里）"
             % (tag, float(plan.get("model_mb") or 0), int(plan.get("ctx_mb") or 0),
                ("（本进程已持有，不计入）" if plan.get("resident_mb") else ""),
                ((" + 运行时工作集 %.0f（%s）" % (float(plan.get("sess_extra_mb") or 0),
                                                 plan.get("sess_src") or ""))
                 if float(plan.get("sess_extra_mb") or 0) > 0 else ""),
                float(plan.get("aux_mb") or 0),
                int(plan.get("frames") or 0), float(plan.get("frame_mb") or 0.0),
                plan.get("batch_mb") if plan.get("batch_mb") is not None else "?",
                int(plan.get("arena_mb") or 0))]
    # ---- 回收结果单独一行：成功要能看见「卸了多少 / 空闲涨到哪 / 花了多久」，
    #      失败**必须写出原因**（用户口径：不能只打一个 0）----
    _rec = dict(plan.get("reclaim") or {})
    # 「worker 里本来就没有常驻模型」不是故障：那一栏空着，别写成「没回收掉」吓人。
    if _rec and not _rec.get("did") and float(plan.get("reclaimable_mb") or 0.0) <= 0.0 \
            and not _rec.get("freed_mb"):
        _rec = {}
    if _rec:
        _freed = float(_rec.get("freed_mb") or 0.0)
        _why = str(_rec.get("why") or "").strip()
        _rb = _rec.get("before_mb")
        _ra = _rec.get("after_mb")
        # 形状只有一个：「回收预览模型 <freed> MB（<ms>ms · …）：空闲 A → B MB · 附注」
        # —— 人眼和验收脚本的正则都只认这一种，别按分支换写法。
        if _rec.get("did"):
            _tail = (" · " + _why) if _why else (
                (" · " + str(_rec.get("note") or "")) if _rec.get("note") else "")
            lines.insert(2, "%s：回收预览模型 %.0f MB（%.0fms%s）：空闲 %s → %s MB%s"
                         % (tag, _freed, float(_rec.get("ms") or 0.0),
                            " · 进程已回收" if _rec.get("recycled") else "",
                            "?" if _rb is None else int(_rb),
                            "?" if _ra is None else int(_ra), _tail))
        else:
            lines.insert(2, "%s：预览 worker 的模型没回收掉 —— %s"
                         "（可回收 %.0f MB 已按「计入可用」算，导出起手时会再收一次）"
                         % (tag, _why or "没有原因（worker 没回 why）",
                            float(plan.get("reclaim_left_mb")
                                  if plan.get("reclaim_left_mb") is not None
                                  else plan.get("reclaimable_mb") or 0.0)))
    if plan.get("mode") == "procs":
        lines.append("%s：计划 %d 个 worker × %d 线程（请求 %d × %d）· 预计峰值 %.0f MB"
                     % (tag, int(plan.get("procs") or 1), int(plan.get("batch") or 1),
                        int(plan.get("requested_procs") or 1),
                        int(plan.get("requested_batch") or 1),
                        float(plan.get("peak_mb") or plan.get("need_mb") or 0)))
    else:
        lines.append("%s：计划 帧预算 %d / dfm 批 %d%s · 预计峰值 %s（可用 %.0f MB）%s"
                     % (tag, int(plan.get("frames") or 0), int(plan.get("batch") or 0),
                        "（已降档）" if plan.get("downgraded") else "",
                        (("%.0f MB" % plan["peak_mb"]) if plan.get("peak_mb") is not None
                         else "装不下（没有一档塞得进预算）"),
                        float(plan.get("avail_mb") or 0.0),
                        ("（会话已在本进程里，%.0f MB 不计入需求）"
                         % float(plan.get("resident_mb") or 0.0))
                        if plan.get("resident_mb") else ""))
    # 逐档记账：拿到「采用」那一档就停（后面的档位这次用不上，刷日志只会淹掉重点）；
    # 全都装不下时把 12 档全列出来 —— 拒绝的理由要看得出「差多少」。
    _stop = False
    for st in (plan.get("steps") or []):
        if _stop:
            break
        lines.append("%s：%s 帧 %s / 批 %s / 进程 %s → 要 %.0f MB %s %.0f MB"
                     % (tag, "采用" if st.get("fit") else "否决",
                        st.get("frames"), st.get("batch"), st.get("procs"),
                        float(st.get("need_mb") or 0),
                        "≤" if st.get("fit") else ">", float(st.get("avail_mb") or 0)))
        _stop = bool(st.get("fit"))
    if plan.get("forced"):
        lines.append("%s：⚠ 强制尝试（用户点了「仍然尝试」）—— 需要 %.0f MB / 可用 %.0f MB，"
                     "差 %.0f MB；真跑不动会自动回滚到拒绝态"
                     % (tag, float(plan.get("need_mb") or 0.0),
                        float(plan.get("avail_mb") or 0.0),
                        max(0.0, float(plan.get("need_mb") or 0.0)
                            - float(plan.get("avail_mb") or 0.0))))
    if plan.get("cpu"):
        lines.append("%s：★ 当前在 CPU 上跑 ★（%s；不设 %s 时这里会明确拒绝）"
                     % (tag, plan.get("reason") or "", _VRAM_ENV_CPU))
    return lines


def _vram_apply(plan: dict, label: str = "", sticky: bool = False) -> dict:
    """落实一份计划：切 CPU/GPU 开关、记进 _VRAM_LAST（status 端点读它）。

    sticky=True 表示「只许开、不许关」：预览每拖一帧都会算一次计划，它没有权力替一个
    正在跑的导出把 CPU 开关关掉（那会让半个任务在 CPU、半个在 GPU）。关这个开关的
    只有「新任务起手时自己算的那一份计划」。
    """
    label = str(label or plan.get("label") or "vram")
    plan["label"] = label
    plan["at"] = round(time.time(), 3)
    if plan.get("cpu") or not sticky:
        _vram_set_cpu(bool(plan.get("cpu")), plan.get("reason") or "")
    with _VRAM_LOCK:
        _VRAM_LAST[label] = dict(plan)
        sig = (label, bool(plan.get("ok")), plan.get("frames"), plan.get("batch"),
               plan.get("procs"), bool(plan.get("cpu")),
               None if plan.get("avail_mb") is None else int(float(plan["avail_mb"]) // 256))
        fresh = _VRAM_LOGGED.get(label) != sig
        if fresh:
            _VRAM_LOGGED[label] = sig
    # 服务端日志留一份账（预览这种每秒都可能问一次的，只在「决定变了」时打）
    if fresh:
        for _ln in _vram_log_lines(plan, label):
            print("[vram] " + _ln, flush=True)
    return plan


def _vram_last(label: str = None) -> dict:
    with _VRAM_LOCK:
        if label:
            return dict(_VRAM_LAST.get(str(label)) or {})
        return {k: dict(v) for k, v in _VRAM_LAST.items()}


def _preview_guard(dfms, label: str = "预览") -> dict:
    """预览的显存闸门（sticky：只许开 CPU，不许关别人的）。

    特例 —— **模型已经常驻在预览 worker 里**时不算增量：那一帧只跑一次前向，不新建
    session，显存本来就已经占着了（这是唯一一种「空闲 0MB 但预览照样能出图」的情形）。
    不认这个特例会误杀：stage6 验收里 4.0 对拍把卡占满的那几分钟，预览是被拒的，
    而它连一个字节都不用再分配。**只有在要新加载模型时才真的需要预算**。
    """
    paths = [str(p) for p in (dfms or []) if str(p or "")]
    try:
        live = _WORKER.alive()
        loaded = {os.path.normcase(str(k)) for k in (_WORKER._models or {})} if live else set()
    except Exception:                                       # noqa: BLE001
        live, loaded = False, set()
    if live and paths and all(os.path.normcase(p) in loaded for p in paths):
        plan = _vram_plan(_vram_models_mb(paths),
                          {"mode": "none", "procs": 1, "label": label})
        plan.update({"ok": True, "cpu": False, "resident": True,
                     "reason": "模型已在预览 worker 里常驻（这一帧不再新占显存）",
                     "msg": "%s：模型已常驻，不再额外占显存（空闲 %s MB / 增量 0 MB）"
                            % (label, plan.get("free_mb"))})
        return _vram_apply(plan, label, sticky=True)
    # 预览从不回收**自己**（worker 里的模型正是它要用的），但可以放掉父进程里那些
    # 「导出留下的」dfm/XSeg 会话 —— 没有任务在跑时那是纯占用（见 _vram_release_inproc）。
    return _vram_guard(paths, {"mode": "none", "procs": 1, "reclaim_inproc": True},
                       label=label, sticky=True, reclaim=False)


def _vram_guard(dfms, opts: dict = None, label: str = "任务",
                sticky: bool = False, reclaim: bool = None, force: bool = None) -> dict:
    """加载模型**之前**的一次预算检查：装得下返回计划，装不下抛 400（消息带数字）。

    sticky=True（预览用）：只在「要在 CPU 上跑」时短路全链路，不允许把别人的 CPU 开关关掉。

    reclaim：算账之前先回收**我们自己**占的显存（常驻预览 worker 里那份 dfm 会话）。
    默认 = 非 sticky，也就是所有导出/阶段闸门都回收，预览自己不回收（它正是要靠那份
    常驻模型才快，而且 _preview_guard 有「已常驻 = 增量 0」的特例）。

    force：用户点了拒绝面板上的「仍然尝试（可能 OOM）」—— 装不下也放行最小档
    （plan.forced=True）。**只放行、不改账**：need/avail/差多少 MB 全留在 plan 上，
    真 OOM 时 _vram_oom_rollback 收现场并回到拒绝态。
    """
    if isinstance(dfms, (str, Path)):
        paths = [str(dfms)]
    else:
        paths = [str(p) for p in (dfms or []) if str(p or "")]
    opts = dict(opts or {})
    if not opts.get("size_px"):
        opts["size_px"] = _dfm_size_hint(paths[0] if paths else "")
    if reclaim is None:
        reclaim = (not sticky)
    if force is None:
        force = bool(opts.get("force"))
    plan = _vram_plan(_vram_models_mb(paths),
                      dict(opts, label=label, reclaim=bool(reclaim), force=bool(force)))
    _vram_apply(plan, label, sticky=bool(sticky))
    if not plan.get("ok"):
        msg = plan.get("msg") or _vram_refuse_msg(plan, label)
        # 拒绝必须有日志（排障第一现场）：这一行进后端 stdout（→ server.log）
        print("[vram] ★ 拒绝 %s ★ %s" % (label, msg), flush=True)
        # 400 带上 need_mb / avail_mb / free_mb / suggest[]（前端只弹一条提示、
        # 不再去正则抠中文），并且**在起 job / 加载模型之前**抛出去。
        raise _BadInput(msg, 400, extra=_vram_refuse_extra(plan, label))
    return plan

def _vram_forced_rollback(label: str, model_mb: float, opts: dict, err) -> dict:
    """强制尝试失败之后：该不该按「确实不够 / 已回滚」收口？要的话返回回滚现场。

    判据两条，满足其一就算**真的不够**：
      ① 报错文本是显存类（ORT / torch / TRT 的说法）；
      ② 把现场收干净、重探一次之后，按当下的空闲**仍然装不下**。

    ② 是必要的兜底：ORT 建 session 失败时给的异常五花八门（有的只有 "CUDA failure"），
    只看文本会漏判，用户就只看到一句「导出失败」，而真相是「这块卡确实不够」。
    反过来说，如果重探之后其实装得下（内存被别的程序还回来了 / 是我们自己的 bug），
    这里**不**改写成「显存不够」—— 那种错该老实报错。
    """
    rb = _vram_oom_rollback(label, model_mb, opts, err)
    if _vram_oom_is_text(err):
        return rb
    if not bool((rb.get("plan") or {}).get("ok")):
        return rb
    return {}


def _vram_oom_inject_on() -> bool:
    """验收专用：DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1 时，_dfm_session 立刻抛 OOM。

    为什么需要这个钩子：WDDM 下「显存不够」通常**不会** OOM —— 驱动会把别人的分配换页
    出去，本机实测把空闲压到 600MB 再 force=1 竟然还能出片（慢，但不失败）。要让
    「强制尝试 → OOM → 回滚 → 还能再试」这条路可复现，只能注入一次 OOM。
    双开关是照 _vram_free_mb.txt 那条规矩来的：生产环境不读任何测试钩子。
    """
    if str(os.environ.get("DFL_ALLOW_TEST_HOOK") or "").strip() not in ("1", "true", "yes", "on"):
        return False
    return str(os.environ.get("DFL_VRAM_OOM_INJECT") or "").strip() in ("1", "true", "yes", "on")


def _vram_reserve_rule_text() -> str:
    """预留这条规则的一句话（文案/日志共用，避免口径和数字对不上）。"""
    if str(os.environ.get("DFL_VRAM_RESERVE_MODE") or "").strip().lower() in ("legacy",
                                                                             "legacy20", "old"):
        return "老口径 空闲×20% + 512MB"
    return "max(256MB, 空闲×5%)"


def _vram_oom_is_text(text) -> bool:
    """这段文字/异常是不是「真的没显存了」（ORT / torch / TRT 的说法都认）。"""
    s = str(text or "").lower()
    return any(k in s for k in ("out of memory", "cuda_error_out_of_memory",
                               "cudaerrormemoryallocation", "failed to allocate",
                               "insufficient memory", "resource_exhausted",
                               "cudnn_status_alloc_failed"))


def _vram_oom_rollback(label: str = "任务", model_mb: float = 0.0, opts: dict = None,
                       err="") -> dict:
    """强制尝试撞上 OOM 之后：**干净地**回到拒绝态（用户那条「别把后端搞坏」）。

    四件事，顺序不能反：
      ① 放掉本进程的 dfm / XSeg 会话 + gc + empty_cache —— OOM 之后 ORT 的 arena
         可能还挂在那个建到一半的 session 上，不清掉下一次连预览都抢不到显存；
      ② 重新探一次空闲（fresh，不吃 2~4s 缓存），按**当下的数**重算一份拒绝计划
         （不是拿 OOM 之前那份旧账忽悠用户）；
      ③ 一句话说清「确实不够 / 已回滚 / 差多少 / 可以再试」；
      ④ 别人的东西一概不动：预览 worker 不重载、后台进程不杀 —— 所以下次还能再试。

    返回的 dict 会挂到 /export/status 与 /stage/status 的 vram_oom 键上（前端据此
    把界面从「进度页」搬回「被拒态」并把 current 的数字重画一遍）。
    """
    rel = {}
    try:
        rel = _vram_release_inproc("强制尝试 OOM 回滚")
    except Exception as e:                                  # noqa: BLE001
        print("[vram] OOM 回滚：放掉本进程会话失败：%s" % str(e)[:160], flush=True)
    try:
        import gc
        gc.collect()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:                                       # noqa: BLE001
        pass
    o2 = dict(opts or {})
    for k in ("force", "free_mb", "total_mb", "reclaim", "reclaim_inproc", "resident"):
        o2.pop(k, None)          # 回滚这一拍：不许再强制，也不许拿一份旧 free 充数
    o2.update({"label": label, "reclaim": False, "force": False})
    try:
        plan = _vram_plan(float(model_mb or 0.0), o2)
    except Exception as e:                                  # noqa: BLE001
        plan = {"ok": False, "need_mb": None, "avail_mb": None,
                "reason": "重算预算失败：%s" % str(e)[:160], "msg": ""}
    try:
        free = plan.get("free_mb")
        summary = "强制尝试失败：%s" % _vram_summary_line(plan)
        gap = max(0.0, float(plan.get("need_mb") or 0.0) - float(plan.get("avail_mb") or 0.0))
        msg = ("%s：强制尝试失败了 —— 确实不够（运行中 OOM：%s）。"
               "已回滚：放掉本进程会话 %s，空闲现在 %s MB（%s）。差 %.0f MB，"
               "关掉占 GPU 的程序后可以再点一次。"
               % (label, str(err or "").strip()[:200] or "显存分配失败",
                  "、".join(list((rel or {}).get("released") or [])[:3]) or "无（本来就没有）",
                  ("?" if free is None else int(free)), summary, gap))
    except Exception:                                       # noqa: BLE001
        summary, msg = "强制尝试失败：显存确实不够，已回滚", "强制尝试失败：显存确实不够，已回滚"
    print("[vram] ★ %s ★" % msg, flush=True)
    return {"ok": False, "kind": "vram_oom", "label": label,
            "released": list((rel or {}).get("released") or []),
            "need_mb": plan.get("need_mb"), "avail_mb": plan.get("avail_mb"),
            "free_mb": plan.get("free_mb"), "total_mb": plan.get("total_mb"),
            "reserve_mb": plan.get("reserve_mb"), "own_mb": plan.get("own_mb"),
            "other_mb": plan.get("other_mb"), "can_force": False,
            "summary": summary, "msg": msg, "error": msg,
            "suggest": ["关掉占 GPU 的程序后重试", "换更小的 .dfm / 调低帧预算",
                        "设 %s=1 用 CPU 跑" % _VRAM_ENV_CPU],
            "plan": dict(plan)}

_PRED_VRAM_SAFETY = 0.20      # 留 20% 余量：探测量打八折才是敢用的预算


def _pred_pick_batch(cfg: dict, ent) -> tuple:
    """挑 batch：请求里写了就照用，没写就**把阶梯真跑一遍再决定**（不写死、也不猜）。

    为什么不能写死 8，也不能只按显存估算：
      · 8GB 卡上 8 是个好值，但同一张卡只剩 1.2GB 空闲时 8 就是一路 OOM 退到 1；
      · 反过来，实测这台机上 batch 16/32 的每张耗时是 batch 1 的 5 倍以上
        （ORT 的 arena 撑爆后整个进程都变慢，之后不再恢复）—— 光看「可用显存够不够」
        会一路选到 32，然后整条链路崩掉。所以**耗时也要量**。

    阶梯规则（全部实测，约 1s 的预热，同时也是真正的 warm-up）：
      1. 依次跑 batch = 1 / 4 / 8 的空张量，记下「每张耗时」和「可用显存」；
      2. 选「每张耗时 ≤ 最快值 × 1.05」里最大的那个 batch（本机：1/4/8 → 选 8）；
      3. 只有在「8 比 1 快 2 倍以上（说明是小模型 + 大卡，批量真的有用）+ 还剩 ≥6GB
         + 下一档塞得进 80% 预算」三条同时成立时，才继续试 16 / 32。
         本机上 8 只比 1 快 1.3×，而 16 会把 8GB 顶到 7806/8151 MB 且并不更快，
         还会让 ORT 的 arena 撑爆、之后整个进程掉到 15 张/秒 —— 所以老实停在 8；
      4. 每一步都按实测单张开销核对 80% 显存预算，不够就停。
    运行中还有两道保险：OOM → 8→4→2→1 退档；比历史最好慢 60% → 主动退半（见 _flush）。

    返回 (batch, {"source","ladder","free_mb","total_mb","per_item_mb","best_per_face_ms"})
    """
    want = int(cfg.get("pred_batch") or 0)
    if want > 0:
        return max(1, min(64, want)), {"source": "request", "batch": int(want),
                                       "ladder": [], "free_mb": None, "total_mb": None,
                                       "per_item_mb": None, "best_per_face_ms": None}
    if _vram_cpu_on():
        # CPU 上跑还探阶梯没意义（每档都慢 17 倍，而且没有显存可省）：直接 1 张。
        return 1, {"source": "cpu", "batch": 1, "ladder": [], "free_mb": None,
                   "total_mb": None, "per_item_mb": None, "best_per_face_ms": None}
    import numpy as np
    _sess, _name, size, _nchw = ent[0], ent[1], ent[2], ent[3]
    _cap = int(ent[5] or 1) if len(ent) > 5 else 8
    size = int(size)
    free0, total0 = _vram_free_mb()
    # 阶梯不能超过 session 真正能吃的批（原生 TRT 引擎的优化档位可能就是 batch 1）
    ladder = [1]
    for _b in (4, 8):
        if _b <= _cap and _b not in ladder:
            ladder.append(_b)
    rows = []
    best_per = None
    try:
        i = 0
        while i < len(ladder):
            b = min(int(ladder[i]), _PRED_BATCH_CAP, _cap)
            x = np.zeros((b, size, size, 3), np.float32)
            # 每个档位先跑一次不计时：第一次跑某个形状要编 kernel / 选算法，实测能把
            # batch 1 的量测放大到 1000 ms（真值是 36 ms）—— 以前就是拿这个被污染的
            # 数字去选 batch，结果选了更慢的那一档。取两次里快的那次才是稳态。
            _dfm_run_batch(ent, x)
            t0 = time.perf_counter()
            _dfm_run_batch(ent, x)
            dt = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            _dfm_run_batch(ent, x)
            dt = min(dt, (time.perf_counter() - t0) * 1000.0)
            per = dt / b
            free_now, _t = _vram_free_mb()
            rows.append({"batch": b, "ms": round(dt, 1), "per_face_ms": round(per, 2),
                         "free_mb": free_now})
            best_per = per if best_per is None else min(best_per, per)
            # 显存：按实测单张开销核对 80% 预算
            fits = None
            if free_now is not None and total0:
                per_item = max(size * size * 3 * 4 * 6 / 1048576.0,
                               max(0.0, (free0 - free_now)) / float(max(1, b)))
                budget = free_now * (1.0 - _PRED_VRAM_SAFETY)
                fits = int(budget / per_item) if per_item > 0 else b
                rows[-1]["per_item_mb"] = round(per_item, 1)
                rows[-1]["budget_batch"] = fits
                if fits <= b or free_now < 1024:
                    break
            # 大卡才继续往上探（两个条件都要满足）：
            #   ① 8 张/批确实比单张快（GPU 跑满的机器不会踏进 16/32 那个坑）；
            #   ② 下一档还塞得进 80% 显存预算 —— 否则试一次就把显存顶到墙，
            #      整条流水线都会被拖慢（实测这台机 batch 16 每张 40ms → 峰值 7.3GB）。
            if i == len(ladder) - 1 and b >= 8 and best_per is not None:
                nxt = 16
                first = float(rows[0]["per_face_ms"] or 0.0)
                gain = (first / float(rows[-1]["per_face_ms"] or 1.0)) if first else 1.0
                # 越过 8 要三个条件同时成立：批量确实带来 2× 以上的单张收益（说明是
                # 小模型 + 大卡，批量有用）、显存至少还剩 6GB、下一档塞得进 80% 预算。
                # 本机（416 dfm、8GB、批量收益 1.3×）三个都不满足 → 老实停在 8：
                # 实测 batch 16 会把显存顶到墙，之后整个进程每张 40ms → 60ms，越试越慢。
                if gain >= 2.0 and (free_now is None or free_now >= 6144)                         and (fits is None or fits >= nxt):
                    ladder.extend([16, 32])
                else:
                    break
            i += 1
    except Exception as e:                                  # noqa: BLE001
        _stage_log("换脸：batch 探测中断（%s）→ 用已测到的档位" % str(e)[:100])
        if not rows:
            return 4, {"source": "probe-failed", "batch": 4, "ladder": [],
                       "free_mb": free0, "total_mb": total0,
                       "per_item_mb": None, "best_per_face_ms": None}
    if not rows:
        return 4, {"source": "no-vram-info", "batch": 4, "ladder": [],
                   "free_mb": free0, "total_mb": total0,
                   "per_item_mb": None, "best_per_face_ms": None}
    ok = [r for r in rows if r["per_face_ms"] <= best_per * 1.05]
    pick = max(ok or rows[:1], key=lambda r: r["batch"])
    meta = {"source": "auto", "batch": int(pick["batch"]), "ladder": rows,
            "free_mb": free0, "total_mb": total0,
            "per_item_mb": pick.get("per_item_mb"),
            "best_per_face_ms": round(min(r["per_face_ms"] for r in rows), 2),
            "safety": _PRED_VRAM_SAFETY}
    _stage_log("换脸：batch 阶梯 %s → 选 %d"
               % (" / ".join("B%d=%.1fms(%.2fms/张)" % (r["batch"], r["ms"], r["per_face_ms"])
                             for r in rows), int(pick["batch"])))
    return int(pick["batch"]), meta


def _is_oom(e) -> bool:
    s = str(e).lower()
    return ("allocate" in s or "out of memory" in s or "memory" in s and "insufficient" in s)


def _gpu_mem_used_mb():
    """当前显存占用（MB）。拿不到就 None —— 只在验收/报告里用，不影响主流程。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        vals = [int(x) for x in (out.stdout or "").split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:                                       # noqa: BLE001
        return None

_SESS_LOCK = threading.Lock()
_SESS_BUSY = {}          # kind -> 上一轮实测的段占用（0..1）：derive 的输入
_SESS_BENCH = {}         # key -> {"gpu_ms","cpu_ms",...}（进程内缓存）
_SESS_PLAN_LAST = {}     # kind -> plan（status 回显）
_DET_EP = {}             # (name, ep) -> 检测器实例（GPU 那份走 app 的缓存）


def _sess_bench_path() -> Path:
    return Path(WORKSPACE) / "_sess_bench.json"


def _sess_bench_load() -> dict:
    try:
        p = _sess_bench_path()
        if p.is_file():
            d = json.loads(p.read_text("utf-8", "ignore") or "{}")
            if isinstance(d, dict):
                return d
    except Exception:                                       # noqa: BLE001
        pass
    return {}


def _sess_bench_put(key: str, row: dict):
    with _SESS_LOCK:
        _SESS_BENCH[str(key)] = dict(row)
    try:
        d = _sess_bench_load()
        d[str(key)] = dict(row)
        _sess_bench_path().write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    except Exception:                                       # noqa: BLE001
        pass


def _sess_bench_get(key: str) -> dict:
    with _SESS_LOCK:
        hit = _SESS_BENCH.get(str(key))
    if hit:
        return dict(hit)
    row = _sess_bench_load().get(str(key)) or {}
    return dict(row) if isinstance(row, dict) else {}


def _sess_ensure(key: str) -> dict:
    """读一遍磁盘缓存（跨进程复用，同一份模型只量一次）。"""
    row = _sess_bench_get(key)
    if row.get("gpu_ms") is not None or row.get("cpu_ms") is not None:
        with _SESS_LOCK:
            _SESS_BENCH[key] = row
    return row


def _detect_for(name, ep="gpu"):
    """按 EP 取一个检测器。GPU 那份复用 app 的缓存（别建第二份显存）；CPU 那份单独建。"""
    key = (str(name), str(ep))
    with _SESS_LOCK:
        hit = _DET_EP.get(key)
    if hit is not None:
        return hit
    det = None
    if ep == "gpu":
        det = _detector(name)            # 复用 app 的缓存（别为同一份权重占第二块显存）
    elif ep == "gpu2":
        try:                             # 第二份 CUDA session：显存够时才能多开一路
            from ..app import _DET_ONNX, _detector_path
            from .._ortcuda import providers as _prov
            canon = _detector_val(name)
            p = _detector_path(_DET_ONNX[canon])
            if p is None:
                return None
            from .._detect import RetinaFaceDet, TinyMogDet
            if canon == "tinymog":
                det = TinyMogDet(str(p), workspace=WORKSPACE, providers=list(_prov()))
            else:
                det = RetinaFaceDet(p, 640)          # 自带 CUDA providers
        except Exception as e:                              # noqa: BLE001
            _stage_log("异构池：第二份 GPU 检测器起不来（%s）" % str(e)[:160])
            return None
    else:
        # CPU 那一路（DFL_DET_EP=cpu|auto 选 CPU 的那档，见 _det_ep_resolve）。
        # 现在的做法是**直接按 CPU EP 造**，两个类都开了会话通道（providers / sess_options）：
        # 以前是「先照原样造好（会顺手加载 TRT 引擎、建一块 CUDA context）、再把 sess 换掉」，
        # 给 CPU 用的检测器白占 5MB 引擎 + 一份 context，而且日志里会冒出
        # 「[TRT] tinymog: 使用 bf16 引擎」这种把人带偏的行。
        # intra_op 线程数：DFL_DET_EP_THREADS（默认 3，夹在 2..4）；老的 DFL_SESS_THREADS
        # 仍然优先（那是池级口径，显式设了就听它的）。
        try:
            from ..app import _DET_ONNX, _detector_path
            from .. import _detect as _dk
            canon = _detector_val(name)
            p = _detector_path(_DET_ONNX[canon])
            if p is None:
                return None
            nthr = _sess.int_env(_sess.ENV_THREADS, 0) or _dk.threads()
            so = _dk.cpu_session_options(nthr)
            from .._detect import RetinaFaceDet, TinyMogDet
            if canon == "tinymog":
                det = TinyMogDet(str(p), workspace=None, providers=["CPUExecutionProvider"],
                                 sess_options=so)
            else:
                det = RetinaFaceDet(p, 640, providers=["CPUExecutionProvider"],
                                    sess_options=so)
            _stage_log("检测：CPU EP · intra_op=%d · inter_op=1 · %s"
                       % (int(nthr), Path(str(p)).name))
        except Exception as e:                              # noqa: BLE001
            _stage_log("异构池：CPU 检测器起不来（%s）" % str(e)[:160])
            return None
    with _SESS_LOCK:
        _DET_EP[key] = det
    return det


def _detect_on(det, frame, thresh: float) -> list:
    """用**指定**的检测器跑一帧（不碰共享实例上的 thresh 字段，可并发）。"""
    det.thresh = float(thresh)
    return det.detect(frame) or []


def _det_prep(det, frame):
    """把检测器的 letterbox 先算出来（decode 线程干，检测线程只留前向 + NMS）。

    检测器没有 prep（老/第三方实现）就返回 None，调用方退回「帧直接丢给 detect」。
    prep 只读 self.input_size、不碰 session，所以可以和安全跑着的 detect 并发。
    """
    fn = getattr(det, "prep", None)
    if fn is None:
        return None
    return fn(frame)


def _detect_on_ctx(det, ctx, thresh: float) -> list:
    """已经 prep 好的上下文 → 人脸列表（自己的 session 用，不加锁）。"""
    det.thresh = float(thresh)
    return det.detect_ctx(ctx) or []


def _detect_boxes_ctx(ctx, detector: str, thresh: float) -> list:
    """共享实例版：和 _detect_boxes 同一把锁、同一个 thresh 口径，只是跳过 resize。"""
    det = _detector(detector)
    with _det_run_lock:
        det.thresh = float(thresh)
        try:
            return det.detect_ctx(ctx) or []
        except Exception as e:                            # noqa: BLE001
            raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)


def _det_quick_profile(det, w: int, h: int, n: int = 3) -> dict:
    """单帧检测的三段实测（1080p 真尺寸零图，n 次取中位）—— 只用来打日志。

    为什么要在**起流水线之前**量这一段：用户报的是「界面选了 TinyMog，日志却 48ms/帧」。
    有这张表就能一眼分开三件事：模型本身多贵（前向）、resize 多贵（预处理）、NMS 多贵；
    剩下的差额就是「在等 GPU/CPU」——那属于整条流水线的账，不属于检测器。
    """
    import time as _t

    import numpy as np

    frame = np.zeros((max(2, int(h or 1080)), max(2, int(w or 1920)), 3), np.uint8)
    out = {"w": int(frame.shape[1]), "h": int(frame.shape[0]), "n": int(max(1, n)),
           "sess": type(getattr(det, "sess", None)).__name__}
    full, prep, fwd, post = [], [], [], []
    try:
        # 先空跑一张：第一次前向可能包含 CUDA Graph 捕获 / torch 张量路径初始化
        # （实测 tinymog 第一张 76ms、之后 2ms）—— 把那次算进「每帧耗时」就是纯误导。
        det.detect(frame)
        for _ in range(max(1, int(n))):
            t0 = _t.perf_counter()
            det.detect(frame)
            t1 = _t.perf_counter()
            full.append((t1 - t0) * 1000.0)
        prep_fn = getattr(det, "prep", None)
        ctx_fn = getattr(det, "detect_ctx", None)
        if prep_fn is not None and ctx_fn is not None:
            for _ in range(max(1, int(n))):
                t0 = _t.perf_counter()
                ctx = prep_fn(frame)
                t1 = _t.perf_counter()
                fw = det._forward(ctx["canvas"])
                t2 = _t.perf_counter()
                ctx_fn(ctx)
                t3 = _t.perf_counter()
                prep.append((t1 - t0) * 1000.0)
                fwd.append((t2 - t1) * 1000.0)
                # 注意：detect_ctx 内部**又跑了一次**前向，所以 post 这一栏 = 前向 + NMS/解码；
                # 报出去时减掉前向的中位数，剩下的才是 NMS/解码本身。
                post.append((t3 - t2) * 1000.0)
        med = lambda v: (sorted(v)[len(v) // 2] if v else None)          # noqa: E731
        out["full_ms"] = None if not full else round(med(full), 2)
        out["prep_ms"] = None if not prep else round(med(prep), 2)
        out["fwd_ms"] = None if not fwd else round(med(fwd), 2)
        out["post_ms"] = (None if not post or not fwd
                          else round(max(0.0, med(post) - med(fwd)), 2))
    except Exception as e:                                            # noqa: BLE001
        out["err"] = str(e)[:160]
    return out


def _detect_bench(name, w: int, h: int) -> dict:
    """检测段的微基准：GPU / CPU 各 2 张取中位（1080p 真尺寸的零图）。

    为什么用 1080p 而不是 640 的方块：这一步里有一大块是 **cv2 的 letterbox resize**
    （1080p → 640），那部分不管在哪个 EP 上都吃 CPU —— 只量模型会把 CPU 那一路
    算得偏快，配比就偏激进。
    """
    import numpy as np
    key = "detect:%s:%dx%d:%dt" % (_detector_val(name), int(w or 1080), int(h or 1920),
                                   _dv_threads())
    row = _sess_ensure(key)
    if row.get("gpu_ms") and row.get("cpu_ms") and row.get("gpu_fwd_ms") is not None:
        return row
    frame = np.zeros((int(h or 1920), int(w or 1080), 3), np.uint8)
    out = {"key": key, "at": round(time.time(), 1)}
    for ep in ("gpu", "cpu"):
        det = None
        try:
            det = _detect_for(name, ep)
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
        if det is None:
            out["%s_ms" % ep] = None
            out["%s_fwd_ms" % ep] = None
            continue
        try:
            ms = _sess.bench(lambda: _detect_on(det, frame, 0.5), n=2, warmup=1)
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
            ms = None
        out["%s_ms" % ep] = ms
        # 前向单独量：DFL_DET_EP=auto 的判据就是这个数（见 server/_detect.py 的 decide）。
        # 为什么不用 gpu_ms（整帧）：那里面有 letterbox(0.4~0.9ms)+NMS/解码(0.1ms)，
        # 这部分两个 EP 完全一样，混进去只会把「模型本身多贵」稀释掉。
        # warmup=1 必须留着：TRT 的第一张要捕获 CUDA Graph（16ms），算进去就是假数。
        try:
            ctx = det.prep(frame) if hasattr(det, "prep") else None
            out["%s_fwd_ms" % ep] = (None if ctx is None else
                                     _sess.bench(lambda: det._forward(ctx["canvas"]),
                                                 n=3, warmup=1))
        except Exception as e:                              # noqa: BLE001
            out["%s_err" % ep] = str(e)[:160]
            out["%s_fwd_ms" % ep] = None
    _sess_bench_put(key, out)
    return out


def _dv_threads() -> int:
    """CPU 检测会话的 intra_op 线程数（缓存键带上它：线程数一改，cpu_ms 就不是同一个数）。"""
    try:
        from .. import _detect as _dk
        nthr = _sess.int_env(_sess.ENV_THREADS, 0) or _dk.threads()
        return int(nthr)
    except Exception:                                       # noqa: BLE001
        return 0


def _det_ep_resolve(det_name, w=None, h=None) -> dict:
    """这一单检测走哪个 EP → {"ep","why","gpu_fwd_ms","cpu_fwd_ms","thr","src","mode","bench"}。

    判据与阈值在 server/_detect.py 的 decide()（DFL_DET_EP=auto 时用实测的前向 ms，
    量不到就查已知表）。这里只负责「把实测喂进去 + 把结论摆成字典」，不自己拍数。
    """
    b = _detect_bench(det_name, w or 1920, h or 1080)
    from .. import _detect as _dk
    pol = _dk.decide(det_name, b.get("gpu_fwd_ms"),
                     gain_ok=_sess.cpu_ep_ok("detect", WORKSPACE),
                     gain_pct=_sess.cpu_ep_gain("detect", WORKSPACE))
    pol["bench"] = b
    pol["cpu_fwd_ms"] = b.get("cpu_fwd_ms")
    pol["name"] = _detector_val(det_name)
    return pol


def _det_ep_line(pol) -> str:
    """一行日志：选了哪个 EP + 为什么 + 两边的实测数字。"""
    if not pol:
        return "-"
    return ("检测 EP：%s · %s · GPU 前向 %s ms / CPU 前向 %s ms（整帧 GPU %s / CPU %s ms）"
            % (str(pol.get("ep") or "?").upper(), "；".join(pol.get("why") or []),
               (("%.2f" % pol["gpu_fwd_ms"]) if pol.get("gpu_fwd_ms") is not None else "未量到"),
               (("%.2f" % pol["cpu_fwd_ms"]) if pol.get("cpu_fwd_ms") is not None else "未量到"),
               (pol.get("bench") or {}).get("gpu_ms"),
               (pol.get("bench") or {}).get("cpu_ms")))


def _det_ep_apply(plan: dict, pol: dict, over=None) -> dict:
    """把「检测走哪个 EP」落进配比。

    · ep=cpu：这个检测器**整段**挪出 GPU 流 → 池里就只留 CPU 那一路（gpu=0, cpu=1）。
      为什么不是「两边各一份、谁快谁上」：检测是逐帧串行的（一张脸一帧），
      两路只会让同一帧的 letterbox/NMS 做两遍，而 GPU 那一路正是要躲开的东西。
    · ep=gpu：不动 GPU 份数（derive 已经按显存/边际算过了），但**清零 CPU 份数** ——
      「政策说要 GPU」时再挂一路 CPU 检测器就是把同一帧分回 GPU 队列之外，
      端到端反而多一份排队（这就是这次实验 A 档的口径）。
      请求体显式给了 sess_cpu（比如做实验）就尊重它，不清零。
    """
    why = plan.setdefault("why", [])
    ep = str((pol or {}).get("ep") or "gpu")
    plan["det_ep"] = dict(pol or {})
    if ep == "cpu":
        if int(plan.get("gpu") or 0) != 0 or int(plan.get("cpu") or 0) != 1:
            why.append("检测 EP=CPU → 池改成 0×GPU + 1×CPU（原 %d×GPU + %d×CPU）"
                       % (int(plan.get("gpu") or 0), int(plan.get("cpu") or 0)))
        plan["gpu"] = 0
        plan["cpu"] = 1
    else:
        if int(plan.get("gpu") or 0) < 1:
            why.append("检测 EP=GPU → 至少 1 份 GPU 检测会话")
            plan["gpu"] = 1
        exp_cpu = (over or {}).get("cpu")
        if not (isinstance(exp_cpu, str) and exp_cpu.strip()) and exp_cpu not in (0,) \
                and int(plan.get("cpu") or 0) > 0:
            why.append("检测 EP=GPU → CPU 那几路（%d 份）清零，别让同一帧又排到 CPU"
                       % int(plan.get("cpu") or 0))
            plan["cpu"] = 0
    return plan


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 vram 与 _app_mod、_engine_root、_sess、app、compose、fused、preview、stages
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from .._paths import engine_root as _engine_root

from .. import _sess

from ..app import (
    WORKSPACE, _merge_job,
)

from .compose import (
    _det_run_lock, _detector, _detector_val,
)

from .pipeline_fused import (
    _FUSED_BATCH_TARGET, _FUSED_BUDGETS, _FUSED_ENV_VRAM,
)

from .preview import (
    _WORKER,
)

from .pipeline_stages import (
    _DFM_SESS, _PRED_LOCK, _STAGE_JOB, _XSEG_SESS, _XSEG_SESS_LOCK, _dfm_run_batch, _stage_log,
)
