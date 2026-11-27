# -*- coding: utf-8 -*-
"""标准 DFL 项目逐帧导出。（上半是 _proj_pipe.py 原身：项目流水线；下半从 routes_merge 切出来：导出作业 _proj_*）

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations
import numpy as np

import collections
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from .._common import encoder_sel as _esel
from .._common.http import (
    _BadInput,
)
from .._common.proc import (
    _kill_proc_tree,
)
from ._shared import (
    _norm_path,
)


# （已无开关：老路删除后只剩流水线实现，见 docs/21-PROJ-PIPELINE.md）


# 合成的画布：aligned（默认，image_to_face_mat 原生画布）| facelib（68 点现算 WHOLE_FACE）。
# 兼容老写法：值全是数字时不当模式用，仍旧当「合成线程数」。
_ENV_COMPOSE = "DFL_PROJ_COMPOSE"


_ENV_COMPOSE_N = "DFL_PROJ_COMPOSE_THREADS"   # 合成线程数（默认 4）


_ENV_READ = "DFL_PROJ_READ"              # 读帧/对齐线程数（默认 3）


_ENV_ENC = "DFL_PROJ_ENC"                # 编码线程数（默认 2）


_ENV_DFM_BATCH = "DFL_PROJ_DFM_BATCH"    # dfm 攒批目标（默认 max(计划 batch, 4)）


_ENV_WAIT = "DFL_PROJ_BATCH_WAIT_MS"     # 攒批超时（默认 20ms）


_ENV_PNG = "DFL_PROJ_PNG_LEVEL"          # .png 帧的压缩级别（默认见 _PNG_LEVEL_DEFAULT）


# .png 帧的压缩级别默认值 = 1 + Z_RLE（和 ⑤ 合成阶段、快速导出同档）。
#
# 改之前的默认是 3 且**不带策略**（裸 cv2.imencode），实测是本仓最贵的一次编码：
# 1080×1920 单线程 6 帧中位 ——
#     level 3 · 默认策略  144.3 ms/帧  · 1.249 MB
#     level 3 · Z_RLE      75.7 ms/帧  · 1.231 MB
#     level 1 · 默认策略   74.5 ms/帧  · 1.371 MB
#     level 1 · Z_RLE      65.4 ms/帧  · 1.231 MB
#     level 0              40.2 ms/帧  · 5.944 MB（体积翻 4.8 倍，不划算）
# PNG 无损，解码出来的像素逐位相同（8 档全部实测 np.array_equal），只有文件字节变；
# 这一档**又快又小**（比 level 3 默认策略少 55% 时间、还小 1.4%），所以直接做成默认。
# 老行为仍然可复现：DFL_PROJ_PNG_LEVEL=3（策略固定 RLE，只是 level 不同）。
_PNG_LEVEL_DEFAULT = 1


_QDEPTH = 24            # 段间队列深度（和 fused 同口径）


_BATCH_WAIT_MS = 20.0   # 攒批超时：正常出片时到脸间隔 30~80ms


_DFM_BATCH_MIN = 4      # 攒批下限（一次前向至少 4 张）


_DFM_BATCH_MAX = 8      # 攒批上限（再多每张反而更贵，见 20-PROJ-ADAPTIVE 的阶梯）


# 画布模式：只看 DFL_PROJ_COMPOSE 里的字符串值。数字（老写法）不算模式。
_COMPOSE_FACELIB = ("facelib", "face", "legacy", "old", "off", "false", "no")


_COMPOSE_ALIGNED = ("aligned", "align", "native", "new", "on", "true", "yes", "1")


def compose_mode() -> str:
    """合成的画布：aligned（默认）| facelib。

    aligned  = 画布就是 aligned/ 里那张对齐脸（DFLJPG 的 image_to_face_mat），
               整条链只有一张画布，贴回用它的逆 —— 少一次重采样，也不存在
               「68 点 → WHOLE_FACE 矩阵」这一步。
    facelib  = 老行为：用 68 点现算 facelib 的 WHOLE_FACE 矩阵，pred 与贴回都在
               那张画布上（A/B、排障用）。
    """
    v = str(os.environ.get(_ENV_COMPOSE) or "").strip().lower()
    if v in _COMPOSE_FACELIB:
        return "facelib"
    if v in _COMPOSE_ALIGNED:
        return "aligned"
    return "aligned"                   # 默认；数字/空/认不出来的都归这里


def compose_threads(default=4) -> int:
    """合成线程数。DFL_PROJ_COMPOSE_THREADS 优先；老写法 DFL_PROJ_COMPOSE=8 也认。"""
    v = str(os.environ.get(_ENV_COMPOSE_N) or "").strip()
    if v:
        return _int_env(_ENV_COMPOSE_N, default, 1, 16)
    old = str(os.environ.get(_ENV_COMPOSE) or "").strip()
    if old.isdigit():
        return _int_env(_ENV_COMPOSE, default, 1, 16)
    return max(1, min(16, int(default)))


def _resize_canvas(img, dst):
    """对齐画布 → dst×dst。缩小用 INTER_AREA（带预滤波），放大用 INTER_CUBIC。

    和「原帧 warp 到画布」不同：这里源和目标都在画布空间，是纯重采样，也不涉及
    任何矩阵 —— mat 由调用方按 dst/S_al 线性缩放，几何仍然精确。
    """
    import cv2
    h = int(img.shape[0])
    dst = int(dst)
    if h == dst:
        return img
    return cv2.resize(img, (dst, dst),
                      interpolation=(cv2.INTER_AREA if dst < h else cv2.INTER_CUBIC))


def _int_env(name, default, lo, hi) -> int:
    try:
        v = int(str(os.environ.get(name) or "").strip() or 0)
    except ValueError:
        v = 0
    if v <= 0:
        v = int(default)
    return max(int(lo), min(int(hi), int(v)))


# ------------------------------------------------------------
# aligned/ → 每一帧有哪几张脸（DFLJPG 里那几样东西直接拿来用）
# ------------------------------------------------------------
def _lm_of(meta: dict):
    """DFLJPG → 原帧坐标的 68 点（新老字段名都认）。拿不到返回 None。"""
    for k in ("source_landmarks", "source_landmarks_68"):
        v = meta.get(k)
        if v:
            try:
                import numpy as np
                a = np.asarray(v, np.float32)
                if (a.ndim == 2 and a.shape[1] >= 2 and a.shape[0] >= 5
                        and np.all(np.isfinite(a))):
                    return a[:, :2].copy()
            except Exception:                               # noqa: BLE001
                continue
    return None


def _rect_of(meta: dict, lm) -> list:
    """检测框（只用它挑「主角脸」，不用来合成）。缺字段就退回 68 点的包围盒。"""
    v = meta.get("source_rect")
    try:
        if v and len(v) == 4:
            return [float(x) for x in v]
    except TypeError:
        pass
    if lm is not None:
        import numpy as np
        a = np.asarray(lm, np.float32)
        return [float(a[:, 0].min()), float(a[:, 1].min()),
                float(a[:, 0].max()), float(a[:, 1].max())]
    return [0.0, 0.0, 0.0, 0.0]


def scan(aligned_dir, frames, rm) -> dict:
    """aligned/ → {帧名: [脸…]}。配对规则和 rm._proj_scan_aligned 完全一样。

    脸字典：{"file","meta","lm","rect","k","area","mat"}
      k    = face_index（切脸时按检测顺序 x1 排出来的序号）；缺字段就按文件名序
      area = 框面积（挑主角脸用，对应 _pick_face 的「先比面积」）
      lm   = 原帧 68 点（**合成用的就是它**，不需要再检测）
    """
    frame_set = set(frames)
    by_stem = {}
    for n in frames:
        by_stem.setdefault(Path(n).stem, n)
    try:
        names = sorted(n for n in os.listdir(str(aligned_dir))
                       if Path(n).suffix.lower() in (".jpg", ".jpeg"))
    except OSError:
        names = []
    raw, no_meta = [], 0
    for n in names:
        meta = rm._proj_head_meta(Path(str(aligned_dir)) / n)
        if not meta:
            no_meta += 1
            continue
        src = str(meta.get("source_filename") or "").strip()
        if not src:                       # 没有这个字段：按文件名前缀认
            src = n.rsplit("_", 1)[0] if "_" in n else Path(n).stem
        raw.append((n, src, meta))

    def _hits(shift):
        return sum(1 for _n, s, _m in raw
                   if rm._proj_match_frame(s, frame_set, by_stem, shift))

    shift = 0
    if raw and _hits(0) <= 0:
        scored = sorted(((_hits(s), s) for s in (1, -1)), reverse=True)
        if scored and scored[0][0] > 0:
            shift = scored[0][1]

    out, missing = {}, []
    for n, src, meta in raw:
        cand = rm._proj_match_frame(src, frame_set, by_stem, shift)
        if cand is None:
            if len(missing) < 64:
                missing.append(str(src) or n)
            continue
        lm = _lm_of(meta)
        rect = _rect_of(meta, lm)
        try:
            k = int(meta.get("face_index"))
        except (TypeError, ValueError):
            k = len(out.get(cand) or [])
        area = max(0.0, (rect[2] - rect[0])) * max(0.0, (rect[3] - rect[1]))
        out.setdefault(cand, []).append({
            "file": Path(str(aligned_dir)) / n, "meta": meta, "lm": lm,
            "rect": rect, "k": int(k), "area": float(area),
            "mat": meta.get("image_to_face_mat") or meta.get("face_mat")})
    for v in out.values():
        v.sort(key=lambda f: (int(f["k"]), str(f["file"].name)))
    no_lm = sum(1 for v in out.values() for f in v if f["lm"] is None)
    return {"map": out, "missing": missing, "no_meta": int(no_meta),
            "shift": int(shift), "faces": len(raw), "no_lm": int(no_lm)}


def _plan_faces(faces, models, single):
    """这一帧要换哪几张脸（口径和老路一致）。

    单模型 = 只换「主角」那张（_pick_face：**先比框面积、再比置信度**）——
    这一版没有置信度（检测已经不在链路里了），所以比 source_rect 的面积，
    同面积取 face_index 小的那张；多模型 = 每张脸按 _model_for_face 的分配
    各用各的 dfm（顺序仍是 face_index）。
    """
    if not faces:
        return []
    if single:
        best = max(faces, key=lambda f: (float(f.get("area") or 0.0), -int(f["k"])))
        return [(best, 0)]
    from .. import routes_merge as rm
    out = []
    for f in faces[: rm._PREVIEW_MAX_SWAPS]:
        try:
            mi = int(rm._model_for_face(int(f["k"]), models))
        except Exception:                                   # noqa: BLE001
            mi = 0
        out.append((f, mi))
    return out


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def run(cfg: dict):
    from .. import routes_merge as rm
    import cv2
    import numpy as np
    from .compose import (compose_aligned_frame, compose_frame,
                                ensure_engine_path)
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat

    project = Path(cfg["project"])
    frames_dir = Path(cfg["frames_dir"])
    aligned_dir = Path(cfg["aligned"])
    merged_dir = Path(cfg["merged"])
    force = bool(cfg.get("force"))

    merged_dir.mkdir(parents=True, exist_ok=True)
    frames = rm._proj_frame_files(frames_dir)
    if not frames:
        raise rm._BadInput("项目里没有帧图（.jpg/.png）：%s" % frames_dir, 422)
    ext_count = {}
    for n in frames:
        e = Path(n).suffix.lower()
        ext_count[e] = ext_count.get(e, 0) + 1
    frame_ext = sorted(ext_count, key=lambda k: (-ext_count[k], k))[0]
    if len(ext_count) > 1:
        rm._proj_warn("帧目录里有多种扩展名（%s），只用 %s 参与编码"
                      % ("、".join(sorted(ext_count)), frame_ext))

    info = scan(aligned_dir, frames, rm)
    face_map = info["map"]
    rm._proj_set(total=len(frames), aligned_faces=int(info["faces"]),
                 missing_source=list(info["missing"]))
    rm._proj_log("项目 %s：%d 帧，%d 张对齐脸，能对上帧的 %d 帧"
                 % (project.name, len(frames), info["faces"], len(face_map)))
    if info["no_meta"]:
        rm._proj_warn("%d 张对齐脸读不出 DFLJPG 元数据（不是 DFLJPG？），已跳过"
                      % int(info["no_meta"]))
    if info["missing"]:
        rm._proj_warn("aligned 里有 %d 条 source_filename 找不到对应原帧（帧被删了？）：%s"
                      % (len(info["missing"]), "、".join(info["missing"][:6])))
    if info["shift"]:
        rm._proj_warn("对齐脸的 source_filename 与帧号整体差 %+d（老项目），已按偏移对上"
                      % int(info["shift"]))
    if not face_map:
        rm._proj_warn("没有任何一张对齐脸能对上帧 —— 这次只是把原帧原样搬进 merged/")

    settings = cfg.get("settings") or {}
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int(settings.get("mask_mode",
                                             comp_cfg.get("mask_mode", 6)) or 0)
    mask_mode = int(comp_cfg["mask_mode"])
    xseg_path = str(cfg.get("xseg") or "")
    need_xseg = bool(xseg_path) and mask_mode >= 6
    need_dst = bool(need_xseg and mask_mode in (7, 8, 9))

    # ---- 模型：一份会话；多模型时按 _model_for_face 分派 ----
    raw_models = [m for m in (cfg.get("models") or []) if m.get("dfm")]
    single = not raw_models
    models = raw_models or [{"name": str(cfg.get("model") or ""),
                             "dfm": str(cfg["dfm"]), "faces": []}]
    plan_paths = [str(m["dfm"]) for m in models]

    # ---- 档位：一份会话 + 合成线程；dfm 攒批是唯一还要按显存取舍的旋钮 ----
    # 计划里的 batch 是**老路的语义**（一条请求带几帧 = 同时在飞的帧数，每帧一份
    # 1080p float32 工作区）。到了流水线里，dfm 的批维只多一份**激活**（416 实测
    # ~40MB/张），所以默认至少攒 4 张；真撞 OOM 有 _flush 的折半重试兜着。
    batch_plan = max(1, int(cfg.get("batch") or 1))
    batch = _int_env(_ENV_DFM_BATCH, max(_DFM_BATCH_MIN, batch_plan),
                     1, _DFM_BATCH_MAX)
    npmode = compose_mode()
    n_cmp = compose_threads(4)
    n_read = _int_env(_ENV_READ, 4, 1, 8)
    n_enc = _int_env(_ENV_ENC, 6, 1, 8)
    png_level = _int_env(_ENV_PNG, _PNG_LEVEL_DEFAULT, 0, 9)
    wait_s = max(0.0, float(_int_env(_ENV_WAIT, int(_BATCH_WAIT_MS), 0, 500)) / 1000.0)

    quit_event = threading.Event()
    fail = {"err": None}
    # 逐段耗时（ms，累计；最后除以帧数就是「每帧在这段上花多少」）—— 排障与调档位
    # 全靠它：帧速上不去时先看哪一段的累计 ÷ 线程数 顶到了墙钟。
    prof = {"prep": 0.0, "dfm": 0.0, "xseg": 0.0, "compose": 0.0,
            "encode": 0.0, "write": 0.0}
    prof_lock = threading.Lock()

    def _acc(key, t_start):
        if not t_start:
            return
        with prof_lock:
            prof[key] = prof.get(key, 0.0) + (time.perf_counter() - t_start) * 1000.0
    stats = {"done": 0, "swapped": 0, "copied": 0, "skipped": 0, "failed": 0,
             "faces": 0, "detected": 0, "aligned": 0, "no_mat": 0}
    failed_frames = []
    face_miss = []
    swap_ms = []
    t0 = time.time()

    try:
        _bf, _bt, _bu = rm._vram_free_smi(fresh=True)
        base_used = _bu
    except Exception:                                       # noqa: BLE001
        base_used = None
    # 采样线程要在**加载会话之前**起来：一份会话涨 2.3GB 就发生在预热那 2 秒里，
    # 起晚了一整单的峰值都看不见（实测会话体量那条记录会因此永远记不上）。
    samp = {"peak": 0, "total": 0}
    samp_stop = threading.Event()

    def _sampler():
        t_begin = time.time()
        while not samp_stop.is_set():
            try:
                _f, _t, _u = rm._vram_free_smi(fresh=True)
                if _u is not None:
                    samp["peak"] = max(int(samp["peak"]), int(_u))
                    samp["total"] = int(_t or 0)
            except Exception:                               # noqa: BLE001
                pass
            # 前 12 秒密采（0.4s）：一份会话是在预热那 2 秒里涨起来的，2 秒一次的
            # 采样常常整段错过 —— 「实测会话体量」那条账就永远记不上。稳定之后
            # 回到 2 秒一次（nvidia-smi 一次 80ms，别一直烧 CPU）。
            samp_stop.wait(0.4 if (time.time() - t_begin) < 12.0 else 2.0)

    threading.Thread(target=_sampler, daemon=True, name="proj-pipe-vram").start()

    # ---- 预热：把 dfm / XSeg 装进显存（帧速从这之后开始算）----
    t_warm = time.time()
    sizes = {}
    _f0 = rm._vram_measure_begin()          # 量「这一次加载到底涨了多少」：段内自有探针，
    for path in plan_paths:                 # 比 2 秒一次的采样准（会话是 2 秒内涨起来的）
        sizes[str(path)] = int(rm._dfm_session(str(path))[2])
    try:
        own_mb = rm._vram_measure_end("dfm:proj_pipe", _f0)
    except Exception:                                       # noqa: BLE001
        own_mb = None
    try:
        cap = int(rm._sess_batch_cap(rm._dfm_session(plan_paths[0])[0]) or 8)
    except Exception:                                       # noqa: BLE001
        cap = 8
    if cap > 1 and batch > cap:
        rm._proj_log("dfm 攒批 %d → %d（这份会话一次最多 %d 张）" % (batch, cap, cap))
        batch = int(cap)
    masker = None
    if need_xseg:
        label = "XSegLite" if "lite" in Path(xseg_path).name.lower() else "XSeg"
        try:
            masker, xs_res = rm._stage_xseg(label)
            got = str(xs_res.get("path") or "")
            if got and os.path.normcase(got) != os.path.normcase(xseg_path):
                rm._proj_warn("遮罩模型按名字取到的是 %s，与请求的 %s 不是同一个文件"
                              % (Path(got).name, Path(xseg_path).name))
            rm._proj_log("遮罩：%s（%s）" % (str(xs_res.get("label") or "-"),
                                           Path(got).name if got else "-"))
        except Exception as e:                              # noqa: BLE001
            masker = None
            rm._proj_warn("遮罩模型加载失败（mask_mode=%d 会退回全 1 遮罩）：%s: %s"
                          % (mask_mode, e.__class__.__name__, str(e)[:160]))
            need_xseg = need_dst = False

    vp = dict(cfg.get("vram_plan") or {})
    if vp.get("downgraded"):
        rm._proj_warn("显存预算只装得下 %d 个 worker（请求 %d）→ 按 %d 开"
                      "（流水线里 worker 恒为一份会话，削减落在攒批/预算上）"
                      % (int(vp.get("procs") or 1), int(vp.get("requested_procs") or 1),
                         int(vp.get("procs") or 1)))
    want_w = int(cfg.get("workers") or 1)
    if want_w > 1:
        # 老路的 workers = 几份 dfm 会话（几个子进程）。这一版全程只有一份会话，
        # 并行度改由「合成/编码线程 + dfm 攒批」决定 —— 手动要 2 个 worker 的请求
        # 仍然接受，但要说清楚它现在落在哪儿（不然日志里出现「请求 2 / 实际 1」会
        # 像降级 bug）。
        rm._proj_warn("流水线只有一份 dfm 会话（请求 %d 个 worker）→ 不再按进程数并行："
                      "合成 %d 线程 · 编码 %d 线程 · dfm 攒批 %d 张"
                      % (want_w, n_cmp, n_enc, batch))
    sz0 = sizes.get(plan_paths[0], 0)
    line = ("流水线：一份 dfm 会话（%s · %dx%d）· 攒批 %s 张 · 合成 %s 线程（%s 画布）· "
            "读帧 %s · 编码 %s · 队列 %s"
            % (Path(plan_paths[0]).name, sz0, sz0, batch, n_cmp, npmode, n_read,
               n_enc, _QDEPTH))
    rm._proj_log(line)
    # 这一行同时进 status.pipeline_line：status 的 log 只回最后 12 行，120 帧跑完
    # 它早被「合成 N/120」挤掉了 —— 验收/排障要能随时读到「这一单按什么档跑的」。
    rm._proj_set(workers=1, batch=int(batch), compose=int(n_cmp), pipeline="fused",
                 pipeline_line=line, compose_mode=str(npmode),
                 adaptive={"src": "pipeline", "note": "流水线：档位不再是 worker×线程，"
                                                     "dfm 攒批 + 合成线程见日志"})
    if npmode == "aligned":
        rm._proj_log("合成画布：aligned（dfm 输入直接取自 DFLJPG 的对齐脸 · 贴回用 "
                     "image_to_face_mat 的逆 · 68 点不参与合成）")
    else:
        rm._proj_log("合成画布：facelib（68 点 → WHOLE_FACE 矩阵 · 原帧再 warp 一次）"
                     " —— DFL_PROJ_COMPOSE=aligned 可切回原生画布")
    rm._proj_log("对齐元数据：%d 张脸的 68 点直接取自 DFLJPG（%d 张缺 landmarks，按需回退检测）"
                 % (sum(len(v) for v in face_map.values()), info["no_lm"]))
    rm._proj_log("预热 %.1fs（一份会话 + %s）"
                 % (time.time() - t_warm, "XSeg" if masker is not None else "无遮罩模型"))

    # ---- 队列 / 段 ----
    END = object()
    q_gpu = queue.Queue(maxsize=_QDEPTH * 2)
    q_cmp = queue.Queue(maxsize=_QDEPTH)
    q_enc = queue.Queue(maxsize=_QDEPTH)
    q_done = queue.Queue()
    # 流水线实况（导出进度页那条 SVG 图的数据源，见 server/routes_merge.py 的 _StageMeter）：
    # lane 数一律取**这一单实际的线程数** —— 合成 n_cmp 条、编码 n_enc 条、dfm 一份会话 1 条。
    # 这一条链没有检测/切帧（aligned 已经是产物），所以 frames/det/xseg 三列留空。
    rm._STAGE_METER.reset(lanes={
        "dfm": ["dfm#0"],
        "compose": ["t%d" % _i for _i in range(max(1, int(n_cmp)))],
        "encode": ["e%d" % _i for _i in range(max(1, int(n_enc)))],
    })
    rm._STAGE_METER.bind({"dfm": q_gpu, "compose": q_enc})
    it = iter(range(len(frames)))
    it_lock = threading.Lock()
    shut_lock = threading.Lock()
    live = {"read": int(n_read), "cmp": int(n_cmp)}
    # ★ 92 是**预览**的传输质量（preview.py:46），不是导出落盘的画质。以前 92 从这里
    #   渗进导出：routes.py 的 export-project 默认 quality = _PREVIEW_QUALITY，于是
    #   同样一单换条路（项目 vs 分段），merged/*.jpg 差一档；分段链用的是 100
    #   （pipeline_stages.py:1894）。对齐到 100。
    q = int(cfg.get("quality") or 100)

    def _put(qq, item):
        while not quit_event.is_set():
            try:
                qq.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _get(qq, timeout=0.2):
        while not quit_event.is_set():
            try:
                return qq.get(timeout=timeout)
            except queue.Empty:
                continue
        return END

    def _drain(qq, n):
        for _i in range(int(n)):
            _put(qq, END)

    t_swap = time.time()

    # ---------- ① 读帧 + 选脸 + warp 到 S（DFLJPG 的 68 点，不检测）----------
    def _fallback_lm(frame):
        """aligned 里连 68 点都没有（5 点模式切的）才回退检测 —— 并计数。"""
        boxes = rm._detect_boxes(frame, cfg["detector"], float(cfg["thresh"]))
        best = rm._pick_face(boxes)
        if best is None:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in best["box"]]
        except (TypeError, ValueError):
            return None
        lm = rm._frame_landmarks(frame, best, [x1, y1, x2, y2])
        return None if lm is None else np.asarray(lm, np.float32)

    def _reader():
        try:
            while not quit_event.is_set():
                with it_lock:
                    try:
                        i = next(it)
                    except StopIteration:
                        return
                t_prep = time.perf_counter()
                name = frames[i]
                src_p = frames_dir / name
                dst_p = merged_dir / name
                job = {"i": int(i), "name": name, "src": src_p, "dst": dst_p,
                       "kind": "swap", "t0": time.time(), "faces": [], "frame": None}
                if not force and dst_p.is_file() and dst_p.stat().st_size > 0:
                    job["kind"] = "skip"
                    _acc("prep", t_prep)
                    _put(q_done, (i, job))
                    continue
                faces = face_map.get(name) or []
                if not faces:
                    job["kind"] = "copy"
                    _put(q_done, (i, job))
                    continue
                try:
                    frame = _app_mod._imread_any(str(src_p))
                    why = ""
                except Exception as e:                      # noqa: BLE001
                    # 别把真实原因吞成「这一帧读不出来（损坏？）」：拆分后有 12 帧全走这条
                    # 分支（rm._app_mod 在 routes_merge 上不存在 → AttributeError），收尾只
                    # 报「merged/ 里一帧都没有」，真实错误一点痕迹都没有。异常类型跟到
                    # failed_frames 里，下次一眼能看出是读图坏了还是代码缺符号。
                    why = "%s: %s" % (e.__class__.__name__, str(e)[:160])
                    frame = None
                if frame is None:
                    job["kind"] = "err"
                    job["error"] = ("这一帧读不出来（损坏？）"
                                    + ("：" + why if why else ""))
                    _put(q_done, (i, job))
                    continue
                job["frame"] = frame
                use = _plan_faces(faces, models, single)
                if not use:
                    job["kind"] = "copy"
                    _put(q_done, (i, job))
                    continue
                try:
                    for face, mi in use:
                        dfm = str(models[mi]["dfm"])
                        sz = int(sizes.get(dfm) or 256)
                        fc = {"k": int(face["k"]), "lm": face["lm"],
                              "dfm": dfm, "size": sz, "mi": int(mi),
                              "dfm_in": None, "dst256": None,
                              "mat": None, "dw": None, "canvas": "facelib",
                              "pred": None, "prd": None, "dst": None,
                              "xp": None, "xd": None}
                        if npmode == "aligned" and face.get("mat") is not None:
                            # ---- aligned 原生：dfm 输入就是 ② 切脸写出来的那张脸
                            # 画布尺寸可能是 512、也可能是 462（老版本按框裁的），
                            # 一律缩到模型输入 S：mat 乘以 S/S_al，几何仍然精确
                            # （整张画布按比例缩放，仿射矩阵线性缩放即等价）。
                            aligned = _app_mod._imread_any(str(face["file"]))
                            if aligned is None:
                                raise RuntimeError("对齐脸读不出来：%s" % face["file"])
                            s_al = int(aligned.shape[0])
                            img = (aligned if s_al == sz
                                   else _resize_canvas(aligned, sz))
                            arr = np.clip(img.astype(np.float32) / 255.0, 0, 1)
                            fc["dfm_in"] = arr
                            fc["dw"] = arr          # ct 的参考就是画布上的这张脸
                            fc["mat"] = (np.asarray(face["mat"], np.float64)
                                         * (float(sz) / float(s_al)))
                            fc["canvas"] = "aligned"
                            if need_dst:
                                fc["dst256"] = _resize_canvas(aligned, 256)
                            stats["aligned"] += 1
                        else:
                            # ---- facelib 画布（老行为 / 没有 image_to_face_mat 的兜底）
                            lm = face["lm"]
                            if lm is None:
                                lm = _fallback_lm(frame)
                                stats["detected"] += 1
                                if lm is None:
                                    continue        # 没点就跳过这张脸
                                face["lm"] = lm
                            if npmode == "aligned":
                                stats["no_mat"] += 1      # 缺 mat，退回 facelib 画布
                            fmat = get_transform_mat(np.asarray(lm, np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            warped = cv2.warpAffine(frame, fmat, (sz, sz),
                                                    flags=cv2.INTER_CUBIC)
                            fc["lm"] = np.asarray(lm, np.float32)
                            fc["dfm_in"] = np.clip(warped.astype(np.float32) / 255.0,
                                                   0, 1)
                            if need_dst:
                                # XSeg 的 dst 画布：和 MergeMaskedFace 里那一趟同序
                                # （原帧 → facelib WHOLE_FACE 画布 → 256）。256 那张矩阵
                                # 就是 S 那张乘 256/S（同一套模板只差缩放，实测 1e-13），
                                # 所以不用再算一次 get_transform_mat。
                                m256 = np.array(fmat, np.float64) * (256.0 / float(sz))
                                fc["dst256"] = cv2.warpAffine(frame, m256, (256, 256),
                                                              flags=cv2.INTER_CUBIC)
                        job["faces"].append(fc)
                except Exception as e:                      # noqa: BLE001
                    traceback.print_exc()
                    job["kind"] = "err"
                    job["error"] = "%s: %s" % (e.__class__.__name__, str(e)[:160])
                    _put(q_done, (i, job))
                    continue
                if not job["faces"]:
                    job["kind"] = "copy_noface"
                    _put(q_done, (i, job))
                    continue
                _acc("prep", t_prep)
                for fc in job["faces"]:
                    rm._stage_tick("dfm")        # 交接点：读帧/warp → dfm（每张脸一次）
                    if not _put(q_gpu, (job, fc)):
                        return
        except BaseException as e:                          # noqa: BLE001
            fail["err"] = fail["err"] or ("read: %r" % (e,))
            quit_event.set()
        finally:
            with shut_lock:
                live["read"] -= 1
                last = live["read"] == 0
            if last:
                _put(q_gpu, END)

    # ---------- ② dfm 攒批 + XSeg（只在一条线程上碰 GPU）----------
    def _gpu():
        pend = {}
        born = {}
        frames_of = {}

        def _flush(only=None):
            for path in (list(pend) if only is None else [only]):
                part = pend.get(path) or []
                if not part:
                    continue
                del pend[path]
                born.pop(path, None)
                i = 0
                while i < len(part):
                    chunk = part[i:i + batch]
                    i += batch
                    sz = int(chunk[0][1]["size"])
                    arr = np.empty((len(chunk), sz, sz, 3), np.float32)
                    for n, (_jb, fc) in enumerate(chunk):
                        arr[n] = fc["dfm_in"]
                    t_dfm = time.perf_counter()
                    try:
                        outs = rm._dfm_run_batch(rm._dfm_session(path), arr)
                    except Exception as e:                  # noqa: BLE001
                        if not (rm._is_oom(e) and len(chunk) > 1):
                            raise
                        half = max(1, len(chunk) // 2)
                        rm._proj_log("dfm batch=%d 显存不够（%s）→ 退到 %d 再试"
                                     % (len(chunk), str(e)[:80], half))
                        outs = []
                        for j in range(0, len(chunk), half):
                            outs.extend(rm._dfm_run_batch(rm._dfm_session(path),
                                                          arr[j:j + half]))
                    _acc("dfm", t_dfm)
                    for n, (_jb, fc) in enumerate(chunk):
                        face01, prd, dst = outs[n]
                        fc["pred"] = np.clip(np.asarray(face01, np.float32), 0, 1)
                        fc["prd"] = prd
                        fc["dst"] = dst
                    if masker is not None:
                        t_xs = time.perf_counter()
                        for jb, fc in chunk:
                            imgs = [np.clip(fc["pred"] * 255.0, 0, 255).astype(np.uint8)]
                            if fc.get("dst256") is not None:
                                imgs.append(fc["dst256"])
                            try:
                                ms = masker.mask_batch(imgs, out_size=256) or []
                            except Exception as e:          # noqa: BLE001
                                raise rm._BadInput("遮罩推理失败：%s" % str(e)[:160], 500)
                            fc["xp"] = ms[0] if len(ms) > 0 else None
                            fc["xd"] = ms[1] if len(ms) > 1 else None
                        _acc("xseg", t_xs)
                    for jb, _fc in chunk:
                        if not jb.get("t_gpu"):
                            jb["t_gpu"] = time.time()      # 这张帧第一次进 GPU 段的时刻
                        rec = frames_of.get(id(jb))
                        if rec is None:
                            continue
                        rec[0] -= 1
                        if rec[0] <= 0:
                            frames_of.pop(id(jb), None)
                            _put(q_cmp, jb)

        try:
            while not quit_event.is_set():
                try:
                    item = q_gpu.get(timeout=0.005)
                except queue.Empty:
                    item = None
                if item is END:
                    _flush()
                    _drain(q_cmp, n_cmp)
                    return
                if item is None:
                    if wait_s > 0.0 and pend:
                        now = time.perf_counter()
                        for pth in list(pend):
                            if now - float(born.get(pth, now)) >= wait_s:
                                _flush(pth)
                    continue
                job, fc = item
                if id(job) not in frames_of:
                    frames_of[id(job)] = [len(job["faces"])]
                if not pend.get(fc["dfm"]):
                    born[fc["dfm"]] = time.perf_counter()
                pend.setdefault(fc["dfm"], []).append((job, fc))
                if len(pend[fc["dfm"]]) >= batch:
                    _flush(fc["dfm"])
        except BaseException as e:                          # noqa: BLE001
            fail["err"] = fail["err"] or ("gpu: %r" % (e,))
            quit_event.set()

    # ---------- ③ 合成（ROI 版；画布按 npmode 分两条）----------
    def _composer():
        try:
            while not quit_event.is_set():
                job = _get(q_cmp)
                if job is END:
                    return
                # 按**画布**分组：aligned 那群走 compose_aligned_frame（image_to_face_mat），
                # facelib 那群走 compose_frame（68 点现算 WHOLE_FACE）。正常一帧里只有
                # 一种；缺 image_to_face_mat 的个别脸会退到 facelib，这时按组各贴一次。
                grp_a, grp_l = [], []
                for fc in job["faces"]:
                    common = {"pred": fc["pred"], "prd_dfm": fc["prd"],
                              "dst_dfm": fc["dst"], "xseg_prd": fc["xp"],
                              "xseg_dst": fc["xd"], "landmarks": fc["lm"]}
                    if fc.get("canvas") == "aligned":
                        common["mat"] = fc["mat"]
                        common["dw"] = fc["dw"]
                        grp_a.append(common)
                    else:
                        grp_l.append(common)
                t_cmp = time.perf_counter()
                try:
                    out = job["frame"]
                    tms = job.setdefault("timings", {})
                    if grp_a:
                        out = compose_aligned_frame(out, grp_a, comp_cfg, tms)
                    if grp_l:
                        out = compose_frame(out, grp_l, comp_cfg, tms)
                except Exception as e:                      # noqa: BLE001
                    job["kind"] = "err"
                    job["error"] = "第 %d 帧合成失败：%s" % (job["i"], str(e)[:160])
                    _put(q_done, (job["i"], job))
                    continue
                _acc("compose", t_cmp)
                job["out"] = out
                job["frame"] = None
                # swap_ms 口径 = 「换脸+遮罩+合成」这一段（从第一张脸进 GPU 到合成完），
                # 和老路 worker 里的 swap_ms 可比；排队等位不算在内（那是流水线深度，
                # 31 帧/秒时当然能积到 1~2 秒，混进来会让人以为单帧变慢了）。
                job["swap_ms"] = (time.time() - (job.get("t_gpu") or job["t0"])) * 1000.0
                rm._stage_tick("compose")        # 交接点：dfm/遮罩/合成 → 编码
                _put(q_enc, job)
        except BaseException as e:                          # noqa: BLE001
            fail["err"] = fail["err"] or ("compose: %r" % (e,))
            quit_event.set()
        finally:
            with shut_lock:
                live["cmp"] -= 1
                last = live["cmp"] == 0
            if last:
                _drain(q_enc, n_enc)

    # ---------- ④ 编码（.png 帧编 PNG、.jpg 帧编 JPEG q=quality）----------
    def _encoder():
        try:
            while not quit_event.is_set():
                job = _get(q_enc)
                if job is END:
                    return
                t_enc = time.perf_counter()
                try:
                    if Path(job["name"]).suffix.lower() == ".png":
                        # 和 ⑤ 合成阶段 / 快速导出**同一个**编码函数（含 Z_RLE 策略）：
                        # 以前这里是裸 imencode(.png, level=3)，是三条导出链里唯一没走
                        # _stage_imencode_png 的一处，也是唯一还在用 level 3 的一处。
                        job["data"] = rm._stage_imencode_png(job["out"], png_level)
                    else:
                        ok, buf = cv2.imencode(
                            ".jpg", job["out"], [int(cv2.IMWRITE_JPEG_QUALITY), q])
                        if not ok:
                            raise RuntimeError("编码失败")
                        job["data"] = buf.tobytes()
                except Exception as e:                      # noqa: BLE001
                    job["kind"] = "err"
                    job["error"] = "编码失败：%s" % str(e)[:120]
                _acc("encode", t_enc)
                job["out"] = None
                rm._stage_tick("encode")         # 交接点：编码 → 写盘
                _put(q_done, (job["i"], job))
        except BaseException as e:                          # noqa: BLE001
            fail["err"] = fail["err"] or ("encode: %r" % (e,))
            quit_event.set()

    # ---------- 主线程：按帧序提交（断点续跑语义）----------
    pending = {}
    next_i = [0]

    def _commit(job):
        kind = job.get("kind")
        name = job["name"]
        t_wr = time.perf_counter()
        try:
            if kind == "skip":
                stats["skipped"] += 1
            elif kind in ("copy", "copy_noface"):
                rm._proj_put(job["dst"], src=job["src"])
                stats["copied"] += 1
                if kind == "copy_noface" and len(face_miss) < rm._PROJ_MAX_LIST:
                    face_miss.append(name)
            elif kind == "err":
                stats["failed"] += 1
                if len(failed_frames) < rm._PROJ_MAX_LIST:
                    failed_frames.append({"frame": name,
                                          "error": str(job.get("error") or "未知错误")})
            else:
                rm._proj_put(job["dst"], data=job.get("data") or b"")
                stats["swapped"] += 1
                stats["faces"] += len(job.get("faces") or [])
                if job.get("swap_ms"):
                    swap_ms.append(float(job["swap_ms"]))
        except Exception as e:                              # noqa: BLE001
            stats["failed"] += 1
            failed_frames.append({"frame": name, "error": "落盘失败：%s" % str(e)[:120]})
        _acc("write", t_wr)
        tm = job.get("timings") or {}
        for src_key, dst_key in (("mask_ms", "mask"), ("ct_ms", "ct"),
                                 ("paste_ms", "paste"), ("blend_ms", "blend")):
            try:
                v = float(tm.get(src_key) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v:
                with prof_lock:
                    prof[dst_key] = prof.get(dst_key, 0.0) + v
        next_i[0] = int(job["i"]) + 1
        done = next_i[0]
        el_swap = max(0.001, time.time() - t_swap)
        rate = (stats["swapped"] + stats["copied"] + stats["failed"]) / el_swap
        vals = sorted(x for x in swap_ms if x)
        p50 = round(vals[len(vals) // 2], 1) if vals else None
        rm._proj_set(done=done, swapped=stats["swapped"], copied=stats["copied"],
                     skipped=stats["skipped"], failed=stats["failed"],
                     elapsed=round(time.time() - t0, 1),
                     swap_elapsed=round(el_swap, 1), swap_frames=len(swap_ms),
                     swap_p50=p50, frame_fps=round(rate, 2),
                     eta_s=(int((len(frames) - done) / rate) if rate > 0 else 0),
                     pct=min(rm._PROJ_PCT_SWAP,
                             int(rm._PROJ_PCT_SWAP * done / float(len(frames)))),
                     msg="合成 %d/%d 帧 · %.1f 帧/秒%s"
                         % (done, len(frames), rate,
                            ("（跳过 %d）" % stats["skipped"]) if stats["skipped"] else ""))
        if done % 20 == 0 or done == len(frames):
            rm._proj_log("合成 %d/%d 帧 · 换脸 %d · 原样 %d · 跳过 %d · 失败 %d · 单帧 p50 %sms"
                         % (done, len(frames), stats["swapped"], stats["copied"],
                            stats["skipped"], stats["failed"],
                            p50 if p50 is not None else "-"))

    cancelled = False
    ths = []
    cv2_prev = 0
    try:
        for _k in range(n_read):
            ths.append(threading.Thread(target=_reader, name="proj-read%d" % _k,
                                        daemon=True))
        ths.append(threading.Thread(target=_gpu, name="proj-gpu", daemon=True))
        for _k in range(n_cmp):
            ths.append(threading.Thread(target=_composer, name="proj-cmp%d" % _k,
                                        daemon=True))
        for _k in range(n_enc):
            ths.append(threading.Thread(target=_encoder, name="proj-enc%d" % _k,
                                        daemon=True))
        try:
            cv2_prev = int(cv2.getNumThreads())
            cv2.setNumThreads(max(1, min(4, max(1, (os.cpu_count() or 8)
                                                  // max(1, n_cmp)))))
        except Exception:                                   # noqa: BLE001
            cv2_prev = 0
        for th in ths:
            th.start()
        while True:
            if rm._proj_cancelled():
                cancelled = True
                quit_event.set()
            try:
                item = q_done.get(timeout=0.05)
            except queue.Empty:
                item = None
            if item is not None:
                pending[item[0]] = item[1]
            while next_i[0] in pending:
                _commit(pending.pop(next_i[0]))
            if fail["err"]:
                quit_event.set()
            if not any(th.is_alive() for th in ths):
                # 线程全退了不代表队列空了（尤其「全部续跑跳过」那种：读者几百毫秒
                # 就干完了，主线程可能只收到一条）。把 q_done 收干净再走，否则
                # done 会停在 1、后面的帧被当成「没做」。
                while True:
                    try:
                        item = q_done.get(timeout=0.2)
                    except queue.Empty:
                        break
                    pending[item[0]] = item[1]
                    while next_i[0] in pending:
                        _commit(pending.pop(next_i[0]))
                break
        while next_i[0] in pending:
            _commit(pending.pop(next_i[0]))
    finally:
        samp_stop.set()
        quit_event.set()
        # 流水线实况收单：不再涨，但保留 lane/ticks 给界面画最后一帧
        rm._STAGE_METER.stop()
        for th in ths:
            try:
                th.join(timeout=5.0)
            except Exception:                               # noqa: BLE001
                pass
        if cv2_prev:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                               # noqa: BLE001
                pass

    if fail["err"]:
        raise rm._BadInput("项目导出流水线出错：%s" % str(fail["err"])[:300], 500)
    if cancelled:
        raise rm._ProjCancel()

    # 会话体量记账：把「这一次加载实测涨了多少」并进峰值再交给预算记账。为什么不能只靠
    # 采样：一份会话是预热那 2 秒里涨起来的，2 秒一次的采样常常整段错过（尤其 60 帧的
    # 小项目，整单才 4 秒），记出来的 delta 会小于权重 → 那一单白量。
    peak_rec = int(samp.get("peak") or 0)
    if own_mb and base_used:
        # 把这一单自己的加载量还原成「等效峰值」：预算记账那边会再把 dfm 批与
        # 检测/遮罩那两项扣回去（它们不在这 2.2GB 里），不补回来的话 384 这种
        # 「会话 2.16GB、权重 1.53GB」的账会被扣到阈值以下而整单白量。
        extra = 0.0
        try:
            extra = float(rm._vram_batch_mb(int(sizes.get(plan_paths[0]) or 416),
                                            int(batch), float(rm._vram_models_mb(plan_paths)),
                                            "procs")) + float(rm._VRAM_AUX_MB)
        except Exception:                                   # noqa: BLE001
            extra = 0.0
        peak_rec = max(peak_rec, int(float(base_used) + float(own_mb) + extra))
    rm._proj_log("会话体量：本次加载涨 %.0f MB（开跑前整卡 %s MB · 采样峰值 %d MB）"
                 % (float(own_mb or 0.0), base_used if base_used is not None else "?",
                    int(samp.get("peak") or 0)))
    try:
        rm._proj_record_sess_peak(plan_paths, 1, int(batch), base_used,
                                  peak_rec, samp.get("total"),
                                  size_px=int(sizes.get(plan_paths[0]) or 0),
                                  ok=not failed_frames)
    except Exception:                                       # noqa: BLE001
        pass
    n_face = max(1, int(stats["swapped"]))
    if stats["aligned"] or stats["no_mat"]:
        rm._proj_log("合成画布统计：aligned %d 张 · facelib 兜底 %d 张（缺 image_to_face_mat）"
                     % (int(stats["aligned"]), int(stats["no_mat"])))
    rm._proj_log("分段耗时（每帧 ms · 除以本段线程数才是墙钟）：读帧/warp %.1f（%d 线程）"
                 " · dfm %.1f · xseg %.1f · 合成 %.1f（%d 线程）· 编码 %.1f（%d 线程）"
                 " · 落盘 %.1f —— 按 %d 张有脸帧算"
                 % (prof["prep"] / n_face, n_read, prof["dfm"] / n_face,
                    prof["xseg"] / n_face, prof["compose"] / n_face, n_cmp,
                    prof["encode"] / n_face, n_enc, prof["write"] / n_face, n_face))
    # 合成内部逐段（compose_frame 的 timings）：遮罩准备 / 色彩迁移 / 反贴 / 融合。
    # **ct 在 aligned（S×S）域上算**：_merge_compose 里 fo=pred(S×S)、dw=原帧 warp 到
    # S×S 的那张，_color_transfer 的统计量只在这两张 S×S 图上取 —— 与整帧 1080p 无关。
    rm._proj_log("合成内部（每帧 ms）：遮罩准备 %.1f · 色彩迁移 %.1f（S×S 域）"
                 " · 反贴 %.1f · 融合 %.1f"
                 % (prof.get("mask", 0.0) / n_face, prof.get("ct", 0.0) / n_face,
                    prof.get("paste", 0.0) / n_face, prof.get("blend", 0.0) / n_face))
    rm._proj_set(profile={k: round(v / n_face, 2) for k, v in prof.items()})
    if stats["detected"]:
        rm._proj_warn("%d 张对齐脸缺 68 点（5 点模式切的？），这些帧回退过一次检测"
                      % int(stats["detected"]))
    if face_miss:
        rm._proj_warn("%d 帧有对齐脸但一张都没换上，已按原帧落盘：%s"
                      % (len(face_miss), "、".join(face_miss[:6])))
    rm._proj_set(failed_frames=failed_frames[: rm._PROJ_MAX_LIST])

    return rm._proj_finish(cfg, project, frames, frame_ext, merged_dir, t0, len(frames))


# ============================================================
# DFL 项目模式（原版 布局）· 逐帧合成落盘 + 按原 fps/音轨重新编码
# ------------------------------------------------------------
# 项目目录（和 原版的 data_dst 结构一致）：
#
#   <项目>/000000.jpg              原帧（6 位序号，按顺序）
#   <项目>/aligned/000000_0.jpg    对齐脸 DFLJPG；元数据 source_filename 指回
#                                  "000000"（也认 "000000.jpg"）
#   <项目>/merged/000000.jpg       产物：换脸后的**整帧**（本接口写的）
#
# 和 /api/merge/export 的分工：
#   /export          -> 4.0 MergeStudio 子进程一口气出 mp4，断了要从头再跑
#   /export-project  -> 逐帧合成落盘到 merged/，再单独编码；能断点续跑、
#                       能报失败帧、产物可以一张张复查
#
# 「逐帧合成」这一段不另写一套：直接复用单帧预览的 _preview_job
# （检测 -> 2D106 的 68 点 -> DFM 换脸 + 遮罩 + 合成），只把「从视频里取帧」
# 换成「从 <项目>/<帧名> 读图」。同一帧走预览和走导出，输出是同一份字节。
# ============================================================

_PROJ_LOCK = threading.RLock()
_PROJ_PCT_SWAP = 96          # 逐帧合成占 0~96%，剩下的给编码
_PROJ_MAX_LIST = 200         # 失败帧 / 缺帧清单最多回传这么多条
_PROJ_FRAME_EXT = {".jpg", ".jpeg", ".png"}
_PROJ_JOB = {
    "running": False, "ok": None, "phase": "idle", "cancelled": False,
    "error": "", "error_detail": "", "stage": "idle", "pct": 0, "msg": "",
    "project": "", "frames_dir": "", "aligned": "", "merged": "",
    "video": "", "out_video": "", "fps": 0.0, "width": 0, "height": 0,
    "video_frames": 0, "total": 0, "done": 0, "swapped": 0, "copied": 0,
    "skipped": 0, "failed": 0, "frame_fps": 0.0, "eta_s": None, "elapsed": 0.0,
    "started_at": None, "finished_at": None, "force": False,
    "dfm": "", "model": "", "models": [], "settings": {}, "detector": "",
    "warnings": [], "failed_frames": [], "missing": [], "missing_source": [],
    "aligned_faces": 0, "merged_frames": 0, "encode": {}, "log": [],
    "want_cancel": False,
}


class _ProjCancel(Exception):
    """用户取消（和失败分开记：取消不是错误）。"""


def _proj_snap() -> dict:
    with _PROJ_LOCK:
        snap = dict(_PROJ_JOB)
        for k in ("warnings", "eff_warnings", "failed_frames", "missing",
                  "missing_source", "models", "log"):
            snap[k] = list(snap.get(k) or [])
        snap["settings"] = dict(snap.get("settings") or {})
        snap["encode"] = dict(snap.get("encode") or {})
        return snap


def _proj_set(**kw):
    with _PROJ_LOCK:
        _PROJ_JOB.update(kw)


def _proj_log(msg):
    line = str(msg)[:300]
    with _PROJ_LOCK:
        log = _PROJ_JOB.get("log")
        if not isinstance(log, list):
            log = []
            _PROJ_JOB["log"] = log
        log.append(line)
        if len(log) > 300:
            del log[:-300]


def _proj_warn(msg):
    """记一条警告：进 warnings（给界面看）同时进 log（给排障用）。"""
    text = str(msg)[:400]
    with _PROJ_LOCK:
        warns = _PROJ_JOB.get("warnings")
        if not isinstance(warns, list):
            warns = []
            _PROJ_JOB["warnings"] = warns
        if text not in warns:
            warns.append(text)
            if len(warns) > 40:
                del warns[:-40]
    _proj_log("[warn] " + text)


def _proj_cancelled() -> bool:
    with _PROJ_LOCK:
        return bool(_PROJ_JOB.get("want_cancel"))


def _proj_frame_files(dirp) -> list:
    """目录里的帧图名（按文件名排序 = ffmpeg image2 的读取顺序）。"""
    out = []
    try:
        with os.scandir(str(dirp)) as it:
            for e in it:
                try:
                    if e.is_file() and Path(e.name).suffix.lower() in _PROJ_FRAME_EXT:
                        out.append(e.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(out)


def _proj_head_meta(path, limit=96 * 1024) -> dict:
    """只读文件头就把 DFLJPG 里的 APP15（pickle）解出来。

    app._dfljpg_dict() 是整份读进内存 —— 四万张对齐脸就是几 GB 的读放大。
    APP15 紧跟 SOI，96KB 足够；段长超过窗口时才回退到整份读。
    """
    try:
        with open(str(path), "rb") as fh:
            data = fh.read(int(limit))
    except OSError:
        return {}
    i, n = 0, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            return {}
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return {}
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            return {}
        if marker == 0xEF:                        # APP15
            end = i + 2 + seg_len
            if end > n:                           # 段比窗口还长：退回整份读
                try:
                    return _app_mod._dfljpg_dict(Path(path)) or {}
                except Exception:
                    return {}
            try:
                d = _app_mod._safe_loads(data[i + 4:end])
            except Exception:
                return {}
            return d if isinstance(d, dict) else {}
        i += 2 + seg_len
    return {}


def _proj_match_frame(src, frame_set, by_stem, shift=0):
    """DFLJPG 的 source_filename -> 真实存在的帧名；认不出来回 None（不猜）。"""
    s = str(src or "").strip().replace("\\", "/")
    if not s:
        return None
    base = s.rsplit("/", 1)[-1]
    if base in frame_set:
        return base
    stem = Path(base).stem
    if stem in by_stem:
        return by_stem[stem]
    if shift and stem.isdigit():
        alt = str(int(stem) + int(shift))
        if len(alt) < len(stem):
            alt = alt.zfill(len(stem))
        if alt in by_stem:
            return by_stem[alt]
    return None


def _proj_scan_aligned(aligned_dir, frames) -> dict:
    """aligned 目录 -> 每个源帧对应哪几张对齐脸。

    返回 {map: {帧名: [脸文件名…]}, missing: [认不出原帧的 source_filename…],
          no_meta: 几张没有元数据, shift: 用到的人数偏移, faces: 总脸数}

    认帧顺序：source_filename 原名 -> 去扩展名 -> 整体平移 ±1（老项目帧从
    000001 起、对齐从 000000 起，差一位）。三种都对不上就进 missing，绝不瞎配。
    """
    frame_set = set(frames)
    by_stem = {}
    for n in frames:
        by_stem.setdefault(Path(n).stem, n)
    try:
        names = sorted(n for n in os.listdir(str(aligned_dir))
                       if Path(n).suffix.lower() in (".jpg", ".jpeg"))
    except OSError:
        names = []
    pairs, no_meta = [], 0
    for n in names:
        meta = _proj_head_meta(Path(str(aligned_dir)) / n)
        if not meta:
            no_meta += 1
            continue
        src = str(meta.get("source_filename") or "").strip()
        if not src:                       # 没有这个字段：按文件名前缀认
            src = n.rsplit("_", 1)[0] if "_" in n else Path(n).stem
        pairs.append((n, src))
    def _hits(shift):
        return sum(1 for _n, s in pairs
                   if _proj_match_frame(s, frame_set, by_stem, shift))

    best_shift = 0
    if pairs and _hits(0) <= 0:
        # 一条都对不上才怀疑是「整体差一位」的老项目（帧 000001 起、对齐 000000 起）。
        # 只要原名能对上就别平移：总数上「多对上一张」也可能把 A 的脸贴到 B 上。
        scored = sorted(((_hits(s), s) for s in (1, -1)), reverse=True)
        if scored and scored[0][0] > 0:
            best_shift = scored[0][1]
    out, missing = {}, []
    for n, src in pairs:
        cand = _proj_match_frame(src, frame_set, by_stem, best_shift)
        if cand is None:
            if len(missing) < _PROJ_MAX_LIST:
                missing.append(str(src) or n)
            continue
        out.setdefault(cand, []).append(n)
    return {"map": out, "missing": missing, "no_meta": int(no_meta),
            "shift": int(best_shift), "faces": len(pairs)}


def _proj_seq_info(names, contiguous=True):
    """帧名列表 -> (prefix, 数字位宽, 后缀, 起始编号, 编号列表)；认不出就 None。

    认不出来（前缀/后缀/位宽不统一、编号不连号）的帧序列没法用 image2 的
    %0Nd 写法喂给 ffmpeg —— 有洞时它会在第一个洞停住，成片被悄悄截短。
    """
    if not names:
        return None
    rx = re.compile(r"^(.*?)(\d+)(\D*)$")
    parsed = []
    for n in names:
        m = rx.match(str(n))
        if not m:
            return None
        parsed.append((m.group(1), int(m.group(2)), len(m.group(2)), m.group(3)))
    prefix, suffix, width = parsed[0][0], parsed[0][3], parsed[0][2]
    for p, _num, w, suf in parsed:
        if p != prefix or suf != suffix or w != width:
            return None
    nums = [x[1] for x in parsed]
    if sorted(nums) != nums:
        return None
    if contiguous and nums[-1] - nums[0] + 1 != len(nums):
        return None                        # 编号有洞（image2 会在这儿停住）
    return prefix, int(width), suffix, int(nums[0]), nums


def _proj_pattern(names):
    """帧名列表 -> (image2 的 pattern, 起始编号)；认不出 / 编号有洞就 None。"""
    info = _proj_seq_info(names)
    if info is None:
        return None
    prefix, width, suffix, start, _nums = info
    return prefix + "%0" + str(width) + "d" + suffix, int(start)


def _proj_put(dst: Path, data=None, src=None):
    """原子落盘：先写 .part 再 os.replace。

    中途被杀 / 断电只会留下 .part，不会把「写了一半的图」当成已完成 ——
    断点续跑靠的就是「文件存在且非空」这一条判据。
    """
    tmp = Path(str(dst) + ".part")
    try:
        if data is not None:
            with open(str(tmp), "wb") as fh:
                fh.write(data)
        else:
            shutil.copyfile(str(src), str(tmp))
        os.replace(str(tmp), str(dst))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _proj_encode(pattern, start, fps, video, out_video, total) -> dict:
    """按需求里那条命令编码：merged 帧 + 原视频音轨，libx264 / crf16 / faststart。

    刻意**不加 -shortest**：音轨比画面长时它会把成片截短（4.0 那边踩过）。

    -c:v 那一段走 _common/encoder_sel（和快速导出 / 分段导出同一个函数）：默认 libx264 crf16
    逐字节不变，DFL_FUSED_ENCODER=nvenc-* 才换。**这里不给宽高** —— 判 4K 要读源帧 / 读 PNG，
    比这一单本身还贵；所以 auto 在项目导出上永远等于 x264（那本来就是默认），
    显式点名 nvenc 时照样生效（含 4K）。见 docs/29-ENCODER.md。
    """
    ffmpeg = str(_app_mod._ffmpeg_exe())
    ext = Path(str(out_video)).suffix.lower()
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-framerate", "%.6g" % float(fps),
           "-start_number", str(int(start)), "-i", str(pattern),
           "-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           # 1) 奇数宽高在 yuv420p 下会被 libx264 直接拒绝，垫一像素最省事；
           # 2) merged 是 JPEG（full range），不显式转 limited 的话成片的 pix_fmt
           #    会被标成 yuvj420p(pc)，很多播放器按 tv 解释 → 整体发灰。
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv"] + \
          _esel.encoder_args() + \
          ["-pix_fmt", "yuv420p",
           "-c:a", "copy"]
    if ext in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", str(out_video)]

    err_path = None
    code = -1
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=".ffmpeg.log")
    err_path = fh.name
    fh.close()
    try:
        with open(str(err_path), "wb") as errfp:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfp,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
            try:
                for line in proc.stdout:
                    if _proj_cancelled():
                        _kill_proc_tree(proc)
                        raise _ProjCancel()
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    k, _eq, v = line.partition("=")
                    if k != "frame":
                        continue
                    try:
                        done = int(v)
                    except ValueError:
                        continue
                    pct = _PROJ_PCT_SWAP
                    if total > 0:
                        pct = _PROJ_PCT_SWAP + int((100 - _PROJ_PCT_SWAP) * done / total)
                    _proj_set(pct=min(99, pct), stage="编码",
                              msg="编码 %d/%d 帧" % (done, total))
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                raw = rf.read()[-4000:]
            tail = raw.decode("utf-8", "ignore").strip()
        except OSError:
            tail = ""
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    return {"code": int(code), "tail": tail[-1200:],
            "cmd": " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd)}


def _proj_source_video(raw, project: Path, frames_dir: Path):
    """原视频（fps / 分辨率 / 音轨的唯一来源）：显式 > prepare 记下的 > 项目里唯一的片子。

    找不到就回 (None, 原因) 让调用方报错 —— 这里绝不猜一个 fps 出来。
    """
    cands = []
    raw = str(raw or "").strip()
    if raw:
        p = _norm_path(raw)
        if p is None or not p.is_file():
            raise _BadInput("指定的原视频不存在：%s" % (p or raw), 404)
        cands.append(p)
    for d in (str(frames_dir), str(project)):
        try:
            src = _app_mod._merge_source_video(d)
        except Exception:
            src = ""
        if src:
            cands.append(Path(src))
            break
    for d in (project, project.parent):
        try:
            vids = sorted(x for x in Path(str(d)).iterdir()
                          if x.is_file() and x.suffix.lower() in _VIDEO_EXT)
        except OSError:
            vids = []
        vids = [v for v in vids if ".merge_" not in v.name]
        same = [v for v in vids
                if v.stem in (Path(str(project)).name, Path(str(frames_dir)).name)]
        cands.extend(same or (vids if len(vids) == 1 else []))
    for p in cands:
        try:
            if p is not None and Path(str(p)).is_file():
                return Path(str(p)), ""
        except OSError:
            continue
    return None, "项目目录里找不到原视频（.mp4）—— 用 video 参数指定一个"


def _proj_finish(cfg: dict, project, frames, frame_ext, merged_dir, t0, total):
    """项目导出的收尾：对账（缺帧要说清楚）+ ffmpeg 编码成片。

    老路（_proj_run）和流水线（_proj_pipe.run）共用这一段 —— 成片的 ffmpeg
    命令、缺帧清单、进度口径只有一个实现，不会两条路各写一套再慢慢漂。
    """
    # ---- 编码前对账：缺帧必须说清楚缺哪几张，别让成片悄悄变短 ----
    merged_names = [n for n in _proj_frame_files(merged_dir)
                    if Path(n).suffix.lower() == frame_ext]
    merged_set = set(merged_names)
    frames_set = set(frames)
    missing = [n for n in frames if n not in merged_set]       # 有源帧、没产物
    v_frames = int(cfg.get("video_frames") or 0)
    fps = max(0.001, float(cfg["fps"]))
    # 项目和视频本该一一对应才比较帧数（抽帧按 2fps 之类的本来就对不上）
    near_full = bool(v_frames) and (len(frames) + 10 >= v_frames)
    if v_frames and near_full and len(frames) < v_frames:
        # 按命名规律点名缺了哪几张（编号本来就有洞，contiguous=False）
        info_seq = _proj_seq_info(frames, contiguous=False)
        if info_seq is not None:
            prefix, width, suffix, start, _nums = info_seq
            want = [prefix + str(i).zfill(width) + suffix
                    for i in range(int(start), int(start) + v_frames)]
            gone = [n for n in want if n not in frames_set]
            if gone:
                _proj_warn("源视频 %d 帧，项目里缺 %d 帧：%s（这些帧不会出现在成片里）"
                           % (v_frames, len(gone), "、".join(gone[:12])))
                missing = gone + missing
    _proj_set(missing=missing[: _PROJ_MAX_LIST], merged_frames=len(merged_names))
    if len(merged_names) < len(frames):
        lack = [n for n in missing if n in frames_set]
        _proj_warn("merged 只有 %d 帧 < 源帧 %d 帧 —— 缺 %d 帧%s"
                   % (len(merged_names), len(frames), len(frames) - len(merged_names),
                      ("：" + "、".join(lack[:12])) if lack else ""))
    if v_frames and near_full and len(merged_names) < v_frames:
        _proj_warn("成片只会有 %d 帧（源视频 %d 帧），比原片短 %.2f 秒"
                   % (len(merged_names), v_frames,
                      (v_frames - len(merged_names)) / fps))
    if not merged_names:
        raise _BadInput("merged/ 里一帧都没有，没法编码：%s" % merged_dir, 422)

    # ---- 编码 ----
    _proj_set(stage="编码", pct=_PROJ_PCT_SWAP,
              msg="编码 %d 帧 → %s" % (len(merged_names),
                                       Path(str(cfg["out_video"])).name))
    pat = _proj_pattern(merged_names)
    seq_dir = None
    try:
        if pat is None:
            # 编号有洞（缺帧 / 名字不规则）：按顺序硬链接成 %06d 再编码。
            # 不这么做的话 image2 会在第一个洞停住，成片被截短，而界面上写着「成功」。
            seq_dir = project / "_encode_seq"
            shutil.rmtree(str(seq_dir), ignore_errors=True)
            seq_dir.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(merged_names):
                d = seq_dir / ("%06d%s" % (i, frame_ext))
                try:
                    os.link(str(merged_dir / n), str(d))
                except OSError:
                    shutil.copyfile(str(merged_dir / n), str(d))
            # 注意别写成 "%06d%s" % (...)：那样 %06d 会被当场格式化掉，
            # image2 拿到的是一个具体文件名 → 只编出 1 帧
            pat = ("%06d" + frame_ext, 0)
            _proj_log("帧编号不连续，已按顺序重排 %d 帧到 %s" % (len(merged_names), seq_dir))
        pattern = str(Path(str(merged_dir if seq_dir is None else seq_dir)) / pat[0])
        t_enc = time.time()
        enc = _proj_encode(pattern, pat[1], float(cfg["fps"]), cfg["video"],
                           cfg["out_video"], len(merged_names))
        enc["seconds"] = round(time.time() - t_enc, 2)
        _proj_set(encode=enc)
        _proj_log("编码：%s" % enc.get("cmd"))
        if enc.get("code") != 0:
            raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s"
                            % (enc.get("code"), (enc.get("tail") or "")[-500:]), 500)
    finally:
        if seq_dir is not None:
            shutil.rmtree(str(seq_dir), ignore_errors=True)

    el = time.time() - t0
    _proj_set(done=total, pct=100, stage="完成", msg="完成：%d 帧 → %s"
              % (len(merged_names), Path(str(cfg["out_video"])).name),
              elapsed=round(el, 1), eta_s=0)

def _proj_run(cfg: dict):
    """项目导出：只走流水线实现（docs/21-PROJ-PIPELINE.md）。

    老路（worker 子进程 + MergeMaskedFace，原 1511-2228 行 / 718 行）已删除：
    新流水在真机跑过 4K 与 1080p 的项目导出，帧号/成片与老路一致；留两套只会让排障时
    先问一句「这单走哪套」。要回退请 checkout 提交历史（本次删除的前一个提交）。
    """
    return run(cfg)



def _proj_worker(cfg: dict):
    """线程入口：所有异常在这里落成终态，绝不让状态永远停在 running。"""
    try:
        _proj_run(cfg)
        if cfg.get("downgrade_warn"):
            _proj_warn(str(cfg["downgrade_warn"]))     # 收尾再提一次：status.log 只看最后 12 行
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": True, "phase": "done",
                              "cancelled": False, "finished_at": time.time(),
                              "stage": "完成",
                              "msg": "完成：%d 帧 → %s"
                                     % (int(_PROJ_JOB.get("merged_frames") or 0),
                                        Path(str(cfg["out_video"])).name)})
    except _ProjCancel:
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "cancelled",
                              "cancelled": True, "stage": "已取消", "pct": 0,
                              "msg": "已取消", "finished_at": time.time(),
                              "error": "已取消（merged/ 里已写好的帧留着，下次接着跑）"})
    except _BadInput as e:
        with _PROJ_LOCK:
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "failed",
                              "stage": "失败", "pct": 0, "msg": "导出失败",
                              "error": str(e.msg)[:1200],
                              "finished_at": time.time()})
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        with _PROJ_LOCK:
            log = list(_PROJ_JOB.get("log") or [])
            _PROJ_JOB.update({"running": False, "ok": False, "phase": "failed",
                              "stage": "失败", "pct": 0, "msg": "导出失败",
                              "error": "%s: %s" % (e.__class__.__name__, str(e)[:600]),
                              "error_detail": "\n".join(log[-20:])[:4000],
                              "finished_at": time.time()})


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 proj 与 _app_mod、_projplan、app、media_frames、preview、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from .. import _projplan as _projplan

from ..app import (
    WORKSPACE,
)

from .._media.frames import (
    _VIDEO_EXT,
)

from .preview import (
    _PREVIEW_MAX_BATCH, _PREVIEW_QUALITY, _PreviewWorkerPool, _img_bytes, _preview_finish,
    _preview_prepare, _proj_batch_default, _proj_out_raw, _proj_send_raw, _proj_workers_default,
    _worker_swap,
)

from .vram import (
    _dfm_size_hint, _proj_record_sess_peak, _vram_free_smi, _vram_log_lines, _vram_plan,
)
