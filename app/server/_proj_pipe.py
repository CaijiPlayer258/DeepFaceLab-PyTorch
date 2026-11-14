# -*- coding: utf-8 -*-
"""项目导出 · 「读帧 → 对齐/dfm 攒批 → 合成 → 编码落盘」流水线（默认路线）。

这条路的由来是三个结构性问题的核实（见 app/docs/21-PROJ-PIPELINE.md）：

  ① **重复检测**：老路（_proj_run → _preview_prepare → _frame_landmarks）对每一帧
     重新跑一次检测器 + 2D106 回归，可 aligned/ 里的 DFLJPG 自己就带着
     source_landmarks（原帧 68 点）/ image_to_face_mat / source_rect / face_index
     —— 那是切脸阶段（②）自己写进去的。这一版直接读它，**一帧都不检测**；
     只有连 landmarks 都没有的对齐脸（5 点模式切的）才回退检测，并且计数报出来。
  ② **合成单线程**：老路把合成放在 worker 子进程里，由 MergeMaskedFace 一帧一次
     单线程跑完整帧（seamlessClone 156ms + ct 95ms ≈ 330ms/帧）。这一版改用视频
     fused 那条路同一个合成器 _merge_compose.compose_frame（ROI 版，纯 CPU，
     默认 8 线程并行），dfm 与 XSeg 只在 GPU 那一条线程上跑。
  ③ **没有 dfm 批**：老路一帧一次 dfm 前向（batch=1）。这一版按 dfm 攒批，
     一次喂 4~8 张对齐脸（_dfm_run_batch，和 ③ 换脸阶段同一个函数）。

显存：全程**只有一份** dfm 会话（384 实测 2.3GB）。线程共享它，不像老路那样
一个 worker 一份会话 —— 8GB 卡上那等于把换页余量让出来。

四段 + 一条兜底（段间都是有界队列，乱序完成后按帧序提交）：

    读帧/对齐(n_read)   对齐脸 68 点（DFLJPG）→ warp 到 S → dfm 输入
        ↓ q_warp
    dfm 攒批(1)         _dfm_run_batch(batch) → pred / prd / dst；XSeg → xseg_prd/dst
        ↓ q_gpu
    合成(n_cmp)         _merge_compose.compose_frame（ROI 版）
        ↓ q_cmp
    编码(n_enc)         PNG / JPEG 编码（.png 帧编 PNG、.jpg 帧编 q92 JPEG）
        ↓
    主线程              按帧序提交（.part → os.replace）→ 断点续跑语义不变

取消：只提交「达到过的**前缀**」，merged/ 里绝不会出现中间的洞。
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from pathlib import Path

# 开关：DFL_PROJ_PIPELINE=legacy 回到老路（worker 子进程 + MergeMaskedFace）。
_ENV_PIPE = "DFL_PROJ_PIPELINE"
_ENV_COMPOSE = "DFL_PROJ_COMPOSE"        # 合成线程数（默认 8）
_ENV_READ = "DFL_PROJ_READ"              # 读帧/对齐线程数（默认 3）
_ENV_ENC = "DFL_PROJ_ENC"                # 编码线程数（默认 2）
_ENV_DFM_BATCH = "DFL_PROJ_DFM_BATCH"    # dfm 攒批目标（默认 max(计划 batch, 4)）
_ENV_WAIT = "DFL_PROJ_BATCH_WAIT_MS"     # 攒批超时（默认 20ms）
_ENV_PNG = "DFL_PROJ_PNG_LEVEL"          # .png 帧的压缩级别（默认 3，与老路逐字节同档）

_QDEPTH = 24            # 段间队列深度（和 fused 同口径）
_BATCH_WAIT_MS = 20.0   # 攒批超时：正常出片时到脸间隔 30~80ms
_DFM_BATCH_MIN = 4      # 攒批下限（一次前向至少 4 张）
_DFM_BATCH_MAX = 8      # 攒批上限（再多每张反而更贵，见 20-PROJ-ADAPTIVE 的阶梯）


def enabled() -> bool:
    """走不走流水线。DFL_PROJ_PIPELINE=legacy → 回老路（对拍 / 排障用）。"""
    v = str(os.environ.get(_ENV_PIPE) or "").strip().lower()
    return not (v in ("legacy", "old", "0", "off", "false", "no"))


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
    from . import routes_merge as rm
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
    from . import routes_merge as rm
    import cv2
    import numpy as np
    from ._merge_compose import compose_frame, ensure_engine_path
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
    n_cmp = _int_env(_ENV_COMPOSE, 4, 1, 16)
    n_read = _int_env(_ENV_READ, 4, 1, 8)
    n_enc = _int_env(_ENV_ENC, 6, 1, 8)
    png_level = _int_env(_ENV_PNG, 3, 0, 9)
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
             "faces": 0, "detected": 0}
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
    line = ("流水线：一份 dfm 会话（%s · %dx%d）· 攒批 %s 张 · 合成 %s 线程 · "
            "读帧 %s · 编码 %s · 队列 %s（DFL_PROJ_PIPELINE=legacy 可回退老路）"
            % (Path(plan_paths[0]).name, sz0, sz0, batch, n_cmp, n_read, n_enc,
               _QDEPTH))
    rm._proj_log(line)
    # 这一行同时进 status.pipeline_line：status 的 log 只回最后 12 行，120 帧跑完
    # 它早被「合成 N/120」挤掉了 —— 验收/排障要能随时读到「这一单按什么档跑的」。
    rm._proj_set(workers=1, batch=int(batch), compose=int(n_cmp), pipeline="fused",
                 pipeline_line=line,
                 adaptive={"src": "pipeline", "note": "流水线：档位不再是 worker×线程，"
                                                     "dfm 攒批 + 合成线程见日志"})
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
    it = iter(range(len(frames)))
    it_lock = threading.Lock()
    shut_lock = threading.Lock()
    live = {"read": int(n_read), "cmp": int(n_cmp)}
    q = int(cfg.get("quality") or 92)

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
                    frame = rm._app_mod._imread_any(str(src_p))
                except Exception:                           # noqa: BLE001
                    frame = None
                if frame is None:
                    job["kind"] = "err"
                    job["error"] = "这一帧读不出来（损坏？）"
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
                        lm = face["lm"]
                        if lm is None:
                            lm = _fallback_lm(frame)
                            stats["detected"] += 1
                            if lm is None:
                                continue        # 没点就跳过这张脸
                            face["lm"] = lm
                        dfm = str(models[mi]["dfm"])
                        sz = int(sizes.get(dfm) or 256)
                        fmat = get_transform_mat(np.asarray(lm, np.float32), sz,
                                                 FaceType.WHOLE_FACE, scale=1.0)
                        warped = cv2.warpAffine(frame, fmat, (sz, sz),
                                                flags=cv2.INTER_CUBIC)
                        d256 = None
                        if need_dst:
                            # XSeg 的 dst 画布：和 MergeMaskedFace 里那一趟同序
                            # （原帧 → facelib WHOLE_FACE 画布 → 256）。256 那张矩阵
                            # 就是 S 那张乘 256/S（同一套模板只差缩放，实测 1e-13），
                            # 所以不用再算一次 get_transform_mat。
                            m256 = np.array(fmat, np.float64) * (256.0 / float(sz))
                            d256 = cv2.warpAffine(frame, m256, (256, 256),
                                                  flags=cv2.INTER_CUBIC)
                        job["faces"].append({
                            "k": int(face["k"]), "lm": np.asarray(lm, np.float32),
                            "dfm": dfm, "size": sz, "mi": int(mi),
                            "dfm_in": np.clip(warped.astype(np.float32) / 255.0, 0, 1),
                            "dst256": d256,
                            "pred": None, "prd": None, "dst": None,
                            "xp": None, "xd": None})
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

    # ---------- ③ 合成（compose_frame，ROI 版）----------
    def _composer():
        try:
            while not quit_event.is_set():
                job = _get(q_cmp)
                if job is END:
                    return
                args = []
                for fc in job["faces"]:
                    args.append({"landmarks": fc["lm"], "pred": fc["pred"],
                                 "prd_dfm": fc["prd"], "dst_dfm": fc["dst"],
                                 "xseg_prd": fc["xp"], "xseg_dst": fc["xd"]})
                t_cmp = time.perf_counter()
                try:
                    out = compose_frame(job["frame"], args, comp_cfg,
                                        job.setdefault("timings", {}))
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
                        ok, buf = cv2.imencode(
                            ".png", job["out"],
                            [int(cv2.IMWRITE_PNG_COMPRESSION), png_level])
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
