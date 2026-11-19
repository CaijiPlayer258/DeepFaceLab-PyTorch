# -*- coding: utf-8 -*-
"""视频导出作业：/api/merge/export 的状态机、逐帧换脸贴图、日志尾巴、取消看门狗。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from .._common.http import (
    _BadInput, _as_dict, _bool_val, _ok,
)
from .._common.proc import (
    _kill_proc_tree,
)


# ============================================================
# 合成子进程的收尾 / 失败信息（stderr 尾巴）
# ============================================================
_ERR_HINT = re.compile(r"(Traceback|Error|error\b|Exception|failed|失败|"
                       r"No such|not found|[Cc]annot|CUDA|out of memory|内存|"
                       r"ModuleNotFound|DLL)", re.I)


def _log_tail(log, n=20):
    lines = [str(x).rstrip() for x in (log or []) if str(x).strip()]
    return lines[-n:]


def _brief_failure(log, n=4) -> str:
    """从 stdout+stderr 合流日志里挑一条能当错误摘要的行。

    子进程硬崩（DLL 缺失 / CUDA OOM / python 直接 traceback 退出）时不会打
    "@@ERR"，父进程以前只能说一句"进程退出码 1"，用户完全不知道为什么。
    """
    tail = _log_tail(log, 60)
    if not tail:
        return ""
    hits = [ln for ln in tail if _ERR_HINT.search(ln)]
    pick = (hits or tail)[-n:]
    return " ｜ ".join(x.strip()[:200] for x in pick)[:600]


def _finalize():
    """合成收尾：把 running / ok / stage / pct / msg 收敛成终态。

    _merge_worker 正常结束时只写 running/ok（成功时补 pct=100/msg），失败时
    pct、stage、msg 全停在"跑到一半"的样子，前端就会把一次崩掉的任务显示成
    47% 还在跑。这里统一收尾。
    """
    with _merge_job_lock:
        _merge_job["running"] = False
        _merge_job["finished_at"] = time.time()
        cancelled = bool(_merge_job.get("cancelled"))
        ok = bool(_merge_job.get("ok"))
        log = list(_merge_job.get("log") or [])
        _merge_job["error_detail"] = "\n".join(_log_tail(log, 30))[:4000]
        if ok:
            _merge_job["pct"] = 100
            if _merge_job.get("stage") in (None, -1):
                _merge_job["stage"] = 6          # 4.0 的最后一步是编码
            _merge_job["msg"] = ("合成完成（取消命令来晚了，成片已生成）" if cancelled
                                 else "合成完成")
            return
        _merge_job["pct_at_failure"] = int(_merge_job.get("pct") or 0)
        _merge_job["pct"] = 0                    # 别让调用方把失败读成"还在 47%"
        _merge_job["stage"] = -1
        if cancelled:
            _merge_job["msg"] = "已取消"
            _merge_job["error"] = "已取消合成（用户手动停止）"
            return
        brief = _brief_failure(log)
        err = str(_merge_job.get("error") or "").strip()
        if not err:
            err = brief or "合成失败：子进程没有给出原因（见 error_detail）"
        elif brief and brief not in err:
            err = err + " ｜ " + brief
        _merge_job["msg"] = "合成失败"
        _merge_job["error"] = err[:1200]


def _run_export(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx,
                fps, workers, encoder, hwaccel, detector=""):
    """合成线程。

    _merge_worker 在起子进程之前就会抛（找不到引擎 python / MergeStudio / 桥脚本），
    以前线程里抛出去没人接：状态永远停在 running，前端一直转圈、取消也点不动。
    """
    try:
        _merge_worker(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx,
                      fps, workers, encoder, hwaccel, detector=detector)
    except Exception as e:                       # noqa: BLE001
        traceback.print_exc()
        with _merge_job_lock:
            _merge_job["ok"] = False
            if not _merge_job.get("error"):
                _merge_job["error"] = "合成没起来：%s" % (str(e)[:300] or e.__class__.__name__)
    _finalize()


_cancel_guard = threading.Lock()
_cancel_watching = False                        # 看门狗只留一个，重复取消不叠加线程


def _ensure_cancel_watch():
    global _cancel_watching
    with _cancel_guard:
        if _cancel_watching:
            return
        _cancel_watching = True
    threading.Thread(target=_cancel_watch, daemon=True, name="merge-cancel").start()


def _cancel_watch():
    """取消的看门狗：进程还没起来就等它起来补刀，worker 卡住就兜底收尾。

    取消请求落地时子进程可能还没 Popen 出来（线程刚起、还在 import MergeStudio），
    以前这种情况直接返回 "no running export"，用户点了取消什么也没发生。
    """
    global _cancel_watching
    try:
        deadline = time.time() + 60
        killed_pid = None
        while time.time() < deadline:
            if not _merge_job.get("running"):
                break
            proc = getattr(_app_mod, "_merge_proc", None)
            try:
                if (proc is not None and proc.poll() is None
                        and getattr(proc, "pid", None) != killed_pid):
                    killed_pid = getattr(proc, "pid", None)
                    _kill_proc_tree(proc)
            except Exception:
                pass
            time.sleep(0.3)
        with _merge_job_lock:
            if not _merge_job.get("running"):
                return                           # _finalize 已经写好终态
            if _merge_job.get("cancelled"):
                _merge_job.update({"running": False, "ok": False, "stage": -1,
                                   "pct_at_failure": int(_merge_job.get("pct") or 0),
                                   "pct": 0, "msg": "已取消",
                                   "error": (_merge_job.get("error")
                                             or "已取消合成（用户手动停止）"),
                                   "finished_at": time.time()})
    finally:
        with _cancel_guard:
            _cancel_watching = False


# ============================================================
# 视频模式导出（不需要先建 DFL 项目）
# ------------------------------------------------------------
# 用户定的链路，一步一步照它实现（别加前置条件）：
#   读源视频 -> 逐帧解码 -> 检测 -> 68 点 -> 决定模型 -> dfm 推理 -> 遮罩推理
#   -> 合成 -> 写出帧 -> ffmpeg 合成视频（音轨 copy）
#
# 「哪张脸用哪个 dfm」只有一处判定，见 _vid_models_for_frame()：
#   * 没有人脸库（库不存在 / 库里没有任何组 / 组里一张有特征的脸都没有）
#     -> 不做 embedding 判定，**所有脸都用左栏当前选中的那个 dfm**
#        （models=[{faces: []}]，空 faces 就是「我管全部脸」）；
#   * 有人脸库 -> _auto_face_plan()：每张脸算 ArcFace embedding，和每个组的均值
#     比余弦，最大且 >= 0.35 归该组、用该组的 dfm；**不匹配的脸进 no_swap
#     （保持原像素）**，绝不兜给任何一个模型。
# 无人脸的帧：检测之后一步都不跑（不算 68 点、不跑 dfm），原帧直出。
#
# 进度 / 取消沿用 /api/merge/export 那一套（_merge_job + /export/status +
# /export/cancel），前端拿到的进度字段一个都没变，只是多了 vid 那一段明细。
# ============================================================
_VID_PCT_SWAP = 92                 # 逐帧合成占 0~92%，剩下 8% 给编码
_VID_MAX_LOG = 4000                # 逐帧日志保留上限（超了丢最老的）
_VID_MAX_RECORDS = 20000           # 逐帧记录（给验收 / 排障看）


# 逐帧日志同时落一份到 <成片>.log：状态接口只回最后 50 行，整片每帧都走了 dfm
# 这件事得有个能 grep 的地方（用户要的是「日志显示每帧都走了 dfm」）。
_VID_LOG_LOCK = threading.Lock()
_VID_LOG_FH = {"fh": None, "path": ""}


def _vid_open_log(path: Path) -> str:
    """开一份逐帧日志（写不进去就算了，不影响导出）。"""
    try:
        fh = open(str(path), "w", encoding="utf-8")
    except OSError:
        return ""
    with _VID_LOG_LOCK:
        _VID_LOG_FH["fh"] = fh
        _VID_LOG_FH["path"] = str(path)
    return str(path)


def _vid_close_log() -> str:
    with _VID_LOG_LOCK:
        fh = _VID_LOG_FH.get("fh")
        _VID_LOG_FH["fh"] = None
        path = str(_VID_LOG_FH.get("path") or "")
        _VID_LOG_FH["path"] = ""
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass
    return path


def _vid_log(msg):
    line = str(msg)[:300]
    with _VID_LOG_LOCK:
        fh = _VID_LOG_FH.get("fh")
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except OSError:
                pass
    with _merge_job_lock:
        log = _merge_job.get("log")
        if not isinstance(log, list):
            log = []
            _merge_job["log"] = log
        log.append(line)
        if len(log) > _VID_MAX_LOG:
            del log[:-_VID_MAX_LOG]


def _vid_stats(**kw):
    with _merge_job_lock:
        st = dict(_merge_job.get("stats") or {})
        st.update(kw)
        _merge_job["stats"] = st


def _vid_record(rec: dict):
    with _merge_job_lock:
        rows = _merge_job.get("frames")
        if not isinstance(rows, list):
            rows = []
            _merge_job["frames"] = rows
        rows.append(dict(rec))
        if len(rows) > _VID_MAX_RECORDS:
            del rows[:-_VID_MAX_RECORDS]


def _vid_cancelled() -> bool:
    with _merge_job_lock:
        return bool(_merge_job.get("cancelled"))


def _vid_models_for_frame(cfg: dict, frame, boxes, idx: int):
    """（判定点）这一帧每张脸用哪个 dfm。**只有这一处**决定换不换、用谁换。

    返回 (models, no_swap, plan)：
      plan is None —— 没有人脸库：models = [左栏选中的那个 dfm，faces=[] 管全部脸]，
                     no_swap 空，所有脸都用它换；
      plan 有值    —— 有库：models = 命中的那些组（每条的 faces 是按检测顺序的脸序号），
                     no_swap = 谁都不像的脸（保持原像素，不兜给任何模型）；
                     plan["models"] 为空 = 这一帧没一张脸匹配上 -> 整帧不换。
    """
    singles = list(cfg["single_models"] or [])
    # ---- 显式点名优先（和预览同一套契约）----
    # payload 里某条 models 带非空 faces = 用户在合成器上逐脸点了模型。这时不能再让库的
    # embedding 覆盖它 —— 否则就是「预览按 chip 走、导出按库走」，用户看到预览对、导出错。
    _explicit = [m for m in singles if (m.get("faces") or [])]
    if _explicit:
        _all = bool(any(not (m.get("faces") or []) for m in singles))
        _named = set()
        for _m in _explicit:
            _named.update(int(x) for x in (_m.get("faces") or []))
        _skip = [] if _all else [int(_bi) for _bi in range(len(boxes or []))
                                 if int(_bi) not in _named]
        return singles, _skip, None
    plan = _auto_face_plan(cfg["plan_payload"], frame, cfg["snap"], "video", "",
                          int(idx), cfg["detector"], cfg["thresh"], boxes=boxes)
    if plan is None:
        return singles, [], None
    return list(plan.get("models") or []), list(plan.get("unmatched") or []), plan


def _vid_encode(cmd: list, part: Path, total: int, on_progress=None,
                cancelled=None) -> dict:
    """跑一条 ffmpeg 编码命令（cmd 末尾已经是 part 路径）→ {code, tail, seconds}。

    取消时收掉进程树再抛 _StageCancel（分段链和快速导出都按「取消不是失败」处理）。
    失败时**不动 dest**：.part 由调用方删 —— 半截文件绝不能看起来像成片。
    """
    cancelled = cancelled or _stage_cancelled
    t0 = time.time()
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
                    if cancelled():
                        _kill_proc_tree(proc)
                        raise _StageCancel()
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
                    if on_progress is not None:
                        on_progress(done)
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                tail = rf.read()[-4000:].decode("utf-8", "ignore").strip()
        except OSError:
            tail = ""
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    return {"code": int(code), "tail": tail,
            "seconds": round(time.time() - t0, 2), "part": str(part), "cmd": list(cmd)}


def _start_video_export(body: dict):
    """POST /api/merge/export 的视频分支。**两条路，按 fused 选**：

      · fused=1（默认，快速导出）：_fused_run 的全内存流水线 —— 解码 → 检测 → 对齐/归组
        → dfm → XSeg → 合成 → 编码 同时跑，段间只有有界队列，**不落中间产物**
        （<视频名>_cache/ 一个都不建）。原型（app/tools/bench/fused_pipeline_bench.py）同段
        1080p 120 帧量到 20.55 帧/秒（分段 7.20）；这份生产实现的验收见
        app/tools/accept/fused_export_accept.py：同一段片 18.21 帧/秒（分段 7.20 → 2.5x），
        换 fast 档检测器 20.87 —— 硬底是 retinaface_10g 单帧 47~50ms（它跑在
        onnxruntime 上，workspace/_engines 里那个 retinaface TRT 引擎 app 层没接）。
        代价：不支持续跑（取消就整段重跑）。
      · fused=0（分段链）：切帧 → 切脸 → 换脸 → 遮罩 → 合成 → 导出，每步独立产物、
        可单跑可续跑（<视频名>_cache/ 成功后删除）。留给排查和对照。

    两条路的进度都镜像到 /api/merge/export/status（老前端和既有验收脚本继续能用），
    fused 还额外回显 frame_budget / 逐段耗时。
    """
    body = _as_dict(body)
    fused = _fused_on(body)
    # ---- 多模型分趟（docs/37）：命中就不再走快速导出那条六段并行 ----
    # 旧路径让每个 dfm 各持一份会话（每条 lane 再乘一份），8GB 卡上 warm 会卡死。
    # 分趟是「一趟只装一个模型」，显存与模型数解耦。
    _dfm_set = set()
    for _m in (body.get("models") or []):
        _p = str((_m or {}).get("dfm") or "").strip()
        if _p:
            _dfm_set.add(_p)
    # force_passes：调试/验收用，单模型也走分趟 —— 只有它能和快速导出做「同分配对拍」。
    # 让"这一单到底走哪条路"可见：之前前端没有任何地方能看出后端选了 fused，
    # 用户选了 2 个模型却看到六列图时无从判断是"没传上去"还是"传上去但被判成 1 个"。
    # 后端再判一次：接口只拿到 1 个 dfm（或一个都没有）时，看**人脸库**里有几个组。
    # 库里有两个组 = 两个不同模型要换不同的人，这就是 passes 的判据；
    # 以前只看 body.models，前端少传一个就整单退化成 fused（用户看到"池没了"）。
    _lib_dfms = []
    try:
        from .compose import _auto_library
        from .. import _faces as _store
        _lib = _auto_library(body)
        if _lib is not None:
            for _g in (_store.load_index(_lib).get("groups") or []):
                _p2 = str(_g.get("dfm") or "").strip()
                if _p2 and _p2 not in _lib_dfms:
                    _lib_dfms.append(_p2)
        for _p2 in _lib_dfms:
            if _p2 not in _dfm_set:
                _dfm_set.add(_p2)
    except Exception as _e:                                    # noqa: BLE001
        _vid_log("导出模式：读人脸库失败（%s）" % str(_e)[:100])
    _names = [str((m or {}).get("name") or "") for m in (body.get("models") or [])]
    _vid_log("导出模式：distinct dfm=%d · models=%s → %s"
             % (len(_dfm_set), (_names + ["库:" + len(_lib_dfms).__str__()]),
                "分趟（passes）" if (len(_dfm_set) > 1 or _bool_val(body.get("force_passes"), False))
                else "快速导出（fused）"))
    if fused and (len(_dfm_set) > 1 or _bool_val(body.get("force_passes"), False)):
        return _start_passes_export(body)
    cfg = _fused_cfg(body) if fused else _stage_cfg(body, "frames", chain=True)
    if not fused:
        cfg["stages"] = list(_STAGE_ORDER)
    # ---- 显存闸门：**在这里**判（HTTP 400），不能在 worker 线程里判 ----
    # 线程里抛 _BadInput 只会变成「任务失败」，前端拿不到 400，也拿不到那句
    # 「需要 X GB / 可用 Y GB / 建议」。真正的加载在 _fused_run / _stage_worker 里，
    # 这里只算账（_vram_plan 不加载任何模型）—— 拒绝时**不建 job、不开线程**。
    #
    # 两条路都要判（用户口径，截图 2）：
    #   · fused  —— 帧预算 + dfm 批是它的两个旋钮，按 mode=frames 算；
    #   · 分段链 —— 一次一帧一张脸，没有帧缓冲，按 mode=none（一份会话）算。
    # 参与这一单的**全部** dfm：以前写成 cfg["dfm"] or [...]，只要 cfg["dfm"] 有值就
    # 只算其中一个 —— 双模型导出时显存账少算一整个模型，计划说装得下、真跑就换页。
    _dfms = []
    for _p in ([cfg.get("dfm")] + [m.get("dfm") for m in (cfg.get("single_models") or [])]):
        _p = str(_p or "").strip()
        if _p and _p not in _dfms:
            _dfms.append(_p)
    # force = 用户在拒绝面板上点了「仍然尝试（可能 OOM）」：装不下也放行最小档。
    # 只放行、不改账（plan.forced=True 且 need/avail 照旧），真 OOM 由 worker 里的
    # _vram_oom_rollback 收现场并回到拒绝态 —— 见 _fused_worker / _stage_worker。
    _force = _bool_val(body.get("force"), False)
    # dry_run 是预检（前端拖一次滑杆就打一次）：**不卸**预览模型，靠「可回收计入可用」
    # 拿到同样的结论；真起手这一拍才卸（reclaim_first），因为那时模型才真的要腾地方。
    _dry = _bool_val(body.get("dry_run"), False)
    if fused:
        plan = _vram_guard(_dfms,
                           {"mode": "frames", "frames": int(cfg.get("frame_budget_want") or 0),
                            "batch": int(cfg.get("fused_batch") or 0), "procs": 1,
                            "reclaim_first": (not _dry)},
                           label="快速导出", force=_force)
    else:
        plan = _vram_guard(_dfms, {"mode": "none", "procs": 1,
                                   "reclaim_first": (not _dry)},
                           label="分段导出", force=_force)
    cfg["vram_plan"] = plan
    cfg["vram_force"] = bool(plan.get("forced"))
    if _dry:
        # 预检（前端在起 job 之前问一次）：只回计划，不建 job、不开线程、不碰 status。
        return _ok(dry_run=True, fused=bool(fused), stages=(["fused"] if fused
                                                           else list(_STAGE_ORDER)),
                   video=str(cfg["video"]), output=str(cfg["out"]),
                   vram_plan=dict(plan), need_mb=plan.get("need_mb"),
                   avail_mb=plan.get("avail_mb"), free_mb=plan.get("free_mb"),
                   suggest=list(plan.get("suggest") or []),
                   frames=int(plan.get("frames") or 0),
                   batch=int(plan.get("batch") or 0),
                   on_cpu=bool(_vram_cpu_on()))
    with _STAGE_LOCK:
        if _STAGE_JOB.get("running"):
            raise _BadInput("已经有阶段在跑（%s）—— 等它结束或点取消"
                            % _STAGE_LABEL.get(_STAGE_JOB.get("stage"), "?"), 409)
    started = time.time()
    with _merge_job_lock:
        if _merge_job.get("running"):
            raise _BadInput("已经在导出（%s）—— 等它结束或点取消"
                            % (_merge_job.get("out") or "上一个任务"), 409)
        _merge_job.update({
            "running": True, "active": True,
            "log": [], "ok": None, "error": None, "error_detail": "",
            "out": str(cfg["out"]), "pct": 0,
            "msg": ("启动快速导出（融合流水线）…" if fused else "启动导出…"),
            "stage": 0,
            "cancelled": False, "pct_at_failure": None, "started_at": started,
            "finished_at": None, "fps": float(cfg.get("fps") or 0.0), "workers": 1,
            "encoder": "libx264", "hwaccel": "", "dfm": str(cfg["dfm"]),
            "mode": "video", "video": str(cfg["video"]), "total": int(cfg.get("total") or 0),
            "frames_done": 0, "frames": [], "stats": {}, "cache": {}, "encode": {},
            "fused": bool(fused), "frame_budget": None, "cache_dir": str(cfg.get("cache") or ""),
            "vram_plan": dict(cfg.get("vram_plan") or {}),
            # 坏帧降级（DFL_FUSED_SKIP_BAD）：上一单跳过的帧号绝不能被当成这一单的
            # —— 和上面 frames/stats 一起清，否则新一单刚开始就会显示上一单的「跳过 N 帧」。
            "skip_bad": False, "skipped_count": 0, "skipped_frames": [],
        })
    with _STAGE_LOCK:
        _STAGE_JOB.update({
            "running": True, "ok": None, "cancelled": False, "error": "", "error_detail": "",
            "phase": "running",
            "stage": ("fused" if fused else "frames"),
            "stage_index": (-1 if fused else 0), "stage_pct": 0,
            "pct": 0,
            "msg": ("启动快速导出（融合流水线）…" if fused else "启动导出（分段）…"),
            "video": str(cfg["video"]),
            "cache_dir": str(cfg.get("cache") or ""), "out_video": str(cfg["out"]),
            "frames_total": int(cfg.get("range_total") or cfg.get("total") or 0),
            "range": dict(cfg.get("range") or {}), "frames_done": 0,
            "frame_fps": 0.0, "eta_s": None, "swapped": 0, "copied": 0, "no_face": 0,
            "stages_done": ([] if fused else _stage_done_list(cfg["meta"])),
            "log": [], "meta": ({} if fused else cfg["meta"]),
            "seconds": {}, "skipped": [], "cache_removed": False,
            # ★ 上面那个 skipped 是「跳过**阶段**」的名单（同名不同物）；
            # 坏帧降级用的是 skipped_count / skipped_frames（帧），一起清干净
            "skip_bad": False, "skipped_count": 0, "skipped_frames": [], "skip_hint": "",
            "counts": ({"frames": 0, "faces": 0, "pred": 0, "masks": 0} if fused
                       else _stage_counts(cfg["cache"])),
            "started_at": started,
            "finished_at": None, "chain": True,
            "keep_cache": bool(cfg.get("keep_cache")), "force": bool(cfg.get("force")),
            "frames": [], "fused": bool(fused), "frame_budget": None, "resumed": False,
            "vram_plan": dict(cfg.get("vram_plan") or {}),
            "eff": None, "eff_warnings": [], "warnings": [],
        })
    # ---- 生效配置：一行 [eff] + 一致性自检 ----
    # 用户要的就是这一行：把「我设的」和「它在跑的」摆在一起，对不上就 [eff-warn]。
    eff = None
    try:
        eff = _eff_publish("/api/merge/export", ("fused" if fused else "staged"), body, cfg)
        patch = _eff_status_patch(eff)
        with _merge_job_lock:
            _merge_job.update(patch)
        with _STAGE_LOCK:
            _STAGE_JOB.update(patch)
    except Exception as e:                                  # noqa: BLE001
        print("[eff] 生成失败（不影响这一单）：%s: %s"
              % (e.__class__.__name__, str(e)[:200]), flush=True)
    cfg["log_file"] = _vid_open_log(Path(cfg["out"]).with_suffix(".log"))
    # 生效配置也贴进产物日志（<out>.log）：这是用户报 bug 时最常附的那份文件
    if eff:
        try:
            _vid_log("[eff] " + _eff.line_of(eff))
            for _w in (eff.get("warnings") or []):
                _vid_log("[eff-warn] " + str(_w.get("text") or _w))
        except Exception:                                   # noqa: BLE001
            pass
    # 帧图编码口径（两条路同一句）：PNG 无损 / JPEG 固定 quality=100 + 4:4:4。
    # 设置页那颗「质量」旋钮已经删掉 —— 后端不再从界面读它，日志里照旧回显真实值。
    _vid_log("帧图编码：PNG 无损（compression level %s）/ JPEG quality=%d + 4:4:4"
             "（界面没有该旋钮，固定质量；PNG 不受影响）"
             % (cfg.get("png_level"), int(cfg.get("jpeg_quality")
                                          or _STAGE_JPEG_QUALITY)))
    if fused:
        threading.Thread(target=_fused_worker, args=(cfg,), daemon=True,
                         name="merge-export-fused").start()
        _vid_log("导出链路（快速导出 · 融合流水线）：%s -> %s"
                 % (cfg["video"], cfg["out"]))
        _vid_log("中间产物：不落 <视频>_cache/（帧只在 %%TEMP%% 里经过，成功/取消都删）")
        _vid_log("续跑：不支持 —— 取消就是整段重跑；要续跑请把「快速导出」关掉走分段链")
        _vid_log("帧范围：第 %d..%d 帧（%d 帧）%s"
                 % (int((cfg.get("range") or {}).get("start") or 1),
                    int((cfg.get("range") or {}).get("end") or 0),
                    int(cfg.get("range_total") or 0),
                    "· 全片" if cfg.get("range_full") else "· 只处理这一段"))
        return _ok(mode="video", fused=True, staged=False, stages=["fused"],
                   video=str(cfg["video"]), output=str(cfg["out"]),
                   output_dir=str(cfg["out"]), cache_dir="",
                   no_cache=True, resumable=False,
                   frame_budget=None, frame_budget_want=int(cfg.get("frame_budget_want") or 0),
                   vram_plan=dict(cfg.get("vram_plan") or {}),
                   on_cpu=bool(_vram_cpu_on()),
                   dfm=str(cfg["dfm"]), model=str(cfg["model"]),
                   settings=dict(cfg.get("settings") or {}),
                   detector=str(cfg["detector"]), thresh=float(cfg.get("thresh") or 0.5),
                   xseg=str(cfg["xseg"]), fps=float(cfg.get("fps") or 0.0),
                   range=dict(cfg.get("range") or {}),
                   range_key=str(cfg.get("range_key") or ""),
                   face_threshold=cfg.get("face_threshold"),
                   library=cfg.get("library") or "",
                   models=[{"name": m.get("name"), "dfm": m.get("dfm"),
                            "faces": list(m.get("faces") or [])}
                           for m in cfg["single_models"]])
    threading.Thread(target=_stage_worker, args=(cfg,), daemon=True,
                     name="merge-export-video").start()
    _vid_log("导出链路（分段）：%s -> 切帧 -> 切脸 -> 合成 -> 导出 -> %s"
             % (cfg["video"], cfg["out"]))
    _vid_log("cache：%s（导出成功后整目录删除）" % cfg["cache"])
    return _ok(mode="video", fused=False, staged=True, stages=list(_STAGE_ORDER),
               video=str(cfg["video"]), output=str(cfg["out"]),
               output_dir=str(cfg["out"]), cache_dir=str(cfg["cache"]),
               dfm=str(cfg["dfm"]), model=str(cfg["model"]),
               settings=dict(cfg.get("settings") or {}),
               detector=str(cfg["detector"]), thresh=float(cfg.get("thresh") or 0.5),
               xseg=str(cfg["xseg"]), fps=float(cfg.get("fps") or 0.0),
               face_threshold=cfg.get("face_threshold"), library=cfg.get("library") or "",
               models=[{"name": m.get("name"), "dfm": m.get("dfm"),
                        "faces": list(m.get("faces") or [])}
                       for m in cfg["single_models"]])


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 pipeline_video 与 _app_mod、_eff、app、compose、eff、fused、stages、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from .. import _eff as _eff

from ..app import (
    _merge_job, _merge_job_lock, _merge_worker,
)

from .compose import (
    _auto_face_plan,
)

from .eff import (
    _eff_publish, _eff_status_patch,
)

from .pipeline_fused import (
    _fused_cfg, _fused_on, _fused_worker,
)

from .pipeline_stages import (
    _STAGE_JOB, _STAGE_JPEG_QUALITY, _STAGE_LABEL, _STAGE_LOCK, _STAGE_ORDER, _StageCancel,
    _stage_cancelled, _stage_cfg, _stage_counts, _stage_done_list, _stage_worker,
)

from .vram import (
    _vram_cpu_on, _vram_guard,
)

# ============================================================
# 多模型分趟导出（docs/37）
# ------------------------------------------------------------
# 握手点只有一个：_start_video_export 里判定「distinct dfm > 1」就转到这里。
# 进度复用 _stage_progress(stage="fused") —— 前端那张车道图不用改就能显示。
# ============================================================
def _start_passes_export(body: dict):
    from . import pipeline_passes as _pp
    body = _as_dict(body)
    cfg = _fused_cfg(body)
    if os.environ.get("DFL_AB"):
        print("[ab] passes out=%s · config=%r" % (cfg.get("out"), dict(cfg.get("config") or {})), flush=True)
    cfg["start"] = int(body.get("start_frame") or 0)
    cfg["end"] = int(body.get("end_frame") or 0) or None
    cfg["crf"] = int(body.get("crf") or 17)
    cfg["encoder"] = str(body.get("encoder") or "")
    # 分相位链路：趟是**串行**的，六列（并发视图）表达不了 —— 这是前端缺的那件事。
    try:
        _models_n = len([m for m in (cfg.get("single_models") or []) if m.get("dfm")])
        with _merge_job_lock:
            _merge_job["pipeline"] = {"shape": "passes", "phases": [
                {"id": "scan", "label": "扫描", "steps": ["解码", "检测", "68点", "embedding → 分配表"]},
                {"id": "pass", "label": "逐模型趟", "count": _models_n,
                 "steps": ["解码", "对齐", "换脸", "遮罩", "写人脸池"]},
                {"id": "gate", "label": "闸门", "steps": ["等所有趟写完"]},
                {"id": "merge", "label": "合流", "steps": ["读池", "合成", "编码", "成片"]},
            ]}
    except Exception:                                          # noqa: BLE001
        pass
    # 命名统一交给 _stage_out_path（_fused_cfg 已经算好）；要区分两单用 out_tag，
    # 不要在这里另起一套 —— 那正是上一轮对拍拿不到可信产物原因。
    with _merge_job_lock:
        if _merge_job.get("running"):
            raise _BadInput("已经在导出（%s）—— 等它结束或点取消"
                            % (_merge_job.get("out") or "上一个任务"), 409)
        _merge_job.update({"running": True, "active": True, "log": [], "ok": None,
                           "error": None, "error_detail": "", "out": str(cfg["out"]),
                           "pct": 0, "msg": "分趟导出：启动 …", "passes": True})
    def _run():
        try:
            def _prog(pct, msg, done=None, total=None, meta=None):
                with _merge_job_lock:
                    _merge_job["pct"] = int(pct)
                    _merge_job["msg"] = str(msg)
                    if done is not None:
                        _merge_job["frames_done"] = int(done)
                    if total:
                        _merge_job["frames_total"] = int(total)
                    if meta:
                        _merge_job["phase"] = str(meta.get("phase") or "")
                        _merge_job["pass"] = {"i": int(meta.get("pass_i") or 0),
                                              "n": int(meta.get("pass_n") or 0),
                                              "label": str(meta.get("pass_label") or "")}
                        _merge_job["pool"] = {"faces": int(meta.get("pool") or 0),
                                              "faces_total": int(meta.get("faces_total") or 0),
                                              "kind": "disk"}
                try:
                    _stage_progress("fused", int(pct), str(msg),
                                    frames_done=(int(done) if done is not None else None),
                                    frames_total=(int(total) if total else None))
                except Exception:                          # noqa: BLE001
                    pass
            res = _pp.run(cfg, log=_vid_log, progress=_prog,
                          cancelled=lambda: bool(_merge_job.get("cancel")))
            with _merge_job_lock:
                _merge_job.update({"running": False, "ok": True, "pct": 100,
                                   "msg": "分趟导出完成：%d 帧" % int(res.get("frames") or 0),
                                   "finished_at": time.time()})
        except BaseException as e:                             # noqa: BLE001
            with _merge_job_lock:
                _merge_job.update({"running": False, "ok": False,
                                   "error": str(e)[:300],
                                   "msg": "分趟导出失败：%s" % str(e)[:160],
                                   "finished_at": time.time()})
    threading.Thread(target=_run, name="passes-export", daemon=True).start()
    return _ok(ok=True, fused=False, passes=True, mode="passes", out=str(cfg["out"]),
               vram_plan={}, msg="分趟导出已启动")
