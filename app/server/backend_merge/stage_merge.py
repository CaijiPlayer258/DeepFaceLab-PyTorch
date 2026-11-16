# -*- coding: utf-8 -*-
"""第 5 段：合成（帧+pred+mask → merged/）。从 pipeline_stages.py 搬出。"""
from __future__ import annotations

from pathlib import Path
import threading
import time

from .pipeline_stages import (
    _BadInput, _DFM_SESS, _INFER_CALLS, _STAGE_METER,
    _StageCancel, _gpu_mem_used_mb, _mask_signature, _mask_src,
    _merge_cv2_threads, _merge_worker_count, _stage_aligned_dir, _stage_cancelled,
    _stage_files, _stage_filter_range, _stage_frames_dir, _stage_imencode_png,
    _stage_log, _stage_merge_face_inputs, _stage_meta_save, _stage_profile_blank,
    _stage_progress, _stage_range, _stage_read_frame, _stage_tick,
    _stage_write_atomic,
)

def _stage_do_merge(cfg: dict) -> dict:
    """阶段 5 · 合成：**纯 CPU**。frames/ + aligned/ + pred/ + mask/ -> 就地覆盖 frames/。

    语义基准 = 4.0 MergeStudio/core/export_pipeline.py 的 _phase_merge_worker()
    （见 app/server/_merge_compose.py 顶部的逐段对照表）：遮罩准备（mask_mode 组合 +
    erode/blur）→ 色彩迁移（ct_mode）→ 融合（overlay / hist-match / seamless /
    seamless-hist-match）→ 反贴回原帧（face_mat 逆变换）→ 写盘。

    这一阶段**不许碰推理**：dfm / XSeg 都已经在 ③④ 落盘，这里只读文件。
    验收会断言「推理调用计数 = 0」且显存不增长（profile 里也有 gpu_mem 前后值）。
    续跑游标 = meta.stages.merge.done_frames（就地覆盖只往前推）。
    """
    import cv2
    import numpy as np
    from .compose import compose_frame

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("merge", {})
    frames_all = _stage_files(_stage_frames_dir(cache))
    if not frames_all:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    # 帧范围：只覆盖 [start, end] 里的帧（就地覆盖 frames/，别的帧一个都不碰）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    frames = _stage_filter_range(rng, frames_all)
    total = len(frames)
    if ((not cfg["force"]) and not rchg and st.get("done")
            and int(st.get("done_frames") or 0) >= total and total):
        _stage_log("合成：复用已完成的结果（第 %d..%d 帧，不重跑）"
                   % (int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": total, "reason": "already-done"}
    for need, label in (("faces", "切脸"), ("pred", "换脸"), ("mask", "遮罩")):
        if not (meta["stages"].get(need) or {}).get("done"):
            raise _BadInput("合成要用 %s 的产物，但「%s」阶段还没完成 —— 先跑%s"
                            "（POST /api/merge/stage/%s）" % (label, label, label, need), 409)
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    sig = _mask_signature(src, cfg)
    mst = meta["stages"].get("mask") or {}
    if mst.get("signature") and mst.get("signature") != sig:
        raise _BadInput("遮罩来源变了（mask/ 是 %s 产出的，现在是 %s）—— 先重跑遮罩阶段"
                        "（force=true），别拿旧遮罩合成" % (mst.get("signature"), sig), 409)

    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    by_frame = {}
    for ap in aligned:
        by_frame.setdefault(ap.stem.split("_")[0], []).append(ap)

    # 游标 **相对范围**：0..total
    cursor = 0 if (rchg or (cfg["force"] and not meta.get("frame_swapped"))) \
        else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    if meta.get("frame_swapped") and cursor == 0 and int(st.get("done_frames") or 0) == 0:
        raise _BadInput("frames/ 已经是合成结果（frame_swapped=true），但 meta 里没有续跑位置 —— "
                        "只能重跑切帧 + 切脸 + 换脸 + 遮罩", 409)
    if meta.get("frame_swapped") and rchg:
        _stage_log("合成：frames/ 已经就地合成过，这次换到第 %d..%d 帧 —— 这一段会在已合成的帧上"
                   "再合成一次；要绝对干净请先 force 重切帧"
                   % (int(rng["start"]), int(rng["end"])))
    if cursor:
        _stage_log("合成：从范围内第 %d 帧续跑（前 %d 帧已经是合成结果）" % (cursor + 1, cursor))

    meta["frame_swapped"] = True
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int((cfg.get("settings") or {}).get("mask_mode", 6) or 0)
    threads = _merge_worker_count(cfg)
    cv2_threads = _merge_cv2_threads(cfg, threads)
    png_level = int(cfg.get("png_level") if cfg.get("png_level") is not None else 1)
    _stage_log("合成（纯 CPU）：第 %d..%d 帧（%d 帧）· 遮罩 %s/%s · ct=%s · blend=%s · "
               "%d 线程（每个 cv2 %d 线程）· PNG level %d"
               % (int(rng["start"]), int(rng["end"]), total,
                  src.get("label") or src.get("source"),
                  comp_cfg.get("mask_mode"), comp_cfg.get("color_transfer_mode"),
                  comp_cfg.get("mode"), threads, cv2_threads, png_level))

    prof = dict(_stage_profile_blank())
    for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms", "encode_ms",
              "write_ms", "frame_ms"):
        prof[k] = float(st.get("profile", {}).get(k) or 0.0)
    prof["frames"] = cursor
    swapped = int(st.get("swapped") or 0)
    copied = int(st.get("copied") or 0)
    no_face = int(st.get("no_face") or 0)
    failed = int(st.get("failed") or 0)
    t0 = time.time()
    start_cursor = cursor
    infer0 = int(sum(_INFER_CALLS.values()))
    sess0 = len(_DFM_SESS)
    gpu0 = _gpu_mem_used_mb()
    stop = threading.Event()
    lock = threading.Lock()

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "frames_all": len(frames_all),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "swapped": swapped, "copied": copied, "no_face": no_face, "failed": failed,
            "resumed_from": int(start_cursor), "in_place": True,
            "aligned_faces": len(aligned), "pure_cpu": True,
            "threads": threads, "png_level": png_level,
            "mask_mode": comp_cfg.get("mask_mode"),
            "ct_mode": comp_cfg.get("color_transfer_mode"),
            "blend": comp_cfg.get("mode"),
            "profile": {k: round(float(v), 2) for k, v in prof.items()},
            "msg": ("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d）"
                    % (int(rng["start"]), int(rng["end"]), total, swapped, copied)) if done
                   else ("合成 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["merge"] = node
        _stage_meta_save(cache, meta)

    def _one(fp):
        """一帧：读图 → 逐脸合成 → PN G编码 → 写盘。返回 (rec, 各段耗时)。"""
        if stop.is_set():
            # 取消后不再开新帧（排队里的那些也直接放掉，别让 shutdown 干等）。
            return {"key": fp.stem, "ok": True, "no_face": True, "skipped": True}, {}
        t = time.time()
        tim = {}
        frame = _stage_read_frame(fp)
        if frame is None:
            return {"key": fp.stem, "ok": False, "why": "读不出来"}, tim
        tim["read_ms"] = (time.time() - t) * 1000.0
        aps = by_frame.get(fp.stem) or []
        faces = []
        for ap in aps:
            f = _stage_merge_face_inputs(cache, ap, cfg)
            if f is not None:
                faces.append(f)
        if not faces:
            return {"key": fp.stem, "ok": True, "no_face": True, "faces": len(aps)}, tim
        t_c = time.time()
        out = compose_frame(frame, faces, comp_cfg, tim)
        tim["frame_ms"] = (time.time() - t_c) * 1000.0
        t_e = time.time()
        data = _stage_imencode_png(out, png_level)
        tim["encode_ms"] = (time.time() - t_e) * 1000.0
        _stage_tick("compose")   # 交接点：合成 → merged/*.png
        t_w = time.time()
        _stage_write_atomic(fp, data)
        tim["write_ms"] = (time.time() - t_w) * 1000.0
        return {"key": fp.stem, "ok": True, "swapped": 1, "faces": len(faces)}, tim

    from concurrent.futures import ThreadPoolExecutor
    chunk = max(threads * 2, 8)
    # 流水线实况：合成 lane 数 = 这个池**实际**的线程数（不分段链写死 8）
    _STAGE_METER.set_lanes({"compose": ["t%d" % _i for _i in range(int(threads))]})
    prog = {"done": cursor}

    def _report():
        """进度 + meta（每 chunk 一次；和以前同一个口径）。"""
        el = max(0.001, time.time() - t0)
        rate = (prog["done"] - start_cursor) / el
        eta = int((total - prog["done"]) / rate) if rate > 0 and prog["done"] < total else None
        n = max(1, prof["frames"] - cursor)
        _stage_progress(
            "merge", 100.0 * prog["done"] / total,
            "合成 %d/%d 帧（第 %d..%d 帧）· %.1f 帧/秒（换脸 %d · 原样 %d）"
            % (prog["done"], total, int(rng["start"]), int(rng["end"]),
               rate, swapped, copied),
            frames_done=prog["done"], frames_total=total,
            frame_fps=round(rate, 2), eta_s=eta,
            swapped=swapped, copied=copied, no_face=no_face,
            profile={"per_frame_ms":
                     {k: round(float(prof.get(k) or 0.0) / n, 2)
                      for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms",
                                "blend_ms", "encode_ms", "write_ms", "frame_ms")},
                     "fps": round(rate, 2)})
        _save(prog["done"])

    # cv2 内部线程数：跟 worker 数一起调（见 _merge_cv2_threads 的注释）。
    cv2_prev = 0
    cv2_set = 0
    if cv2_threads > 0:
        try:
            cv2_prev = int(cv2.getNumThreads())
            if cv2_prev != int(cv2_threads):
                cv2.setNumThreads(int(cv2_threads))
                cv2_set = int(cv2_threads)
        except Exception:                                         # noqa: BLE001
            cv2_prev, cv2_set = 0, 0
    pool = None
    try:
        # **整段一个池**：以前是每个 chunk 重建一次 ThreadPoolExecutor，
        # 每个 chunk 末尾都有一道隐式栅栏（shutdown(wait=True)）—— 最后一批 worker
        # 空等，下一批才开工。8 worker / chunk=16 时白丢 6~12% 吞吐。
        pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="merge")
        done_in_chunk = 0
        # 一次性把所有帧提交进池：worker 干完一帧立刻拿下一帧，没有 chunk 边界。
        for rec, tim in pool.map(_one, frames[cursor:]):
            if _stage_cancelled():
                stop.set()
                raise _StageCancel()
            if rec.get("skipped"):
                continue                  # 取消尾巴：不算帧、不动游标
            if not rec.get("ok"):
                failed += 1
                _stage_log("合成跳过 %s：%s" % (rec.get("key"), rec.get("why")))
            elif rec.get("no_face"):
                copied += 1
                no_face += 1
            else:
                swapped += 1
            prof["frames"] += 1
            prof["faces"] += int(rec.get("faces") or 0)
            for kk, vv in tim.items():
                prof[kk] = float(prof.get(kk) or 0.0) + float(vv or 0.0)
            prog["done"] += 1
            done_in_chunk += 1
            if done_in_chunk >= chunk:
                done_in_chunk = 0
                _report()
        _report()                     # 收尾也报一次（进度到 100%），和以前每 chunk 末尾一致
    except _StageCancel:
        _save(prog["done"])
        _stage_log("合成取消：停在 %d/%d 帧（游标已存）" % (prog["done"], total))
        raise
    finally:
        if pool is not None:
            # 取消时别等已经排队的帧：置 stop + cancel_futures，worker 下一帧开头就退出。
            pool.shutdown(wait=not stop.is_set(), cancel_futures=True)
        if cv2_set:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                                     # noqa: BLE001
                pass

    sec = round(time.time() - t0, 2)
    n = max(1, prof["frames"] - cursor)
    per = {k: round(float(prof.get(k) or 0.0) / n, 2)
           for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms",
                     "encode_ms", "write_ms", "frame_ms")}
    infer_n = int(sum(_INFER_CALLS.values())) - infer0
    gpu1 = _gpu_mem_used_mb()
    sess_after = len(_DFM_SESS)
    fps = round((total - start_cursor) / sec, 2) if sec > 0 else 0.0
    prof_out = {
        "per_frame_ms": per,
        "frames": int(prof["frames"]), "faces": int(prof["faces"]),
        "threads": threads, "png_level": png_level,
        "fps": fps, "pure_cpu": True,
        "inference_calls": infer_n,
        "dfm_sessions_before": sess0, "dfm_sessions_after": sess_after,
        "gpu_mem_mb_before": gpu0, "gpu_mem_mb_after": gpu1,
        "wall_s": sec,
    }
    _save(total, done=True, extra={"seconds": sec, "profile": prof_out,
                                   "fps": fps, "inference_calls": infer_n})
    _stage_log("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d，无人脸 %d · 失败 %d）· "
               "%.2f s · %.2f 帧/秒"
               % (int(rng["start"]), int(rng["end"]), total, swapped, copied,
                  no_face, failed, sec, fps))
    _stage_log("合成 profile（每帧 ms）：%s"
               % " · ".join("%s=%.2f" % (k, v) for k, v in per.items()))
    _stage_log("合成阶段推理调用：%d（必须为 0）· 显存 %s → %s MB"
               % (infer_n, gpu0, gpu1))
    if infer_n:
        _stage_log("警告：合成阶段发生了 %d 次推理 —— 违反「纯 CPU」约定" % infer_n)
    return {"count": total, "swapped": swapped, "copied": copied, "no_face": no_face,
            "failed": failed, "seconds": sec, "fps": fps,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "inference_calls": infer_n, "profile": per}
