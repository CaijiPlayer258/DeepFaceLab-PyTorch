# -*- coding: utf-8 -*-
"""第 3 段：换脸（对齐脸 → pred/ 的 PNG+meta）。从 pipeline_stages.py 搬出。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import os
import queue
import threading
import time

from .pipeline_stages import (
    _BadInput, _PRED_VRAM_SAFETY, _STAGE_METER, _StageCancel,
    _dfm_run_batch, _dfm_session, _is_oom, _pred_pick_batch,
    _stage_aligned_dir, _stage_check_cancel, _stage_face_rows, _stage_files,
    _stage_frame_path, _stage_frames_dir, _stage_imencode_png, _stage_in_range,
    _stage_log, _stage_meta_save, _stage_pred_assign, _stage_pred_dir,
    _stage_progress, _stage_range, _stage_read_frame, _stage_tick,
    _stage_write_atomic,
)

def _stage_do_pred(cfg: dict) -> dict:
    """阶段 3 · 换脸：frames/ + aligned/ -> pred/（换脸正脸 + dfm 自带两张遮罩）。

    语义照 4.0 export_pipeline.py:769（fo = warpAffine(frame, get_transform_mat(lm,
    swap_sz, WHOLE_FACE)) → dfm 前向）。

    **吞吐**（本机实测，1080×1920、每帧 1 张脸、app/tools/bench/stage_perf_profile.py）：
        read 28.5ms/张（4 线程摊薄后，整帧 PNG 解码 17.7ms 是底）· prep 9ms/张 ·
        infer 37.6ms(ort-CUDA) / 30.4ms(原生 TRT bf16) ← 硬底 · write 26ms/张（2 线程）
    墙钟 ≈ max(三段)：25.6 张/秒(ort-CUDA) / 31.8 张/秒(原生 TRT)。
    所以这一段**已经贴着推理跑了**，再优化 CPU 侧不会让吞吐变高，只会让它更抗噪。
    老实现是一条串行链（读帧 → warp → 攒 batch → 前向 → 写 3 张 PNG），
    一张脸 = 22(读) + 8(float 转换) + 36(前向) + 21(三张 PNG 编码) ≈ 100ms，实测 9.6 张/秒，
    而且 batch 从 1 加到 16 吞吐几乎不动 —— 说明瓶颈根本不在推理。
    现在拆成三段，用队列连起来（谁慢谁是瓶颈，其它两段被完全藏住）：
        读线程 ×N    PNG 解码 + 整帧 → float32 + 按 68 点 warp（1080p ≈ 30ms/帧，
                      4 线程实测摊到 7.3ms/帧）
        推理线程     攒够 batch 一次前向（416 dfm ≈ 36ms/张，GPU 99% 占用，是硬底）
        写线程 ×M    PNG 编码 + 原子落盘（416×416 ×3 ≈ 21ms/张）
    墙钟 ≈ max(三段)，而不是三段之和。

    已经踩过的两个坑（别再踩）：
      · 读线程以前先把 1080p 整帧 astype(float32)/255 再 warp（实测多花 25ms/帧），
        而 4.0 是「uint8 整帧 warp → 再 /255」，两边只差舍入 —— 现在和 4.0 同序；
      · 进度里的「帧/秒」以前拿续跑游标 k 算，而 k 为了不留空洞会滞后 readers 倍：
        25.4 张/秒被显示成 6.3 帧/秒，用户报的「5.8~6.2 帧/秒」就是它，不是真变慢。

    其它三处通用优化（都不是针对某台机器）：
      · batch 按可用显存实测自动选（_pred_pick_batch），OOM / 明显变慢自动退档 8→4→2→1；
      · 输入缓冲按 batch 预分配，每批只做一次 memcpy，不再是 np.stack(...).astype(float32)
        的「分配 + 转换 + 再拷一遍」；
      · 人脸库归组（embedding）仍然在主线程做 —— 它要用整帧，且非默认路径。

    多模型：有人脸库时按 embedding 归组，每组用各自的 dfm；不匹配的脸**不产出 pred**
    （⑤ 保持原像素）。游标 = meta.stages.pred.done_frames（帧数），取消可续。
    """
    import queue
    import sys as _sys
    import cv2
    import numpy as np
    from .compose import ensure_engine_path
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    from .._align import read_dfljpg_meta

    # 流水线实况：换脸 = 1 个 dfm session（一条 lane；写盘线程不产 bit，不算 lane）
    _STAGE_METER.set_lanes({"dfm": ["dfm#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("pred", {})
    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再换脸", 409)
    frames_dir = _stage_frames_dir(cache)
    # 帧范围：只对 [start, end] 里的帧做推理（groups 的下标/游标都按过滤后的列表数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    groups_all = _stage_face_rows(aligned)
    groups = [(k, ps) for k, ps in groups_all if _stage_in_range(rng, k)]
    total = len(groups)
    faces_in_range = sum(len(ps) for _k, ps in groups)
    pred_dir = _stage_pred_dir(cache)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dfm = str(cfg.get("dfm") or "")
    if not dfm or not Path(dfm).is_file():
        raise _BadInput("没有 .dfm 模型（换脸阶段要它）：%s" % (dfm or "空"), 409)

    ent0 = _dfm_session(dfm)                       # 先加载：batch 要按它的显存开销算
    batch, bmeta = _pred_pick_batch(cfg, ent0)
    size0 = int(ent0[2])
    # 并行度：读线程按 CPU 核数给（PNG 解码在 OpenCV 里是放 GIL 的），写线程固定 2 个
    readers = max(1, int(cfg.get("pred_readers")
                         or min(4, max(1, (os.cpu_count() or 4) // 3))))
    writers = max(1, int(cfg.get("pred_writers") or 2))
    use_lib = bool(cfg.get("use_library") and cfg.get("library"))

    # 游标 **相对范围**：0..total（total = 范围内有脸的帧数）
    rchg = bool(cfg.get("range_changed"))
    cursor = 0 if (cfg["force"] or rchg) else max(0, min(int(st.get("done_frames") or 0), total))
    if (not cfg["force"]) and not rchg and st.get("done") and cursor >= total and total:
        _stage_log("换脸：复用已有 %d 帧的 pred/（第 %d..%d 帧，不重跑）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}
    if cursor:
        _stage_log("换脸：从范围内第 %d 帧续跑（前 %d 帧的 pred/ 已经在）" % (cursor + 1, cursor))
    _stage_log("换脸：dfm=%s · batch=%d(%s) · 帧范围 %d..%d（%d 帧 / %d 张脸）· "
               "读 %d 线程 / 写 %d 线程%s"
               % (Path(dfm).name, batch, bmeta.get("source"),
                  int(rng["start"]), int(rng["end"]), total, faces_in_range,
                  readers, writers, " · 人脸库归组" if use_lib else ""))
    if bmeta.get("source") == "auto":
        _stage_log("换脸：batch 自动选择 %d（可用显存 %s / 共 %s MB · 单张约 %s MB · "
                   "留 %d%% 余量 · 最快 %s ms/张）"
                   % (batch, bmeta.get("free_mb"), bmeta.get("total_mb"),
                      bmeta.get("per_item_mb"), int(_PRED_VRAM_SAFETY * 100),
                      bmeta.get("best_per_face_ms")))

    t0 = time.time()
    # force 重跑时从 0 数起：以前会把上一轮的 count 累加（90 张脸跑完显示 180），
    # 于是 faces_per_sec 一直偏大 —— 吞吐对比全是错的数
    done_faces = 0 if (cfg.get("force") or rchg) else int(st.get("count") or 0)
    if cfg.get("force") or rchg:
        st["count"] = 0
    unmatched = int(st.get("unmatched") or 0)
    failed = int(st.get("failed") or 0)
    sizes = {}
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "frames": 0}
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": done_faces, "unmatched": unmatched, "failed": failed,
            "faces_total": int(faces_in_range), "dfm": str(dfm), "model": Path(dfm).name,
            "batch": int(batch), "batch_source": str(bmeta.get("source") or ""),
            "batch_meta": dict(bmeta), "sizes": dict(sizes), "pred_dir": str(pred_dir),
            "readers": int(readers), "writers": int(writers),
            "resumed_from": int(cursor),
            "msg": ("换脸完成：%d 张脸 / %d 帧（第 %d..%d 帧 · 未匹配 %d）"
                    % (done_faces, total, int(rng["start"]), int(rng["end"]), unmatched))
                   if done else ("换脸 %d/%d 帧（第 %d..%d 帧）"
                                 % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["pred"] = node
        _stage_meta_save(cache, meta)

    # ---- 队列与共享状态 ----
    q_frames = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, batch * 2))
    stop = threading.Event()
    boss = {"exc": None}
    pre = np.empty((batch, size0, size0, 3), np.float32)      # 预分配输入缓冲（NHWC）
    pend, completed, plock2 = {}, set(), threading.Lock()
    commit = {"n": cursor}

    def _commit_upto():
        with plock2:
            n = commit["n"]
            while n in completed:
                completed.discard(n)
                n += 1
            commit["n"] = n
            return n

    def _pend_add(gi: int):
        with plock2:
            pend[gi] = int(pend.get(gi) or 0) + 1

    def _pend_done(gi: int):
        nonlocal failed
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    # ---- 写线程：PNG 编码 + 原子落盘 ----
    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, face, prd, dst = item
            _stage_tick("dfm")       # 交接点：dfm 换脸 → pred/*.png
            tw = time.perf_counter()
            try:
                _stage_write_atomic(
                    pred_dir / (key + ".png"),
                    _stage_imencode_png(np.clip(face * 255.0, 0, 255).astype(np.uint8), 1))
                if prd is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_prdm.png"),
                        _stage_imencode_png(np.clip(prd * 255.0, 0, 255).astype(np.uint8), 1))
                if dst is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_dstdm.png"),
                        _stage_imencode_png(np.clip(dst * 255.0, 0, 255).astype(np.uint8), 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("换脸写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _put_frame(item):
        """带超时的 put：主线程没了（取消/异常）也别把读线程吊死在队列上。"""
        while not stop.is_set():
            try:
                q_frames.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ---- 读线程：解码 + warp（不碰推理，也不碰写盘）----
    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                key, paths = groups[gi]
                fp = _stage_frame_path(frames_dir, key)
                if fp is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                tr = time.perf_counter()
                frame = _stage_read_frame(fp)
                _bump("read_ms", time.perf_counter() - tr)
                if frame is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                crops = None
                if not use_lib:
                    tp = time.perf_counter()
                    crops = []
                    for ap in paths:
                        m = read_dfljpg_meta(ap) or {}
                        lm = m.get("source_landmarks") or m.get("landmarks")
                        if not lm or len(lm) != 68:
                            crops.append((ap.stem, None, None))
                            continue
                        try:
                            sz = size0
                            fmat = get_transform_mat(np.asarray(lm, np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            # 顺序和 4.0 export_pipeline:720-721 完全一致：**uint8 整帧
                            # warp → 再 /255**。以前是先把 1080p 整帧转 float32 再 warp，
                            # 实测这多花的 25ms/帧（astype 11.8 + 浮点 warp 13.4 vs
                            # uint8 warp 0.18）只是把同样的双线性插值算在 float 上，
                            # 结果只差舍入 —— 却把 4 个读线程全拖成瓶颈。
                            crop = cv2.warpAffine(frame, fmat, (sz, sz), flags=cv2.INTER_CUBIC)
                            crops.append((ap.stem,
                                          np.clip(crop.astype(np.float32) / 255.0, 0, 1), None))
                        except Exception as e:              # noqa: BLE001
                            crops.append((ap.stem, None, str(e)[:100]))
                    _bump("prep_ms", time.perf_counter() - tp)
                # ok=True 才是「这一帧能处理」；frame 只在人脸库归组时才需要带回主线程
                _put_frame((gi, key, True, frame if use_lib else None, None, crops))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_frames.put((None, None, False, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    # ---- 一批：拷进预分配缓冲 → 一次前向 → 交给写线程 ----
    jobs = {}          # dfm -> ([gi], [keys], [crops])
    # best_ms 用阶梯探测的最好成绩起手：否则第一批（可能已经偏大）会被当成基准，
    # 后面的退档守卫就永远不会触发
    stat = {"best_ms": bmeta.get("best_per_face_ms") or None, "shrunk": 0}

    def _flush(use_dfm: str, force: bool = False):
        nonlocal batch, done_faces, failed
        bucket = jobs.get(use_dfm)
        if not bucket:
            return
        gis, keys, crops = bucket
        while keys and (len(keys) >= batch or (force and keys)):
            n = min(len(keys), batch)
            ent = _dfm_session(use_dfm)
            size = int(ent[2])
            sizes[Path(use_dfm).name] = size
            if size0 == size and pre.shape[0] >= n:
                arr = pre[:n]
            else:
                arr = np.empty((n, size, size, 3), np.float32)
            for i in range(n):
                arr[i] = crops[i]
            ti = time.perf_counter()
            try:
                outs = _dfm_run_batch(ent, arr)
            except Exception as e:                          # noqa: BLE001
                if not (_is_oom(e) and n > 1):
                    raise
                step = max(1, n // 2)
                _stage_log("换脸：batch=%d 显存不够（%s）→ 退到 %d 再试"
                           % (n, str(e)[:80], step))
                batch = max(1, min(batch, step))
                stat["shrunk"] += 1
                outs = []
                for i in range(0, n, step):
                    outs.extend(_dfm_run_batch(ent, arr[i:i + step]))
            dt = time.perf_counter() - ti
            per_face = dt * 1000.0 / max(1, n)
            _bump("infer_ms", dt, n)
            # 变慢守卫：比历史最好成绩慢 60% 说明 batch 太大（显存抖动 / 换出），主动退档
            if stat["best_ms"] is None or per_face < stat["best_ms"]:
                stat["best_ms"] = per_face
            elif n > 1 and per_face > stat["best_ms"] * 1.6 and stat["shrunk"] < 4:
                half = max(1, n // 2)
                _stage_log("换脸：batch=%d 时每张 %.1f ms（最好 %.1f ms）→ 退到 %d"
                           % (n, per_face, stat["best_ms"], min(batch, half)))
                batch = max(1, min(batch, half))
                stat["shrunk"] += 1
            for i in range(n):
                q_write.put((gis[i], keys[i], outs[i][0], outs[i][1], outs[i][2]))
                done_faces += 1
            del gis[:n], keys[:n], crops[:n]

    def _submit(gi: int, stem: str, use_dfm: str, crop):
        _pend_add(gi)
        bucket = jobs.setdefault(use_dfm, ([], [], []))
        bucket[0].append(gi)
        bucket[1].append(stem)
        bucket[2].append(crop)
        _flush(use_dfm)

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="pred-w%d" % i)
        t.start()
        threads.append(t)
    hi_all = total
    for r in range(readers):
        lo = cursor + (hi_all - cursor) * r // readers
        hi = cursor + (hi_all - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="pred-r%d" % r)
        t.start()
        threads.append(t)

    k = cursor
    cancelled = {"flag": False}
    got = 0            # 已经处理完的帧数（显示用）。k 是**续跑游标**，它会为了
                       # 「不留空洞」而滞后（4 个读线程各跑一段，游标只认连续前缀），
                       # 以前拿 k 算帧/秒 → 正好少报 readers 倍（实测 6.3 帧/秒 vs
                       # 真实 25.4 张/秒，差 4 倍，就是被当成「慢」的那个数）。
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_frames.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, ok, frame, fc, crops = item
            if not ok:
                _stage_log("换脸跳过 %s：frames/ 里没有这一帧（或读不出来）" % key)
                with plock2:
                    completed.add(gi2)
            elif use_lib:
                # 人脸库归组：要整帧 + embedding，只能在主线程做
                metas = []
                for ap in groups[gi2][1]:
                    m = read_dfljpg_meta(ap) or {}
                    metas.append({"ap": ap,
                                  "lm": m.get("source_landmarks") or m.get("landmarks"),
                                  "box": m.get("source_rect") or None})
                plans = _stage_pred_assign(cfg, frame, metas)
                for meta_f, use_dfm in zip(metas, plans):
                    lm = meta_f["lm"]
                    if not lm or len(lm) != 68:
                        failed += 1
                        _stage_log("换脸跳过 %s：没有 68 点" % meta_f["ap"].name)
                        continue
                    if not use_dfm:
                        unmatched += 1
                        continue
                    size = int(_dfm_session(use_dfm)[2])
                    fmat = get_transform_mat(np.asarray(lm, np.float32), size,
                                             FaceType.WHOLE_FACE, scale=1.0)
                    crop = cv2.warpAffine(frame, fmat, (size, size), flags=cv2.INTER_CUBIC)
                    _submit(gi2, meta_f["ap"].stem, use_dfm,
                            np.clip(crop.astype(np.float32) / 255.0, 0, 1))
            else:
                for stem, crop, err in (crops or []):
                    if crop is None:
                        failed += 1
                        _stage_log("换脸跳过 %s%s" % (stem, ("：" + err) if err else "：没有 68 点"))
                        continue
                    _submit(gi2, stem, dfm, crop)
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("pred", 100.0 * got / total,
                            "换脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.1f 帧/秒 · "
                            "%.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]), done_faces,
                               got / el, done_faces / el, batch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
            if (gi - cursor) % 20 == 0:
                _save(k)
        else:
            k = total
    except _StageCancel:
        # 取消也要把游标落盘 —— 但**必须等排空之后**再落：以前在 except 里立刻
        # _save(commit)，那一刻队里还有没写完的脸，游标会停在 0（实测 60 张脸
        # 取消时 done_frames=0，续跑等于从头再来）。现在置标志位，排空在 finally 里做。
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            for use_dfm in list(jobs):
                _flush(use_dfm, force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
            # 等写盘排空：游标必须对得上磁盘上真的有的文件（否则续跑会跳过没写的帧）
            t_wait = time.time()
            while time.time() - t_wait < 600:
                with plock2:
                    left = sum(int(v) for v in pend.values())
                if left <= 0:
                    break
                time.sleep(0.05)
        k = _commit_upto()
        for t in threads:
            t.join(timeout=5.0)
    if cancelled["flag"]:
        _save(k)
        _stage_log("换脸取消：停在 %d/%d 帧（游标已存；已跑完 %d 张脸）"
                   % (k, total, done_faces))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    sec = round(time.time() - t0, 2)
    prof_out = {"read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
                "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
                "readers": int(readers), "writers": int(writers),
                "per_face_ms": {
                    "read": round(prof["read_ms"] / max(1, total - cursor), 2),
                    "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
                    "infer": round(prof["infer_ms"] / max(1, prof["faces"]), 2),
                    "write": round(prof["write_ms"] / max(1, prof["faces"]), 2),
                }}
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "faces_per_sec": round(done_faces / sec, 2) if sec else None,
        "frame_fps": round((total - cursor) / sec, 2) if sec else None,
        "profile": prof_out, "batch_final": int(batch)})
    _stage_log("换脸完成：第 %d..%d 帧（%d 帧）· %d 张脸 · 未匹配 %d · 失败 %d · %.2f s · "
               "%.1f 张/秒（batch %d · 读 %d/写 %d 线程 · 每张 读 %.1f + 预处理 %.1f + "
               "推理 %.1f + 写 %.1f ms）"
               % (int(rng["start"]), int(rng["end"]), total, done_faces, unmatched,
                  failed, sec, done_faces / sec if sec > 0 else 0.0, batch, readers,
                  writers, prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_face_ms"]["write"]))
    return {"count": done_faces, "frames": total, "unmatched": unmatched, "failed": failed,
            "seconds": sec, "faces_per_sec": round(done_faces / sec, 2) if sec else None,
            "batch": int(batch), "lane": "pipeline",
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}
