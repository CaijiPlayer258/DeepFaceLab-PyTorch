# -*- coding: utf-8 -*-
"""第 2 段：切脸（帧 → 对齐脸 DFLJPG）。从 pipeline_stages.py 搬出。"""
from __future__ import annotations

from pathlib import Path
import collections
import numpy as np
import queue
import threading
import time

from .pipeline_stages import (
    _BadInput, _DEFAULT_DETECTOR, _FACES_END, _STAGE_JPEG_QUALITY,
    _STAGE_METER, _StageCancel, _StageTicks, _det_run_lock,
    _detector, _landmarks_model, _stage_aligned_dir, _stage_check_cancel,
    _stage_clear_dir, _stage_face_name, _stage_faces_batch, _stage_faces_batchable,
    _stage_faces_detect, _stage_faces_queue, _stage_faces_read_threads, _stage_files,
    _stage_frames_dir, _stage_in_range, _stage_log, _stage_meta_save,
    _stage_progress, _stage_range, _stage_read_image, _stage_tick,
)

def _stage_do_faces(cfg: dict) -> dict:
    """阶段 2 · 切脸：frames/ -> aligned/000001_00.jpg（DFLJPG：original_name + 68 点 + face_mat）。"""
    import cv2
    import numpy as np
    from .._align import (COARSE_SIZE, align_face_68, coarse_align_from_5,
                         landmarks68_in_source, pose_from_kps, write_dfljpg)

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    # 流水线实况：切脸 = 1 个检测 session（读线程不算 lane，它们不产 bit）
    _STAGE_METER.set_lanes({"det": ["det#0"]})
    st = meta["stages"].setdefault("faces", {})
    frames = _stage_files(_stage_frames_dir(cache))
    if not frames:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    aligned_dir = _stage_aligned_dir(cache)
    # 帧范围：只处理 [start, end] 里的帧（帧索引仍是全局的 —— aligned/ 的命名不许挪位）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rows = [i for i, p in enumerate(frames) if _stage_in_range(rng, p.stem)]
    total = len(rows)                     # 范围内的帧数（游标从这里数到 0）
    have = len(_stage_files(aligned_dir, (".jpg", ".jpeg")))

    if ((not cfg["force"]) and not cfg.get("range_changed") and st.get("done") and have
            and int(st.get("frames") or 0) == len(frames)):
        _stage_log("切脸：复用已有 %d 张对齐脸（第 %d..%d 帧，不重跑）"
                   % (have, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": have, "reason": "already-done"}

    if cfg["force"]:
        n = _stage_clear_dir(aligned_dir, (".jpg", ".jpeg"))
        _stage_log("切脸：force=true，清掉旧的 %d 张对齐脸" % n)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # 以前兜底写死 "retinaface_10g"：cfg 里没有 det_align 时静默换成一个**重档**检测器，
    # 日志里看不出这一跳。现在兜底 = _DEFAULT_DETECTOR，并且**明写一行**。
    det_name = str(cfg.get("det_align") or "").strip()
    if not det_name:
        det_name = _DEFAULT_DETECTOR
        _stage_log("切脸：cfg 里没有 det_align，用了默认检测器 %s" % det_name)
    thresh = float(cfg.get("align_thresh") or 0.5)
    lm = _landmarks_model()          # 和预览/合成同款：纯 CPU onnxruntime 的 2D106
    if lm is None:
        raise _BadInput("2D106 关键点模型不可用（weights/insightface_2d106.onnx）—— "
                        "没有 68 点就不写对齐脸", 500)

    # 游标 **相对范围**：0..total（total = end-start+1），换范围时旧游标（相对旧范围）作废
    cursor = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    faces_total = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("count") or 0)
    with_faces = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("frames_with_faces") or 0)
    skipped = int(st.get("skipped") or 0)
    if cursor:
        _stage_log("切脸：从第 %d 帧续跑（前 %d 帧的对齐脸已经在）"
                   % (int(rng["start"]) + cursor, cursor))
    _stage_log("切脸：检测器 %s（%s）· thresh=%.2f · 68 点仿射 · 帧范围 %d..%d（%d 帧）"
               % (det_name, str(cfg.get("det_align_src") or "cfg.det_align"), thresh,
                  int(rng["start"]), int(rng["end"]), total))
    # 对齐脸是这条链上唯一的 JPEG 产物：质量回显出来（验收/排障要能一眼看到，
    # 别再靠"默认值"猜）。默认 100 + 4:4:4；PNG 帧完全不受这个数影响。
    _stage_log("切脸：对齐脸 JPEG quality=%d · 4:4:4（界面无旋钮，固定质量）"
               % int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY))
    # 检测器先建好（加载要几百毫秒）：流水线里只有一个线程用它，不再每帧走 _detector()
    det = _detector(det_name)

    t0 = time.time()

    def _save_progress(up_to: int, done: bool = False):
        meta["stages"]["faces"] = {
            "done": bool(done), "done_frames": int(up_to), "frames": len(frames),
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": faces_total, "frames_with_faces": with_faces, "skipped": skipped,
            "detector": det_name, "align": "68", "thumb": "2x box",
            "msg": ("切脸完成：%d 张脸 / %d 帧（第 %d..%d 帧）"
                    % (faces_total, total, int(rng["start"]), int(rng["end"]))) if done
                   else ("切脸 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        _stage_meta_save(cache, meta)

    # ---- 流水线：读帧线程池 → 检测线程 → 本线程做 2D106 / 裁剪 / 编码写盘 ----
    # 三段用的资源互不相干（磁盘+解码 / GPU / CPU），串行做等于每段都在等另外两段。
    from concurrent.futures import ThreadPoolExecutor

    nread = _stage_faces_read_threads()
    depth = _stage_faces_queue(nread)
    batch_n = _stage_faces_batch(det)
    ticks = _StageTicks()
    count0 = faces_total
    _stage_log("切脸流水线：读帧 %d 线程 · 队列 %d · 检测批次 %d%s"
               % (nread, depth, batch_n,
                  "" if batch_n > 1 else "（%s 的 onnx 输入 batch 维固定=1 → 逐帧）" % det_name))

    read_q = queue.Queue(maxsize=depth)     # (pos, frame, err)；满了读帧线程就等，内存可控
    det_q = queue.Queue()                   # (pos, frame, boxes) / ("err", None, 异常)
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=nread, thread_name_prefix="faces-read")

    def _read_job(path):
        t = time.perf_counter()
        try:
            return _stage_read_image(path), None
        except Exception as e:                                 # noqa: BLE001
            return None, e
        finally:
            ticks.add("read", time.perf_counter() - t)

    def _q_put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _feeder():
        """按帧序读图：最多 depth 张在飞，读出来的帧立即进 read_q。

        注意「读完了」和「在飞的都取完了」是两件事：帧迭代器耗尽后 pending 里还压着
        最多 depth 张，必须继续取完再收尾（曾经在这里直接 break，末尾 12 帧被静默丢掉，
        阶段照样报 40/40 完成 —— 所以主循环末尾还有一道 consumed 对账）。
        """
        pending = collections.deque()
        it = iter(range(cursor, total))
        exhausted = False
        try:
            while True:
                while (not exhausted) and (not stop.is_set()) and len(pending) < depth:
                    try:
                        pos = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        pending.append((pos, pool.submit(_read_job, frames[rows[pos]])))
                    except RuntimeError:            # 取消时主线程已经 shutdown 了线程池
                        exhausted = True
                        break
                if not pending:
                    break
                pos, fut = pending.popleft()
                try:
                    img, err = fut.result()
                except Exception as e:                         # noqa: BLE001
                    img, err = None, e
                if not _q_put(read_q, (pos, img, err)):
                    return
        except Exception:                       # 收尾阶段的竞态不该把线程栈打到控制台
            pass
        finally:
            _q_put(read_q, _FACES_END)

    def _detect_worker():
        """从 read_q 取帧、按 batch_n 攒一批送检测器，结果按帧序进 det_q。"""
        ended = False
        try:
            while True:
                item = read_q.get()
                if item is _FACES_END:
                    ended = True
                    item = None
                group = []
                if item is not None:
                    if item[2] is not None:
                        det_q.put(item)
                        continue
                    group.append(item)
                while (not ended) and len(group) < batch_n:
                    try:
                        nxt = read_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _FACES_END:
                        ended = True
                        break
                    if nxt[2] is not None:
                        det_q.put(nxt)
                        continue
                    group.append(nxt)
                if group:
                    _stage_check_cancel()
                    imgs = [g[1] for g in group]
                    t = time.perf_counter()
                    try:
                        with _det_run_lock:                     # thresh 是共享检测器上的字段
                            det.thresh = float(thresh)
                            got = _stage_faces_detect(det, imgs, thresh)
                    except _StageCancel:
                        raise
                    except Exception as e:                      # noqa: BLE001
                        raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)
                    finally:
                        ticks.add("detect", time.perf_counter() - t)
                    for g, boxes in zip(group, got):
                        det_q.put((g[0], g[1], boxes))
                if ended:
                    return
        except BaseException as e:                              # noqa: BLE001
            det_q.put(("err", None, e))
        finally:
            det_q.put(_FACES_END)

    feeder_th = threading.Thread(target=_feeder, name="faces-feed", daemon=True)
    det_th = threading.Thread(target=_detect_worker, name="faces-detect", daemon=True)
    feeder_th.start()
    det_th.start()

    consumed = 0
    try:
        while True:
            item = det_q.get()
            if item is _FACES_END:
                break
            pos, frame, extra = item
            if pos == "err":
                raise extra
            if frame is None:
                raise _BadInput("这一帧读不出来：%s" % frames[rows[pos]], 500)
            if isinstance(extra, BaseException):
                raise extra
            boxes = sorted(list(extra or []),
                           key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
            n = 0
            for k, b in enumerate(boxes):
                _stage_check_cancel()
                box = b.get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(v) for v in box]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                side = int(round(max(bw, bh) * 2.0))
                side -= side % 2
                kps = b.get("kps") or []
                if len(kps) < 5 or side < 32:
                    skipped += 1
                    continue
                try:
                    cmat = coarse_align_from_5(kps)
                    coarse = cv2.warpAffine(frame, cmat, (COARSE_SIZE, COARSE_SIZE),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                    t_lm = time.perf_counter()
                    lm68a = lm.landmarks68(coarse)
                    ticks.add("lm68", time.perf_counter() - t_lm)
                    if lm68a is None:
                        raise ValueError("2D106 没有输出")
                    pts = landmarks68_in_source(lm68a, cmat)
                    t_cr = time.perf_counter()
                    crop, meta_f = align_face_68(frame, np.asarray(pts, np.float64),
                                                 box=[int(round(x1)), int(round(y1)),
                                                      int(round(x2)), int(round(y2))],
                                                 size=side)
                    ticks.add("crop", time.perf_counter() - t_cr)
                except Exception as e:                          # noqa: BLE001
                    skipped += 1
                    _stage_log("切脸跳过 %s 第 %d 张：%s" % (frames[rows[pos]].name, k,
                                                          str(e)[:120]))
                    continue
                if crop is None or not meta_f:
                    skipped += 1
                    _why = (meta_f or {}).get("reject_reason") if isinstance(meta_f, dict) else ""
                    if _why:
                        _stage_log("切脸拒收 %s 第 %d 张：%s"
                                   % (frames[rows[pos]].name, k, _why))
                    # ★ 拒收原因要落盘，不能只滚一行日志。以前 append_bad_frame
                    #   全仓只有 backend_extract/align_stream 在调，4.1 的这条链从不
                    #   写 _bad_frames.txt —— 于是 DFL_ALIGN_GATE 的 warn 模式在这里
                    #   等于空转：拒了哪几张、为什么拒，事后没有任何东西可复盘
                    #   （align.py:612 的 _say 只打到 stdout，滚过去就没了）。
                    try:
                        from .._vision.align import append_bad_frame as _abf
                        _abf(aligned_dir, _stage_face_name(rows[pos], k),
                             {"bad": True, "span": 0.0, "iod": 0.0,
                              "reason": str(_why or "对齐没有返回可用的脸")[:120]})
                    except Exception:                      # noqa: BLE001
                        pass
                    continue
                # 配对键 / 68 点 / face_mat 三样都写进 DFLJPG 的 APP15
                meta_f.update({
                    "original_name": frames[rows[pos]].name,
                    "source_filename": frames[rows[pos]].stem,
                    "face_mat": meta_f.get("image_to_face_mat"),
                    "landmarks_68": meta_f.get("landmarks"),
                    "source_landmarks_68": meta_f.get("source_landmarks"),
                    "frame_index": int(rows[pos]),
                    "frame_name": frames[rows[pos]].name, "face_index": int(k),
                    "detector": det_name, "side": int(side),
                })
                try:
                    yaw, pitch, roll = pose_from_kps(kps, box)
                    if yaw is not None:
                        meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = yaw, pitch, roll
                except Exception:                               # noqa: BLE001
                    pass
                _stage_tick("det")       # 交接点：切脸 → aligned/*.jpg（下一段从这里读）
                t_en = time.perf_counter()
                write_dfljpg(aligned_dir / _stage_face_name(rows[pos], k), crop, meta_f,
                             quality=int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY),
                             subsampling=0)
                ticks.add("encode", time.perf_counter() - t_en)
                n += 1
            faces_total += n
            if n:
                with_faces += 1
            else:
                # 这一帧一张脸都没送出（检出 0 张 / 全被拒收）→ 进度页旁路线上的那一帧
                _STAGE_METER.note_no_face()
            el = max(0.001, time.time() - t0)
            _stage_progress("faces", 100.0 * (pos + 1) / total,
                            "切脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.2f 帧/秒"
                            % (pos + 1, total, int(rng["start"]), int(rng["end"]),
                               faces_total, (pos + 1 - cursor) / el),
                            frames_done=pos + 1, frames_total=total,
                            frame_fps=round((pos + 1 - cursor) / el, 2))
            consumed += 1
            if (pos + 1) % 20 == 0:
                _save_progress(pos + 1)
    finally:
        stop.set()
        pool.shutdown(wait=False)
        det_th.join(timeout=1.5)
        feeder_th.join(timeout=1.5)

    sec = round(time.time() - t0, 2)
    done_n = max(0, total - cursor)
    # 对账：流水线少喂了帧就等于少切了脸，绝不能把「跑到 30/40」报成「40/40 完成」
    if consumed != done_n:
        raise _BadInput("切脸流水线提前结束：只有 %d/%d 帧走完了检测（产物不完整，"
                        "重跑 force=true）" % (consumed, done_n), 500)
    fps = float(done_n) / max(1e-6, sec)
    # 三段是**并行**的，所以「每帧在各段上花的时间」加起来会大于每帧墙钟：
    #   read     = 读图总耗时 / 读帧线程数（摊到单帧、按并行度折算）
    #   others   = 每帧真的花在那一段上的时间
    #   stage_util = 这一段占用了整段墙钟的百分之多少（>100% = 多线程并行）
    post = ticks.total("lm68") + ticks.total("crop") + ticks.total("encode")
    prof = {"frames": int(done_n), "faces": int(faces_total - count0), "seconds": sec,
            "fps": round(fps, 2),
            "per_frame_ms": {"read": round(1000.0 * ticks.total("read") / max(1, nread)
                                           / max(1, done_n), 2),
                             "detect": ticks.ms("detect", done_n),
                             "lm68": ticks.ms("lm68", done_n),
                             "crop": ticks.ms("crop", done_n),
                             "encode": ticks.ms("encode", done_n)},
            "stage_util": {"read": round(100.0 * ticks.total("read") / max(1e-6, sec), 1),
                           "detect": round(100.0 * ticks.total("detect") / max(1e-6, sec), 1),
                           "post": round(100.0 * post / max(1e-6, sec), 1)},
            "lm68_ms_per_face": ticks.ms("lm68", max(1, faces_total - count0)),
            "pipeline": {"read_threads": int(nread), "queue": int(depth),
                         "detect_batch": int(batch_n),
                         "batchable": bool(_stage_faces_batchable(det)),
                         "overlap": "read || detect || post"}}
    # 谁在拖后腿：按「占用墙钟的比例」比，读帧那一路要除以线程数
    # （read 的 stage_util 是 6 个线程之和，直接比会把多线程误判成瓶颈）
    eff = {"read": prof["stage_util"]["read"] / max(1, int(nread)),
           "detect": prof["stage_util"]["detect"],
           "post": prof["stage_util"]["post"]}
    prof["stage_share"] = {k: round(v, 1) for k, v in eff.items()}
    prof["bottleneck"] = max(eff.items(), key=lambda kv: kv[1])[0] if eff else ""
    prof["per_frame_total_ms"] = round(1000.0 * sec / max(1, done_n), 2)
    prof["per_frame_ms"]["other"] = round(max(
        0.0, prof["per_frame_total_ms"] - sum(prof["per_frame_ms"].values())), 2)
    _save_progress(total, done=True)
    meta["stages"]["faces"]["seconds"] = sec
    meta["stages"]["faces"]["fps"] = round(fps, 2)
    meta["stages"]["faces"]["profile"] = prof
    _stage_meta_save(cache, meta)
    _stage_log("切脸完成：第 %d..%d 帧（%d 帧）· %d 张脸（%d 帧有脸）· 跳过 %d · %.2f s · "
               "%.2f 帧/秒（读 %s / 检测 %s / 2D106 %s / 裁剪 %s / 写盘 %s ms，"
               "读帧 %d 线程 · 批次 %d）"
               % (int(rng["start"]), int(rng["end"]), total, faces_total,
                  with_faces, skipped, sec, fps,
                  prof["per_frame_ms"]["read"], prof["per_frame_ms"]["detect"],
                  prof["per_frame_ms"]["lm68"], prof["per_frame_ms"]["crop"],
                  prof["per_frame_ms"]["encode"], nread, batch_n))
    return {"count": faces_total, "frames": total, "skipped": skipped, "seconds": sec,
            "fps": round(fps, 2), "profile": prof,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}
