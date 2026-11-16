# -*- coding: utf-8 -*-
"""第 4 段：遮罩（pred/ → mask/）。从 pipeline_stages.py 搬出。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import os
import queue
import threading
import time

from .pipeline_stages import (
    _BadInput, _INFER_CALLS, _MASK_DST_INPUT, _STAGE_METER,
    _StageCancel, _mask_signature, _mask_src, _stage_aligned_dir,
    _stage_check_cancel, _stage_files, _stage_filter_range, _stage_imencode_png,
    _stage_log, _stage_mask_dir, _stage_meta_save, _stage_pred_dir,
    _stage_progress, _stage_range, _stage_read_dst_face, _stage_read_image,
    _stage_tick, _stage_write_atomic, _stage_xseg,
)

def _stage_do_mask(cfg: dict) -> dict:
    """阶段 4 · 遮罩推理：aligned/ + pred/ -> mask/（每个对齐脸两张灰图）。

    语义照 4.0 export_pipeline.py:758-799（_phase_mask_worker）：
      · XSeg / XSegLite 来源：mask/<key>_xseg_prd.png = XSeg(pred)，
        mask/<key>_xseg_dst.png = XSeg(dst 对齐脸)
      · DFM 来源：遮罩就是 dfm 自带的那两条分支（③ 已经落盘 prdm/dstdm），这一步
        只把它们按 mask/ 的命名**落盘留档**（便于检查），不跑任何推理。
    两种来源都会把 dfm 的两张遮罩一起放进 mask/（mask_mode=9 全乘要用）。
    游标 = meta.stages.mask.done_frames（对齐脸张数）。

    **dst 输入 = aligned/<frame>_<fidx>.jpg：不读 frames/，只补一张几何补偿矩阵**
      以前为了拿「dst 对齐脸」先解码 1080p 原帧（≈24 ms/张脸，同一帧多张脸还各解码一次），
      再用 facelib 的 WHOLE_FACE 变换压到 256 —— 解码原帧这一步是纯浪费：同一张脸就在
      aligned/*.jpg 里（切脸阶段 68 点仿射 _align.align_mat_from_68 的直接产物）。
      现在读 aligned，再用一张补偿矩阵（image_to_face_mat⁻¹ × facelib WHOLE_FACE，见
      _stage_dst_face_mat）映射到合成画布。**为什么要补偿**：这条链的合成画布是 facelib 的
      （③ pred、⑤ compose_frame 都用它），而 aligned 芯片在 2026-12 之前是另一张画布
      （68 点拟合，中位 0.36% 缩放 + 2.5° 旋转，尾部 13%）—— 直接 resize 会让 dst 遮罩
      整体错开，mask_mode 8/7 贴回原帧就是边缘一条挖进脸里的缝。重切之后两边已经是同一张
      矩阵，这张补偿对它们约等于恒等，留着是为了兼容没重切过的老项目。
      dst 输入整条读下来实测 5.6 ms/张（解码 3.4 + 元数据 0.6 + 矩阵 0.5 + 映射 1.1），
      省掉的是 1080p 解码（≈24 ms）和每张脸重算一遍的 68 点变换。
      读不出来 / 损坏的那一张**跳过并计数**（failed + 日志点名 + meta 的 dst_bad_aligned），
      既不拿全帧兜底、也不算进成功数；一张都没成就报错。

    **吞吐**：这条链以前是纯串行（读 pred → 元数据 → 读原帧 → XSeg(prd) → 写盘 →
    XSeg(dst) → 写盘 → 复制 dfm 遮罩），一台机器上 2059 张脸 ≈ 26 张/秒，而原生 TRT
    的 XSegLite 纯推理只要 1.44 ms/图（实测），也就是 92% 的时间不在推理上。现在拆成
    读 N 线程 / 推理 1 线程 / 写 M 线程三段（和 ③ 换脸同一套结构），墙钟从「各段相加」
    变成「取最长的那段」；dst 输入换成 aligned 之后读线程不再是瓶颈（口径见
    app/tools/bench/stage_perf_profile.py 与 %TEMP% 里的 before/after 基准）。
    """
    import sys as _sys
    import numpy as np

    # 流水线实况：遮罩 = 1 个 XSeg session（DFM 来源时这块不推理，仍然是 1 条 lane）
    _STAGE_METER.set_lanes({"xseg": ["xse#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("mask", {})
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    cfg["mask_src"] = src
    sig = _mask_signature(src, cfg)
    aligned_all = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned_all:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再补遮罩", 409)
    if not (meta["stages"].get("pred") or {}).get("done"):
        raise _BadInput("换脸（pred）还没完成 —— 遮罩阶段要用换脸后的正脸，先跑换脸", 409)
    # 帧范围：只对 [start, end] 里的对齐脸推理（total/游标都按这一段的张数数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    aligned = _stage_filter_range(rng, aligned_all, lambda p: p.stem.split("_")[0])
    total = len(aligned)
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if (st.get("signature") and st.get("signature") != sig and not cfg["force"]
            and int(st.get("done_frames") or 0) > 0):
        raise _BadInput("遮罩来源变了（%s → %s）：mask/ 里的东西已经失效，"
                        "不能静默复用。传 force=true 重跑遮罩阶段"
                        % (st.get("signature"), sig), 409)
    # 老版本（dst 输入 = 原帧重新对齐）算出来的 mask/ 不能静默复用：来源没变，但几何变了。
    # 这里不报 409（挡的是「换了模型还复用旧遮罩」），直接当 force 整段重算并说清原因。
    stale_dst = (bool(st.get("done")) and bool(st.get("inference"))
                 and str(st.get("dst_input") or "") != _MASK_DST_INPUT)
    if stale_dst:
        _stage_log("遮罩：现有 mask/ 是「dst 输入 = 原帧重新对齐」那一版算的（dst_input=%s）"
                   "→ 口径已改成 aligned/，几何变了，整段重算"
                   % (st.get("dst_input") or "-",))
    redo = bool(cfg["force"] or rchg or stale_dst)
    if (not redo) and st.get("done") and st.get("signature") == sig \
            and int(st.get("done_frames") or 0) >= total:
        _stage_log("遮罩：复用已有 %d 张遮罩（第 %d..%d 帧，不重跑；要重做传 force=true）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}

    # ---- 按界面选择加载模型；后端 = TRT 有就用、没有就 onnxruntime（结论全部回显）----
    xseg = None
    backend = "none"
    engine_file = ""
    engine_prec = ""
    fallback = ""
    model_onnx = ""
    model_label = str(src.get("label") or "")
    if src["source"] == "xseg":
        xseg, res = _stage_xseg(src.get("model"))
        info = dict(getattr(xseg, "info", {}) or {})
        backend = str(info.get("backend") or "ort")
        engine_file = str(info.get("engine") or "")
        engine_prec = str(info.get("precision") or "")
        fallback = str(info.get("reason") or "")
        model_onnx = str(info.get("onnx") or res.get("path") or "")
        model_label = res.get("label") or model_label
        if backend == "trt":
            _stage_log("遮罩：模型=%s（%s）· 后端=TRT(%s) · 引擎=%s"
                       % (model_label, Path(model_onnx).name, engine_prec or "?", engine_file))
        else:
            _stage_log("遮罩：模型=%s（%s）· 后端=onnxruntime · 没有可用的 TRT 引擎 → 回落"
                       "（原因：%s）" % (model_label, Path(model_onnx).name,
                                        fallback or "未知"))
    engine = backend          # 老字段（导出/验收还在用它）—— 语义就是 trt|ort|none
    cursor = 0 if redo else max(0, min(int(st.get("done_frames") or 0), total))
    if cursor:
        _stage_log("遮罩：从范围内第 %d 张续跑（前 %d 张已经在）" % (cursor + 1, cursor))
    _stage_log("遮罩：来源=%s · 模型=%s · 引擎=%s%s · 第 %d..%d 帧（%d 张对齐脸）%s"
               % (src.get("label") or src.get("source"), model_label or "-", engine,
                  ("(%s)" % engine_file) if engine_file else "",
                  int(rng["start"]), int(rng["end"]), total,
                  ("" if not src.get("model_note") else
                   " · 注意：" + str(src.get("model_note")))))

    t0 = time.time()
    # force / 换范围 / dst 输入口径变了 都从 0 数（游标和计数都相对范围）
    wrote = 0 if redo else int(st.get("count") or 0)
    if redo:
        st["count"] = 0
    failed = int(st.get("failed") or 0)
    means = []
    t_infer = 0.0          # 纯推理累计（两张 256 的 XSeg 前向，不含读图/写盘）
    # 逐段打点（和 ③ 换脸同一套口径：读图 / 预处理 / 推理 / 写盘，各自累计 ms）。
    # dst 输入改成 aligned 之后，几何映射并在读图段里（一次 558→256 的重采样）→ 预处理段
    # prep_ms 恒为 0（保留这个打点名，是为了和上一版的 profile 直接对比）；
    # 读图 = pred PNG 解码 + aligned JPEG 解码 + 补偿矩阵 + 映射到 256。
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "images": 0, "bad_aligned": 0}
    bad_aligned = []       # 读不出来的 aligned 文件名（最多留 20 个，进 meta / 日志）
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0, images: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n
            if images:
                prof["images"] += images

    def _bad_aligned(name: str):
        """记一张读不出来的 aligned 文件（计数 + 最多留 20 个名字）。"""
        with plock:
            prof["bad_aligned"] = int(prof.get("bad_aligned") or 0) + 1
            if len(bad_aligned) < 20:
                bad_aligned.append(str(name))

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "total": total,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": wrote, "failed": failed, "signature": sig,
            "mask_source": src.get("label") or src.get("source"),
            "mask_model": str(src.get("model") or ""),
            "mask_model_from": str(src.get("model_from") or ""),
            "mask_onnx": model_onnx, "mask_label": model_label,
            "engine": engine,                 # 老字段，语义 = backend
            "backend": backend,               # trt | ort | none
            "engine_file": engine_file,       # 实际加载的引擎文件名（trt 才有）
            "engine_precision": engine_prec,  # int8 / bf16
            "fallback_reason": fallback,      # 没上 TRT 的一句话原因（有引擎时为空）
            "size": 256, "inference": (src["source"] == "xseg"),
            # dst 遮罩的像素来源：aligned（aligned/*.jpg 本身）/ dfm（dfm 自带，不推理）。
            # 老版本没有这个字段 —— _stage_do_mask 靠它认出「从原帧重建」那一版并重算。
            "dst_input": (_MASK_DST_INPUT if src["source"] == "xseg" else "dfm"),
            "dst_bad_aligned": int(prof.get("bad_aligned") or 0),
            "dir": str(mask_dir), "resumed_from": int(cursor),
            "msg": ("遮罩完成：%d 张脸（第 %d..%d 帧 · 来源 %s）"
                    % (wrote, int(rng["start"]), int(rng["end"]),
                       src.get("label") or src.get("source"))) if done
                   else ("遮罩 %d/%d 张（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["mask"] = node
        _stage_meta_save(cache, meta)

    def _copy_dfm_masks(key: str):
        """把 ③ 的 dfm 遮罩按 mask/ 的命名落盘（来源=DFM 时就是这一步的产物）。"""
        n = 0
        for src_name, dst_name in (("_prdm", "_dfm_prd"), ("_dstdm", "_dfm_dst")):
            a = pred_dir / (key + src_name + ".png")
            if not a.is_file():
                continue
            try:
                _stage_write_atomic(mask_dir / (key + dst_name + ".png"), a.read_bytes())
                n += 1
            except OSError:
                pass
        return n

    # ---- 串行链 → 三段流水线 ----
    # 以前这里是一条串行链：读 pred PNG → 读 DFLJPG 元数据 → 读 1080p 原帧 → 对齐到
    # 256 → XSeg(prd) → 编码写盘 → XSeg(dst) → 编码写盘 → 复制 dfm 的两张遮罩。
    # 纯推理（原生 TRT int8，见 stage_perf_profile.py 的实测）只有 1.44 ms/张，
    # 两张也就 2.9 ms；2059 张脸却只有 26 张/秒（≈38 ms/张）—— 也就是 92% 的时间
    # 不在推理上，而且串行意味着「各段相加」。现在拆成三段用队列连起来：
    #     读线程 ×N   pred PNG 解码 + aligned JPEG 解码并映射到 256（_stage_read_dst_face）
    #     推理线程    攒够 mask_batch 张脸，一次前向（ORT 动态 batch 或 TRT 逐张）
    #     写线程 ×M   PNG 编码 + 原子落盘 + dfm 两张遮罩的文件复制
    # 墙钟 ≈ max(三段) 而不是三段之和。
    # dst 输入换成 aligned 之后，读线程从「每张脸解码一张 1080p PNG（≈24 ms/张）」降到
    # 「解码一张 558 的 JPEG 再映射到 256（≈4 ms/张）」，同一帧的多张脸也不再共用帧缓存
    # （每张脸读自己那张 aligned）；写线程仍是「每张脸 4 个 PNG + 2 次文件复制」（~24 ms/张）。
    readers = max(1, int(cfg.get("mask_readers")
                         or min(6, max(2, (os.cpu_count() or 4) // 4))))
    writers = max(1, int(cfg.get("mask_writers")
                         or min(6, max(2, (os.cpu_count() or 4) // 8))))
    # 默认**不批量**（1）：本机实测 256² 逐张 4.79 ms/图 比 batch4 的 5.24 ms/图 更快
    # （ORT 在 batch>1 时换 kernel，流水线里还被队列拉长），而批量会换规约顺序、遮罩
    # 逐像素差 1~2 LSB。要试就传 mask_batch=N（引擎真吃批时的收益见 _xseg.mask_batch）。
    mbatch = max(1, int(cfg.get("mask_batch") or 1))
    if src["source"] == "xseg" and xseg is not None:
        try:
            # TRT 的 xseglite 引擎是静态 batch=1（_engines.PROFILES 单档位）→ 老实逐张；
            # ort-CUDA 的输入是动态 batch → 一次多张（256² 实测 4.02 → 2.68 ms/张）
            mbatch = max(1, min(mbatch, int(xseg.batch_limit())))
        except Exception:                                   # noqa: BLE001
            mbatch = 1
    _stage_log("遮罩：流水线 读 %d 线程 / 推理 batch %d / 写 %d 线程 · 后端 %s"
               % (readers, mbatch, writers, backend))

    if src["source"] != "xseg":
        # 来源=DFM：遮罩就是 ③ 落盘的那两张，这里只按 mask/ 的命名留档，不推理。
        k = cursor
        try:
            for k in range(cursor, total):
                _stage_check_cancel()
                key = aligned[k].stem
                try:
                    _copy_dfm_masks(key)
                    wrote += 1
                except Exception as e:                      # noqa: BLE001
                    failed += 1
                    _stage_log("遮罩跳过 %s：%s" % (key, str(e)[:140]))
                el = max(0.001, time.time() - t0)
                _stage_progress("mask", 100.0 * (k + 1) / total,
                                "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒"
                                % (k + 1, total, int(rng["start"]), int(rng["end"]),
                                   (k + 1 - cursor) / el),
                                frames_done=k + 1, frames_total=total,
                                frame_fps=round((k + 1 - cursor) / el, 2))
        except _StageCancel:
            _save(k)                # 取消也要把游标落盘：下次从这里接着跑
            _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
            raise
        sec = round(time.time() - t0, 2)
        done_n = max(1, total - cursor)
        _save(total, done=True, extra={
            "seconds": sec, "finished_at": time.time(),
            "fps": round((total - cursor) / sec, 2) if sec else None,
            "mask_mean": None, "infer_seconds": 0.0, "infer_ms_per_face": 0.0,
        })
        _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d · %.2f s · %.1f 张/秒 · "
                   "来源 %s（不推理，只落 dfm 自带遮罩）"
                   % (int(rng["start"]), int(rng["end"]), total, failed, sec,
                      (total - cursor) / sec if sec > 0 else 0.0,
                      src.get("label") or src.get("source")))
        return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
                "mask_source": src.get("label") or src.get("source"),
                "signature": sig, "engine": engine, "backend": backend,
                "engine_file": engine_file,
                "range": {"start": int(rng["start"]), "end": int(rng["end"])},
                "infer_ms_per_face": 0.0}

    # ---------------- XSeg：读 / 推理 / 写 三段流水线 ----------------
    q_prep = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, writers * mbatch * 2))
    stop = threading.Event()
    boss = {"exc": None}
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
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    def _put_prep(item):
        while not stop.is_set():
            try:
                q_prep.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _u8(m):
        return None if m is None else np.clip(m * 255.0, 0, 255).astype(np.uint8)

    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, prd_u8, dst_u8 = item
            _stage_tick("xseg")      # 交接点：XSeg 遮罩 → mask/*.png
            tw = time.perf_counter()
            try:
                _copy_dfm_masks(key)
                if prd_u8 is not None:
                    # PNG level=1（无损，只是少压一层）：掩码是灰度，level 3 每张只快一点点
                    _stage_write_atomic(mask_dir / (key + "_xseg_prd.png"),
                                        _stage_imencode_png(prd_u8, 1))
                if dst_u8 is not None:
                    _stage_write_atomic(mask_dir / (key + "_xseg_dst.png"),
                                        _stage_imencode_png(dst_u8, 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("遮罩写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                ap = aligned[gi]
                key = ap.stem
                tr = time.perf_counter()
                pm = pred_dir / (key + ".png")
                pred = _stage_read_image(pm) if pm.is_file() else None
                # dst 输入 = aligned/*.jpg（切脸阶段 68 点仿射的直接产物），只做一次「映射到
                # 合成画布」的重采样：不读 1080p 原帧（见 _stage_read_dst_face）。
                dst, why_dst = _stage_read_dst_face(ap)
                _bump("read_ms", time.perf_counter() - tr)
                why = ""
                if pred is None:
                    why = "pred/ 里没有换脸结果（先跑换脸）"
                elif dst is None:
                    why = "aligned 用不了 %s：%s" % (ap.name, why_dst)
                    _bad_aligned(ap.name)
                if why:
                    _put_prep((gi, key, None, None, why))
                    continue
                _put_prep((gi, key, pred, dst, ""))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_prep.put((None, None, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    jobs = []

    def _flush(force: bool = False):
        nonlocal t_infer, wrote
        while jobs and (len(jobs) >= mbatch or (force and jobs)):
            n = min(len(jobs), mbatch)
            chunk = jobs[:n]
            imgs, slots = [], []
            for i, (_gi, _k, pred, dst, _why) in enumerate(chunk):
                imgs.append(pred)
                slots.append((i, 0))
                if dst is not None:
                    imgs.append(dst)
                    slots.append((i, 1))
            ti = time.perf_counter()
            outs = xseg.mask_batch(imgs, out_size=256)
            dt = time.perf_counter() - ti
            t_infer += dt
            _bump("infer_ms", dt, n, len(imgs))
            _INFER_CALLS["xseg"] += len(imgs)
            prd_m = [None] * n
            dst_m = [None] * n
            for (i, slot), mk in zip(slots, outs):
                if slot:
                    dst_m[i] = mk
                else:
                    prd_m[i] = mk
                    means.append(float(np.asarray(mk).mean()))
            for i, (gi, key, _p, _d, _w) in enumerate(chunk):
                q_write.put((gi, key, _u8(prd_m[i]), _u8(dst_m[i])))
                wrote += 1
            del jobs[:n]

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="mask-w%d" % i)
        t.start()
        threads.append(t)
    for r in range(readers):
        lo = cursor + (total - cursor) * r // readers
        hi = cursor + (total - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="mask-r%d" % r)
        t.start()
        threads.append(t)

    got = 0                     # 已经处理完的张数（显示用；游标 commit 会为了不留空洞而滞后）
    k = cursor
    cancelled = {"flag": False}
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_prep.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, pred, dst, why = item
            if pred is None:
                failed += 1
                _stage_log("遮罩跳过 %s：%s" % (key, why or "pred/ 里没有换脸结果（先跑换脸）"))
                with plock2:
                    completed.add(gi2)
            else:
                _pend_add(gi2)
                jobs.append((gi2, key, pred, dst, ""))
                _flush()
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("mask", 100.0 * got / total,
                            "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]),
                               got / el, mbatch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
    except _StageCancel:
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            _flush(force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
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
        # 取消也要把游标落盘（排空之后再落，才和磁盘上真的有的文件对得上）
        _save(k)
        _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    if total > 0 and int(prof["faces"]) == 0:
        # 一张都没推理成功 → 别把空 mask/ 当「已完成」（下游合成会拿不到遮罩却以为跑过了）
        raise _BadInput("遮罩：%d 张对齐脸一张都没推理成功（aligned 读不出来 %d 张 / pred 缺失？）"
                        "—— 空 mask/ 不算成功。先看 aligned/ 和 pred/：%s"
                        % (total, int(prof.get("bad_aligned") or 0),
                           "、".join(bad_aligned[:5]) or "-"), 500)
    sec = round(time.time() - t0, 2)
    done_n = max(1, total - cursor)
    n_faces = max(1, prof["faces"])
    prof_out = {
        "read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
        "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
        "readers": int(readers), "writers": int(writers), "mask_batch": int(mbatch),
        "backend": backend, "engine_file": engine_file,
        "images": int(prof["images"]),
        "dst_input": _MASK_DST_INPUT, "bad_aligned": int(prof.get("bad_aligned") or 0),
        "dst_mat": "image_to_face_mat⁻¹ · facelib WHOLE_FACE",
        # 每张脸各段各花多少 ms（读图 / 预处理（这一版恒 0）/ 2 次推理 / 编码+落盘）
        "per_face_ms": {
            "read": round(prof["read_ms"] / max(1, total - cursor), 2),
            "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
            "infer": round(prof["infer_ms"] / n_faces, 2),
            "write": round(prof["write_ms"] / n_faces, 2),
        },
        # 单张纯推理 = 总推理时间 / 推理张数（一张脸两张图）
        "per_image_ms": round(prof["infer_ms"] / max(1, prof["images"]), 3),
    }
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "fps": round((total - cursor) / sec, 2) if sec else None,
        "mask_mean": (round(sum(means) / len(means), 4) if means else None),
        # TRT vs ORT 的对比就靠这两个数：同一个视频、同一份模型、同一个预处理
        "infer_seconds": round(t_infer, 3),
        "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                              if src["source"] == "xseg" else 0.0),
        "profile": prof_out,
    })
    _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d%s · %.2f s · %.1f 张/秒 · "
               "后端 %s%s · 来源 %s · dst 输入 %s · 每张 读 %.1f + 预处理 %.1f + 推理 %.1f"
               "（%.2f ms/图 × %d 图）+ 写 %.1f ms（读 %d / 写 %d 线程 · batch %d）"
               % (int(rng["start"]), int(rng["end"]), total, failed,
                  ("（其中 aligned 读不出来 %d）" % int(prof.get("bad_aligned") or 0))
                  if int(prof.get("bad_aligned") or 0) else "", sec,
                  (total - cursor) / sec if sec > 0 else 0.0,
                  backend, ("(%s)" % engine_file) if engine_file else "",
                  src.get("label") or src.get("source"), _MASK_DST_INPUT,
                  prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_image_ms"],
                  prof["images"], prof_out["per_face_ms"]["write"],
                  readers, writers, mbatch))
    return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
            "mask_source": src.get("label") or src.get("source"),
            "signature": sig, "engine": engine, "backend": backend,
            "engine_file": engine_file, "dst_input": _MASK_DST_INPUT,
            "bad_aligned": int(prof.get("bad_aligned") or 0),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "mask_batch": int(mbatch), "profile": prof_out,
            "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                                  if src["source"] == "xseg" else 0.0)}
