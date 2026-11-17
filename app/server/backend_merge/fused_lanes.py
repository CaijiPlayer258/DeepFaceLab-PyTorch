# -*- coding: utf-8 -*-
"""融合流水线的"车道"体：从 _fused_run 里搬出来的闭包。

搬法：**函数体逐字不动**，只把原先靠闭包捕获的名字改成同名的显式关键字参数 ——
搬前搬后是同一条代码路径，不是重写。判据（app/docs/notes/fused-baseline.md）：
  · fused_pipeline_bench.py --frames 30 --modes B  → 30 个 PNG 的 md5 逐行相同
  · fused_export_accept.py --cases a               → PASS 20 / FAIL 0

第一代只搬了 decode。detect / prep / gpu / compose / encode 仍在 _fused_run 里，
按 docs/notes/fused-run-map.md（六段 + 行号）逐代搬。
"""
from __future__ import annotations


def decode_lane(*, readinto_on, probe, budget, video, start_i, dec_mode, src_bits, dec_threads, gen, stop, end_i, ticks, sem, live_lock, live, det_prep_on, _det_prep, det, q_dec, _fail, _put, _stage_tick, _FusedFrame, np, time, _fused_open_frames, END):
    """① 解码：ffmpeg 的 rawvideo 流 → 帧对象，塞进 q_dec（原样搬自 _fused_run._decode）。"""
    proc = None
    # readinto 的环形缓冲：槽数 = 帧预算 + 4。为什么不担心被覆盖：在飞帧由 sem 卡在
    # budget 帧（decode 这里是「先读、后 acquire」，所以最多 budget+1 块同时活着），
    # 而 frame 引用在合成之后就放了（_compose 里 job.frame = None）。
    pool = []
    if readinto_on:
        try:
            _pw, _ph = int(probe.get("w") or 0), int(probe.get("h") or 0)
            if _pw > 0 and _ph > 0:
                pool = [np.empty((_ph, _pw, 3), np.uint8)
                        for _ in range(int(budget) + 4)]
        except Exception:                                  # noqa: BLE001
            pool = []
    pi = 0
    try:
        proc, next_frame = _fused_open_frames(video, start_i,
                                             size=(probe.get("w"), probe.get("h")),
                                             mode=dec_mode, bits=(src_bits or None),
                                             threads=dec_threads)
        gen["proc"] = proc
        n = start_i - 1
        while not stop.is_set():
            if n >= end_i:
                break
            t_rd = time.perf_counter()
            if pool:
                frame = next_frame(pool[pi])
                if frame is not None:
                    pi = (pi + 1) % len(pool)
            else:
                frame = next_frame()
            ticks.add("decode", time.perf_counter() - t_rd)
            if frame is None:
                break
            n += 1
            t_wait = time.perf_counter()
            while not stop.is_set():
                if sem.acquire(timeout=0.2):
                    break
            ticks.add("dec_wait", time.perf_counter() - t_wait)
            if stop.is_set():
                break
            with live_lock:
                live["n"] += 1
                if live["n"] > live["peak"]:
                    live["peak"] = live["n"]
            job = _FusedFrame(n, n - start_i + 1, frame)
            if det_prep_on:
                # 检测的 letterbox 在这条线程做：它只忙 15ms/帧（30%），有富余；
                # 检测线程是硬底，把 0.7ms 的 resize + 填边搬走就是净赚。
                # 逐位一致由「同一个 prep()」保证（见 _detect.py 的 _prep_ctx）。
                t_prep = time.perf_counter()
                job.det_ctx = _det_prep(det, frame)
                if job.det_ctx is not None:
                    ticks.add("det_prep", time.perf_counter() - t_prep)
            _stage_tick("frames")            # 交接点：解码 → 检测队列
            if not _put(q_dec, job):
                break
    except BaseException as e:                             # noqa: BLE001
        _fail("decode", e)
    finally:
        _put(q_dec, END)


def detect_lane(*, k=0, det_obj=None, n_workers=1, time, _get, q_dec, ticks, END, _put, n_prep, q_det, det_prep_pool, det_prep_on, _det_prep, det, _detect_on_ctx, det_thresh, det_name, _detect_boxes_ctx, _detect_on, _detect_boxes, _fail, _stage_tick):
    """② 检测：从 q_dec 取帧 → 前向 + NMS → 按 x1 排序 → q_det（原样搬自 _fused_run._detect）。"""
    try:
        while True:
            t_wait = time.perf_counter()
            job = _get(q_dec)
            ticks.add("det_wait", time.perf_counter() - t_wait)
            if job is END:
                # 毒丸只放了一颗：谁吃到谁负责传下去（n_workers-1 颗），
                # 最后各自退出 —— 不然其余几路会在 _get 里等到天荒地老。
                for _i in range(max(0, int(n_workers) - 1)):
                    _put(q_dec, END)
                for _i in range(n_prep):
                    _put(q_det, END)
                break
            t = time.perf_counter()
            try:
                if job.det_ctx is None and det_prep_pool and not det_prep_on:
                    # pool 档：letterbox 在本线程做，并复用本线程那张画布
                    # （prep → 前向 → 立刻 drop，所以复用安全；见 _prep_ctx 的注释）
                    tq = time.perf_counter()
                    _ctx = _det_prep(det if det_obj is None else det_obj, job.frame,
                                     reuse=True)
                    if _ctx is not None:
                        ticks.add("det_prep", time.perf_counter() - tq)
                        job.det_ctx = _ctx
                if job.det_ctx is not None:
                    # 主路径：canvas 已经在 decode 线程做好，这里只剩前向 + NMS
                    boxes = (_detect_on_ctx(det_obj, job.det_ctx, det_thresh)
                             if det_obj is not None
                             else _detect_boxes_ctx(job.det_ctx, det_name, det_thresh))
                    job.det_ctx = None       # 1.2MB/帧：用完就放，别带到合成再扔
                else:
                    boxes = (_detect_on(det_obj, job.frame, det_thresh)
                             if det_obj is not None
                             else _detect_boxes(job.frame, det_name, det_thresh))
            except Exception as e:                         # noqa: BLE001
                _fail("detect", e)
                for _i in range(n_prep):
                    _put(q_det, END)
                break
            ticks.add("detect", time.perf_counter() - t)
            # 和 ② 切脸同一个次序（按 x1 排）：脸序号 = 这个顺序，compose 也按它贴
            job.boxes = sorted(list(boxes or []),
                               key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
            job.rec["detected"] = len(job.boxes)
            _stage_tick("det")               # 交接点：检测 → 对齐/预处理
            _put(q_det, job)
    except BaseException as e:                             # noqa: BLE001
        _fail("detect", e)

# ---- ③ 对齐/预处理 + 归组：68 点、dfm 输入、xseg 的 dst 画布、这一帧用哪个 dfm ----
