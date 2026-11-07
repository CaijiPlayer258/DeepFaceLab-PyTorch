"""抽取流水线：解码 → 检测 →（调用线程）对齐/落盘，级间有界队列。

为什么单独一个模块：这段逻辑只有三件事 —— 有界队列、取消、错误回传 ——
塞进 app.py 会跟 JOB / 检测器 / 对齐三套状态搅在一起，改一处要看五处。

本机实测（201 帧 1080x1920 HEVC，RTX 5070 Laptop，quality=high）：
    只解码                1.40~1.50 s（其中 1.34~1.44 s 是 ffmpeg 管道读取）
    检测（孤立、无并发）  7.6~9.1 ms/帧  → 约 1.5~1.8 s
    检测（有解码线程并发）11.2~12.3 ms/帧 → 约 2.3~2.5 s
    三级三线程（改前）    2.40~2.69 s，其中检测占 2.28~2.65 s
所以这条流水线**省不出总时间**：它跑在 max(各段) 上，而最大的一段是检测，
检测自己的耗时又会被并发抬高（诊断过：拿掉写盘只省 0.2 s，软解/零拷贝读取
都不改善 —— 不是 GIL 单点、也不是 NVDEC 抢 GPU）。真正的收益是别的东西：
内存有界、可取消、错误能回传、帧序不变、允许把写盘藏到检测背后。
「加了线程」不等于「并行打满」—— 结构只决定谁等谁，决定不了最慢那一级的速度。

设计口径：
    · 生产 = 解码线程；消费 = 检测线程；下游（对齐 + JPEG + DFLJPG 落盘）在
      **调用线程**里跑：谁写文件谁顺序，不需要额外排序或序号重排。
    · 两级队列都有界：q_in 卡住解码（背压），q_out 卡住检测。常驻内存约为
      (q_in + q_out + 批上限 + 2) 帧：1080p 每帧 6.2 MB，默认 6/6 ≈ 90 MB 峰期，
      长视频不增长。DFL_EXTRACT_Q=2 时 ≈ 30 MB。
    · q_in 至少等于 batch：TinyMog 批 8 而队列 6 的话永远凑不满一批，只能靠"等"
      凑批 —— 实测 fast 档因此慢 17%（2.60s → 2.18s 就是去掉等待换来的）。
    · 任一级抛异常：记下 (阶段, 异常, 远端 traceback)，置 stop，调用线程收尾后
      用 PipelineError 抛回 —— 不让异常烂在线程里变成「进度卡在 37%」。
    · 取消：调用方给 should_stop；所有 put/get 都用 poll 粒度超时，所以 stop 之后
      最迟 poll 秒内醒；收尾顺序是 置 stop → 排空两级队列 → join(超时)。
    · 两个 worker 都是 daemon；join 超时只是「报告还活着」，不会拖住调用方退出。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

__all__ = ["PipelineError", "capacity", "run", "pipeline_enabled"]

# 队列哨兵：和 (idx, frame) 不同型，必须用 is 判
_SENTINEL = object()

DEFAULT_Q = 6          # 4~8 是实测甜区：再深只多占内存，再浅解码会被检测拖住


class PipelineError(RuntimeError):
    """worker 线程里的异常带回调用线程。原始异常挂在 .original 上。"""

    def __init__(self, stage, exc, remote_tb=""):
        super().__init__("抽取流水线在 [%s] 阶段失败：%s: %s"
                         % (stage, type(exc).__name__, exc))
        self.stage = stage
        self.original = exc
        self.remote_traceback = remote_tb


class _Stopped(Exception):
    """内部信号：已取消 / 已出错，该级收工。不是错误，不往外抛。"""


def pipeline_enabled(default=True) -> bool:
    """DFL_EXTRACT_PIPELINE=0/off/false/no → 回退 app.py 里原来的多线程实现。"""
    v = os.environ.get("DFL_EXTRACT_PIPELINE")
    if v is None or not str(v).strip():
        return bool(default)
    return str(v).strip().lower() not in ("0", "off", "false", "no", "none")


def capacity(default=DEFAULT_Q) -> int:
    """队列容量：DFL_EXTRACT_Q，默认 6。下限 1（背压测试要用 2），上限 64。"""
    try:
        n = int(str(os.environ.get("DFL_EXTRACT_Q") or default).strip())
    except (TypeError, ValueError):
        n = default
    return max(1, min(64, n))


def run(*, decode, detect_batch, emit, batch=1, q_in=None, q_out=None,
        should_stop=None, poll=0.1, join_timeout=5.0,
        prof=None, name="extract"):
    """跑一次三级流水线。

    decode(put)               解码线程：逐帧 put(item)（item 的格式由调用方定），返回即结束
    detect_batch([item, ...]) 检测线程：返回与入参等长的 [[face, ...], ...]
    emit(item, faces)         调用线程：对齐 + 落盘，返回值累加进 made
    batch                     单次检测最多凑几帧（RetinaFace=1，TinyMog=8）
    should_stop()             取消判据（app 里是 not JOB.running）
    prof                      计时累加用的 dict（put1/get1/det/put2/get2/emit）

    返回 {"made", "cancelled", "alive", "wall"}；worker 出错时抛 PipelineError。
    """
    T = time.perf_counter
    prof = prof if prof is not None else {}
    for k in ("put1", "get1", "det", "put2", "get2", "emit"):
        prof.setdefault(k, 0.0)

    batch = max(1, int(batch))
    # 队列至少要放得下一整批：TinyMog 的 batch 是 8，队列只有 6 的话
    # 永远凑不满一批 —— 那就只能靠"等"，等来的延迟比省下的固定开销更贵
    # （实测 quality=fast：等待版 2.60s vs 别人 2.21s）。
    cap_in = max(batch, int(q_in or capacity()))
    cap_out = max(1, int(q_out or cap_in))
    q_frames = queue.Queue(maxsize=cap_in)
    q_faces = queue.Queue(maxsize=cap_out)
    stop = threading.Event()
    lock = threading.Lock()
    state = {"err": None, "cancelled": False, "made": 0, "q1max": 0, "q2max": 0}

    def _wants_stop():
        if stop.is_set():
            return True
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:
            return True                  # 取消判据自己炸了，按取消处理

    def _fail(stage):
        exc = sys.exc_info()[1]
        with lock:
            if state["err"] is None:
                state["err"] = (stage, exc, traceback.format_exc())
        stop.set()

    def _put(q, item, qkey):
        """阻塞放；满就按 poll 粒度醒来看看是不是该收工了。顺手记队列峰值深度。"""
        while True:
            if stop.is_set():
                raise _Stopped()
            try:
                q.put(item, timeout=poll)
                depth = q.qsize()
                if depth > state[qkey]:
                    state[qkey] = depth
                return
            except queue.Full:
                continue

    def _drain(q):
        """排空：让还在 put 的线程下一次醒来就能看到 stop。"""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _decoder():
        try:
            def put(item):
                t = T()
                _put(q_frames, item, "q1max")
                prof["put1"] += T() - t
            decode(put)
        except _Stopped:
            pass
        except BaseException:               # noqa: BLE001 —— 任何异常都要带回主线程
            _fail("decode")
        finally:
            try:
                q_frames.put(_SENTINEL, timeout=max(0.05, poll))
            except queue.Full:
                pass                        # 消费者已收工，没人读了，不硬塞

    def _detector():
        pending = []

        def flush():
            if not pending:
                return
            items = pending[:]
            del pending[:]
            t = T()
            faces_list = detect_batch(items)
            prof["det"] += T() - t
            faces_list = list(faces_list or [])
            if len(faces_list) < len(items):
                faces_list += [[] for _ in range(len(items) - len(faces_list))]
            t = T()
            for it, faces in zip(items, faces_list):
                _put(q_faces, (it, faces), "q2max")
            prof["put2"] += T() - t

        try:
            while True:
                if _wants_stop():
                    return
                t = T()
                try:
                    item = q_frames.get(timeout=poll)
                except queue.Empty:
                    continue
                prof["get1"] += T() - t
                if item is _SENTINEL:
                    flush()                 # 生产者结束：不满一批也发小批
                    break
                pending.append(item)
                if len(pending) >= batch:
                    flush()
                    continue
                # 凑批：只吃队列里**已经有**的帧，绝不为凑批去等 ——
                # 等来的那点批处理收益，抵不过一段 20ms 的空转（实测 fast 档慢 17%）。
                done = False
                while len(pending) < batch:
                    try:
                        nxt = q_frames.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _SENTINEL:
                        done = True
                        break
                    pending.append(nxt)
                flush()
                if done:
                    break
        except _Stopped:
            pass
        except BaseException:               # noqa: BLE001
            _fail("detect")
        finally:
            try:
                q_faces.put(_SENTINEL, timeout=max(0.05, poll))
            except queue.Full:
                pass

    dec_th = threading.Thread(target=_decoder, name=name + "-decode", daemon=True)
    det_th = threading.Thread(target=_detector, name=name + "-detect", daemon=True)
    t0 = T()
    dec_th.start()
    det_th.start()
    try:
        while True:
            if _wants_stop():
                state["cancelled"] = True
                break
            if state["err"] is not None:
                break
            t = T()
            try:
                item = q_faces.get(timeout=poll)
            except queue.Empty:
                prof["get2"] += T() - t
                if not det_th.is_alive() and q_faces.empty():
                    break               # 检测线程收工且下游已空 —— 正常结束的兜底
                continue
            prof["get2"] += T() - t
            if item is _SENTINEL:
                break
            t = T()
            state["made"] += int(emit(item[0], item[1]) or 0)
            prof["emit"] += T() - t
    finally:
        stop.set()
        _drain(q_frames)
        _drain(q_faces)
        deadline = T() + max(0.0, float(join_timeout))
        for th in (dec_th, det_th):
            th.join(max(0.0, deadline - T()))

    alive = [th.name for th in (dec_th, det_th) if th.is_alive()]
    if state["err"] is not None:
        stage, exc, tb = state["err"]
        raise PipelineError(stage, exc, tb) from exc
    return {"made": state["made"], "cancelled": bool(state["cancelled"]),
            "alive": alive, "wall": T() - t0,
            "q1max": state["q1max"], "q2max": state["q2max"]}
