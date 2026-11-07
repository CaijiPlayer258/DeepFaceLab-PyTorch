"""B 站批量下载队列（从 4.0 的 _DownloadManager 移植，去掉 Qt 依赖）。

4.0 里是一个 Qt 对象 + 2 个 worker 线程 + 信号回传进度；这里换成
纯线程实现，状态存在一个 dict 里，前端轮询 /api/bili/batch/status。
下载本身是 bili_video.download_single_video（真实 DASH 音视频），失败会带
error 字段回来，前端必须显示，不允许默默变成「已完成」。
"""
from __future__ import annotations

import threading
import time

from .bili_video import download_single_video
from .bili_http import BiliError

_LOCK = threading.Lock()
_STATE = {
    "running": False,
    "paused": False,
    "cancelled": False,
    "items": [],          # [{bvid, state, pct, speed, stage, error, path}]
    "started": 0,
    "finished": 0,
}
_THREAD = None


def snapshot() -> dict:
    with _LOCK:
        return {
            "running": _STATE["running"],
            "paused": _STATE["paused"],
            "cancelled": _STATE["cancelled"],
            "started": _STATE["started"],
            "finished": _STATE["finished"],
            "items": [dict(it) for it in _STATE["items"]],
        }


def cancel() -> dict:
    """取消：未开始的直接标记，正在下的靠 progress_callback 返回 False 中断。"""
    with _LOCK:
        _STATE["cancelled"] = True
        _STATE["paused"] = False
        for it in _STATE["items"]:
            if it["state"] == "queued":
                it["state"] = "cancelled"
    return snapshot()


def set_paused(paused: bool) -> dict:
    with _LOCK:
        _STATE["paused"] = bool(paused)
    return snapshot()


def _gate(idx: int) -> bool:
    """暂停就在这里卡住；返回 False 表示已被取消。"""
    while True:
        with _LOCK:
            paused = _STATE["paused"]
            cancelled = _STATE["cancelled"]
        if cancelled:
            return False
        if not paused:
            return True
        time.sleep(0.2)


def _worker(idx: int, cookie: str, out_dir: str, ffmpeg: str, qn: int):
    with _LOCK:
        _STATE["items"][idx]["state"] = "downloading"

    def cb(fraction, speed, stage):
        if not _gate(idx):
            return False
        with _LOCK:
            it = _STATE["items"][idx]
            it["pct"] = round(float(fraction) * 100, 1)
            it["speed"] = round(float(speed), 2)
            it["stage"] = stage or ""
        return True

    ok = False
    err = ""
    path = ""
    try:
        path = download_single_video(_STATE["items"][idx]["bvid"], cookie, out_dir,
                                     progress_callback=cb, ffmpeg_path=ffmpeg, qn=qn)
        ok = bool(path)
    except Exception as e:
        err = str(e)[:300]
        if isinstance(e, BiliError):
            err = "[%s] %s" % (e.kind, e)

    with _LOCK:
        it = _STATE["items"][idx]
        if _STATE["cancelled"] and not ok:
            it["state"] = "cancelled"
        else:
            it["state"] = "done" if ok else "failed"
            it["pct"] = 100.0 if ok else it.get("pct", 0.0)
            it["path"] = path or ""
            it["error"] = err
        _STATE["finished"] += 1
        remaining = [x for x in _STATE["items"] if x["state"] in ("queued", "downloading")]
        if not remaining:
            _STATE["running"] = False


def start(bvids, cookie: str, out_dir: str, ffmpeg: str, workers: int = 2,
          qn: int = 120) -> dict:
    global _THREAD
    with _LOCK:
        if _STATE["running"]:
            return {"ok": False, "error": "已有下载任务在跑"}
        _STATE["items"] = [{"bvid": b, "state": "queued", "pct": 0.0,
                            "speed": 0.0, "stage": "", "error": "", "path": ""}
                           for b in bvids]
        _STATE["running"] = True
        _STATE["paused"] = False
        _STATE["cancelled"] = False
        _STATE["started"] = len(bvids)
        _STATE["finished"] = 0

    q = list(range(len(bvids)))
    q_lock = threading.Lock()

    def runner():
        while True:
            with q_lock:
                if not q:
                    return
                idx = q.pop(0)
            _worker(idx, cookie, out_dir, ffmpeg, qn)

    threads = [threading.Thread(target=runner, daemon=True)
               for _ in range(max(1, min(workers, len(bvids))))]
    for t in threads:
        t.start()
    _THREAD = threads[0] if threads else None
    return {"ok": True, "count": len(bvids)}
