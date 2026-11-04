"""B 站批量下载队列（从 4.0 的 _DownloadManager 移植，去掉 Qt 依赖）。

4.0 里是一个 Qt 对象 + 2 个 worker 线程 + 信号回传进度；这里换成
纯线程实现，状态存在一个 dict 里，前端轮询 /api/bili/batch/status。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .bili_video import download_single_video

_LOCK = threading.Lock()
_STATE = {
    "running": False,
    "items": [],          # [{bvid, state, pct, speed, error, path}]
    "started": 0,
    "finished": 0,
}
_THREAD = None


def snapshot() -> dict:
    with _LOCK:
        return {
            "running": _STATE["running"],
            "started": _STATE["started"],
            "finished": _STATE["finished"],
            "items": [dict(it) for it in _STATE["items"]],
        }


def _worker(idx: int, cookie: str, out_dir: str, ffmpeg: str):
    with _LOCK:
        it = _STATE["items"][idx]
        it["state"] = "downloading"

    def cb(fraction, speed, stage):
        with _LOCK:
            it = _STATE["items"][idx]
            it["pct"] = round(float(fraction) * 100, 1)
            it["speed"] = round(float(speed), 2)
            it["stage"] = stage or ""
        return True

    try:
        ok = download_single_video(it["bvid"], cookie, out_dir,
                                   progress_callback=cb, ffmpeg_path=ffmpeg)
    except Exception as e:
        ok = False
        with _LOCK:
            _STATE["items"][idx]["error"] = str(e)[:200]

    with _LOCK:
        it = _STATE["items"][idx]
        it["state"] = "done" if ok else "failed"
        it["pct"] = 100.0 if ok else it.get("pct", 0.0)
        _STATE["finished"] += 1
        remaining = [x for x in _STATE["items"] if x["state"] in ("queued", "downloading")]
        if not remaining:
            _STATE["running"] = False


def start(bvids, cookie: str, out_dir: str, ffmpeg: str, workers: int = 2) -> dict:
    global _THREAD
    with _LOCK:
        if _STATE["running"]:
            return {"ok": False, "error": "已有下载任务在跑"}
        _STATE["items"] = [{"bvid": b, "state": "queued", "pct": 0.0,
                            "speed": 0.0, "stage": "", "error": "", "path": ""}
                           for b in bvids]
        _STATE["running"] = True
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
            _worker(idx, cookie, out_dir, ffmpeg)

    threads = [threading.Thread(target=runner, daemon=True)
               for _ in range(max(1, min(workers, len(bvids))))]
    for t in threads:
        t.start()
    _THREAD = threads[0] if threads else None
    return {"ok": True, "count": len(bvids)}
