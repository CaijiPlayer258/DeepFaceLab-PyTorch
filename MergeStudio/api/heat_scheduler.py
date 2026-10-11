# -*- coding: utf-8 -*-
"""热区调度器：把用户的交互轨迹变成预取区间并集，按优先级填满。

设计（2026-10-11 检查模式融入主 UI）：
- 交互轨迹（拖动经过的帧、停留点、框选区间）→ 记录为区间
- 区间自动合并（重叠/相邻 30 帧内合并）
- 单一后台预取线程消化区间；交互帧请求永远优先（不与本调度器争 executor）
- cache-status 升级：返回区间列表 + 各区间已缓存数（供时间线热区渲染）

对外 API（routes_preview 内使用）：
- heat_report(frame_idx)             交互信号：记录访问点（拖动/停留）
- heat_declare(start, end)           显式区间（框选/慢放目标段）
- heat_regions() / heat_status()     状态查询
- heat_start() / heat_stop()         生命周期（select-video 时重启）
"""
import threading
import time
from pathlib import Path

_MERGE_GAP = 30          # 区间间距小于此值则合并
_INTERACT_RADIUS = 80    # 交互点向后预取半径（与旧行为一致）
_MAX_REGIONS = 8         # 最多保留的热区数（LRU 淘汰最旧）

_lock = threading.Lock()
_regions = []            # [[start, end, last_touch], ...] 升序
_disk_dir = None         # Path 或 None（None=未激活）
_total = 0
_worker = None
_stop_evt = None


def _normalize(regions):
    regions.sort(key=lambda r: r[0])
    out = []
    for r in regions:
        if out and r[0] - out[-1][1] <= _MERGE_GAP:
            out[-1][1] = max(out[-1][1], r[1])
            out[-1][2] = max(out[-1][2], r[2])
        else:
            out.append(r)
    return out


def reset(disk_dir, total):
    """切换视频：重置区间，重启工作线程。disk_dir=None 时停用。"""
    global _regions, _disk_dir, _total, _worker, _stop_evt
    with _lock:
        _regions = []
        _disk_dir = Path(disk_dir) if disk_dir else None
        _total = int(total) if total else 0
        if _stop_evt is not None:
            _stop_evt.set()
        _stop_evt = None
        _worker = None
        if _disk_dir is not None and _total > 0:
            _stop_evt = threading.Event()
            _worker = threading.Thread(target=_digest_loop, daemon=True,
                                       name='heat-predecode')
            _worker.start()


def _touch_locked(start, end):
    global _regions
    now = time.time()
    _regions.append([start, end, now])
    _regions = _normalize(_regions)
    while len(_regions) > _MAX_REGIONS:
        _regions.pop(0)   # 最旧（已按 start 排序，pop(0) 简化为丢最靠前的）

def heat_report(frame_idx):
    """交互信号：访问点 → [idx, idx+_INTERACT_RADIUS] 热区。"""
    if _disk_dir is None:
        return
    with _lock:
        _touch_locked(max(0, frame_idx), min(frame_idx + _INTERACT_RADIUS, _total - 1))


def heat_declare(start, end):
    """显式声明区间（框选/慢放段）。"""
    if _disk_dir is None:
        return
    s, e = max(0, int(start)), min(int(end), _total - 1)
    if e < s:
        return
    with _lock:
        _touch_locked(s, e)


def heat_regions():
    with _lock:
        return [(r[0], r[1]) for r in _regions]


def heat_status():
    """区间 + 每区间磁盘命中数（热区渲染数据源）。"""
    with _lock:
        regions = [(r[0], r[1]) for r in _regions]
    out = []
    if _disk_dir is not None:
        for s, e in regions:
            have = 0
            for i in range(s, e + 1):
                if (_disk_dir / f"{i}.jpg").exists():
                    have += 1
            out.append({"start": s, "end": e, "cached": have,
                        "total": e - s + 1,
                        "pct": round(have / (e - s + 1) * 100, 1)})
    return out


def _missing_chunk(region):
    """区间内第一个未缓存连续缺口 (s,e)；全满返回 None。"""
    s, e = region
    i = s
    while i <= e:
        if not (_disk_dir / f"{i}.jpg").exists():
            j = i
            while j < e and not (_disk_dir / f"{j+1}.jpg").exists():
                j += 1
            return (i, j)
        i += 1
    return None


def _digest_loop():
    """预取工作线程：循环消化最新声明的区间缺口（新交互永远先于旧尾巴）。"""
    import cv2
    global _regions
    evt = _stop_evt
    while not evt.is_set() and _disk_dir is not None:
        # 取「最新触达且未满」的区间（尾部优先=最近的交互）
        with _lock:
            pending = None
            for r in reversed(_regions):
                if _missing_chunk((r[0], r[1])) is not None:
                    pending = (r[0], r[1])
                    break
        if pending is None:
            time.sleep(0.4)   # 无事可做
            continue
        chunk = _missing_chunk(pending)
        if chunk is None:
            continue
        cs, ce = chunk
        cap = None
        try:
            from MergeStudio.api import routes_preview as _rp
            src = _rp._resolve_preview_source(_rp._current_video) or _rp._current_video
            if src is None:
                return
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
            n_yield = 0
            for i in range(cs, ce + 1):
                if evt.is_set():
                    return
                if (_disk_dir / f"{i}.jpg").exists():
                    cap.grab()
                    continue
                n_yield += 1
                if n_yield >= 8:
                    time.sleep(0.02)   # 让出：交互帧优先
                    n_yield = 0
                ret, frame = cap.read()
                if not ret:
                    break
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,
                                                       _rp.FRAME_CACHE_QUALITY])
                if ok:
                    (_disk_dir / f"{i}.jpg").write_bytes(jpeg.tobytes())
        except Exception:
            time.sleep(1.0)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
