# -*- coding: utf-8 -*-
"""合成用的取帧 / 建项目（**不是提取器**）：从 backend_extract/frames.py 搬出来。

  _merge_prepare_worker   视频 → 项目（000000.jpg + aligned/000000_0.jpg）
  _merge_frame_image      合成器按当前打开的 视频/项目 取一帧

为什么搬：它们不属于「提取页」，调用方全在 backend_merge（routes.py / compose.py）。
留在提取器包里会让人以为"提取器还有第二个职责"，照着名字找会找不到。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .._common.bootstrap import _NO_WINDOW
from .._common.jobs import _merge_cache, _merge_prep, _merge_prep_lock
from ..backend_extract.media import _dfl_project_scan, _extract_frames, _imread_any


def _merge_prepare_worker(video: Path, project: Path):
    import threading as _th

    def _set(**kw):
        with _merge_prep_lock:
            _merge_prep.update(kw)

    try:
        project.mkdir(parents=True, exist_ok=True)
        aligned = project / "aligned"
        aligned.mkdir(parents=True, exist_ok=True)

        _set(stage="抽帧", pct=0, msg="分析视频…")
        try:
            import imageio_ffmpeg as _iff
            n, secs = _iff.count_frames_and_secs(str(video))
            total = int(n) if n else 0
            src_fps = (n / secs) if secs else 30.0
        except Exception:
            total, src_fps = 0, 30.0

        # 快路径：同一个视频已经准备过（帧数对得上、对齐脸也在），直接复用
        have = len(list(project.glob("*.jpg")))
        have_aligned = len(list(aligned.glob("*.jpg")))
        if total and have == total and have_aligned > 0:
            _set(frames=have, aligned=have_aligned, pct=100,
                 msg="复用已准备好的项目：%d 帧 · %d 张对齐脸" % (have, have_aligned))
            try:
                names, srcdir, kind = _dfl_project_scan(project)
                _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                     "names": names, "path": "", "frames": len(names),
                                     "fps": src_fps, "w": 0, "h": 0})
            except Exception:
                pass
            _set(running=False, ok=True, stage="完成")
            return

        def _on_frame(made):
            pct = int(35 * made / total) if total else 0
            _set(pct=min(35, pct), msg="抽帧 %d/%s" % (made, total or "?"))

        # 只有确认要重跑才清对齐脸（上面快路径会复用它们）
        for old in aligned.glob("*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
        got = _extract_frames(video, project, total, _on_frame)
        _set(frames=got, pct=35, msg="抽帧完成：%d 帧" % got)

        # 对齐：复用提取器的解码+检测+对齐流水线（它会自己开 JOB）
        _set(stage="对齐", msg="加载检测器…")
        holder = {}

        def _run_align():
            # _align_stream 自己不置 running —— 提取器那边是调用方置的。
            # 不置的话它内部那句 "if not JOB.running: break" 会立刻退出，产出 0 张脸。
            try:
                JOB.running = True
                holder["n"] = _align_stream(str(video), aligned, fps=src_fps,
                                            rotate=0, quality="high", skip=1,
                                            src_fps=src_fps)
            except Exception as e:                     # noqa: BLE001
                holder["err"] = e
            finally:
                JOB.running = False

        th = _th.Thread(target=_run_align, daemon=True)
        th.start()
        while th.is_alive():
            try:
                done = int(getattr(JOB, "done", 0) or 0)
                tot = int(getattr(JOB, "total", 0) or 0) or got or 1
                pct = 35 + int(63 * min(1.0, done / float(tot)))
                _set(pct=min(98, pct), msg="对齐 %d/%d" % (done, tot))
            except Exception:
                pass
            th.join(0.4)
        if "err" in holder:
            raise holder["err"]
        made = int(holder.get("n") or 0)
        _set(aligned=made, pct=100, msg="完成：%d 帧 · %d 张对齐脸" % (got, made))
        if not made:
            _set(ok=False, error="对齐没有产出（视频里没检出人脸？）",
                 msg="对齐失败：0 张脸")

        # 直接把准备好的项目打开，前端不用再点一次
        try:
            names, srcdir, kind = _dfl_project_scan(project)
            _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                 "names": names, "path": "", "frames": len(names),
                                 "fps": src_fps, "w": 0, "h": 0})
        except Exception:
            pass
        _set(running=False, ok=True, stage="完成")
    except Exception as e:                             # noqa: BLE001
        _set(running=False, ok=False, error=str(e)[:300], msg="失败", stage="出错")

def _merge_frame_image(idx: int):
    """按当前打开的东西取一帧：视频模式解码，项目模式直接读图。

    返回 (图像, 总帧数)。合成器两种模式都要能检测/分析，所以抽出来共用。
    """
    import cv2
    total = int(_merge_cache.get("frames") or 0)
    if _merge_cache.get("mode") == "project":
        names = _merge_cache.get("names") or []
        srcdir = _merge_cache.get("srcdir") or ""
        if not names:
            raise HTTPException(409, "还没有打开项目")
        idx = max(0, min(int(idx), len(names) - 1))
        img = _imread_any(Path(srcdir) / names[idx])
        if img is None:
            raise HTTPException(500, "读不了这张图: " + names[idx])
        return img, len(names)

    path = _merge_cache.get("path")
    if not path:
        raise HTTPException(409, "还没有打开视频")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(500, "打不开视频")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "取不到第 %d 帧" % idx)
    return frame, total

