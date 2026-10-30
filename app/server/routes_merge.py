# -*- coding: utf-8 -*-
"""合成器路由（从 app.py 按路径前缀抽出）。"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import subprocess as _sp
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse

from . import app as _app_mod
from .app import (WORKSPACE, _dfl_project_scan, _dfljpg_dict, _get_detector, _imread_any, _merge_cache, _merge_frame_image, _merge_job, _merge_job_lock, _merge_prep, _merge_prep_lock, _merge_prepare_worker, _merge_worker, _safe_dirname, _video_frame_jpeg, _video_probe)
import base64 as _b64

router = APIRouter()

@router.get("/api/merge/videos")
def merge_videos(dir: str = ""):
    """列出视频文件。dir 给了就扫它，否则扫工作区。"""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    roots = [Path(dir)] if dir else []
    roots.append(WORKSPACE)
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            # 合成产物 / 缓存目录不作为素材列出来（否则列表里全是自己的输出）
            if f.name.endswith("_merged.mp4") or ".merge_" in str(f.parent):
                continue
            out.append({"path": str(f), "name": f.name,
                        "mb": round(f.stat().st_size / 1048576, 1)})
            if len(out) >= 200:
                break
    return {"dir": str(roots[0]), "videos": out}


@router.post("/api/merge/prepare")
def merge_prepare(body: dict):
    """视频 → 可合成项目（抽帧 + 对齐）。跑在后台，进度用 /status 拉。"""
    body = body or {}
    raw = str(body.get("video") or "").strip()
    if not raw:
        raise HTTPException(400, "缺少 video")
    video = Path(raw)
    if not video.is_file():
        raise HTTPException(404, "视频不存在: " + raw)

    with _merge_prep_lock:
        if _merge_prep["running"]:
            raise HTTPException(409, "正在准备另一个视频")
        _merge_prep.update({"running": True, "ok": None, "error": "", "pct": 0,
                            "msg": "启动…", "stage": "准备", "frames": 0,
                            "aligned": 0, "video": str(video)})

    target = str(body.get("dir") or "").strip()
    project = Path(target) if target else (WORKSPACE / "merge" / _safe_dirname(video.stem))
    with _merge_prep_lock:
        _merge_prep["project"] = str(project)

    threading.Thread(target=_merge_prepare_worker, args=(video, project),
                     daemon=True).start()
    return {"ok": True, "project": str(project), "video": str(video)}


@router.get("/api/merge/prepare/status")
def merge_prepare_status():
    with _merge_prep_lock:
        return dict(_merge_prep)


@router.post("/api/merge/open")
def merge_open(body: dict):
    raw = (body or {}).get("path") or ""
    p = Path(raw)

    # 目录 = DFL 项目（data_dst / aligned / 裸帧目录）
    if p.is_dir():
        names, srcdir, kind = _dfl_project_scan(p)
        if not names:
            raise HTTPException(404, "目录里没有可用的图: " + raw)
        _merge_cache.update({
            "mode": "project", "dir": str(p), "srcdir": srcdir, "names": names,
            "path": "", "frames": len(names), "fps": 0, "w": 0, "h": 0,
        })
        sample = names[:24]
        lm = sum(1 for n in sample
                 if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None)
        return {"path": str(p), "name": p.name, "mode": "project", "kind": kind,
                "frames": len(names), "fps": 0, "width": 0, "height": 0,
                "with_landmarks": lm, "sampled": len(sample)}

    if not p.is_file():
        raise HTTPException(404, "路径不存在: " + raw)
    try:
        frames, fps, w, h = _video_probe(p)
    except Exception as e:
        raise HTTPException(500, "无法解析视频: " + str(e))
    _merge_cache.update({"mode": "video", "path": str(p), "dir": "", "list": [],
                         "frames": frames, "fps": fps, "w": w, "h": h})
    return {"path": str(p), "name": p.name, "mode": "video", "frames": frames,
            "fps": fps, "width": w, "height": h}


@router.get("/api/merge/frame/{idx}")
def merge_frame(idx: int, w: int = 900):
    # 项目模式：直接从列表里取那张图
    if _merge_cache.get("mode") == "project":
        names = _merge_cache.get("names") or []
        srcdir = _merge_cache.get("srcdir") or ""
        if not names:
            raise HTTPException(409, "还没有打开项目")
        idx = max(0, min(idx, len(names) - 1))
        fp = Path(srcdir) / names[idx]
        if not fp.is_file():
            raise HTTPException(404, "图片不见了: " + str(fp))
        try:
            import cv2
            img = _imread_any(fp)
            if img is None:
                raise RuntimeError("解不开")
            h0, w0 = img.shape[:2]
            maxw = max(200, min(w, 1600))
            if w0 > maxw:
                img = cv2.resize(img, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                raise RuntimeError("编码失败")
        except Exception as e:
            raise HTTPException(500, "取图失败: " + str(e))
        return {"idx": idx, "total": len(names), "name": names[idx],
                "png": "data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode()}

    path = _merge_cache.get("path")
    if not path:
        raise HTTPException(409, "还没有打开视频")
    total = _merge_cache.get("frames") or 0
    if total:
        idx = max(0, min(idx, total - 1))
    try:
        png = _video_frame_jpeg(path, idx, maxw=max(200, min(w, 1600)))
    except Exception as e:
        raise HTTPException(500, "取帧失败: " + str(e))
    return {"idx": idx, "png": "data:image/jpeg;base64," + _b64.b64encode(png).decode()}


@router.post("/api/merge/export")
def merge_export(body: dict):
    """把一段帧目录合成成换脸结果。真要跑，不是占位。"""
    body = body or {}
    # project 和 input_dir 是同一样东西：合成器里叫「项目」，接口里叫「输入帧目录」
    input_dir = str(body.get("input_dir") or body.get("project") or "").strip()
    if not input_dir:
        raise HTTPException(400, "缺少 input_dir（或 project）")
    ip = Path(input_dir)
    if not ip.is_dir():
        raise HTTPException(404, "输入目录不存在: " + input_dir)

    dfm = str(body.get("dfm") or "").strip()
    model_name = str(body.get("model") or "").strip()
    if not dfm and model_name:
        # 界面传的是模型名（384wf / DFLitetest …），这里落成 .dfm 路径
        cand = WORKSPACE / "model" / (model_name + ".dfm")
        if cand.is_file():
            dfm = str(cand)
    if not dfm:
        # 自动挑第一个 .dfm
        cands = sorted((WORKSPACE / "model").glob("*.dfm"))
        dfm = str(cands[0]) if cands else ""
    if not dfm or not Path(dfm).is_file():
        who = ("模型 %s 还没有导出 .dfm —— 去训练页点「导出」" % model_name) if model_name else "没有可用的 .dfm 模型"
        raise HTTPException(400, who)

    out_raw = str(body.get("output_dir") or "")
    aligned_dir = str(body.get("aligned_dir") or "")

    # 对齐集：显式传入 > 输入目录下的 aligned/ > 同级的 <name>_aligned/
    # 合成靠 DFLJPG 里的元数据定位人脸，没有对齐集就只能瞎猜，所以这里必须定死
    if not aligned_dir:
        cands = [ip / "aligned", ip.parent / (ip.name + "_aligned")]
        for c in cands:
            if c.is_dir():
                aligned_dir = str(c)
                break
    if not aligned_dir or not Path(aligned_dir).is_dir():
        raise HTTPException(400,
            "找不到对齐集（DFLJPG）：期望 %s/aligned 或 %s_aligned。"
            "先跑一次「抽帧 + 人脸对齐」。" % (ip, ip))
    if not list(Path(aligned_dir).glob("*.jpg")):
        raise HTTPException(400, "对齐集是空的: " + aligned_dir)

    if out_raw.lower().endswith(".mp4"):
        out_mp4 = out_raw
    elif out_raw:
        out_mp4 = str(Path(out_raw) / "merged.mp4")
    else:
        out_mp4 = str(ip.parent / (ip.name + "_merged.mp4"))

    mask_onnx = str(body.get("xseg_onnx") or "")
    if not mask_onnx:
        for c in (WORKSPACE / "model" / "XSegLite" / "xseglite.onnx",
                  WORKSPACE / "model" / "XSeg" / "XSeg.onnx"):
            if c.is_file():
                mask_onnx = str(c)
                break
    settings = body.get("settings") or {}

    with _merge_job_lock:
        if _merge_job["running"]:
            raise HTTPException(409, "已经在合成")
        # 先把状态置成 running 再起线程：否则前端刚拿到 ok=True 就来轮询，
        # 会读到线程启动前的旧状态（running=False / ok=None），当成"已完成"。
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "out": out_mp4, "pct": 0, "msg": "启动合成…"})

    try:
        fps = float(body.get("fps") or _merge_cache.get("fps") or 30.0)
    except Exception:
        fps = 30.0
    try:
        workers = int(body.get("workers") or 0)
    except Exception:
        workers = 0
    threading.Thread(target=_merge_worker,
                args=(input_dir, out_mp4, aligned_dir, dfm, settings, mask_onnx, fps, workers,
                      str(body.get("encoder") or "h264_nvenc"),
                      str(body.get("hwaccel") if body.get("hwaccel") is not None else "auto")),
                daemon=True).start()
    return {"ok": True, "input_dir": input_dir, "aligned_dir": aligned_dir,
            "output": out_mp4, "output_dir": out_mp4, "dfm": dfm, "settings": settings}


@router.get("/api/merge/export/status")
def merge_export_status():
    with _merge_job_lock:
        return {"running": _merge_job["running"], "ok": _merge_job["ok"],
                "error": _merge_job["error"], "out": _merge_job["out"],
                "pct": _merge_job.get("pct", 0), "msg": _merge_job.get("msg", ""),
                "stage": _merge_job.get("stage", -1),
                "cancelled": _merge_job.get("cancelled", False),
                "log": _merge_job["log"][-50:]}


@router.get("/api/merge/merged/{idx}")
def merge_merged_frame(idx: int, w: int = 900):
    """从「最近一次导出的成片」里取一帧，给工作台的「合成」标签用。"""
    import cv2
    out = (_merge_job.get("out") or "").strip()
    if not out or not Path(out).is_file():
        # 重启之后 _merge_job 是空的 —— 按当前项目的默认命名找一遍
        for cand in (str(_merge_cache.get("dir") or "") + "_merged.mp4",
                     str(_merge_cache.get("dir") or "") + os.sep + "merged.mp4"):
            if cand and Path(cand).is_file():
                out = cand
                break
    if not out or not Path(out).is_file():
        raise HTTPException(409, "还没有导出的成片")
    cap = cv2.VideoCapture(out)
    if not cap.isOpened():
        raise HTTPException(500, "打不开成片: " + out)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "成片里没有第 %d 帧" % idx)
    h0, w0 = frame.shape[:2]
    maxw = max(200, min(int(w), 1600))
    if w0 > maxw:
        frame = cv2.resize(frame, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
    ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok2:
        raise HTTPException(500, "编码失败")
    return {"idx": int(idx), "out": out,
            "png": "data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode()}


@router.post("/api/merge/export/cancel")
def merge_export_cancel():
    """终止正在跑的合成子进程。"""
    with _merge_job_lock:
        running = _merge_job["running"]
    if not running or _app_mod._merge_proc is None:
        return {"ok": False, "reason": "no running export"}
    try:
        _app_mod._merge_proc.terminate()
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}
    with _merge_job_lock:
        _merge_job["cancelled"] = True
        _merge_job["msg"] = "cancelling"
    return {"ok": True}


@router.get("/api/merge/detect/{idx}")
def merge_detect(idx: int, model: str = "det_500m", thresh: float = 0.5):
    """对当前帧做人脸检测，返回原图坐标的框。视频 / 项目两种模式都支持。"""
    frame, _total = _merge_frame_image(idx)
    det = _get_detector(model)
    det.thresh = float(thresh)
    boxes = det.detect(frame)
    h, w = frame.shape[:2]
    return {"idx": int(idx), "width": int(w), "height": int(h),
            "model": model, "boxes": boxes}


@router.get("/api/merge/analyze")
def merge_analyze(step: int = 12, model: str = "det_500m", thresh: float = 0.5):
    """按步长扫全片，统计每帧的人脸数，给时间轴的人脸轨用。
    486 帧 / step=12 → 41 次推理，约 20 秒。"""
    total = int(_merge_cache.get("frames") or 0)
    step = max(1, min(int(step), 200))
    det = _get_detector(model)
    det.thresh = float(thresh)

    marks = []
    idx = 0
    scanned = 0
    while idx < total:
        try:
            frame, _n = _merge_frame_image(idx)
        except HTTPException:
            break
        boxes = det.detect(frame)
        marks.append({"idx": idx, "count": len(boxes),
                      "box": boxes[0]["box"] if boxes else None})
        scanned += 1
        idx += step
    return {"total": total, "step": step, "scanned": scanned, "marks": marks}


@router.post("/api/merge/project")
def merge_project(body: dict):
    """看一眼目录是不是能用的 DFL 项目：有几张帧、有没有 aligned、带不带 landmarks。"""
    raw = str((body or {}).get("path") or "").strip()
    p = Path(raw)
    if not p.is_dir():
        raise HTTPException(404, "不是目录: " + raw)
    names, srcdir, kind = _dfl_project_scan(p)
    aligned = p / "aligned"
    sample = names[:24]
    lm = sum(1 for n in sample
             if _dfljpg_dict(Path(srcdir) / n).get("landmarks") is not None)
    return {
        "path": str(p), "kind": kind, "total": len(names),
        "has_aligned": aligned.is_dir(),
        "with_landmarks": lm,          # 采样 24 张的结果，不是全量
        "sampled": len(sample),
        "sample": names[:6],
    }


@router.get("/api/merge/info")
def merge_info():
    return dict(_merge_cache)
