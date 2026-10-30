# -*- coding: utf-8 -*-
"""遮罩编辑 / 工程（从 app.py 按路径前缀抽出）。"""
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

from .app import (IMAGE_EXT, WORKSPACE, _mask_ai_stub, _mask_state)

router = APIRouter()

@router.post("/api/project/open")
def mask_project_open(body: dict):
    """打开一个已对齐的人脸集目录，列出图片供遮罩编辑器使用。"""
    raw = (body or {}).get("path") or ""
    rel = raw
    low = rel.lower().replace("/", "\\")
    for pre in ("workspace\\", "workspace"):
        if low.startswith(pre):
            rel = rel[len(pre):].lstrip("\\/")
            break
    d = Path(raw) if (raw and Path(raw).is_absolute()) else (WORKSPACE / rel if rel else WORKSPACE / "data_src" / "aligned")
    if not d.is_dir():
        raise HTTPException(404, "目录不存在: " + str(d))
    files = sorted([p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT])
    resume = 0
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            memo = json.loads(fp.read_text(encoding="utf-8"))
            if memo.get("dir") and Path(memo["dir"]) == d:
                resume = int(memo.get("index") or 0)
        except Exception:
            pass
    _mask_state.update({"dir": str(d), "files": [p.name for p in files], "index": resume})
    return {"path": str(d), "count": len(files), "resume": resume,
            "files": [{"name": p.name, "size": p.stat().st_size} for p in files[:500]]}


@router.get("/api/image/{index}")
def mask_image(index: int):
    """返回当前项目的第 index 张图（base64），以及是否已有遮罩。"""
    import base64
    files = _mask_state.get("files") or []
    if not files:
        raise HTTPException(409, "还没有打开项目")
    index = max(0, min(int(index), len(files) - 1))
    d = Path(_mask_state["dir"])
    p = d / files[index]
    if not p.is_file():
        raise HTTPException(404, "文件不存在: " + files[index])
    raw = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    _mask_state["index"] = index
    return {"index": index, "name": files[index], "total": len(files),
            "image": base64.b64encode(raw).decode(),   # 纯 base64，前端自己拼 data URL
            "mime": mime, "has_mask": True}


@router.post("/api/mask/predict")
def mask_predict(body: dict):
    _mask_ai_stub()


@router.post("/api/mask/text")
def mask_text(body: dict):
    _mask_ai_stub()


@router.post("/api/mask/bisenet")
def mask_bisenet(body: dict):
    _mask_ai_stub()


@router.post("/api/model/preload")
def mask_preload(body: dict | None = None):
    return {"ok": True, "note": "模型预加载未接"}


@router.get("/api/image/{index}/mask")
def mask_existing(index: int):
    """读这张图已有的 XSeg 遮罩。DFLJPG 里的遮罩段暂时不解析，返回空当新图处理。"""
    return {"mask": None}


@router.post("/api/progress/save")
def mask_progress_save(body: dict | None = None):
    """记住看到第几张，下次进来接着干。落盘，重启也还在。"""
    body = body or {}
    idx = int(body.get("index") or 0)
    _mask_state["index"] = idx
    try:
        (WORKSPACE / "_mask_progress.json").write_text(
            json.dumps({"index": idx, "dir": _mask_state.get("dir") or ""}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass
    return {"ok": True, "index": idx}


@router.get("/api/progress/load")
def mask_progress_load():
    """读上次的进度；文件里还有上次打开的目录，前端可以据此恢复。"""
    idx = _mask_state.get("index") or 0
    d = ""
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            idx = int(data.get("index") or idx)
            d = data.get("dir") or ""
        except Exception:
            pass
    return {"index": idx, "dir": d}


@router.post("/api/mask/save")
def mask_save(body: dict):
    """把画好的遮罩存成 PNG，写到 _masks/<同名>.png。"""
    import base64
    files = _mask_state.get("files") or []
    if not files:
        raise HTTPException(409, "还没有打开项目")
    idx = int((body or {}).get("index", _mask_state.get("index") or 0))
    data = (body or {}).get("mask") or ""
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception:
        raise HTTPException(400, "遮罩数据不是合法 base64")
    d = Path(_mask_state["dir"]) / "_masks"
    d.mkdir(exist_ok=True)
    out = d / (Path(files[idx]).stem + ".png")
    out.write_bytes(raw)
    return {"ok": True, "path": str(out)}


@router.get("/api/mask/models")
def mask_models(kind: str = "XSegLite"):
    """扫描 workspace/model/<kind>/ 下的 .onnx / .engine，供遮罩绘制页的模型下拉用。
    4.0 的 _refresh_xseg_file_list() 就是这个逻辑：优先 .engine（TRT 最快）。"""
    base = WORKSPACE / "model" / (kind if kind in ("XSeg", "XSegLite") else "XSegLite")
    files = []
    if base.is_dir():
        for f in sorted(base.iterdir()):
            if f.is_file() and f.suffix.lower() in (".onnx", ".engine"):
                files.append({"name": f.name, "path": str(f),
                              "mb": round(f.stat().st_size / 1048576, 1),
                              "trt": f.suffix.lower() == ".engine"})
    files.sort(key=lambda x: (not x["trt"], x["name"]))
    return {"kind": kind, "dir": str(base), "files": files}
