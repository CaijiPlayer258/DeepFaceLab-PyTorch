# -*- coding: utf-8 -*-
"""面部 68 点（从 app.py 拆出）。"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .app import (WORKSPACE)

router = APIRouter()

def _face_path() -> Path:
    return WORKSPACE / "face68.json"


@router.get("/api/face")
def face_get():
    try:
        return {"points": json.loads(_face_path().read_text(encoding="utf-8"))}
    except Exception:
        return {"points": None}


@router.post("/api/face")
def face_save(payload: dict):
    pts = payload.get("points")
    if not isinstance(pts, list) or len(pts) != 68:
        raise HTTPException(status_code=400, detail="需要 68 个点")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    _face_path().write_text(json.dumps(pts, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@router.delete("/api/face")
def face_clear():
    try:
        _face_path().unlink()
    except Exception:
        pass
    return {"ok": True}
