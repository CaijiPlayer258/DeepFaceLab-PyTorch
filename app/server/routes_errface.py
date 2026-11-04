# -*- coding: utf-8 -*-
"""错脸检测（ErrFaceFilter / xgboost 68 点几何）路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import _errface

router = APIRouter()


@router.get("/api/errface/info")
def errface_info():
    return {
        "ok": True,
        "available": _errface.available(),
        "model": _errface.model_path().name,
    }


@router.post("/api/errface/scan")
def errface_scan(body: dict):
    body = body or {}
    d = str(body.get("dir") or "").strip()
    if not d:
        raise HTTPException(400, "缺少目录")
    try:
        return _errface.scan(d, threshold=float(body.get("threshold") or 0.5),
                             limit=int(body.get("limit") or 0),
                             percent=float(body.get("percent") or 0))
    except Exception as e:
        raise HTTPException(500, str(e)[:300])
