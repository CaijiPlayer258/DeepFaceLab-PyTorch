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
    """读 68 点。成功 {ok, points}；**没有就是 404 + detail**。

    以前这里是「200 + {points: null}」—— 全站唯一一个用空数据表达失败的接口，
    前端只能额外判 d.points == null（app.js:8990 那处特判）。失败也是失败，
    给 404，让 r.ok 成为唯一判据。
    """
    try:
        return {"ok": True, "points": json.loads(_face_path().read_text(encoding="utf-8"))}
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail="这一帧没有关键点（还没保存过 face68.json）")
    except Exception as e:                                      # noqa: BLE001
        raise HTTPException(status_code=404,
                            detail="这一帧没有关键点（face68.json 读不出来：%s）" % e)


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
