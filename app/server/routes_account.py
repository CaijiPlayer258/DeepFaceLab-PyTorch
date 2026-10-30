# -*- coding: utf-8 -*-
"""账号（从 app.py 拆出）。"""
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

from .app import (WORKSPACE, app)

router = APIRouter()

# ---------------- 账号（B 站） ----------------
AUTH_FILE = None      # 运行时按工作区解析


def _auth_path() -> Path:
    global AUTH_FILE
    if AUTH_FILE is None:
        AUTH_FILE = WORKSPACE / "auth.json"
    return AUTH_FILE


def _auth_read() -> dict:
    try:
        return json.loads(_auth_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.get("/api/auth")
def auth_state():
    a = _auth_read()
    return {
        "logged_in": bool(a.get("uid")),
        "uid": a.get("uid", 0),
        "name": a.get("name", ""),
        "avatar": a.get("avatar", ""),
        "since": a.get("since", 0),
        "onboarded": bool(a.get("onboarded")),
    }


@router.post("/api/auth/login")
def auth_login(payload: dict):
    """真正的扫码流程接上之前，先用它写入一次登录态。"""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    data = {
        "uid": int(payload.get("uid") or 0),
        "name": payload.get("name") or "",
        "avatar": payload.get("avatar") or "",
        "since": int(__import__("time").time()),
        "onboarded": True,
    }
    _auth_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **data}


@router.post("/api/auth/skip")
def auth_skip():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    a = _auth_read()
    a["onboarded"] = True
    _auth_path().write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@router.post("/api/auth/logout")
def auth_logout():
    try:
        _auth_path().unlink()
    except Exception:
        pass
    return {"ok": True}
