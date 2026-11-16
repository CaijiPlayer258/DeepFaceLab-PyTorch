# -*- coding: utf-8 -*-
"""账号（从 app.py 拆出）：/api/auth*，读写 workspace/auth.json。

**环形依赖是既有事实，不是这次搬家引入的**：本模块 import app 的 WORKSPACE/app，
而 app.py 在模块末尾才 include 本模块的 router。所以只有 `import server.app` 是合法入口，
直接 import 本模块会 ImportError: cannot import name 'router' from partially initialized module
（搬家前逐字相同：实测 `import server.routes_account` 在 HEAD 版本上报的就是这句）。旧路径 app/server/routes_account.py 因此必须是
惰性代理 shim —— 静态 re-export 会在 app 半初始化时就把 router 取走。
"""
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

from ..app import (WORKSPACE, app)      # 环：app.py 在模块末尾才 include 路由，只有 import server.app 是合法入口

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
