# -*- coding: utf-8 -*-
"""模型转换（.npy -> .pth）路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import _convert

router = APIRouter()


@router.get("/api/convert/list")
def convert_list():
    return {"ok": True, "items": _convert.scan()}


@router.post("/api/convert/start")
def convert_start(body: dict):
    body = body or {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少模型名")
    try:
        return _convert.start(name)
    except FileNotFoundError as e:
        raise HTTPException(404, "找不到源文件或转换脚本：%s" % str(e)[:200])
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/convert/status")
def convert_status():
    return _convert.status()
