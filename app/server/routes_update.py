# -*- coding: utf-8 -*-
"""自动更新路由：检查 / 安装。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel

from . import backend_update as U

router = APIRouter(prefix="/api/update", tags=["update"])

@router.get("/progress")
def update_progress():
    """安装进度（下载 % / 解包 / 覆盖）。前端在 apply 期间轮询这个接口。"""
    return U.progress()


class ApplyBody(BaseModel):
    asset: Optional[str] = None


@router.get("/check")
def check() -> dict:
    return U.check()


@router.get("/version")
def version() -> dict:
    return U.parse_version()          # {version, base, when, human}


@router.post("/apply")
def apply(body: ApplyBody = Body(default=ApplyBody())) -> dict:
    return U.apply(body.asset)
