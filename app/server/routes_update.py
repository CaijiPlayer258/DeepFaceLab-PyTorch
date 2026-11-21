# -*- coding: utf-8 -*-
"""自动更新路由：检查 / 安装。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel

from . import backend_update as U

router = APIRouter(prefix="/api/update", tags=["update"])


class ApplyBody(BaseModel):
    asset: Optional[str] = None


@router.get("/check")
def check() -> dict:
    return U.check()


@router.get("/version")
def version() -> dict:
    return {"version": U.local_version()}


@router.post("/apply")
def apply(body: ApplyBody = Body(default=ApplyBody())) -> dict:
    return U.apply(body.asset)
