# -*- coding: utf-8 -*-
"""提取器的 HTTP 薄层：进度轮询 /api/extract/status。

路由函数体原样从 app/server/app.py 搬过来（只把 @app.get 换成 @router.get），
app.py 那边改成 include_router(_extract_router)。
"""
from __future__ import annotations

from fastapi import APIRouter

from .._common.jobs import _job_status

router = APIRouter()


@router.get("/api/extract/status")
def extract_status():
    """GET 一次拿到 running/pct/msg/fps + ok/error（ok=False 时 error 必非空）。"""
    return _job_status()
