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
    """GET 一次拿到 running/pct/msg/fps + ok/error（ok=False 时 error 必非空）。

    .. deprecated::
        **前端不用这条**，界面轮询的是 GET /api/extract/progress
        （SSE，backend_models/trt.py）。这条只被验收脚本
        app/tools/accept/aligned444_accept.py 用来「拉一次看看」。

        留着不删是因为脚本依赖；但新代码别再调它 —— 两条读的是同一个
        _job_status()，语义却一个是轮询、一个是流，容易踩。
        响应里的 deprecated 字段就是给调用方的一句提醒（行为零改动）。
    """
    st = _job_status()
    if isinstance(st, dict):
        st = dict(st)
        st["deprecated"] = "改用 GET /api/extract/progress（SSE）；这条只给验收脚本"
    return st
