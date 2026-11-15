# -*- coding: utf-8 -*-
"""数据处理的 HTTP 薄层：/api/state（版本 / 运行时 / 端口 / 数据集 / 模型 / 能力）。

路由函数体原样从 app/server/app.py 搬过来（只把 @app.get 换成 @router.get）。
"""
from __future__ import annotations

from fastapi import APIRouter

from .stats import (_capabilities, _dataset, _models, _ports, _ports_status,
                    _runtime, _version)

router = APIRouter()


@router.get("/api/state")
def state():
    return {
        "version": _version(),
        "runtime": _runtime(),
        "ports_status": _ports_status(),
        "datasets": {"src": _dataset("data_src"), "dst": _dataset("data_dst")},
        "models": _models(),
        "ports": _ports(),
        "capabilities": _capabilities(),
    }
