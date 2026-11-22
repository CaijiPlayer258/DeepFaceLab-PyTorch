# -*- coding: utf-8 -*-
"""数据处理的 HTTP 薄层：/api/state（版本 / 运行时 / 端口 / 数据集 / 模型 / 能力）。

路由函数体原样从 app/server/app.py 搬过来（只把 @app.get 换成 @router.get）。
"""
from __future__ import annotations

import os

import threading

from fastapi import APIRouter, HTTPException

from .._common.jobs import JOB

from .stats import (_capabilities, _dataset, _models, _ports, _ports_status,
                    _runtime, _version)
from .maskops import _mask_worker

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
        # 预览模式（serve.py --preview）：给设计工作台那类独立浏览器会话用，
        # 前端据此跳过协议门、把登录态当已登录，好让画布直接看到真实界面。
        "preview": os.environ.get("DFL_PREVIEW") == "1",
    }


@router.get("/api/dataset/mask/progress")
def dataset_mask_progress():
    """应用遮罩的进度：直接取 JOB.snapshot()（pct/done/total/rate/eta/msg）。"""
    snap = JOB.snapshot()
    return {"running": snap["running"], "pct": int(snap["pct"] or 0),
            "msg": snap["msg"], "done": snap["done"], "total": snap["total"],
            "speed": snap["rate"], "eta": snap["eta"], "stage": snap["stage"]}


@router.post("/api/dataset/mask")
def dataset_mask(payload: dict | None = None):
    """POST {dir: 'data_src/aligned', workers: 8} —— 给整个目录补 XSeg 遮罩。"""
    payload = payload or {}
    # 路径按常规解析：绝对路径原样用，相对路径相对软件根目录（.）；
    # 不做额外的前缀限制，也不做任何自动拼接
    raw_dir = str(payload.get("dir") or "data_src/aligned")
    p = Path(raw_dir)
    root = (p if p.is_absolute() else (_APP_ROOT / p)).resolve()
    if not root.is_dir():
        raise HTTPException(404, "目录不存在: %s" % rel)
    workers = int(payload.get("workers") or 8)
    # 遮罩类型：lite / xseg（对应 XSegLite / XSeg 两个模型）；res 是写入 DFLJPG 的遮罩尺寸
    model = str(payload.get("model") or "lite").lower()
    if model not in ("lite", "xseg"):
        raise HTTPException(400, "model 只能是 lite 或 xseg")
    res = int(payload.get("res") or 256)
    if res not in (256, 512):
        raise HTTPException(400, "res 只能是 256 或 512")
    with JOB.lock:
        JOB.running = True
        JOB.msg = "启动应用遮罩…"
    threading.Thread(target=_mask_worker, args=(str(root), workers, model, res), daemon=True).start()
    return {"ok": True, "dir": str(root), "workers": workers, "model": model, "res": res}






# 启动时后台核对一次引擎状态（md5 + SM 对不上会被标成过期/未编译）
