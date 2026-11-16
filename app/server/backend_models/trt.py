# -*- coding: utf-8 -*-
"""TensorRT 引擎（从 app.py 拆出）。"""
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

from ..app import (JOB, WEB, WORKSPACE, _APP_ROOT, _NO_WINDOW, _frames_worker, _mask_worker, _get_xseg, app)

router = APIRouter()

# ---------------- TensorRT 引擎（编译/状态）：整块搬到 backend_engines/routes.py ----------------
# 这里保留 import：本模块末尾那根校验线程、以及别处对 _engine_roots 的引用继续可用。
from ..backend_engines.routes import (  # noqa: E402,F401
    _engine_build_worker, _engine_job, _engine_job_lock, _engine_roots, _engines_verify_all,
)

def _weights_dir():
    """权重根（平铺布局：onnx 直接在根下，没有模型子目录）。"""
    from .._paths import weights_dir
    return weights_dir()


@router.post("/api/perf/bench")
def perf_bench(payload: dict | None = None):
    """跑一遍各环节的实测耗时（推理/解码/编码），给「速度测试」页用。

    只读：不写任何文件，不动用户数据。
    """
    import time
    import numpy as np
    import cv2

    rows = []
    root = WORKSPACE / "data_src" / "aligned"
    files = sorted(root.glob("*.jpg"))[:8] if root.is_dir() else []
    if not files:
        return {"ok": False, "detail": "workspace/data_src/aligned 里没有图片，无法测速"}
    # DFLJPG 尾部带 APP15 段，直接 imdecode 会失败，必须走 _imread_any
    from ..app import _imread_any
    imgs = []
    for f in files:
        im = _imread_any(f)
        if im is not None:
            imgs.append(im)
    if not imgs:
        return {"ok": False, "detail": "图片都读不出来"}

    def timed(fn, n=10):
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    # 1) 解码
    fp = files[0]
    rows.append({"k": "解码（DFLJPG）", "ms": timed(lambda: _imread_any(fp), 20)})

    # 2) 遮罩
    xseg = _get_xseg(None)
    if xseg is not None:
        base = imgs[0]
        rows.append({"k": "遮罩推理（%s）" % type(xseg).__name__, "ms": timed(lambda: xseg.mask(base, out_size=256), 10)})

    # 3) 检测
    try:
        from .._detect import TinyMogDet
        onnx = _weights_dir() / "tinymog.onnx"
        if onnx.is_file():
            det = TinyMogDet(str(onnx), workspace=WORKSPACE)
            canvas = cv2.resize(imgs[0], (640, 640))
            rows.append({"k": "检测（TinyMog）", "ms": timed(lambda: det.detect(canvas), 10)})
    except Exception as e:
        rows.append({"k": "检测（TinyMog）", "ms": None, "note": str(e)[:60]})

    # 4) 特征点
    try:
        from .._landmarks106 import InsightFace106
        onnx = _weights_dir() / "insightface_2d106.onnx"
        if onnx.is_file():
            lm = InsightFace106(str(onnx), workspace=WORKSPACE)
            face = cv2.resize(imgs[0], (192, 192))
            rows.append({"k": "特征点（2D106）", "ms": timed(lambda: lm.landmarks106(face), 10)})
    except Exception as e:
        rows.append({"k": "特征点（2D106）", "ms": None, "note": str(e)[:60]})

    # 5) 编码
    face = cv2.resize(imgs[0], (256, 256))
    rows.append({"k": "JPEG 编码", "ms": timed(lambda: cv2.imencode(".jpg", face, [int(cv2.IMWRITE_JPEG_QUALITY), 95]), 20)})

    return {"ok": True, "rows": rows}


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
threading.Thread(target=_engines_verify_all, daemon=True).start()
