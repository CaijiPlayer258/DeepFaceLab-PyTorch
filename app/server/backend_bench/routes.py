# -*- coding: utf-8 -*-
"""速度测试页：POST /api/perf/bench —— 量解码/遮罩/检测/特征点/编码各自多少 ms。

从 backend_models/trt.py 搬出（逐字搬，行为零变化）。只读：不写文件、不动用户数据。
"""
from __future__ import annotations

from fastapi import APIRouter

from .._common.paths import WORKSPACE

router = APIRouter()

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


