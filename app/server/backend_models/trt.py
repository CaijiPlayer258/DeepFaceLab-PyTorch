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


def _resolve_dir(dst: str) -> Path:
    r"""把界面里填的目录解析成绝对路径。

    界面上显示的是「workspace\data_src\aligned」这种带前缀的相对路径。
    以前后端直接 Path(dst) —— 相对的是 serve.py 的当前目录（app/），
    于是帧被写到 app\workspace\...，用户在自己的 workspace 里什么都看不到。
    现在：绝对路径原样用；相对路径去掉开头的 workspace 段后挂到 WORKSPACE 下。
    """
    raw = str(dst or "").strip().strip('"')
    if not raw:
        return WORKSPACE
    q = Path(raw)
    if q.is_absolute():
        return q
    parts = [x for x in raw.replace("/", "\\").split("\\") if x and x != "."]
    if not parts:
        return WORKSPACE
    # workspace\... 仍然指向真实工作区（4.0 那边），保持老行为不改
    if parts[0].lower() == "workspace":
        rest = parts[1:]
        return WORKSPACE.joinpath(*rest) if rest else WORKSPACE
    # 其余相对路径一律相对**本仓库根目录**：
    #   abc\data_src\aligned → D:\MyProject\DeepFaceLab-PyTorch-4.1\abc\data_src\aligned
    return _APP_ROOT.joinpath(*parts)


@router.post("/api/extract/start")
def extract_start(payload: dict | None = None):
    """真抽帧：src = 视频文件，dst = 输出目录（默认 workspace/_frames），fps = 抽帧频率。"""
    payload = payload or {}
    src = (payload.get("src") or "").strip()
    dst = (payload.get("dst") or "").strip()
    fps = float(payload.get("fps") or 2)
    skip = int(payload.get("skip") or 0)      # 每 N 帧取 1；给了就以源帧率为准换算 fps
    rotate = int(payload.get("rotate") or 0)
    quality = str(payload.get("quality") or "high")
    align = bool(payload.get("align", True))
    size = int(payload.get("size") or 0)   # 0 = 不限制；分辨率跟着源图和检测框走
    # 5 点退化闸门：None = 跟 DFL_ALIGN_GATE（默认 warn，只记 _bad_frames.txt）；
    # True = strict（--strict-align，真拒并计进 rejected）；False = 强制 warn。
    _sa = payload.get("strict_align", None)
    strict_align = None if _sa is None else bool(_sa)

    if not src:
        raise HTTPException(400, "缺少输入视频路径")
    p = Path(src)
    if not p.is_file():
        raise HTTPException(404, "视频不存在: " + src)
    face_dir = _resolve_dir(dst) if dst else (WORKSPACE / "_frames")
    face_dir.mkdir(parents=True, exist_ok=True)
    dst = str(face_dir)

    JOB.reset()
    # 起线程前就把 running 置上：前端拿到 ok 立刻订阅 SSE，
    # 若这时还是 idle，收流逻辑会以为任务没起来（见 extract_progress 里的 idle 判定）。
    with JOB.lock:
        JOB.running = True
        JOB.msg = "启动抽帧…"
    threading.Thread(target=_frames_worker,
                     args=(str(p), dst, fps, rotate, align, quality, size, skip,
                           strict_align),
                     daemon=True).start()
    return {"ok": True, "src": str(p), "dst": dst,
            "frames": str(face_dir.parent / (face_dir.name + "_frames")),
            "fps": fps, "skip": skip,
            "rotate": rotate, "align": align, "quality": quality, "size": size,
            "strict_align": strict_align}


@router.post("/api/extract/cancel")
def extract_cancel():
    JOB.running = False
    return {"ok": True}


@router.get("/api/extract/progress")
async def extract_progress():
    async def gen():
        last = None
        idle = 0.0
        while True:
            snap = JOB.snapshot()
            if snap != last:
                yield "data: " + json.dumps(snap, ensure_ascii=False) + "\n\n"
                last = snap
                idle = 0.0
            if not snap["running"] and snap["pct"] >= 100:
                return
            if not snap["running"]:
                # 任务没在跑、进度又没到 100：多半是"点了开始但工作线程还没起来"，
                # 给 1 秒宽限；再没有动静就收流。否则这条 SSE 会一直挂着不返回
                # （实测客户端等 60s 也拿不到响应）。
                idle += 0.08
                if idle > 1.0:
                    return
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="text/event-stream")




# 启动时后台核对一次引擎状态（md5 + SM 对不上会被标成过期/未编译）
threading.Thread(target=_engines_verify_all, daemon=True).start()
