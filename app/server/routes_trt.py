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

from .app import (JOB, WEB, WORKSPACE, _APP_ROOT, _NO_WINDOW, _frames_worker, _mask_worker, _get_xseg, app)

router = APIRouter()

# ---------------- TensorRT 引擎（编译/状态） ----------------
_engine_job = {"running": False, "name": "", "log": [], "error": "", "done": False}


def _engine_roots():
    """能找 onnx 的根目录：工作区优先，其次引擎仓。"""
    return [WORKSPACE, Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3] / "DeepFaceLab-PyTorch"]


def _engines_verify_all():
    """启动时在后台把引擎状态核一遍（md5 + SM），不阻塞启动。"""
    try:
        from ._engines import status as _st
        _st(WORKSPACE, _engine_roots())
    except Exception:
        pass


@router.get("/api/devices")
def api_devices():
    """可用训练设备：GPU 的真实名字 + 显存。

    界面上的"设备"下拉以前只有 0 / 1 / 0,1 这种序号 —— 用户得自己开命令行
    跑 nvidia-smi 才知道 0 是哪张卡。这里直接把型号返给前端。
    """
    gpus = []
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=6,
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        for line in (r.stdout or "").strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                gpus.append({"idx": int(parts[0]), "name": parts[1],
                             "mem_mb": int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0})
    except Exception:
        pass
    cpu = os.environ.get("PROCESSOR_IDENTIFIER") or "CPU"
    try:
        cpu = (os.cpu_count() or 0) and ("CPU (%d 线程)" % (os.cpu_count() or 0)) or "CPU"
    except Exception:
        pass
    return {"gpus": gpus, "cpu": cpu}


@router.get("/api/build")
def api_build():
    """界面构建时间：拿 app.js / app.css / index.html 的最新修改时间。

    用途是把"你看到的到底是哪一版"变成可核对的事实 —— 之前排查样式问题时
    反复在猜客户端有没有重载。页面右下角会显示这个时间。
    """
    files = [WEB / "index.html", WEB / "js" / "app.js", WEB / "css" / "app.css"]
    best = 0.0
    names = {}
    for f in files:
        try:
            m = f.stat().st_mtime
        except Exception:
            continue
        names[f.name] = int(m)
        best = max(best, m)
    return {"ts": int(best),
            "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(best)) if best else "",
            "files": names}


@router.get("/api/engines")
def engines_status():
    from ._engines import status as _st
    st = _st(WORKSPACE, _engine_roots())
    st["job"] = dict(_engine_job)
    return st


def _engine_build_worker(name: str):
    import subprocess
    import sys
    from ._engines import status as _st, engine_path as _ep
    from ._engines import MODEL_SPECS

    st = _st(WORKSPACE, _engine_roots())
    row = next((m for m in st["models"] if m["name"] == name), None)
    if not row or not row.get("onnx"):
        with _engine_job_lock:
            _engine_job.update({"running": False, "error": "找不到模型 " + name, "done": True})
        return
    eng = _ep(WORKSPACE, name, row["onnx"], st["sm"])
    if eng is None:
        with _engine_job_lock:
            _engine_job.update({"running": False, "error": "没有可用的 GPU", "done": True})
        return

    from ._engines import PROFILES
    cmd = [sys.executable, str(Path(__file__).parent / "_trt_build.py"),
           "--onnx", row["onnx"], "--out", str(eng), "--name", name, "--bf16"]
    prof = PROFILES.get(name)
    if prof:
        cmd += ["--profile", ";".join(",".join(str(v) for v in p) for p in prof)]
    with _engine_job_lock:
        _engine_job.update({"running": True, "name": name, "log": ["开始编译 " + name],
                            "error": "", "done": False, "pct": 0})
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, creationflags=_NO_WINDOW)
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("@@P "):
                with _engine_job_lock:
                    try:
                        pct = int(line.split()[1])
                        _engine_job["pct"] = pct
                    except Exception:
                        pass
                    _engine_job["log"].append(line[4:])
                    _engine_job["log"] = _engine_job["log"][-40:]
            elif line.startswith("@@OK "):
                with _engine_job_lock:
                    _engine_job["log"].append("完成: " + Path(line[5:]).name)
            elif line.startswith("@@ERR "):
                with _engine_job_lock:
                    _engine_job["error"] = line[6:]
                    _engine_job["log"].append("失败: " + line[6:])
        proc.wait()
    except Exception as e:
        with _engine_job_lock:
            _engine_job["error"] = "%s: %s" % (type(e).__name__, e)
    with _engine_job_lock:
        _engine_job["running"] = False
        _engine_job["done"] = True


_engine_job_lock = threading.Lock()


@router.post("/api/engines/build")
def engines_build(payload: dict | None = None):
    payload = payload or {}
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少 name")
    with _engine_job_lock:
        if _engine_job.get("running"):
            raise HTTPException(409, "已有编译任务在跑：" + str(_engine_job.get("name")))
        _engine_job.update({"running": True, "name": name, "log": [], "error": "", "done": False, "pct": 0})
    threading.Thread(target=_engine_build_worker, args=(name,), daemon=True).start()
    return {"ok": True, "name": name}


def _modelhub_dir():
    from ._paths import modelhub_dir
    return modelhub_dir()


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
    from .app import _imread_any
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
        from ._detect import TinyMogDet
        onnx = _modelhub_dir() / "onnx" / "TinyMog" / "TinyMog.onnx"
        if onnx.is_file():
            det = TinyMogDet(str(onnx), workspace=WORKSPACE)
            canvas = cv2.resize(imgs[0], (640, 640))
            rows.append({"k": "检测（TinyMog）", "ms": timed(lambda: det.detect(canvas), 10)})
    except Exception as e:
        rows.append({"k": "检测（TinyMog）", "ms": None, "note": str(e)[:60]})

    # 4) 特征点
    try:
        from ._landmarks106 import InsightFace106
        onnx = _modelhub_dir() / "onnx" / "InsightFace2d106" / "InsightFace2D106.onnx"
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
                     args=(str(p), dst, fps, rotate, align, quality, size, skip),
                     daemon=True).start()
    return {"ok": True, "src": str(p), "dst": dst,
            "frames": str(face_dir.parent / (face_dir.name + "_frames")),
            "fps": fps, "skip": skip,
            "rotate": rotate, "align": align, "quality": quality, "size": size}


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
