# -*- coding: utf-8 -*-
"""提取器的 HTTP 层：起跑 / 取消 / 进度流（SSE）/ 一次性状态。

路由函数体原样从 app/server/app.py 搬过来（只把 @app.get 换成 @router.get），
app.py 那边改成 include_router(_extract_router)。
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .._common.jobs import JOB, _job_status
from .._common.paths import WORKSPACE, _APP_ROOT
from .frames import _frames_worker

router = APIRouter()


@router.get("/api/extract/status")
def extract_status():
    """GET 一次拿到 running/pct/msg/fps + ok/error（ok=False 时 error 必非空）。

    .. deprecated::
        **前端不用这条**，界面轮询的是 GET /api/extract/progress
        （SSE，本文件下面那条）。这条只被验收脚本
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
    # 资源管理器「复制为路径」给的是 "D:\..." —— 带引号。以前只 strip 空白，
    # 于是 Path 拿到的字面量里含引号，exists() 永远 False，报「路径不存在」/「视频不存在」。
    # 这里统一去掉首尾引号（半角/全角都算），空白在引号内外都清。
    src = (payload.get("src") or "").strip().strip('\'"“”').strip()
    dst = (payload.get("dst") or "").strip().strip('\'"“”').strip()
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
    # 输入框以前提示「源视频文件或源图片目录」，但这里只吃视频 —— 拖一个切帧输出目录进来
    # 会得到「视频不存在」，看着像路径写错，其实是选错了东西。分开报，并说清楚该去哪。
    # 输入可以是**视频文件**，也可以是**图片目录 / DFL 项目目录**。
    # align_stream 里两条分支都在（视频走 ffmpeg，目录逐张读），这条校验别再把目录挡掉。
    if not p.exists():
        raise HTTPException(404, "路径不存在：%s" % p)
    if p.is_dir():
        _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        try:
            _n = sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in _exts)
            _has_aligned = (p / "aligned").is_dir()
        except OSError:
            _n, _has_aligned = 0, False
        if not _n and not _has_aligned:
            raise HTTPException(400, "这个目录里没有图片：%s" % p)
    elif not p.is_file():
        raise HTTPException(400, "不是普通文件或目录：%s" % p)
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
