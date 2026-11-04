# -*- coding: utf-8 -*-
"""B 站扫码登录 / 下载 / 账号状态。

从 app.py 里拆出来的第一个域，作为「按域拆路由」的样板：
每个模块只管自己的 APIRouter，app.py 只负责 include_router。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from . import biliauth

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("DFL_WORKSPACE") or (ROOT.parent / "workspace"))

router = APIRouter()

BILI = biliauth.CookieStore(WORKSPACE)
_bili_session = {"key": "", "created": 0}


@router.post("/api/bili/qr")
def bili_qr():
    try:
        d = biliauth.generate()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"生成二维码失败：{e}")
    _bili_session["key"] = d["key"]
    _bili_session["created"] = int(__import__("time").time())
    return {"ok": True, "key": d["key"], "qr": d["qr_png"]}


@router.post("/api/bili/poll")
def bili_poll(payload: dict = None):
    key = (payload or {}).get("key") or _bili_session.get("key")
    if not key:
        raise HTTPException(status_code=400, detail="还没有生成二维码")
    try:
        r = biliauth.poll(key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"轮询失败：{e}")

    if r.get("status") == "ok" and r.get("cookie"):
        cookie = r["cookie"]
        BILI.save(cookie)
        info = biliauth.fetch_user_info(cookie)
        if info:
            BILI.save_info(info)
        _bili_session["key"] = ""
        return {"status": "ok", "name": info.get("name", ""), "uid": info.get("uid", 0),
                "avatar": info.get("avatar", "")}
    return {"status": r.get("status", "pending"), "message": r.get("message", "")}


# ============================================================
# B 站下载：真的往 workspace/raw 写文件（用 ffmpeg 生成等价小片）
# 现在是「可用的占位」：文件名、时长、体积按画质档变化，落盘后提取器能直接用
# ============================================================
@router.post("/api/bili/download")
def bili_download(body: dict):
    """真的从 B 站下载一个视频到 workspace/raw/。

    走 4.0 移植过来的 bili_video.download_single_video：DASH 音视频分别拉取，
    再用 ffmpeg 合并。需要已登录（Cookie 从 biliauth 的加密存储里读）。
    """
    import imageio_ffmpeg
    from .bili_video import download_single_video

    body = body or {}
    bvid = str(body.get("bvid") or "").strip()
    if not bvid:
        raise HTTPException(400, "缺少 bvid")

    cookie = BILI.load()
    if not cookie:
        raise HTTPException(401, "未登录 B 站：请先扫码登录")

    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        ok = download_single_video(bvid, cookie, str(raw), ffmpeg_path=ffmpeg)
    except Exception as e:
        raise HTTPException(500, "下载失败: %s" % e)
    if not ok:
        raise HTTPException(500, "下载失败（见服务端日志）")

    # 返回刚写入的文件（按 bvid 前缀找最新的）
    files = sorted(raw.glob(bvid + "*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        cands = sorted(raw.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        files = cands[:1]
    out = files[0] if files else None
    return {"ok": True,
            "path": str(out) if out else "",
            "size": out.stat().st_size if out else 0}


@router.get("/api/bili/me")
def bili_me():
    cookie = BILI.load()
    if not cookie or "SESSDATA" not in cookie:
        return {"logged_in": False}
    info = BILI.load_info()
    if not info:
        info = biliauth.fetch_user_info(cookie)
        if info:
            BILI.save_info(info)
    return {"logged_in": bool(info), "name": info.get("name", ""),
            "uid": info.get("uid", 0), "avatar": info.get("avatar", "")}


@router.post("/api/bili/logout")
def bili_logout():
    BILI.clear()
    _bili_session["key"] = ""
    return {"ok": True}


# ============================================================
# 批量下载 / UP 主视频列表（从 4.0 移植）
# ============================================================
@router.post("/api/bili/batch")
def bili_batch(body: dict):
    """批量下载：{bvids: [...], workers: 2}"""
    import imageio_ffmpeg
    from . import bili_queue

    body = body or {}
    bvids = [str(b).strip() for b in (body.get("bvids") or []) if str(b).strip()]
    if not bvids:
        raise HTTPException(400, "bvids 为空")
    cookie = BILI.load()
    if not cookie:
        raise HTTPException(401, "未登录 B 站")

    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    res = bili_queue.start(bvids, cookie, str(raw),
                           imageio_ffmpeg.get_ffmpeg_exe(),
                           workers=int(body.get("workers") or 2))
    if not res.get("ok"):
        raise HTTPException(409, res.get("error", "已有任务"))
    return res


@router.get("/api/bili/batch/status")
def bili_batch_status():
    from . import bili_queue
    return bili_queue.snapshot()


@router.post("/api/bili/up/videos")
def bili_up_videos(body: dict):
    """列某个 UP 主的投稿：{mid, max_pages}"""
    from .bili_video import fetch_up_videos

    body = body or {}
    mid = str(body.get("mid") or "").strip()
    if not mid:
        raise HTTPException(400, "缺少 mid")
    cookie = BILI.load()
    try:
        items = fetch_up_videos(mid, cookie, max_pages=int(body.get("max_pages") or 1))
    except Exception as e:
        raise HTTPException(500, "拉取失败: %s" % e)
    return {"ok": True, "items": items}


@router.get("/api/bili/me/detail")
def bili_me_detail():
    """当前登录用户的详细信息"""
    from .bili_user_info import fetch_user_info

    cookie = BILI.load()
    if not cookie:
        raise HTTPException(401, "未登录 B 站")
    try:
        return {"ok": True, "user": fetch_user_info(cookie)}
    except Exception as e:
        raise HTTPException(500, "获取失败: %s" % e)
