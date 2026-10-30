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
    """把一个视频「下载」到 workspace/raw/。返回真实落盘路径和大小。"""
    import subprocess
    import imageio_ffmpeg
    body = body or {}
    bvid = str(body.get("bvid") or "").strip()
    title = str(body.get("title") or "").strip() or bvid or "untitled"
    quality = str(body.get("quality") or "best")

    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    safe = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:60] or "video"
    name = (bvid + "_" if bvid else "") + safe + ".mp4"
    out = raw / name
    if out.exists():
        return {"ok": True, "path": str(out), "size": out.stat().st_size, "skipped": True}

    # 画质决定时长（30/20/12 秒），顺便决定文件体积
    secs = {"best": 30, "1080": 20, "720": 12}.get(quality, 20)
    size = {"best": "1280x720", "1080": "1280x720", "720": "960x540"}.get(quality, "1280x720")

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-v", "error", "-y",
           "-f", "lavfi", "-i", "color=c=0x202024:s=%s:d=%d:r=24" % (size, secs),
           "-f", "lavfi", "-i", "sine=frequency=440:duration=%d" % secs,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
           "-c:a", "aac", "-b:a", "64k", "-shortest", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, creationflags=_NO_WINDOW)
    except Exception as e:
        raise HTTPException(500, "生成失败: " + str(e))
    if proc.returncode != 0 or not out.exists():
        raise HTTPException(500, "ffmpeg 退出码 %d: %s" % (
            proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[:200]))
    return {"ok": True, "path": str(out), "size": out.stat().st_size,
            "seconds": secs, "quality": quality}


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
