# -*- coding: utf-8 -*-
"""B 站扫码登录 / 投稿列表 / 下载 / 账号状态。

每个模块只管自己的 APIRouter，app.py 只负责 include_router。

关于「占位」的历史包袱（重要）：
以前这里 /api/bili/download 上的注释写着「现在是可用的占位：文件名、时长、
体积按画质档变化」，但代码其实早就是真的 DASH 下载了；真正在造假的是前端
app/web/js/app.js 的 biliMockFetch()/biliMockDownload() —— 列表纯本地随机生成，
进度条纯 Math.random()，一条网络请求都不发。这一版：

  · /api/bili/up/videos 走真实 /x/space/wbi/arc/search（WBI 签名），
    失败按 401（未登录）/ 412（风控）/ 502（网络）明确返回，绝不返回空列表充数；
  · /api/bili/download 真下 DASH（cid + qn + fnval=4048），ffmpeg 合并；
  · 占位数据只在 --mock（DFL_BILI_MOCK=1）下由 bili_placeholder 提供，
    响应里带 mock=True，前端会标出来。
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from . import biliauth, bili_placeholder
from .bili_http import BiliError, mask_cookie

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("DFL_WORKSPACE") or (ROOT.parent / "workspace"))

router = APIRouter()

BILI = biliauth.CookieStore(WORKSPACE)
_bili_session = {"key": "", "created": 0}
# 登录态复核缓存：{cookie 指纹, 上次校验时间}，5 分钟内不重复打 nav
_login_probe = {"fp": "", "ts": 0.0}


def _log(msg: str):
    print("[bili] " + msg, flush=True)


def _err(e: BiliError) -> HTTPException:
    """BiliError -> HTTPException：kind 决定状态码，payload 原样给前端。"""
    return HTTPException(e.status, e.payload())


_logged_cookie_fp = {"fp": None}


def _log_cookie(cookie: str) -> None:
    """cookie 只在指纹变化时打一行（值永远不出现，只有名字+长度）。"""
    fp = hashlib.sha1(cookie.encode("utf-8")).hexdigest()
    if _logged_cookie_fp["fp"] != fp:
        _logged_cookie_fp["fp"] = fp
        _log("加载 cookie：%s" % mask_cookie(cookie))


def _load_cookie(require: bool = True) -> str:
    cookie = BILI.load()
    if require and (not cookie or "SESSDATA" not in cookie):
        _log("未登录：cookie 缺失（%s）" % mask_cookie(cookie))
        raise HTTPException(401, {"code": "not_logged_in",
                                  "message": "未登录 B 站：请先扫码登录"})
    _log_cookie(cookie)
    return cookie


def _require_login(cookie: str) -> None:
    """cookie 存在但已过期 -> 401；连不上 B 站 -> 502。

    只存 sha1 指纹，5 分钟内不重复请求 nav；日志里绝不出现 cookie 值。
    """
    fp = hashlib.sha1(cookie.encode("utf-8")).hexdigest()
    now = time.time()
    if _login_probe["fp"] == fp and (now - _login_probe["ts"]) < 300:
        return
    try:
        info = biliauth.nav(cookie)
    except BiliError as e:               # 网络/证书错误：502，不冒充「没登录」
        _log("登录态校验失败：%s" % e)
        raise _err(e)
    if not info.get("logged_in"):
        _log("登录态失效：%s code=%s" % (mask_cookie(cookie), info.get("code")))
        raise HTTPException(401, {"code": "not_logged_in",
                                  "message": "B 站登录已失效（%s），请重新扫码登录"
                                             % (info.get("message") or "code=%s" % info.get("code"))})
    _login_probe["fp"] = fp
    _login_probe["ts"] = now


@router.post("/api/bili/qr")
def bili_qr():
    try:
        d = biliauth.generate()
    except BiliError as e:
        raise _err(e)
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
    except BiliError as e:
        raise _err(e)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"轮询失败：{e}")

    if r.get("status") == "ok" and r.get("cookie"):
        cookie = r["cookie"]
        BILI.save(cookie)
        _log("登录成功，保存 cookie：%s" % mask_cookie(cookie))
        info = biliauth.fetch_user_info(cookie)
        if info:
            BILI.save_info(info)
        _bili_session["key"] = ""
        return {"status": "ok", "name": info.get("name", ""), "uid": info.get("uid", 0),
                "avatar": info.get("avatar", "")}
    return {"status": r.get("status", "pending"), "message": r.get("message", "")}


# ============================================================
# B 站下载：真下 DASH 音视频，写 workspace/raw/
# ============================================================
@router.post("/api/bili/download")
def bili_download(body: dict):
    """下载一条视频。

    真实链路：/x/web-interface/view 拿 cid -> /x/player/wbi/playurl 拿 dash
    音视频分流地址（fnval=4048 才有 dash）-> 分别拉取 -> ffmpeg 合流。
    需要的参数就是 bvid、cid、qn（清晰度），外加有效 cookie。
    """
    from .bili_video import QN_MAP, download_single_video, resolve_ffmpeg

    body = body or {}
    bvid = str(body.get("bvid") or "").strip()
    if not bvid:
        raise HTTPException(400, {"code": "bad_request", "message": "缺少 bvid"})
    quality = str(body.get("quality") or "best").strip()
    qn = QN_MAP.get(quality, QN_MAP["best"])

    cookie = _load_cookie()
    _require_login(cookie)
    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    try:
        ffmpeg = resolve_ffmpeg()
    except BiliError as e:
        raise _err(e)

    _log("开始下载 %s (quality=%s qn=%d)" % (bvid, quality, qn))
    try:
        path = download_single_video(bvid, cookie, str(raw), ffmpeg_path=ffmpeg, qn=qn)
    except BiliError as e:
        _log("下载失败 %s: [%s] %s" % (bvid, e.kind, e))
        raise _err(e)
    except Exception as e:
        _log("下载异常 %s: %s" % (bvid, e))
        raise HTTPException(500, {"code": "api", "message": "下载失败: %s" % e})
    if not path:
        raise HTTPException(500, {"code": "api", "message": "下载失败（没有产出文件）"})

    out = Path(path)
    return {"ok": True, "path": str(out), "size": out.stat().st_size if out.exists() else 0}


@router.get("/api/bili/me")
def bili_me():
    """登录态。cookie 缺失 -> logged_in=False；cookie 在但失效 -> logged_in=False + 原因；
    网络/证书错误 -> 502，不允许把「连不上」说成「没登录」。"""
    cookie = BILI.load()
    if not cookie or "SESSDATA" not in cookie:
        return {"logged_in": False, "code": "not_logged_in", "message": "还没有登录"}
    try:
        info = biliauth.nav(cookie)
    except BiliError as e:
        _log("nav 失败：%s" % e)
        raise _err(e)
    if info.get("logged_in"):
        BILI.save_info({"uid": info["uid"], "name": info["name"],
                        "avatar": info["avatar"], "level": info["level"]})
    _log("登录态：%s uid=%s" % (mask_cookie(cookie), info.get("uid")))
    return {"logged_in": bool(info.get("logged_in")),
            "name": info.get("name", ""),
            "uid": info.get("uid", 0),
            "avatar": info.get("avatar", ""),
            "level": info.get("level", 0),
            "code": info.get("code"),
            "message": info.get("message", "")}


@router.post("/api/bili/logout")
def bili_logout():
    BILI.clear()
    _bili_session["key"] = ""
    return {"ok": True}


# ============================================================
# UP 主投稿列表（真实接口）
# ============================================================
@router.post("/api/bili/up/videos")
def bili_up_videos(body: dict):
    """列某个 UP 主的投稿：{mid, page=1, ps=30}

    真实接口：/x/space/wbi/arc/search（需要 WBI 签名，需要登录 cookie）。
    失败一定报错：401 未登录 / 412 风控 / 404 用户不存在 / 502 网络问题。
    """
    from .bili_video import fetch_up_page

    body = body or {}
    mid = str(body.get("mid") or "").strip()
    page = int(body.get("page") or 1)
    ps = int(body.get("ps") or 30)
    if not mid.isdigit():
        raise HTTPException(400, {"code": "bad_request", "message": "mid 必须是纯数字 UID"})

    if bili_placeholder.enabled():
        _log("!! MOCK 模式：返回占位列表 mid=%s（DFL_BILI_MOCK=1）" % mid)
        data = bili_placeholder.placeholder_page(mid, page=page, ps=ps)
        data["ok"] = True
        return data

    cookie = _load_cookie()
    _require_login(cookie)          # 401/502 在这里就出去，不落进下面的 502 兜底
    try:
        data = fetch_up_page(mid, cookie, page=page, ps=ps)
    except BiliError as e:
        _log("拉取投稿失败 mid=%s page=%d: [%s] %s" % (mid, page, e.kind, e))
        raise _err(e)
    except Exception as e:
        _log("拉取投稿异常 mid=%s: %s" % (mid, e))
        raise HTTPException(502, {"code": "api", "message": "拉取失败: %s" % e})

    _log("投稿列表 mid=%s page=%d/%s total=%s 本页=%d"
         % (mid, data["page"], "?" if not data["total"] else data["total"],
            data["total"], len(data["items"])))
    data["ok"] = True
    data["mock"] = False
    return data


# ============================================================
# 批量下载
# ============================================================
@router.post("/api/bili/batch")
def bili_batch(body: dict):
    """批量下载：{bvids: [...], workers: 4, quality: 'best'}"""
    from . import bili_queue
    from .bili_video import QN_MAP, resolve_ffmpeg

    body = body or {}
    bvids = [str(b).strip() for b in (body.get("bvids") or []) if str(b).strip()]
    if not bvids:
        raise HTTPException(400, {"code": "bad_request", "message": "bvids 为空"})
    cookie = _load_cookie()
    _require_login(cookie)

    try:
        ffmpeg = resolve_ffmpeg()
    except BiliError as e:
        raise _err(e)

    qn = QN_MAP.get(str(body.get("quality") or "best"), QN_MAP["best"])
    raw = WORKSPACE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    res = bili_queue.start(bvids, cookie, str(raw), ffmpeg,
                           workers=int(body.get("workers") or 2), qn=qn)
    if not res.get("ok"):
        raise HTTPException(409, {"code": "busy", "message": res.get("error", "已有任务")})
    return res


@router.get("/api/bili/batch/status")
def bili_batch_status():
    from . import bili_queue
    return bili_queue.snapshot()


@router.post("/api/bili/batch/pause")
def bili_batch_pause(body: dict):
    from . import bili_queue
    paused = bool((body or {}).get("paused"))
    return bili_queue.set_paused(paused)


@router.post("/api/bili/batch/cancel")
def bili_batch_cancel():
    from . import bili_queue
    return bili_queue.cancel()


@router.get("/api/bili/me/detail")
def bili_me_detail():
    """当前登录用户的详细信息"""
    from .bili_user_info import fetch_user_info

    cookie = _load_cookie()
    try:
        return {"ok": True, "user": fetch_user_info(cookie)}
    except Exception as e:
        raise HTTPException(500, "获取失败: %s" % e)
