# -*- coding: utf-8 -*-
"""自动更新路由：检查 / 安装。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel

from . import backend_update as U

router = APIRouter(prefix="/api/update", tags=["update"])

@router.get("/auto")
def update_auto():
    """启动时后台查的那一次结果（前端首页横幅用）。

    checked=False 表示还没查完（进程刚起来），前端可以过几秒再问；
    behind=True 才提示有新版本。
    """
    return U.auto_state()


@router.get("/progress")
def update_progress():
    """安装进度（下载 % / 解包 / 覆盖）。前端在 apply 期间轮询这个接口。"""
    return U.progress()


class ApplyBody(BaseModel):
    asset: Optional[str] = None


@router.get("/check")
def check() -> dict:
    return U.check()


@router.get("/version")
def version() -> dict:
    return U.parse_version()          # {version, base, when, human}


@router.get("/selfcheck")
def selfcheck() -> dict:
    """这份安装是不是「修好的那一版」—— **不读版本号，直接探测特征串**。

    为什么需要它：用户可能只拷了几个文件、或者改了源码没重启，`version.txt`
    写着 b45 而跑着 b23 的代码。版本号会说谎，`inspect.getsource` 不会。
    返回 ok=False 时 missing 里就是缺哪几项修复 —— 直接告诉用户该更新。
    """
    from ._selfcheck import selfcheck as _sc
    out = dict(U.parse_version() or {})
    try:
        chk = _sc()
        out.update(chk)
    except Exception as e:                                 # noqa: BLE001
        out.update({"ok": None, "missing": [], "items": [],
                    "error": "%s: %s" % (e.__class__.__name__, str(e)[:200])})
    return out


@router.post("/apply")
def apply(body: ApplyBody = Body(default=ApplyBody())) -> dict:
    return U.apply(body.asset)
