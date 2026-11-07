# -*- coding: utf-8 -*-
"""错脸检测（ErrFaceFilter / xgboost 68 点几何）路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import _errface

router = APIRouter()


def _status(force: bool = False) -> dict:
    try:
        return _errface.status(force=force)
    except Exception as e:                      # 探测自己炸了也不能 500：能力状态照回
        return {"capability": "errface", "available": False,
                "model": _errface.model_path().name,
                "reason": "能力探测失败：%s" % str(e)[:200], "reason_en": "",
                "install": "", "install_en": "", "deps": {}}


@router.get("/api/errface/info")
def errface_info(refresh: str = ""):
    """能力状态。available=false 时 reason/install 直接用给人看的中英文。

    以前只有 available 一个布尔 —— 前端只能等用户点了 /api/dataset/scan 再吃 409。
    """
    st = _status(force=str(refresh or "").strip().lower() in ("1", "true", "yes", "on"))
    return dict({"ok": True}, **st)


@router.post("/api/errface/scan")
def errface_scan(body: dict):
    body = body or {}
    d = str(body.get("dir") or "").strip()
    if not d:
        raise HTTPException(400, "缺少目录")
    st = _status()
    if not st.get("available"):
        # 能力没装 = 状态，不是故障：200 + 说清怎么装，别抛 409/500
        return {"ok": False, "unavailable": True, "capability": "errface",
                "reason": st.get("reason") or "", "install": st.get("install") or "",
                "status": st}
    try:
        res = _errface.scan(d, threshold=float(body.get("threshold") or 0.5),
                            limit=int(body.get("limit") or 0),
                            percent=float(body.get("percent") or 0))
    except Exception as e:
        raise HTTPException(500, str(e)[:300])
    if not res.get("ok") and res.get("unavailable"):
        return {"ok": False, "unavailable": True, "capability": "errface",
                "reason": res.get("reason") or "", "install": res.get("install") or "",
                "status": res.get("status") or st}
    return res
