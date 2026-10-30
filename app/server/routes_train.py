# -*- coding: utf-8 -*-
"""训练控制（从 app.py 按路径前缀抽出）。"""
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

from . import app as _app_mod
from ._paths import engine_python as _engine_python
from .app import (WEBUI_PORT, _DFL_ROOT, _kill_train_tree, _last_preview_state, _model_dat, _repair_option_strings, _sweep_orphan_trainers, _train_job, _train_job_lock, _train_worker, _webui_post)

router = APIRouter()

@router.post("/api/train/start")
def train_start(body: dict):
    body = body or {}
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "缺少模型名")
    dat = _model_dat(name)
    if dat is None:
        raise HTTPException(404, "找不到模型 %s —— 先用「新建模型」建一个，或检查 workspace/model 里有没有 %s_SAEHD_data.dat" % (name, name))
    if not (_engine_python()).is_file():
        raise HTTPException(500, "找不到 4.0 的引擎（%s）—— 见 app/server/_paths.py：可以写 engine.txt 或设 DFL_ENGINE_ROOT" % _DFL_ROOT)
    with _train_job_lock:
        if _train_job["running"]:
            raise HTTPException(409, "已经在训练 " + _train_job["name"])
    fixed = _repair_option_strings(name)
    if fixed:
        print("[trainer] 修好了参数里的说明文字: " + ", ".join(fixed), flush=True)
    threading.Thread(target=_train_worker, args=(name, bool(body.get("cpu_only"))), daemon=True).start()
    return {"ok": True, "name": name, "running": True}


@router.post("/api/train/save")
def train_save():
    """立即保存：让 4.0 在下一个迭代结束后落盘（不打断训练）。"""
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        _train_job["saving"] = True
        _train_job["saving_until"] = time.time() + 20
    try:
        _webui_post("/request-save")
    except Exception as e:
        with _train_job_lock:
            _train_job["saving"] = False
        raise HTTPException(503, "训练还没加载完（WebUI 未就绪）：%s" % str(e)[:120])
    return {"ok": True, "msg": "已请求保存，下一个迭代后落盘"}


@router.post("/api/train/pause")
def train_pause():
    """暂停：挂起训练进程（psutil suspend），迭代停在原地，显存不释放。"""
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        if _train_job["paused"]:
            return {"ok": True, "paused": True}
    proc = _app_mod._train_proc
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "训练进程不在")
    try:
        import psutil
        p = psutil.Process(proc.pid)
        p.suspend()
        for c in p.children(recursive=True):
            try:
                c.suspend()
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(500, "挂起失败：%s" % str(e)[:120])
    with _train_job_lock:
        _train_job["paused"] = True
    return {"ok": True, "paused": True}


@router.post("/api/train/resume")
def train_resume():
    with _train_job_lock:
        if not _train_job["running"]:
            raise HTTPException(409, "现在没有在训练")
        if not _train_job["paused"]:
            return {"ok": True, "paused": False}
    proc = _app_mod._train_proc
    if proc is None or proc.poll() is not None:
        raise HTTPException(409, "训练进程不在")
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for c in p.children(recursive=True):
            try:
                c.resume()
            except Exception:
                pass
        p.resume()
    except Exception as e:
        raise HTTPException(500, "恢复失败：%s" % str(e)[:120])
    with _train_job_lock:
        _train_job["paused"] = False
    return {"ok": True, "paused": False}


@router.post("/api/train/stop")
def train_stop():
    """停止：先请 4.0 保存并退出（/request-quit），超时再强杀整棵树。"""
    with _train_job_lock:
        running = _train_job["running"]
        _train_job["stopping"] = True
    if not running:
        n = _sweep_orphan_trainers("(停止时发现遗留)")
        return {"ok": True, "msg": ("已清理 %d 个遗留训练进程" % n) if n else "没有在训练"}
    # 挂起状态下整个进程（包括 WebUI 线程）都冻着，先恢复再发指令 ——
    # 否则每一次 /request-quit 都会超时，白等 90 秒最后强杀。
    with _train_job_lock:
        paused = _train_job["paused"]
    if paused:
        try:
            import psutil
            if _app_mod._train_proc:
                p_ = psutil.Process(_app_mod._train_proc.pid)
                for c_ in p_.children(recursive=True):
                    try:
                        c_.resume()
                    except Exception:
                        pass
                p_.resume()
        except Exception:
            pass
        with _train_job_lock:
            _train_job["paused"] = False
        time.sleep(0.5)

    graceful = False
    for attempt in range(6):
        try:
            _webui_post("/request-quit", timeout=4.0)
            graceful = True
            break
        except Exception:
            time.sleep(1.5)

    # 等它自己收尾（保存 + 退出）；最多 90 秒，然后再强杀
    for _ in range(90):
        if _app_mod._train_proc is None or _app_mod._train_proc.poll() is not None:
            break
        time.sleep(1)
    if _app_mod._train_proc is not None and _app_mod._train_proc.poll() is None:
        _kill_train_tree()
        return {"ok": True, "graceful": False, "msg": "已强制结束（没等到它自己退出）"}
    return {"ok": True, "graceful": graceful,
            "msg": "已保存并退出" if graceful else "已结束"}


@router.get("/api/train/preview")
def train_preview(wait: float = 0.0):
    """把 4.0 WebUI 的实时预览转出来。

    训练进程会在 127.0.0.1:6790 起一个监控服务：
      GET /request-preview  → 让它下一轮吐一张
      GET /preview          → pop 语义，取走就没了

    wait=N 会**一直等到有新图**（最多 N 秒）再返回。前端点「刷新」时用它，
    否则拿到的是上一次的缓存 —— 看起来就是「点了没反应 / 有滞后」。

    返回体在最外层补三个字段（4.0 原样透传 previews/labels）：
      fresh  这次是不是真的拿到了新图
      iter   这张图对应的迭代数（拿到新图时记录）
      age_s  这张图距今多少秒
    """
    import json as _json
    import time as _time
    import urllib.request

    port = WEBUI_PORT
    base = "http://127.0.0.1:%d" % port
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(path, timeout=1.2):
        req = urllib.request.Request(base + path, headers={"User-Agent": "dfl41"})
        with opener.open(req, timeout=timeout) as r:
            return r.read()

    def _pack(raw, fresh):
        try:
            d = _json.loads(raw)
        except Exception:
            d = {"previews": [], "labels": {}}
        d["fresh"] = bool(fresh)
        d["iter"] = int(_last_preview_state.get("iter") or 0)
        at = float(_last_preview_state.get("at") or 0)
        d["age_s"] = round(time.time() - at, 1) if at else None
        return Response(content=_json.dumps(d).encode("utf-8"),
                        media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    deadline = time.time() + max(0.0, min(float(wait or 0.0), 25.0))
    requested = False
    raw = None

    while True:
        if not requested:
            try:
                _get("/request-preview", timeout=0.8)
                requested = True
            except Exception:
                # 训练没在跑（或还没起 WebUI）：别让前端干等
                cached = _last_preview_state.get("body")
                if cached:
                    return _pack(cached, False)
                return _pack(b'{"previews":[],"labels":{}}', False)
        try:
            d = _json.loads(_get("/preview"))
            if d.get("previews"):
                raw = _json.dumps(d).encode("utf-8")
                _last_preview_state.update({"body": raw,
                                            "iter": _train_job.get("iter") or 0,
                                            "at": time.time()})
                return _pack(raw, True)
        except Exception:
            pass
        if time.time() >= deadline:
            break
        _time.sleep(0.2)

    cached = _last_preview_state.get("body")
    if cached:
        return _pack(cached, False)
    return _pack(b'{"previews":[],"labels":{}}', False)


@router.get("/api/train/status")
def train_status():
    with _train_job_lock:
        j = _train_job
        it = int(j.get("iter") or 0)
        rate = float(j.get("it_s") or 0.0)
        started = float(j.get("started_at") or 0.0)
        target = int(j.get("target") or 0)
        elapsed = (time.time() - started) if (started and j["running"]) else 0.0
        eta = None
        if target and rate > 0 and it < target:
            eta = int((target - it) / rate)
        return {"running": j["running"], "name": j["name"],
                "iter": it, "ok": j["ok"],
                "error": j["error"], "log": j["log"][-80:],
                "it_s": rate, "elapsed_s": int(elapsed),
                "target_iter": target, "eta_s": eta,
                "iter_seen": bool(j.get("iter_seen")),
                "paused": bool(j.get("paused")),
                "saving": bool(j.get("saving")) and time.time() < float(j.get("saving_until") or 0),
                "stopping": bool(j.get("stopping")),
                "last_backup_iter": j.get("last_backup_iter") or 0,
                "lr": j.get("lr") or 0.0,
                "loss_src": j.get("loss_src"),
                "loss_dst": j.get("loss_dst"),
                "history": j.get("history") or [],
                "has_preview": bool(_last_preview_state.get("body")),
                "preview_iter": _last_preview_state.get("iter") or 0}
