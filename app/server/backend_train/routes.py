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

from .. import app as _app_mod
from . import events as _load_events
from .._paths import engine_python as _engine_python
from .. import _train_device as _tdev
from ..app import (WEBUI_PORT, _DFL_ROOT, _engine_supports_cpu_only, _gpu_list, _kill_train_tree, _last_preview_state, _model_dat, _repair_option_strings, _sweep_orphan_trainers, _train_job, _train_job_lock, _train_worker, _webui_post)

router = APIRouter()


def _warn_payload(job):
    """把加载失败事件整理成界面要的形状（每条带一句可直接显示的 text）。

    训练没在跑就直接给空：横幅是"本次训练的状态"，跟着训练进程一起结束。
    """
    if not job.get("running"):
        return []
    return _load_events.warnings_payload(job.get("warnings") or [])


def _check_target(body):
    """控制类接口（暂停/继续/保存/停止）都作用于"正在训练的那个模型"。
    界面可能切到了别的模型再点暂停 —— 以前会不声不响停掉另一个，现在明确拒绝。"""
    name = str((body or {}).get("name") or "").strip()
    if not name:
        return
    with _train_job_lock:
        cur = _train_job.get("name")
        running = bool(_train_job.get("running"))
    if running and cur and name != cur:
        raise HTTPException(409, "现在训练的是 %s，不是 %s" % (cur, name))

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
    # 可选 options：一次请求就把"参数 + 启动"做完（前端不必先 POST /config 再 start）
    opts_in = body.get("options")
    if isinstance(opts_in, dict) and opts_in:
        from ..routes_models import _clean_options
        import pickle as _pk
        try:
            _data = _pk.loads(dat.read_bytes())
            _cur = _data.get("options") or {}
            _clean, _bad = _clean_options(opts_in, _cur)
            if _bad:
                raise HTTPException(400, "这些参数类型不对: " + ", ".join(_bad))
            _cur.update(_clean)
            _data["options"] = _cur
            _tmp = dat.with_suffix(".dat.tmp")
            _tmp.write_bytes(_pk.dumps(_data))
            _tmp.replace(dat)
            try:
                from ..backend_models.routes import _write_opt_cache as _woc
                _woc(dat, _data.get("iter") or 0, _data.get("options") or {})
            except Exception:
                pass
        except HTTPException:
            raise
        except Exception as _e:
            raise HTTPException(500, "写入训练参数失败: " + str(_e)[:120])
    fixed = _repair_option_strings(name)
    if fixed:
        print("[trainer] 修好了参数里的说明文字: " + ", ".join(fixed), flush=True)

    # ── 设备：参数已落盘，这里先按 dat 里的 device 验一遍 ──
    # 放在起线程之前，非法值才能变成一条能读懂的 400（带上可用 GPU 列表）；
    # 否则错误只会留在后台线程里，前端拿到的仍是 {"ok": true}，用户以为开训了。
    dev_raw, plan, dev_err = None, None, None
    try:
        import pickle as _pk2
        dev_raw = (_pk2.loads(dat.read_bytes()).get("options") or {}).get("device")
    except Exception:
        dev_raw = None
    try:
        plan = _tdev.plan(dev_raw, _gpu_list(), _engine_supports_cpu_only())
    except _tdev.DeviceOptionError as _e:
        dev_err = str(_e)
    if dev_err:
        raise HTTPException(400, dev_err)

    # 训练要独占显存：先把合成器/预览 worker 常驻的 dfm / XSeg 会话放掉，
    # 否则「没在跑合成，却提示显存不足」或训练峰值被挤爆。
    try:
        from ..backend_merge import vram as _vram
        _r1 = _vram._vram_release_inproc("训练启动")
        _r2 = _vram._free_preview_sessions("训练启动")
        print("[trainer] 释放合成会话: %s / %s" % (_r1.get("why") or _r1.get("freed"), _r2.get("did")), flush=True)
    except Exception as _e:
        print("[trainer] 释放合成会话失败（忽略）: %s" % str(_e)[:120], flush=True)

    threading.Thread(target=_train_worker, args=(name, bool(body.get("cpu_only"))), daemon=True).start()
    return {"ok": True, "name": name, "running": True,
            "device": plan, "device_banner": plan["banner"]}


@router.post("/api/train/save")
def train_save(body: dict = None):
    _check_target(body)
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
def train_pause(body: dict = None):
    _check_target(body)
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
def train_resume(body: dict = None):
    _check_target(body)
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
def train_stop(body: dict = None):
    _check_target(body)
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
            _webui_post("/request-quit", timeout=2.0)
            graceful = True
            break
        except Exception:
            # 前两次就够判断它是不是还在加载模型：加载期 WebUI 端口没起，
            # 以前会连等 6×4 秒再等 90 秒，用户感受就是点了停止没反应。
            if attempt >= 1:
                if _app_mod._train_proc is not None and _app_mod._train_proc.poll() is None:
                    time.sleep(2.0)
                    _kill_train_tree()
                    return {"ok": True, "graceful": False,
                            "msg": "训练还在加载模型（未进入迭代），已直接结束"}
                break
            time.sleep(1.0)

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




def _last_notice(lines):
    """从引擎日志里取最后一条 [notice] 提示（常驻信息条用）。"""
    for line in reversed(list(lines or [])):
        text = str(line or "").strip()
        if text.startswith("[notice]"):
            return text[len("[notice]"):].strip()
    return None
@router.get("/api/train/status")
def train_status():
    """训练状态。

    warnings[] 是**模块加载失败**的结构化事件（引擎打 [DFL-LOAD-EVENT] 行，
    app._train_worker 收进 _train_job）。它刻意做成状态而不是一次性通知：
    训练还在跑的时候，前端每次轮询 / 刷新页面 / 重进训练页都能再拿到一遍，
    于是那条"某模块被重置"的常驻横幅会重新出现 —— 这正是想要的行为，
    因为模型现在确实是从那个模块的初始状态在训练。
    """
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
                # 引擎打 [notice] 前缀的常驻提示（预训练 inter 重置等），
                # 前端用它渲染一条蓝色信息条；没有就是 None。
                "notice": _last_notice(j["log"]),
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
                "warnings": _warn_payload(j),
                "has_preview": bool(_last_preview_state.get("body")),
                "preview_iter": _last_preview_state.get("iter") or 0,
                # 设备回显：device=<值> → CUDA_VISIBLE_DEVICES=<env> 由 _train_worker
                # 算好存在 _train_job 里。老前端不读这个字段也不受影响。
                "device": j.get("device")}


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


