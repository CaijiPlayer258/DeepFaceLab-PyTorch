# -*- coding: utf-8 -*-
"""引擎编译页的后端：GET /api/engines（状态）+ POST /api/engines/build（编译）。

从 backend_models/trt.py 搬出（原第 23-176 行，逐字搬，行为零变化）：
  _engine_job / _engine_job_lock   编译任务状态（前端轮询同一个 job）
  _engine_roots()                  找 onnx 的三个根：工作区 → weights 平铺 → 仓库根
  _engines_verify_all()            启动时后台核一遍（md5 + SM）
  _engine_build_worker(name)       子进程跑 _trt_build.py，解析 @@P/@@OK/@@ERR
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..app import WORKSPACE, _NO_WINDOW

router = APIRouter()

# ---------------- TensorRT 引擎（编译/状态） ----------------
_engine_job = {"running": False, "name": "", "log": [], "error": "", "done": False}


def _engine_roots():
    """能找 onnx 的根目录：工作区 → weights/ 平铺根 → 仓库根。

    4.0 仓库（parents[4]/DeepFaceLab-PyTorch）不再兜底：权重已经全部平铺在
    <仓库>/weights/ 下，继续认 4.0 只会把「没下载」报成「找到了」。
    """
    from .._paths import weights_dir
    return [WORKSPACE, weights_dir(), Path(__file__).resolve().parents[3]]


def _engines_verify_all():
    """启动时在后台把引擎状态核一遍（md5 + SM），不阻塞启动。"""
    try:
        from ..backend_models.engines import status as _st
        _st(WORKSPACE, _engine_roots())
    except Exception:
        pass


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
    from ..backend_models.engines import status as _st
    st = _st(WORKSPACE, _engine_roots())
    st["job"] = dict(_engine_job)
    return st


def _engine_build_worker(name: str):
    import subprocess
    import sys
    from ..backend_models.engines import status as _st, engine_path as _ep
    from ..backend_models.engines import MODEL_SPECS

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

    from ..backend_models.engines import PROFILES
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "backend_models" / "trt_build.py"),
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


# 启动时后台核对一次引擎状态（md5 + SM 对不上会被标成过期/未编译）。
# 原来在 backend_models/trt.py 末尾 —— 那个文件已拆空并删除，线程挪到这里（它是引擎自己的事）。
threading.Thread(target=_engines_verify_all, daemon=True).start()
