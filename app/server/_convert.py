# -*- coding: utf-8 -*-
"""TF/Legacy .npy 模型 -> torch .pth 的转换任务（跑在 4.1 WebGUI 里）。

真正的转换由 4.0 的 tools/convert_dfl_tf_to_torch.py 完成——它已经把
权重名映射、卷积核 perm(3,2,0,1) 重排这些事做对了，这里只负责：
  1) 找到 workspace/model 里带 *_weights.npy 的模型
  2) 拉起那个脚本（用引擎 python，有 torch）
  3) 边跑边看目标目录里逐个模块的 .pth 出现，把进度喂给前端
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ._paths import engine_python

# 4.0 引擎目录（和训练共用同一套路径解析）
def _dfl_root() -> Path:
    from .app import _DFL_ROOT
    return Path(_DFL_ROOT)

MODULE_SUFFIXES = ["encoder", "inter_AB", "inter_B", "decoder", "inter", "decoder_src", "decoder_dst"]

CONVERT_JOB = {
    "running": False,
    "name": "",
    "file": "",
    "started": 0.0,
    "ended": 0.0,
    "step": "idle",          # idle | load | convert | done | error
    "modules": [],           # [{"name": "encoder", "state": "wait|run|ok", "items": 0}]
    "percent": 0,
    "error": "",
    "log": [],
}
_LOCK = threading.Lock()


def model_dir() -> Path:
    from .app import WORKSPACE
    return Path(WORKSPACE) / "model"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return str(n)


def scan() -> list:
    """列出所有还没转换的老模型（有 *_weights.npy，没有 *_encoder.pth）。"""
    md = model_dir()
    out = []
    if not md.is_dir():
        return out
    for npy in sorted(md.glob("*_weights.npy")):
        stem = npy.name[: -len("_weights.npy")]        # 例：_smoketrain_SAEHD
        arch = ""
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            arch = parts[1].lower()
        converted = (md / (stem + "_encoder.pth")).exists()
        dat = md / (stem + "_data.dat")
        out.append({
            "name": stem,
            "file": npy.name,
            "arch": arch,
            "size": npy.stat().st_size,
            "sizeText": _fmt_size(npy.stat().st_size),
            "hasDat": dat.exists(),
            "converted": converted,
        })
    return out


def _set(**kw):
    with _LOCK:
        CONVERT_JOB.update(kw)


def _log(line: str):
    with _LOCK:
        CONVERT_JOB["log"].append(line)
        if len(CONVERT_JOB["log"]) > 200:
            CONVERT_JOB["log"] = CONVERT_JOB["log"][-200:]


def _watch(md: Path, stem: str, proc, modules):
    """轮询目标目录，哪个模块的 .pth 出现了就把它标成 ok。"""
    seen = set()
    while proc.poll() is None:
        for m in modules:
            if m["state"] == "ok":
                continue
            p = md / (stem + "_" + m["name"] + ".pth")
            if p.exists() and p.stat().st_size > 0:
                m["state"] = "ok"
                m["items"] = 1
                seen.add(m["name"])
                done = sum(1 for x in modules if x["state"] == "ok")
                _set(modules=modules, percent=int(done * 100 / max(1, len(modules))))
        time.sleep(0.3)


def start(name: str) -> dict:
    md = model_dir()
    stem = str(name).strip()
    if not stem:
        raise ValueError("缺少模型名")
    npy = md / (stem + "_weights.npy")
    if not npy.exists():
        raise FileNotFoundError(str(npy))
    exe = engine_python()
    conv = _dfl_root() / "tools" / "convert_dfl_tf_to_torch.py"
    if not conv.exists():
        raise FileNotFoundError(str(conv))

    # 模块清单：优先按 weights.npy 里的键前缀推断，读不出来就按 SAEHD 默认给四个
    modules = [{"name": n, "state": "wait", "items": 0} for n in ("encoder", "inter_AB", "inter_B", "decoder")]
    try:
        import pickle
        with open(npy, "rb") as f:
            data = pickle.load(f)
        keys = list(data.keys()) if isinstance(data, dict) else []
        pref = []
        for k in keys:
            head = k.split("/")[0].split(":")[0]
            if head and head not in pref:
                pref.append(head)
        pref = [p for p in pref if p in MODULE_SUFFIXES]
        if pref:
            modules = [{"name": p, "state": "wait", "items": 0} for p in pref]
    except Exception as e:                     # 读不出来不影响转换本身
        _log("读取 npy 键失败（用默认模块表）: %s" % e)

    _set(running=True, name=stem, file=npy.name, started=time.time(), ended=0.0,
         step="load", modules=modules, percent=0, error="", log=[])

    def worker():
        try:
            cmd = [str(exe), str(conv), "--src", str(md), "--dst", str(md),
                   "--model", stem.rsplit("_", 1)[-1] or "SAEHD", "--name", stem]
            _log("$ " + " ".join(cmd))
            proc = subprocess.Popen(
                cmd, cwd=str(_dfl_root()),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _set(step="convert")
            threading.Thread(target=_watch, args=(md, stem, proc, modules), daemon=True).start()
            for line in proc.stdout:            # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    _log(line)
            rc = proc.wait()
            if rc == 0:
                for m in modules:
                    m["state"] = "ok"
                    m["items"] = 1
                _set(running=False, step="done", modules=modules, percent=100, ended=time.time())
            else:
                _set(running=False, step="error", ended=time.time(),
                     error="转换脚本退出码 %d（看 log 最后几行）" % rc)
        except Exception as e:
            _set(running=False, step="error", ended=time.time(), error=str(e)[:300])

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "name": stem, "modules": modules}


def status() -> dict:
    with _LOCK:
        return json.loads(json.dumps(CONVERT_JOB))
