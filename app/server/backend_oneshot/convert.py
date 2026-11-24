# -*- coding: utf-8 -*-
"""TF/Legacy .npy 模型 -> torch .pth 的转换任务（跑在 4.1 WebGUI 里）。

真正的转换由 4.0 的 tools/migrate/convert_dfl_tf_to_torch.py 完成——它已经把
权重名映射、卷积核 perm(3,2,0,1) 重排这些事做对了，这里只负责：
  1) 找到 workspace/model 里带 *_weights.npy 的模型
  2) 拉起那个脚本（用引擎 python，有 torch）
  3) 边跑边看目标目录里逐个模块的 .pth 出现，把进度喂给前端
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .._paths import engine_python

# 4.0 引擎目录（和训练共用同一套路径解析）
def _dfl_root() -> Path:
    from ..app import _DFL_ROOT
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
    from ..app import WORKSPACE
    return Path(WORKSPACE) / "model"


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return str(n)


def _migrate_dir() -> Path:
    """app/tools/migrate/ —— 迁移类脚本（导出 dfm / npy→pth）真正放的地方。

    引擎目录下**没有** tools/，以前按 <engine>/tools/ 找，两个脚本都找不到。
    """
    return Path(__file__).resolve().parents[2] / "tools" / "migrate"


def scan() -> list:
    """列出所有还没转成 torch 的 iperov TF 模型（有 *.npy，没有 *_encoder.pth）。"""
    md = model_dir()
    out = []
    if not md.is_dir():
        return out
    # iperov 原版（TF 2.0）有两套 npy 命名，都得认 —— 4.0/4.1 是本项目的 torch 版，权重是 .pth：
    #   单文件：<stem>_weights.npy（一个文件装全部权重）
    #   分模块：<stem>_encoder.npy / _decoder_src.npy / _inter.npy …
    # 以前只 glob *_weights.npy，于是分模块那批被判成「没有 TF 模型」，
    # 卡片上却写着「npy · 需转换」，点进去一片空白。
    _cands = {}
    for _f in sorted(md.glob("*.npy")):
        for _suf in ("_weights.npy", "_encoder.npy"):
            if _f.name.endswith(_suf):
                _stem = _f.name[: -len(_suf)]
                _prev = _cands.get(_stem)
                # 同一个模型两种命名都在时，优先 weights.npy（转换流程按它走）
                if _prev is None or _suf == "_weights.npy":
                    _cands[_stem] = (_f, _suf)
                break
    for stem, (npy, _suffix) in sorted(_cands.items()):
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
            "kind": ("modules" if _suffix == "_encoder.npy" else "weights"),
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


def _resolve_stem(md: Path, name: str) -> str:
    """模型卡片上写的是显示名（DF-UDT416），文件却是 DF-UDT416_SAEHD_*.npy。

    这里把两种写法都认下来：精确的 stem 直接用；否则按 <name>_* 去 glob 一次。
    """
    name = str(name).strip()
    for suf in ("_weights.npy", "_encoder.npy"):
        if (md / (name + suf)).exists():
            return name
    for suf in ("_weights.npy", "_encoder.npy"):
        hits = sorted(md.glob(name + "_*" + suf))
        if hits:
            return hits[0].name[: -len(suf)]
    return name


def start(name: str) -> dict:
    md = model_dir()
    stem = _resolve_stem(md, name)
    _log("resolve %s -> %s" % (name, stem))
    if not stem:
        raise ValueError("缺少模型名")
    npy = md / (stem + "_weights.npy")
    _mods_npy = md / (stem + "_encoder.npy")
    if not npy.exists() and not _mods_npy.exists():
        # 4.0 后期那批模型是「分模块 npy」（<stem>_encoder.npy / _decoder_src.npy …），
        # 本来就不需要转换：导出 dfm 的脚本直接读 npy。以前这里抛 FileNotFoundError，
        # 用户点「转换」只看到一句找不到文件，不知道该怎么办。
        raise FileNotFoundError(str(npy))
    exe = engine_python()
    conv = _migrate_dir() / "convert_tf_to_torch.py"
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
            # 专用转换器 app/tools/migrate/convert_tf_to_torch.py：只做 npy -> pth，
            # 不碰 .dfm。它内部复用 export_dfm.py 的 convert_npy_to_pth()（4.0 验证过
            # 的那条 models.import_model(is_exporting=True) + model.save() 路径）。
            cmd = [str(exe), str(conv),
                   "--model", stem.rsplit("_", 1)[-1] or "SAEHD",
                   "--model-dir", str(md), "--name", stem]
            _log("$ " + " ".join(cmd))
            _env = dict(os.environ)
            _env["DFL_TF_MATCH_LOG"] = "1"   # 让 Saveable 打出 [match] 行
            proc = subprocess.Popen(
                cmd, cwd=str(_dfl_root()), env=_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _set(step="convert")
            threading.Thread(target=_watch, args=(md, stem, proc, modules), daemon=True).start()
            # 逐行解析脚本输出，把模块一个个点亮 —— 光靠看 .pth 出现是不够的：
            # model.save() 会一口气把 4 个文件写完，中间态只有一瞬，前端根本看不到。
            _done_n = 0
            _seen = set()
            for line in proc.stdout:            # type: ignore[union-attr]
                line = line.rstrip()
                if not line:
                    continue
                _log(line)
                # 形如「├ encoder: .npy → .pth」「└ decoder_dst: .npy → .pth」
                _mm = re.match(r"^\[match\]\s+(.+)$", line.strip())
                if _mm:
                    # 真实的名字/形状匹配：把这一条塞进 msg，前端显示成滚动的一行
                    _set(msg="匹配 " + _mm.group(1)[:120])
                    continue
                _m = re.match(r"^[├└│\s]*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\.npy", line.strip())
                if _m:
                    _nm = _m.group(1)
                    if _nm in _seen:
                        continue
                    _seen.add(_nm)
                    if _done_n < len(modules):
                        modules[_done_n]["state"] = "ok"
                        modules[_done_n]["items"] = 1
                        _done_n += 1
                        _set(modules=modules,
                             percent=int(_done_n * 100 / max(1, len(modules))))

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
