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
    # 结果落在哪：转换产物是 model_dir 下的 <stem>_*.pth，以前界面上一个字都不提，
    # 用户看完「已完成」不知道文件去了哪。outDir 给目录，files 给实际写出的文件。
    "outDir": "",
    "files": [],             # [{"name": "...pth", "module": "encoder", "size": 0, "sizeText": "98.0 MB"}]
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


def _scan_outputs(md: Path, stem: str) -> list:
    """转换到底写出了哪几个文件 —— 直接问磁盘，不猜命名规则。

    model.save() 写的是 <model_dir>/<stem>_<module>.pth（stem 已含 _SAEHD）。
    以前界面上只有「已完成」，用户不知道文件落在哪个目录、叫什么名字。
    优化器状态（*_opt.pth）不是权重，单独滤掉。
    """
    out = []
    try:
        _hit = sorted(md.glob(stem + "_*.pth"), key=lambda x: x.stat().st_mtime)
    except Exception:
        return out
    for p in _hit:
        _tail = p.name[len(stem) + 1:] if p.name.startswith(stem + "_") else p.name
        if not _tail.endswith(".pth"):
            continue
        _mod = _tail[:-4]
        if _mod.endswith("_opt") or "_opt_" in _mod:
            continue
        try:
            _sz = p.stat().st_size
        except OSError:
            _sz = 0
        out.append({"name": p.name, "module": _mod, "size": _sz, "sizeText": _fmt_size(_sz)})
    return out


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

    # 模块清单：**从模型权重里读真实的那几个**，不要套模板。
    # 以前这里过滤 MODULE_SUFFIXES（那是 LIAE 的 encoder/inter_AB/inter_B/decoder），
    # DF 模型的 inter 直接被滤掉、decoder 还只剩一个 —— 界面上就凭空多出 inter_B、
    # 少掉 decoder_dst。现在原样保留权重里出现的模块名（去重、保序）。
    modules = []
    # ① 先看模型配置里的 archi_type —— 这是权威来源：
    #    df   -> encoder / inter / decoder_src / decoder_dst
    #    liae -> encoder / inter_AB / inter_B / decoder
    try:
        import pickle as _pk
        _dat = md / (stem + "_data.dat")
        if _dat.exists():
            _cfg = _pk.loads(_dat.read_bytes())
            _opts = _cfg.get("options") or {}
            # 字段名是 archi，值是 liae-ud / df-udt 这种；archi_type 是另外算出来的
            _archi = str(_opts.get("archi_type") or _opts.get("archi") or "").lower()
            if "liae" in _archi:
                modules = [{"name": n, "state": "wait", "items": 0}
                           for n in ("encoder", "inter_AB", "inter_B", "decoder")]
            elif "df" in _archi:
                modules = [{"name": n, "state": "wait", "items": 0}
                           for n in ("encoder", "inter", "decoder_src", "decoder_dst")]
    except Exception:
        pass

    # ② 配置读不出来时才退回看权重键名（单文件 _weights.npy 才有全量键）
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
        # 只留看起来像模块名的（模型自己的模块就是这些前缀）
        pref = [p for p in pref if p and not p.startswith("_")]
        if pref:
            modules = [{"name": p, "state": "wait", "items": 0} for p in pref]
    except Exception as e:
        _log("读取 npy 键失败: %s" % e)
    if not modules:
        modules = [{"name": n, "state": "wait", "items": 0} for n in ("encoder", "inter", "decoder")]

    _set(running=True, name=stem, file=npy.name, started=time.time(), ended=0.0,
         step="load", modules=modules, percent=0, error="", log=[],
         outDir=str(md), files=[])

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
            _env["PYTHONIOENCODING"] = "utf-8"   # 否则中文异常在 GBK 下变乱码，页面上只剩「退出码 1」
            _env["PYTHONUTF8"] = "1"
            _set(step="convert", msg="正在匹配权重名与形状…")
            proc = subprocess.Popen(
                cmd, cwd=str(_dfl_root()), env=_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _set(step="convert", msg="正在匹配权重名与形状…")
            # _watch 是旧的「看 .pth 出现」轮询：它会把每个模块的 items 一律设成 1，
            # 和我们从 [match] 行累计出来的真实张量数（18 / 4 / 0 / 88）冲突。停用它。
            # 逐行解析脚本输出，把模块一个个点亮 —— 光靠看 .pth 出现是不够的：
            # model.save() 会一口气把 4 个文件写完，中间态只有一瞬，前端根本看不到。
            _log_lines = []
            _done_n = 0
            _seen = set()
            for line in proc.stdout:            # type: ignore[union-attr]
                line = line.rstrip()
                if not line:
                    continue
                _log(line)
                # 形如「├ encoder: .npy → .pth」「└ decoder_dst: .npy → .pth」
                _mm = re.match(r"^\[match\]\s+([^|]+)\|(.+)$", line.strip())
                if _mm:
                    # 形如「[match] encoder|down1/conv1/weight:0 (3,3,4,64) -> (64,4,3,3) 转置 (名字)」
                    # 归到对应模块上：累计命中的张量数（真实项数）+ 记下最后一条明细
                    _mn, _rest = _mm.group(1).strip(), _mm.group(2).strip()
                    for _mod in modules:
                        _n = _mod.get("name") or ""
                        if _n and (_n == _mn or _n in _mn or _mn in _n):
                            _mod["items"] = int(_mod.get("items") or 0) + 1
                            _mod["detail"] = _rest[:120]
                            break
                    _set(msg="匹配 " + _rest[:110], modules=modules)
                    continue
                _m = re.match(r"^[├└│\s]*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\.npy", line.strip())
                if _m:
                    _nm = _m.group(1)
                    if _nm in _seen:
                        continue
                    _seen.add(_nm)
                    if _done_n < len(modules):
                        modules[_done_n]["state"] = "ok"
                        # items 别动：那是 [match] 行逐个张量累计出来的真实个数
                        _done_n += 1
                        _set(modules=modules,
                             percent=int(_done_n * 100 / max(1, len(modules))))

            rc = proc.wait()
            if rc == 0:
                for m in modules:
                    m["state"] = "ok"
                    m.pop("detail", None)   # 匹配日志只在跑的时候看，收尾别留在卡片上
                    # items 保留 [match] 累计出来的真实张量数（18 / 4 / 0 / 88），
                    # 以前这里粗暴覆盖成 1，页面上每个模块都显示「1 项」
                _files = _scan_outputs(md, stem)
                for _m in modules:
                    _hit = [f for f in _files if f["module"].endswith(_m.get("name") or "")
                            or _m.get("name") in f["module"]]
                    if _hit:
                        _m["sizeText"] = _hit[0]["sizeText"]
                        _m["file"] = _hit[0]["name"]
                _set(running=False, step="done", modules=modules, percent=100,
                     ended=time.time(), outDir=str(md), files=_files)
            else:
                # 失败也要给出「哪个模块 / 为什么」，否则前端只有整张卡变红，
                # 看不到设计稿里 .cvreason 那一行（形状不符/名字对不上/文件损坏）。
                # 优先找真正的异常行（RuntimeError / TF 权重没能全部对上 / shape），
                # 找不到才退回「退出码 N」。以前直接把退出码当原因，页面上只看到
                # 一句「转换脚本退出码 1（退出码 1）」，等于没说。
                _all = [x for x in (_log_lines or []) if x.strip()]
                _kinds = ("tf 权重没能全部对上", "runtimeerror", "shape",
                          "mismatch", "无匹配", "候选", "[warn]", "error")
                _why = ""
                for _ln in reversed(_all):
                    if any(k in _ln.lower() for k in _kinds):
                        _why = _ln.strip()
                        break
                # 首选脚本自己落的错误文件（traceback 全在里面，不受 stdout 被吞的影响）
                _errf = md / "_convert.error.log"
                if _errf.exists():
                    try:
                        _tb = [x.strip() for x in
                               _errf.read_text(encoding="utf-8", errors="replace").splitlines()
                               if x.strip()]
                        if _tb:
                            _why = _tb[-1]
                            _prev = _tb[-2] if len(_tb) >= 2 else ""
                            if 'File "' in _prev:
                                _why = _prev + " → " + _why
                    except Exception:
                        pass
                if not _why:
                    # traceback 的最后一行通常是 `模块.异常: 说明`，比退出码有用得多
                    _tail2 = [x.strip() for x in _all if x.strip()][-2:]
                    _why = " / ".join(_tail2) if _tail2 else ("转换脚本退出码 %d" % rc)
                # 失败归到哪个模块：日志里出现 <模块>: 就归给它，否则归最后一步
                _who = None
                # 先从错误文本里认（traceback 那行就写着 `encoder: shape ...`）
                for _mm2 in modules:
                    if (_mm2["name"] + ":") in (_why or ""):
                        _who = _mm2["name"]
                        break
                if not _who:
                    for _ln in reversed(_all):
                        for _mm2 in modules:
                            if (_mm2["name"] + ":") in _ln:
                                _who = _mm2["name"]
                                break
                        if _who:
                            break
                if not _who:
                    _who = modules[-1]["name"] if modules else "转换"
                _detail = _why[:300]
                _marked = False
                if modules:
                    for _mm3 in modules:
                        if _mm3["name"] == _who:
                            _mm3["state"] = "bad"
                            _mm3["reason"] = _detail
                            _marked = True
                            break
                if not _marked and modules is not None:
                    # 不能用 modules = modules + [...] —— 那会让 Python 把 modules 当成
                    # worker 的局部变量，前面 _set(modules=modules) 直接 UnboundLocalError，
                    # 整段 error 处理全崩，页面上就是「失败」却一个字原因都没有。
                    modules.append({"name": _who, "state": "bad", "reason": _detail})
                # 失败也把已经写出来的文件列出来 —— 用户要知道「转了一半」时
                # 目录里到底多了什么，能不能直接用。
                _set(running=False, step="error", ended=time.time(),
                     error=_detail, modules=modules, outDir=str(md),
                     files=_scan_outputs(md, stem))
        except Exception as e:
            # 兜底：这里以前只写 error，不写 modules —— 前端拿到「全 wait」的模块表，
            # 页面上就是「失败」但四行全是「待转换」，一句原因都看不到。
            import traceback as _tb3
            _line = _tb3.format_exc().strip().splitlines()
            _last = (_line[-1] if _line else "%s: %s" % (type(e).__name__, e))[:300]
            _set(running=False, step="error", ended=time.time(), error=_last,
                 modules=[{"name": "转换", "state": "bad", "reason": _last}],
                 outDir=str(md), files=_scan_outputs(md, stem))

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "name": stem, "modules": modules}


def status() -> dict:
    with _LOCK:
        return json.loads(json.dumps(CONVERT_JOB))
