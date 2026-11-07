# -*- coding: utf-8 -*-
"""服务器侧读 ``*_data.dat`` 的唯一入口：4.1 的 dat 原样通过，TF 时代的就地翻译。

为什么要有这一层
----------------
界面只认 4.1 的 options 键名。iperov/TF 时代（DFL 1.x / 2.0）的 dat 里是
``learn_mask`` / ``ed_ch_dims`` / ``face_type='whole_face'`` / ``archi='dfhd'`` 那一套，
裸 pickle 读出来直接喂给参数页的后果是：一半输入框空着、face_type 下拉框空白，
点「保存」还会把老键原样写回去。翻译规则（哪个键哪年改的、缺的键缺省多少）只有
一份出处：``app/engine/core/legacy_dat.py``。这里不复制任何一张映射表，只做三件事：

  1. 惰性 import 那个模块（路径问 ``_paths.engine_root()``，和 serve.py 同源，不硬编码）；
  2. 按 ``(路径, mtime_ns, size)`` 缓存结果 —— 读 dat 是 ms 级，但「用 AST 从
     Model_pytorch.py 抽默认值」不该每个请求跑一遍，文件一动整条缓存失效；
  3. 给界面拼一条能直接复制的转权重命令。

两条硬约束
----------
* server 的解释器里没有 torch、可能也没有 numpy。``legacy_dat`` 顶层只 import
  ast / pickle / pathlib（numpy 是惰性 import），所以能直接拿来用。
* 新格式 dat 必须零影响：``detect_legacy`` 说「不是老 dat」时，连键序都不动。
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
import threading
from pathlib import Path

_LOCK = threading.RLock()
_MOD = None
_MOD_TRIED = False

# (路径, mtime_ns, size) → 迁移结果。只存 options/report 这类小字典（几 KB），
# 不存整份 dat —— 122MB 的模型缓存不得。
_CACHE = {}
_CACHE_MAX = 64


def _engine_module():
    """惰性 import ``core.legacy_dat``；拿不到就返回 None（退化成改动前的行为）。"""
    global _MOD, _MOD_TRIED
    if _MOD is not None or _MOD_TRIED:
        return _MOD
    with _LOCK:
        if _MOD is not None or _MOD_TRIED:
            return _MOD
        _MOD_TRIED = True
        roots = []
        try:
            from ._paths import engine_root
            roots.append(Path(engine_root()))
        except Exception:
            pass
        # 兜底：和 server/ 同级的 engine/（engine_root() 找不到时也不至于傻掉）
        roots.append(Path(__file__).resolve().parents[1] / "engine")
        for root in roots:
            if (root / "core" / "legacy_dat.py").is_file():
                s = str(root)
                if s not in sys.path:
                    sys.path.append(s)   # append，绝不插队抢 server 自己的同名模块
                break
        try:
            _MOD = importlib.import_module("core.legacy_dat")
        except Exception:
            _MOD = None
    return _MOD


def _stat_key(p: Path):
    st = p.stat()
    try:
        real = str(p.resolve())
    except OSError:
        real = str(p)
    return (os.path.normcase(real), st.st_mtime_ns, st.st_size)


def _remember(key, payload):
    with _LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            # 够用就行：早期条目先走，反正文件没变的话重算一次也就一遍 pickle
            for old in list(_CACHE)[: _CACHE_MAX // 2]:
                _CACHE.pop(old, None)
        _CACHE[key] = payload


def _norm_top(data):
    """dat 顶层键也可能是 py2 的 bytes（'iter' → b'iter'）。返回 (归一后的 dict, 改了几个)。"""
    if not any(isinstance(k, (bytes, bytearray)) for k in data):
        return data, 0
    out, n = {}, 0
    for k, v in data.items():
        if isinstance(k, (bytes, bytearray)):
            out[bytes(k).decode("utf-8", "replace")] = v
            n += 1
        else:
            out[k] = v
    return out, n


def _pickle_load(p: Path, mod):
    if mod is not None:
        # 先按 py3 读；py2 写的（str 里带非 ASCII）由它退回 encoding='bytes'
        return mod.read_dat(p)
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _as_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _norm_data(raw_data):
    """顶层归一：py2 的 bytes 键 → str；DFL 1.x 只存 epoch → 补一个 iter。

    返回 (dict, 被换掉的 bytes 键个数)。
    """
    data, top_bytes = _norm_top(raw_data)
    if "iter" not in data and "epoch" in data:
        # 引擎那边是 iter = max(iter, epoch)；界面显示的就是 iter，不补会写成「0 次迭代」
        data["iter"] = data.get("epoch")
    return data, top_bytes


def _load(p: Path):
    """一次 pickle，同时给出 (完整 dat, 迁移结果)。"""
    mod = _engine_module()
    data = _pickle_load(p, mod)
    if not isinstance(data, dict):
        raise ValueError("%s 顶层不是 dict（%s），不像 DFL 的 data.dat"
                         % (p, type(data).__name__))
    data, top_bytes = _norm_data(data)
    raw = data.get("options")
    if not isinstance(raw, dict):
        raw = {}

    info = {
        "detected": False,
        "source": "pickle" if mod is None else "modern",
        "reasons": [],
        "report": [],
        "counts": {},
        "options": dict(raw),
        "iter": _as_int(data.get("iter", data.get("epoch"))),
        "dat": str(p),
    }
    if mod is None:
        return data, info

    reasons = mod.detect_legacy(raw)
    if not reasons:
        return data, info            # 4.1 自己的 dat：一个字都不改

    try:
        options, report = mod.migrate_options(raw, model_py=mod.model_py_for(p))
    except Exception as e:
        info["reasons"] = list(reasons) + ["迁移失败：%s" % e]
        return data, info

    if top_bytes:
        report.insert(0, "[迁移] dat 顶层有 %d 个键是 py2 的 bytes（如 b'iter'），已归一"
                      % top_bytes)
    info.update({
        "detected": True,
        "source": "legacy",
        "reasons": list(reasons),
        "report": list(report),
        "counts": mod.report_counts(report),
        "options": options,
    })
    return data, info


def describe(path):
    """只要有 options / 迁移报告（带缓存）。参数页、模型详情用这个。"""
    p = Path(path)
    key = _stat_key(p)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    _, info = _load(p)
    _remember(key, info)
    return info


def read_dat(path):
    """(完整 dat, 迁移信息)；dat['options'] 已经是 4.1 的形状。

    预览图 / loss 曲线要整份 dat，所以命中缓存也照样 pickle 一遍 —— 省掉的是
    AST 抽默认值和映射表那部分，不是磁盘读。
    """
    p = Path(path)
    key = _stat_key(p)
    hit = _CACHE.get(key)
    if hit is not None:
        data = _pickle_load(p, _engine_module())
        if isinstance(data, dict):
            data, _ = _norm_data(data)
            data["options"] = dict(hit["options"])
            return data, hit
        return {}, hit
    data, info = _load(p)
    _remember(key, info)
    if info["detected"]:
        data["options"] = dict(info["options"])
    return data, info


# --------------------------------------------------------------------------
#  权重现状 + 可复制的转权重命令
# --------------------------------------------------------------------------
def _dat_stem(dat_path: Path) -> str:
    name = dat_path.name
    return name[: -len("_data.dat")] if name.endswith("_data.dat") else dat_path.stem


def weights_of(dat_path: Path):
    """TF 的 .npy 和 4.1 的 .pth 各有多少（只数文件名，不打开）。"""
    p = Path(dat_path)
    stem = _dat_stem(p)
    try:
        pth = sorted(p.parent.glob(stem + "_*.pth"))
        npy = sorted(p.parent.glob(stem + "_*.npy"))
    except OSError:
        pth, npy = [], []
    return {"torch": len(pth), "tf": len(npy),
            "pth": [f.name for f in pth[:6]], "npy": [f.name for f in npy[:6]]}


def _tool_and_python():
    repo = Path(__file__).resolve().parents[2]
    tool = repo / "app" / "tools" / "import_legacy_dat.py"
    try:
        from ._paths import engine_python
        py = Path(engine_python())
    except Exception:
        py = Path(sys.executable)
    return tool, py


def convert_command(dat_path) -> str:
    """两行、路径全填好、能直接粘贴的两条命令。

    import_legacy_dat.py 不允许 --out 和 --dat 是同一个文件（它不覆盖原始 dat），
    而 --convert 又要求 --out 的文件名和原 dat 一致 —— 所以先把 dat 拷进
    .legacy_src/ 再就地写回模型目录：原文件在 .legacy_src 里留了一份底。
    """
    p = Path(dat_path)
    tool, py = _tool_and_python()
    stash = p.parent / ".legacy_src"
    src = stash / p.name
    if os.name == "nt":
        copy = 'mkdir "%s" ; Copy-Item "%s" "%s"' % (stash, p, src)
    else:
        copy = 'mkdir -p "%s" && cp "%s" "%s"' % (stash, p, src)
    run = '"%s" "%s" --dat "%s" --weights-dir "%s" --apply --out "%s" --convert' % (
        py, tool, src, p.parent, p)
    return copy + "\n" + run


def notice(dat_path):
    """给界面的一条 JSON：检测结果 + 计数 + 报告 + 权重现状 + 可复制命令。"""
    p = Path(dat_path)
    info = describe(p)
    w = weights_of(p)
    needs = (w["torch"] == 0) and (info["detected"] or w["tf"] > 0)

    hint, note = "", ""
    if needs:
        if info["detected"]:
            hint = ("该 dat 来自 TF 版本，需用 app/tools/import_legacy_dat.py --convert "
                    "转换权重")
        else:
            hint = ("这份 dat 是 4.1 格式，但目录里的权重还是 TF 的 *.npy，"
                    "需用 app/tools/import_legacy_dat.py --convert 转换权重")
        if w["tf"] == 0:
            note = "目录里连 TF 权重（*.npy）都没有：这份 dat 只有参数，训练会从零开始。"

    return {
        "detected": bool(info["detected"]),
        "source": info["source"],
        "counts": info["counts"],
        "report": info["report"],
        "reasons": info["reasons"],
        "iter": info["iter"],
        "weights": w,
        "needs_convert": bool(needs),
        "hint": hint,
        "note": note,
        "command": convert_command(p) if needs else "",
        "dat": str(p),
    }


def forget(path=None):
    """测试/调试用：清掉缓存（path=None 全清）。"""
    with _LOCK:
        if path is None:
            _CACHE.clear()
        else:
            try:
                key = _stat_key(Path(path))
            except OSError:
                return
            _CACHE.pop(key, None)
