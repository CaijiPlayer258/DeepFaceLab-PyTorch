# -*- coding: utf-8 -*-
"""发动机侧的唯一兼容入口：把 landmark106to68 指向 **server 里的那一份**。

以前这里抄了一份 196 行的索引表（源头是 4.0 的 Extractor/Extractor.py:1171-1352），
和 `app/server/_landmarks106.py` 并存 —— 两份映射，谁改了另一边不知道。
用户决定：**只留 server 那一份**，引擎这边只做「按路径加载」。

两种加载方式（都不需要改 sys.path）：
  1) 若 `app/` 在 sys.path 上（引擎被 server 进程内调用时就是），直接 `from server._landmarks106 import ...`；
  2) 否则（训练子进程：只有 app/engine 在 sys.path 上）按文件路径 spec_from_file_location 加载
     `<repo>/app/server/_landmarks106.py` —— 它只 import numpy，没有别的依赖。

两份都失败就抛 ImportError（**不留第三份副本**）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL = None
_ERR = ""


def _load():
    """返回 server 侧的 module；失败时把原因留在 _ERR 里。"""
    global _IMPL, _ERR
    if _IMPL is not None:
        return _IMPL
    # ① 正常 import（app/ 在 sys.path 上）
    try:
        from server import _landmarks106 as mod          # noqa: WPS433
        _IMPL = mod
        return _IMPL
    except Exception as e:                                # noqa: BLE001
        _ERR = "import server._landmarks106 失败：%r" % (e,)
    # ② 按文件路径加载（训练子进程的 sys.path 只有 app/engine）
    here = Path(__file__).resolve()                       # app/engine/Extractor/_compat.py
    cand = here.parents[2] / "server" / "_landmarks106.py"   # app/server/_landmarks106.py
    try:
        spec = importlib.util.spec_from_file_location("_dfl_server_landmarks106", str(cand))
        if spec is None or spec.loader is None:
            raise FileNotFoundError(str(cand))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IMPL = mod
        return _IMPL
    except Exception as e:                                # noqa: BLE001
        _ERR += "；按路径加载 %s 也失败：%r" % (cand, e)
    raise ImportError("landmark106to68 找不到实现（server 侧 _landmarks106.py）：" + _ERR)


def landmark106to68(pt106):
    """106 点 → 68 点（索引映射）。实现在 app/server/_landmarks106.py，本函数只转发。"""
    return _load().landmark106to68(pt106)


def __getattr__(name):
    """其余名字（InsightFace106 之类）一并转发，避免以后有人再从这儿 import 别的东西。"""
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_load(), name)
