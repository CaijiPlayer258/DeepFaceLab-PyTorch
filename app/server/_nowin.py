"""兼容层（DEPRECATED）：真身已搬到 app/server/_common/nowin.py。

**不能惰性**：_merge_runner.py 是**被当脚本跑**的（4.0 python 直接执行），它那行是
`from _nowin import install as _install_nowin` —— 没有包上下文，惰性代理拼不出口号。
所以这里直接把真身的名字搬进来（真身只用 sys，没有相对 import），再把 sys.modules
指到真身，包内 import 拿到的仍是同一个模块对象。
"""
import importlib as _importlib
import sys as _sys


def _load_real():
    if __package__:                                           # 包内：server._nowin
        return _importlib.import_module(__package__ + "._common.nowin")
    from pathlib import Path                                  # 脚本方式：app/server 在 sys.path
    app_dir = str(Path(__file__).resolve().parents[1])        # app/
    if app_dir not in _sys.path:
        _sys.path.append(app_dir)
    return _importlib.import_module("server._common.nowin")


_impl = _load_real()
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
_sys.modules[__name__] = _impl
