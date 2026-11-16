"""兼容层（DEPRECATED）：真身已搬到 app/server/_common/engine_paths.py。

全仓**三种**加载形态都要活，所以这里用 PEP 562 的模块级 `__getattr__`（惰性，且不要求
本模块已经登记在 sys.modules 里），而不是改 `__class__` 的代理：
  1) 包内：`from ._paths import engine_root`（app.py、backend_* 一路，包名 server.*）
  2) 顶层：app/server 在 sys.path 时 `from _paths import engine_root`
     （backend_train/dfl_train.py 的 except 分支、tools/ 下的命令行脚本）
  3) 按文件 exec：app/tools/accept/merge_accept.py 用 spec_from_file_location 读 ROOT ——
     它没把模块注册进 sys.modules，改 `__class__` 的代理在那儿会 KeyError，模块级
     `__getattr__` 不受影响。
惰性还有一层必要：`_common/paths.py`（界面/工作区/settings 常量那个模块）反过来要
`from .._paths import engine_python, engine_root`，真身若在 import 期再拉回来就成环。

（名字说明：`_common/paths.py` 已被"界面/工作区常量"占走，引擎根解析落在
engine_paths.py；两者不是一回事。）
"""
import importlib as _importlib
import sys as _sys

_TARGET = "_common.engine_paths"


def _real():
    """真身模块。包内相对取；没有包上下文时把 app/ 追加进 sys.path 再按 server.* 取。"""
    if __package__:
        return _importlib.import_module(__package__ + "." + _TARGET)
    from pathlib import Path
    app_dir = str(Path(__file__).resolve().parents[1])        # app/
    if app_dir not in _sys.path:
        _sys.path.append(app_dir)                             # 追加，不插前面：别抢 4.0 引擎里的同名模块
    return _importlib.import_module("server." + _TARGET)


def __getattr__(name):                                        # PEP 562
    return getattr(_real(), name)


def __dir__():
    return dir(_real())
