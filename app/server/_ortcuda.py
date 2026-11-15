"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_models/ortcuda.py。

**额外要照顾上游的裸 import**：app/tools/bench/dfm_*.py 只把 app/server 塞进 sys.path 再
`from _ortcuda import add_cuda_dll_dirs` —— 那种形态下 __package__ 是空串，所以 _real()
还要能按 server.* 把真身取回来（见下面注释）。
"""

import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_models.ortcuda"


def _real():
    """真身模块。

    两种 import 形态都要成立：
      · 包内：server._ortcuda / app.server._ortcuda —— __package__ 有值，相对取；
      · 顶层：app/tools/bench/dfm_*.py 把 app/server 塞进 sys.path 后裸 import _ortcuda
        —— __package__ 是空串，这时把 app/ 追加进 sys.path 再按 server.* 取，
        拿到的是**同一个模块对象**（_CACHE 不会分叉成两份）。
    """
    if __package__:
        return _importlib.import_module(__package__ + "." + _TARGET)
    from pathlib import Path
    app_dir = str(Path(__file__).resolve().parents[1])          # app/
    if app_dir not in _sys.path:
        _sys.path.append(app_dir)
    return _importlib.import_module("server." + _TARGET)


class _LazyModule(_types.ModuleType):
    """转发到真身模块：取属性 / 设属性 / dir() 全走真身。"""

    @staticmethod
    def _real():
        return _real()

    def __getattr__(self, name):
        return getattr(self._real(), name)

    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            return _types.ModuleType.__setattr__(self, name, value)
        setattr(self._real(), name, value)

    def __dir__(self):
        return dir(self._real())


_sys.modules[__name__].__class__ = _LazyModule
