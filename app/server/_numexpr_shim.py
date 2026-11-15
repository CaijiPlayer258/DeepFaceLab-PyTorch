"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_models/shim.py。

**这个被当模块对象用**：backend_merge/compose.py 拿它 `sys.modules["numexpr"] = _numexpr_shim` 顶替真的
numexpr。转发类照样成立 —— ne.evaluate / ne.set_num_threads 都落到真身上。
"""

import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_models.shim"


def _real():
    return _importlib.import_module(__package__ + "." + _TARGET)


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
