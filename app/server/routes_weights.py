"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_models/weights.py。

router / ROOT / _state 都在真身模块里（ROOT 在真身里已按新的目录深度改成 parents[3]）。
"""

import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_models.weights"


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
