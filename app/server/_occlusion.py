# -*- coding: utf-8 -*-
"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_mask/occlusion.py。

**惰性转发**：本模块自己不 import 真身，属性访问 / 赋值 / dir() 全落到真身模块上，
所以 scan / move_to_trash / write_manifest / jpeg_size 与私有名（_one_file / _stats /
_within / _mask_float）都不分叉，模块级状态和猴子补丁也不会只改到代理这一层。

等价写法：from app.server.backend_mask.occlusion import *（`*` 不带私有名，所以用代理）。
现有调用点 routes_oneshot.py 里 `from . import _occlusion as OCC` + `OCC.scan(root)` 照旧。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_mask.occlusion"


class _LazyModule(_types.ModuleType):
    """转发到真身模块：取属性 / 设属性 / dir() 全走真身。"""

    @staticmethod
    def _real():
        return _importlib.import_module(__package__ + "." + _TARGET)

    def __getattr__(self, name):
        return getattr(self._real(), name)

    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            return _types.ModuleType.__setattr__(self, name, value)
        setattr(self._real(), name, value)

    def __dir__(self):
        return dir(self._real())


_sys.modules[__name__].__class__ = _LazyModule
