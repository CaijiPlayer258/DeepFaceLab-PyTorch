"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_oneshot/routes.py。

四个 `/api/dataset/*` 系列路由（oneshot / scan / apply / restore / sort / occlusion / thumb /
faceid 常驻进程）都在真身里，模块级状态（_oneshot_job、_faceid_proc）也只在那里。
router 就在真身模块里定义：app.py 末尾那串 from .routes_oneshot import router 仍然拿得到（旧路径不炸就是这条）。
"""

import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_oneshot.routes"


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
