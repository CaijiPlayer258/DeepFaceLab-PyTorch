"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_extract/pipeline.py。

现有调用点：app.py 的抽取接口（pipeline_enabled / capacity / run）与 `/api/extract/status`
里的 current 结论；`__all__` 见真身。

**惰性转发**：本模块自己不 import 真身（真身在 import 期会拿锁和哨兵对象），
属性访问 / 赋值 / dir() 全落到真身模块上 —— 模块级状态、猴子补丁、私有名都不分叉成两份。

等价写法：from app.server.backend_extract.pipeline import *（`*` 不带私有名，所以用代理）。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_extract.pipeline"


def _real():
    """真身模块。

    两种 import 形态都要成立（_ortcuda.py 那个 shim 同一套写法）：
      · 包内：server.backend_extract / app.server.* —— __package__ 有值，相对取；
      · 顶层：app/server 被塞进 sys.path 后裸 import（tools/ 下的脚本、bench）——
        __package__ 是空串，这时把 app/ 追加进 sys.path 再按 server.* 取，
        拿到的是**同一个模块对象**，模块级状态不会分叉成两份。
    """
    if __package__:
        return _importlib.import_module(__package__ + "." + _TARGET)
    from pathlib import Path
    app_dir = str(Path(__file__).resolve().parents[1])          # app/
    if app_dir not in _sys.path:
        _sys.path.append(app_dir)                                # 追加，不插前面
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
