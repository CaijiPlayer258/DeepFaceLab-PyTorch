# -*- coding: utf-8 -*-
"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_train/events.py。

**惰性转发**：本模块自己不 import 真身。app.py 顶层 `from . import _load_events`，训练进程 stdout 每来一行就要 `parse_line`；
保持惰性代理可让「只 import 单体解析函数」的单测不必连带拉起事件收集那一套。改成 PEP 562 代理后，属性访问/赋值都落到真身
模块上，所以模块级状态、猴子补丁、私有名都不分叉。

等价写法：from app.server.backend_train/events.py import *。后端以 app/ 为 import 根
（serve.py 把 app/ 插进 sys.path[0]，包名是 server.*），绝对形式在那里取不到；相对写法在
server.* 与 app.server.* 两种根下都成立。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_train.events"


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
