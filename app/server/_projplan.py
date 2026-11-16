"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_merge/proj_plan.py。

**惰性转发**：本模块自己不 import 真身 —— app.py ↔ routes_merge ↔ backend_merge.* 那条链
会立刻成环。改成 PEP 562 代理后，属性访问/赋值都落到真身模块上，所以模块级状态、猴子补丁、
私有名都不分叉。

为什么单独一个 proj_plan.py 而不是并进 vram.py：vram.py 由 routes_merge 的切片生成器维护，
重跑一次就会把追加进来的段落冲掉（实测已经冲掉三次）。自适应档位这块代码独立成文件，
生成器不碰，才不会再丢。

等价写法：from app.server.backend_merge.proj_plan import *。后端以 app/ 为 import 根
（serve.py 把 app/ 插进 sys.path[0]，包名是 server.*），绝对形式在那里取不到；相对写法在
server.* 与 app.server.* 两种根下都成立。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_merge.proj_plan"


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
