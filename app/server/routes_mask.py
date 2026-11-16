# -*- coding: utf-8 -*-
"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_mask/{routes,painter}.py。

**惰性转发**：本模块自己不 import 真身。app.py 里
    from .routes_mask import router as _mask_router
而真身 routes.py 顶部有 `from ..app import ...` —— 本文件要是直接
`from .backend_mask.routes import *`，就会在 app.py 还没走完时把真身拉进来（和拆分前
直接 `import server.routes_mask` 就炸 ImportError 是同一个环）。改成 PEP 562 代理后，
属性访问先问 backend_mask.routes，找不到再问 backend_mask.painter，所以
_mask_png_path / _decode_mask / _infer / _WORKER_SRC 这些私有名照旧摸得到，
模块级状态（_mask_state 在 app.py，_infer 在 painter）也不分叉。

等价写法：from app.server.backend_mask.routes import *（`*` 不带私有名，老调用点会静默
变成 AttributeError，所以这里只用代理）。后端以 app/ 为 import 根（serve.py 把 app/ 插进
sys.path[0]，包名是 server.*），绝对形式在那里取不到；相对写法在 server.* 与
app.server.* 两种根下都成立。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGETS = ("backend_mask.routes", "backend_mask.painter")   # 主真身 / 兜底（私有名的老家）


def _real_mods():
    pkg = __package__
    return tuple(_importlib.import_module(pkg + "." + t) for t in _TARGETS)


class _LazyModule(_types.ModuleType):
    """转发到真身模块：取属性 / 设属性 / dir() 全走真身（routes 优先，painter 兜底）。"""

    def __getattr__(self, name):
        miss = None
        for mod in _real_mods():
            try:
                return getattr(mod, name)
            except AttributeError as ex:
                miss = miss or ex
        raise miss

    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            return _types.ModuleType.__setattr__(self, name, value)
        mods = _real_mods()
        for mod in mods:
            if hasattr(mod, name):
                setattr(mod, name, value)
                return
        setattr(mods[0], name, value)

    def __dir__(self):
        names = set()
        for mod in _real_mods():
            names.update(dir(mod))
        return sorted(names)


_sys.modules[__name__].__class__ = _LazyModule
