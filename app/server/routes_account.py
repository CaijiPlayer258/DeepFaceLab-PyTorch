"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_bili/account.py。

PEP 562 惰性代理：本模块自己不 import 真身，属性访问时才转发。这里必须惰性 ——
真身要 `from ..app import (WORKSPACE, app)`，而 app.py 在模块末尾才 include 本路由；
静态 re-export 会在 app 半初始化时就把 router 取走并炸在 import 期。

注意：转发名没有静态绑定，**对这个旧路径星号 import 拿不到东西** —— 请写
`from .routes_account import 具体名字` 或 `from . import routes_account` 再用属性。

_real() 对空 __package__ 有兜底：`import server.routes_account`（app.py 的取法）和
`import routes_account`（app/server 直接在 sys.path 上）两种根都能落到真身。
"""
import importlib as _importlib

_TARGET = "backend_bili.account"


def _real():
    pkg = __package__ or ""
    return _importlib.import_module(pkg + "." + _TARGET if pkg else _TARGET)


def __getattr__(name):
    return getattr(_real(), name)
