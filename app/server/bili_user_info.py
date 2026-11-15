"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_bili/user.py。

PEP 562 惰性代理：本模块自己不 import 真身，属性访问时才转发。为什么不用
`from .backend_bili.user import *`：星号只复制当下的绑定（本模块没有可变态，代理只为统一写法）
会分成两份，一边改了另一边看不见；走代理则属性、异常类、模块状态全是同一个对象，
isinstance 与 is 判定都还成立。

注意：转发名没有静态绑定，**对这个旧路径星号 import 拿不到东西** —— 请写
`from .bili_user_info import 具体名字` 或 `from . import bili_user_info` 再用属性。

_real() 对空 __package__ 有兜底：`import server.bili_user_info`（serve.py 那种根）和
`import bili_user_info`（app/server 直接在 sys.path 上）两种取法都能落到真身。
"""
import importlib as _importlib

_TARGET = "backend_bili.user"


def _real():
    pkg = __package__ or ""
    return _importlib.import_module(pkg + "." + _TARGET if pkg else _TARGET)


def __getattr__(name):
    return getattr(_real(), name)
