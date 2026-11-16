"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_faces/faceid.py。

这一条**不能惰性**：python -m server._faceid_cli --serve 是活跃调用路径（routes_oneshot 起的
常驻引擎进程），__main__ 分支必须把 main() 转发过去。它只在子进程里被加载，不在 app.py 的
import 链上，所以没有成环风险。
"""
from .backend_faces.faceid import *            # noqa: F401,F403  —— 旧路径 import 不炸
import sys as _sys                   # noqa: E402
from .backend_faces import faceid as _impl     # noqa: E402,F401

if __name__ == "__main__":            # python -m server._faceid_cli --serve
    raise SystemExit(main())

_sys.modules[__name__] = _impl        # 旧路径 = 新模块本体
