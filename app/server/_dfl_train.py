# -*- coding: utf-8 -*-
"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_train/dfl_train.py。

**惰性转发**：本模块自己不 import 真身。它还被人当**脚本**跑（见下），import 期又会立刻给 multiprocessing / webbrowser 打补丁；
惰性代理能保证「谁 import 谁才吃补丁」。改成 PEP 562 代理后，属性访问/赋值都落到真身
模块上，所以模块级状态、猴子补丁、私有名都不分叉。

等价写法：from app.server.backend_train/dfl_train.py import *。后端以 app/ 为 import 根
（serve.py 把 app/ 插进 sys.path[0]，包名是 server.*），绝对形式在那里取不到；相对写法在
server.* 与 app.server.* 两种根下都成立。
"""
import importlib as _importlib
import sys as _sys
import types as _types

_TARGET = "backend_train.dfl_train"


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


if __name__ == "__main__":
    # ⚠ 本文件是**可执行入口**，不能只做代理：app.py 有两处硬编码指着它 ——
    #   · _train_worker(): boot = Path(__file__).parent / "_dfl_train.py"，有就拿来当
    #     `python <boot> train --...` 跑（走不通会退化成裸跑 4.0 main.py，WebUI 端口不同）；
    #   · _sweep_orphan_trainers(): 按命令行里含 "_dfl_train.py" 认遗留训练进程。
    # 所以被当脚本跑时转给真身的 main()，被 import 时才是惰性代理。
    # sys.path **追加** app/（不插前面）：只为让 `server` 这个包名可导入，
    # 不抢 4.0 引擎里同名的模块（engine / tools / web）。
    import os as _os

    _PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _PKG_ROOT not in _sys.path:
        _sys.path.append(_PKG_ROOT)
    from server.backend_train.dfl_train import main as _real_main

    raise SystemExit(_real_main())
else:
    _sys.modules[__name__].__class__ = _LazyModule
