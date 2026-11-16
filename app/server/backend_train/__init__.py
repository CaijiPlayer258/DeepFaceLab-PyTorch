# -*- coding: utf-8 -*-
"""训练器：训练控制路由、4.0 引擎启动包装、模块加载失败事件、错脸（ErrFace）数据集体检。

**不在 __init__ 里 import 子模块**：
  · routes / errface_routes 会把 fastapi 拖进启动路径；
  · dfl_train 在 **import 期**就会给 multiprocessing 打补丁（set_start_method 强制 spawn
    并换成空操作）——只是 `import server.backend_train` 的进程也会被改掉，绝不能顺手加载。
一律走 `from .子模块 import 名字`。
"""
