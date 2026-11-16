# -*- coding: utf-8 -*-
"""引擎编译：TensorRT 引擎的**状态查询**与**编译**（一个独立界面 = 一个独立包）。

原来这两条路由住在 backend_models/trt.py（那个文件还兼着 /api/devices、/api/extract/*、
/api/perf/bench 等别的东西），而前端的「引擎编译页」要的是一个只属于它的入口。
只做命名空间：**不要**在这里 import 子模块（会把 onnxruntime/tensorrt 拖进启动路径）。
"""
