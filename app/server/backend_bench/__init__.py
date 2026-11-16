# -*- coding: utf-8 -*-
"""速度测试页的后端：POST /api/perf/bench（只读，不写任何文件）。

原来住在 backend_models/trt.py —— 那个文件当时是「什么都往里塞」的回收站。
只做命名空间：不 import 子模块（测量本身要 import cv2/numpy，别拖进启动路径）。
"""
