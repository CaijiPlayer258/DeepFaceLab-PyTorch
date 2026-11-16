# -*- coding: utf-8 -*-
"""遮罩绘制：遮罩落盘/读回（_masks PNG + DFLJPG APP15）、SAM2/GroundedDINO 推理桥、遮挡率排序。

只做命名空间：**不要**在这里 import 子模块 —— painter 会建推理进程桥（threading/subprocess/
atexit），occlusion 会拖 numpy/cv2 进来，按需 import 就够。

routes.py（HTTP 层，薄）· painter.py（落盘/读回 + 推理进程桥）· occlusion.py（遮挡率排序器）
"""
