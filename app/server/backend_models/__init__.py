"""模型层：TensorRT 引擎缓存、旧版 dat 读取、TF→torch 转换任务。

**不在 __init__ 里 import 子模块**：会把 cv2/torch 拖进启动路径，
而且 backend_merge 里 vram / pipeline_project 有同名公开名（enabled / candidates），
星号 re-export 会让其中一个静默覆盖另一个 —— 一律走 `from .子模块 import 名字`。
"""
