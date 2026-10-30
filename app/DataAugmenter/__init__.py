"""
DataAugmenter — 数据增强器模块

当前功能:
- XSegAugmenter: XSeg/XSegLite ONNX 批量遮罩应用
"""

__version__ = "1.0.0"

from .XSegAugmenter import XSegAugmenter

__all__ = ["XSegAugmenter"]
