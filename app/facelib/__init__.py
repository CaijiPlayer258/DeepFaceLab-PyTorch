from .FaceType import FaceType

# ⚠️ 下面这些名字都【惰性导入】，不要写回模块顶部。
#
# 两种代价，都踩过：
#  1. S3FDExtractor / FANExtractor / FaceEnhancer / XSegNet 都会 import torch。
#     任何一句 `from facelib import FaceType`（比如 DFLIMG/DFLJPG.py）都会把
#     474 MB 的 torch 拖进来 —— 实测子进程光 import 就 +1.5s、+430MB。
#     SampleLoaderV4 的预取子进程只要 LandmarksProcessor 的一个角度函数，
#     却为此付了整份 torch。
#  2. FaceDetector 是唯一在顶部 import xlib.onnxruntime 的模块，onnxruntime
#     带原生扩展：Qt 先加载过 DLL 之后再 import 会以
#        WinError 1114 动态链接库(DLL)初始化例程失败
#     直接崩。FacesetProcessor.Filter / Sorter 只想要 LandmarksProcessor，
#     在 GUI 里一 import facelib 就整个崩掉。
#
# 惰性化之后，调用方写法不变：from facelib import XSegNet 照旧能用，
# 只是真正需要的那一刻才加载。FaceType 是纯枚举，保持立即导入。
_LAZY_NAMES = {
    'S3FDExtractor': '.S3FDExtractor',
    'FANExtractor': '.FANExtractor',
    'FaceEnhancer': '.FaceEnhancer',
    'XSegNet': '.XSegNet',
    'FaceDetector': '.FaceDetector',
}

_loaded = {}


def __getattr__(name):
    """PEP 562：按需加载上面那几个重模块，其余名字照常抛 AttributeError。"""
    mod_path = _LAZY_NAMES.get(name)
    if mod_path is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    obj = _loaded.get(name)
    if obj is None:
        import importlib
        obj = getattr(importlib.import_module(mod_path, __name__), name)
        _loaded[name] = obj
    return obj
