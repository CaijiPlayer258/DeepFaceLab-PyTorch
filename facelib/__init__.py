from .FaceType import FaceType
from .S3FDExtractor import S3FDExtractor
from .FANExtractor import FANExtractor
from .FaceEnhancer import FaceEnhancer
from .XSegNet import XSegNet

# ⚠️ FaceDetector 必须【惰性导入】，不能写在这里。
#    它是 facelib 里唯一在模块顶部 import xlib.onnxruntime 的模块，而 onnxruntime
#    带原生扩展：在 Qt 已经加载过 DLL 之后再 import 会以
#        WinError 1114 动态链接库(DLL)初始化例程失败
#    直接崩（和 torch / c10.dll 是同一个机制，已实测复现）。
#    后果是「任何 import facelib 的东西都会被拽进 onnxruntime」——
#    例如 FacesetProcessor.Filter / Sorter 只想要 facelib.LandmarksProcessor，
#    在 GUI 里一 import 就整个崩掉。
#    惰性化之后：需要 FaceDetector 的代码照旧写 from facelib import FaceDetector，
#    不需要的（清晰度排序/筛选、DFLJPG）不再为它付出代价。
_FaceDetector = None


def __getattr__(name):
    """PEP 562：按需加载 FaceDetector，其余名字照常抛 AttributeError。"""
    global _FaceDetector
    if name == 'FaceDetector':
        if _FaceDetector is None:
            from .FaceDetector import FaceDetector as _FD
            _FaceDetector = _FD
        return _FaceDetector
    raise AttributeError("module %r has no attribute %r" % (__name__, name))