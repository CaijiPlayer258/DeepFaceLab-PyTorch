"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_models/trt_build.py。

这个文件**是被当脚本执行的**（routes_trt 的编译线程用 sys.executable 起子进程，
命令行里写的就是这个路径；app/docs 里的手动命令也指这里），所以除了包内 import 的
PEP 562 惰性转发，__main__ 分支必须把入口转给真身 —— 否则 `python _trt_build.py --onnx …`
会因为没有包上下文、做不了相对 import 而直接炸。
"""
import sys as _sys
import types as _types

_TARGET = "backend_models.trt_build"

if __name__ == "__main__":
    # 当脚本跑：按文件路径加载真身（它只用标准库，不需要包上下文），再交给它的 main()
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _impl = _Path(__file__).resolve().parent / "backend_models" / "trt_build.py"
    _spec = _ilu.spec_from_file_location("_trt_build_impl", str(_impl))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _sys.exit(_mod.main())
else:
    import importlib as _importlib

    def _real():
        return _importlib.import_module(__package__ + "." + _TARGET)

    class _LazyModule(_types.ModuleType):
        """转发到真身模块：取属性 / 设属性 / dir() 全走真身。"""

        @staticmethod
        def _real():
            return _real()

        def __getattr__(self, name):
            return getattr(self._real(), name)

        def __setattr__(self, name, value):
            if name.startswith("__") and name.endswith("__"):
                return _types.ModuleType.__setattr__(self, name, value)
            setattr(self._real(), name, value)

        def __dir__(self):
            return dir(self._real())

    _sys.modules[__name__].__class__ = _LazyModule
