"""兼容层（DEPRECATED）：真身已搬到 app/server/backend_extract/scan_runner.py。

这个文件**是能被当脚本执行的**（4.0 的 python 跑数据集体检），所以除了把名字搬进来，
__main__ 分支必须把入口转给真身的 main()。真身的导入在"当脚本跑"和"当模块 import"
两种形态下都对（见那边的注释）。
注：routes 里现在指向的是真身的绝对路径，本文件只是旧路径的兜底。
"""
import importlib as _importlib
import sys as _sys


def _load_real():
    if __package__:                                           # 包内：server._scan_runner
        return _importlib.import_module(__package__ + ".backend_extract.scan_runner")
    from pathlib import Path                                  # 脚本方式：app/server 在 sys.path
    app_dir = str(Path(__file__).resolve().parents[1])        # app/
    if app_dir not in _sys.path:
        _sys.path.append(app_dir)
    return _importlib.import_module("server.backend_extract.scan_runner")


_impl = _load_real()
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
_sys.modules[__name__] = _impl

if __name__ == "__main__":                                    # python _scan_runner.py --dir ...
    raise SystemExit(_impl.main())
