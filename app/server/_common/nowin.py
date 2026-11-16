"""让 4.1 拉起的整棵进程树都不弹控制台窗口。

背景：4.1 是 GUI（Electron 壳，父进程没有控制台）。Windows 上从"没有控制台"的
进程里 CreateProcess 一个控制台程序（python.exe / ffmpeg.exe），系统会给它**新建**
一个黑框 —— 用户看到的就是一闪而过的 cmd。

我们的直接子进程都加了 CREATE_NO_WINDOW，但再往下一层就管不到了：
  · SampleLoaderV4 的预取进程（multiprocessing spawn）
  · 合成管线里的 ffmpeg
它们由 CPython 自己 CreateProcess，不看我们的参数。

这里直接包住 _winapi.CreateProcess（subprocess 和 multiprocessing/popen_spawn_win32
都走它），给所有子进程补上 CREATE_NO_WINDOW。父进程已有的控制台照用，所以
"在终端里直接跑"不受影响 —— 只是不再**新建**窗口。

install() 幂等；非 Windows 直接返回 False。
"""
from __future__ import annotations

import sys

CREATE_NO_WINDOW = 0x08000000


def install() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import _winapi
    except Exception:
        return False
    if getattr(_winapi, "_dfl_nowin", False):
        return True

    _orig = _winapi.CreateProcess

    def _patched(app_name, cmd_line, proc_attrs=None, thread_attrs=None,
                 inherit_handles=False, creation_flags=0, env=None,
                 current_directory=None, startup_info=None):
        try:
            creation_flags = int(creation_flags) | CREATE_NO_WINDOW
        except Exception:
            creation_flags = CREATE_NO_WINDOW
        return _orig(app_name, cmd_line, proc_attrs, thread_attrs,
                     inherit_handles, creation_flags, env,
                     current_directory, startup_info)

    _winapi.CreateProcess = _patched
    _winapi._dfl_nowin = True
    return True
