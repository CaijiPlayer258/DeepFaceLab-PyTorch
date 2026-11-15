# -*- coding: utf-8 -*-
"""进程引导：Windows 上给派生的每个子进程补 CREATE_NO_WINDOW。

从 app/server/app.py 顶部原样切出来的（_NO_WINDOW + _install_no_window + 调用点），
函数体 / 注释一字未改 —— 只是现在归 _common 所有：路径 / 任务 / 提取器各模块都从这里
取同一个常量（POSIX 上必须是 0，否则 subprocess 会 ValueError）。
"""
from __future__ import annotations

import os


# 4.1 是 GUI 应用：任何子进程只要弹一下黑色控制台窗口，用户都会以为出事了。
# Windows 上统一用 CREATE_NO_WINDOW（POSIX 下是 0，无副作用）。
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _install_no_window():
    """全局兜底：这个进程派生的任何子进程都不弹控制台窗口。

    以前只有显式传了 creationflags 的地方才安静，剩下十几处（nvidia-smi、
    ffmpeg、人脸身份聚类用的引擎 python…）都会闪一个黑框 —— 点一下「人脸身份」
    就弹窗，就是这么来的。这里统一打补丁，调用点不用再记得传参。
    """
    import subprocess as _sp

    if os.name != "nt" or getattr(_sp, "_dfl_nowindow", False):
        return
    _orig_run = _sp.run
    _orig_popen_init = _sp.Popen.__init__

    def run(*a, **kw):
        kw.setdefault("creationflags", _NO_WINDOW)
        return _orig_run(*a, **kw)

    def popen_init(self, *a, **kw):
        kw.setdefault("creationflags", _NO_WINDOW)
        return _orig_popen_init(self, *a, **kw)

    _sp.run = run
    _sp.Popen.__init__ = popen_init
    _sp._dfl_nowindow = True


_install_no_window()
