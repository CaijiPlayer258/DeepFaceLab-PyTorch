# -*- coding: utf-8 -*-
"""进程小工具：_kill_proc_tree（按 pid 杀整棵树）、_proc_mem_mb（读本进程 RSS）。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import sys


def _kill_proc_tree(proc) -> bool:
    """停掉合成子进程和它拉起的 ffmpeg（Windows 上 terminate 只杀父进程）。"""
    if proc is None:
        return False
    try:
        if proc.poll() is not None:
            return False
    except Exception:
        return False
    try:
        import psutil
        children = []
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            children = []
        for c in children:
            try:
                c.terminate()
            except Exception:
                pass
        try:
            proc.terminate()
        except Exception:
            pass
        for c in children:
            try:
                c.wait(timeout=3)
            except Exception:
                try:
                    c.kill()
                except Exception:
                    pass
        return True
    except Exception:
        try:
            proc.terminate()
            return True
        except Exception:
            return False


def _proc_mem_mb() -> dict:
    """本进程内存（MB）：rss = 工作集（任务管理器「内存」那一列）·
    private = 提交（「已提交」那一列）· peak_* = 峰值。

    为什么不用 psutil：后端不是到处都装了它，而 psapi 就在 kernel32 旁边 ——
    和任务管理器同源，打进日志的数字能和用户截图对上（不然「日志 800MB、任务管理器
    1.3GB」又是一轮谁也说不清的对话）。

    两个数说的不是一回事：**提交**包含压到页面文件里的、以及 WDDM 上由显存兜着的
    那部分；**工作集**才是当下真占着物理内存的量。用户看到的「内存 13/15.2」是
    工作集口径，「已提交 37/129」是提交口径 —— 拿提交去喊内存泄漏是错的。
    """
    try:
        import ctypes
        if sys.platform.startswith("win"):
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            k32 = ctypes.WinDLL("kernel32.dll")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.WinDLL("psapi.dll")
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()),
                                          ctypes.byref(pmc), ctypes.sizeof(pmc)):
                return {"rss": round(pmc.WorkingSetSize / 1048576.0, 1),
                        "private": round(pmc.PagefileUsage / 1048576.0, 1),
                        "peak_rss": round(pmc.PeakWorkingSetSize / 1048576.0, 1),
                        "peak_private": round(pmc.PeakPagefileUsage / 1048576.0, 1)}
    except Exception:                                       # noqa: BLE001
        pass
    try:                                          # 非 Windows：只有 rss 这一列
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int((f.read() or "0 0").split()[1])
        return {"rss": round(pages * 4096 / 1048576.0, 1), "private": None,
                "peak_rss": None, "peak_private": None}
    except Exception:                                       # noqa: BLE001
        return {}
