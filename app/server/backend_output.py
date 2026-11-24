"""「输出」页的后端：把 serve 进程的 stdout/stderr 回显给界面。

为什么要单独一页：用户遇到报错时手上只有截图，而 Electron 那份
_electron_backend.log 每次启动被 "w" 截断。这一页让用户自己就能把完整日志
截下来 / 复制出去，不用来回猜「你看到什么了」。
"""
from __future__ import annotations

from fastapi import APIRouter

from ._common import output_log

router = APIRouter()


@router.get("/api/output/tail")
def output_tail(since: int = 0, limit: int = 3000):
    """按**累计行号**增量取。since=0 表示「给我最后 limit 行」。

    返回的 total 是当前累计行号，前端把它存起来当下次的 since —— 环形缓冲淘汰
    旧行也不会错位，因为行号只增不减。
    """
    try:
        return output_log.tail(int(since or 0), int(limit or 3000))
    except Exception as e:                                  # noqa: BLE001
        return {"total": 0, "lines": [], "error": str(e)[:160]}


@router.post("/api/output/clear")
def output_clear():
    """只清内存里的缓冲（文件不动）—— 前端「清屏」按钮用。"""
    try:
        output_log.clear()
        return {"ok": True}
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "error": str(e)[:160]}
