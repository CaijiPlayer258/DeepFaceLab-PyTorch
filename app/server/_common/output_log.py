"""把后端的 stdout / stderr 留住，供界面「输出」页回显。

为什么要它：训练/提取/合成出问题时，用户手上只有一张截图，而我们能看到的
只有 Electron 写的那份 `_electron_backend.log`（而且 desktop/main.js:29 每次
启动都用 "w" 截断）。于是在「你说有问题、我看不到日志」之间来回猜。

这里在进程内装一个 tee：
  · 内存里留最近 N 行（环形），`/api/output/tail` 按行号增量取；
  · 同时追加到 `_server_output.log`（超过 MAX_FILE 就轮转一次），重启也能查。

**不能再往 stdout 写任何东西** —— write() 里 print 会无限递归。所有意外都吞掉：
日志采集本身绝不能影响正常功能。
"""
from __future__ import annotations

import collections
import io
import os
import sys
import threading
import time
from pathlib import Path

MAX_LINES = 6000          # 内存里留多少行
MAX_FILE = 8 * 1024 * 1024  # 落盘超过这个大小就轮转
_FILE = None
_lock = threading.Lock()
_buf: "collections.deque" = collections.deque(maxlen=MAX_LINES)
_seq = 0                  # 累计行号（环形淘汰后仍然连续，前端靠它做增量）
_installed = False


def _path() -> Path:
    global _FILE
    if _FILE is None:
        root = Path(__file__).resolve().parents[2]      # app/
        _FILE = root.parent / "_server_output.log"      # 项目根，和 _electron_backend.log 并排
    return _FILE


def _append_file(text: str) -> None:
    p = _path()
    try:
        if p.exists() and p.stat().st_size > MAX_FILE:
            os.replace(str(p), str(p) + ".1")
    except Exception:
        pass
    try:
        with open(p, "a", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except Exception:
        pass


class _Tee(io.TextIOBase):
    """写一份到原流，同时记进环形缓冲 + 落盘。"""

    def __init__(self, stream):
        self._s = stream

    def write(self, s):                                     # noqa: D401
        try:
            if self._s is not None:
                self._s.write(s)
        except Exception:
            pass
        if not s:
            return 0
        global _seq
        try:
            # 只按 \n 断行：4.0 大量用 print(end='\r') 原地刷新，splitlines() 会把
            # 那些 \r 也当换行，界面上一半是空行。行尾的 \r 顺手剥掉。
            lines = [ln.rstrip("\r") for ln in s.split("\n")]
            while lines and lines[-1] == "":
                lines.pop()
            if not lines:
                return len(s)
            with _lock:
                for ln in lines:
                    _seq += 1
                    _buf.append((_seq, ln))
            _append_file(s if s.endswith("\n") else s + "\n")
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            if self._s is not None:
                self._s.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._s is not None and self._s.isatty())
        except Exception:
            return False

    @property
    def encoding(self):
        return "utf-8"

    def fileno(self):
        if self._s is None:
            raise OSError("no fileno")
        return self._s.fileno()


def install():
    """在 serve.py 启动时调一次。重复调用安全。"""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        if not isinstance(sys.stdout, _Tee):
            sys.stdout = _Tee(sys.stdout)
        if not isinstance(sys.stderr, _Tee):
            sys.stderr = _Tee(sys.stderr)
    except Exception:
        pass
    try:
        _append_file("\n===== serve 启动 %s =====\n"
                     % time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass


def tail(since: int = 0, limit: int = 3000) -> dict:
    """返回 since 之后的行。since=0 表示「给我最后 limit 行」。

    seq 是**累计行号**，环形淘汰不影响它，所以前端拿最后一条的 seq 当下次的 since
    就能稳定做增量，不需要 diff 内容。
    """
    with _lock:
        rows = list(_buf)
    total = _seq
    if since and since > 0:
        rows = [r for r in rows if r[0] > since]
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return {
        "total": total,
        "lines": [{"n": n, "s": s} for n, s in rows],
        "file": str(_path()),
        "max_lines": MAX_LINES,
    }


def clear() -> None:
    with _lock:
        _buf.clear()
