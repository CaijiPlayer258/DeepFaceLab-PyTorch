"""只跑后端，不建窗口。给 Electron / 浏览器调试用。

启动后会往 stdout 打一行 PORT=xxxx，外壳读它拿端口。
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _free_port(start: int = 8760) -> int:
    for p in range(start, start + 60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def main() -> int:
    import uvicorn

    from server.app import app as fastapi_app

    port = _free_port()
    os.environ["DFL_PORT"] = str(port)
    print("PORT=%d" % port, flush=True)

    # 上次开着远程访问的话，这次启动自动恢复（失败静默跳过）
    try:
        from server.app import _remote_boot
        _remote_boot()
    except Exception:
        pass

    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
