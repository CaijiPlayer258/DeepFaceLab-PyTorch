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

# 4.0 引擎代码收在 app/engine/：server 里那些 "from core… / from DFLIMG…"
# 是惰性 import，不在这里加路径会在第一次调用时炸。
ENGINE = ROOT / "engine"
if ENGINE.is_dir() and str(ENGINE) not in sys.path:
    sys.path.append(str(ENGINE))


def _free_port(start: int = 8760) -> int:
    for p in range(start, start + 60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def _apply_flags(argv) -> None:
    """命令行开关。目前只有 --mock：把 B 站相关路由切到占位数据。

    占位数据以前是默认行为（前端 biliMockFetch 直接随机生成），现在反过来：
    默认全真，只有显式 --mock（或等价的环境变量 DFL_BILI_MOCK=1）才给占位。
    """
    if "--mock" in argv:
        os.environ["DFL_BILI_MOCK"] = "1"
        print("[mock] DFL_BILI_MOCK=1：B 站列表/下载将返回占位数据", flush=True)


def main() -> int:
    # 先把 stdout/stderr 接上 tee，后面所有输出（含 uvicorn、训练、提取、合成）
    # 都会进内存环形缓冲 + _server_output.log，界面的「输出」页读的就是它。
    # 必须在 import uvicorn 之前装，否则 import 期的告警会漏掉。
    try:
        from server._common import output_log
        output_log.install()
    except Exception as _e:
        print('[output] tee 安装失败（不影响使用）：%s' % _e, flush=True)

    import uvicorn

    # --preview（或 DFL_PREVIEW=1）：给「设计工作台」这类独立浏览器会话用。
    # 那个会话没有登录态、也没在本地同意过协议，页面会被协议门整块盖住，
    # 看不到真实界面。这个开关只做两件事：跳过协议门、把登录态当已登录；
    # 数据仍从本机真实后端读，改代码照样热更新。
    if "--preview" in sys.argv or os.environ.get("DFL_PREVIEW") == "1":
        os.environ["DFL_PREVIEW"] = "1"
        print("[preview] 预览模式：跳过协议门与登录校验，仅供本机画布查看", flush=True)

    _apply_flags(sys.argv[1:])

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
