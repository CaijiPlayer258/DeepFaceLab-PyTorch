"""DeepFaceLab 4.1 入口。

架构：本地 FastAPI 起在 127.0.0.1 上，外面套一层系统 WebView（pywebview / WebView2）。
界面是纯静态网页，所以把壳换成 Electron 或 Tauri 也只是换 main，不用动 app/web。

规则：界面层（app/web）不碰 Python；所有重活走 HTTP API。
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _free_port(start: int = 8760) -> int:
    for p in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


# 系统代理（Steam++ 之类）会把 127.0.0.1 的请求也劫走，导致探测拿到 404。
# 探测必须显式绕开代理。
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _enable_dpi_awareness() -> None:
    """必须在建窗口之前声明 DPI 感知。

    否则 Windows 会把进程当成 DPI 无关的：窗口按物理像素创建，而 WebView2
    仍按 150% 渲染，于是 CSS 视口只剩 1180/1.5 ≈ 787px，整个界面看起来被放大。
    """
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE
        return
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _dpi_scale() -> float:
    """从注册表读系统 DPI 缩放（96/120/144...）。

    不依赖进程 DPI 感知，也不依赖网页回报 —— pywebview 的 api 注入时机不稳定，
    用它做尺寸校正会出现"有时正确有时不校正"的抖动。
    """
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\Desktop\WindowMetrics")
        val, _ = winreg.QueryValueEx(key, "AppliedDPI")
        return max(1.0, float(val) / 96.0)
    except Exception:
        return 1.0


def _work_area() -> tuple:
    """屏幕可用区域（物理像素，已扣掉任务栏）。"""
    try:
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0)
        return (r.right - r.left, r.bottom - r.top)
    except Exception:
        return (1920, 1080)


def _wait(url: str, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with _NO_PROXY.open(url, timeout=0.6) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.08)
    return False


class WindowApi:
    """暴露给网页的窗口控制（标题栏那三个圆点用）。"""

    def __init__(self):
        self.win = None

    def minimize(self):
        if self.win:
            self.win.minimize()

    def maximize(self):
        if not self.win:
            return
        self.win.toggle_fullscreen()

    def close(self):
        if self.win:
            self.win.destroy()

    def toggle_theme(self):
        return None

    def fit(self, dpr: float = 1.0):
        """按网页报告的 devicePixelRatio 校正窗口尺寸。

        pywebview 在 Windows 上把 width/height 当物理像素，而 WebView2 按系统
        缩放渲染；高分屏下 CSS 视口会变成 1180/1.5 ≈ 787px，界面整体被放大。
        让网页把 dpr 报回来，这里反算物理尺寸，视口就正好是 1180×760。
        """
        try:
            k = float(dpr) or 1.0
        except Exception:
            k = 1.0
        aw, ah = _work_area()
        w = int(min(1180 * k, aw - 24))
        h = int(min(760 * k, ah - 24))
        self.win.resize(w, h)
        return {"dpr": k, "w": w, "h": h}


def main() -> int:
    _enable_dpi_awareness()

    import uvicorn
    import webview

    from server.app import app as fastapi_app

    port = _free_port()
    cfg = uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()

    url = "http://127.0.0.1:%d/" % port
    if not _wait(url.rstrip("/") + "/api/state"):
        print("backend did not start on port %d" % port, file=sys.stderr)
        return 1

    k = _dpi_scale()
    aw, ah = _work_area()
    win_w = int(min(1180 * k, aw - 24))
    win_h = int(min(760 * k, ah - 24))
    api = WindowApi()
    win = webview.create_window(
        "DeepFaceLab 4.1",
        url,
        width=win_w,
        height=win_h,
        min_size=(int(900 * k), int(600 * k)),
        frameless=True,
        easy_drag=True,
        background_color="#FFFFFF",
        js_api=api,
    )
    api.win = win
    webview.start()
    server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
