# -*- coding: utf-8 -*-
"""训练进程内的最小控制口（替代 4.0 的 WebUI 监控页）。

只做 4.1 后端真正会用到的几个动作，UI 一律没有：
    POST /request-preview      请求下一轮出一张预览
    GET  /preview              取走缓存（pop 语义，与 4.0 一致）
    POST /request-save         请求保存
    POST /request-quit         请求退出（保存后退出）
    POST /request-settings     热更新生成器参数
    POST /update-model-options 热更新模型参数
    GET  /status               取当前 iter / loss（排障用）

接口形状与 4.0 的 WebUI 完全一致，所以 app/server 那边一行都不用改。
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

_cache: Dict[str, Any] = {}
_lock = threading.Lock()
_flags = {"preview": False, "save": False, "close": False}
_pending: Dict[str, Any] = {}
_server: Optional[ThreadingHTTPServer] = None
_password = ""


def update_cache(patch: Dict[str, Any]) -> None:
    with _lock:
        _cache.update(patch or {})


def get_cache() -> Dict[str, Any]:
    with _lock:
        return dict(_cache)


def _set(flag: str) -> None:
    with _lock:
        _flags[flag] = True


def _clear(flag: str) -> None:
    with _lock:
        _flags[flag] = False


def is_preview_requested() -> bool:
    return _flags["preview"]


def clear_preview_request() -> None:
    _clear("preview")


def is_save_requested() -> bool:
    return _flags["save"] or _flags["close"]


def clear_save_request() -> None:
    _clear("save")


def is_close_requested() -> bool:
    return _flags["close"]


def clear_close_request() -> None:
    _clear("close")


def get_settings_update():
    with _lock:
        out = _pending.pop("settings", None)
    return out


def get_model_options_update():
    with _lock:
        out = _pending.pop("model_options", None)
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # 静默：训练控制口不打访问日志
        return

    def _send(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8", "replace")) or {}
        except Exception:
            return {}

    def do_GET(self):                 # noqa: N802
        path = self.path.split("?")[0]
        if path == "/preview":
            with _lock:
                data = _cache.pop("preview", None)
            return self._send({"ok": True, "preview": data})
        if path == "/status":
            with _lock:
                return self._send({"ok": True, "iter": _cache.get("iter", 0),
                                   "loss": _cache.get("loss_history", [])[-1:] or []})
        return self._send({"ok": False, "error": "unknown path"}, 404)

    def do_POST(self):                # noqa: N802
        path = self.path.split("?")[0]
        body = self._read_json()
        if _password and str(body.get("password") or "") != _password:
            return self._send({"ok": False, "error": "bad password"}, 403)
        if path == "/request-preview":
            _set("preview")
            return self._send({"ok": True})
        if path == "/request-save":
            _set("save")
            return self._send({"ok": True})
        if path == "/request-quit":
            _set("close")
            _set("save")
            return self._send({"ok": True})
        if path == "/request-settings":
            with _lock:
                _pending["settings"] = {k: v for k, v in body.items() if k != "password"}
            return self._send({"ok": True})
        if path == "/update-model-options":
            with _lock:
                _pending["model_options"] = {k: v for k, v in body.items() if k != "password"}
            return self._send({"ok": True})
        return self._send({"ok": False, "error": "unknown path"}, 404)


def start_webui_server(port: int = 6790, model_dir: str = ".", password: Optional[str] = None) -> bool:
    """保持与 4.0 相同的函数名/签名，Trainer 那边调用点不用改。"""
    global _server, _password
    _password = password or ""
    if _server is not None:
        return True
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), _Handler)
    srv.daemon_threads = True
    _server = srv
    threading.Thread(target=srv.serve_forever, name="train-control", daemon=True).start()
    print("[control] 训练控制口已启动 127.0.0.1:%d（无界面）" % int(port), flush=True)
    return True
