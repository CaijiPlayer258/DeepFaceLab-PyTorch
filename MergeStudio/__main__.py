"""
MergeStudio - DeepFaceLab Torch Merge Studio
Standalone web server for face swap merging.
"""
import sys
import os
import socket
import subprocess
from pathlib import Path

# Ensure project root is in sys.path (for modelhub, facelib, core, etc.)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn
from MergeStudio.api.server import create_app
import webbrowser
import threading

HOST = "127.0.0.1"
PORT = 8002

# ── 自动加载配置（ui-dark）──
# workspace/.merge_auto.json 格式：
# {"workspace": "F:\\DFL-PyTorch\\workspace", "model": "Anna"}
# 存在则启动后自动打开 workspace + 加载模型
_auto_config_path = Path(__file__).parent.parent / "workspace" / ".merge_auto.json"
_auto_config = None
if _auto_config_path.exists():
    import json as _json
    try:
        _auto_config = _json.loads(_auto_config_path.read_text(encoding="utf-8"))
    except Exception:
        pass

def _auto_load_delayed():
    """服务启动后延迟执行自动加载（等 uvicorn ready）。
    序列: workspace → model → video(explicit) — 每步更新 AUTOLOAD_STATE 供页面轮询。"""
    import time
    import json as json_mod
    import urllib.request
    from pathlib import Path as _P
    from MergeStudio.api import routes_preview as _rp
    time.sleep(3)
    base = "http://127.0.0.1:%d" % PORT

    def _post(path, payload, timeout):
        data = json_mod.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    if _auto_config:
        _rp.AUTOLOAD_STATE["configured"] = True
    had_error = False
    if _auto_config and _auto_config.get("workspace"):
        _rp.AUTOLOAD_STATE.update(stage="opening-workspace", detail=str(_auto_config["workspace"]))
        try:
            _post("/api/project/open", {"path": _auto_config["workspace"]}, 60)
            print("[AutoLoad] workspace opened:", _auto_config["workspace"])
        except Exception as e:
            had_error = True
            _rp.AUTOLOAD_STATE.update(stage="error", detail="workspace: %s" % e)
            print("[AutoLoad] workspace open failed:", e)
    if _auto_config and _auto_config.get("model"):
        _rp.AUTOLOAD_STATE.update(stage="loading-model", detail=str(_auto_config["model"]))
        try:
            _post("/api/models/load", {"name": _auto_config["model"]}, 600)
            print("[AutoLoad] model loaded:", _auto_config["model"])
        except Exception as e:
            had_error = True
            _rp.AUTOLOAD_STATE.update(stage="error", detail="model: %s" % e)
            print("[AutoLoad] model load failed:", e)
    if _auto_config and _auto_config.get("video"):
        _v = _P(_auto_config["video"])
        if not _v.is_absolute() and _auto_config.get("workspace"):
            _v = _P(_auto_config["workspace"]) / _v
        _rp.AUTOLOAD_STATE.update(stage="selecting-video", detail=_v.name)
        try:
            _post("/api/select-video", {"path": str(_v)}, 300)
            print("[AutoLoad] video selected:", _v.name)
        except Exception as e:
            had_error = True
            _rp.AUTOLOAD_STATE.update(stage="error", detail="video: %s" % e)
            print("[AutoLoad] video select failed:", e)
    if _auto_config:
        if not had_error:
            _rp.AUTOLOAD_STATE.update(stage="done", detail="")
            print("[AutoLoad] all done (workspace + model + video)")


def _check_port():
    """Check if port is in use and optionally kill the owning process."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, PORT))
        sock.close()
        return  # port is free
    except OSError:
        sock.close()
        pass

    # Port is in use — find the process
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, shell=True
        )
        lines = result.stdout.splitlines()
        pids = set()
        for line in lines:
            if f"{HOST}:{PORT}" in line or f"0.0.0.0:{PORT}" in line:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[1] == "LISTENING":
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
    except Exception:
        pids = set()

    if not pids:
        print(f"ERROR: Port {PORT} is in use but could not identify the process.")
        sys.exit(1)

    # Get process names for display
    proc_info = []
    for pid in pids:
        try:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, shell=True)
            name = "unknown"
            for line in r.stdout.splitlines():
                if str(pid) in line:
                    name = line.strip().split()[:3]
                    name = " ".join(name) if isinstance(name, list) else name
                    break
            proc_info.append((pid, name))
        except Exception:
            proc_info.append((pid, str(pid)))

    print(f"Port {PORT} is already in use by:")
    for pid, name in proc_info:
        print(f"  PID {pid} — {name}")

    try:
        resp = input("Kill these processes and restart? [Y/n]: ").strip().lower()
    except EOFError:
        resp = "y"

    if resp in ("", "y", "yes"):
        for pid, _ in proc_info:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, check=True, shell=True)
                print(f"  Killed PID {pid}")
            except subprocess.CalledProcessError:
                print(f"  Failed to kill PID {pid} (access denied)")
        # Small delay to let the port be released
        import time
        time.sleep(0.5)
    else:
        print("Exiting.")
        sys.exit(1)


def main():
    _check_port()
    app = create_app()
    threading.Timer(2.0, lambda: webbrowser.open(f'http://{HOST}:{PORT}')).start()
    threading.Thread(target=_auto_load_delayed, daemon=True).start()

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level='info',
        access_log=False,
    )


if __name__ == '__main__':
    main()
