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
PORT = 8000


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
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
