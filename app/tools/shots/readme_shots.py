# -*- coding: utf-8 -*-
r"""README 截图：无头 Edge 打开真后端，切英文 + 白色主题，逐页截一张。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\shots\readme_shots.py
产物：app/docs/readme/*.png（入库，README 直接引用）

为什么不复用 accept 里那套：那套带着几十条断言，跑一轮好几分钟；
这里只要「起后端 → 开页 → 切语言/主题 → 截图」，一条命令几十秒。
"""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
OUT = REPO / "app" / "docs" / "readme"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
TMP = Path(os.environ.get("TEMP") or ".")

# (文件名, 页面 key, 说明) —— 顺序就是 README 里的顺序
PAGES = [
    ("01-home", "home", "首页：根据真实现状推下一步"),
    ("02-extract", "extract", "提取：视频 → 帧 → 人脸"),
    ("03-dataset", "dataset", "数据集：过滤 / 排序 / 去重"),
    ("04-train", "train", "训练：模型列表与新建"),
    ("05-merge", "merge", "合成器：逐脸分配 / 逐帧预览"),
    ("06-engine", "tool-engine", "工具 · 引擎编译（TensorRT）"),
    ("07-mask", "tool-mask", "工具 · 遮罩绘制"),
    ("08-settings", "settings", "设置"),
    # 下面这些是旧 README 里"没截图"的页面，补齐
    ("09-about", "about", "关于：版本 / 许可 / 免责声明"),
    ("10-bili", "tool-bili", "工具 · B 站下载"),
    ("11-color", "tool-color", "工具 · 色彩迁移"),
    ("12-angle", "tool-angle", "工具 · 角度视图"),
    ("13-train-cfg", "train-cfg", "训练参数（含搜索框）"),
    ("14-extract-run", "extract-run", "提取进度（六阶段 / 实时读数）"),
]


def start_server():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # 系统代理（Watt Toolkit / 任何 VPN）会把 127.0.0.1 也劫走，本地探测必须绕开
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    proc = subprocess.Popen([str(PY), "serve.py"],
                            cwd=str(REPO / "app"), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")
    port = 0
    t0 = time.time()
    while time.time() - t0 < 120:
        line = proc.stdout.readline()
        if not line:
            break
        if line.startswith("PORT="):
            port = int(line.strip().split("=")[1])
            break
    if not port:
        proc.kill()
        raise SystemExit("后端没打印 PORT=")
    # 等 /api/state 能通
    _plain = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(120):
        try:
            _plain.open("http://127.0.0.1:%d/api/state" % port, timeout=2).read()
            return proc, port
        except Exception:                                  # noqa: BLE001
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("后端起来了但 /api/state 不通")


def edge(port):
    from websockets.sync.client import connect
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dbg = s.getsockname()[1]
    profile = TMP / ("readme_shots_%s" % uuid.uuid4().hex[:8])
    proc = subprocess.Popen(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--window-size=1500,950", "--no-first-run", "--no-default-browser-check",
         "--user-data-dir=%s" % profile, "--remote-debugging-port=%d" % dbg,
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000)
    ws_url = ""
    t0 = time.time()
    while time.time() - t0 < 60 and not ws_url:
        try:
            raw = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                "http://127.0.0.1:%d/json/list" % dbg, timeout=3).read()
            pages = [t for t in json.loads(raw.decode("utf-8"))
                     if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
        except Exception:                                  # noqa: BLE001
            time.sleep(0.4)
    if not ws_url:
        proc.kill()
        raise SystemExit("连不上无头 Edge")
    cm = connect(ws_url, max_size=None, open_timeout=20)
    ws = cm.__enter__()
    seq = [0]

    def call(method, params=None, timeout=90):
        seq[0] += 1
        mid = seq[0]
        ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(ws.recv(timeout=timeout))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("%s: %s" % (method, msg["error"]))
                return msg.get("result") or {}

    def ev(expr, timeout=120):
        r = call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                      "awaitPromise": True,
                                      "timeout": int(timeout * 1000)}, timeout + 30)
        return (r.get("result") or {}).get("value")

    call("Page.enable")
    call("Runtime.enable")

    return proc, cm, call, ev


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    OUT.mkdir(parents=True, exist_ok=True)
    srv, port = start_server()
    brow, cm, call, ev = None, None, None, None
    try:
        brow, cm, call, ev = edge(port)
        # 先落语言/主题，再进站：这一条 setItem 在 about:blank 上做不了，所以先开页
        call("Page.navigate", {"url": "http://127.0.0.1:%d/" % port})
        time.sleep(4)
        ev("(function(){try{localStorage.setItem('dfl-lang','en');}catch(e){}"
           "document.documentElement.setAttribute('data-theme','light');return 1;})()")
        call("Page.navigate", {"url": "http://127.0.0.1:%d/" % port})
        time.sleep(5)
        ev("(function(){try{window.I18N.apply('en');}catch(e){}"
           "try{window.applyTheme('light',{instant:true});}catch(e){}"
           "document.documentElement.setAttribute('data-theme','light');return 1;})()")
        time.sleep(1.5)
        for name, key, desc in PAGES:
            if only and not any(name.startswith(o) for o in only):
                continue
            ok = ev("(function(){try{window.go('%s');return 1;}catch(e){return String(e);}})()" % key)
            time.sleep(2.2)
            ev("(function(){try{window.scrollTo(0,0);}catch(e){}"
               "document.documentElement.style.scrollBehavior='auto';"
               "return 1;})()")
            # 关掉可能挡住的欢迎/教程层
            ev("(function(){['welcome','tutMask','onboard'].forEach(function(id){"
               "var el=document.getElementById(id); if(el){el.style.display='none';}});return 1;})()")
            time.sleep(0.4)
            time.sleep(0.6)
            res = {}
            for _try in range(3):
                try:
                    res = call("Page.captureScreenshot", {"format": "png"}, timeout=45)
                    if res.get("data"):
                        break
                except Exception as e:                     # noqa: BLE001
                    print("    capture retry: %s" % str(e)[:80])
            data = base64.b64decode(res.get("data") or "")
            if not data:
                print("  FAIL %s（截图空）" % name)
                continue
            (OUT / (name + ".png")).write_bytes(data)
            print("  OK   %-12s %s  (%s)" % (name, desc, ok))
    finally:
        try:
            if cm is not None:
                cm.__exit__(None, None, None)
        except Exception:                                  # noqa: BLE001
            pass
        if brow is not None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(brow.pid)],
                           capture_output=True)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
    print("产物:", OUT)


if __name__ == "__main__":
    main()
