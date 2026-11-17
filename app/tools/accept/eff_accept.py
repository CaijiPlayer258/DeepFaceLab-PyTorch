# -*- coding: utf-8 -*-
r"""「生效配置」验收：一行 [eff] + 请求体脱敏回显 + 三类 warn + 前端横幅，全在 %TEMP%。

验的东西（对应用户这一轮的要求）：
  A  正常路径：请求体带 detector=tinymog → [eff] 里 detector=tinymog、detector_src=ui、
     这一段没有 [eff-warn]；status.warnings 为空。
  B  复现老 bug：请求体不带 detector、只带 quality=high → [eff-warn] 出现、
     status.warnings[] 里有它、detector_src=inferred、前端顶部横幅可见（截图）。
  C  未知键：body 里塞 {"foo":1} → [req-unused] foo。
  D  脱敏：整段视频路径只留 clip12.mp4、faces_project 只留目录名、base64 换成长度，
     [req] 行里不许出现 C:\Users / 超长 base64。
  E  六阶段链：POST /api/merge/stage/frames 也有 [eff]（kind=stage:frames），
     /api/merge/stage/status 回 eff + warnings。
  F  前端「当前生效」面板：14 个字段、来源标签（界面/推断/默认）都在。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\eff_accept.py
    （--no-browser 跳过截图那一段）

退出码 0 = 全过。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_eff_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"
SHOT = BOX / "banner.png"

DFM = "\u963f\u8a3c\u8a3c\u8a3c\u8a3c-416"
SETTINGS = {"mask_mode": 8, "mask_source": "XSegLite", "color_transfer": "mt",
            "blend": "seamless", "mask_erode": 0, "mask_blur": 4}
# 14 → 16：多了两个用户可调的性能旋钮（dfm_dtype / compose_backend，设置页「性能」区）。
# 它们和其他 14 个字段同一套「值 + _src」，来源多一档 "auto"（界面选了自动）。
FIELDS = ("detector", "mask_source", "ct_mode", "blend", "erode", "blur", "models",
          "faces_project", "library_enabled", "fast_export", "frame_range",
          "workers", "batch", "jpeg_quality", "dfm_dtype", "compose_backend")

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def head(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def http(method, path, body=None, timeout=120.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        blob = e.read()
        try:
            return e.code, json.loads(blob.decode("utf-8") or "{}")
        except Exception:                                   # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def post(path, body, timeout=120.0):
    return http("POST", path, body, timeout=timeout)


def get(path, timeout=60.0):
    return http("GET", path, timeout=timeout)[1]


# ============================================================
# 沙箱 / 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        try:
            os.rmdir(str(WS / "model"))
        except OSError:
            pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / "model"),
                        str(REPO / "workspace" / "model")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
    src = TEMP / "dfl_b_accept" / "media" / "clip12.mp4"
    dst = MEDIA / "clip12.mp4"
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "testsrc=size=192x192:rate=10:duration=1.2",
                        str(dst)], check=True)
    say("沙箱：%s" % BOX)
    return dst


def pick_dfm() -> str:
    p = REPO / "workspace" / "model" / (DFM + ".dfm")
    if p.is_file():
        return DFM
    cands = sorted((REPO / "workspace" / "model").glob("*.dfm"))
    if not cands:
        raise SystemExit("workspace\\model 里没有 .dfm —— 先导出一个模型")
    return cands[0].stem


def start_server():
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    # G 段断言的是「seamless 被显式关掉 GPU 版 → 退回 CPU + 一条 [eff-warn]」这条**可复盘路径**。
    # 现在 seamless 默认也走 GPU（compose_gpu.seamless_on 默认 True，见 knobs.refine_compose），
    # 不显式关掉的话 G4/G5/G6/G7/H1 会去断言一个已经不存在的默认行为。
    env["DFL_COMPOSE_GPU_SEAMLESS"] = "0"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)
    logf = open(str(SERVER_LOG), "ab")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                               # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                           # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 240:
        time.sleep(0.3)
    if port["v"] is None:
        raise SystemExit("后端没起来（见 %s）" % SERVER_LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端：%s（pid %d）" % (BASE, proc.pid))
    return proc


# ============================================================
# 日志解析：[eff] 行 + 紧跟其后的 [eff-warn]
# ============================================================
def read_log():
    try:
        return SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def eff_blocks(text):
    """-> [{"eff": {...}, "warns": [text, ...], "req": ...}]（按出现顺序）。"""
    out = []
    cur = None
    for line in text.splitlines():
        if line.startswith("[eff] "):
            if cur:
                out.append(cur)
            try:
                cur = {"eff": json.loads(line[len("[eff] "):]), "warns": [], "req": None}
            except Exception:                               # noqa: BLE001
                cur = None
        elif line.startswith("[eff-warn] "):
            if cur is not None:
                cur["warns"].append(line[len("[eff-warn] "):])
        elif line.startswith("[req] "):
            if cur:
                out.append(cur)
                cur = None
    if cur:
        out.append(cur)
    return out


def wait_terminal(path, timeout=180.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = get(path)
        if not last.get("running"):
            return last
        time.sleep(0.4)
    return last


# ============================================================
# 浏览器（Edge headless + CDP）：真实页面里渲染横幅 + 面板，截一张图
# ============================================================
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def shot_banner(eff, warns):
    try:
        return _shot_banner(eff, warns)
    except Exception as e:                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        say("浏览器那一段失败：%s: %s" % (e.__class__.__name__, str(e)[:300]))
        return None


def _shot_banner(eff, warns):
    try:
        from websockets.sync.client import connect
    except Exception as e:                                  # noqa: BLE001
        say("没装 websockets（%s）—— 跳过截图" % e)
        return None
    if not Path(EDGE).is_file():
        say("没有 Edge —— 跳过截图")
        return None
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    import uuid
    # 每次全新的 profile：复用被强杀实例留下的目录会撞锁，Edge 直接退出（调试口永远不出现）
    profile = BOX / ("edge_profile_%d_%s" % (os.getpid(), uuid.uuid4().hex[:6]))
    # 直连 127.0.0.1：系统代理会把调试口也代理掉（表现为「连不上」）
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    proc = subprocess.Popen(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--window-size=1680,1100", "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", "--remote-allow-origins=*",
         "--user-data-dir=%s" % profile, "--remote-debugging-port=%d" % port,
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000)
    try:
        ws_url = ""
        deadline = time.time() + 60
        while time.time() < deadline and not ws_url:
            try:
                raw = opener.open(
                    "http://127.0.0.1:%d/json/list" % port, timeout=3).read()
                tabs = json.loads(raw.decode("utf-8"))
                pages = [t for t in tabs if t.get("type") == "page"
                         and t.get("webSocketDebuggerUrl")]
                # 起手可能还塞了 edge:// 引导页：优先 about:blank
                pages.sort(key=lambda t: 0 if str(t.get("url") or "").startswith("about:")
                           else 1)
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
            except Exception:                               # noqa: BLE001
                time.sleep(0.4)
        if not ws_url:
            say("连不上无头 Edge 的调试口")
            return None

        seq = [0]
        with connect(ws_url, max_size=None, open_timeout=20) as ws:
            def call(method, params=None):
                seq[0] += 1
                mid = seq[0]
                ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(ws.recv(timeout=60))
                    if msg.get("id") == mid:
                        if "error" in msg:
                            raise RuntimeError("%s: %s" % (method, msg["error"]))
                        return msg.get("result") or {}

            def ev(expr, timeout=60):
                res = call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                                "awaitPromise": True,
                                                "timeout": int(timeout * 1000)})
                if res.get("exceptionDetails"):
                    raise RuntimeError(json.dumps(res["exceptionDetails"],
                                                  ensure_ascii=False)[:400])
                return (res.get("result") or {}).get("value")

            call("Page.enable")
            call("Runtime.enable")
            call("Emulation.setDeviceMetricsOverride", {"width": 1680, "height": 1100,
                                                        "deviceScaleFactor": 1,
                                                        "mobile": False})
            call("Page.navigate", {"url": BASE + "/"})
            # 先等 app.js 把 go() 挂上；**进合成页**才会 bindMergeUI()，mgStep /
            # mgpEffRender 都是那一步才定义的（只在 #home 上等它俩会永远等不到）。
            booted = False
            for _ in range(180):
                time.sleep(0.4)
                try:
                    if ev("typeof window.go === 'function'"):
                        booted = True
                        break
                except Exception:                           # noqa: BLE001
                    pass
            if not booted:
                say("页面没就绪（window.go 没挂上）")
                return None
            ev("window.go('merge'); true")
            ready = False
            for _ in range(120):
                time.sleep(0.4)
                try:
                    if ev("typeof window.mgpEffRender === 'function' "
                          "&& typeof window.mgStep === 'function'"):
                        ready = True
                        break
                except Exception:                           # noqa: BLE001
                    pass
            if not ready:
                say("合成页没就绪（mgpEffRender 没挂上）")
                return None
            ev("window.mgStep('progress'); true")
            # 首启欢迎层（扫码/法务）会盖住整页：截图前收起（它和这一屏无关）
            ev("(function(){var w=document.getElementById('welcome');"
               "if(w) w.style.display='none';"
               "var lg=document.getElementById('legalGate');"
               "if(lg) lg.style.display='none'; return true;})()")
            time.sleep(0.6)
            payload = json.dumps({"eff": eff, "warns": warns}, ensure_ascii=False)
            ev("(function(){var d=%s; window.mgpEffRender(d.eff, d.warns); return true;})()"
               % payload)
            time.sleep(0.5)
            # 14 个字段不再常驻：点右下角的「查看生效配置」才渲染进抽屉
            ev("(function(){var o=document.getElementById('mgpEffOpen');"
               "if(o) o.click(); return true;})()")
            time.sleep(0.4)
            state = ev("(function(){var b=document.getElementById('mgpEffWarn');"
                       "var body=document.getElementById('mgpEffWarnBody');"
                       "var t=document.getElementById('mgpEffWarnTitle');"
                       "var g=document.getElementById('mgpEffGrid');"
                       "var p=document.getElementById('mgpEffDrawer');"
                       "return JSON.stringify({banner: !!b && !b.hidden,"
                       "text: body?String(body.textContent):'',"
                       "/* 一条警告一个 div：lines = 真的摆出来几行（不是 join 出来的一个字符串） */"
                       "lines: body?body.children.length:0,"
                       "title: t?String(t.textContent):'',"
                       "items: g?g.children.length:0,"
                       "panel: !!p && !p.hidden,"
                       "gone: !document.getElementById('mgpEff')});})()")
            rect = ev("(function(){var e=document.querySelector('#p-merge .mgview"
                      "[data-view=\"progress\"] .mgprog');"
                      "if(!e) return '';var r=e.getBoundingClientRect();"
                      "return JSON.stringify({x:r.x,y:r.y,width:r.width,height:r.height});})()")
            clip = json.loads(rect) if rect else None
            if clip:
                clip["scale"] = 1          # CDP 的 clip 必须带 scale，不给就 Invalid parameters
            res = call("Page.captureScreenshot",
                       {"format": "png", "captureBeyondViewport": True, "clip": clip})
            import base64 as b64
            SHOT.write_bytes(b64.b64decode(res.get("data") or ""))
            say("截图：%s（%d 字节）" % (SHOT, SHOT.stat().st_size))
            try:
                call("Browser.close")
            except Exception:                               # noqa: BLE001
                pass
            return json.loads(state) if state else None
    finally:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True)
        except Exception:                                   # noqa: BLE001
            pass


# ============================================================
# 主流程
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    clip = prepare_sandbox()
    model = pick_dfm()
    proc = start_server()
    text0 = ""
    try:
        head("A  正常路径：detector=tinymog（界面发的）")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "detector": "tinymog", "model": model,
            "settings": SETTINGS, "faces_project": "C:\\Users\\30764\\faces_project"})
        check(st == 200 and d.get("ok"), "A1 导出启动", "HTTP %s ok=%s" % (st, d.get("ok")))
        time.sleep(0.8)
        blocks = eff_blocks(read_log())
        a = [b for b in blocks if b["eff"].get("kind") == "fused"]
        check(bool(a), "A2 有 [eff] 行", "共 %d 条 [eff]" % len(blocks))
        if a:
            e = a[-1]["eff"]
            check(e.get("detector") == "tinymog" and e.get("detector_src") == "ui",
                  "A3 detector=tinymog + src=ui",
                  "%s/%s" % (e.get("detector"), e.get("detector_src")))
            missing = [f for f in FIELDS
                       if f not in e or (f + "_src") not in e]
            check(not missing, "A4 字段表齐全（值 + 来源）",
                  "缺 %s" % missing if missing else "%d/%d" % (len(FIELDS), len(FIELDS)))
            # 默认态（请求体没给这两个键）：值报「自动」、来源 default —— 不冒充具体档
            check(e.get("dfm_dtype") == "auto" and e.get("dfm_dtype_src") == "default",
                  "A4b 请求体没给 dfm_dtype → auto / default",
                  "%s/%s" % (e.get("dfm_dtype"), e.get("dfm_dtype_src")))
            check(e.get("compose_backend") == "cpu" and e.get("compose_backend_src") == "default",
                  "A4c 请求体没给 compose_gpu → cpu（现在的 CPU 路径）/ default",
                  "%s/%s" % (e.get("compose_backend"), e.get("compose_backend_src")))
            check(not a[-1]["warns"], "A5 这一段没有 [eff-warn]", "; ".join(a[-1]["warns"]))
        s = wait_terminal("/api/merge/export/status")
        check(not (s.get("warnings") or []), "A6 status.warnings 为空",
              str(s.get("warnings")))
        check((s.get("eff") or {}).get("detector") == "tinymog",
              "A7 status.eff 回显 detector", str((s.get("eff") or {}).get("detector")))

        head("B  复现老 bug：不带 detector、只带 quality=high")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "quality": "high", "model": model,
            "settings": SETTINGS})
        check(st == 200 and d.get("ok"), "B1 导出启动", "HTTP %s ok=%s" % (st, d.get("ok")))
        time.sleep(0.8)
        blocks = eff_blocks(read_log())
        bug = [b for b in blocks if b["eff"].get("kind") == "fused"]
        check(bool(bug), "B2 有 [eff] 行", "%d 条" % len(bug))
        if bug:
            e = bug[-1]["eff"]
            check(e.get("detector") == "retinaface_10g" and e.get("detector_src") == "inferred",
                  "B3 detector 由 quality 推出来", "%s/%s" % (e.get("detector"),
                                                              e.get("detector_src")))
            check(any("quality=high" in w and "detector" in w for w in bug[-1]["warns"]),
                  "B4 [eff-warn] 触发", " | ".join(bug[-1]["warns"]))
        s = wait_terminal("/api/merge/export/status")
        warns = s.get("warnings") or []
        check(bool(warns) and any("detector" in str(w) for w in warns),
              "B5 status.warnings[] 里有这条", " | ".join(str(w) for w in warns)[:200])
        check(any(w.get("code") == "eff-inferred"
                  for w in (s.get("eff_warnings") or [])),
              "B6 eff_warnings 结构（code=eff-inferred）",
              str(s.get("eff_warnings"))[:160])

        # ---- G：两个用户可调的性能旋钮（请求体 → [eff]，被跳过时 [eff-warn]）----
        head("G  性能旋钮：dfm_dtype=fp32 / compose_gpu=1（选了就要能看到，跳过了要喊）")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "detector": "tinymog", "model": model,
            "settings": SETTINGS, "dfm_dtype": "fp32", "compose_gpu": 1})
        check(st == 200 and d.get("ok"), "G1 带旋钮的导出启动", "HTTP %s" % st)
        time.sleep(0.8)
        blocks = eff_blocks(read_log())
        kn = [b for b in blocks if b["eff"].get("kind") == "fused"]
        check(bool(kn), "G2 有 [eff] 行（kind=fused）", "%d 条" % len(kn))
        e = kn[-1]["eff"] if kn else {}
        check(e.get("dfm_dtype") == "fp32" and e.get("dfm_dtype_src") == "ui",
              "G3 dfm_dtype 回显实际值 fp32（不是 auto 解析后的档）+ src=ui",
              "%s/%s" % (e.get("dfm_dtype"), e.get("dfm_dtype_src")))
        # 这一单的 blend 是 seamless：docs/26 实测无缝融合在 GPU 上更慢，所以 compose_gpu.ready
        # 之前就把它判回 CPU —— 于是 [eff] 必须写 cpu（不是 gpu），并**必须**有一条
        # [eff-warn] 说清「选了 gpu、实际跑 cpu、原因」，和检测器那次事故同一个口径。
        check(e.get("compose_backend") == "cpu" and e.get("compose_backend_src") == "ui",
              "G4 compose_backend 写实际值 cpu（选了 gpu 但 seamless 不进 GPU）",
              "%s/%s want=%s" % (e.get("compose_backend"), e.get("compose_backend_src"),
                                 "cpu/ui"))
        g_warns = kn[-1]["warns"] if kn else []
        joined = " | ".join(g_warns)
        check(any(("选了 gpu" in w) and ("实际跑 cpu" in w) and ("原因" in w) for w in g_warns),
              "G5 [eff-warn] 三件事都说了（选了 X / 实际 Y / 原因 Z）", joined[:240] or "(无)")
        s = wait_terminal("/api/merge/export/status")
        check(any(w.get("field") == "compose_backend"
                  for w in (s.get("eff_warnings") or [])),
              "G6 status.eff_warnings[] 里也有这条（进度页横幅才有得显示）",
              str(s.get("eff_warnings"))[:200])
        check((s.get("eff") or {}).get("compose_backend") == "cpu",
              "G7 status.eff 回显 cpu（界面抽屉那一格不会继续说 gpu）",
              str((s.get("eff") or {}).get("compose_backend")))
        # 日志里那一行 dfm 也必须是 fp32：界面选的档 = 真在跑的档
        check(any("fp32" in str(x) for x in (s.get("eff") or {}).values()) or
              re.search(r'"dfm": "[^"]*fp32', read_log()) is not None,
              "G8 [eff] 的 dfm 那一列写着 fp32", str((s.get("eff") or {}).get("dfm")))

        # overlay + compose_gpu=1：要么真跑 GPU，要么回退到 CPU 并留下同一条口径的告警。
        # 两种情况都算过 —— 断言的是「**永远可复盘**」，不是「这台机器一定有 GPU」。
        head("G9  合成后端 gpu（overlay）：真跑 or 可复盘的回退")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "detector": "tinymog", "model": model,
            "settings": dict(SETTINGS, blend="overlay"), "compose_gpu": 1})
        check(st == 200 and d.get("ok"), "G9a 导出启动", "HTTP %s" % st)
        time.sleep(0.8)
        blocks = eff_blocks(read_log())
        ov = [b for b in blocks if b["eff"].get("kind") == "fused"]
        e2 = ov[-1]["eff"] if ov else {}
        back = str(e2.get("compose_backend") or "")
        check(back in ("gpu", "cpu"), "G9b compose_backend 有明确结论", back)
        if back == "cpu":
            check(any("选了 gpu" in w for w in (ov[-1]["warns"] if ov else [])),
                  "G9c 回退了 → 有「选了 gpu、实际跑了什么」的 [eff-warn]",
                  " | ".join((ov[-1]["warns"] if ov else []))[:200])
        else:
            say("这一单真跑在 GPU 上（%s.torch 可用）→ 没有回退告警是正确行为" % "compose_gpu")
        wait_terminal("/api/merge/export/status")

        # ---- H：一单同时踩两件事（检测器推断 + 合成后端降级）----
        # 为什么要有这一段：F 是**前端**那一段，它只能显示喂进去的那份 status。
        # 以前 F 喂的是上一单（G）的 status —— 那一单 detector 是显式给的，本来就没有
        # detector 那条警告，于是「横幅里没有 detector」看着像前端丢字，其实是**数据只有一条**。
        # 这里用一单同时踩两件事的请求（不带 detector + quality=high，blend=seamless 又带
        # compose_gpu=1）把「后端到底累不累加」摆出来：两条都得在，横幅才谈得上「都看得见」。
        head("H  一单两条警告：检测器推断 + 合成后端降级（横幅要两条都显示）")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "quality": "high", "model": model,
            "settings": SETTINGS, "compose_gpu": 1})
        say("H 启动：HTTP %s ok=%s" % (st, d.get("ok")))
        time.sleep(0.8)
        s = wait_terminal("/api/merge/export/status")
        ew = s.get("eff_warnings") or []
        # ★ 证据：**那一刻** status 里到底有几条（横幅只能显示这里有的东西）
        say("status.eff_warnings = %s" % json.dumps(ew, ensure_ascii=False)[:800])
        say("status.warnings     = %s"
            % json.dumps(s.get("warnings") or [], ensure_ascii=False)[:800])
        _codes = [str(w.get("code")) for w in ew]
        _texts = [str(w.get("text")) for w in ew]
        check(len(ew) >= 2
              and any("detector" in t for t in _texts)
              and any("选了 gpu" in t for t in _texts),
              "H1 eff_warnings[] 里两条都在（后一条没把前一条盖掉）",
              "%d 条 · code=%s" % (len(ew), _codes))

        if not args.no_browser:
            head("F  前端横幅 + 「当前生效」面板（真实页面，Edge headless）")
            say("喂给横幅的 status：eff.detector=%s/%s · eff_warnings=%d 条"
                % ((s.get("eff") or {}).get("detector"),
                   (s.get("eff") or {}).get("detector_src"), len(ew)))
            ui = shot_banner(s.get("eff"), ew)
            if ui is None:
                check(False, "F1 页面渲染", "浏览器那一段没跑起来")
            else:
                # ★ 证据：DOM 里那一刻的横幅标题与正文（正文 = 每行一条 div 的 textContent）
                say("DOM 横幅：title=%r lines=%s text=%r"
                    % (str(ui.get("title")), ui.get("lines"),
                       str(ui.get("text"))[:400]))
                check(ui.get("banner") is True, "F1 顶部横幅可见", str(ui.get("banner")))
                check("detector" in str(ui.get("text")), "F2 横幅文案里有 detector",
                      str(ui.get("text"))[:160])
                # 真需求：两条警告要能**同时**看见（不是后一条挤掉前一条，也不是只有标题说 2 项）
                check(ui.get("lines") == len(ew)
                      and "detector" in str(ui.get("text"))
                      and "选了 gpu" in str(ui.get("text")),
                      "F2b 两条警告同时可见（detector 推断 + 合成后端降级）",
                      "lines=%s/%s · title=%s"
                      % (ui.get("lines"), len(ew), str(ui.get("title"))[:60]))
                check(ui.get("gone") is True, "F3 常驻面板 #mgpEff 已删除", str(ui.get("gone")))
                check(int(ui.get("items") or 0) >= len(FIELDS),
                      "F4 抽屉里 %d 个字段（14 → 16）" % len(FIELDS),
                      "items=%s" % ui.get("items"))
                check(ui.get("panel") is True, "F5 抽屉（#mgpEffDrawer）可见",
                      str(ui.get("panel")))
        else:
            say("--no-browser：跳过截图")

        head("C  未知键：body 里塞 {\"foo\":1}")
        st, d = post("/api/merge/export", {"video": str(clip), "fused": 1, "dry_run": 1,
                                           "detector": "tinymog", "model": model,
                                           "foo": 1})
        time.sleep(0.5)
        text = read_log()
        check("[req-unused] foo" in text, "C1 [req-unused] foo 出现",
              "HTTP %s" % st)
        check(bool(re.search(r"^\[req\] .*\"foo\": 1", text, re.M)),
              "C2 [req] 里有 foo 的值（未脱敏的普通字段）")

        head("D  脱敏：路径只留文件名、base64 换长度")
        st, d = post("/api/merge/export", {
            "video": str(clip), "fused": 1, "dry_run": 1, "detector": "tinymog",
            "model": model, "faces_project": "C:\\Users\\30764\\faces_project",
            "blob": "data:image/png;base64," + "QUJD" * 200, "bar": 1})
        time.sleep(0.5)
        text = read_log()
        reqs = [l for l in text.splitlines() if l.startswith("[req] ")]
        check(bool(reqs), "D1 有 [req] 行", "%d 条" % len(reqs))
        if reqs:
            last = reqs[-1]
            check('"video": "clip12.mp4"' in last, "D2 视频路径只留文件名",
                  last[:180])
            check('"faces_project": "faces_project"' in last, "D3 faces_project 只留目录名",
                  last[:180])
            check("C:\\Users" not in last and "C:\\\\Users" not in last,
                  "D4 [req] 里没有 C:\\Users", last[:180])
            check("<base64" in last and "QUJDQUJD" not in last,
                  "D5 base64 换成长度", last[:180])

        head("E  六阶段链：/api/merge/stage/frames 也要有 [eff]")
        st, d = post("/api/merge/stage/frames", {
            "video": str(clip), "detector": "tinymog", "model": model,
            "settings": SETTINGS})
        check(st == 200, "E1 frames 阶段启动", "HTTP %s" % st)
        time.sleep(0.8)
        blocks = eff_blocks(read_log())
        stg = [b for b in blocks if str(b["eff"].get("kind")).startswith("stage:")]
        check(bool(stg), "E2 有 stage [eff] 行", "%d 条" % len(stg))
        if stg:
            e = stg[-1]["eff"]
            check(e.get("detector") == "tinymog" and e.get("detector_src") == "ui",
                  "E3 阶段 eff detector", "%s/%s" % (e.get("detector"),
                                                     e.get("detector_src")))
        s = wait_terminal("/api/merge/stage/status")
        check((s.get("eff") or {}).get("detector") == "tinymog",
              "E4 stage/status 回 eff", str((s.get("eff") or {}).get("detector")))
        check(isinstance(s.get("warnings"), list), "E5 stage/status 有 warnings[]",
              str(s.get("warnings"))[:120])

    finally:
        try:
            proc.kill()
        except Exception:                                   # noqa: BLE001
            pass
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                                   # noqa: BLE001
            pass

    head("结果")
    bad = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + detail if detail else ""))
    print("")
    print("%d / %d 通过" % (len(RESULTS) - len(bad), len(RESULTS)))
    if SHOT.is_file():
        print("截图：%s" % SHOT)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
