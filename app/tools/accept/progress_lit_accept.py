# -*- coding: utf-8 -*-
r"""导出进度页「流水线实况」验收：6 列 lane + 贝塞尔曲线 + 光点 + 卡住 + 抽屉。

这一版验的东西（对应用户逐轮确认过的规格）：
  A  后端：status.stages 六个键齐全（frames/det/dfm/xseg/compose/encode）；
     lanes 数量 == **实际线程数**（合成池 8 → 8 条 t0..t7、编码 6 → 6 条 e0..e5、
     dfm/xseg/det 各一个 session → 1 条）；ticks 单调不减；inflight 是整数；
     并且拿后端日志里那行「线程 检测…/合成N/编码M」当第二把尺子对一遍。
  B  DOM：没有 #mgpEff（常驻面板已拆），有 #mgpEffWarn（告警横幅）；
     右下角「查看生效配置」→ 抽屉里 14 个字段 + 来源标签。
  C  图：六列标题、lane 方块数 == 所有 lanes 之和；相邻两列**共用一个腰点**
     （源 lane → 腰点 → 目标 lane，条数 = 源+目标，不再是交叉的 N×M），每条只是一段
     贝塞尔（d 里一个 C）、进出腰点坐标集合完全相同；光点从源 lane 经腰点走到目标 lane，
     密度按「速率 240 个/秒 + 同屏 200」两个物理上限（不再按每次轮询硬切 10 个），
     轮询窗口 250ms 且光点铺在窗口里；lane 点亮 → 1.15s 暗回；>600ms 没 tick 的列整列琥珀、恢复后复原；
     **图上没有「在飞 N」那行字**（用户要求删掉，inflight 仍在 status 里）。
  D  减动效：prefers-reduced-motion:reduce 时点亮动画是 none。
  E  复用：同一份 lanes 连喂两拍，SVG 节点身份不变（不重建 DOM）。

截图（%TEMP%/dfl_progress_lit_accept/）：prog-live.png / prog-stall.png / prog-done.png

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\progress_lit_accept.py
退出码 0 = 全过。

注意（用户抱怨过机器卡）：
  · 整轮只起一个无头 Edge，跑完立刻 taskkill；
  · 后端只真跑一单导出（72 帧 320x240），拿到真实 status 之后浏览器那半段全是
    喂数据，不再为了截图起第二单。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_progress_lit_accept"
WS = BOX / "ws"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

FIELDS = ("detector", "mask_source", "ct_mode", "blend", "erode", "blur", "models",
          "faces_project", "library_enabled", "fast_export", "frame_range",
          "workers", "batch", "jpeg_quality")
STAGES = ("frames", "det", "dfm", "xseg", "compose", "encode")

# 这一单要的线程数（写进请求体，后端按它起线程 —— 断言就断「lanes == 这个数」）
WANT_WORKERS = 8       # 合成池
WANT_WRITERS = 6       # 编码线程


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


# ============================================================
# 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        try:
            os.rmdir(str(WS / "model"))          # 目录联接：先拆再删，否则 rmtree 会把真模型删了
        except OSError:
            pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / "model"),
                        str(REPO / "workspace" / "model")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
    say("沙箱：%s" % BOX)


def make_clip():
    """72 帧 320x240（3 秒 24fps）：让流水线真的跑起来、能采到「在飞」的那几拍。

    素材优先用训练预览图（4x16 的小脸网格）裁一小块再放大 —— 有脸，换脸/遮罩两段才
    会有 tick（彩条测试图里一张脸都没有，六列里那两列几乎不动，截图看着像坏了）。
    预览图不存在就退回 testsrc：那是数据真没脸，不是图坏了。
    这张图只当「有脸的素材」，不参与任何断言。
    """
    dst = BOX / "clip72.mp4"
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    face = REPO / "workspace" / "model" / "_smoketrain_SAEHD_preview.jpg"
    if face.is_file():
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-loop", "1",
               "-i", str(face), "-t", "3", "-r", "24",
               "-vf", "crop=160:120:8:8,scale=320:240:flags=lanczos",
               "-pix_fmt", "yuv420p", str(dst)]
        kind = "训练预览图裁块（有脸）"
    else:
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=3",
               "-pix_fmt", "yuv420p", str(dst)]
        kind = "testsrc 彩条（无脸）"
    subprocess.run(cmd, check=True)
    say("素材：%s（%d 字节 · %s）" % (dst.name, dst.stat().st_size, kind))
    return dst


def pick_dfm():
    cands = sorted((REPO / "workspace" / "model").glob("*.dfm"))
    if not cands:
        raise SystemExit("workspace/model 里没有 .dfm")
    pick = [c for c in cands if "416" in c.stem] or cands
    return pick[0].stem


def start_server():
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    logf = open(str(SERVER_LOG), "ab")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                                # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                            # noqa: BLE001
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
        except Exception:                                    # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def read_log():
    try:
        return SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_eff():
    """后端打的那行 [eff] 就是「同一份配置」，直接拿来喂前端（不自己造 fixture）。"""
    best = None
    for line in read_log().splitlines():
        if line.startswith("[eff] "):
            try:
                best = json.loads(line[len("[eff] "):])
            except Exception:                                # noqa: BLE001
                pass
    return best


_THREADS_RE = re.compile(r"线程 检测(\d+)/解码(\d+)/对齐(\d+)/GPU(\d+)/合成(\d+)/编码(\d+)")


def threads_of(lines):
    """从任意几行日志里捞出「快速导出：线程 检测1/解码1/对齐4/GPU1/合成8/编码6」→ dict。

    为什么要它：lanes 必须等于**实际**线程数，光看 status 自己说自己对不算数 ——
    这里拿后端起线程时打的那行当第二把尺子。
    去哪找：这行走 _vid_log（进 <成片>.log + status.log），**不进进程 stdout**，
    所以从轮询到的 status.log / log_tail 里捞（status 尾巴只留 50 行，得边轮边捞）。
    """
    hit = None
    for line in (lines or []):
        m = _THREADS_RE.search(str(line))
        if m:
            hit = {"det": int(m.group(1)), "frames": int(m.group(2)),
                   "prep": int(m.group(3)), "gpu": int(m.group(4)),
                   "compose": int(m.group(5)), "encode": int(m.group(6))}
    return hit


# ============================================================
# 浏览器（Edge headless + CDP）
# ============================================================
def edge_session():
    """起一个无头 Edge，返回 (proc, cm, call, ev)。"""
    from websockets.sync.client import connect
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    profile = BOX / ("edge_profile_%d_%s" % (os.getpid(), uuid.uuid4().hex[:6]))
    proc = subprocess.Popen(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--window-size=1680,1100", "--no-first-run", "--no-default-browser-check",
         "--disable-extensions", "--remote-allow-origins=*",
         "--user-data-dir=%s" % profile, "--remote-debugging-port=%d" % port,
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000)
    ws_url = ""
    deadline = time.time() + 60
    while time.time() < deadline and not ws_url:
        try:
            raw = OPENER.open("http://127.0.0.1:%d/json/list" % port, timeout=3).read()
            tabs = json.loads(raw.decode("utf-8"))
            pages = [t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            pages.sort(key=lambda t: 0 if str(t.get("url") or "").startswith("about:") else 1)
            if pages:
                ws_url = pages[0]["webSocketDebuggerUrl"]
        except Exception:                                    # noqa: BLE001
            time.sleep(0.4)
    if not ws_url:
        raise SystemExit("连不上无头 Edge 的调试口")

    cm = connect(ws_url, max_size=None, open_timeout=20)
    ws = cm.__enter__()
    seq = [0]

    def call(method, params=None, recv_timeout=90):
        seq[0] += 1
        mid = seq[0]
        ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(ws.recv(timeout=recv_timeout))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("%s: %s" % (method, msg["error"]))
                return msg.get("result") or {}

    def ev(expr, timeout=120):
        res = call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                        "awaitPromise": True,
                                        "timeout": int(timeout * 1000)},
                   recv_timeout=timeout + 30)
        if res.get("exceptionDetails"):
            raise RuntimeError(json.dumps(res["exceptionDetails"], ensure_ascii=False)[:400])
        return (res.get("result") or {}).get("value")

    call("Page.enable")
    call("Runtime.enable")
    call("Emulation.setDeviceMetricsOverride", {"width": 1680, "height": 1100,
                                                "deviceScaleFactor": 1, "mobile": False})
    return proc, cm, call, ev


def boot_progress(ev, call):
    """打开页面 -> 进合成页 -> 切到进度页。返回 True/False。"""
    call("Page.navigate", {"url": BASE + "/"})
    ok = False
    for _ in range(180):
        time.sleep(0.4)
        try:
            if ev("typeof window.go === 'function'"):
                ok = True
                break
        except Exception:                                    # noqa: BLE001
            pass
    if not ok:
        return False
    ev("window.go('merge'); true")
    ok = False
    for _ in range(120):
        time.sleep(0.4)
        try:
            if ev("typeof window.mgpPaint === 'function' && "
                  "typeof window.mgpEffRender === 'function' && "
                  "typeof window.mgpPipe === 'function'"):
                ok = True
                break
        except Exception:                                    # noqa: BLE001
            pass
    if not ok:
        return False
    ev("window.mgStep('progress'); true")
    # 首启欢迎层（扫码/法务）会盖住整页：它和这一屏无关，收起来
    ev("(function(){var w=document.getElementById('welcome'); if(w) w.style.display='none';"
       "var lg=document.getElementById('legalGate'); if(lg) lg.style.display='none';"
       "return true;})()")
    time.sleep(0.4)
    return True


def ev_json(call, expr):
    res = call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                    "awaitPromise": True, "timeout": 60000})
    if res.get("exceptionDetails"):
        raise RuntimeError(json.dumps(res["exceptionDetails"], ensure_ascii=False)[:400])
    return (res.get("result") or {}).get("value")


def rect_of(call, selector):
    """量一个元素的框（截图前先量好，别在点亮只剩几百毫秒时再往返一次）。"""
    try:
        raw = ev_json(call, "(function(){var e=document.querySelector(%s);"
                            "if(!e) return '';var r=e.getBoundingClientRect();"
                            "return JSON.stringify({x:r.x,y:r.y,width:r.width,"
                            "height:r.height,scale:1});})()" % json.dumps(selector))
        return json.loads(raw) if raw else None
    except Exception:                                        # noqa: BLE001
        return None


def shot(call, name, selector="", clip=None):
    """按元素裁一张；selector 空 = 整屏。

    两个坑（上一版踩过）：
      · captureBeyondViewport=true + clip 在 headless 下会卡到超时 —— 元素本来就在视口里，
        不带它一样能裁；
      · 截图失败不该把整轮断言带走 —— 打不出来就记一张整屏的退路。
    """
    import base64 as b64
    out = BOX / name
    plans = []
    if selector and not clip:
        clip = rect_of(call, selector)
    if clip and clip.get("width") and clip.get("height"):
        c = dict(clip)
        c["scale"] = 1
        plans.append({"format": "png", "clip": c})
    plans.append({"format": "png"})
    err = None
    for _try in range(2):                       # headless 偶发 capture 卡住：再试一次
        for params in plans:
            try:
                res = call("Page.captureScreenshot", params, recv_timeout=60)
                data = b64.b64decode(res.get("data") or "")
                if data:
                    out.write_bytes(data)
                    return out
            except Exception as e:                           # noqa: BLE001
                err = e
        if _try == 0:
            say("截图 %s 第一次没拿到（%s），重试" % (name, err))
            time.sleep(0.4)
    say("截图失败（%s）：%s" % (name, err))
    return None


# ============================================================
# 页面里的小工具（注入一次）
# ============================================================
INIT_JS = r"""
(function(){
  if (!window.__litProbe) {
    window.__litProbe = { starts: 0 };
    var box = document.getElementById('mgPipeSvg');
    box.addEventListener('animationstart', function(){ window.__litProbe.starts += 1; }, true);
  }
  return true;
})()
"""


def probe(ev):
    return int(ev("window.__litProbe.starts") or 0)


def wait_lit(ev, call, want, tries=8):
    """等「亮着的 lane 条数」达到要求（want=0 表示等它全暗回去）。

    为什么要轮询：点亮是 rAF 里挂的类，一轮 rAF 不一定在喂完数据的那一帧就跑；
    固定 sleep 要么太短（断言偶发失败）要么太长（拖时间）。这里最多等约 1s。
    """
    n = -1
    for _ in range(tries):
        flush_frames(ev, 2)
        n = int(ev_json(call, "(function(){return document.querySelectorAll("
                               "'#mgPipeSvg rect.lane.lit').length;})()") or 0)
        if (want == 0 and n == 0) or (want > 0 and n >= want):
            return n
        time.sleep(0.08)
    return n


def feed_and_probe(ev, call, live_status, live, want_lit=True, want_pulse=True,
                   tries=4):
    """喂一拍带增量的 status，走一帧，再在**同一批**里读「亮了几条 / 几个光点」。

    为什么不在两步之间 sleep：点亮挂类和光点推进都在 rAF 里，而 headless 的 CDP
    往返随便就几百毫秒 —— 固定 sleep 会让「点亮」衰减完、「光点」也跑完，读出来是 0
    （偶发就是这么来的）。这里喂完立刻走一帧、立刻读，读不到就多一点增量再来。
    返回 (lit_n, pulses_n, fed_status)。
    """
    lit_n, p_n, b = 0, 0, live_status
    for i in range(tries):
        b = json.loads(json.dumps(live_status))
        for k in STAGES:
            b["stages"][k]["ticks"] = int((live["stages"].get(k) or {}).get("ticks")
                                          or 0) + 1 + i
        ev("window.mgpPipe(%s); true" % json.dumps(b, ensure_ascii=False))
        flush_frames(ev, 1)
        out = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var dbg = window.__mgpPipe ? window.__mgpPipe() : {};
          return JSON.stringify({lit: svg.querySelectorAll('rect.lane.lit').length,
                                 pulses: (dbg && dbg.pulses) || 0});
        })()"""))
        lit_n = int(out.get("lit") or 0)
        p_n = int(out.get("pulses") or 0)
        if (not want_lit or lit_n > 0) and (not want_pulse or p_n > 0):
            return lit_n, p_n, b
        time.sleep(0.05)
    return lit_n, p_n, b


def wait_pulses(ev, call, want=1, tries=8):
    """等「正在跑的光点」达到 want 个（0 表示等它们跑完/收干净）。"""
    n = -1
    for _ in range(tries):
        flush_frames(ev, 2)
        n = int(json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())"))
                .get("pulses") or 0)
        if (want == 0 and n == 0) or (want > 0 and n >= want):
            return n
        time.sleep(0.06)
    return n


def flush_frames(ev, n=2):
    """等 n 个 rAF 真的跑完。

    为什么要它：点亮（classList 摘/挂）和光点（circle 的 cx/cy）都是 rAF 里做的。
    喂完数据只 sleep 的话，无头 Edge 不一定会派发帧（Runtime.evaluate 会），
    截图就会拍到「动画第 0 毫秒」——看起来像根本没点亮。
    """
    try:
        ev("new Promise(function(res){var k=0;(function s(){"
           "if(++k>=%d) return res(1); requestAnimationFrame(s);})();})" % int(n))
    except Exception:                                        # noqa: BLE001
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    # 只重拍截图：复用上一次真跑留下的 status（不跑导出、不起流水线）。
    # 用户有「别反复起无头浏览器」的要求，改样式后重拍就用它，别整轮重跑。
    ap.add_argument("--shots-only", action="store_true")
    args = ap.parse_args(argv)

    if args.shots_only:
        # 只重拍截图：**不能**清沙箱（那些 status/eff/threads 就是这次要复用的现场）
        for d in (WS, LOGDIR):
            d.mkdir(parents=True, exist_ok=True)
    else:
        prepare_sandbox()
    proc = start_server()
    edge = None
    ws = None
    shots = {}
    try:
        # ---------------------------------------------------------------
        head("A  后端：跑一单真导出，把 status.stages 采下来")
        clip = make_clip()
        model = pick_dfm()
        samples = []
        final = {}
        th = None
        if args.shots_only:
            # 复用上一次真跑留下的两拍 status（终态 + 中段）+ [eff] + 线程数那行
            need = ("status-final.json", "status-live.json", "eff.json", "threads.json")
            miss = [n for n in need if not (BOX / n).is_file()]
            if miss:
                raise SystemExit("--shots-only 缺文件（先跑一次完整验收）：%s"
                                 % "、".join(miss))
            final = json.loads((BOX / "status-final.json").read_text(encoding="utf-8"))
            _live = json.loads((BOX / "status-live.json").read_text(encoding="utf-8"))
            th = json.loads((BOX / "threads.json").read_text(encoding="utf-8"))
            samples = [{"at": 0.0, "running": True, "stages": _live.get("stages") or {}},
                       {"at": 0.1, "running": False, "stages": final.get("stages") or {}}]
            say("--shots-only：复用 %s 里的 status（不跑导出）" % BOX.name)
        else:
            st, d = http("POST", "/api/merge/export", {
                "video": str(clip), "fused": 1, "detector": "tinymog", "model": model,
                "merge_workers": WANT_WORKERS, "pred_writers": WANT_WRITERS,
                "settings": {"mask_mode": 8, "mask_source": "XSegLite",
                             "color_transfer": "mt", "blend": "seamless",
                             "mask_erode": 0, "mask_blur": 4}})
            check(st == 200 and (d or {}).get("ok") is not False,
                  "A0 快速导出已起单",
                  "HTTP %s · %s" % (st, str((d or {}).get("error") or "")[:120]))

            t0 = time.time()
            while time.time() - t0 < 240:
                _s = http("GET", "/api/merge/export/status")[1] or {}
                th = threads_of(_s.get("log")) or threads_of(_s.get("log_tail")) or th
                if _s.get("stages"):
                    samples.append({"at": round(time.time() - t0, 2),
                                    "running": bool(_s.get("running")),
                                    "stages": _s.get("stages")})
                if not _s.get("running") and not _s.get("active"):
                    final = _s
                    break
                time.sleep(0.15)
            say("采样 %d 拍 · 终态 running=%s ok=%s"
                % (len(samples), final.get("running"), final.get("ok")))
            (BOX / "status-final.json").write_text(
                json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
            if samples:
                (BOX / "status-samples.json").write_text(
                    json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")

        check(bool(samples), "A1 status 里出现了 stages（有数据才画得出来）",
              "%d 拍" % len(samples))
        if not samples:
            return finish(shots)

        # ---- 六个键齐全、形状对 ----
        first = samples[0]["stages"]
        check(sorted(first.keys()) == sorted(STAGES),
              "A2 stages 六个键齐全（frames/det/dfm/xseg/compose/encode）",
              str(sorted(first.keys())))
        shape_bad = []
        for k in STAGES:
            row = first.get(k) or {}
            if not isinstance(row.get("ticks"), int):
                shape_bad.append(k + ".ticks")
            if not isinstance(row.get("inflight"), int):
                shape_bad.append(k + ".inflight")
            if not isinstance(row.get("lanes"), list):
                shape_bad.append(k + ".lanes")
        check(not shape_bad, "A3 ticks/inflight 是整数、lanes 是数组", ",".join(shape_bad))
        neg = [k for k in STAGES
               for s in samples
               if int((s["stages"].get(k) or {}).get("inflight") or 0) < 0]
        check(not neg, "A4 inflight 不为负", str(sorted(set(neg))))

        # ---- ticks 单调不减 + 真的在涨 ----
        bad = []
        for k in STAGES:
            last = None
            for s in samples:
                v = int((s["stages"].get(k) or {}).get("ticks") or 0)
                if last is not None and v < last:
                    bad.append("%s %d->%d" % (k, last, v))
                last = v
        check(not bad, "A5 ticks 单调不减（只增）", "; ".join(bad[:4]))
        grew = [k for k in STAGES
                if int((samples[-1]["stages"].get(k) or {}).get("ticks") or 0)
                > int((samples[0]["stages"].get(k) or {}).get("ticks") or 0)]
        check(len(grew) >= 4, "A6 至少四段真的在涨（流水线不是摆设）", str(grew))

        # ---- lanes = 实际线程数 ----
        last_st = samples[-1]["stages"]
        lanes = {k: list((last_st.get(k) or {}).get("lanes") or []) for k in STAGES}
        check(len(lanes["compose"]) == WANT_WORKERS,
              "A7 合成 lane 数 == 合成池线程数（%d）" % WANT_WORKERS,
              "%d 条 · %s" % (len(lanes["compose"]), lanes["compose"]))
        check(len(lanes["encode"]) == WANT_WRITERS,
              "A8 编码 lane 数 == 编码线程数（%d）" % WANT_WRITERS,
              "%d 条 · %s" % (len(lanes["encode"]), lanes["encode"]))
        check(lanes["dfm"] == ["dfm#0"], "A9 dfm 一个 session = 1 条 lane",
              str(lanes["dfm"]))
        check(len(lanes["det"]) == 1 and len(lanes["xseg"]) == 1
              and len(lanes["frames"]) == 1,
              "A10 det / xseg / frames 各 1 条 lane",
              "det=%s xseg=%s frames=%s" % (lanes["det"], lanes["xseg"], lanes["frames"]))
        check(lanes["compose"] == ["t%d" % i for i in range(WANT_WORKERS)]
              and lanes["encode"] == ["e%d" % i for i in range(WANT_WRITERS)],
              "A11 lane 名字按序号（t0.. / e0..）",
              "%s / %s" % (lanes["compose"][:3], lanes["encode"][:3]))
        check(bool(th) and th.get("compose") == len(lanes["compose"])
              and th.get("encode") == len(lanes["encode"]),
              "A12 和后端日志那行「合成N/编码M」对得上（第二把尺子）",
              "log=%s status=%d/%d" % (th, len(lanes["compose"]), len(lanes["encode"])))

        # ---- 「在飞」真的出现过 ----
        inflight_max = max([int((s["stages"].get(k) or {}).get("inflight") or 0)
                            for s in samples for k in STAGES] or [0])
        check(inflight_max > 0, "A13 采样里出现过 inflight > 0（队列真有在飞）",
              "max=%d" % inflight_max)

        # 挑一拍做前端输入的「进行中」素材：取中间那拍（六段都在动）。
        # --shots-only 时只有两拍（盘上的中段 + 终态），中段就是 samples[0]。
        if args.shots_only:
            live = samples[0]
        else:
            live = samples[min(len(samples) - 1, max(0, len(samples) // 2))]
        live_status = dict(final)
        live_status.update({"running": True, "ok": None, "stages": live["stages"],
                            "phase": "running", "pct": 45, "video": str(clip),
                            "vid": {"frame_fps": 11.3, "total": 72, "done": 32}})
        (BOX / "status-live.json").write_text(
            json.dumps(live_status, ensure_ascii=False, indent=1), encoding="utf-8")

        eff = last_eff()
        if args.shots_only:
            eff = json.loads((BOX / "eff.json").read_text(encoding="utf-8")) or None
        check(bool(eff) and all(k in (eff or {}) for k in FIELDS),
              "A14 拿到 [eff] 回显（14 个字段齐全）",
              "keys=%d" % len((eff or {}).keys()))
        if not args.shots_only:
            # 给 --shots-only 留现场：以后改样式重拍截图，不用再跑一遍导出
            (BOX / "eff.json").write_text(json.dumps(eff or {}, ensure_ascii=False),
                                          encoding="utf-8")
            (BOX / "threads.json").write_text(json.dumps(th or {}, ensure_ascii=False),
                                              encoding="utf-8")
        hints = [{"code": "eff-inferred", "field": "detector", "level": "warn",
                  "text": "detector 由 quality 推出来（界面没给）"}]

        if args.no_browser:
            say("--no-browser：不跑浏览器那一段")
            return finish(shots)

        # ---------------------------------------------------------------
        head("B  DOM：常驻面板没了、横幅还在、抽屉能开")
        edge, ws, call, ev = edge_session()
        if not boot_progress(ev, call):
            check(False, "B0 页面就绪",
                  "没进到合成页/进度页（window.mgpPipe 没挂上）")
            return finish(shots)
        ev(INIT_JS)

        check(ev("document.getElementById('mgpEff') === null") is True,
              "B1 DOM 里没有 #mgpEff（常驻面板已拆）")
        check(ev("!!document.getElementById('mgpEffWarn')") is True,
              "B2 告警横幅 #mgpEffWarn 还在")
        dead = ev_json(call, r"""(function(){
          var bad = [];
          for (var i=0;i<document.styleSheets.length;i++){
            var ss = document.styleSheets[i], rs;
            try { rs = ss.cssRules; } catch(e) { continue; }
            for (var j=0;j<rs.length;j++){
              var sel = rs[j] && rs[j].selectorText;
              if (!sel) continue;
              if (sel.indexOf('#mgpEff') >= 0) bad.push(sel);
            }
          }
          return JSON.stringify(bad.slice(0, 6));
        })()""")
        check(dead == "[]", "B3 没有 #mgpEff 的死 CSS", str(dead))

        check(ev("document.getElementById('mgpEffOpen').disabled") is True,
              "B4 没有回显时入口是禁用的")
        ev("(function(){window.mgpEffRender(%s, %s); return true;})()"
           % (json.dumps(eff, ensure_ascii=False), json.dumps(hints, ensure_ascii=False)))
        time.sleep(0.3)
        check(ev("document.getElementById('mgpEffOpen').disabled") is False,
              "B5 有回显后入口启用")
        check(ev("document.getElementById('mgpEffDrawer').hidden") is True,
              "B6 抽屉默认关着（不是常驻面板）")
        ev("document.getElementById('mgpEffOpen').click(); true")
        time.sleep(0.4)
        drawer = json.loads(ev_json(call, r"""(function(){
          var b = document.getElementById('mgpEffDrawer');
          var g = document.getElementById('mgpEffGrid');
          return JSON.stringify({
            open: !!b && !b.hidden,
            items: g ? g.children.length : 0,
            keys: g ? Array.prototype.map.call(g.children,
                     function(n){ return n.getAttribute('data-k'); }) : [],
            srcs: g ? Array.prototype.map.call(g.querySelectorAll('.mgeff-i .s'),
                     function(n){ return String(n.textContent); }) : []
          });
        })()"""))
        check(drawer.get("open") is True, "B7 点开后抽屉可见")
        check(int(drawer.get("items") or 0) == len(FIELDS), "B8 抽屉里 14 个字段",
              "items=%s" % drawer.get("items"))
        check(list(drawer.get("keys") or []) == list(FIELDS), "B9 字段顺序/键名对得上",
              str(drawer.get("keys"))[:120])
        check(all(s in ("界面", "推断", "默认") for s in (drawer.get("srcs") or [])),
              "B10 每行都有来源小标签", str(sorted(set(drawer.get("srcs") or []))))
        ev("document.getElementById('mgpEffClose').click(); true")
        time.sleep(0.2)

        # ---------------------------------------------------------------
        head("C  图：六列 / lane 数 / 腰点收束的连线 / 光点 / 点亮 / 卡住")
        ev("window.mgpReset(); true")
        ev("window.mgpPipe(%s); true" % json.dumps(live_status, ensure_ascii=False))
        time.sleep(0.5)
        dom = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var st = svg.querySelectorAll('text.t-stage');
          var lanes = svg.querySelectorAll('rect.lane');
          var off = svg.querySelectorAll('rect.lane.off');
          var paths = svg.querySelectorAll('g.edges path');
          var one = 0, bad = 0, ds = [], starts = [], ends = [];
          function nums(s){                     /* 抽出一段 d 里的所有数字（字母/逗号都当分隔） */
            var out = [], parts = String(s).split(/[^0-9.\-]+/);
            for (var q=0;q<parts.length;q++){
              if (!parts[q]) continue;
              var v = parseFloat(parts[q]);
              if (isFinite(v)) out.push(v);
            }
            return out;
          }
          function pt(s){                       /* 最后两个数 = 这条路径的终点 */
            var n = nums(s);
            return { x: n[n.length-2], y: n[n.length-1] };
          }
          function head(s){                     /* "M x,y" 那一段的坐标（起点） */
            return pt(String(s).slice(0, String(s).indexOf('C') + 1));
          }
          for (var i=0;i<paths.length;i++){
            var d = paths[i].getAttribute('d') || '';
            ds.push(d);
            if (d.indexOf('C') >= 0 && d.lastIndexOf('C') === d.indexOf('C')) one += 1;
            else bad += 1;
            starts.push(head(d));
            ends.push(pt(d));
          }
          function key(p){ return Math.round(p.x*10)/10 + ',' + Math.round(p.y*10)/10; }
          var uk = {};
          for (var s=0;s<starts.length;s++) uk[key(starts[s])] = 1;
          var ek = {};
          for (var e=0;e<ends.length;e++) ek[key(ends[e])] = 1;
          var cnt = [];
          var cs = svg.querySelectorAll('text.t-count');
          for (var k=0;k<cs.length;k++) cnt.push(String(cs[k].textContent));
          return JSON.stringify({
            cols: st.length, colTxt: Array.prototype.map.call(st,
                    function(n){ return String(n.textContent); }),
            lanes: lanes.length, off: off.length,
            paths: paths.length, one: one, bad: bad,
            starts: uk, ends: ek,
            queueNodes: svg.querySelectorAll('text.t-queue').length,
            queueTxt: String(svg.textContent).indexOf('在飞') >= 0,
            counts: cnt, firstD: ds[0] || ''
          });
        })()"""))
        want_lanes = sum(len(lanes[k]) for k in STAGES)
        check(int(dom.get("cols")) == 6, "C1 六个列标题", str(dom.get("colTxt")))
        check(int(dom.get("lanes")) == want_lanes,
              "C2 lane 方块数 == status 里 lanes 的总数（不写死）",
              "DOM %s / status %d" % (dom.get("lanes"), want_lanes))
        check(int(dom.get("off")) == 0, "C3 这一单六列都有真 lane（没有占位格）",
              "off=%s" % dom.get("off"))
        # 腰点几何：相邻两列共用 1 个腰点 → 条数 = 源 lane 数 + 目标 lane 数
        want_paths = 0
        for i in range(len(STAGES) - 1):
            want_paths += len(lanes[STAGES[i]]) + len(lanes[STAGES[i + 1]])
        want_waists = len(STAGES) - 1
        check(int(dom.get("paths")) == want_paths,
              "C4 连线条数 = Σ(源 lane + 目标 lane)（合成8×导出6 的 48 条交叉没了）",
              "DOM %s / 期望 %d（旧写法会是 %d）"
              % (dom.get("paths"), want_paths,
                 sum(len(lanes[STAGES[i]]) * len(lanes[STAGES[i + 1]])
                     for i in range(len(STAGES) - 1))))
        check(int(dom.get("one")) == int(dom.get("paths")) and int(dom.get("bad")) == 0,
              "C5 每条连线都是单段三次贝塞尔（d 里一个 C）",
              "paths=%s one=%s bad=%s" % (dom.get("paths"), dom.get("one"), dom.get("bad")))
        # 腰点 = 既出现在「起点集合」又出现在「终点集合」里的那些坐标：
        #   源侧：起点 = lane 边、终点 = 腰点；目标侧：起点 = 腰点、终点 = lane 边。
        #   所以 lane 边只会在其中一侧出现，而腰点两侧都有 —— 交集恰好就是腰点。
        starts_k = set((dom.get("starts") or {}).keys())
        ends_k = set((dom.get("ends") or {}).keys())
        shared = sorted(starts_k & ends_k)
        n_src = sum(len(lanes[STAGES[i]]) for i in range(len(STAGES) - 1))
        n_tgt = sum(len(lanes[STAGES[i + 1]]) for i in range(len(STAGES) - 1))
        check(len(starts_k) == n_src + want_waists
              and len(ends_k) == n_tgt + want_waists,
              "C5b 起点/终点个数 = (源 lane 数 + 腰点数) / (目标 lane 数 + 腰点数)",
              "starts=%d(期望 %d) ends=%d(期望 %d)"
              % (len(starts_k), n_src + want_waists,
                 len(ends_k), n_tgt + want_waists))
        check(len(shared) == want_waists,
              "C5c 收束到 %d 个腰点，且进出**共用同一个**点（起点∩终点 = 腰点）"
              % want_waists,
              "交集 %d 个：%s" % (len(shared), shared[:3]))
        check(int(dom.get("queueNodes") or 0) == 0 and dom.get("queueTxt") is False,
              "C5d 没有「在飞 N」那行字（.t-queue 节点 0 个、SVG 里没有「在飞」）",
              "nodes=%s txt=%s" % (dom.get("queueNodes"), dom.get("queueTxt")))

        # 第一拍 prev 是空的（= 重连语义：不补放历史光点），所以先喂一拍**增量**再断光点
        big = json.loads(json.dumps(live_status))
        for k in STAGES:
            big["stages"][k]["ticks"] = int((live["stages"].get(k) or {}).get("ticks") or 0) + 2
        _lit_n, p_n, _fed = feed_and_probe(ev, call, live_status, live,
                                           want_lit=False, want_pulse=True)
        pipe = json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())"))
        check(int(p_n) > 0, "C6 有 tick 增量就真的有光点在跑", "pulses=%s" % p_n)
        # C7 的口径（用户要求「把上限去掉、闪烁要跟真实交接一样密」）：
        #   旧：同屏 60 个、每次轮询最多 10 个（TICK_MAX）—— 那条上限把「20 帧/秒」也一起
        #       砍了（500ms × 20fps = 每段正好 10 个），而且 10 个全在同一瞬间放出来，
        #       肉眼就是「每半秒跳一波」。
        #   新：两个**物理上限**，都不是为了让测试变绿：
        #       · 同屏在飞 MAX_PULSE = 200（≈ 40 帧/秒 × 8 张脸 × 0.6s 飞行时间的极端值；
        #         常见档 20~30 帧/秒、每帧 1~4 张脸，在飞只有几十个，根本碰不到）；
        #       · 速率 PULSE_PER_S = 240 个/秒（按 250ms 窗口折算 = 每次轮询最多 60 个）。
        #   这一条同时锁住「上限值」和「轮询窗口已经降到 250ms」两个设计决定。
        check(int(pipe.get("pulses") or 0) <= int(pipe.get("max_pulse") or 200)
              and int(pipe.get("max_pulse") or 0) == 200,
              "C7 同屏在飞 ≤ MAX_PULSE=200（不再是 60）",
              "pulses=%s max_pulse=%s" % (pipe.get("pulses"), pipe.get("max_pulse")))
        check(int(pipe.get("poll_ms") or 0) == 250
              and int(pipe.get("pulse_per_s") or 0) == 240,
              "C7b 轮询窗口 250ms、速率上限 240 个/秒（光点铺在窗口里，不再一拍一堆）",
              "poll_ms=%s pulse_per_s=%s"
              % (pipe.get("poll_ms"), pipe.get("pulse_per_s")))
        on_curve = ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var paths = svg.querySelectorAll('path');
          var circles = svg.querySelectorAll('circle');
          if (!paths.length || !circles.length) return -1;
          var best = 1e9;
          for (var c=0;c<circles.length;c++){
            var cx = parseFloat(circles[c].getAttribute('cx'));
            var cy = parseFloat(circles[c].getAttribute('cy'));
            if (!isFinite(cx) || !isFinite(cy)) continue;
            for (var i=0;i<paths.length;i++){
              var L = paths[i].getTotalLength(), n = 24;
              for (var k=0;k<=n;k++){
                var pt = paths[i].getPointAtLength(L*k/n);
                var dd = Math.hypot(pt.x-cx, pt.y-cy);
                if (dd < best) best = dd;
              }
            }
          }
          return best;
        })()""")
        check(isinstance(on_curve, (int, float)) and float(on_curve) <= 30.0,
              "C8 光点落在那条贝塞尔曲线上（离最近采样点 <=30）", str(on_curve))

        # C9 的口径也变了，改的理由写在下面（不是「为了变绿把 60 改成 200」）：
        #   旧断言是「一次 delta 也只放 60 个」—— 那其实是 TICK_MAX=10 × 6 段的上限，
        #   它同时也是「20 帧/秒也只能 10 个/段」的罪魁祸首，跟用户这次的诉求直接冲突。
        #   新策略：**按速率预算**放（240 个/秒 × 250ms 窗口 = 每拍最多 60 个），超出就按
        #   比例稀释（各段的光点比例不变，不是一刀切成一样多），并发另有 200 的上限兜底。
        #   所以这条断言验的是两件事：
        #     ① 巨大 delta（每段 +2000 = 12000 次交接）也只放出预算内那么多（≤60）；
        #     ② 同时那些光点没有超过并发上限（≤200）。
        big2 = json.loads(json.dumps(live_status))
        for k in STAGES:
            big2["stages"][k]["ticks"] = int((live["stages"].get(k) or {}).get("ticks") or 0) + 2000
        ev("window.mgpPipe(%s); true" % json.dumps(big2, ensure_ascii=False))
        dump = json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())"))
        budget = int(round(240 * 250 / 1000.0))
        check(0 < int(dump.get("last_spawn") or 0) <= budget,
              "C9 巨大 delta（每段 +2000）也只放「速率预算」内那么多",
              "last_spawn=%s（预算 %d）· 交接总数 %d"
              % (dump.get("last_spawn"), budget, 2000 * len(STAGES)))
        check(int(dump.get("pulses") or 0) <= 200,
              "C9b 同时光点总数 ≤ 并发上限 200",
              "pulses=%s" % dump.get("pulses"))

        ev("window.__litProbe.starts = 0; true")
        base_starts = probe(ev)
        ev("window.mgpReset(); true")
        ev("window.mgpPipe(%s); true" % json.dumps(live_status, ensure_ascii=False))
        flush_frames(ev, 2)
        ev("window.mgpPipe(%s); true" % json.dumps(big, ensure_ascii=False))
        lit_n = wait_lit(ev, call, 1)     # 点亮在 rAF 里挂类：等到真的亮了再断言
        lit = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var r = svg.querySelector('rect.lane.lit');
          var cs = r ? getComputedStyle(r) : null;
          return JSON.stringify({lit: !!r, anim: cs ? cs.animationName : '',
                                 dur: cs ? cs.animationDuration : ''});
        })()"""))
        check(lit.get("lit") is True, "C10 有新 tick 的 lane 挂上了 .lit", str(lit))
        check(str(lit.get("anim") or "") not in ("", "none"), "C11 点亮是 CSS 动画",
              "%s" % lit.get("anim"))
        check(str(lit.get("dur") or "").startswith("1.15"), "C12 动画 1.15s（衰减回常态）",
              str(lit.get("dur")))
        flush_frames(ev, 2)
        starts_after = probe(ev)
        check(starts_after > base_starts, "C13 点亮真的触发了 animationstart",
              "%d -> %d" % (base_starts, starts_after))
        time.sleep(1.45)
        steady_n = wait_lit(ev, call, 0, tries=8)      # 最多再等 ~1s：1.25s 的摘类要跑完
        check(int(steady_n) == 0, "C14 1.25s 后摘掉 .lit（回稳态）",
              "lit=%s · 亮过 %s 条" % (steady_n, lit_n))

        # 「进行中」截图：喂一拍真实的中段 status，等光点跑起来再拍。
        # 走 mgpFusedPaint（= 页面真实的 paint 入口：进度条 / 帧数 / 文件名 / 六列状态 / 图）
        ev("window.mgpReset(); true")
        ev("window.mgpFusedPaint(%s); true" % json.dumps(live_status, ensure_ascii=False))
        time.sleep(0.2)
        # 这张照片要拍到「点亮衰减中 + 曲线上的光点」，而两者都比一次 headless 截图往返短
        # （点亮 1.15s、光点 470~750ms；实测连拍三次都只拿到「已经暗回去」的稳态）。
        # 所以**只在这一张照片**里按住两个时钟：
        #   · mgppipeLit 用 Web Animations 暂停在 6%（峰值 ≈70ms）；
        #   · 光点用 mgpPipeFreeze 钉在出生后约 45%（默认 0 = 正常跑，运行时路径不看它）。
        # 动画/光点本身是真的，被按住的只是这张照片的时钟 —— C10~C14、C6/C8 验的是真跑。
        clip_rect = rect_of(call, "#p-merge .mgview[data-view='progress'] .mgprog")
        # 定格在「起飞后 150~400ms」这个区间：光点铺在 250ms 的窗口里（最晚的 t0 = now+250），
        # 所以把假时钟推到 now+400，飞得最早的那个走到 ~85%、最晚的 ~27% —— 都还在空中，
        # 照片里是一串连续的点，而不是挤在 lane 出口。
        ev("window.mgpPipeFreeze(performance.now() + 400); true")
        frozen = 0
        # 点亮是在 rAF 里挂类的：喂完必须真的走几帧才有动画对象可暂停。偶尔那一帧
        # 还没跑到（实测约三成概率），所以最多试 3 次，每次多喂一拍 ticks。
        for _try in range(3):
            _big = json.loads(json.dumps(live_status))
            for _k in STAGES:
                _big["stages"][_k]["ticks"] = int((live["stages"].get(_k) or {})
                                                  .get("ticks") or 0) + 2 * (_try + 1)
            ev("window.mgpFusedPaint(%s); true" % json.dumps(_big, ensure_ascii=False))
            flush_frames(ev, 3)             # 这一帧：挂 .lit + 光点落到曲线上 45% 处
            # 定格手法：按「刚才真正亮过的那几条 lane」重新挂 .lit（时钟从 0 重启，
            # 应用自己的 1.25s 摘类定时器已经全部跑过了，不会再来抢），再用内联
            # animation-play-state:paused + delay:-70ms 摆在动画第 6% 的峰值。
            # 为什么不用 Web Animations 的 pause()+currentTime：headless 里那是
            # 「pending pause」，同一份代码有时拍到峰值、有时拍到稳态。内联 style 同步生效。
            last_fl = json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())")
                                 ).get("last_flashed") or []
            frozen = ev("(function(){var svg=document.getElementById('mgPipeSvg');"
                        "var all=svg.querySelectorAll('rect.lane');var list=%s;var n=0;"
                        "for(var i=0;i<list.length;i++){"
                        "var want=list[i].k+'#'+list[i].i,hit=null;"
                        "for(var r=0;r<all.length;r++){"
                        "if(all[r].getAttribute('data-lane')===want){hit=all[r];break;}}"
                        "if(!hit)continue;"
                        "hit.classList.remove('lit');void hit.getBoundingClientRect();"
                        "hit.classList.add('lit');"
                        "hit.style.animationPlayState='paused';hit.style.animationDelay='-70ms';n++;}"
                        "return n;})()" % json.dumps(last_fl, ensure_ascii=False)) or 0
            if frozen:
                break
        say("定格：点亮 %s 条（截完就恢复）" % frozen)
        shots["live"] = shot(call, "prog-live.png", clip=clip_rect)
        # 拍完立刻恢复：解开光点时钟、把暂停的动画取消（照片的时钟不能漏到后面的断言里）
        ev("window.mgpPipeFreeze(0); true")
        ev("(function(){var svg=document.getElementById('mgPipeSvg');"
           "var rs=svg.querySelectorAll('rect.lane.lit');"
           "for(var i=0;i<rs.length;i++) rs[i].classList.remove('lit');"
           "var all=svg.querySelectorAll('rect.lane');"
           "for(var k=0;k<all.length;k++){all[k].style.animationPlayState='';"
           "all[k].style.animationDelay='';}"
           "var l=document.getAnimations?document.getAnimations():[];"
           "for(var j=0;j<l.length;j++){try{l[j].cancel();}catch(e){}}"
           "return rs.length;})()")
        say("截图：%s" % shots["live"])

        # 卡住：同一份 status 隔 >600ms 再喂一次（ticks 没涨）→ 整列琥珀
        ev("window.mgpReset(); true")
        ev("window.mgpFusedPaint(%s); true" % json.dumps(live_status, ensure_ascii=False))
        time.sleep(0.75)
        ev("window.mgpFusedPaint(%s); true" % json.dumps(live_status, ensure_ascii=False))
        flush_frames(ev, 3)
        time.sleep(0.1)
        stall = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var st = svg.querySelectorAll('rect.lane.stall');
          var l0 = st[0];
          var cs = l0 ? getComputedStyle(l0) : null;
          return JSON.stringify({stalled: st.length,
                                 stroke: cs ? cs.stroke : '',
                                 fill: cs ? cs.fill : ''});
        })()"""))
        check(int(stall.get("stalled")) > 0, "C15 >600ms 没 tick 的列变琥珀（.stall）",
              "stalled=%s" % stall.get("stalled"))
        check("240" in str(stall.get("stroke")) or "180" in str(stall.get("stroke")),
              "C16 琥珀色描边走 --warn（240,180,41 / 浅色 194,116,11）",
              str(stall.get("stroke")))
        shots["stall"] = shot(call, "prog-stall.png",
                              "#p-merge .mgview[data-view='progress'] .mgprog")
        say("截图：%s" % shots["stall"])
        ev("window.mgpPipe(%s); true" % json.dumps(big, ensure_ascii=False))
        time.sleep(0.15)
        back = json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())"))
        check(not any(bool(v) for v in (back.get("stalled") or {}).values()),
              "C17 tick 一恢复，琥珀色就撤掉", str(back.get("stalled")))

        # 「在飞 N」已按用户要求删掉：把 inflight 喂成 5 也不许长出任何文字节点
        q_live = json.loads(json.dumps(live_status))
        q_live["stages"]["dfm"]["inflight"] = 5
        q_live["stages"]["det"]["inflight"] = 0
        before_txt = int(ev_json(call, "(function(){return document.querySelectorAll("
                                       "'#mgPipeSvg text').length;})()") or 0)
        ev("window.mgpPipe(%s); true" % json.dumps(q_live, ensure_ascii=False))
        time.sleep(0.15)
        now = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          return JSON.stringify({n: svg.querySelectorAll('text').length,
                                 inair: String(svg.textContent).indexOf('在飞') >= 0,
                                 q: svg.querySelectorAll('.t-queue').length});
        })()"""))
        check(int(now.get("n") or 0) == before_txt, "C18 inflight 变了也不多长文字节点",
              "text %s -> %s" % (before_txt, now.get("n")))
        check(now.get("inair") is False and int(now.get("q") or 0) == 0,
              "C19 整张图里没有「在飞」字样、没有 .t-queue 节点", str(now))

        # 节点复用：同一份 lanes 再喂一拍，lane 元素身份不变（没重建 DOM）
        ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var a = svg.querySelectorAll('rect.lane');
          window.__laneRef = a[a.length-1];
          return 'ok';
        })()""")
        ev("window.mgpPipe(%s); true" % json.dumps(q_live, ensure_ascii=False))
        time.sleep(0.05)
        same = ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var a = svg.querySelectorAll('rect.lane');
          return (a[a.length-1] === window.__laneRef) ? 'same' : 'rebuilt';
        })()""")
        check(same == "same", "C20 同一份 lanes 不重建 SVG 节点（复用）", str(same))

        # 完成态
        done_status = dict(final)
        done_status.update({"running": False, "ok": True, "phase": "done", "pct": 100,
                            "vid": {"frame_fps": 11.9, "total": 72, "done": 72}})
        ev("window.mgpChainDone(); true")
        ev("window.mgpFusedPaint(%s); true" % json.dumps(done_status, ensure_ascii=False))
        flush_frames(ev, 2)
        time.sleep(1.4)          # 等 1.15s 的点亮衰减完：拍稳态，不拍「闪到一半」
        shots["done"] = shot(call, "prog-done.png",
                             "#p-merge .mgview[data-view='progress'] .mgprog")
        say("截图：%s" % shots["done"])
        dstate = json.loads(ev_json(call, "JSON.stringify(window.__mgpPipe())"))
        check(all(v == "done" for v in (dstate.get("state") or {}).values()),
              "C21 完成态六列都是 done", str(dstate.get("state")))

        # ---------------------------------------------------------------
        head("D  减动效")
        call("Emulation.setEmulatedMedia",
             {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
        ev("window.mgpPipeFreeze(0); true")
        ev("window.mgpReset(); true")
        ev("window.mgpPipe(%s); true" % json.dumps(live_status, ensure_ascii=False))
        flush_frames(ev, 1)
        d1_lit, _d1_p, _d1_fed = feed_and_probe(ev, call, live_status, live,
                                                want_lit=True, want_pulse=False)
        rm = json.loads(ev_json(call, r"""(function(){
          var svg = document.getElementById('mgPipeSvg');
          var r = svg.querySelector('rect.lane.lit');
          var cs = r ? getComputedStyle(r) : null;
          return JSON.stringify({lit: !!r, anim: cs ? cs.animationName : ''});
        })()"""))
        check(str(rm.get("anim") or "") == "none" and rm.get("lit") is True,
              "D1 减动效时点亮照挂（.lit）但动画是 none",
              "%s · 本轮亮过 %s 条" % (rm, d1_lit))
        call("Emulation.setEmulatedMedia", {"features": []})

    except SystemExit:
        raise
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, "脚本异常", "%s: %s" % (e.__class__.__name__, str(e)[:300]))
    finally:
        try:
            if edge is not None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(edge.pid)],
                               capture_output=True)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            if ws is not None:
                ws.__exit__(None, None, None)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            proc.kill()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            pass
    return finish(shots)


def finish(shots=None):
    shots = shots or {}
    if not RESULTS:
        return 1
    head("结果")
    bad = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + detail if detail else ""))
    print("")
    print("%d / %d 通过" % (len(RESULTS) - len(bad), len(RESULTS)))
    for k in ("live", "stall", "done"):
        v = shots.get(k)
        if v and Path(v).is_file():
            print("截图[%s]：%s" % (k, v))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
