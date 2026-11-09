# -*- coding: utf-8 -*-
r"""合成器 · 单脸六宫格调试 验收（无头 Edge + 真后端 + %TEMP% 工作区）

用法
    python app/tools/merge_face_debug_accept.py

    解释器：要带 fastapi / uvicorn / websockets / cv2 的那个。

判读
    PASS / FAIL / SKIP，退出码 0 = 没有 FAIL · 1 = 有 FAIL · 2 = 脚本自己炸了

断言（这次改动的验收条目）
    1) 右上角不再有「合成 / 调试」切换（#mgViewSwitch 不存在）
    2) 每个框旁三个按钮（chip / 调试 / ＋）都是圆角矩形：border-radius <= 8px、高度一致
    3) 框内无覆盖：框中心 20x20 与底图（同一张合成图重绘到 canvas）逐像素差 0
    4) 「盖层」（脸层 + 描边 canvas）整层去掉/加回，框区域截图逐像素相同
    5) 点「调试」-> 舞台出现六宫格：6 格都有图、尺寸一致
    6) 六格顺序与命名 = 用户指定
    7) 点「返回」-> 舞台截图与进入前逐像素一致

隔离：工作区 = %TEMP%\mgdbg_accept\ws（仓库 workspace 一个字节都不动）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

TMP = Path(tempfile.gettempdir())
BASE = TMP / "mgdbg_accept"
WS = BASE / "ws"
VIDEO = BASE / "inbox" / "synth.mp4"
SHOT_FACES = TMP / "mgdbg_faces.png"
SHOT_GRID = TMP / "mgdbg_grid.png"
SHOT_AFTER = TMP / "mgdbg_after.png"
SHOT_STAGE = TMP / "mgdbg_stage.png"
SHOT_PAGE = TMP / "mgdbg_page.png"          # 整页（没有右上角切换 + 三个圆角按钮）
SHOT_GRID_PAGE = TMP / "mgdbg_grid_page.png"  # 整页 + 六宫格
LOG_PATH = BASE / "accept.log"

T_SERVER = 180
T_PAGE = 90
T_PREPARE = 900
T_FACES = 240
T_COMPOSE = 300
T_DEBUG = 300

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "app"
SRC_LIB = REPO / "workspace" / "merge" / "_faces"
SRC_MODEL = REPO / "workspace" / "model"

EDGE_CANDIDATES = (
    os.environ.get("EDGE_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    shutil.which("msedge") or "",
    shutil.which("chrome") or "",
)

try:
    from websockets.sync.client import connect as _ws_connect
except Exception:                                            # noqa: BLE001
    _ws_connect = None

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ITEMS = []

# ------------------------------------------------------------------ 基础工具
def log(line):
    line = str(line)
    print(line)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def check(ok, what, detail="", ctx=None):
    log("%-4s %s%s" % ("PASS" if ok else "FAIL", what, (" · " + detail) if detail else ""))
    if not ok and ctx is not None:
        try:
            log("     实际：" + json.dumps(ctx, ensure_ascii=False)[:1500])
        except Exception:                                    # noqa: BLE001
            log("     实际：" + str(ctx)[:1500])
    ITEMS.append({"what": what, "ok": bool(ok)})
    return bool(ok)


def skip(what, why):
    log("SKIP %s · %s" % (what, why))
    ITEMS.append({"what": what, "ok": True})


def http_get(port, path="/json/list", timeout=3.0):
    return OPENER.open("http://127.0.0.1:%d%s" % (port, path), timeout=timeout).read().decode()


def api(port, path, body=None, method="GET", timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data,
                                 method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(OPENER.open(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except Exception:                                    # noqa: BLE001
            return {"ok": False, "detail": "HTTP %s" % e.code}


def free_port(start=9500):
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("9500~9700 都占用了")


def edge_path():
    for c in EDGE_CANDIDATES:
        if c and Path(c).is_file():
            return c
    return ""


def server_python():
    for c in (str(REPO / "runtime" / "server" / "Scripts" / "python.exe"), sys.executable):
        if not c or not Path(c).is_file():
            continue
        try:
            r = subprocess.run([c, "-c", "import fastapi, uvicorn, websockets, cv2, numpy"],
                               capture_output=True, timeout=180)
            if r.returncode == 0:
                return c
        except Exception:                                    # noqa: BLE001
            continue
    return ""


# ------------------------------------------------------------------ 临时工作区
def ensure_ws():
    if WS.is_dir():
        shutil.rmtree(WS, ignore_errors=True)
    (WS / "merge").mkdir(parents=True, exist_ok=True)
    (WS / "inbox").mkdir(parents=True, exist_ok=True)
    # 登录/引导门：不写这一份，页面会被扫码浮层盖住
    (WS / "auth.json").write_text(json.dumps(
        {"uid": 100001, "name": "验收账号", "avatar": "", "since": int(time.time()),
         "onboarded": True}, ensure_ascii=False), encoding="utf-8")
    # 模型目录用 junction 指到仓库（只读使用；2GB 权重不复制）
    dst = WS / "model"
    if not dst.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(SRC_MODEL)],
                       capture_output=True, text=True)
    lib = WS / "merge" / "_faces"
    lib.mkdir(parents=True, exist_ok=True)
    for f in SRC_LIB.iterdir():
        if f.is_file():
            shutil.copy2(f, lib / f.name)


def make_video():
    import cv2
    import numpy as np
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    faces = sorted(SRC_LIB.glob("f_*.png"))[:2]
    if len(faces) < 2:
        return False
    ims = []
    for f in faces:
        im = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        ims.append(cv2.resize(im, (300, 300), interpolation=cv2.INTER_CUBIC))
    w, h = 1280, 720
    vw = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for i in range(12):
        fr = np.full((h, w, 3), 46, np.uint8)
        fr[180:480, 180:480] = ims[0]
        fr[180:480, 800:1100] = ims[1]
        fr = cv2.add(fr, np.full((h, w, 3), i % 3, np.uint8))
        vw.write(fr)
    vw.release()
    return VIDEO.is_file() and VIDEO.stat().st_size > 1000


def start_server(py):
    env = dict(os.environ, DFL_WORKSPACE=str(WS), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([py, "serve.py"], cwd=str(APP_DIR), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, env=env,
                            creationflags=0x08000000)
    tail, port = [], None
    t0 = time.time()
    while time.time() - t0 < T_SERVER:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        tail.append(line.rstrip())
        del tail[:-60]
        if line.startswith("PORT="):
            port = int(line.strip().split("=")[1])
            break

    def pump():
        try:
            for ln in proc.stdout:
                tail.append(ln.rstrip())
                del tail[:-60]
        except Exception:                                    # noqa: BLE001
            pass
    threading.Thread(target=pump, daemon=True).start()
    return proc, port, tail


# ------------------------------------------------------------------ CDP
class Browser:
    def __init__(self, edge):
        self.port = free_port()
        self.profile = Path(tempfile.mkdtemp(prefix="edge-mgdbg-", dir=str(TMP)))
        self.proc = subprocess.Popen(
            [edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--disable-extensions", "--disable-sync", "--no-first-run",
             "--no-default-browser-check", "--disable-background-networking",
             "--remote-debugging-port=%d" % self.port,
             "--user-data-dir=%s" % self.profile, "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws = None
        self._id = 0
        self.console = []
        self._open()

    def _open(self):
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                pages = [t for t in json.loads(http_get(self.port)) if t.get("type") == "page"]
                if pages:
                    self.ws = _ws_connect(pages[0]["webSocketDebuggerUrl"],
                                          max_size=128 * 1024 * 1024)
                    self.call("Page.enable")
                    self.call("Runtime.enable")
                    self.call("Log.enable")
                    self.call("Emulation.setDeviceMetricsOverride", width=1680, height=1000,
                              deviceScaleFactor=1, mobile=False)
                    return
            except Exception:                                # noqa: BLE001
                time.sleep(0.4)
        raise RuntimeError("Edge CDP 连不上")

    def call(self, method, **p):
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": p}))
        deadline = time.time() + 600
        while time.time() < deadline:
            m = json.loads(self.ws.recv())
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError("%s: %s" % (method, m["error"]))
                return m.get("result", {})
            if m.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded"):
                try:
                    self.console.append(json.dumps(m.get("params"), ensure_ascii=False)[:260])
                    del self.console[:-20]
                except Exception:                            # noqa: BLE001
                    pass
        raise RuntimeError("CDP %s 超时" % method)

    def eval(self, js, timeout=300):
        r = self.call("Runtime.evaluate", expression=js, returnByValue=True,
                      awaitPromise=True, timeout=int(timeout * 1000))
        if r.get("exceptionDetails"):
            raise RuntimeError("页面里抛异常：%s"
                               % json.dumps(r["exceptionDetails"], ensure_ascii=False)[:400])
        return r.get("result", {}).get("value")

    def js(self, js):
        v = self.eval(js)
        return json.loads(v) if isinstance(v, str) else v

    def goto(self, url):
        self.call("Page.navigate", url=url)
        t0 = time.time()
        while time.time() - t0 < T_PAGE:
            try:
                if self.eval("document.readyState") == "complete":
                    return True
            except Exception:                                # noqa: BLE001
                pass
            time.sleep(0.25)
        return False

    def pixel(self, clip=None):
        """截一张，解成 numpy（BGR）。clip = 页面坐标 {x,y,width,height}。"""
        import cv2
        import numpy as np
        p = {"format": "png"}
        if clip:
            p["clip"] = dict(clip, scale=1)
        data = self.call("Page.captureScreenshot", **p).get("data") or ""
        raw = base64.b64decode(data)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img, raw

    def shot(self, path, clip=None):
        img, raw = self.pixel(clip)
        Path(path).write_bytes(raw)
        return raw

    def close(self):
        try:
            self.ws.close()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            pass
        shutil.rmtree(self.profile, ignore_errors=True)


# ------------------------------------------------------------------ 页面探针
SNAP_JS = r"""(function(){
  var st = document.getElementById("mgStage");
  var cv = st ? st.querySelector("canvas.mgovl") : null;
  var img = document.getElementById("mgPreview");
  var grid = document.getElementById("mgFaceGrid");
  function rect(el){ if(!el) return null; var r = el.getBoundingClientRect();
    return {x:r.x, y:r.y, w:r.width, h:r.height, cx:r.x+r.width/2, cy:r.y+r.height/2}; }
  function btn(el){
    if(!el) return null;
    var cs = getComputedStyle(el);
    var r = el.getBoundingClientRect();
    return {txt:String(el.textContent||"").trim(), br:parseFloat(cs.borderTopLeftRadius)||0,
            brRaw:cs.borderTopLeftRadius, h:r.height, w:r.width, x:r.x, y:r.y,
            vis: el.offsetParent !== null, bg:cs.backgroundColor};
  }
  var cards = Array.prototype.map.call(document.querySelectorAll(".mgfc"), function(c){
    return {i: c.getAttribute("data-face"),
            chip: btn(c.querySelector(".mgfb")),
            dbg: btn(c.querySelector(".mgfdbg")),
            plus: btn(c.querySelector(".mgplus")),
            layerBg: getComputedStyle(c).backgroundColor};
  });
  return JSON.stringify({
    viewSwitch: !!document.getElementById("mgViewSwitch"),
    viewSwAny: document.querySelectorAll("#mgViewSwitch, .mgviewsw").length,
    grid: grid ? {hidden: !!grid.hidden, cells: grid.querySelectorAll(".mgfg-cell").length,
                  rect: rect(grid)} : null,
    stage: rect(st), canvas: rect(cv), preview: rect(img),
    canvasZ: cv ? getComputedStyle(cv).zIndex : "",
    imgZ: img ? getComputedStyle(img).zIndex : "",
    imgSrc: img ? String(img.getAttribute("src")||"").length : -1,
    imgShown: !!(img && getComputedStyle(img).display !== "none"),
    frames: (window.__mv||{}).frames || 0,
    video: String((window.__mv||{}).video||""),
    prep: !!(window.__mergePrep && window.__mergePrep.running),
    cards: cards,
    faceRects: (window.mgFaceRects ? window.mgFaceRects() : []),
    models: Array.prototype.map.call(document.querySelectorAll("#mgModels .mgitem"), function(x){
      return {n: String((x.querySelector(".nm")||{}).textContent||"").trim(),
              on: x.classList.contains("on")};})
  });
})()"""

DBG_JS = r"""(function(){
  var s = (window.mgFaceDebugState ? window.mgFaceDebugState() : null);
  return JSON.stringify(s || {missing:1});
})()"""

# 框中心 20x20：底图（同一张合成图按 object-fit 重绘）的像素 + 页面坐标
REF_JS = r"""(function(){
  var img = document.getElementById("mgPreview");
  if (!img || !img.naturalWidth) return JSON.stringify({err:"no img"});
  var ir = img.getBoundingClientRect();
  var nw = img.naturalWidth, nh = img.naturalHeight;
  var s = Math.min(ir.width / nw, ir.height / nh);
  var dw = nw * s, dh = nh * s;
  var dx = ir.x + (ir.width - dw) / 2, dy = ir.y + (ir.height - dh) / 2;
  var c = document.createElement("canvas");
  c.width = Math.round(dw); c.height = Math.round(dh);
  var ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, c.width, c.height);
  var out = [];
  (window.mgFaceRects ? window.mgFaceRects() : []).forEach(function(r){
    var x0 = Math.round(r.cx - 10), y0 = Math.round(r.cy - 10);
    var kx = Math.round((x0 - dx) * (c.width / dw));
    var ky = Math.round((y0 - dy) * (c.height / dh));
    var d = ctx.getImageData(kx, ky, 20, 20).data;
    var arr = []; for (var i = 0; i < d.length; i++) arr.push(d[i]);
    out.push({i: r.i, page: [x0, y0], px: arr});
  });
  return JSON.stringify({blocks: out});
})()"""


def wait(b, js, pred, timeout, interval=0.4):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = b.js(js)
        except Exception as e:                               # noqa: BLE001
            last = {"err": str(e)[:200]}
        if pred(last):
            return True, last, round(time.time() - t0, 1)
        time.sleep(interval)
    return False, last, round(time.time() - t0, 1)


SPEC_LABELS = ["mask(dfm-pred)", "mask(dst)", "merged(mask(dfm-pred), mask(dst))",
               "dfm-pred", "dst-original", "merged"]


def run(b, port, opt):
    # ---------- 进合成页 + 导入素材 ----------
    if not b.goto("http://127.0.0.1:%d/" % port):
        check(False, "打开界面", "readyState 一直不到 complete")
        return
    b.eval('localStorage.setItem("dfl-legal-v3","1"), 1')      # 许可门：验收不测它
    b.goto("http://127.0.0.1:%d/" % port)
    b.eval('window.go && window.go("merge"), 1')
    ok, s, dt = wait(b, SNAP_JS, lambda x: x.get("prep") is not None, 60)
    check(ok, "进合成页", "%.1fs" % dt, ctx=s)
    if not ok:
        return

    b.eval("window.mgPrepare(%s), 1" % json.dumps(str(VIDEO)))
    ok, s, dt = wait(b, SNAP_JS,
                     lambda x: (not x.get("prep")) and (x.get("frames") or 0) > 0,
                     T_PREPARE, 0.6)
    check(ok, "导入素材", "%s 帧 · %.1fs" % ((s or {}).get("frames"), dt), ctx=s)
    if not ok:
        return
    ok, s, dt = wait(b, SNAP_JS, lambda x: len(x.get("cards") or []) > 0, T_FACES, 0.5)
    check(ok, "检测出框", "%d 个框 · %.1fs" % (len((s or {}).get("cards") or []), dt), ctx=s)
    if not ok:
        return

    # ---------- 给每张脸指定一个真模型（小、CPU 也快），等合成图落地 ----------
    names = [m["n"] for m in (s.get("models") or []) if m.get("n")]
    pick = next((n for n in names if "416" in n), (names[0] if names else ""))
    if not pick:
        skip("选模型", "工作区里没有模型（%s）" % (WS / "model"))
        return
    for i in range(len(s.get("cards") or [])):
        b.eval("window.mgAssignFace(%d, %s), 1" % (i, json.dumps(pick)))
    ok, s, dt = wait(b, SNAP_JS,
                     lambda x: x.get("imgSrc", 0) > 10000 and x.get("imgShown"),
                     T_COMPOSE, 0.5)
    check(ok, "合成出图", "模型 %s · 图 %.0fKB · %.1fs"
          % (pick, (s or {}).get("imgSrc", 0) / 1024.0, dt), ctx=s)
    if not ok:
        return
    time.sleep(0.8)          # 让 img.onload 的那一次补画跑完，后面比像素才稳

    # ---------- 1) 右上角不再有「合成 / 调试」 ----------
    check(not s.get("viewSwitch") and not s.get("viewSwAny"),
          "右上角没有「合成/调试」切换",
          "#mgViewSwitch / .mgviewsw 都不在 DOM 里", ctx={"viewSwitch": s.get("viewSwitch")})

    # ---------- 2) 三个按钮：圆角矩形 + 等高 ----------
    cards = s.get("cards") or []
    bad = []
    for c in cards:
        for k in ("chip", "dbg", "plus"):
            bt = c.get(k)
            if not bt:
                bad.append({"face": c["i"], "btn": k, "why": "缺按钮"})
                continue
            if not bt.get("vis"):
                bad.append({"face": c["i"], "btn": k, "why": "不可见"})
            if bt.get("br", 99) > 8.0:
                bad.append({"face": c["i"], "btn": k, "br": bt.get("brRaw")})
    heights = sorted({round(c[k]["h"], 1) for c in cards for k in ("chip", "dbg", "plus")
                      if c.get(k)})
    same_h = len(heights) <= 1
    check(bool(cards) and not bad,
          "三个按钮都是圆角矩形（border-radius <= 8px）",
          "faces=%d · chip/调试/＋ 圆角 %s"
          % (len(cards), [c["chip"]["brRaw"] if c.get("chip") else "?" for c in cards]), ctx=bad)
    check(bool(cards) and same_h, "三个按钮高度一致",
          "height = %s px（chip=%s 调试=%s ＋=%s）"
          % (heights[0] if heights else "?", (cards[0]["chip"] or {}).get("h") if cards else "?",
             (cards[0]["dbg"] or {}).get("h") if cards else "?",
             (cards[0]["plus"] or {}).get("h") if cards else "?"),
          ctx={"heights": heights})

    # ---------- 3) 框内无覆盖：框中心 20x20 与底图逐像素差 0 ----------
    ref = b.js(REF_JS)
    full, _ = b.pixel()
    worst, per = 0, []
    for blk in ref.get("blocks") or []:
        x0, y0 = blk["page"]
        want = blk["px"]
        got = full[y0:y0 + 20, x0:x0 + 20]
        if got.shape[0] != 20 or got.shape[1] != 20:
            per.append({"face": blk["i"], "why": "越界"})
            continue
        dmax = 0
        for yy in range(20):
            for xx in range(20):
                b_, g_, r_ = int(got[yy, xx, 0]), int(got[yy, xx, 1]), int(got[yy, xx, 2])
                for ch, v in ((0, want[(yy * 20 + xx) * 4 + 2]),
                              (1, want[(yy * 20 + xx) * 4 + 1]),
                              (2, want[(yy * 20 + xx) * 4 + 0])):
                    dmax = max(dmax, abs(v - (b_, g_, r_)[ch]))
        per.append({"face": blk["i"], "max": dmax})
        worst = max(worst, dmax)
    check(bool(per) and worst <= 1, "框内无覆盖（框中心 20x20 与底图差 0）",
          "最大差值 %d px（faces=%s）" % (worst, [p.get("max") for p in per]), ctx=per)

    # ---------- 4) 盖层整层去掉/加回：**框内**（描边以内）像素一个都不变 ----------
    # 描边本身当然会画（2px 亮线 + 外面一圈 2px 深色对比线），chip / 调试 / ＋ 也都在框外；
    # 所以这里只比「框内缩进 3px」那块 —— 那正是用户说的「脸被白蒙版盖住」的地方。
    st = s.get("stage")
    clip = {"x": st["x"], "y": st["y"], "width": st["w"], "height": st["h"]}
    b.js("""(function(){
        document.getElementById("mgFaceLayer").style.display = "none";
        document.querySelector("#mgStage canvas.mgovl").style.display = "none"; return 1;})()""")
    time.sleep(0.35)
    with_on, _ = b.pixel(clip)
    b.js("""(function(){
        document.getElementById("mgFaceLayer").style.display = "";
        document.querySelector("#mgStage canvas.mgovl").style.display = ""; return 1;})()""")
    time.sleep(0.35)
    without, _ = b.pixel(clip)
    inset = 3
    worst2, per2 = 0, []
    if with_on.shape == without.shape:
        for r in (s.get("faceRects") or []):
            x0 = int(round(r["x"] - clip["x"])) + inset
            y0 = int(round(r["y"] - clip["y"])) + inset
            w0 = int(round(r["w"])) - inset * 2
            h0 = int(round(r["h"])) - inset * 2
            if w0 <= 0 or h0 <= 0:
                per2.append({"face": r["i"], "why": "框太小"})
                continue
            a = with_on[y0:y0 + h0, x0:x0 + w0].astype("int32")
            c2 = without[y0:y0 + h0, x0:x0 + w0].astype("int32")
            d = int(abs(a - c2).max()) if (a.size and a.shape == c2.shape) else -1
            per2.append({"face": r["i"], "max": d, "box": [w0, h0]})
            worst2 = max(worst2, d)
    check(with_on.shape == without.shape and bool(per2) and worst2 == 0,
          "盖层（脸层 + 描边 canvas）在框内不画任何像素",
          "框内缩进 %dpx · 最大差值 %d（faces=%s）"
          % (inset, worst2, [p.get("max") for p in per2]),
          ctx={"per": per2})

    # ---------- 截图：框特写 ----------
    r0 = (s.get("faceRects") or [{}])[0]
    if r0:
        b.shot(str(SHOT_FACES), {"x": max(0, r0["x"] - 34), "y": max(0, r0["y"] - 46),
                                 "width": r0["w"] + 78, "height": r0["h"] + 96})
    b.shot(str(SHOT_STAGE), clip)
    b.shot(str(SHOT_PAGE))

    # ---------- 5) 点「调试」→ 六宫格 ----------
    before_stage, _ = b.pixel(clip)
    before_snap = b.js(SNAP_JS)
    clicked = b.eval("""(function(){
        var b = document.querySelector('.mgfc[data-face="0"] .mgfdbg');
        if (!b) return 0; b.click(); return 1;})()""")
    ok, g, dt = wait(b, DBG_JS, lambda x: x.get("open") and
                     all(c.get("has") for c in (x.get("cells") or [])) and
                     len(x.get("cells") or []) == 6, T_DEBUG, 0.4)
    check(clicked and ok, "点「调试」→ 六宫格出图",
          "%d 格 · %.1fs" % (len((g or {}).get("cells") or []), dt), ctx=g)
    if clicked and ok:
        cells = g["cells"]
        check(len(cells) == 6, "六宫格是 3 列 × 2 行 = 6 格", "%d 格" % len(cells), ctx=cells)
        names = [c.get("nm") for c in cells]
        check(names == SPEC_LABELS, "六格顺序与命名符合指定", " / ".join(names), ctx=names)
        nums = [c.get("no") for c in cells]
        check(nums == ["1", "2", "3", "4", "5", "6"], "每格左上角有数字", " ".join(nums), ctx=nums)
        sizes = sorted({(c.get("nw"), c.get("nh")) for c in cells})
        check(len(sizes) == 1 and sizes[0][0] > 0, "六格的图尺寸一致",
              "natural = %s" % (sizes,), ctx=cells)
        bw = sorted({(c.get("w"), c.get("h")) for c in cells})
        check(len(bw) == 1, "六格的格子尺寸一致（3 列 × 2 行等分）", "box = %s" % (bw,))
        gw = g["rect"][2]
        check(abs(gw - s["stage"]["w"]) <= 2, "六宫格整块贴合舞台宽度",
              "grid=%.1f stage=%.1f" % (gw, s["stage"]["w"]))
        b.shot(str(SHOT_GRID), {"x": g["rect"][0], "y": g["rect"][1],
                                "width": g["rect"][2], "height": g["rect"][3]})
        b.shot(str(SHOT_GRID_PAGE))
        # 点一格放大（可选功能，做了就得能用）：再点一次缩回去
        z1 = b.js("""(function(){
            var c = document.querySelector('.mgfg-cell[data-k="dfm-pred"]');
            if (!c) return JSON.stringify({err:"no cell"});
            c.click();
            var grid = document.getElementById("mgFaceGrid");
            var st = window.mgFaceDebugState();
            return JSON.stringify({zoom: st.zoom, cls: grid.className,
                                   shown: Array.prototype.filter.call(
                                     grid.querySelectorAll(".mgfg-cell"),
                                     function(x){ return x.offsetParent !== null; }).length});
        })()""")
        z2 = b.js("""(function(){
            var c = document.querySelector('.mgfg-cell[data-k="dfm-pred"]');
            c.click(); var st = window.mgFaceDebugState();
            var grid = document.getElementById("mgFaceGrid");
            return JSON.stringify({zoom: st.zoom,
                                   shown: Array.prototype.filter.call(
                                     grid.querySelectorAll(".mgfg-cell"),
                                     function(x){ return x.offsetParent !== null; }).length});
        })()""")
        check(z1.get("zoom") == "dfm-pred" and z1.get("shown") == 1
              and not z2.get("zoom") and z2.get("shown") == 6,
              "点一格能放大、再点缩回", "放大 %s → 缩回 %s" % (z1.get("zoom"), z2.get("zoom") or "无"),
              ctx={"zoom_in": z1, "zoom_out": z2})

    # ---------- 6) 点「返回」→ 画面逐像素回到进入前 ----------
    back = b.eval("""(function(){
        var b = document.getElementById("mgFaceGridBack");
        if (!b) return 0; b.click(); return 1;})()""")
    ok2, g2, dt2 = wait(b, DBG_JS, lambda x: (x.get("present") is not None) and
                        (not x.get("open")), 30, 0.2)
    after_snap = b.js(SNAP_JS)
    after_stage, _ = b.pixel(clip)
    same = (before_stage.shape == after_stage.shape and
            int(abs(before_stage.astype("int32") - after_stage.astype("int32")).max()) == 0)
    check(back and ok2 and same, "点「返回」→ 回合成视图，画面与进入前逐像素一致",
          "格 %s · 图 %s → %s" % ("已收" if ok2 else "还在",
                                before_snap.get("imgSrc"), after_snap.get("imgSrc")),
          ctx={"before": list(before_stage.shape), "after": list(after_stage.shape),
               "grid": g2})
    b.shot(str(SHOT_AFTER), clip)

    # ---------- 7) 顺手验一条被我改过的老交互：点框 = 这张脸切「原图 / 合成」 ----------
    if r0:
        b.eval("""(function(){
            var r = (window.mgFaceRects()||[])[0];
            window.__mgAcceptClick = {x: r.cx, y: r.cy}; return 1;})()""")
        pt = b.js("JSON.stringify(window.__mgAcceptClick)")
        for kind in ("mousePressed", "mouseReleased"):
            b.call("Input.dispatchMouseEvent", type=kind, x=float(pt["x"]), y=float(pt["y"]),
                   button="left", buttons=1, clickCount=1)
        ok3, s3, dt3 = wait(b, SNAP_JS,
                            lambda x: ((x.get("faceRects") or [{}])[0] or {}).get("raw") is True,
                            30, 0.3)
        for kind in ("mousePressed", "mouseReleased"):
            b.call("Input.dispatchMouseEvent", type=kind, x=float(pt["x"]), y=float(pt["y"]),
                   button="left", buttons=1, clickCount=1)
        ok4, s4, dt4 = wait(b, SNAP_JS,
                            lambda x: ((x.get("faceRects") or [{}])[0] or {}).get("raw") is False,
                            60, 0.3)
        check(ok3 and ok4, "点框仍然切这张脸的「原图 / 合成」",
              "原图 %s → 合成 %s" % (ok3, ok4),
              ctx={"after_first": (s3.get("faceRects") or [{}])[0],
                   "after_second": (s4.get("faceRects") or [{}])[0]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    opt = ap.parse_args(argv)
    if LOG_PATH.parent.is_dir():
        try:
            LOG_PATH.unlink()
        except OSError:
            pass
    BASE.mkdir(parents=True, exist_ok=True)

    edge = edge_path()
    py = server_python()
    if not edge:
        print("SKIP 找不到 Edge（装 Edge 或设 EDGE_PATH）")
        return 0
    if not py:
        print("SKIP 没有能 import fastapi/uvicorn/websockets/cv2 的 python")
        return 0
    if _ws_connect is None:
        print("SKIP 当前解释器没有 websockets（用 runtime\\server\\Scripts\\python.exe 跑）")
        return 0
    if not SRC_LIB.is_dir() or not SRC_MODEL.is_dir():
        print("SKIP 仓库里没有 workspace/merge/_faces 或 workspace/model")
        return 0

    proc = browser = None
    tail = []
    try:
        ensure_ws()
        if not make_video():
            print("SKIP 造不出合成素材视频")
            return 0
        proc, port, tail = start_server(py)
        if not port:
            check(False, "后端启动", "serve.py 没打出 PORT=", ctx={"log": tail[-12:]})
            return 1
        log("[后端] %s · 工作区 %s · port=%s" % (py, WS, port))
        browser = Browser(edge)
        run(browser, port, opt)
        if browser.console:
            log("[页面 console] " + " | ".join(browser.console[-3:]))
    finally:
        if browser:
            browser.close()
        if proc:
            try:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, timeout=30)
            except Exception:                                # noqa: BLE001
                pass
        if not opt.keep:
            shutil.rmtree(WS, ignore_errors=True)

    fails = [i for i in ITEMS if not i["ok"]]
    log("-" * 72)
    log("共 %d 项 · PASS %d · FAIL %d" % (len(ITEMS), len(ITEMS) - len(fails), len(fails)))
    for f in fails:
        log("  FAIL " + f["what"])
    log("截图：" + " · ".join(str(p) for p in (SHOT_FACES, SHOT_GRID, SHOT_AFTER,
                                              SHOT_PAGE, SHOT_GRID_PAGE) if p.is_file()))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
