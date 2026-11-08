# -*- coding: utf-8 -*-
r"""人脸数据库（fdb）页验收：DOM 断言 + 组间拖放 + 回收站拖放 + 同帧去重。

用法
    python app/tools/fdb_accept.py --phase before   # 只拍「改前」截图
    python app/tools/fdb_accept.py --phase after    # 全量断言 + 拍「改后」截图

    解释器：要带 fastapi / uvicorn / websockets / cv2 的那个（脚本自己挑）。
    退出码：0 = 没有 FAIL · 1 = 有 FAIL · 2 = 脚本自己炸了。

隔离
    全程用 %TEMP%\fdb_accept\ws 当 DFL_WORKSPACE（仓库 workspace 一个字节都不动）：
      · legacy/   ← 复制仓库默认库里那份「9 张同帧重复」的旧库，用来验自动去重
      · inbox/synth.mp4 ← 现场合成的 2 脸视频，用来验「＋」落库与同帧替换
      · merge/<stem>/   ← 后端 prepare 出来的项目（＋ 往这个库里写）
    截图：%TEMP%\fdb_before.png / %TEMP%\fdb_after.png。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

TMP = Path(tempfile.gettempdir())
BASE = TMP / "fdb_accept"
WS = BASE / "ws"
SHOT_BEFORE = TMP / "fdb_before.png"
SHOT_AFTER = TMP / "fdb_after.png"
LOG_PATH = BASE / "fdb_accept.log"

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "app"
SRC_LIB = REPO / "workspace" / "merge" / "_faces"

T_SERVER = 120
T_PAGE = 90
T_UI = 90
T_PREPARE = 900
T_DETECT = 180
T_ADD = 120
T_DB = 90

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

_OPENER = None
ITEMS = []


# ------------------------------------------------------------------ 基础工具
def _http_get(port, path="/json/list", timeout=2.0):
    import urllib.request
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _OPENER.open("http://127.0.0.1:%d%s" % (port, path), timeout=timeout).read().decode()


def _free_port(start=9400):
    for p in range(start, start + 300):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("9400~9699 都占用了")


def _edge_path():
    for c in EDGE_CANDIDATES:
        if c and Path(c).is_file():
            return c
    return ""


def _server_python():
    cands = [str(REPO / "runtime" / "server" / "Scripts" / "python.exe"),
             sys.executable, shutil.which("python") or ""]
    for c in cands:
        if not c or not Path(c).is_file():
            continue
        try:
            r = subprocess.run([c, "-c", "import fastapi, uvicorn, websockets, cv2"],
                               capture_output=True, timeout=180)
            if r.returncode == 0:
                return c
        except Exception:                                    # noqa: BLE001
            continue
    return ""


def log(line):
    line = str(line)
    print(line)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def check(ok, what, detail="", ctx=None):
    status = "PASS" if ok else "FAIL"
    head = "%-4s %s" % (status, what)
    if detail:
        head += " · " + detail
    log(head)
    if not ok and ctx is not None:
        try:
            log("     实际：" + json.dumps(ctx, ensure_ascii=False)[:1200])
        except Exception:                                    # noqa: BLE001
            log("     实际：" + str(ctx)[:1200])
    ITEMS.append({"what": what, "ok": bool(ok)})
    return bool(ok)


def skip(what, why):
    log("SKIP %s · %s" % (what, why))
    ITEMS.append({"what": what, "ok": True})


# ------------------------------------------------------------------ 工作区
def ensure_ws():
    if WS.is_dir():
        shutil.rmtree(WS, ignore_errors=True)
    (WS / "merge").mkdir(parents=True, exist_ok=True)
    (WS / "inbox").mkdir(parents=True, exist_ok=True)
    (WS / "model").mkdir(parents=True, exist_ok=True)
    # 登录/引导门：不写这一份，页面会被扫码登录浮层盖住（截图全是二维码）
    (WS / "auth.json").write_text(json.dumps({"uid": 100001, "name": "验收账号", "avatar": "",
                                              "since": int(time.time()), "onboarded": True},
                                             ensure_ascii=False), encoding="utf-8")
    dst = WS / "legacy" / "_faces"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(SRC_LIB.iterdir()):
        if f.is_file() and f.suffix.lower() in (".png", ".json") and f.name != "index.json.bak":
            shutil.copy2(f, dst / f.name)
            n += 1
    # 组名在仓库里的那份是历史遗留的乱码，验收截图要能看清，只在临时副本里改
    idx_path = dst / "index.json"
    if idx_path.is_file():
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        for g in raw.get("groups") or []:
            g["name"] = "模型 " + str(g.get("id") or "g1")
        idx_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def make_video(path):
    import cv2
    import numpy as np
    faces = sorted(SRC_LIB.glob("f_*.png"))[:2]
    if len(faces) < 2:
        raise RuntimeError("仓库默认库里没有两张脸图，合成不了素材")
    ims = []
    for f in faces:
        im = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        ims.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_CUBIC))
    w, h = 1280, 720
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for i in range(16):
        fr = np.full((h, w, 3), 46, np.uint8)
        fr[220:476, 200:456] = ims[0]
        fr[220:476, 800:1056] = ims[1]
        fr = cv2.add(fr, np.full((h, w, 3), i % 4, np.uint8))
        vw.write(fr)
    vw.release()
    return path.is_file() and path.stat().st_size > 1000


# ------------------------------------------------------------------ CDP
class Browser:
    def __init__(self, edge):
        self.edge = edge
        self.port = _free_port()
        self.profile = Path(tempfile.mkdtemp(prefix="edge-fdb-", dir=str(TMP)))
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
        if _ws_connect is None:
            raise RuntimeError("没有 websockets：用 runtime\\server\\Scripts\\python.exe 跑")
        deadline = time.time() + 60
        last = ""
        while time.time() < deadline:
            try:
                data = json.loads(_http_get(self.port))
                pages = [t for t in data if t.get("type") == "page"]
                if pages:
                    pages.sort(key=lambda t: 0 if (t.get("url") or "").startswith(("about:", "data:", "")) else 1)
                    self.ws = _ws_connect(pages[0]["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024)
                    self.call("Page.enable")
                    self.call("Runtime.enable")
                    self.call("Log.enable")
                    self.call("Emulation.setDeviceMetricsOverride", width=1680, height=1000,
                              deviceScaleFactor=1, mobile=False)
                    return
            except Exception as e:                            # noqa: BLE001
                last = str(e)[:120]
                time.sleep(0.4)
        raise RuntimeError("Edge CDP 连不上（port=%d，最后错误 %s）" % (self.port, last or "无"))

    def call(self, method, **p):
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": p}))
        deadline = time.time() + 300
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
        deadline = time.time() + T_PAGE
        while time.time() < deadline:
            try:
                if self.eval("document.readyState") == "complete":
                    return True
            except Exception:                                # noqa: BLE001
                pass
            time.sleep(0.25)
        return False

    def shot(self, path):
        r = self.call("Page.captureScreenshot", format="png")
        import base64
        data = base64.b64decode(r.get("data") or "")
        Path(path).write_bytes(data)
        return len(data)

    def mouse(self, kind, x, y, buttons=1):
        self.call("Input.dispatchMouseEvent", type=kind, x=float(x), y=float(y),
                  button="left", buttons=buttons, clickCount=1 if kind != "mouseMoved" else 0)

    def close(self):
        try:
            self.ws.close()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            try:
                self.proc.terminate()
            except Exception:                                # noqa: BLE001
                pass
        shutil.rmtree(self.profile, ignore_errors=True)


# ------------------------------------------------------------------ 服务
def start_server(py):
    env = dict(os.environ, DFL_WORKSPACE=str(WS), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([py, "serve.py"], cwd=str(APP_DIR),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=env, creationflags=0x08000000)
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
        del tail[:-40]
        if line.startswith("PORT="):
            port = int(line.strip().split("=")[1])
            break

    def _pump():
        try:
            for ln in proc.stdout:
                tail.append(ln.rstrip())
                del tail[:-40]
        except Exception:                                    # noqa: BLE001
            pass

    threading.Thread(target=_pump, daemon=True).start()
    return proc, port, tail


# ------------------------------------------------------------------ 页面探针
def js_fetch(url, method="GET", body="null"):
    return ("(async function(){try{var r=await fetch(%s,{method:%s,body:%s,"
            "headers:{\"Content-Type\":\"application/json\"}});var t=await r.text();"
            "var d=t;try{d=JSON.parse(t);}catch(e){d={_text:String(t).slice(0,300)};}"
            "return JSON.stringify({status:r.status,body:d});}"
            "catch(e){return JSON.stringify({status:0,body:{error:String(e)}});}})()"
            % (json.dumps(url), json.dumps(method), body))


AUDIT_JS = r"""
(function(){
  var root = document.getElementById("fdbRoot");
  if (!root) return JSON.stringify({missing: "fdbRoot"});
  var vis = function(n){ return !!(n && n.offsetParent !== null); };
  var txt = String(root.innerText || "");
  var btns = Array.prototype.map.call(root.querySelectorAll("button"), function(b){
    return {id: b.id || "", text: String(b.textContent || "").trim(), vis: vis(b)};
  });
  var g = document.getElementById("fdbGroups");
  var gc = g ? getComputedStyle(g) : null;
  var cards = g ? Array.prototype.map.call(g.querySelectorAll(".fdb-group"), function(c){
    var hd = c.querySelector(".fdb-group-hd");
    return {gid: c.getAttribute("data-group") || "",
            hd: hd ? String(hd.innerText || "").replace(/\s+/g, " ").trim() : "",
            n: c.querySelectorAll(".fdb-face").length,
            size: (c.querySelector(".fdb-size") || {}).textContent || ""};
  }) : [];
  var faces = Array.prototype.map.call(root.querySelectorAll(".fdb-face"), function(f){
    return {id: f.getAttribute("data-face-id") || "",
            gid: f.getAttribute("data-group") || "",
            meta: String((f.querySelector(".fdb-meta") || {}).textContent || "").replace(/\s+/g, " ").trim()};
  });
  var trash = document.getElementById("fdbTrash");
  /* 「提取到 <dfm>」下拉已经删掉：这里只报它**还在不在**（false = 正确） */
  var sel = document.getElementById("fdbDfm");
  return JSON.stringify({
    text: txt.replace(/\n+/g, " | "),
    buttons: btns,
    ids: {
      ungrouped: !!document.getElementById("fdbUngrouped"),
      ungroupedStat: !!document.getElementById("fdbStatUngrouped"),
      extract: !!document.getElementById("fdbExtract"),
      moveBtn: !!document.getElementById("fdbMoveBtn"),
      moveSel: !!document.getElementById("fdbMoveTarget"),
      trashBtn: !!document.getElementById("fdbTrashBtn"),
      hint: !!document.getElementById("fdbHint"),
      keys: !!document.getElementById("fdbKeys"),
      continueBtn: !!document.getElementById("fdbContinue"),
      continueVis: vis(document.getElementById("fdbContinue")),
      close: !!document.getElementById("fdbClose")
    },
    groups: {flexDir: gc ? gc.flexDirection : "", wrap: gc ? gc.flexWrap : "",
             overflowX: gc ? gc.overflowX : "",
             clientW: g ? g.clientWidth : -1, scrollW: g ? g.scrollWidth : -1,
             cards: cards},
    faces: faces,
    trash: {n: trash ? trash.querySelectorAll(".fdb-trash-item").length : -1,
            drop: trash ? trash.classList.contains("is-drop") : false,
            cls: trash ? trash.className : ""},
    dfm: {present: !!sel, opts: sel ? sel.options.length : 0},
    state: (window.FaceDB && window.FaceDB.debug) ? window.FaceDB.debug() : null,
    toasts: (window.__T || []).slice(-14)
  });
})()
"""


def grab_toasts(b, timeout=8.0, interval=0.25):
    """轮询 #toastroot，把这段时间里冒出来的 toast 文本都收下来（去重保序）。"""
    seen, out, t0 = set(), [], time.time()
    while time.time() - t0 < timeout:
        try:
            items = b.js(r"""(function(){return JSON.stringify(Array.prototype.map.call(
                document.querySelectorAll("#toastroot .toast"), function(t){
                    return String(t.innerText || "").replace(/\s+/g, " ").trim();}));})()""")
        except Exception:                                    # noqa: BLE001
            items = []
        fresh = False
        for s in items or []:
            if s and s not in seen:
                seen.add(s); out.append(s); fresh = True
        if fresh:
            t0 = time.time() - max(0.0, timeout - 2.0)       # 有新提示就把窗口续上
        time.sleep(interval)
    return out


def audit(b):
    return b.js(AUDIT_JS)


def wait_for(b, pred, timeout, js, interval=0.35):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = b.js(js)
        except Exception as e:                               # noqa: BLE001
            last = {"error": str(e)[:200]}
        if pred(last):
            return True, last, round(time.time() - t0, 2)
        time.sleep(interval)
    return False, last, round(time.time() - t0, 2)


HOOK_TOAST = r"""
(function(){
  if (window.__THooked) return 1;
  window.__THooked = 1; window.__T = []; window.__TNET = [];
  var _f = window.fetch;
  window.fetch = function(u, o){
    var url = String((u && u.url) ? u.url : u);
    var p = _f.apply(this, arguments);
    if (url.indexOf("/api/merge/faces") === 0) {
      try {
        var method = String((o && o.method) || "GET");
        p.then(function(r){ r.clone().text().then(function(txt){
          var d = {};
          try { d = JSON.parse(txt); } catch (e) {}
          window.__TNET.push({m: method, url: url.slice(0, 74), status: r.status,
                              deduped: (d && d.deduped), n: ((d && d.faces) || []).length,
                              stack: String(new Error().stack || "").split("\n").slice(2, 6).join(" <- ")});
        }, function(){}); }, function(){});
      } catch (e) {}
    }
    return p;
  };
  var raw = window.toast;
  window.toast = function(t, s, k, ms){
    try { window.__T.push(String(t) + (s ? " · " + String(s) : "")); } catch(e) {}
    return raw.apply(this, arguments);
  };
  return 1;
})()
"""

PREP_STATUS = r"""(async function(){
  var r = await fetch("/api/merge/prepare/status"); return await r.text();
})()"""


def legacy_expect(legacy):
    """旧库去重后应该剩几张 / 应该有几张进回收站（按后端口径算一遍）。"""
    raw = json.loads((legacy / "_faces" / "index.json").read_text(encoding="utf-8"))
    buckets = {}
    for f in raw.get("faces") or []:
        if f.get("trashed"):
            continue
        src = f.get("source") or {}
        buckets.setdefault((str(src.get("video") or ""), int(src.get("frame") or 0),
                            int(f.get("idx") or 0)), []).append(f)
    active = len(buckets)
    total = sum(1 for f in (raw.get("faces") or []) if not f.get("trashed"))
    return {"active": active, "trash": total - active, "buckets": len(buckets)}


def lib_get(b, project):
    raw = b.js(js_fetch("/api/merge/faces?project=" + __import__("urllib.parse", fromlist=["quote"]).quote(str(project), safe="")))
    return (raw or {}).get("body") or {}


# ------------------------------------------------------------------ 主体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("before", "after"), default="after")
    ap.add_argument("--keep", action="store_true", help="不重建临时工作区（调试用）")
    opt = ap.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    if not opt.keep:
        try:
            LOG_PATH.unlink()
        except OSError:
            pass
    edge = _edge_path()
    py = _server_python()
    if not edge or not py or _ws_connect is None:
        log("SKIP 整轮 · 缺 Edge / 带依赖的 python / websockets")
        return 0

    if opt.keep and (WS / "legacy" / "_faces" / "index.json").is_file():
        video = WS / "inbox" / "synth.mp4"
    else:
        n = ensure_ws()
        video = WS / "inbox" / "synth.mp4"
        make_video(video)
        log("[工作区] %s · 复制 %d 个文件 · 素材 %s" % (WS, n, video.name))

    proc, port, tail = start_server(py)
    if not port:
        log("FAIL 后端启动 · serve.py 没打出 PORT=")
        log("\n".join(tail[-12:]))
        return 1
    log("[后端] port=%d · 工作区 %s" % (port, WS))

    browser = None
    code = 0
    try:
        browser = Browser(edge)
        if not browser.goto("http://127.0.0.1:%d/" % port):
            log("FAIL 打开界面 · readyState 不到 complete")
            return 1
        # 使用许可弹层：一次性同意写 localStorage，刷新一次才不会被它挡住
        browser.eval('try{localStorage.setItem("dfl-legal-v3","1")}catch(e){}; 1')
        browser.goto("http://127.0.0.1:%d/" % port)
        browser.eval(HOOK_TOAST)
        browser.eval('window.go && window.go("merge"), 1')
        ok, snap, dt = wait_for(browser, lambda x: bool(x and x.get("hasMergeUI")),
                                T_UI, r"""(function(){return JSON.stringify({
                                  hasMergeUI: !!(window.mgPrepare && window.MergePages),
                                  ready: document.readyState});})()""")
        if not ok:
            log("FAIL 进合成页 · mgPrepare/MergePages 没挂出来")
            return 1
        try:
            if opt.phase == "before":
                code = phase_before(browser, port)
            else:
                code = phase_after(browser, port, video)
        except Exception:                                    # noqa: BLE001
            import traceback
            log("SCRIPT ERROR\n" + traceback.format_exc())
            code = 2
    finally:
        if browser:
            browser.close()
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            proc.terminate()
    fails = [x for x in ITEMS if not x["ok"]]
    log("-" * 66)
    log("汇总：%d 项 · FAIL %d 项" % (len(ITEMS), len(fails)))
    return 1 if fails else code


def open_db(b, project):
    b.eval("window.FaceDB.open({project: %s, mode: \"export\", allowExtract: true}), 1"
           % json.dumps(str(project)))
    def _ready(x):
        st = (x or {}).get("state") or {}
        return bool(st.get("open")) and not st.get("loading")
    return wait_for(b, _ready, T_DB, AUDIT_JS)


def phase_before(b, port):
    legacy = WS / "legacy"
    ok, a, dt = open_db(b, legacy)
    if not ok:
        log("FAIL 打开人脸库 · 数据没渲染出来")
        return 1
    gate = b.eval('(function(){var w=document.querySelector("#welcome");var l=document.getElementById("legalGate");'
                  'return JSON.stringify({welcome:!!(w&&getComputedStyle(w).display!=="none"),'
                  'legal:!!(l&&l.style.display==="flex")});})()')
    log("[门] %s" % json.dumps(gate, ensure_ascii=False))
    check(True, "改前页面（旧版）已渲染", "%d 个组 · %d 张脸" % (len(a["groups"]["cards"]), len(a["faces"])))
    n = b.shot(SHOT_BEFORE)
    log("[截图] %s（%d 字节）" % (SHOT_BEFORE, n))
    return 0


EXP = {}


def phase_after(b, port, video):
    EXP["legacy"] = legacy_expect(WS / "legacy")
    # ---------- 1 准备素材（真后端 prepare） ----------
    b.eval("window.mgPrepare(%s), 1" % json.dumps(str(video)))
    ok, s, dt = wait_for(b, lambda x: (not x.get("running")) and bool(x.get("ok")) and x.get("frames"),
                         T_PREPARE, PREP_STATUS, 0.6)
    if not ok:
        log("FAIL 准备素材 · %.0fs 内没到终态" % T_PREPARE)
        log(json.dumps(s, ensure_ascii=False)[:600])
        return 1
    project = str(s.get("project") or "")
    check(True, "准备素材", "%s 帧 · 项目 %s · %.1fs" % (s.get("frames"), project, dt))

    # ---------- 1b 「提取到 <dfm>」下拉必须已经不在页面上 ----------
    # 新脸归哪一组由后端按人脸 embedding 判定（像哪个组里的脸就归哪个组），
    # 所以这个下拉和 window.FaceDB.addTarget 都删了。
    gone = b.js(r"""(function(){return JSON.stringify({
        sel: !!document.getElementById("fdbDfm"),
        wrap: !!document.getElementById("fdbExtractWrap"),
        api: !!(window.FaceDB && window.FaceDB.addTarget)});})()""")
    check(not gone.get("sel") and not gone.get("wrap") and not gone.get("api"),
          "「提取到 <dfm>」下拉与 addTarget 都已删除", json.dumps(gone, ensure_ascii=False),
          {"gone": gone})

    # ---------- 2 造 6 个组（横向滚动 + 拖放目标） ----------
    names = ["模型A", "模型B", "模型C", "模型D", "模型E", "模型F"]
    raw = b.js(js_fetch("/api/merge/faces/groups", "POST",
                        json.dumps(json.dumps({"project": project, "dfms": names}))))
    body = (raw or {}).get("body") or {}
    check(bool(body.get("ok")), "建组", "6 个组 → %s" % str(body.get("project") or ""), body)

    # ---------- 3 等检测框出来，连点 5 次「＋」 ----------
    ok, s, dt = wait_for(b, lambda x: (x.get("fc") or 0) >= 1, T_DETECT,
                         r"""(function(){return JSON.stringify({fc:
                            document.querySelectorAll(".mgfc[data-face]").length});})()""")
    if not ok:
        log("FAIL 检测出框 · 等不到 .mgfc（合成器没出脸？）")
        return 1
    faces_n = len(b.js(r"""(function(){var o=[];document.querySelectorAll(".mgfc[data-face]").forEach(function(n){
        o.push(n.getAttribute("data-face"));});return JSON.stringify(o);})()"""))
    face_ids = b.js(r"""(function(){var o=[];document.querySelectorAll(".mgfc[data-face]").forEach(function(n){
        o.push(n.getAttribute("data-face"));});return JSON.stringify(o);})()""")
    check(len(face_ids) >= 2, "合成器出 2 张脸", "faces=%s" % (face_ids,))

    def plus(i):
        return b.eval("""(function(){var n=document.querySelector('.mgfc[data-face="%d"] .mgplus');
            if(!n) return 0; n.click(); return 1;})()""" % i)

    def lib_after_add():
        d = lib_get(b, project)
        return [f for f in (d.get("faces") or []) if not f.get("trashed")]

    okp = plus(int(face_ids[0]))
    check(okp == 1, "点第 1 张脸的「＋」", "face=%s" % face_ids[0])
    toasts = grab_toasts(b, timeout=10)
    s = {"n": next((x for x in toasts if x.startswith(("已加入人脸数据库", "已更新该帧的人脸"))), "")}
    first = lib_after_add()
    check(len(first) == 1, "首次「＋」新增 1 张", "库内 %d 张 · %s" % (len(first), (first or [{}])[0].get("id")),
          {"faces": first})
    fid = (first or [{}])[0].get("id") or ""
    check(str(s.get("n") or "").startswith("已加入人脸数据库"), "首次提示「已加入人脸数据库」",
          str(s.get("n") or "")[:80], s)

    for k in range(4):
        plus(int(face_ids[0]))
        time.sleep(0.35)
    more = grab_toasts(b, timeout=20)
    same = lib_after_add()
    check(len(same) == 1, "同帧连点 5 次「＋」→ 库里只有 1 张",
          "库内 %d 张" % len(same), {"faces": same})
    check(bool(same) and same[0].get("id") == fid, "替换后 id 不变",
          "首张 %s → 现在 %s" % (fid, (same or [{}])[0].get("id")), {"faces": same})
    check(any(str(x).startswith("已更新该帧的人脸") for x in more),
          "替换提示「已更新该帧的人脸」",
          " · ".join(more[-4:]), {"toasts": more[-6:]})

    # 多脸帧：第二张脸各存一张
    plus(int(face_ids[1]))
    ok, s, dt = wait_for(b, lambda x: len([f for f in (x.get("faces") or []) if not f.get("trashed")]) >= 2,
                         T_ADD, AUDIT_JS, 0.5)
    two = lib_after_add()
    check(len(two) == 2, "多脸帧各存一张", "库内 %d 张 · idx=%s"
          % (len(two), [f.get("idx") for f in two]), {"faces": two})

    # ---------- 4 打开人脸库：DOM 断言 ----------
    ok, a, dt = open_db(b, project)
    if not ok:
        log("FAIL 打开人脸库 · 数据没渲染出来")
        return 1
    a = audit(b)
    ids = a.get("ids") or {}
    check("未分组" not in (a.get("text") or ""), "页面没有「未分组」文本",
          "可见文本 %d 字" % len(a.get("text") or ""), {"text": (a.get("text") or "")[:400]})
    check(not ids.get("ungrouped") and not ids.get("ungroupedStat"), "「未分组」卡片与统计项已删",
          "fdbUngrouped=%s fdbStatUngrouped=%s" % (ids.get("ungrouped"), ids.get("ungroupedStat")))
    btns = [x for x in (a.get("buttons") or []) if x.get("vis")]
    labels = [x.get("text") for x in btns]
    close_btns = [x.get("id") for x in btns if x.get("text") == "关闭"]
    check(close_btns == ["fdbClose"], "全页只剩右上角一个「关闭」",
          "可见按钮：%s" % ("、".join(labels) or "无"), {"buttons": btns})
    check(not ids.get("continueVis"), "底部统计栏里没有「关闭/继续导出」按钮（browse 模式）",
          "fdbContinue 存在=%s 可见=%s" % (ids.get("continueBtn"), ids.get("continueVis")))
    check(ids.get("close"), "右上角关闭按钮还在", "fdbClose=%s" % ids.get("close"))
    for bad in ("移动", "移入回收站", "从当前帧提取人脸"):
        check(bad not in labels, "没有「%s」按钮" % bad, "可见按钮：%s" % ("、".join(labels) or "无"))
    check(not ids.get("moveBtn") and not ids.get("moveSel") and not ids.get("trashBtn")
          and not ids.get("extract"), "移动下拉/移动/移入回收站/提取按钮 DOM 已移除",
          json.dumps({k: ids.get(k) for k in ("moveBtn", "moveSel", "trashBtn", "extract")}))
    check(not ids.get("hint") and not ids.get("keys"), "说明性小字已删",
          "fdbHint=%s fdbKeys=%s" % (ids.get("hint"), ids.get("keys")))
    check("拖动" not in (a.get("text") or ""), "页面没有拖动说明文字",
          "", {"text": (a.get("text") or "")[:400]})
    g = a.get("groups") or {}
    check(g.get("flexDir") == "row", "组卡片横向排列", "flex-direction=%s wrap=%s"
          % (g.get("flexDir"), g.get("wrap")))
    check(g.get("overflowX") in ("auto", "scroll"), "组容器可横向滚动",
          "overflow-x=%s" % g.get("overflowX"))
    check((g.get("scrollW") or 0) > (g.get("clientW") or 0), "一行放不下时溢出（不换行堆叠）",
          "scrollWidth=%s > clientWidth=%s" % (g.get("scrollW"), g.get("clientW")))
    scrolled = b.eval("""(function(){var g=document.getElementById("fdbGroups");
        if(!g) return -1; g.scrollLeft=120; return g.scrollLeft;})()""")
    check((scrolled or 0) > 0, "确实能横向滚", "scrollLeft=%s" % scrolled)
    b.eval('(function(){var g=document.getElementById("fdbGroups"); if(g) g.scrollLeft=0; return 1})()')
    cards = g.get("cards") or []
    check(len(cards) >= 6, "6 个组卡片都在", "组头：%s" % (" | ".join(c.get("hd") or "" for c in cards))[:200])
    if cards:
        hd = cards[0].get("hd") or ""
        check(("×" in hd) and str(cards[0].get("n")) in hd, "组头有 组名+张数+输入尺寸",
              "「%s」" % hd, {"cards": cards})
    check(bool(a.get("faces")) and all("帧" in (f.get("meta") or "") for f in (a.get("faces") or [])),
          "每张脸有 帧号 + id", "faces=%s" % [(f.get("meta") or "") for f in (a.get("faces") or [])])

    n = b.shot(SHOT_AFTER)
    log("[截图] %s（%d 字节）" % (SHOT_AFTER, n))

    # ---------- 5 拖一张脸到另一个组 ----------
    face = (a.get("faces") or [{}])[0]
    src_id, src_gid = face.get("id"), face.get("gid")
    tgt = [c for c in cards if c.get("gid") != src_gid][0]

    def rect(sel):
        return b.js("""(function(){var n=%s; if(!n) return null; var r=n.getBoundingClientRect();
            return JSON.stringify({x:r.left+r.width/2, y:r.top+r.height/2,
                                   w:r.width, h:r.height});})()""" % sel)

    def drag_to(sel_expr, expect_cls):
        r1 = rect("""document.querySelector('.fdb-face[data-face-id="%s"]')""" % src_id)
        r2 = rect(sel_expr)
        if not r1 or not r2:
            return False, {"src": r1, "dst": r2}
        b.mouse("mousePressed", r1["x"], r1["y"])
        b.mouse("mouseMoved", r1["x"] + 10, r1["y"] + 10)
        steps = 4
        for k in range(1, steps + 1):
            b.mouse("mouseMoved", r1["x"] + (r2["x"] - r1["x"]) * k / steps,
                    r1["y"] + (r2["y"] - r1["y"]) * k / steps)
            time.sleep(0.05)
        time.sleep(0.15)
        hit = b.js(AUDIT_JS)
        ok_hi = bool((hit.get("trash") or {}).get("drop")) if expect_cls == "trash" else \
            any(c.get("gid") == (hit.get("state") or {}).get("x", "") for c in [])
        if expect_cls == "trash":
            ok_hi = bool((hit.get("trash") or {}).get("drop"))
        else:
            ok_hi = bool(expect_cls in (hit.get("trash") or {}).get("cls", "")) or True
        b.mouse("mouseReleased", r2["x"], r2["y"], buttons=0)
        return ok_hi, hit

    r2 = rect("""document.querySelector('.fdb-group[data-group="%s"]')""" % tgt.get("gid"))
    hi, mid = drag_to("""document.querySelector('.fdb-group[data-group="%s"]')""" % tgt.get("gid"), "group")
    log("[拖放] %s → %s · 拖动中 state.drag=%s"
        % (src_id, tgt.get("gid"),
           json.dumps((mid.get("state") or {}).get("drag"), ensure_ascii=False)))
    ok, a2, dt = wait_for(b, lambda x: any(f.get("id") == src_id and f.get("gid") == tgt.get("gid")
                                           for f in (x.get("faces") or [])),
                          T_DB, AUDIT_JS, 0.5)
    d = lib_get(b, project)
    moved = [f for f in (d.get("faces") or []) if f.get("id") == src_id]
    check(ok and moved and moved[0].get("group") == tgt.get("gid"), "拖动到组卡片 → move 生效",
          "%s：%s → %s" % (src_id, src_gid, tgt.get("gid")),
          {"want": tgt.get("gid"), "got": (moved or [{}])[0].get("group")})

    # ---------- 5b 拖到空白处（原来「未分组」区的位置）：什么都不做 ----------
    a = audit(b)
    face_b = [f for f in (a.get("faces") or []) if f.get("id") == src_id]
    if face_b:
        g_before = face_b[0].get("gid")
        r1 = rect("""document.querySelector('.fdb-face[data-face-id="%s"]')""" % src_id)
        r3 = rect("""document.querySelector(".fdb-stats")""")
        if r1 and r3:
            b.mouse("mousePressed", r1["x"], r1["y"])
            b.mouse("mouseMoved", r1["x"] + 12, r1["y"] + 12)
            for k in range(1, 5):
                b.mouse("mouseMoved", r1["x"] + (r3["x"] - r1["x"]) * k / 4,
                        r1["y"] + (r3["y"] - r1["y"]) * k / 4)
                time.sleep(0.06)
            b.mouse("mouseReleased", r3["x"], r3["y"], buttons=0)
            time.sleep(1.0)
            d0 = lib_get(b, project)
            still = [f for f in (d0.get("faces") or []) if f.get("id") == src_id]
            check(bool(still) and still[0].get("group") == g_before,
                  "拖到空白处（原未分组区）没有行为",
                  "%s 仍在 %s" % (src_id, (still or [{}])[0].get("group")),
                  {"group": (still or [{}])[0].get("group"), "want": g_before})

    # ---------- 6 拖到回收站 ----------
    a = audit(b)
    picked = [f for f in (a.get("faces") or []) if f.get("id") == src_id]
    face2 = picked[0] if picked else ((a.get("faces") or [{}])[0])
    src_id2 = face2.get("id")
    r1 = rect("""document.querySelector('.fdb-face[data-face-id="%s"]')""" % src_id2)
    r2 = rect("""document.getElementById("fdbTrash")""")
    b.mouse("mousePressed", r1["x"], r1["y"])
    b.mouse("mouseMoved", r1["x"] + 12, r1["y"] + 12)
    for k in range(1, 6):
        b.mouse("mouseMoved", r1["x"] + (r2["x"] - r1["x"]) * k / 5,
                r1["y"] + (r2["y"] - r1["y"]) * k / 5)
        time.sleep(0.06)
    time.sleep(0.2)
    hi = audit(b)
    check(bool((hi.get("trash") or {}).get("drop")), "拖到回收站时高亮",
          "class=%s" % (hi.get("trash") or {}).get("cls"))
    b.mouse("mouseReleased", r2["x"], r2["y"], buttons=0)
    ok, a3, dt = wait_for(b, lambda x: any(f.get("id") == src_id2 and f.get("trashed")
                                           for f in (x.get("faces") or [])) or
                          bool((x.get("trash") or {}).get("n")), T_DB, AUDIT_JS, 0.5)
    d = lib_get(b, project)
    tf = [f for f in (d.get("faces") or []) if f.get("id") == src_id2]
    check(bool(tf) and tf[0].get("trashed"), "拖到回收站 → trash 生效",
          "%s trashed=%s" % (src_id2, (tf or [{}])[0].get("trashed")), {"face": tf})

    # ---------- 7 回收站：还原 / 彻底删除 ----------
    ok = b.eval("""(function(){var row=document.querySelector('#fdbTrashList .fdb-trash-item[data-face-id="%s"]');
        if(!row) return 0; var b=row.querySelector('button[data-act="restore"]');
        if(!b) return -1; b.click(); return 1;})()""" % src_id2)
    check(ok == 1, "回收站「还原」按钮在", "click=%s" % ok)
    ok, a4, dt = wait_for(b, lambda x: any(f.get("id") == src_id2 and not f.get("trashed")
                                           for f in (x.get("faces") or [])), T_DB, AUDIT_JS, 0.5)
    check(ok, "「还原」把脸放回组里", "%s" % src_id2)

    # 再拖进回收站 → 彻底删除
    a = audit(b)
    r1 = rect("""document.querySelector('.fdb-face[data-face-id="%s"]')""" % src_id2)
    r2 = rect("""document.getElementById("fdbTrash")""")
    if r1 and r2:
        b.mouse("mousePressed", r1["x"], r1["y"])
        b.mouse("mouseMoved", r1["x"] + 12, r1["y"] + 12)
        for k in range(1, 6):
            b.mouse("mouseMoved", r1["x"] + (r2["x"] - r1["x"]) * k / 5,
                    r1["y"] + (r2["y"] - r1["y"]) * k / 5)
            time.sleep(0.06)
        b.mouse("mouseReleased", r2["x"], r2["y"], buttons=0)
        wait_for(b, lambda x: bool((x.get("trash") or {}).get("n")), T_DB, AUDIT_JS, 0.5)
    ok = b.eval("""(function(){var row=document.querySelector('#fdbTrashList .fdb-trash-item[data-face-id="%s"]');
        if(!row) return 0; var b=row.querySelector('button[data-act="delete"]');
        if(!b) return -1; b.click(); return 1;})()""" % src_id2)
    check(ok == 1, "回收站「彻底删除」按钮在", "click=%s" % ok)
    ok, s, dt = wait_for(b, lambda x: bool(x.get("sid")), 20,
                         r"""(function(){return JSON.stringify({sid:(document.getElementById("fdbModal")||{}).hidden===false});})()""")
    b.eval('(function(){var b=document.getElementById("fdbModalOk"); if(b) b.click(); return 1})()')
    ok, a5, dt = wait_for(b, lambda x: not any(f.get("id") == src_id2 for f in (x.get("faces") or [])),
                          T_DB, AUDIT_JS, 0.5)
    check(ok, "「彻底删除」把脸从库里去掉", "%s" % src_id2)

    # ---------- 8 右上角关闭 ----------
    b.eval('(function(){var b=document.getElementById("fdbClose"); if(b) b.click(); return 1})()')
    ok, s, dt = wait_for(b, lambda x: (not x.get("open")) and x.get("inMain"), 30,
                         r"""(function(){var v=document.getElementById("mgFaceDb");
                            return JSON.stringify({open:(window.FaceDB&&window.FaceDB.isOpen)?window.FaceDB.isOpen():null,
                              facedbIn:!!(v&&v.classList.contains("in")),
                              inMain:!!(document.querySelector('.mgview[data-view="main"]')||{}).classList.contains("in"),
                              step: window.__mergeStep||""});})()""")
    check(bool(ok), "右上角关闭 → 回到合成视图且不残留状态",
          "FaceDB.isOpen=%s · .mgview.in=%s · step=%s"
          % ((s or {}).get("open"), (s or {}).get("facedbIn"), (s or {}).get("step")), s)

    # ---------- 9 旧库自动去重 ----------
    legacy = WS / "legacy"
    log("[旧库] 打开前 marker=%s · __T=%s"
        % (json.loads((legacy / "_faces" / "index.json").read_text(encoding="utf-8")).get("dedupe_version"),
           len(b.js("(function(){return JSON.stringify(window.__T||[])})()") or [])))
    ok, a6, dt = open_db(b, legacy)
    log("[旧库] 打开后 marker=%s · __T=%s"
        % (json.loads((legacy / "_faces" / "index.json").read_text(encoding="utf-8")).get("dedupe_version"),
           b.js("(function(){return JSON.stringify(window.__T||[])})()")))
    if not ok:
        log("FAIL 打开旧库 · 数据没渲染出来")
        return 1
    a = audit(b)
    tr = (a.get("trash") or {}).get("n")
    exp = EXP["legacy"]
    check(tr == exp["trash"], "旧库同帧重复 → 多出来的进回收站",
          "回收站 %s 张（期望 %s）" % (tr, exp["trash"]), {"trash": a.get("trash"), "exp": exp})
    check(len(a.get("faces") or []) == exp["active"], "旧库去重后每帧只剩 1 张",
          "库内 %d 张（期望 %s）" % (len(a.get("faces") or []), exp["active"]),
          {"faces": a.get("faces"), "exp": exp})
    dedupe_t = [str(x) for x in (a.get("toasts") or []) if "同帧重复" in str(x)]
    check(bool(dedupe_t), "去重提示「已把 N 张同帧重复的脸移入回收站」",
          dedupe_t[-1] if dedupe_t else (" · ".join((a.get("toasts") or [])[-4:]) or "无"),
          {"toasts": (a.get("toasts") or [])[-8:]})
    log("[旧库] 去重后 库内 %d 张 · 回收站 %d 张 · toasts=%d 条"
        % (len(a.get("faces") or []), tr, len(a.get("toasts") or [])))
    first_row = b.eval("""(function(){var r=document.querySelector("#fdbTrashList .fdb-trash-item");
        if(!r) return ""; return r.getAttribute("data-face-id")||"";})()""")
    if first_row:
        b.eval("""(function(){var r=document.querySelector('#fdbTrashList .fdb-trash-item[data-face-id="%s"]');
            var b=r&&r.querySelector('button[data-act="restore"]'); if(b) b.click(); return 1;})()"""
               % first_row)
        ok, a7, dt = wait_for(b, lambda x: bool((x.get("trash") or {}).get("n") == exp["trash"] - 1),
                              T_DB, AUDIT_JS, 0.5)
        check(ok, "旧库回收站里的脸能还原", "%s → 回收站剩 %d" % (first_row, exp["trash"] - 1))
    n = b.shot(TMP / "fdb_after_dedupe.png")
    log("[截图] %s（%d 字节）" % (TMP / "fdb_after_dedupe.png", n))

    # ---------- 10 导出流程的「继续导出」没被删坏 ----------
    ok, a8, dt = open_db(b, project)
    b.eval("""(function(){window.__cont=0; window.FaceDB.open({project: %s, mode: "export",
        onContinue: function(){ window.__cont=1; }}); return 1})()""" % json.dumps(str(project)))
    ok, s, dt = wait_for(b, lambda x: bool((x.get("ids") or {}).get("continueVis")) and
                         not (x.get("state") or {}).get("loading"), 30, AUDIT_JS)
    check(ok, "导出流程里「继续导出」还在（browse 模式才隐藏）",
          "fdbContinue 可见=%s" % (s or {}).get("ids", {}).get("continueVis"))
    b.eval('(function(){var b=document.getElementById("fdbContinue"); if(b) b.click(); return 1})()')
    ok, s, dt = wait_for(b, lambda x: bool(x.get("cont")), 20,
                         r"""(function(){return JSON.stringify({cont: window.__cont||0,
                            open:(window.FaceDB.isOpen?window.FaceDB.isOpen():null)});})()""")
    check(ok, "点「继续导出」把控制权交回合成器", "onContinue 收到=%s" % (s or {}).get("cont"))

    # ---------- 11 全程没有未捕获异常 ----------
    bad = [x for x in (b.console or []) if "exception" in x.lower()]
    check(not bad, "页面没有未捕获异常", "console 记录 %d 条" % len(b.console or []),
          {"console": (b.console or [])[-3:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
