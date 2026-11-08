# -*- coding: utf-8 -*-
r"""合成链路验收：导入 → 选模型 → 检测 → 时间轴 → 预览 → 导出 → 取消。

用法
    python app/tools/merge_accept.py                  # 跑 1 轮
    python app/tools/merge_accept.py --rounds 3       # 连跑 3 轮，最后打印逐项对比表
    python app/tools/merge_accept.py --keep           # 导出跑完不取消（慢，整片）
    python app/tools/merge_accept.py 某段.mp4          # 换素材
    python app/tools/merge_accept.py --frames 600,200 # 换两个时间轴帧号
    python app/tools/merge_accept.py --scale 2        # 所有超时 ×2（慢机器）

    解释器：要带 fastapi / uvicorn / websockets 的那个。脚本自己挑 ——
    当前解释器能 import 就用它，否则退回 <仓库>/runtime/server/Scripts/python.exe。

判读
    PASS  断言成立，括号里是实测值
    FAIL  断言不成立；下跟一行「实际：{...}」，把拿到的 JSON / DOM 快照打出来
    SKIP  前置缺失（没素材 / 没 .dfm / 没成片 / 没 Edge / 没 websockets）—— 不算失败
    退出码：0 = 没有 FAIL（可以有 SKIP）· 1 = 有 FAIL · 2 = 脚本自己炸了

前置
    1. 后端能起：app/serve.py（端口由 serve.py 自己从 8760 起找）
    2. 素材视频：<仓库>/workspace/*.mp4，默认 9月29日 (5).mp4
    3. 工作区：$DFL_WORKSPACE > <引擎根>/workspace（要有 model/*.dfm，默认挑 model_004）
    4. 对齐素材：<工作区>/merge/<视频名>/ —— 首次运行现场抽帧 + 对齐，几分钟
    5. 成片：<工作区>/merge/<视频名>_merged.mp4 —— 只有「预览」这项要它；
       全新环境想看到它绿，先跑一次 --keep，否则这项是 SKIP
    6. Edge（--headless=new）+ websockets

稳定性约定（这一版改的就是这些）
    · 没有一处 sleep 硬等：每个异步动作都轮询到「终态 / 条件成立」再断言，
      超时是显式常量（T_*），失败时把 JSON、DOM 快照、toast 一起打出来。
    · 每轮都是全新的后端 + 全新的 Edge（独立端口 + 独立 profile + taskkill /T /F）：
      旧的「固定 9333 端口 + 固定 %TEMP%\edge-cdp-prof」在上一轮浏览器没死透时，
      新进程会把命令行转发给老实例然后立刻退出，脚本驱动的其实是上一轮那个页面。
    · 断言看「当前模式真正在画图的那个 img」：视频模式是 #mgPreview，
      帧目录（DFL 项目）模式是 #mgStage img.mgfrm。只读 #mgPreview 在项目模式下
      永远是空串 —— 旧脚本「时间轴换图 / 预览」必 FAIL 就是错在这儿。
"""

from __future__ import annotations

import argparse
import importlib.util
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
LOG_DIR = TMP / "merge_accept"

# ============================== 超时（秒） ==============================
# 一轮正常耗时：起后端 3s + 导入 10s + 检测 7s + 时间轴 2s + 预览 2s + 取消 2s
T_SERVER_START = 90      # serve.py 打出 PORT= 之前
T_BROWSER = 40           # Edge 起来 + CDP 连上
T_PAGE = 60              # 首页 readyState=complete
T_MERGE_UI = 60          # 合成页把 mgPrepare / MergePages 挂出来
T_PREPARE = 1800         # 抽帧 + 对齐（首次）；复用已准备好的项目只要几秒
T_MODELS = 90            # 模型列表渲染（/api/state 回来才会画）
T_FRAME = 60             # 时间轴改帧 → 舞台出图
T_DETECT = 120           # 检测视图出框（首次要加载 onnx，实测 6~8s）
T_PREVIEW = 90           # 合成视图出成片帧
T_EXPORT_START = 180     # 导出真的在干活（子进程报了 stage/pct）
T_CANCEL = 300           # 取消到终态
T_KEEP = 3600            # --keep：整片跑完

DET_NAMES = ("RetinaFace_10g", "TinyMog", "TinyMog")
ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv",
            "h264_amf", "hevc_amf", "libx264", "libx265")


# ============================== 路径 ==============================
def _repo_root():
    """仓库根。app/tools/<本文件> → parents[1]=app、parents[2]=仓库根。

    优先用 app/server/_paths.py 里的 ROOT（唯一的真源），拿不到才退回 parents[2]；
    两条路都不写死盘符。旧版写成 parents[1]（以为自己在 <仓库>/tools/），
    于是 RL = <仓库>/app，后面 cwd=<仓库>/app/app → NotADirectoryError 直接跑不起来。
    """
    here = Path(__file__).resolve()
    fallback = here.parents[2] if len(here.parents) > 2 else here.parent
    for cand in (fallback, here.parents[1] if len(here.parents) > 1 else fallback):
        mod = cand / "app" / "server" / "_paths.py"
        if not mod.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("dfl_paths_for_accept", mod)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)                  # 只 import os/pathlib，没有副作用
            root = Path(getattr(m, "ROOT"))
            if (root / "app" / "server" / "_paths.py").is_file():
                return root, m
        except Exception:                               # noqa: BLE001
            pass
    return fallback, None


ROOT, _PATHS = _repo_root()
RL = str(ROOT)                       # 兼容旧脚本里的写法
APP_DIR = ROOT / "app"
DEFAULT_VIDEO = ROOT / "workspace" / "9月29日 (5).mp4"

EDGE_CANDIDATES = (
    os.environ.get("EDGE_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    shutil.which("msedge") or "",
    shutil.which("chrome") or "",
)

try:
    from websockets.sync.client import connect as _ws_connect
except Exception:                    # noqa: BLE001
    _ws_connect = None

_OPENER = None


def _http_get(port, path="/json/list", timeout=2.0):
    import urllib.request
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _OPENER.open("http://127.0.0.1:%d%s" % (port, path), timeout=timeout).read().decode()


def _free_port(start=9333):
    for p in range(start, start + 300):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("9333~9632 都被占用了，腾一个出来再跑")


def _edge_path():
    for c in EDGE_CANDIDATES:
        if c and Path(c).is_file():
            return c
    return ""


def _server_python():
    """能 import fastapi/uvicorn/websockets 的解释器。"""
    cands = [sys.executable,
             str(ROOT / "runtime" / "server" / "Scripts" / "python.exe"),
             shutil.which("python") or "",
             shutil.which("python3") or ""]
    seen = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        if not Path(c).is_file():
            continue
        try:
            r = subprocess.run([c, "-c", "import fastapi, uvicorn, websockets"],
                               capture_output=True, timeout=120)
            if r.returncode == 0:
                return c
        except Exception:                               # noqa: BLE001
            continue
    return ""


def _workspace_candidates():
    r"""后端工作区候选（模型、merge 产物都在这儿）。

    旧版写死成 D:\MyProject\DeepFaceLab-PyTorch\workspace：换台机器就「没有模型」，
    而这台机器上仓库自己的 workspace/ 里又只有半个工作区（没备好的项目、没成片）。
    $DFL_WORKSPACE 永远最高优先且唯一；没设就下面这几个一起打分挑。
    """
    env = os.environ.get("DFL_WORKSPACE", "").strip()
    if env:
        return [Path(env)]
    out, seen = [], set()
    roots = [ROOT / "workspace", ROOT.parent / "DeepFaceLab-PyTorch" / "workspace"]
    if _PATHS is not None:
        try:
            roots.append(Path(_PATHS.engine_root()) / "workspace")
        except Exception:                               # noqa: BLE001
            pass
    for c in roots:
        k = str(c).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _workspace_note(p, stem=""):
    r"""给工作区打分：备好本素材的项目(+10) > 有成片(+5) > 有 .dfm(+1)。"""
    if not p.is_dir():
        return -1, "目录不存在"
    score, notes = 0, []
    proj = p / "merge" / stem if stem else None
    if proj is not None and proj.is_dir() and any(proj.glob("*.jpg")):
        score += 10
        notes.append("已备好本素材的项目")
    if stem and (p / "merge" / (stem + "_merged.mp4")).is_file():
        score += 5
        notes.append("有成片")
    try:
        n = len(list((p / "model").glob("*.dfm"))) if (p / "model").is_dir() else 0
    except OSError:
        n = 0
    if n:
        score += 1
        notes.append("%d 个 .dfm" % n)
    return score, "；".join(notes) or "空"


def _workspace(stem=""):
    r"""挑一个工作区，返回 (Path, 为什么是它)。"""
    cands = _workspace_candidates()
    if len(cands) == 1:
        return cands[0], "DFL_WORKSPACE 指定"
    best, best_score, best_why = cands[0], -2, "默认"
    for c in cands:
        sc, why = _workspace_note(c, stem)
        if sc > best_score:
            best, best_score, best_why = c, sc, why
    return best, best_why


def _weights_file(name):
    base = None
    if _PATHS is not None:
        try:
            base = Path(_PATHS.weights_dir())
        except Exception:                               # noqa: BLE001
            base = None
    return (base or (ROOT / "weights")) / name


# ============================== 最小 CDP 客户端 ==============================
class Browser:
    """独立端口 + 独立 profile；close() 用 taskkill /T /F 连子进程一起收。"""

    def __init__(self, edge):
        self.edge = edge
        self.port = _free_port()
        self.profile = Path(tempfile.mkdtemp(prefix="edge-merge-accept-", dir=str(TMP)))
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

    # ---- 连接 ----
    def _open(self):
        if _ws_connect is None:
            raise RuntimeError("没有 websockets：用 runtime\\server\\Scripts\\python.exe 跑")
        deadline = time.time() + T_BROWSER
        last = ""
        while time.time() < deadline:
            try:
                data = json.loads(_http_get(self.port))
                pages = [t for t in data if t.get("type") == "page"]
                if pages:
                    # 挑一个“干净”的页：优先 about:blank / data: / 空 url，别挑扩展页、引导页
                    pages.sort(key=lambda t: 0 if (t.get("url") or "").startswith(("about:", "data:", "")) else 1)
                    self.ws = _ws_connect(pages[0]["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024)
                    self.call("Page.enable")
                    self.call("Runtime.enable")
                    self.call("Log.enable")
                    self.call("Emulation.setDeviceMetricsOverride", width=1624, height=1046,
                              deviceScaleFactor=1, mobile=False)
                    return
            except Exception as e:                      # noqa: BLE001
                last = str(e)[:120]
                time.sleep(0.4)
        raise RuntimeError("Edge CDP 连不上（port=%d，最后错误 %s）" % (self.port, last or "无"))

    def _drain(self, m):
        if m.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded", "Runtime.consoleAPICalled"):
            try:
                self.console.append(json.dumps(m.get("params"), ensure_ascii=False)[:300])
            except Exception:                           # noqa: BLE001
                pass
            del self.console[:-20]

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
            self._drain(m)
        raise RuntimeError("CDP %s 超时" % method)

    # ---- 求值 ----
    def eval(self, js, timeout=300):
        r = self.call("Runtime.evaluate", expression=js, returnByValue=True,
                      awaitPromise=True, timeout=int(timeout * 1000))
        if r.get("exceptionDetails"):
            raise RuntimeError("页面里抛异常：%s" % json.dumps(r["exceptionDetails"], ensure_ascii=False)[:400])
        return r.get("result", {}).get("value")

    def js(self, js):
        """页面返回 JSON 字符串 → dict。"""
        v = self.eval(js)
        if v is None:
            return {}
        return json.loads(v) if isinstance(v, str) else v

    def goto(self, url):
        self.call("Page.navigate", url=url)
        deadline = time.time() + T_PAGE
        while time.time() < deadline:
            try:
                if self.eval("document.readyState") == "complete":
                    return True
            except Exception:                           # noqa: BLE001
                pass
            time.sleep(0.25)
        return False

    def close(self):
        try:
            self.ws.close()
        except Exception:                               # noqa: BLE001
            pass
        try:                                            # 只 terminate 会漏掉渲染/GPU 子进程
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:                               # noqa: BLE001
            try:
                self.proc.terminate()
            except Exception:                           # noqa: BLE001
                pass
        try:
            shutil.rmtree(self.profile, ignore_errors=True)
        except Exception:                               # noqa: BLE001
            pass


# ============================== 页面探针 ==============================
SNAP_JS = r"""
(function(){
  var q = function(s){ return document.querySelector(s); };
  var st = q("#mgStage");
  var vid = q("#mgPreview");
  var frm = st ? st.querySelector("img.mgfrm") : null;
  var videoMode = !!(window.mgVideoMode && window.mgVideoMode());
  var cur = videoMode ? vid : frm;
  var other = videoMode ? frm : vid;
  var cv = st ? st.querySelector("canvas.mgovl") : null;
  var ink = -1;
  if (cv && cv.width && cv.height) {
    try {
      var d = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data, n = 0;
      for (var i = 3; i < d.length; i += 4) if (d[i] > 10) n++;
      ink = n;
    } catch (e) { ink = -2; }
  }
  var hintEl = document.getElementById("mgStageHint");
  var rows = Array.prototype.map.call(document.querySelectorAll("#mgModels .mgitem"), function(x){
    var nm = x.querySelector(".nm");
    return { n: nm ? String(nm.textContent||"").trim() : "",
             on: x.classList.contains("on"),
             dfm: !!x.querySelector(".tag.ok") };
  });
  var s = {};
  try {
    var p = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {};
    s = { mode:p.mode, fmt:p.image_format, enc:p.encoder, color:p.color_transfer,
          nw:p.num_workers, detector:p.detector, model:p.model };
  } catch (e) { s = { err:String(e) }; }
  return JSON.stringify({
    url: location.href,
    ready: document.readyState,
    hasMergeUI: !!(window.mgPrepare && window.MergePages),
    videoMode: videoMode,
    streamDown: !!(window.__mgVideo||{}).streamDown,
    mvFrames: (window.__mv||{}).frames || 0,
    mvFps: (window.__mv||{}).fps || 0,
    mvVideo: (window.__mv||{}).video || "",
    mvPath: (window.__mv||{}).path || "",
    mvT: (window.__mv||{}).t || 0,
    prepRunning: !!(window.__mergePrep||{}).running,
    inputDir: window.__mergeInputDir || "",
    exporting: !!window.__mergeExporting,
    frameInput: (document.getElementById("mtlFrame")||{}).value || "",
    tlFrame: (window.MergeTimeline && window.MergeTimeline.getFrame) ? window.MergeTimeline.getFrame() : null,
    view: (function(){ var b = q("#mgViews b.on"); return b ? b.getAttribute("data-v") : null; })(),
    imgWhich: videoMode ? "#mgPreview" : "img.mgfrm",
    imgLen: cur ? String(cur.src||"").length : -1,
    imgSig: cur ? String(cur.src||"").slice(0, 72) : "",
    otherLen: other ? String(other.src||"").length : -1,
    ink: ink,
    ovl: cv ? [cv.width, cv.height] : null,
    hint: hintEl ? (hintEl.style.display === "none" ? "" : String(hintEl.textContent||"").trim()) : "",
    det: (document.getElementById("mgDetector")||{}).value || "",
    models: rows,
    toasts: Array.prototype.map.call(document.querySelectorAll("#toastroot .toast"), function(t){
      return String(t.textContent||"").trim().slice(0, 90);
    }),
    exportNet: (window.__mgAcceptNet || []).slice(-3),
    settings: s
  });
})()
"""


# 页面里给 fetch 挂钩子：只记 /api/merge/export* 的真实响应体，失败时拿它当证据。
# 不拦不改，原请求照走（clone 出来的副本读 text）。
HOOK_JS = r"""
(function(){
  if (window.__mgAcceptHooked) return 1;
  window.__mgAcceptHooked = 1;
  window.__mgAcceptNet = [];
  var of = window.fetch;
  window.fetch = function(u, o){
    var url = String((u && u.url) ? u.url : u);
    var p = of.apply(this, arguments);
    if (url.indexOf("/api/merge/export") === 0) {
      try {
        p.then(function(r){ r.clone().text().then(function(t){
              window.__mgAcceptNet.push({url:url, status:r.status, body:String(t).slice(0,400)});
            }, function(){}); },
          function(e){ window.__mgAcceptNet.push({url:url, error:String(e)}); });
      } catch(e) {}
    }
    return p;
  };
  return 1;
})()
"""


def _js_fetch(url, method="GET", body="null"):
    """页面里发一次 fetch，返回 JSON 字符串 {status, body}。"""
    return ("(async function(){try{var r=await fetch(%s,{method:%s,body:%s,headers:{\"Content-Type\":\"application/json\"}});"
            "var t=await r.text();var d=t;try{d=JSON.parse(t);}catch(e){d={_text:String(t).slice(0,300)};}"
            "return JSON.stringify({status:r.status,body:d});}catch(e){return JSON.stringify({status:0,body:{error:String(e)}});}})()"
            % (json.dumps(url), json.dumps(method), body))


def _short(d, n=700):
    try:
        s = json.dumps(d, ensure_ascii=False)
    except Exception:                                   # noqa: BLE001
        s = str(d)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def wait_snap(b, pred, timeout, interval=0.4):
    """轮询页面快照到 pred(snap) 为真。返回 (ok, 最后一份 snap, 秒)。"""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = b.js(SNAP_JS)
        except Exception as e:                          # noqa: BLE001
            last = {"error": str(e)[:200]}
        if pred(last):
            return True, last, round(time.time() - t0, 2)
        time.sleep(interval)
    return False, last, round(time.time() - t0, 2)


def wait_status(b, url, pred, timeout, interval=0.5, method="GET"):
    """轮询一个接口的 body 到 pred(body) 为真。返回 (ok, 最后 body, 秒)。"""
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        try:
            raw = b.js(_js_fetch(url, method))
            last = (raw or {}).get("body") or {}
        except Exception as e:                          # noqa: BLE001
            last = {"error": str(e)[:200]}
        if pred(last):
            return True, last, round(time.time() - t0, 2)
        time.sleep(interval)
    return False, last, round(time.time() - t0, 2)


def _set_frame(b, n):
    return b.eval("""(function(){var f=document.getElementById("mtlFrame");
        if(!f) return 0; f.value=String(%d); f.dispatchEvent(new Event("change")); return 1})()""" % int(n))


# ============================== 一轮 ==============================
class Round:
    def __init__(self, idx, opt, log_fh):
        self.idx = idx
        self.opt = opt
        self.fh = log_fh
        self.items = []

    def _emit(self, status, what, detail="", ctx=None):
        # 项目名里本来就有「：」，明细用 " · " 隔开，别出现「：：」
        head = "%-4s %s" % (status, what)
        if detail:
            head += " · " + detail
        self.fh.write(head + "\n")
        print(head)
        if ctx is not None and status != "PASS":
            line = "     实际：%s" % _short(ctx, 900)
            self.fh.write(line + "\n")
            print(line)
        self.fh.flush()
        self.items.append({"what": what, "status": status, "detail": detail})

    def ok(self, flag, what, detail="", ctx=None):
        self._emit("PASS" if flag else "FAIL", what, detail, None if flag else (ctx if ctx is not None else {}))
        return bool(flag)

    def skip(self, what, reason, ctx=None):
        self._emit("SKIP", what, reason, ctx)

    def fail(self, what, detail="", ctx=None):
        self._emit("FAIL", what, detail, ctx if ctx is not None else {})


def _server_start(py, workspace, log_fh):
    # PYTHONIOENCODING 定死成 utf-8：否则遇到外部把控制台设成别的编码时，
    # 读 serve.py 的 stdout 会在解码上炸（UnicodeDecodeError，一行都读不到）。
    env = dict(os.environ, DFL_WORKSPACE=str(workspace),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([py, "serve.py"], cwd=str(APP_DIR),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=env, creationflags=0x08000000)
    tail = []
    port = None
    t0 = time.time()
    while time.time() - t0 < T_SERVER_START:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        tail.append(line.rstrip())
        del tail[:-40]
        if line.startswith("PORT="):
            try:
                port = int(line.strip().split("=")[1])
            except ValueError:
                port = None
            break

    def _pump():
        try:
            for ln in proc.stdout:
                tail.append(ln.rstrip())
                del tail[:-40]
        except Exception:                               # noqa: BLE001
            pass

    threading.Thread(target=_pump, daemon=True, name="merge-accept-srvlog").start()
    if port:
        log_fh.write("[后端] %s serve.py · 工作区 %s · port=%d\n" % (py, workspace, port))
    return proc, port, tail


def run_round(idx, opt, log_fh):
    r = Round(idx, opt, log_fh)
    video = Path(opt.video)
    f1, f2 = opt.frames
    workspace = Path(opt.workspace)
    project = workspace / "merge" / video.stem
    log_fh.write("=" * 72 + "\n第 %d 轮 · 素材 %s · 帧 %s→%s · 工作区 %s\n"
                 % (idx, video, f1, f2, workspace))

    # ---------- 前置：缺一个就整轮 SKIP，不冒充 ERROR ----------
    edge = _edge_path()
    if not edge:
        r.skip("整轮", "找不到 Edge（装 Edge 或设 EDGE_PATH）")
        return r.items
    py = _server_python()
    if not py:
        r.skip("整轮", "没有能 import fastapi/uvicorn/websockets 的 python（用 runtime\\server\\Scripts\\python.exe 跑）")
        return r.items
    if _ws_connect is None:
        r.skip("整轮", "当前解释器没有 websockets（用 runtime\\server\\Scripts\\python.exe 跑）")
        return r.items
    if not video.is_file():
        r.skip("整轮", "素材视频不在：%s（放一个到 workspace/ 或用参数指定）" % video)
        return r.items

    proc = browser = None
    tail = []
    try:
        proc, port, tail = _server_start(py, workspace, log_fh)
        if not port:
            r.fail("后端启动", "serve.py 没打出 PORT=（%.0fs）" % T_SERVER_START, {"server_log": tail[-12:]})
            return r.items

        t_boot = time.time()
        browser = Browser(edge)
        if not browser.goto("http://127.0.0.1:%d/" % port):
            r.fail("打开界面", "readyState 一直不到 complete（%ds）" % T_PAGE,
                   {"url": "http://127.0.0.1:%d/" % port, "console": browser.console[-4:]})
            return r.items

        # 进合成页：等 mgPrepare / MergePages 真的挂出来（页面自己渲染，不猜 sleep）
        browser.eval('window.go && window.go("merge"), 1')
        ok, s, dt = wait_snap(browser, lambda x: bool(x.get("hasMergeUI")) and x.get("ready") == "complete",
                              T_MERGE_UI, 0.3)
        if not ok:
            r.fail("进合成页", "mgPrepare/MergePages 没挂出来（%.0fs）" % T_MERGE_UI,
                   {"snap": s, "console": browser.console[-4:]})
            return r.items
        browser.eval(HOOK_JS)                      # 记录导出请求的真实响应（失败时当证据）
        log_fh.write("[时序] 起后端 + 进合成页 %.1fs\n" % (time.time() - t_boot))

        # ---------- 1 导入素材 ----------
        t0 = time.time()
        browser.eval("window.mgPrepare(%s), 1" % json.dumps(str(video)))
        # 终态 = 准备任务停了「而且」前端已经把项目 open 上（__mv.frames 落下）——
        # 只看 !__mergePrep.running 会抢在 mgOpenPrepared 的 fetch 前面，读到 frames=0
        ok, s, dt = wait_snap(browser, lambda x: (not x.get("prepRunning")) and (x.get("mvFrames") or 0) > 0,
                              T_PREPARE * opt.scale, 0.5)
        prow = browser.js(_js_fetch("/api/merge/prepare/status"))
        pst = (prow or {}).get("body") or {}
        if pst.get("ok") is False:
            r.fail("导入素材", "准备失败：%s" % str(pst.get("error"))[:120], {"prepare/status": pst})
        elif not ok:
            r.fail("导入素材", "%.0fs 内没到终态（prepRunning=%s, frames=%s）"
                   % (T_PREPARE * opt.scale, (s or {}).get("prepRunning"), (s or {}).get("mvFrames")),
                   {"snap": s, "prepare/status": pst, "server_log": tail[-8:]})
        else:
            r.ok(True, "导入素材", "%d 帧 · %.1f fps · %s · %d 张对齐脸 · 用时 %.1fs"
                 % (s.get("mvFrames") or 0, s.get("mvFps") or 0,
                    "video" if s.get("videoMode") else "project",
                    pst.get("aligned") or 0, time.time() - t0))
        total = int((s or {}).get("mvFrames") or pst.get("frames") or 0)

        # ---------- 2 选模型 ----------
        ok, s, dt = wait_snap(browser, lambda x: len(x.get("models") or []) > 0, T_MODELS, 0.3)
        rows = (s or {}).get("models") or []
        if not ok or not rows:
            r.skip("选模型", "模型列表没渲染出来（工作区 %s 里没有模型？）" % workspace, {"snap": s})
        else:
            want = [m for m in rows if m.get("n") == "model_004" and m.get("dfm")] \
                or [m for m in rows if m.get("dfm")]
            if not want:
                r.skip("选模型", "工作区里没有一个模型导出过 .dfm（先训练页「导出」）", {"models": rows})
            else:
                name = want[0]["n"]
                browser.eval("""(function(){var r=document.querySelectorAll("#mgModels .mgitem");
                    for(var i=0;i<r.length;i++){var nm=r[i].querySelector(".nm");
                      if(nm && String(nm.textContent||"").trim()===%s){r[i].click();return 1}}return 0})()"""
                             % json.dumps(name))
                ok2, s2, _ = wait_snap(browser, lambda x: name in [m.get("n") for m in (x.get("models") or []) if m.get("on")],
                                       T_MODELS, 0.2)
                picked = [m["n"] for m in (s2 or {}).get("models") or [] if m.get("on")]
                r.ok(ok2, "选模型", "%s（列表 %d 个 · 选中 %s）" % (name, len(rows), picked),
                     ctx=None if ok2 else {"snap": s2})
                s = s2 or s

        # ---------- 3 检测器 ----------
        det_now = (s or {}).get("det") or ""
        det_name = det_now if det_now in DET_NAMES else DET_NAMES[0]
        browser.eval("""(function(){var sel=document.getElementById("mgDetector");
            if(sel){sel.value=%s; sel.dispatchEvent(new Event("change"));}return 1})()""" % json.dumps(det_name))
        det_name = browser.js(SNAP_JS).get("det") or det_name

        if total <= 0:
            for w in ("时间轴：定位帧", "检测参数：出框", "时间轴：换帧换图",
                      "预览：合成视图", "合成参数可采集", "导出：任务已启动",
                      "导出：进度推进", "导出：取消已停止"):
                r.skip(w, "总帧数还是 0（素材没打开）")
            return r.items

        f1c = max(0, min(int(f1), total - 1))
        f2c = max(0, min(int(f2), total - 1))
        if f1c == f2c:
            f2c = 0 if f1c > 0 else min(total - 1, 1)

        # ---------- 4 时间轴：定位帧 ----------
        base = browser.js(SNAP_JS)
        _set_frame(browser, f1c)
        ok, s, dt = wait_snap(browser,
                              lambda x: (x.get("imgLen") or -1) > 0 and
                                        (x.get("imgSig") != base.get("imgSig") or x.get("imgLen") != base.get("imgLen")),
                              T_FRAME * opt.scale, 0.25)
        r.ok(ok, "时间轴：定位帧", "帧 %d 出图（%s 长度 %s→%s · view=%s · %.2fs）"
             % (f1c, (s or {}).get("imgWhich"), base.get("imgLen"), (s or {}).get("imgLen"),
                (s or {}).get("view"), dt),
             ctx=None if ok else {"snap": s, "frame": f1c, "base": base})

        # ---------- 5 检测：出框 ----------
        # 前置：这一帧得真有脸，「不出框」才轮得到怪界面。用 /api/merge/detect 探
        # （和点「检测」时同一个接口）；没脸就前后挪几帧 —— 断言仍然只认 canvas 墨迹。
        probe = browser.js(_js_fetch("/api/merge/detect/%d?model=%s&thresh=0.5" % (f1c, det_name)))
        pbody = (probe or {}).get("body") or {}
        det_frame = f1c
        if (probe or {}).get("status") != 200:
            r.skip("检测参数：出框", "检测接口 %s：%s" % ((probe or {}).get("status"), str(pbody.get("error"))[:80]),
                   {"detect": probe})
        elif not (pbody.get("boxes") or []):
            found = None
            for d in (1, -1, 2, -2, 5, 25, -25, 60):
                cand = f1c + d
                if cand < 0 or cand >= total:
                    continue
                pr = browser.js(_js_fetch("/api/merge/detect/%d?model=%s&thresh=0.5" % (cand, det_name)))
                if (pr or {}).get("status") == 200 and ((pr.get("body") or {}).get("boxes") or []):
                    found = (cand, pr)
                    break
            if not found:
                r.skip("检测参数：出框", "帧 %d 前后 8 帧都没人脸（--frames 换一帧，或换素材）" % f1c,
                       {"detect": probe})
            else:
                det_frame, probe = found
                pbody = probe.get("body") or {}
                log_fh.write("[说明] 帧 %d 没人脸，检测这项改用帧 %d（%d 个框）\n"
                             % (f1c, det_frame, len(pbody.get("boxes") or [])))
                base2 = browser.js(SNAP_JS)
                _set_frame(browser, det_frame)
                wait_snap(browser, lambda x: (x.get("imgLen") or -1) > 0 and x.get("imgSig") != base2.get("imgSig"),
                          T_FRAME * opt.scale, 0.25)
        if (probe or {}).get("status") == 200 and (pbody.get("boxes") or []):
            browser.eval('document.querySelector("#mgViews b[data-v=\\"det\\"]").click(), 1')
            ok, s, dt = wait_snap(browser, lambda x: (x.get("ink") or 0) > 500, T_DETECT * opt.scale, 0.25)
            ovl = (s or {}).get("ovl") or [0, 0]
            r.ok(ok, "检测参数：出框", "帧 %d 画布像素 %s（%s×%s）· API boxes=%d · view=%s · %.2fs"
                 % (det_frame, (s or {}).get("ink"), ovl[0], ovl[1],
                    len(pbody.get("boxes") or []), (s or {}).get("view"), dt),
                 ctx=None if ok else {"detect": probe, "snap": s,
                                      "toasts": (s or {}).get("toasts"), "console": browser.console[-4:]})

        # ---------- 6 时间轴：换帧换图 ----------
        before = browser.js(SNAP_JS)
        _set_frame(browser, f2c)
        ok, s, dt = wait_snap(browser,
                              lambda x: (x.get("imgLen") or -1) > 1000 and
                                        (x.get("imgSig") != before.get("imgSig") or x.get("imgLen") != before.get("imgLen")),
                              T_FRAME * opt.scale, 0.25)
        r.ok(ok, "时间轴：换帧换图", "帧 %d→%d · %s 换图（长度 %s→%s · %.2fs）"
             % (f1c, f2c, (s or {}).get("imgWhich"), before.get("imgLen"), (s or {}).get("imgLen"), dt),
             ctx=None if ok else {"snap": s, "before": before})

        # ---------- 7 预览：合成视图 ----------
        outs = [project.parent / (project.name + "_merged.mp4"), project / "merged.mp4"]
        out_mp4 = next((p for p in outs if p.is_file()), None)
        if out_mp4 is None:
            r.skip("预览：合成视图", "还没有成片（%s）—— 先跑一次 --keep 生成它" % outs[0].name)
        else:
            before = browser.js(SNAP_JS)
            idx_swap = int(before.get("tlFrame") if before.get("tlFrame") is not None else f2c)
            browser.eval('document.querySelector("#mgViews b[data-v=\\"swap\\"]").click(), 1')
            ok, s, dt = wait_snap(browser,
                                  lambda x: (x.get("imgLen") or -1) > 10000 and
                                            (x.get("imgSig") != before.get("imgSig") or x.get("imgLen") != before.get("imgLen")),
                                  T_PREVIEW * opt.scale, 0.25)
            api = browser.js(_js_fetch("/api/merge/merged/%d?w=900" % idx_swap))
            abody = (api or {}).get("body") or {}
            r.ok(ok, "预览：合成视图", "帧 %d 显示成片（%s 长度 %s→%s · API %s/%d 字节 · %.2fs）"
                 % (idx_swap, (s or {}).get("imgWhich"), before.get("imgLen"), (s or {}).get("imgLen"),
                    (api or {}).get("status"), len(str(abody.get("png") or "")), dt),
                 ctx=None if ok else {"merged_api": {"status": (api or {}).get("status"),
                                                     "error": abody.get("error")},
                                      "snap": s, "hint": (s or {}).get("hint")})

        # ---------- 8 合成参数可采集 ----------
        st = (browser.js(SNAP_JS).get("settings")) or {}
        good = (str(st.get("enc") or "") in ENCODERS
                and str(st.get("fmt") or "") in ("jpg", "png")
                and bool(st.get("mode")))
        r.ok(good, "合成参数可采集", _short(st, 200), ctx=None if good else {"settings": st, "snap": browser.js(SNAP_JS)})

        # ---------- 9 导出：任务已启动 ----------
        browser.eval('document.getElementById("mgExport").click(), 1')
        input_dir = browser.js(SNAP_JS).get("inputDir") or ""
        if not input_dir:
            r.skip("导出：任务已启动", "前端没有 __mergeInputDir（素材没 open 上）", {"snap": browser.js(SNAP_JS)})
            r.skip("导出：进度推进", "同上")
            r.skip("导出：取消已停止", "同上")
            return r.items

        browser.eval('document.getElementById("mgStartExport2").click(), 1')
        ctrl = "/api/merge/export/status"

        def _ctx(b, extra=None):
            """失败上下文：状态 + 那条 POST 的真实响应 + 页面 toast。"""
            ctx = {"export/status": {k: b.get(k) for k in
                                     ("ok", "phase", "running", "cancelled", "pct", "stage", "msg",
                                      "error", "out", "log_tail")}}
            snap = browser.js(SNAP_JS)
            ctx["export_net"] = [n for n in (snap.get("exportNet") or [])
                                 if str(n.get("url") or "").startswith("/api/merge/export")]
            ctx["toasts"] = snap.get("toasts")
            if extra:
                ctx.update(extra)
            return ctx

        # 「任务已启动」= 后端把任务接住了（running=true）。注意 idle 不是终态：
        # 点完到路由置 running 之间有 1s 级延迟，把 idle 当“结束”会误判（第一版就栽这儿）。
        ok, st, dt = wait_status(
            browser, ctrl,
            lambda b: b.get("running") is True or b.get("phase") in ("failed", "cancelled", "done"),
            T_EXPORT_START * opt.scale, 0.3)
        started = st.get("running") is True
        if started:
            r.ok(True, "导出：任务已启动", "phase=%s · %.2fs" % (st.get("phase"), dt))
        else:
            net = [n for n in (browser.js(SNAP_JS).get("exportNet") or [])
                   if str(n.get("url") or "").startswith("/api/merge/export")]
            if not net:
                why = "点了「开始导出」但页面没发出 /api/merge/export 请求（按钮/处理器没绑上？）"
            elif all(int(n.get("status") or 0) >= 400 for n in net):
                why = "POST /api/merge/export 被拒：%s" % str(net[-1].get("body"))[:120]
            else:
                why = "POST 返回了，但后端没进入 running（phase=%s）" % st.get("phase")
            r.fail("导出：任务已启动", why, _ctx(st))

        # 真的在干活（子进程报了阶段/进度），不是只被接住
        if started:
            ok, st, dt = wait_status(
                browser, ctrl,
                lambda b: (b.get("running") is False) or int(b.get("stage") or -1) >= 0 or int(b.get("pct") or 0) > 0,
                T_EXPORT_START * opt.scale, 0.5)
            moved = int(st.get("stage") or -1) >= 0 or int(st.get("pct") or 0) > 0
            r.ok(moved, "导出：进度推进", "stage=%s · pct=%s · msg=%s · %.1fs"
                 % (st.get("stage"), st.get("pct"), str(st.get("msg"))[:40], dt),
                 ctx=None if moved else _ctx(st))
        else:
            r.skip("导出：进度推进", "任务没启动（见上一条）", _ctx(st))

        # ---------- 10 导出：取消 ----------
        if opt.keep:
            ok, st3, dt = wait_status(browser, ctrl, lambda b: b.get("running") is False, T_KEEP, 5.0)
            outp = str(st3.get("out") or "")
            size = os.path.getsize(outp) if outp and os.path.exists(outp) else -1
            good = bool(st3.get("ok")) and size > 0
            r.ok(good, "导出：整片完成", "%s · %s 字节 · %.0fs" % (outp, size, dt),
                 ctx=None if good else {"export/status": st3})
        elif not started and st.get("phase") == "done":
            r.skip("导出：取消已停止", "取消前整片就跑完了（素材太短）", {"export/status": st})
        else:
            cres = browser.js(_js_fetch("/api/merge/export/cancel", "POST"))
            ok, st2, dt = wait_status(browser, ctrl, lambda b: b.get("running") is False, T_CANCEL * opt.scale, 0.5)
            stopped = (st2.get("running") is False) and bool(
                st2.get("cancelled") or st2.get("phase") in ("cancelled", "done"))
            r.ok(stopped, "导出：取消已停止", "phase=%s · cancelled=%s · %.2fs"
                 % (st2.get("phase"), st2.get("cancelled"), dt),
                 ctx=None if stopped else {"cancel": cres,
                                           "export/status": {k: st2.get(k) for k in
                                                             ("ok", "phase", "running", "cancelled", "pct", "stage", "msg", "error")}})
    except Exception as e:                              # noqa: BLE001
        import traceback
        log_fh.write(traceback.format_exc())
        r.fail("整轮异常", str(e)[:200],
               {"server_log": tail[-10:], "console": (browser.console[-6:] if browser else [])})
    finally:
        try:
            if browser:
                browser.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            if proc:
                proc.terminate()
        except Exception:                               # noqa: BLE001
            pass
    return r.items


# ============================== 入口 ==============================
def _prereq_report(opt):
    edge = _edge_path()
    py = _server_python()
    video = Path(opt.video)
    ws = Path(opt.workspace)
    dfms = sorted((ws / "model").glob("*.dfm")) if (ws / "model").is_dir() else []
    out = ws / "merge" / (video.stem + "_merged.mp4")
    det = _weights_file("retinaface_10g.onnx")
    rows = [
        ("Edge", bool(edge), edge or "找不到（装 Edge 或设 EDGE_PATH）"),
        ("后端 python", bool(py), py or "没有带 fastapi/uvicorn 的解释器"),
        ("websockets", _ws_connect is not None, sys.executable),
        ("素材视频", video.is_file(), "%s %s" % (video, "" if video.is_file() else "（缺）")),
        ("工作区", ws.is_dir(), "%s（%d 个 .dfm）" % (ws, len(dfms))),
        ("模型 .dfm", bool(dfms), ", ".join(p.name for p in dfms[:4]) or "无"),
        ("成片（预览用）", out.is_file(), "%s %s" % (out.name, "" if out.is_file() else "（缺 → 预览这项 SKIP，先 --keep 一次）")),
        ("检测权重", det.is_file(), str(det)),
    ]
    print("前置：")
    for name, ok, detail in rows:
        print("  %-14s %-4s %s" % (name, "OK" if ok else "缺", detail))
    cands = _workspace_candidates()
    if len(cands) > 1:
        print("  工作区候选（按「已备好项目 / 有成片 / 有 .dfm」打分）:")
        for c in cands:
            sc, why = _workspace_note(c, video.stem)
            print("    %+3d  %s  ← %s" % (sc, c, why))
    print("")


def main(argv=None):
    ap = argparse.ArgumentParser(description="合成链路验收（自己起后端 + 驱动无头 Edge）")
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="素材视频")
    ap.add_argument("--keep", action="store_true", help="导出跑完不取消（慢）")
    ap.add_argument("--rounds", type=int, default=1, help="连跑几轮（默认 1；稳定性验证用 3）")
    ap.add_argument("--frames", default="600,200", help="时间轴两个帧号，默认 600,200")
    ap.add_argument("--scale", type=float, default=1.0, help="所有超时 ×N（慢机器）")
    ap.add_argument("--workspace", default="", help="后端工作区（默认：已备好本素材的项目/有成片/有 .dfm 的那个）")
    ap.add_argument("--log", default="", help="日志文件（默认 %%TEMP%%\\merge_accept\\<时间>.log）")
    a = ap.parse_args(argv)
    try:
        fr = tuple(int(x) for x in str(a.frames).replace("，", ",").split(",")[:2])
        if len(fr) != 2:
            raise ValueError
        a.frames = fr
    except ValueError:
        print("--frames 要两个整数，例如 --frames 600,200")
        return 2
    a.rounds = max(1, int(a.rounds))
    a.scale = max(0.2, float(a.scale))
    # 工作区要在解析出素材之后再挑（打分要看 merge/<素材名> 有没有备好）
    ws_note = "命令行 --workspace 指定"
    if not str(a.workspace or "").strip():
        a.workspace, ws_note = _workspace(Path(a.video).stem)
        a.workspace = str(a.workspace)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log) if a.log else (LOG_DIR / ("merge_accept-%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("仓库根：%s" % ROOT)
    print("素材：%s" % a.video)
    print("工作区：%s（%s）" % (a.workspace, ws_note))
    print("日志：%s" % log_path)
    print("")
    _prereq_report(a)

    fh = open(log_path, "w", encoding="utf-8")
    rounds = []
    t_all = time.time()
    try:
        for i in range(1, a.rounds + 1):
            print("-" * 72)
            print("第 %d/%d 轮" % (i, a.rounds))
            rounds.append(run_round(i, a, fh))
    finally:
        fh.write("\n总耗时 %.1fs\n" % (time.time() - t_all))
        fh.close()

    print("")
    print("=" * 72)
    fails = [it for items in rounds for it in items if it["status"] == "FAIL"]
    skips = [it for items in rounds for it in items if it["status"] == "SKIP"]
    if a.rounds > 1:
        names = []
        for items in rounds:
            for it in items:
                if it["what"] not in names:
                    names.append(it["what"])
        width = max([len(n) for n in names] + [10]) + 2
        print("项目".ljust(width) + "".join(("R%d" % (i + 1)).ljust(8) for i in range(len(rounds))))
        for n in names:
            cells = ""
            for items in rounds:
                d = {it["what"]: it["status"] for it in items}
                cells += d.get(n, "-").ljust(8)
            print(n.ljust(width) + cells)
        sig = [tuple((it["what"], it["status"]) for it in items) for items in rounds]
        same = all(s == sig[0] for s in sig)
        print("")
        print("一致性：%s" % ("一致（%d 轮逐项完全相同）" % len(rounds) if same else "不一致 —— 见上表"))
    else:
        items = rounds[0] if rounds else []
        print("汇总：PASS %d · FAIL %d · SKIP %d"
              % (len([1 for it in items if it["status"] == "PASS"]), len(fails), len(skips)))
    print("总耗时：%.1fs" % (time.time() - t_all))
    for it in fails:
        print("  FAIL %s：%s" % (it["what"], it["detail"]))
    for it in skips:
        print("  SKIP %s：%s" % (it["what"], it["detail"]))
    print("日志：%s" % log_path)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
