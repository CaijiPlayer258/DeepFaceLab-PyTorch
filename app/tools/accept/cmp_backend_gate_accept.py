# -*- coding: utf-8 -*-
r"""合成后端 × 融合方式「界面上先互斥」的验收 —— **零负载版**（不导出、不接模型、不碰 GPU）。

★ 为什么是零负载：这一关要证的只有「配置从界面到后端、再从后端回到界面」这一条线的
  每一个转折点，和「这一单跑多快」毫无关系。真跑一次导出会拉起 4.0 子进程 + ffmpeg +
  模型加载，属于用重锤敲螺丝。所以：

  V1  冷启动（记忆 compose_backend=gpu + 融合方式 seamless）：
      #mgCmpBackend.value === 'auto' · localStorage 里也变成 'auto' · console 有那行 warn
  V2  同一页面改 overlay → gpu 档 disabled===false、title 恢复；改回 seamless → 重新禁用并退回 auto
  V3  请求体组装（**不导出**）：调界面自己那套 MergePages.settings() + mgPerfBody()，
      断言默认（自动）时不带 compose_gpu 键；再反向设成 gpu 证明这条线是通的
      （不然「没有这个键」可能只是永远为空）
  V4  后端 refine（**不导出**）：纯函数 knobs.resolve() + knobs.refine_compose()，
      断言 compose_gpu=1 + seamless 会被判回 cpu 并带一条「选了/实际/原因」的警告；
      再验 overlay 不被误伤
  V5  深色主题：抽屉里「已降级」小标 + 来源 chip 的计算色/对比度 + 截图（纯 DOM 渲染）
  S1  静态扫描：导出设置页每个控件 → 请求体键 → 后端消费者，列出「界面有、链路没有」的

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\cmp_backend_gate_accept.py
    ... --tag after     · --no-browser（只跑 V4/S1）
退出码 0 = 全过。

沙箱里**故意不接 workspace\model**：没有 dfm → 页面不会预热模型，这条路全程零 GPU。
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_cmp_gate"
WS = BOX / "ws"
SHOTS = BOX / "shots"
LOG = BOX / "server.log"

MEM_KEY = "dfl-mg-cfg-memory"      # 设置页两个性能下拉的偏好（data-cfg 记忆）
PREFS_KEY = "dfl-merge-prefs"      # 合成器右栏那套偏好（含「模式」= 融合方式）

sys.path.insert(0, str(APP))
RESULTS = []
BASE = ""


# ============================================================
# 小工具
# ============================================================
def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


def head(t):
    print("", flush=True)
    print("=" * 78, flush=True)
    print(t, flush=True)
    print("=" * 78, flush=True)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http(method, path, body=None, timeout=60.0):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except ValueError:
                return r.status, {"raw": raw}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw}


def get(path, timeout=60.0):
    return http("GET", path, timeout=timeout)[1]


# ============================================================
# 沙箱 / 后端（只有 HTTP 服务，没有模型、没有 export）
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, WS / "model", SHOTS):
        d.mkdir(parents=True, exist_ok=True)
    say("沙箱：%s（model 目录空着 = 不预热模型、不碰 GPU）" % BOX)


def start_server():
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    # 这一关验的是「界面/请求体」的口径，进程级 env 会盖住它（knobs: env > body）
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_DFM_DTYPE", "DFL_COMPOSE_GPU",
              "DFL_COMPOSE_GPU_SEAMLESS", "DFL_COMPOSE_GPU_THREADS", "DFL_VRAM_FREE_MB",
              "DFL_ALLOW_CPU_FALLBACK", "DFL_FUSED_BUDGET", "DFL_FUSED_VRAM_FREE_MB"):
        env.pop(k, None)
    logf = open(str(LOG), "ab")
    logf.write(b"\n===== start cmp_backend_gate (light) =====\n")
    logf.flush()
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                            # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                        # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端起不来，见 %s" % LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    while time.time() - t0 < 240:
        try:
            if get("/api/merge/stage/status", timeout=20) is not None:
                say("后端：%s（pid=%d）" % (BASE, proc.pid))
                return proc
        except Exception:                                # noqa: BLE001
            pass
        time.sleep(0.3)
    raise SystemExit("后端不应答，见 %s" % LOG)


def stop_server(proc):
    if proc is None:
        return
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30)
    except Exception:                                    # noqa: BLE001
        pass


# ============================================================
# 颜色（V5）：计算的 rgb + WCAG 对比度
# ============================================================
def _rgba(s):
    m = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(s or ""))]
    if len(m) < 3:
        return (0.0, 0.0, 0.0, 0.0)
    a = m[3] if len(m) > 3 else 1.0
    return (m[0], m[1], m[2], a)


def _over(fg, bg):
    r, g, b, a = fg
    br, bg2, bb, _ = bg
    if a >= 0.999:
        return (r, g, b, 1.0)
    return (r * a + br * (1 - a), g * a + bg2 * (1 - a), b * a + bb * (1 - a), 1.0)


def _lum(c):
    def f(x):
        x = max(0.0, min(1.0, x / 255.0))
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(fg, bg):
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _rgb_text(c):
    return "rgb(%d, %d, %d)" % (round(c[0]), round(c[1]), round(c[2]))


# ============================================================
# 页面探针
# ============================================================
HOOK = r"""
(function(){
  window.__warns = []; window.__errs = [];
  try {
    var w = console.warn;
    console.warn = function(){
      try { window.__warns.push(Array.prototype.map.call(arguments, String).join(' ')); } catch(e){}
      return w.apply(console, arguments);
    };
  } catch(e){}
  try {
    window.addEventListener('error', function(e){
      try { window.__errs.push(String((e && e.message) || e)); } catch(x){}
    });
  } catch(e){}
})();
"""

READ_UI = r"""
(function(){
  var sel = document.getElementById('mgCmpBackend');
  var modeEl = document.getElementById('mgMode');
  var raw = null;
  try { raw = localStorage.getItem('dfl-mg-cfg-memory'); } catch(e) {}
  var mem = {};
  try { mem = JSON.parse(raw || '{}') || {}; } catch(e) { mem = {}; }
  var opt = null;
  if (sel) {
    for (var i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === 'gpu') { opt = sel.options[i]; break; }
    }
  }
  return JSON.stringify({
    mode: modeEl ? String(modeEl.value) : '',
    cmp: sel ? String(sel.value) : '',
    mem_raw: raw, mem: mem,
    gpu_disabled: opt ? !!opt.disabled : null,
    gpu_title: opt ? String(opt.title || '') : '',
    gate: typeof window.mgGateCmpBackend,
    perfBody: typeof window.mgPerfBody,
    warns: (window.__warns || []).slice(-8),
    errs: (window.__errs || []).slice(-4)
  });
})()
"""

# V3：请求体组装（不导出）—— 走界面自己那两个函数，别在验收里另写一份口径
BODY_PROBE = r"""
(function(){
  var s = (window.MergePages && window.MergePages.settings) ? window.MergePages.settings() : {};
  var body = window.mgPerfBody({ video: 'x.mp4', fused: 1 }, s);
  /* 反向对照：把下拉强行设成 gpu（disabled 只挡用户点击，不挡脚本），
     看这条线是不是真通 —— 不然「没有这个键」可能永远为空，断言等于白写。 */
  var sel = document.getElementById('mgCmpBackend');
  var old = sel ? sel.value : '';
  if (sel) sel.value = 'gpu';
  var s2 = window.MergePages.settings();
  var body2 = window.mgPerfBody({ video: 'x.mp4', fused: 1 }, s2);
  if (sel) sel.value = old;
  return JSON.stringify({settings: s, body: body,
                         forced: {settings: s2, body: body2}});
})()
"""


def wait_app(b, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if b.eval("typeof window.go === 'function'"):
                return True
        except Exception:                                # noqa: BLE001
            pass
        time.sleep(0.25)
    return False


def hide_gates(b):
    b.eval("(function(){var w=document.getElementById('welcome'); if(w) w.style.display='none';"
           "var m=document.getElementById('modal'); if(m) m.style.display='none';"
           "var lg=document.getElementById('legalGate'); if(lg) lg.style.display='none';"
           "return true;})()")


def enter_settings(b):
    b.eval("window.go('merge'); true")
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            if b.eval("typeof window.mgStep === 'function'"):
                break
        except Exception:                                # noqa: BLE001
            pass
        time.sleep(0.25)
    b.eval("window.mgStep('settings'); true")
    time.sleep(0.4)
    hide_gates(b)


def seed_storage(b, mem=None, prefs=None):
    """最小 helper：在**目标 origin** 下写 localStorage，再导航让页面按记忆重新载入。

    merge_accept.Browser 没有「设了 localStorage 再 reload」的 API，而 V1 的前提就是
    「记忆里存着 gpu」时的冷启动。做法：先取一次 origin（about:blank 上写不了
    127.0.0.1 的 localStorage），写 key，再 goto。
    """
    b.goto(BASE + "/")
    wait_app(b)
    payload = json.dumps({"mem": mem, "prefs": prefs}, ensure_ascii=False)
    b.eval("(function(){var d=%s;"
           "try{ if (d.mem === null) localStorage.removeItem(%s);"
           "      else localStorage.setItem(%s, JSON.stringify(d.mem)); }catch(e){}"
           "try{ if (d.prefs === null) localStorage.removeItem(%s);"
           "      else localStorage.setItem(%s, JSON.stringify(d.prefs)); }catch(e){}"
           "return true;})()"
           % (payload, json.dumps(MEM_KEY), json.dumps(MEM_KEY),
              json.dumps(PREFS_KEY), json.dumps(PREFS_KEY)))
    b.goto(BASE + "/")
    wait_app(b)


def read_ui(b):
    return b.js(READ_UI) or {}


def shot(b, path, clip=None):
    import base64
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        p = {"format": "png"}
        if clip:
            p["clip"] = clip
            p["captureBeyondViewport"] = True
        data = b.call("Page.captureScreenshot", **p).get("data") or ""
        path.write_bytes(base64.b64decode(data))
        say("截图 %s（%d 字节）" % (path, path.stat().st_size))
        return path
    except Exception as e:                               # noqa: BLE001
        say("截图失败（%s）：%s" % (path.name, e))
        return None


GATE_WARN = "融合方式=seamless，合成后端 gpu 已禁用"
GPU_TITLE_ON = "compose_frame_gpu"


# ============================================================
# V4：后端 refine（纯函数，不导出）
# ============================================================
def v4_backend_refine():
    head("V4 后端 refine：compose_gpu=1 + seamless 仍然判回 cpu（纯函数，不导出）")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    # 走 app.server.backend_merge 这条真包路径：refine_compose 里那句相对导入
    # （from .compose_gpu import seamless_on）用 importlib 单文件加载会静默失败 ——
    # 然后 refine 直接不降级，断言就假绿了。
    from app.server.backend_merge import knobs            # noqa: E402
    kn0 = knobs.resolve({"compose_gpu": 1})
    say("resolve({'compose_gpu':1}) → %s/%s"
        % (kn0.get("compose_backend"), kn0.get("compose_backend_src")))
    check(kn0.get("compose_backend") == "gpu" and kn0.get("compose_backend_src") == "ui",
          "V4a 请求体 compose_gpu=1 → cfg 里是 gpu / ui（键真被消费了）",
          "%s/%s" % (kn0.get("compose_backend"), kn0.get("compose_backend_src")))
    kn = knobs.refine_compose(kn0, "seamless", fused=True)
    w = [x for x in (kn.get("warnings") or []) if len(x) > 2 and x[1] == "compose_backend"]
    say("refine_compose(seamless) → %s · why=%r · warn=%s"
        % (kn.get("compose_backend"), kn.get("compose_why"),
           (w[0][2][:90] + "…") if w else "(无)"))
    check(kn.get("compose_backend") == "cpu", "V4b 退回 cpu",
          "compose_backend=%r" % (kn.get("compose_backend"),))
    check(len(w) == 1 and str(w[0][0]) == "eff-skipped",
          "V4c 有且只有一条 compose_backend 警告（code=eff-skipped）",
          "%s" % (w[0][:2] if w else "(无)",))
    txt = str(w[0][2]) if w else ""
    check(("选了" in txt) and ("实际" in txt) and ("原因" in txt),
          "V4d 警告三件事齐（选了 X / 实际跑 Y / 原因 Z）", txt[:160])
    check(bool(kn.get("compose_why")),
          "V4e compose_why 有值（抽屉「已降级」小标就靠它）",
          "why=%r" % (kn.get("compose_why"),))
    # 不能误伤：overlay 这一单不该被动
    kn2 = knobs.refine_compose(knobs.resolve({"compose_gpu": 1}), "overlay", fused=True)
    check(kn2.get("compose_backend") == "gpu" and not (kn2.get("warnings") or []),
          "V4f overlay + gpu 不被误伤（仍是 gpu、没有警告）",
          "%s/%s" % (kn2.get("compose_backend"), kn2.get("compose_backend_src")))
    # 不导出（分段链）也要退：界面上那颗「快速导出」关掉就是这条
    kn3 = knobs.refine_compose(knobs.resolve({"compose_gpu": 1}), "overlay", fused=False)
    say("fused=False（关掉快速导出）→ %s · why=%r"
        % (kn3.get("compose_backend"), (kn3.get("compose_why") or "")[:60]))
    check(kn3.get("compose_backend") == "cpu",
          "V4g 分段链（fused=False）也退回 cpu —— 这一类**界面还没拦**，见 S1",
          "compose_backend=%r why=%r" % (kn3.get("compose_backend"),
                                         (kn3.get("compose_why") or "")[:60]))
    return kn, w


# ============================================================
# S1：静态扫描 —— 设置页每个控件 → 请求体键 → 后端消费者
# ============================================================
# 这张表是**手写的传送表**（每一行都能在 app.js 里指到行号），不是正则猜的：
# 上一版只认「请求体字面量里出现过的键」，于是把 mgPerfBody / mgMergeSettings 这两条
# 间接通道（compose_gpu / dfm_dtype / settings.*）全判成「哪里都没去」—— 那正是这份
# 报告在批评的同一类错误：只看一眼就下结论。
TRANSPORT = [
    # 控件键             传送位置            请求体里的名字            后端消费者
    ("compose_backend", "mgPerfBody", "compose_gpu", "knobs.resolve → DFL_COMPOSE_GPU"),
    ("dfm_dtype", "mgPerfBody", "dfm_dtype", "knobs.resolve → DFL_DFM_DTYPE（只对 torch）"),
    ("detector", "mgDetectorBody", "detector", "cfg.det_align"),
    ("mode", "mgMergeSettings", "settings.mode", "cfg.config.mode → eff.blend"),
    ("face_type", "mgMergeSettings", "settings.face_type", "cfg.face_type"),
    ("image_format", "mgMergeSettings", "settings.image_format", "cfg.image_format"),
    ("mask_source/mask_mode", "mgMergeSettings", "settings.mask_mode", "cfg.mask_mode"),
    ("color_transfer", "mgMergeSettings", "settings.color_transfer", "cfg.color_transfer_mode"),
    ("output_path", "body 字面量", "output_dir", "cfg.output_dir"),
    ("frame_from/frame_to", "body 字面量", "start_frame/end_frame", "cfg.range"),
    ("keep_cache", "body 字面量", "keep_cache", "pipeline_stages / project"),
    ("encoder", "body 字面量", "encoder", "_merge_runner --encoder（只有项目导出）"),
    ("hwaccel", "body 字面量", "hwaccel", "routes → 4.0 子进程（只有项目导出）"),
    ("num_workers", "body 字面量", "workers", "merge_run --workers（只有项目导出）"),
    ("fast_export", "分支", "fused", "走 fused 还是分段链"),
    ("redetect", "（无）", "", "服务端 grep 无此键：客户端自用"),
]


def s1_scan():
    head("S1 静态扫描：设置页每个控件有没有「消费者」（不导出）")
    js = (APP / "web" / "js" / "app.js").read_text("utf-8", "replace")

    def func_body(name, nxt):
        a = js.find("function " + name)
        bs = js.find("function " + nxt, a + 10) if nxt else a + 14000
        return js[a:bs] if a >= 0 else ""

    fused = func_body("mgFusedExport", "mgFusedStart")
    stage = func_body("mgStageExport", "mgStageStart")
    proj = js[js.find("function mgRunExport"):js.find("window.mgRunExport = mgRunExport")]

    def literal_keys(src):
        keys = set(re.findall(r"^\s{8}([a-z_0-9]+):", src, re.M))
        keys |= set(re.findall(r"body\.([a-z_0-9]+) =", src))
        return keys

    k_fused, k_stage, k_proj = literal_keys(fused), literal_keys(stage), literal_keys(proj)
    calls = {
        "mgPerfBody": [bool(re.search(r"mgPerfBody\(", s)) for s in (fused, stage, proj)],
        "mgDetectorBody": [bool(re.search(r"mgDetectorBody", s)) for s in (fused, stage, proj)],
        "mgMergeSettings": [bool(re.search(r"mgMergeSettings\(", s)) for s in (fused, stage, proj)],
    }
    for _n, _f in calls.items():
        say("%-16s 在 fused/stage/project 里被调用：%s" % (_n, _f))

    def delivered(key, body_key):
        """这一条控件在三条路上有没有把值送到请求体里。"""
        out = []
        for i, src in enumerate((fused, stage, proj)):
            if body_key and re.search(r"body\.%s =" % re.escape(body_key), src):
                out.append(True)
            elif body_key and body_key in (k_fused, k_stage, k_proj)[i]:
                out.append(True)
            elif key in ("compose_backend", "dfm_dtype"):
                out.append(calls["mgPerfBody"][i])
            elif key == "detector":
                out.append(calls["mgDetectorBody"][i])
            elif key in ("mode", "face_type", "image_format", "mask_source/mask_mode",
                         "color_transfer"):
                out.append(calls["mgMergeSettings"][i])
            elif key in ("output_path", "frame_from/frame_to", "fast_export"):
                out.append(True)
            else:
                out.append(bool(body_key) and body_key in (k_fused, k_stage, k_proj)[i])
        return out

    say("")
    print("       %-22s %-7s %-7s %-8s %s"
          % ("控件键", "fused", "stage", "project", "后端消费者"), flush=True)
    print("       " + "-" * 96, flush=True)
    dead_fused = []
    for key, fn, body_key, consumer in TRANSPORT:
        flags = delivered(key, body_key)
        if not flags[0]:
            dead_fused.append(key)
        print("       %-22s %-7s %-7s %-8s %s"
              % (key, *["✔" if f else "✘" for f in flags], consumer), flush=True)
    say("")
    check("num_workers" in dead_fused and "encoder" in dead_fused
          and "hwaccel" in dead_fused and "keep_cache" in dead_fused,
          "S1a 已确认：并行进程 / 编码器 / 解码加速 / 保留 cache 在默认（fused）导出上都不到后端",
          "fused 未送达：%s" % ", ".join(dead_fused))
    check(not ({"compose_backend", "dfm_dtype", "detector"} & set(dead_fused)),
          "S1b 两颗性能旋钮 + 检测器在 fused 上是送达的（「没有键」= 自动，不是断了）",
          "fused 未送达：%s" % (", ".join(dead_fused) or "(无)"))
    for pat, label in ((r"body\.get\(.encoder.\)|--encoder", "encoder"),
                       (r"\"merge_workers\"|body\.get\(.workers.\)", "workers"),
                       (r"keep_cache", "keep_cache"),
                       (r"body\.get\(.hwaccel.\)", "hwaccel"),
                       (r"compose_gpu", "compose_gpu"),
                       (r"dfm_dtype", "dfm_dtype")):
        hits = []
        for f in (REPO / "app" / "server").rglob("*.py"):
            try:
                t = f.read_text("utf-8", "replace")
            except OSError:
                continue
            if re.search(pat, t):
                hits.append(f.name)
        say("   后端「%s」出现在：%s" % (label, ", ".join(sorted(set(hits))[:6]) or "(无)"))
    return dead_fused


# ============================================================
# S2：接线检查（静态）—— 后端发布 compose_why -> 前端读它 -> CSS 画它
# ============================================================
# 这一关防的是「代码在、但接不上」：
#   · dd4fceb 在 app.js 里写了「已降级」小标，可 CSS 里一条规则都没有 -> 标了等于没标；
#   · refine_compose 算了 compose_why，可 eff 里没这个字段 -> 前端拿不到，永远不挂小标。
# 三处少一处，用户看到的还是「cpu · 界面」。
def s2_wire():
    head("S2 接线检查：compose_why -> 抽屉小标 -> CSS（静态）")
    eff_src = (APP / "server" / "backend_merge" / "eff.py").read_text("utf-8", "replace")
    js = (APP / "web" / "js" / "app.js").read_text("utf-8", "replace")
    css = (APP / "web" / "css" / "app.css").read_text("utf-8", "replace")
    check('eff["compose_why"]' in eff_src,
          "S2a 后端把 compose_why 发布进 eff（否则前端永远拿不到原因）",
          "由 _eff_publish 挂到 eff 上")
    check("compose_why" in eff_src and "refine_compose" in eff_src,
          "S2b 这个值来自 knobs.refine_compose（不是另编一份）", "ok")
    check("eff.compose_why" in js,
          "S2c 前端读的是 eff.compose_why（mgpEffPaint 那一处）", "ok")
    check(".mgeff-i .d {" in css,
          "S2d 「已降级」小标有 CSS 规则（以前一条都没有，只继承正文色的斜体）", "ok")
    check(".mgeff-i .s.downgraded" in css,
          "S2e 「降级」chip 有 CSS 规则（否则落回 --ink-3 灰，深色 3.39:1）", "ok")
    check(js.count("mgGateCmpBackend") >= 4,
          "S2f 门控函数有调用点（定义 + window 挂载之外至少两处触发）",
          "出现 %d 次" % js.count("mgGateCmpBackend"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    tag = a.tag

    prepare_sandbox()
    proc = start_server()
    b = None
    try:
        # V4 先跑：纯函数、零负载，而且 V5 要拿它算出来的 compose_why / 警告文案当输入
        kn4, w4 = v4_backend_refine()

        if not a.no_browser:
            from tools.accept.merge_accept import Browser, _edge_path     # noqa: E402
            edge = _edge_path()
            if not edge:
                raise SystemExit("找不到 Edge（装 Edge 或设 EDGE_PATH）")
            b = Browser(edge)
            b.call("Page.addScriptToEvaluateOnNewDocument", source=HOOK)
            b.call("Emulation.setDeviceMetricsOverride", width=1624, height=1046,
                   deviceScaleFactor=1, mobile=False)

        # ------------------------------------------------------------------ V1
        if b is not None:
            head("V1 冷启动：记忆 compose_backend=gpu + 融合方式 seamless")
            seed_storage(b, mem={"compose_backend": "gpu"}, prefs={"mode": "seamless"})
            boot = read_ui(b)
            say("载入后（未导航）：mode=%r cmp=%r mem=%s warns=%s"
                % (boot.get("mode"), boot.get("cmp"), boot.get("mem_raw"),
                   (boot.get("warns") or [])[-2:]))
            check(boot.get("gate") == "function", "V1z 门控函数挂在 window 上",
                  "typeof=%s" % (boot.get("gate"),))
            check(boot.get("mode") == "seamless", "V1a 前提：融合方式 = seamless（记忆回填）",
                  "mode=%r" % (boot.get("mode"),))
            check(boot.get("cmp") == "auto", "V1b 载入后 #mgCmpBackend = auto",
                  "cmp=%r" % (boot.get("cmp"),))
            mem1 = boot.get("mem") or {}
            check(mem1.get("compose_backend") == "auto",
                  "V1c 记忆也改成 auto（localStorage 真写进去了，不是只在内存里改）",
                  "raw=%s" % (boot.get("mem_raw"),))
            v1_warn = [w for w in (boot.get("warns") or []) if GATE_WARN in w]
            check(bool(v1_warn), "V1d console 有那行 warn", " | ".join(v1_warn)[:200] or "(无)")

            enter_settings(b)
            ui = read_ui(b)
            say("进设置页后：mode=%r cmp=%r mem=%s gpu_disabled=%s"
                % (ui.get("mode"), ui.get("cmp"), ui.get("mem_raw"), ui.get("gpu_disabled")))
            check(ui.get("mode") == "seamless" and ui.get("cmp") == "auto",
                  "V1e 进设置页仍是 seamless + auto",
                  "mode=%r cmp=%r" % (ui.get("mode"), ui.get("cmp")))
            check((ui.get("mem") or {}).get("compose_backend") == "auto",
                  "V1f 进设置页后记忆仍是 auto", "raw=%s" % (ui.get("mem_raw"),))
            check(ui.get("gpu_disabled") is True,
                  "V1g seamless 时 gpu 这一档是 disabled（界面选不出来）",
                  "disabled=%s" % (ui.get("gpu_disabled"),))
            shot(b, SHOTS / ("%s-v1-settings-seamless.png" % tag))

            # ------------------------------------------------------------------ V2
            head("V2 同一页面改 overlay（门控要放开）/ 再改回 seamless（要收回）")
            b.eval("(function(){var e=document.getElementById('mgMode');"
                   "e.value='overlay'; e.dispatchEvent(new Event('change',{bubbles:true}));"
                   "return true;})()")
            time.sleep(0.3)
            ov = read_ui(b)
            say("overlay：gpu_disabled=%s title=%r"
                % (ov.get("gpu_disabled"), (ov.get("gpu_title") or "")[:60]))
            check(ov.get("mode") == "overlay", "V2a 融合方式已改成 overlay",
                  "mode=%r" % (ov.get("mode"),))
            check(ov.get("gpu_disabled") is False,
                  "V2b gpu 这一档又能选了（disabled === false）",
                  "disabled=%s" % (ov.get("gpu_disabled"),))
            check(GPU_TITLE_ON in (ov.get("gpu_title") or ""),
                  "V2c title 恢复原文案（compose_frame_gpu…）",
                  "title=%r" % ((ov.get("gpu_title") or "")[:90],))
            b.eval("(function(){var e=document.getElementById('mgCmpBackend');"
                   "e.value='gpu'; e.dispatchEvent(new Event('change',{bubbles:true}));"
                   "return e.value;})()")
            time.sleep(0.2)
            g1 = read_ui(b)
            check(g1.get("cmp") == "gpu", "V2d overlay 下真能选中 gpu（不是「谁也用不了 gpu」）",
                  "cmp=%r" % (g1.get("cmp"),))
            b.eval("(function(){var e=document.getElementById('mgMode');"
                   "e.value='seamless'; e.dispatchEvent(new Event('change',{bubbles:true}));"
                   "return true;})()")
            time.sleep(0.3)
            back = read_ui(b)
            say("改回 seamless：cmp=%r mem=%s gpu_disabled=%s"
                % (back.get("cmp"), back.get("mem_raw"), back.get("gpu_disabled")))
            check(back.get("cmp") == "auto" and back.get("gpu_disabled") is True,
                  "V2e 改回 seamless → gpu 重新禁用且值退回 auto",
                  "cmp=%r disabled=%s" % (back.get("cmp"), back.get("gpu_disabled")))
            check((back.get("mem") or {}).get("compose_backend") == "auto",
                  "V2f 退回时记忆同步成 auto（下次载入不会再冒 gpu）",
                  "raw=%s" % (back.get("mem_raw"),))
            shot(b, SHOTS / ("%s-v2-settings-after-toggle.png" % tag))

            # ------------------------------------------------------------------ V3
            head("V3 请求体组装（不导出）：界面自己的 settings() + mgPerfBody()")
            bp = b.js(BODY_PROBE) or {}
            s0 = bp.get("settings") or {}
            b0 = bp.get("body") or {}
            say("settings.compose_backend=%r · body=%s"
                % (s0.get("compose_backend"), json.dumps(b0, ensure_ascii=False)))
            check(s0.get("compose_backend") == "auto",
                  "V3a seamless 下界面报的是 auto（不是 gpu）",
                  "compose_backend=%r" % (s0.get("compose_backend"),))
            check("compose_gpu" not in b0,
                  "V3b 组装出来的请求体**没有** compose_gpu 键（自动 = 不带键）",
                  "keys=%s" % sorted(b0.keys()))
            f0 = (bp.get("forced") or {}).get("body") or {}
            check(f0.get("compose_gpu") == 1,
                  "V3c 反向对照：强行设 gpu → compose_gpu=1（这条线是通的）",
                  "body=%s" % json.dumps(f0, ensure_ascii=False))
            # dfm_dtype 同一条线（同样的机制，同样的断言）
            b.eval("(function(){var e=document.getElementById('mgDfmDtype');"
                   "if(e){e.value='fp32'; e.dispatchEvent(new Event('change',{bubbles:true}));}"
                   "return true;})()")
            bp2 = b.js(BODY_PROBE) or {}
            b2 = bp2.get("body") or {}
            check(b2.get("dfm_dtype") == "fp32" and "compose_gpu" not in b2,
                  "V3d 另一颗旋钮（推理精度）走同一条线：fp32 带键、compose 仍不带",
                  "keys=%s" % sorted(b2.keys()))
            b.eval("(function(){try{localStorage.removeItem('dfl-mg-cfg-memory');}catch(e){}"
                   "var e=document.getElementById('mgDfmDtype'); if(e) e.value='auto';"
                   "return true;})()")

            # ------------------------------------------------------------------ V5
            head("V5 深色主题：抽屉里「已降级」小标 + 来源 chip（纯 DOM，不导出）")
            # 后端那几个字面值是 V4 真跑出来的 compose_why / 警告文案（不是编的）
            eff = {
                "detector": "tinymog", "detector_src": "ui",
                "mask_source": "XSegLite", "mask_source_src": "inferred",
                "ct_mode": "mt", "ct_mode_src": "ui",
                "blend": "seamless", "blend_src": "ui",
                "erode": 0, "erode_src": "ui", "blur": 4, "blur_src": "ui",
                "models": ["MODEL"], "models_src": "ui",
                "faces_project": "", "faces_project_src": "default",
                "library_enabled": True, "library_enabled_src": "default",
                "fast_export": True, "fast_export_src": "ui",
                "frame_range": {"start": 1, "end": 30}, "frame_range_src": "ui",
                "workers": 8, "workers_src": "default",
                "batch": 8, "batch_src": "default",
                "jpeg_quality": 100, "jpeg_quality_src": "default",
                "dfm_dtype": "auto", "dfm_dtype_src": "default",
                "compose_backend": str(kn4.get("compose_backend") or "cpu"),
                "compose_backend_src": str(kn4.get("compose_backend_src") or "ui"),
                "compose_why": str(kn4.get("compose_why") or ""),
            }
            warns = [{"code": str(x[0]), "field": str(x[1]), "level": "warn", "text": str(x[2])}
                     for x in (w4 or [])]
            payload = json.dumps({"eff": eff, "warns": warns}, ensure_ascii=False)
            dark = b.js(
                "(function(){try{ if (window.setThemeMode) window.setThemeMode('dark'); }catch(e){}"
                "document.documentElement.setAttribute('data-theme','dark');"
                "return JSON.stringify({theme:document.documentElement.getAttribute('data-theme'),"
                "warnVar:getComputedStyle(document.documentElement).getPropertyValue('--warn')||''});})()"
            ) or {}
            say("主题=%s · --warn=%r" % (dark.get("theme"), (dark.get("warnVar") or "").strip()))
            b.eval("window.mgStep('progress'); true")
            time.sleep(0.4)
            b.eval("(function(){var d=%s; window.mgpEffRender(d.eff, d.warns); return true;})()"
                   % payload)
            time.sleep(0.4)
            b.eval("(function(){var o=document.getElementById('mgpEffOpen');"
                   "if(o) o.click(); return true;})()")
            time.sleep(0.6)
            probe = b.js(
                "(function(){var it=document.querySelector('#mgpEffGrid .mgeff-i[data-k=\"compose_backend\"]');"
                "if(!it) return JSON.stringify({found:false});"
                "var d=it.querySelector('em.d'), s=it.querySelector('i.s');"
                "var csd=d?getComputedStyle(d):null, css=s?getComputedStyle(s):null,"
                "  csi=getComputedStyle(it);"
                "var r=function(e){var b=e.getBoundingClientRect();"
                "  return {w:+b.width.toFixed(1),h:+b.height.toFixed(1)};};"
                "return JSON.stringify({found:true,"
                "row:{bg:csi.backgroundColor, cls:it.className},"
                "badge: d?{text:String(d.textContent),color:csd.color,fs:csd.fontSize,"
                "  fw:csd.fontWeight,box:r(d),title:String(d.title||'').slice(0,80)}:null,"
                "chip: s?{text:String(s.textContent),color:css.color,border:css.borderTopColor,"
                "  fs:css.fontSize,cls:s.className}:null,"
                "val:String((it.querySelector('b.v')||{}).textContent||'')});})()") or {}
            say("探测：%s" % json.dumps(probe, ensure_ascii=False)[:600])
            row_bg = _rgba((probe.get("row") or {}).get("bg"))
            bd, cp = probe.get("badge") or {}, probe.get("chip") or {}
            warn_rgb = _rgba("rgb(240, 180, 41)")           # 深色主题的 --warn
            check(bool(probe.get("found")) and "downgraded" in str((probe.get("row") or {}).get("cls")),
                  "V5a 这一行带 .downgraded", "cls=%r" % ((probe.get("row") or {}).get("cls"),))
            check(bd.get("text") == "已降级", "V5b 小标文字 = 「已降级」",
                  "text=%r" % (bd.get("text"),))
            badge_on_bg = _over(_rgba(bd.get("color")), row_bg)
            c_badge = contrast(badge_on_bg, row_bg)
            say("小标：color=%s → 铺在 %s 上 = %s · 对比 %.2f:1 · %sx%s"
                % (bd.get("color"), (probe.get("row") or {}).get("bg"), _rgb_text(badge_on_bg),
                   c_badge, (bd.get("box") or {}).get("w"), (bd.get("box") or {}).get("h")))
            check(c_badge >= 4.5, "V5c 小标对比度 ≥ 4.5:1（深色下可读）",
                  "%s on %s = %.2f:1" % (bd.get("color"), (probe.get("row") or {}).get("bg"),
                                         c_badge))
            check(_rgb_text(badge_on_bg) == _rgb_text(warn_rgb),
                  "V5d 小标用的是 --warn（不是继承正文色）",
                  "实测 %s · 期望 %s" % (_rgb_text(badge_on_bg), _rgb_text(warn_rgb)))
            chip_on_bg = _over(_rgba(cp.get("color")), row_bg)
            c_chip = contrast(chip_on_bg, row_bg)
            say("来源 chip：text=%r color=%s · 对比 %.2f:1 · cls=%r"
                % (cp.get("text"), cp.get("color"), c_chip, cp.get("cls")))
            check(cp.get("text") == "降级", "V5e chip 文字 = 「降级」（不再是「界面」）",
                  "text=%r" % (cp.get("text"),))
            check(c_chip >= 4.5, "V5f chip 对比度 ≥ 4.5:1",
                  "%s on %s = %.2f:1" % (cp.get("color"), (probe.get("row") or {}).get("bg"),
                                         c_chip))
            check(_rgb_text(chip_on_bg) == _rgb_text(warn_rgb),
                  "V5g chip 是黄色（--warn），和「已降级」同一套语义",
                  "实测 %s · 期望 %s" % (_rgb_text(chip_on_bg), _rgb_text(warn_rgb)))
            rect = b.js("(function(){var e=document.getElementById('mgpEffSheet')||"
                        "document.getElementById('mgpEffDrawer');"
                        "if(!e) return ''; var r=e.getBoundingClientRect();"
                        "return JSON.stringify({x:Math.max(0,r.x-8),y:Math.max(0,r.y-8),"
                        "width:Math.min(window.innerWidth,r.width+16),"
                        "height:Math.min(window.innerHeight,r.height+16)});})()")
            if isinstance(rect, str):
                try:
                    rect = json.loads(rect)
                except ValueError:
                    rect = None
            clip5 = rect if isinstance(rect, dict) else None
            if clip5:
                clip5["scale"] = 1
            shot(b, SHOTS / ("%s-v5-drawer-dark.png" % tag), clip5)
            shot(b, SHOTS / ("%s-v5-page-dark.png" % tag))
            (BOX / ("v5-%s.json" % tag)).write_text(
                json.dumps({"probe": probe, "eff": eff, "warns": warns, "dark": dark},
                           ensure_ascii=False, indent=1), "utf-8")
        else:
            say("--no-browser：跳过 V1/V2/V3/V5")

        # ------------------------------------------------------------------ S1
        s1_scan()
        s2_wire()

        (BOX / ("v-%s.json" % tag)).write_text(
            json.dumps({"results": [{"ok": k, "name": n, "detail": d} for k, n, d in RESULTS]},
                       ensure_ascii=False, indent=1), "utf-8")
        return 0
    finally:
        if b is not None:
            try:
                b.close()
            except Exception:                            # noqa: BLE001
                pass
        stop_server(proc)
        fails = [r for r in RESULTS if not r[0]]
        print("", flush=True)
        print("汇总（%s）：PASS %d · FAIL %d" % (tag, len(RESULTS) - len(fails), len(fails)),
              flush=True)
        for ok, name, detail in fails:
            print("  FAIL %s · %s" % (name, detail), flush=True)
        print("截图目录：%s" % SHOTS, flush=True)


if __name__ == "__main__":
    rc = main()
    fails = [r for r in RESULTS if not r[0]]
    raise SystemExit(1 if fails else 0)
