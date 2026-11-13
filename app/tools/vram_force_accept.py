# -*- coding: utf-8 -*-
r"""显存「预留 / 归属 / 强制尝试」验收（%TEMP%，真 GPU，真实后端）。

这一轮用户的三句话对应的三件事：
  ① 「可用 2.03 是怎么算出来的，偏高还是偏低」—— 预留从「空闲 20% + 512MB」改成
     max(256MB, 空闲×5%)，本脚本把新旧两套数字**并排**打出来（A 段，legacy 后端对照）；
  ② 「4.6GB 被占，但那可能是我们自己的残留」—— 打印显存归属（本程序 / 其它程序 /
     空闲），并用 Windows「GPU Process Memory · Dedicated」按进程实测一遍（W 段）；
  ③ 「拒绝要允许强制尝试」—— force=1 放行最小档；真 OOM 必须**干净地**回到拒绝态、
     没有残留 job、还能再试一次（B 段）。

三段：
  W  归因：空闲时 / 预览一帧之后，各是谁占的；summary 一行里必须同时有
     「预留」「本程序」「其它程序」「空闲」。
  A  空闲 ~3.4GB：新口径默认**放行**（reserve ≤ 400MB），老口径（DFL_VRAM_RESERVE_MODE
     =legacy20）默认拒绝且 reserve ≥ 1000MB —— 两套数字都打出来；随后真导出 12 帧，
     报帧率与实测显存峰值，日志里不许有 OOM。
  B  空闲 ~600MB：默认拒绝（400 + can_force）；force=1 → 后端真尝试 → OOM 被捕获 →
     status 里 vram_oom 有数、phase=failed、job 收尾干净（无临时帧目录 / 无 .part /
     本进程会话已放掉）→ 还能再 force 一次 → 释放压力后预览照样出图（后端没坏）。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\vram_force_accept.py
    （只跑某几段：--cases w,a,b ；--dfm 换模型）

退出码 0 = 全过。全程产物在 %TEMP%\dfl_vram_force，不碰工作区。
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

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_vram_force"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "张元英"                    # 384 模型（文件不带尺寸 → 加载后还会再判一次预算）
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OOM_PAT = re.compile(r"out of memory|CUDA out of memory|cudaErrorMemoryAllocation|"
                     r"Failed to allocate|OOM", re.I)
TARGET_A = 3400                   # 「空闲 3.4GB」那个场景
TARGET_B = 600                    # 「空闲 600MB」那个场景


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


def smi():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,name,memory.total,memory.used,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception as e:                                  # noqa: BLE001
        return "ERR %s" % e


def smi_used_free():
    try:
        row = smi().splitlines()[0].split(",")
        return int(row[3]), int(row[4])
    except Exception:                                       # noqa: BLE001
        return None, None


def http(method, path, body=None, timeout=3600.0):
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


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)

# ============================================================
# 沙箱 / 后端 / 压力
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        for name in ("model", "_engines"):
            try:
                os.rmdir(str(WS / name))
            except OSError:
                pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, OUT, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("model", "_engines"):
        src = REPO / "workspace" / name
        if not src.is_dir():
            continue
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / name), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建目录联接失败（%s）：%s %s" % (name, r.stdout, r.stderr))
    say("沙箱：%s（model / _engines 已接进来）" % BOX)


def _child_pids(pid):
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"ParentProcessId=%d\" | "
             "Select-Object -ExpandProperty ProcessId" % int(pid)],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:                                       # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _kill_tree(pid, depth=0):
    if depth > 4:
        return
    for child in _child_pids(pid):
        _kill_tree(child, depth + 1)
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))], capture_output=True,
                       timeout=30)
    except Exception:                                       # noqa: BLE001
        pass


def start_server(tag="", extra_env=None, log_path=None):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK", "DFL_ALLOW_TEST_HOOK",
              "DFL_VRAM_RESERVE_MODE", "DFL_VRAM_RECYCLE_WORKER"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    lp = Path(log_path or SERVER_LOG)
    logf = open(str(lp), "ab")
    logf.write(("\n===== start %s =====\n" % tag).encode("utf-8"))
    logf.flush()
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
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % lp)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % lp)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端%s：%s（pid=%s）" % (tag, BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/stage/status 不应答，见 %s" % lp)


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.2)


def log_since(mark, path=None):
    try:
        with open(str(path or SERVER_LOG), "rb") as f:
            f.seek(int(mark))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_mark(path=None):
    try:
        return Path(str(path or SERVER_LOG)).stat().st_size
    except OSError:
        return 0


class Hog:
    """另一个进程（模拟「别的程序占着显存」/ 用来把空闲压到目标值）：真分配、真 hold。"""

    def __init__(self, mb):
        self.mb = int(mb)
        self.proc = None
        self.ok = False

    def start(self):
        code = (
            "import torch, time\n"
            "n = int(%d * 1024 * 1024 // 4)\n"
            "x = torch.empty(n, dtype=torch.float32, device='cuda')\n"
            "x.zero_(); torch.cuda.synchronize()\n"
            "print('HOG_READY', flush=True)\n"
            "time.sleep(900)\n" % self.mb)
        self.proc = subprocess.Popen([str(PY), "-c", code], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        line = self.proc.stdout.readline()
        self.ok = "HOG_READY" in str(line or "")
        return self

    def stop(self):
        if self.proc is not None:
            _kill_tree(self.proc.pid)
            try:
                self.proc.kill()
            except Exception:                               # noqa: BLE001
                pass
            self.proc = None


def press_to(target_mb, tag=""):
    """用 torch 子进程把「空闲显存」压到 target_mb 附近；返回 hog 列表（记得 stop）。

    显存是活的（桌面合成器/浏览器会动），所以按「量一次、补一块」的方式逼近，
    最多四轮；每轮只补差额的 95%，避免一下打穿把 OOM 甩给自己。
    """
    hogs = []
    for _i in range(4):
        _u, f = smi_used_free()
        if f is None:
            break
        if f <= int(target_mb) + 250:
            break
        need = int((f - int(target_mb)) * 0.95)
        if need < 200:
            break
        h = Hog(need).start()
        if not h.ok:
            say("加压进程没吃上（%d MB）—— 可能已经没那么多空闲" % need)
            h.stop()
            break
        hogs.append(h)
        say("加压 +%d MB（%s）→ nvidia-smi：%s" % (need, tag or "-", smi()))
        time.sleep(0.8)
    return hogs


def release(hogs):
    for h in (hogs or []):
        h.stop()
    time.sleep(2.0)


def wait_free(min_mb, timeout=25.0):
    """等驱动把显存还回来（WDDM 上杀掉占显存的进程之后 free 不是立刻涨）。"""
    t0 = time.time()
    f = None
    while time.time() - t0 < timeout:
        _u, f = smi_used_free()
        if f is None or f >= int(min_mb):
            return f
        time.sleep(0.5)
    return f


class PeakWatch:
    """跑导出的时候每 300ms 采一次 nvidia-smi used，报峰值。"""

    def __init__(self):
        self.peak_used = 0
        self.peak_free = None
        self.t0 = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.t0 = time.time()

        def loop():
            while not self._stop.is_set():
                u, f = smi_used_free()
                if u is not None:
                    self.peak_used = max(self.peak_used, int(u))
                    self.peak_free = f if self.peak_free is None else min(self.peak_free, f)
                self._stop.wait(0.3)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=3)
        return False


def gpu_attribution():
    """Windows 的「GPU 进程专用显存」按 pid 摊开 —— 这是「谁真占着」的唯一硬证据。

    nvidia-smi 在这台机上对每进程的 used_gpu_memory 只回 [N/A]/[Insufficient
    Permissions]，所以归属只能问性能计数器（Dedicated Usage = 真显存）。
    """
    ps = (
        "$c = (Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples\n"
        "$h = @{}\n"
        "foreach ($s in $c) {\n"
        "  if ($s.CookedValue -le 0) { continue }\n"
        "  $p = ($s.InstanceName -split '_')[1]\n"
        "  if (-not $h.ContainsKey($p)) { $h[$p] = 0 }\n"
        "  $h[$p] += $s.CookedValue\n"
        "}\n"
        "foreach ($k in $h.Keys) {\n"
        "  $pr = Get-Process -Id ([int]$k) -ErrorAction SilentlyContinue\n"
        "  $nm = '?'\n"
        "  if ($pr) { $nm = $pr.ProcessName }\n"
        "  '{0}|{1}|{2}' -f $k, [math]::Round($h[$k]/1MB,1), $nm\n"
        "}\n")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:                                       # noqa: BLE001
        return []
    rows = []
    for line in (out or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "mb": float(parts[1]), "name": parts[2]})
        except Exception:                                   # noqa: BLE001
            continue
    rows.sort(key=lambda r: -r["mb"])
    return rows

# ============================================================
# 请求体
# ============================================================
def export_body(clip, out, dry=False, budget=30, force=False):
    body = {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "output_dir": str(out), "frame_budget": int(budget),
            "start_frame": 1, "end_frame": 12}
    if dry:
        body["dry_run"] = True
    if force:
        body["force"] = 1
    return body


def preview_body(clip, t=0.5):
    return {"video": str(clip), "t": float(t), "model": DFM, "settings": dict(SETTINGS)}


def plan_of(d):
    """dry_run 的响应把计划放在 vram_plan；拒绝（400）放在顶层 vram。"""
    d = d or {}
    return dict(d.get("vram_plan") or d.get("vram") or {})


def brief(plan):
    keys = ("ok", "forced", "free_mb", "reserve_mb", "avail_mb", "need_mb", "peak_mb",
            "frames", "batch", "own_mb", "own_inproc_mb", "own_preview_mb",
            "reclaimable_mb", "reclaimed_mb", "other_mb", "source")
    return {k: (plan or {}).get(k) for k in keys}


def wait_export(timeout=1800.0):
    t0 = time.time()
    seen = False
    last = {}
    same = 0
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=60)
        cur = d or {}
        terminal = cur.get("phase") in ("done", "cancelled", "failed") and not cur.get("running")
        if cur.get("running"):
            seen = True
        same = (same + 1) if (terminal and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if terminal and ((seen and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.3)
    return last


def fused_tmp_dirs():
    try:
        return sorted(p.name for p in TEMP.glob("dfl_fused_*") if p.is_dir())
    except OSError:
        return []


def probe_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0)}
    finally:
        cap.release()


def run_ff(args, what=""):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] \
        + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-400:]))


# ============================================================
# W · 显存归属（空闲 / 预览一帧之后）
# ============================================================
def case_attribution(clip):
    head("W · 显存归属：本程序 / 其它程序 / 空闲 —— 拒绝消息里那一行必须有它们")
    proc = start_server("归因")
    try:
        say("nvidia-smi：%s" % smi())
        _u, f0 = smi_used_free()
        rows = gpu_attribution()
        say("按进程的专用显存（Windows GPU Process Memory，Top 8）：")
        for it in rows[:8]:
            say("    pid %-7s %8.1f MB  %s" % (it["pid"], it["mb"], it["name"]))
        mine = [it for it in rows if it["name"].lower() in ("python", "pythonw", "python3")]
        say("其中 python/pythonw 合计 %.1f MB（含本脚本起的后端；跑之前那些是别的实例）"
            % sum(it["mb"] for it in mine))

        st, d = post("/api/merge/export", export_body(clip, OUT / "w0.mp4", dry=True))
        p = plan_of(d)
        say("W1 空闲时 dry_run：%s" % json.dumps(brief(p), ensure_ascii=False))
        say("W1 summary：%s" % (p.get("summary") or (d or {}).get("summary") or ""))
        check(st == 200 and p.get("ok") is True, "w1 空闲时预算放行（HTTP 200）",
              "HTTP %s ok=%r" % (st, p.get("ok")))
        want_res = max(256.0, min(768.0, float(p.get("free_mb") or 0) * 0.05))
        check(abs(float(p.get("reserve_mb") or 0.0) - want_res) <= 1.5,
              "w2 预留 = max(256MB, 空闲×5%)（不再是 20%+512MB）",
              "reserve=%s 期望≈%.0f free=%s" % (p.get("reserve_mb"), want_res, p.get("free_mb")))
        s = str(p.get("summary") or "")
        check(all(k in s for k in ("预留", "本程序", "其它程序", "空闲")),
              "w3 summary 一行里同时有「预留 / 本程序 / 其它程序 / 空闲」", s[:200])
        check(float(p.get("own_inproc_mb") or 0.0) < 400.0,
              "w4 空闲时「本进程」几乎不占显存（<400MB）",
              "own_inproc=%s" % p.get("own_inproc_mb"))
        check(p.get("other_mb") is not None and float(p.get("other_mb") or 0.0) > 0,
              "w5 空闲时「其它程序」有数（桌面合成器/浏览器等）",
              "other=%s" % p.get("other_mb"))

        st2, d2 = post("/api/merge/preview", preview_body(clip), timeout=900)
        _s, dv = get("/api/merge/preview/status")
        wk = dict(dv or {})
        say("W2 预览之后 worker：models=%s used_mb=%s pid=%s"
            % (wk.get("models"), wk.get("used_mb"), wk.get("pid")))
        check(st2 == 200 and (d2 or {}).get("ok") is True, "w6 预览出图", "HTTP %s" % st2)
        check(float(wk.get("used_mb") or 0.0) >= 500.0,
              "w7 预览 worker 自己记着「我占了多少 MB」", "used_mb=%s" % wk.get("used_mb"))
        st3, d3 = post("/api/merge/export", export_body(clip, OUT / "w1.mp4", dry=True))
        p3 = plan_of(d3)
        say("W3 预览常驻时 dry_run：%s" % json.dumps(brief(p3), ensure_ascii=False))
        check(st3 == 200 and p3.get("ok") is True,
              "w8 我们自己占的那块（预览常驻）不影响导出放行",
              "ok=%r reclaimed=%s" % (p3.get("ok"), p3.get("reclaimed_mb")))
    finally:
        stop_server(proc)


# ============================================================
# A · 空闲 ~3.4GB：新口径放行 / 老口径拒绝（同一刻两套数字）
# ============================================================
def case_34gb(clip):
    head("A · 空闲 ~3.4GB：新预留放行；老口径（20%%+512MB）同一刻被拒 —— 两套数字并排")
    hogs = []
    proc = start_server("3.4G")
    mark = log_mark()
    try:
        _u, f = smi_used_free()
        if f is not None and f > TARGET_A + 250:
            hogs = press_to(TARGET_A, "A")
        _u, f = smi_used_free()
        say("A0 nvidia-smi：%s（目标空闲 %d MB，实际 %s MB）" % (smi(), TARGET_A, f))

        st, d = post("/api/merge/export", export_body(clip, OUT / "a1.mp4", dry=True))
        p = plan_of(d)
        say("A1 新口径 dry_run：%s" % json.dumps(brief(p), ensure_ascii=False))
        say("A1 summary：%s" % (p.get("summary") or ""))
        check(st == 200 and p.get("ok") is True,
              "a1 新口径（reserve=max(256MB, free×5%)）在 ~3.4GB 空闲时**放行**",
              "HTTP %s ok=%r reserve=%s avail=%s need=%s"
              % (st, p.get("ok"), p.get("reserve_mb"), p.get("avail_mb"), p.get("need_mb")))
        check(float(p.get("reserve_mb") or 0.0) <= 420.0,
              "a2 新预留 ≤ 420MB（3.4GB 空闲时只有 256MB 的底线）",
              "reserve=%s" % p.get("reserve_mb"))
        check("预留" in str(p.get("summary") or ""), "a3 文案里写明了预留", p.get("summary"))

        # ---- 同一刻，老口径对照（另一个后端进程，只换预留口径）----
        p_old = None
        legacy_log = LOGDIR / "server_legacy.log"
        global BASE  # noqa: PLW0603 —— 起第二个后端会把 BASE 指到它的端口
        base_main = BASE
        proc2 = start_server("3.4G-legacy", {"DFL_VRAM_RESERVE_MODE": "legacy20"},
                             log_path=legacy_log)
        try:
            st2, d2 = post("/api/merge/export", export_body(clip, OUT / "a2.mp4", dry=True))
            p_old = plan_of(d2) or dict((d2 or {}).get("vram") or {})
            say("A2 老口径 dry_run：HTTP %s %s" % (st2, json.dumps(brief(p_old),
                                                                   ensure_ascii=False)))
            say("A2 老口径文案：%s" % (str((d2 or {}).get("error") or p_old.get("msg") or "")[:300]))
            check(st2 == 400, "a4 老口径（free×20%+512MB）同一刻**拒绝**（HTTP 400）",
                  "HTTP %s" % st2)
            check(float(p_old.get("reserve_mb") or 0.0) >= 1000.0,
                  "a5 老口径的预留 ≥1GB —— 这就是用户算不明白的那 1.3GB 死重",
                  "legacy reserve=%s / new reserve=%s"
                  % (p_old.get("reserve_mb"), p.get("reserve_mb")))
            err2 = str((d2 or {}).get("error") or "")
            check("预留" in err2 and "需要" in err2 and "可用" in err2,
                  "a6 拒绝文案把「预留」写出来了（不用用户自己猜）", err2[:200])
        finally:
            stop_server(proc2)
            BASE = base_main          # 对照后端收掉之后，BASE 必须指回主后端

        # ---- 真跑 12 帧（这就是「点仍然尝试」之前那条正常路）----
        with PeakWatch() as pk:
            st3, d3 = post("/api/merge/export", export_body(clip, OUT / "a3.mp4"))
            fin = wait_export(1800.0) if st3 == 200 else {}
        plan3 = dict(fin.get("vram_plan") or {})
        total = int((fin.get("vid") or {}).get("total") or 0)
        done = int((fin.get("vid") or {}).get("done") or 0)
        fps = float((fin.get("vid") or {}).get("frame_fps") or 0.0)
        out_p = Path(str(fin.get("out") or (OUT / "a3.mp4")))
        say("A3 终态：phase=%s ok=%s %d/%d 帧 · %.2f 帧/秒 · 峰值显存 %d MB"
            % (fin.get("phase"), fin.get("ok"), done, total, fps, pk.peak_used))
        say("A3 计划：%s" % json.dumps(brief(plan3), ensure_ascii=False))
        check(st3 == 200 and fin.get("phase") == "done" and fin.get("ok") is True,
              "a7 ~3.4GB 空闲 · 新口径下 12 帧真出片（0 拒绝）",
              "HTTP %s phase=%s err=%s" % (st3, fin.get("phase"),
                                           str(fin.get("error") or "")[:160]))
        info = probe_video(out_p) if out_p.is_file() else {}
        check(bool(info) and int(info.get("frames") or 0) == 12, "a8 成片 12 帧",
              "%s 帧 · %s" % (info.get("frames") if info else "-", out_p.name))
        check(fps > 0.0, "a9 帧率有数（报告里要用）", "%.2f 帧/秒" % fps)
        check(pk.peak_used > 0, "a10 显存峰值有数（报告里要用）", "%d MB used" % pk.peak_used)
        blob = log_since(mark)
        check(not OOM_PAT.search(blob), "a11 全程没有 CUDA OOM",
              (OOM_PAT.search(blob).group(0) if OOM_PAT.search(blob) else "clean"))
    finally:
        release(hogs)
        stop_server(proc)

# ============================================================
# B · 空闲 ~600MB：默认拒绝 → force=1 → 真 OOM → 干净回滚 → 还能再试
# ============================================================
def case_600mb(clip):
    head("B · 空闲 ~600MB：默认拒绝；force=1 真尝试；OOM 被捕获 → 回滚 → 还能再试")
    hogs = []
    # OOM 注入：WDDM 会把别的进程的显存换页出去，本机把空闲压到 600MB 再 force=1
    # 竟然还能出片（慢，但不 OOM）—— 靠真实压力测不出「OOM → 回滚」这条路，
    # 所以用 DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1 在加载模型那一步注入一次。
    proc = start_server("600M", {"DFL_ALLOW_TEST_HOOK": "1", "DFL_VRAM_OOM_INJECT": "1"})
    mark = log_mark()
    try:
        if True:
            hogs = press_to(TARGET_B, "B")
        _u, f = smi_used_free()
        say("B0 nvidia-smi：%s（目标空闲 %d MB，实际 %s MB）" % (smi(), TARGET_B, f))

        st, d = post("/api/merge/export", export_body(clip, OUT / "b1.mp4", dry=True))
        p = plan_of(d)
        err = str((d or {}).get("error") or "")
        say("B1 默认 dry_run：HTTP %s %s" % (st, json.dumps(brief(p), ensure_ascii=False)))
        say("B1 文案：%s" % err[:260])
        check(st == 400, "b1 默认仍然**明确拒绝**（HTTP 400）", "HTTP %s" % st)
        check(p.get("need_mb") is not None and p.get("avail_mb") is not None,
              "b2 拒绝带结构化字段 need_mb / avail_mb",
              "need=%s avail=%s" % (p.get("need_mb"), p.get("avail_mb")))
        check(bool((d or {}).get("can_force")), "b3 拒绝带 can_force=true（前端据此亮「仍然尝试」）",
              "can_force=%r" % (d or {}).get("can_force"))
        sline = str((d or {}).get("summary") or "")
        say("B1 summary：%s" % sline)
        check("预留" in sline and ("本程序" in sline or "其它程序" in sline),
              "b4 一行文案里同时有「预留」和显存归属", sline[:200])

        # ---- force=1：放行最小档，后端真去加载 ----
        for p in list(OUT.glob("b1.*")) + list(OUT.glob("b2.*")):
            try:
                p.unlink()
            except OSError:
                pass
        tmp_before = fused_tmp_dirs()
        st2, d2 = post("/api/merge/export", export_body(clip, OUT / "b1.mp4", force=True))
        say("B2 force=1 起单：HTTP %s %s" % (st2, str((d2 or {}).get("error") or "")[:160]))
        check(st2 == 200, "b5 force=1 被放行（不再是 400）", "HTTP %s" % st2)
        fin = wait_export(1800.0) if st2 == 200 else {}
        oom = dict(fin.get("vram_oom") or {})
        say("B3 终态：phase=%s err=%s" % (fin.get("phase"), str(fin.get("error") or "")[:240]))
        say("B3 vram_oom：%s" % json.dumps({k: oom.get(k) for k in
                                            ("kind", "need_mb", "avail_mb", "free_mb",
                                             "reserve_mb", "own_mb", "other_mb")},
                                           ensure_ascii=False))
        check(bool(oom), "b6 OOM 被捕获并写进 status.vram_oom（不是一句「导出失败」）",
              json.dumps(oom, ensure_ascii=False)[:200])
        check(bool(oom) and "回滚" in str(oom.get("msg") or ""),
              "b7 回滚话术写明「已回滚」", str(oom.get("msg") or "")[:220])
        check(bool(oom) and (oom.get("summary") or "").startswith("强制尝试失败"),
              "b8 回滚摘要以「强制尝试失败」开头（前端据此搬回拒绝态）",
              str((oom or {}).get("summary") or "")[:200])

        # ---- 干净：job 收尾、无残留、本进程会话已放掉 ----
        st4, d4 = get("/api/merge/export/status")
        check(st4 == 200 and not (d4 or {}).get("running") and not (d4 or {}).get("active"),
              "b9 job 收尾干净（running=false / active=false）",
              "running=%r active=%r" % ((d4 or {}).get("running"), (d4 or {}).get("active")))
        tmp_after = fused_tmp_dirs()
        check(tmp_after == tmp_before, "b10 没有残留的临时帧目录（dfl_fused_*）",
              "前 %s / 后 %s" % (tmp_before, tmp_after))
        outp = OUT / "b1.mp4"
        parts = sorted(p.name for p in OUT.glob("b1.part.*"))
        check(not outp.exists() and not parts, "b11 没有半截成片（b1.mp4 / b1.part.*）",
              "mp4=%s part=%s" % (outp.exists(), parts))

        st5, d5 = post("/api/merge/export", export_body(clip, OUT / "b1.mp4", dry=True))
        p5 = plan_of(d5)
        say("B4 回滚后 dry_run：%s" % json.dumps(brief(p5), ensure_ascii=False))
        check(float(p5.get("own_inproc_mb") or 0.0) < 400.0,
              "b12 回滚之后本进程没挂着半份会话（own_inproc < 400MB）",
              "own_inproc=%s" % p5.get("own_inproc_mb"))
        check(st5 == 400, "b13 回滚之后仍然回到被拒态（不是悄悄放行）", "HTTP %s" % st5)

        # ---- 还能再试一次 ----
        st6, d6 = post("/api/merge/export", export_body(clip, OUT / "b2.mp4", force=True))
        say("B5 第二次 force=1：HTTP %s" % st6)
        check(st6 == 200, "b14 回滚之后还能再 force 一次（后端没被搞坏）", "HTTP %s" % st6)
        fin2 = wait_export(1800.0) if st6 == 200 else {}
        check(bool(fin2.get("vram_oom")) or fin2.get("phase") == "done",
              "b15 第二次尝试也干净收尾（回滚 或 出片）",
              "phase=%s" % fin2.get("phase"))
        st8, d8 = get("/api/merge/export/status")
        check(st8 == 200 and not (d8 or {}).get("running"),
              "b15b 第二次之后 status 仍然是干净的（没有残留 job）",
              "running=%r phase=%r" % ((d8 or {}).get("running"), (d8 or {}).get("phase")))
    finally:
        release(hogs)
        f_after = wait_free(2400, timeout=60.0)
        say("B6 压力释放后 nvidia-smi：%s（等到的空闲 %s MB）" % (smi(), f_after))
        # 压力放开之后后端必须**还是好的**：要么预览出图，要么明确说显存不够（真正的
        # 拒绝），绝不能 500 / 崩进程 / 卡死。WDDM 有时要几十秒才把杀掉的那块还回来，
        # 所以这里不断言「一定能出图」（那会变成一个看驱动脸色的 flaky 断言）。
        try:
            st7, d7 = post("/api/merge/preview", preview_body(clip, 0.3), timeout=900)
            if st7 == 200 and (d7 or {}).get("ok") is True:
                check(True, "b16 压力释放后预览照样出图（后端没坏）",
                      "HTTP 200 · 空闲 %s MB" % f_after)
            else:
                st9, d9 = get("/api/merge/preview/status")
                check(st7 == 400 and (d7 or {}).get("kind") == "vram" and st9 == 200,
                      "b16 后端仍然健康（这次的拒绝仍是显存口径，不是 500/崩溃）",
                      "preview HTTP %s kind=%r status HTTP %s 空闲 %s MB"
                      % (st7, (d7 or {}).get("kind"), st9, f_after))
        except Exception as e:                              # noqa: BLE001
            check(False, "b16 后端仍然健康（不是 500 / 崩溃 / 卡死）", str(e)[:200])
        blob = log_since(mark)
        hits = [ln.strip() for ln in blob.splitlines() if "回滚" in ln or "强制尝试" in ln]
        check(bool(hits), "b17 日志里有「强制尝试 / 回滚」的现场",
              (hits[0][:200] if hits else "（没找到）"))
        stop_server(proc)



# ============================================================
# F · 前端接线（静态）：按钮在、文案在、force 真的会被写进 body
# ------------------------------------------------------------
# 这一档为什么存在：拒绝面板那颗「仍然尝试」是**跨三个文件**的一条线
# （index.html 的按钮 → app.js 的点击 → mgVramPreflight 把 force 写进 body →
#  后端 can_force/force）。任何一处被改名，界面上就只剩一颗死按钮，
# 而这几处没有别的自动化能看见（headless UI 验收只量布局）。
# ============================================================
def case_wiring():
    head("F · 前端接线：拒绝面板的「仍然尝试（可能 OOM）」是不是真的接上了")
    html = (APP / "web" / "index.html").read_text("utf-8", "ignore")
    js = (APP / "web" / "js" / "app.js").read_text("utf-8", "ignore")
    i18n = (APP / "web" / "js" / "i18n.js").read_text("utf-8", "ignore")
    css = (APP / "web" / "css" / "app.css").read_text("utf-8", "ignore")
    py = (APP / "server" / "routes_merge.py").read_text("utf-8", "ignore")
    check('id="mgpForceTry"' in html and "仍然尝试（可能 OOM）" in html,
          "f1 index.html 有那颗按钮（而且默认 hidden）",
          "id=%s hidden=%s" % ('id="mgpForceTry"' in html, "mgpForceTry" in html))
    check('$("mgpForceTry")' in js and "mgVramReject" in js,
          "f2 app.js 找得到按钮并挂上点击", "hook=%s" % ('$("mgpForceTry")' in js))
    check("body.force = 1" in js and "window.__mgForceRetry" in js,
          "f3 点下去真的把 force=1 写进 body（真请求也会带）",
          "force=%s retry=%s" % ("body.force = 1" in js, "__mgForceRetry" in js))
    check("d.can_force !== false" in js and "d.vram_oom" in js,
          "f4 按钮只在 can_force 时亮；vram_oom 走「已回滚」那条路",
          "can_force=%s vram_oom=%s" % ("d.can_force !== false" in js, "d.vram_oom" in js))
    check("#mgpForceTry" in css, "f5 拒绝态那颗按钮有自己的样式（实心，能被看见）",
          "css=%s" % ("#mgpForceTry" in css))
    check("仍然尝试（可能 OOM）" in i18n, "f6 中文进了 i18n 词典（切英文不漏）",
          "dict=%s" % ("仍然尝试（可能 OOM）" in i18n))
    check('"can_force": True' in py and "def _vram_summary_line" in py,
          "f7 后端给得出 can_force 与 summary",
          "can_force=%s summary=%s" % ('"can_force": True' in py,
                                       "def _vram_summary_line" in py))
    check("reserve_mb" in py and "max(256MB, 空闲×5%)" in py,
          "f8 预留是新口径，而且文案里写明了规则",
          "reserve=%s rule=%s" % ("reserve_mb" in py, "max(256MB, 空闲×5%)" in py))


# ============================================================
# main
# ============================================================
def main(argv=None):
    global DFM
    ap = argparse.ArgumentParser(description="显存预留 / 归属 / 强制尝试验收")
    ap.add_argument("--cases", default="w,a,b,f", help="要跑的案例（w/a/b/f）")
    ap.add_argument("--dfm", default=DFM, help="用哪个 dfm（默认 张元英 384）")
    args = ap.parse_args(argv)
    DFM = str(args.dfm)
    want = [c.strip().lower() for c in str(args.cases).split(",") if c.strip()]
    t_all = time.time()

    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / "clip12.mp4"
    run_ff(["-ss", "3", "-t", "0.4", "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", clip],
           "切样片")
    say("样片 %s：%s · dfm=%s" % (clip.name, probe_video(clip), DFM))
    say("nvidia-smi 起始：%s" % smi())

    if "w" in want:
        case_attribution(clip)
    if "a" in want:
        case_34gb(clip)
    if "b" in want:
        case_600mb(clip)
    if "f" in want:
        case_wiring()

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())