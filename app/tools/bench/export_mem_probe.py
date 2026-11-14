# -*- coding: utf-8 -*-
r"""快速导出（fused）**内存**验收：导出前 / 峰值 / 结束后 5s·30s·60s 的 RSS 与 private。

为什么单独一个工具：fused_export_accept 验的是「出片对不对」，一句内存都没量。
用户报的是「导出完了 RAM 不降」，这条只有**按进程、按时间点**量出来才能定性：
      · 后端进程（serve.py）
      · 预览 worker（preview_worker_*.py，dfm/XSeg 会话常驻在这里）
      · ffmpeg 子进程（解码一路 + 编码一路）
      · 系统口径（used / commit / nonpaged）：用来分辨「真泄漏」和
        「Windows 的已提交-but-未驻留（standby / pagefile）」这两种完全不同的东西

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\export_mem_probe.py
    runtime\server\Scripts\python.exe app\tools\export_mem_probe.py --runs 2 --seconds 8
    runtime\server\Scripts\python.exe app\tools\export_mem_probe.py --full   (用整片 1439 帧)

退出码 0 = 全部断言通过（前后 RSS 差 / 第二次峰值不高于第一次）。
"""
from __future__ import annotations

import argparse
import json
import os
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
BOX = TEMP / "dfl_fused_mem"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "张元英"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
TOL_DELTA_MB = 300.0        # 验收线：导出前后 RSS 差

class _Tee(object):
    """同时写控制台和 UTF-8 报告文件（控制台是 cp936，会吃掉 ·/→ 这类字符）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:                                   # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:                                   # noqa: BLE001
                pass


BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

try:
    import psutil
except Exception as e:                                          # noqa: BLE001
    raise SystemExit("缺 psutil：%s" % e)


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
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def mb(v):
    return round(float(v) / 1048576.0, 1)


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
        except Exception:                                       # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ff(args, what=""):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-500:]))
    return r


def probe(path):
    import cv2
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True)
    txt = r.stderr.decode("utf-8", "ignore")
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"frames": frames, "fps": fps, "w": w, "h": h,
            "audio": ("Audio:" in txt)}


# ============================================================
# 系统口径（把「真泄漏」和「已提交未驻留」分开）
# ============================================================
def _perf_info():
    import ctypes
    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD),
                    ("ThreadCount", wintypes.DWORD)]

    psapi = ctypes.WinDLL("psapi.dll")
    info = PERFORMANCE_INFORMATION()
    info.cb = ctypes.sizeof(info)
    if not psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return {}
    ps = float(info.PageSize or 4096)
    return {
        "commit_gb": round(info.CommitTotal * ps / 2 ** 30, 2),
        "commit_limit_gb": round(info.CommitLimit * ps / 2 ** 30, 2),
        "commit_peak_gb": round(info.CommitPeak * ps / 2 ** 30, 2),
        "nonpaged_gb": round(info.KernelNonpaged * ps / 2 ** 30, 2),
        "paged_gb": round(info.KernelPaged * ps / 2 ** 30, 2),
    }


def sys_snap():
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    d = {"used_gb": round(vm.used / 2 ** 30, 2),
         "avail_gb": round(vm.available / 2 ** 30, 2),
         "total_gb": round(vm.total / 2 ** 30, 2),
         "pagefile_used_gb": round(sm.used / 2 ** 30, 2)}
    try:
        d.update(_perf_info())
    except Exception:                                           # noqa: BLE001
        pass
    return d


# ============================================================
# 按进程量
# ============================================================
def _classify(cmd):
    c = (cmd or "").lower()
    if "preview_worker" in c:
        return "worker"
    if "serve.py" in c:
        return "backend"
    if "ffmpeg" in c:
        return "ffmpeg"
    if "_faceid_cli" in c:
        return "faceid"
    return "other"


def procs_of(root_pid):
    """root + 全部后代，按角色分好类（worker / backend / ffmpeg / 其它）。"""
    out = {}
    try:
        root = psutil.Process(int(root_pid))
    except Exception:                                           # noqa: BLE001
        return out
    cand = [root]
    try:
        cand += root.children(recursive=True)
    except Exception:                                           # noqa: BLE001
        pass
    for p in cand:
        try:
            cmd = " ".join(p.cmdline() or [])
        except Exception:                                       # noqa: BLE001
            cmd = ""
        kind = _classify(cmd)
        if p.pid == int(root_pid):
            kind = "backend"
        try:
            mi = p.memory_info()
            fi = p.memory_full_info()
            out[p.pid] = {"kind": kind, "name": p.name(),
                          "rss": mi.rss, "priv": getattr(mi, "private", 0),
                          "uss": fi.uss, "threads": p.num_threads(),
                          "cmd": cmd[:120]}
        except Exception:                                       # noqa: BLE001
            continue
    return out


def agg(procs):
    d = {}
    for v in procs.values():
        k = v["kind"]
        cur = d.setdefault(k, {"n": 0, "rss": 0, "priv": 0, "uss": 0})
        cur["n"] += 1
        cur["rss"] += v["rss"]
        cur["priv"] += v["priv"]
        cur["uss"] += v["uss"]
    return d


class Sampler(threading.Thread):
    """每 dt 秒拍一次（角色级聚合），并记录各角色的峰值。线程安全。"""

    def __init__(self, root_pid, dt=0.25):
        super().__init__(daemon=True)
        self.root = int(root_pid)
        self.dt = float(dt)
        self.lock = threading.Lock()
        self.keep = []
        self.peak = {}
        self.peak_pid = {}
        self.stop_ev = threading.Event()
        self.t0 = time.time()

    def tick(self, tag=""):
        pr = procs_of(self.root)
        a = agg(pr)
        row = {"t": round(time.time() - self.t0, 2), "tag": tag, "agg": a,
               "n_proc": len(pr), "sys": sys_snap()}
        with self.lock:
            self.keep.append(row)
            for k, v in a.items():
                p = self.peak.setdefault(k, {"rss": 0, "priv": 0, "uss": 0})
                for key in ("rss", "priv", "uss"):
                    if v[key] > p[key]:
                        p[key] = v[key]
                        self.peak_pid[(k, key)] = (row["t"], tag)
        return row

    def run(self):
        while not self.stop_ev.is_set():
            try:
                self.tick("running")
            except Exception:                                   # noqa: BLE001
                pass
            self.stop_ev.wait(self.dt)

    def sample(self, tag=""):
        r = self.tick(tag)
        time.sleep(0.05)
        return r


# ============================================================
# 沙箱 / 后端
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


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_FUSED_VRAM_FREE_MB", "DFL_VRAM_FREE_MB",
              "DFL_FUSED_BUDGET", "DFL_ALLOW_CPU_FALLBACK", "DFL_STAGE_KEEP"):
        env.pop(k, None)
    logf = open(str(SERVER_LOG), "ab")
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
            except Exception:                                   # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                               # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % SERVER_LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % SERVER_LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端%s：%s（pid=%s）" % (tag, BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                       # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/stage/status 不应答，见 %s" % SERVER_LOG)


def _child_pids(pid, depth=0):
    if depth > 4:
        return []
    out = []
    try:
        for c in psutil.Process(int(pid)).children(recursive=False):
            out.append(c.pid)
            out += _child_pids(c.pid, depth + 1)
    except Exception:                                           # noqa: BLE001
        pass
    return out


def stop_server(proc):
    if proc is None:
        return
    for pid in [proc.pid] + _child_pids(proc.pid):
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))],
                       capture_output=True, timeout=30)
    time.sleep(1.2)


# ============================================================
# 导出
# ============================================================
def start_export(body, tag=""):
    st, d = post("/api/merge/export", body)
    if st != 200 or d.get("ok") is False:
        raise SystemExit("[%s] 导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
    return d


def wait_export(timeout=3600.0):
    t0 = time.time()
    seen = False
    last = {}
    same = 0
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=60)
        cur = d or {}
        terminal = (cur.get("phase") in ("done", "cancelled", "failed")
                    and not cur.get("running"))
        if cur.get("running"):
            seen = True
        same = (same + 1) if (terminal and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if terminal and ((seen and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.3)
    return last


def log_of(out_video):
    p = Path(str(out_video)).with_suffix(".log")
    try:
        return p.read_text("utf-8", "ignore") if p.is_file() else ""
    except OSError:
        return ""


def pick(log, tag):
    return [ln for ln in (log or "").splitlines() if tag in ln]


def row_line(tag, s, kinds=("backend", "worker", "ffmpeg")):
    parts = []
    for k in kinds:
        v = (s.get("agg") or {}).get(k)
        if v:
            parts.append("%s n=%d rss=%s priv=%s uss=%s"
                         % (k, v["n"], mb(v["rss"]), mb(v["priv"]), mb(v["uss"])))
        else:
            parts.append("%s -" % k)
    d = s.get("sys") or {}
    return ("t=%6.1fs %-14s | %s | sys used=%sG avail=%sG commit=%sG nonpaged=%sG"
            % (s["t"], tag, " | ".join(parts), d.get("used_gb"), d.get("avail_gb"),
               d.get("commit_gb"), d.get("nonpaged_gb")))


# ============================================================
# 一次导出：前 / 峰值 / 后 5·30·60
# ============================================================
def run_one(idx, clip, root_pid, post_waits):
    out = OUT / ("mem_%d.mp4" % idx)
    if out.exists():
        out.unlink()
    smp = Sampler(root_pid, 0.25)
    before = smp.sample("before#%d" % idx)
    print(row_line("导出前", before), flush=True)
    t0 = time.time()
    smp.start()
    d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(out)}, "mem%d" % idx)
    st = wait_export()
    wall = time.time() - t0
    smp.stop_ev.set()
    time.sleep(0.4)
    after_rows = []
    prev = 0.0
    for w in post_waits:
        time.sleep(max(0.0, float(w) - prev))
        prev = float(w)
        after_rows.append((w, smp.sample("after+%ds" % int(w))))
    log = log_of(out)
    return {"idx": idx, "before": before, "after": after_rows,
            "peak": dict(smp.peak), "peak_at": dict(smp.peak_pid),
            "status": st, "wall": round(wall, 2), "log": log, "out": str(out),
            "export": d0, "timeline": list(smp.keep)}


def main(argv=None):
    global DFM
    ap = argparse.ArgumentParser(description="快速导出内存验收（RSS / private / 峰 / 前后差）")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="样片时长（--full 时忽略，用整片）")
    ap.add_argument("--full", action="store_true", help="不做样片，直接压整片（1439 帧）")
    ap.add_argument("--post", default="5,30,60", help="结束后采样点（秒，逗号分隔）")
    ap.add_argument("--model", default=DFM)
    ap.add_argument("--report", default=str(BOX / "report.txt"))
    args = ap.parse_args(argv)
    try:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, open(str(args.report), "w", encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        pass
    DFM = str(args.model)
    post_waits = [float(x) for x in str(args.post).split(",") if x.strip()]

    print("python %s · psutil %s · 目标进程口径：RSS=WorkingSet / private=commit / uss=独占"
          % (sys.version.split()[0], psutil.__version__))
    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    if args.full:
        clip = src
    else:
        clip = MEDIA / "clip.mp4"
        run_ff(["-ss", "3", "-t", str(args.seconds), "-i", src, "-c:v", "libx264",
                "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
                "-movflags", "+faststart", clip], "切样片")
        info = probe(clip)
        say("样片 %s：%s" % (clip.name, info))

    proc = start_server("内存验收")
    try:
        smp = Sampler(proc.pid, 0.25)
        time.sleep(2.0)
        base = smp.sample("boot")
        print(row_line("后端刚起", base), flush=True)

        # 预览一次：把「预览 worker」拉起来（真用户就是这么用的：先拖预览再导出）
        head("预热：一次预览（拉起预览 worker + 装载 dfm）")
        pf = OUT / "preview.jpg"
        st, d = post("/api/merge/preview",
                     {"video": str(clip), "t": 1.0, "model": DFM,
                      "settings": dict(SETTINGS)}, timeout=600)
        say("预览 HTTP %s · ms=%s · worker=%s" % (st, (d or {}).get("ms"),
                                                 (d or {}).get("worker", {}).get("pid")))
        time.sleep(2.0)
        warm = smp.sample("preview-warm")
        print(row_line("预览之后", warm), flush=True)

        runs = []
        for i in range(1, int(args.runs) + 1):
            head("导出 #%d" % i)
            r = run_one(i, clip, proc.pid, post_waits)
            runs.append(r)
            pl = pick(r["log"], "流水线跑完")
            say("· %s" % (pl[0] if pl else "(日志里没有『流水线跑完』)"))
            for tag in ("各段占用", "profile（每帧 ms）", "检测池", "内存（", "内存："):
                ln = pick(r["log"], tag)
                if ln:
                    say("· " + ln[0])
            for w, s in r["after"]:
                print(row_line("结束+%ds" % int(w), s), flush=True)
            print("")
            print("  RSS 时间表（每 1s 一行；MB）")
            print("  %6s %9s %9s %9s %9s %8s %8s"
                  % ("t(s)", "后端WS", "后端提交", "workerWS", "worker提交", "ffmWS", "系统used"))
            tl = r["timeline"]
            last_t = -999.0
            for s in tl:
                if s["t"] - last_t < 1.0:
                    continue
                last_t = s["t"]
                a = s["agg"]
                print("  %6.1f %9s %9s %9s %9s %8s %8s"
                      % (s["t"],
                         mb(a.get("backend", {}).get("rss", 0)),
                         mb(a.get("backend", {}).get("priv", 0)),
                         mb(a.get("worker", {}).get("rss", 0)),
                         mb(a.get("worker", {}).get("priv", 0)),
                         mb(a.get("ffmpeg", {}).get("rss", 0)),
                         (s.get("sys") or {}).get("used_gb")))
            p = r["peak"]
            say("· 峰值 backend rss=%s priv=%s uss=%s / worker rss=%s priv=%s uss=%s / "
                "ffmpeg rss=%s"
                % (mb(p.get("backend", {}).get("rss", 0)),
                   mb(p.get("backend", {}).get("priv", 0)),
                   mb(p.get("backend", {}).get("uss", 0)),
                   mb(p.get("worker", {}).get("rss", 0)),
                   mb(p.get("worker", {}).get("priv", 0)),
                   mb(p.get("worker", {}).get("uss", 0)),
                   mb(p.get("ffmpeg", {}).get("rss", 0))))
    finally:
        stop_server(proc)

    # ---- 汇总 ----
    head("汇总：RSS 时间表（MB）")
    hdr = ("%-10s %8s %8s %8s %8s %8s" % ("时点", "后端RSS", "后端priv", "workerRSS",
                                         "workerpriv", "ffmpegRSS"))
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        b = r["before"]["agg"]
        print("%-10s %8s %8s %8s %8s %8s"
              % ("#%d 前" % r["idx"], mb(b.get("backend", {}).get("rss", 0)),
                 mb(b.get("backend", {}).get("priv", 0)),
                 mb(b.get("worker", {}).get("rss", 0)),
                 mb(b.get("worker", {}).get("priv", 0)),
                 mb(b.get("ffmpeg", {}).get("rss", 0))))
        p = r["peak"]
        print("%-10s %8s %8s %8s %8s %8s"
              % ("#%d 峰" % r["idx"], mb(p.get("backend", {}).get("rss", 0)),
                 mb(p.get("backend", {}).get("priv", 0)),
                 mb(p.get("worker", {}).get("rss", 0)),
                 mb(p.get("worker", {}).get("priv", 0)),
                 mb(p.get("ffmpeg", {}).get("rss", 0))))
        for w, s in r["after"]:
            a = s["agg"]
            print("%-10s %8s %8s %8s %8s %8s"
                  % ("#%d +%ds" % (r["idx"], int(w)),
                     mb(a.get("backend", {}).get("rss", 0)),
                     mb(a.get("backend", {}).get("priv", 0)),
                     mb(a.get("worker", {}).get("rss", 0)),
                     mb(a.get("worker", {}).get("priv", 0)),
                     mb(a.get("ffmpeg", {}).get("rss", 0))))
    print("")

    # ---- 断言 ----
    if runs:
        r = runs[-1]
        b = r["before"]["agg"]
        last = r["after"][-1][1]["agg"] if r["after"] else b
        d_rss = mb(last.get("backend", {}).get("rss", 0)) - mb(b.get("backend", {}).get("rss", 0))
        d_wrk = mb(last.get("worker", {}).get("rss", 0)) - mb(b.get("worker", {}).get("rss", 0))
        d_all = (mb(last.get("backend", {}).get("rss", 0))
                 + mb(last.get("worker", {}).get("rss", 0))
                 + mb(last.get("ffmpeg", {}).get("rss", 0))
                 - mb(b.get("backend", {}).get("rss", 0))
                 - mb(b.get("worker", {}).get("rss", 0))
                 - mb(b.get("ffmpeg", {}).get("rss", 0)))
        say("导出前后 RSS 差：后端 %+.1f MB · 预览 worker %+.1f MB · 合计 %+.1f MB"
            % (d_rss, d_wrk, d_all))
        check(abs(d_all) <= TOL_DELTA_MB, "导出前后 RSS 差 ≤ %.0f MB" % TOL_DELTA_MB,
              "实际 %+.1f MB（后端 %+.1f / worker %+.1f）" % (d_all, d_rss, d_wrk))
        if len(runs) >= 2:
            p1 = runs[0]["peak"]
            p2 = runs[1]["peak"]
            s1 = sum(p1.get(k, {}).get("rss", 0) for k in ("backend", "worker", "ffmpeg"))
            s2 = sum(p2.get(k, {}).get("rss", 0) for k in ("backend", "worker", "ffmpeg"))
            check(mb(s2) <= mb(s1) + 1.0, "第二次峰值不高于第一次",
                  "峰值 RSS #1=%.1f MB · #2=%.1f MB" % (mb(s1), mb(s2)))
        for r in runs:
            lg = r["log"]
            say("· #%d 帧目录收尾：%s" % (r["idx"],
                                        (pick(lg, "临时帧目录") or ["-"])[-1][:90]))

    fails = [x for x in RESULTS if not x[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d" % (len(RESULTS) - len(fails), len(fails)))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
