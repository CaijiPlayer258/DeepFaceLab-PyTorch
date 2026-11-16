# -*- coding: utf-8 -*-
r"""速度基准（ONNX only）：dfm 只走 onnxruntime，不编 TRT 引擎、不降精度。

跑这几组（每组 --reps 次，取中位）：

  fused-full    视频导出 fused，**全片**（384 dfm / tinymog / XSegLite / seamless+mt）
  fused-face    同前，但只跑「有人脸」的 120 帧窗口
  fused-noface  同前，但只跑「无人脸」的 120 帧窗口（解释片段之间 fps 不同）
  det-cmp       同段 120 帧：tinymog vs retinaface_10g（检测器对照）
  proj-2x2      DFL 项目导出（默认 2×2 档）
  staged-120    分段导出（fused=0）同段 120 帧（「可续跑」的代价）
  preview       合成器单帧预览 ms/帧

采样：GPU 利用率 / 显存（nvidia-smi）、本进程树 CPU% 与 RSS 峰值（psutil），0.5s 一格。

跑法（仓库根，必须 server venv）：
    runtime\server\Scripts\python.exe app\tools\speed_bench.py
    runtime\server\Scripts\python.exe app\tools\speed_bench.py --scenes fused-full,det-cmp --reps 3
    runtime\server\Scripts\python.exe app\tools\speed_bench.py --json D:\MyProject\_speed41\speed.json

退出码：0 = 跑完（数字是什么都算跑完）；2 = 环境/接口不对。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
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

BOX = Path(os.environ.get("DFL_SPEED_BOX") or r"D:\MyProject\_speed41")
WS = BOX / "ws"                     # DFL_WORKSPACE（模型 / 输出都在这儿）
OUT = BOX / "out"
LOG = BOX / "log"

DFM = "张元英"                       # dfm 里 in_face = [N, 384, 384, 3]（384 whole_face）
VIDEO = REPO / "workspace" / "9月29日 (5).mp4"
PROJECT = Path(os.environ.get("DFL_SPEED_PROJECT")
               or r"D:\MyProject\DeepFaceLab-PyTorch\workspace\merge\9月29日 (5)")

SETTINGS = {"mask_mode": 8, "mask_source": "XSegLite", "color_transfer": "mt",
            "blend": "seamless", "mask_erode": 0, "mask_blur": 4}
DET_FAST = "tinymog"
DET_HIGH = "retinaface_10g"

BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RESULTS = []


# ============================================================
# 基础设施
# ============================================================
def say(msg=""):
    print(str(msg), flush=True)


def head(t):
    say("")
    say("=" * 78)
    say(t)
    say("=" * 78)


def http(method, path, body=None, timeout=1800.0):
    data = None
    hdrs = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=hdrs, method=method)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        blob = e.read()
        try:
            return e.code, json.loads(blob.decode("utf-8") or "{}")
        except Exception:                                   # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(p, **kw):
    return http("GET", p, **kw)


def post(p, body, **kw):
    return http("POST", p, body, **kw)


def kill_tree(pid):
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=30)
    except Exception:                                       # noqa: BLE001
        pass


def start_server():
    global BASE
    LOG.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DFL_WORKSPACE=str(WS), PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1")
    logp = LOG / "server.log"
    fh = open(str(logp), "w", encoding="utf-8")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=env, creationflags=0x08000000)
    port = {}

    def reader():
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if line.startswith("PORT=") and not port:
                try:
                    port["v"] = int(line.strip().split("=")[1])
                except ValueError:
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while not port.get("v") and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端起不来，见 %s" % logp)
        time.sleep(0.2)
    if not port.get("v"):
        raise SystemExit("等不到 PORT=，见 %s" % logp)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端：%s（pid=%s · workspace=%s）" % (BASE, proc.pid, WS))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/merge/stage/status", timeout=10)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端不应答，见 %s" % logp)


# ============================================================
# 采样：GPU 利用率 / 显存 / 进程树 CPU / RSS
# ============================================================
class Sampler(threading.Thread):
    def __init__(self, pid, gap=0.5):
        super().__init__(daemon=True)
        self.pid = int(pid)
        self.gap = gap
        self.stop = threading.Event()
        self.gpu = []
        self.vram = []
        self.cpu = []
        self.rss = []
        self._proc = None
        self._track = {}            # pid → psutil.Process（CPU% 要同一个对象才有意义）
        try:
            import psutil
            self.psutil = psutil
            self._proc = psutil.Process(self.pid)
            self._proc.cpu_percent(None)
            for c in self._proc.children(recursive=True):
                try:
                    c.cpu_percent(None)
                except Exception:                           # noqa: BLE001
                    pass
        except Exception:                                   # noqa: BLE001
            self.psutil = None

    def _gpu_once(self):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=8)
            line = (r.stdout or "").strip().splitlines()[0]
            u, m = [x.strip() for x in line.split(",")[:2]]
            return float(u), float(m)
        except Exception:                                   # noqa: BLE001
            return None, None

    def _proc_once(self):
        """进程树 CPU / RSS。

        cpu_percent(None) 量的是「**同一个 Process 对象**上一次调用到现在」的增量：
        以前每格都重新 children() 建一遍对象，于是每格都是首调 → 恒 0.0。
        这里按 pid 缓存对象，只在 pid 集合变化时增删（新进程首格仍是 0，可接受）。
        """
        if not self.psutil:
            return None, None
        try:
            live = [self._proc] + self._proc.children(recursive=True)
        except Exception:                                   # noqa: BLE001
            live = [self._proc]
        seen = {}
        for p in live:
            try:
                seen[p.pid] = p
            except Exception:                               # noqa: BLE001
                pass
        for pid in list(self._track):
            if pid not in seen:
                self._track.pop(pid, None)
        for pid, p in seen.items():
            if pid not in self._track:
                try:
                    p.cpu_percent(None)
                except Exception:                           # noqa: BLE001
                    pass
                self._track[pid] = p
        cpu = 0.0
        rss = 0
        for p in self._track.values():
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except Exception:                               # noqa: BLE001
                pass
        return cpu, rss / 1048576.0

    def run(self):
        while not self.stop.is_set():
            u, m = self._gpu_once()
            if u is not None:
                self.gpu.append(u)
                self.vram.append(m)
            c, r = self._proc_once()
            if c is not None:
                self.cpu.append(c)
                self.rss.append(r)
            self.stop.wait(self.gap)

    def summary(self):
        def mean(xs):
            return round(statistics.fmean(xs), 1) if xs else None
        ncpu = 1
        try:
            ncpu = max(1, int(self.psutil.cpu_count() or 1))
        except Exception:                                   # noqa: BLE001
            pass
        return {"gpu_util_pct": mean(self.gpu),
                "gpu_util_max": round(max(self.gpu), 1) if self.gpu else None,
                "vram_peak_mb": round(max(self.vram)) if self.vram else None,
                # 归一成「占全机 %」：psutil 的 100% = 一个核
                "cpu_pct": (round(statistics.fmean(self.cpu) / ncpu, 1) if self.cpu else None),
                "cpu_pct_raw": mean(self.cpu), "ncpu": ncpu,
                "rss_peak_mb": round(max(self.rss)) if self.rss else None,
                "n_gpu_samples": len(self.gpu), "n_cpu_samples": len(self.cpu)}


# ============================================================
# 导出
# ============================================================
def wait_export(timeout=3600.0):
    seen = False
    last = {}
    cur = {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=60)
        cur = d or {}
        if cur.get("running"):
            seen = True
        if cur.get("phase") in ("done", "failed", "cancelled") and not cur.get("running"):
            if seen or not cur.get("active"):
                break
        last = cur
        time.sleep(0.3)
    return cur or last


def read_log(out_path):
    p = Path(str(out_path)).with_suffix(".log")
    if not p.is_file():
        return []
    try:
        return p.read_text("utf-8", "ignore").splitlines()
    except OSError:
        return []


def pick(lines, tag):
    return [ln.strip() for ln in lines if tag in ln]


def run_fused(tag, start=None, end=None, detector=DET_FAST, reps=1):
    rows = []
    for i in range(reps):
        out = OUT / ("%s_%d.mp4" % (tag, i))
        if out.exists():
            out.unlink()
        body = {"video": str(VIDEO), "model": DFM, "settings": dict(SETTINGS),
                "output_dir": str(out), "detector": detector}
        if start is not None and end is not None:
            body["start_frame"] = int(start)
            body["end_frame"] = int(end)
        t0 = time.time()
        st, d = post("/api/merge/export", body)
        if st != 200 or d.get("ok") is False:
            raise SystemExit("[%s] 导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
        sm = Sampler(SERVER_PID["v"])          # 采样后端进程树（serve.py + worker 子进程）
        sm.start()
        fin = wait_export()
        sm.stop.set()
        sm.join(timeout=5)
        wall = round(time.time() - t0, 2)
        vid = fin.get("vid") or {}
        stats = vid.get("stats") or {}
        lines = read_log(out)
        row = {"tag": tag, "rep": i, "detector": detector,
               "phase": fin.get("phase"), "fps": stats.get("fps"),
               "frames": vid.get("done"), "total": vid.get("total"),
               "wall_s": wall, "range": [start, end],
               "per_stage_ms": vid.get("per_stage_ms") or {},
               "stage_util_pct": vid.get("stage_util_pct") or {},
               "frame_budget": vid.get("frame_budget"),
               "log_run": pick(lines, "流水线跑完")[-1:] or [],
               "log_mem": pick(lines, "内存（本进程）")[-1:] or [],
               "log_enc": pick(lines, "编码完成")[-1:] or [],
               "err": (fin.get("error") or "")[:200]}
        row.update(sm.summary())
        rows.append(row)
        say("[%s#%d] phase=%s fps=%s 帧=%s/%s 墙钟=%.1fs GPU=%s%% 显存峰=%sMB CPU=%s%% RSS峰=%sMB"
            % (tag, i, row["phase"], row["fps"], row["frames"], row["total"], wall,
               row["gpu_util_pct"], row["vram_peak_mb"], row["cpu_pct"], row["rss_peak_mb"]))
        if row["phase"] != "done":
            say("   错误：%s" % row["err"])
    return rows


def run_project(tag, workers=None, batch=None, reps=1, limit=120):
    rows = []
    for i in range(reps):
        md = OUT / ("proj_%s_%d" % (tag, i))
        shutil.rmtree(str(md), ignore_errors=True)
        md.mkdir(parents=True, exist_ok=True)
        out = OUT / ("%s_%d.mp4" % (tag, i))
        if out.exists():
            out.unlink()
        body = {"project_dir": str(PROJ_SMALL), "frames_dir": str(PROJ_SMALL),
                "aligned_dir": str(PROJ_SMALL / "aligned"), "merged_dir": str(md),
                "video": str(VIDEO), "out_video": str(out), "model": DFM,
                "settings": dict(SETTINGS), "quality": 92, "force": True}
        if workers is not None:
            body["workers"] = int(workers)
        if batch is not None:
            body["batch"] = int(batch)
        t0 = time.time()
        st, d = post("/api/merge/export-project", body)
        if st != 200 or d.get("ok") is False:
            raise SystemExit("[%s] 项目导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
        sm = Sampler(SERVER_PID["v"])
        sm.start()
        snap = {}
        while time.time() - t0 < 1800:
            _s, snap = get("/api/merge/export-project/status", timeout=60)
            if snap.get("phase") in ("done", "failed", "cancelled"):
                break
            time.sleep(0.5)
        sm.stop.set()
        sm.join(timeout=5)
        wall = round(time.time() - t0, 2)
        row = {"tag": tag, "rep": i, "phase": snap.get("phase"),
               "fps": snap.get("frame_fps"), "done": snap.get("done"),
               "wall_s": wall, "workers": snap.get("workers"), "batch": snap.get("batch"),
               "swap_p50": snap.get("swap_p50"), "swap_elapsed": snap.get("swap_elapsed"),
               "encode_s": (snap.get("encode") or {}).get("seconds"),
               "log": list(snap.get("log") or [])[-6:],
               "err": (snap.get("error") or "")[:200]}
        row.update(sm.summary())
        rows.append(row)
        say("[%s#%d] phase=%s fps=%s done=%s workers=%s batch=%s swap_p50=%sms 墙钟=%.1fs GPU=%s%% 显存峰=%sMB"
            % (tag, i, row["phase"], row["fps"], row["done"], row["workers"],
               row["batch"], row["swap_p50"], wall, row["gpu_util_pct"], row["vram_peak_mb"]))
    return rows


def run_staged(tag, start, end, reps=1):
    rows = []
    for i in range(reps):
        out = OUT / ("%s_%d.mp4" % (tag, i))
        if out.exists():
            out.unlink()
        body = {"video": str(VIDEO), "model": DFM, "settings": dict(SETTINGS),
                "output_dir": str(out), "detector": DET_FAST,
                "start_frame": int(start), "end_frame": int(end), "fused": False}
        t0 = time.time()
        st, d = post("/api/merge/export", body)
        if st != 200 or d.get("ok") is False:
            raise SystemExit("[%s] 分段导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
        sm = Sampler(SERVER_PID["v"])
        sm.start()
        fin = wait_export(3600.0)
        sm.stop.set()
        sm.join(timeout=5)
        wall = round(time.time() - t0, 2)
        row = {"tag": tag, "rep": i, "phase": fin.get("phase"),
               "fps": fin.get("frame_fps") or fin.get("fps"),
               "done": (fin.get("vid") or {}).get("done"), "wall_s": wall,
               "stages": fin.get("stage"), "log": list(fin.get("log_tail") or [])[-4:],
               "err": (fin.get("error") or "")[:200]}
        row.update(sm.summary())
        rows.append(row)
        say("[%s#%d] phase=%s fps=%s 墙钟=%.1fs GPU=%s%% 显存峰=%sMB"
            % (tag, i, row["phase"], row["fps"], wall, row["gpu_util_pct"], row["vram_peak_mb"]))
    return rows


def run_preview(tag, n=12, reps=1):
    """合成器单帧预览：POST /api/merge/preview（video + t）。"""
    rows = []
    for i in range(reps):
        ms = []
        detail = []
        for k in range(n):
            t = round((40 + k * 7) / 30.0, 3)               # 第 40/47/... 帧
            t0 = time.time()
            st, d = post("/api/merge/preview",
                         {"video": str(VIDEO), "t": t, "model": DFM,
                          "settings": dict(SETTINGS), "detector": DET_FAST},
                         timeout=300)
            dt = (time.time() - t0) * 1000.0
            if st != 200 or d.get("ok") is False:
                say("   preview 失败：HTTP %s %s" % (st, str(d.get("error"))[:120]))
                continue
            if k < 2:
                continue                                    # 前两发是冷启动（载模型/建 worker）
            ms.append(dt)
            tm = d.get("timings") or {}
            detail.append({k: round(dt, 1), "grab": tm.get("grab_ms"),
                           "detect": tm.get("detect_ms"), "swap": tm.get("swap_ms")})
        med = round(statistics.median(ms), 1) if ms else None
        rows.append({"tag": tag, "rep": i, "ms_median": med, "n": len(ms),
                     "ms_all": [round(x, 1) for x in ms], "detail": detail[-3:]})
        say("[%s#%d] 单帧预览 p50=%.1f ms（%d 帧：%s）"
            % (tag, i, med or -1, len(ms), ", ".join("%.0f" % x for x in ms)))
    return rows


# ============================================================
# 素材准备
# ============================================================
SERVER_PID = {"v": 0}
PROJ_SMALL = BOX / "proj120"


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def prepare_ws():
    WS.mkdir(parents=True, exist_ok=True)
    (WS / "model").mkdir(parents=True, exist_ok=True)
    link_or_copy(REPO / "workspace" / "model" / (DFM + ".dfm"), WS / "model" / (DFM + ".dfm"))
    # 遮罩：XSegLite（和验收脚本同一条路：workspace/model/XSegLite/xseglite.onnx）
    xseg = Path(os.environ.get("DFL_SPEED_XSEG")
                or r"D:\MyProject\DeepFaceLab-PyTorch\workspace\model\XSegLite\xseglite.onnx")
    if xseg.is_file():
        link_or_copy(xseg, WS / "model" / "XSegLite" / "xseglite.onnx")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)


def face_index():
    """每一帧有没有脸：DFL 项目的 aligned/<帧号>_*.jpg 就是答案。"""
    al = PROJECT / "aligned"
    if not al.is_dir():
        return {}, 0
    have = set()
    for p in al.glob("*.jpg"):
        stem = p.stem.split("_")[0]
        if stem.isdigit():
            have.add(int(stem))
    total = 0
    for p in PROJECT.iterdir():
        if p.suffix.lower() in (".jpg", ".png") and p.stem.isdigit():
            total = max(total, int(p.stem))
    return have, total


def window(is_face, width=120):
    """找一个宽度=width、整窗都有脸 / 都没有脸的连续区间（1 基帧号）。"""
    have, total = face_index()
    if not total:
        return None
    run = 0
    for n in range(1, total + 1):
        ok = (n in have) == is_face
        run = run + 1 if ok else 0
        if run >= width:
            return n - width + 1, n
    return None


def build_small_project(start, end):
    """切一个小项目（帧 + aligned），项目导出用 —— 不碰原项目目录。"""
    shutil.rmtree(str(PROJ_SMALL), ignore_errors=True)
    (PROJ_SMALL / "aligned").mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(start, end + 1):
        src = PROJECT / ("%06d.jpg" % i)
        if src.is_file():
            link_or_copy(src, PROJ_SMALL / src.name)
            n += 1
        for a in (PROJECT / "aligned").glob("%06d_*.jpg" % i):
            link_or_copy(a, PROJ_SMALL / "aligned" / a.name)
    return n


# ============================================================
# 汇总
# ============================================================
def med(rows, key):
    xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="速度基准（ONNX only）")
    ap.add_argument("--scenes", default="fused-full,fused-face,fused-noface,det-cmp,proj-2x2,staged-120,preview")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=str(BOX / "speed.json"))
    ap.add_argument("--no-server", action="store_true", help="复用已在跑的后端（BASE 从 env 读）")
    a = ap.parse_args(argv)

    prepare_ws()
    scenes = [s.strip() for s in a.scenes.split(",") if s.strip()]
    reps = max(1, int(a.reps))
    head("速度基准 · dfm=%s（384 · ONNX/onnxruntime）· 素材 %s" % (DFM, VIDEO.name))
    say("工作区 %s · 输出 %s" % (WS, OUT))

    have, total = face_index()
    say("有人脸帧：%d / %d（%.1f%%）· 无人脸 %d 帧"
        % (len(have), total, 100.0 * len(have) / max(1, total), total - len(have)))
    face_win = window(True)
    noface_win = window(False)
    say("有人脸窗口：%s · 无人脸窗口：%s" % (face_win, noface_win))

    proc = None
    if not a.no_server:
        proc = start_server()
        SERVER_PID["v"] = proc.pid
    else:
        global BASE
        BASE = os.environ.get("DFL_SPEED_BASE", "")
        SERVER_PID["v"] = int(os.environ.get("DFL_SPEED_PID", "0"))
        if not BASE or not SERVER_PID["v"]:
            raise SystemExit("--no-server 需要 DFL_SPEED_BASE / DFL_SPEED_PID")

    all_rows = {}
    try:
        if "fused-full" in scenes:
            head("1 · 视频导出 fused · 全片")
            all_rows["fused-full"] = run_fused("fused-full", None, None, DET_FAST, reps)
        if "fused-face" in scenes and face_win:
            head("2 · 视频导出 fused · 有人脸 120 帧（%d-%d）" % face_win)
            all_rows["fused-face"] = run_fused("fused-face", face_win[0], face_win[1], DET_FAST, reps)
        if "fused-noface" in scenes and noface_win:
            head("3 · 视频导出 fused · 无人脸 120 帧（%d-%d）" % noface_win)
            all_rows["fused-noface"] = run_fused("fused-noface", noface_win[0], noface_win[1], DET_FAST, reps)
        if "det-cmp" in scenes and face_win:
            head("4a · 检测器对照 · retinaface_10g（120 帧）")
            all_rows["det-high"] = run_fused("det-high", face_win[0], face_win[1], DET_HIGH, reps)
            head("4b · 检测器对照 · tinymog（同段 120 帧）")
            all_rows["det-fast"] = run_fused("det-fast", face_win[0], face_win[1], DET_FAST, reps)
        if "proj-2x2" in scenes and face_win:
            head("5 · DFL 项目导出（默认 2×2 · 120 帧）")
            n = build_small_project(face_win[0], face_win[1])
            say("小项目：%d 帧 → %s" % (n, PROJ_SMALL))
            all_rows["proj-2x2"] = run_project("proj-2x2", None, None, reps)
        if "staged-120" in scenes and face_win:
            head("6 · 分段导出 fused=0（120 帧）")
            all_rows["staged-120"] = run_staged("staged-120", face_win[0], face_win[1], reps)
        if "preview" in scenes:
            head("7 · 合成器单帧预览")
            all_rows["preview"] = run_preview("preview", 12, reps)
    finally:
        if proc is not None:
            kill_tree(proc.pid)

    # ---- 汇总 ----
    head("汇总（每格 reps=%d · 取中位）" % reps)
    table = []
    for key, rows in all_rows.items():
        if not rows:
            continue
        if key == "preview":
            table.append([key, "%.1f ms/帧" % med(rows, "ms_median"),
                          "-", "-", "-", "%d 次" % rows[0]["n"]])
            continue
        table.append([key,
                      med(rows, "fps"),
                      med(rows, "gpu_util_pct"),
                      med(rows, "vram_peak_mb"),
                      med(rows, "cpu_pct"),
                      med(rows, "rss_peak_mb")])
    say("%-14s %10s %8s %10s %8s %10s" % ("场景", "帧/秒", "GPU%", "显存峰MB", "CPU%", "RSS峰MB"))
    for r in table:
        say("%-14s %10s %8s %10s %8s %10s" % tuple(str(x) for x in r))

    blob = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "dfm": DFM,
            "video": str(VIDEO), "settings": SETTINGS, "reps": reps,
            "faces": {"with_face": len(have), "total": total},
            "windows": {"face": face_win, "noface": noface_win},
            "runs": all_rows, "table": table}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    say("")
    say("JSON：%s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
