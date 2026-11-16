# -*- coding: utf-8 -*-
r"""四臂基准：A 基线 / B +NVENC / C +GPU 合成 / D 两者（只看 CPU 侧还剩多少空间）。

口径（四臂完全一样，只有「合成在哪儿算」和「用哪个编码器」两处不同）：

    素材   workspace/9月29日 (5).mp4（1080x1920 · HEVC · 30fps），默认取 200-349 帧 = 150 帧
    设置   seamless + mt + mask_mode 6 + mask_blur 4（= 用户这一轮的口径）
    模型   阿証証証証-416（torch bf16w，默认档），检测器不点名（= 后端默认 tinymog）
    出口   raw 管道（默认），x264 crf16 或 h264_nvenc CQ21

四臂：
    A  compose=cpu（8 线程，默认）· encoder=x264（默认）
    B  compose=cpu（默认）        · encoder=nvenc-h264（CQ 21）
    C  compose=gpu（DFL_COMPOSE_GPU=1，seamless 默认跟着上 GPU，4 线程）
    D  compose=gpu + nvenc-h264

编码器只能在**进程启动时**钉（DFL_FUSED_ENCODER 没有请求体入口），合成后端可以逐单切
（请求体的 compose_gpu，走 knobs.apply 写进这一单的 env）。所以按「编码器」分两个后端进程、
每轮交替起：A1 C1 | B1 D1 | A2 C2 | B2 D2 | A3 C3 | B3 D3 —— 组内 A vs C、B vs D 是同一
进程里交替量的，跨组（A vs B）也是交替量的，热漂移对四臂一样。

测量（**无探针**：不往流水线进程里塞任何采样代码，全部在**另一个进程**里做）：
  · 每段 ms / 各段占用：后端自己写的 <成片>.log（_StageTicks，本来就有，不受这里影响）
  · 进程树核数：psutil 每 100ms 采一次 server 进程树的 cpu_times，按进程拆 py / 解码 ffmpeg /
    编码 ffmpeg（Δcpu/Δwall = 核）
  · GPU：NVML（nvml.dll + ctypes，10Hz）——SM 占用 / NVENC 引擎占用 / 显存 / 功耗 / 温度；
    NVML 起不来才退回 nvidia-smi -lms 100
  · 窗口：以 <成片>.log 里的「管道命令」行（t_all 之前最后一行）为起点、「流水线跑完」行
    为终点，只在窗口内统计 —— 会把「加载会话」那段排除掉，和 profile 的 sec 同一个窗口

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\arm_bench.py
    ... --rounds 3 --frames 150 --start-frame 200 --json out.json
    ... --arms A,C --frames 40 --rounds 1        （先验管道，两分钟）

产物：%TEMP%\dfl_arm_bench\{ws,out,log}（工作区是沙箱，model/_engines 是目录联接；
仓库 workspace 只读，不写）。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
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
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_arm_bench"
WS = BOX / "ws"
OUT = BOX / "out"
LOGS = BOX / "log"
SRC_VIDEO = REPO / "workspace" / "9月29日 (5).mp4"
MODEL = "阿証証証証-416"
SETTINGS = {"blend": "seamless", "color_transfer": "mt", "mask_mode": 6,
            "mask_erode": 0, "mask_blur": 4}
# 沙箱工作区要从真工作区拿一份「起手状态」：这两单的会话配比/攒批滞回都读它们
SEED = ("_sess_bench.json", "_pred_batch_pick.json", "_vram_sess_peak.json",
        "settings.json")

ARMS = {
    "A": {"enc": None, "gpu": 0, "what": "基线（CPU 合成 8 线程 + x264）"},
    "B": {"enc": "nvenc-h264", "gpu": 0, "what": "A + NVENC（CQ 21）"},
    "C": {"enc": None, "gpu": 1, "what": "A + GPU 合成（seamless 也上 GPU）"},
    "D": {"enc": "nvenc-h264", "gpu": 1, "what": "B + C"},
}
ENC_ENV = {"nvenc-h264": {"DFL_FUSED_ENCODER": "nvenc-h264"}}

BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CLOCK0 = 0.0


def say(msg=""):
    print(str(msg), flush=True)


def head(t):
    say("")
    say("=" * 96)
    say(t)
    say("=" * 96)


# ============================================================
# HTTP
# ============================================================
def http(method, path, body=None, timeout=1800.0):
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
            return e.code, {"error": blob[:400].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


# ============================================================
# NVML（ctypes，10Hz，几乎零成本；不起子进程）
# ============================================================
class _NvmlUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class Nvml:
    """NVML 薄封装；任何一步失败就 ok=False，采样器自己退回 nvidia-smi。"""

    def __init__(self, index=0):
        self.ok = False
        self.err = ""
        self.h = None
        try:
            last = None
            dll = None
            for cand in ("nvml.dll", r"C:\Windows\System32\nvml.dll"):
                try:
                    dll = ctypes.WinDLL(cand)
                    break
                except OSError as e:                             # noqa: PERF203
                    last = e
            if dll is None:
                raise last or OSError("nvml.dll 找不到")
            self.dll = dll
            dll.nvmlInit_v2()
            h = ctypes.c_void_p()
            fn = getattr(dll, "nvmlDeviceGetHandleByIndex_v2", None) \
                or dll.nvmlDeviceGetHandleByIndex
            rc = fn(ctypes.c_uint(index), ctypes.byref(h))
            if rc != 0:
                raise OSError("nvmlDeviceGetHandleByIndex rc=%d" % rc)
            self.h = h
            self.ok = True
        except Exception as e:                                   # noqa: BLE001
            self.err = "%s: %s" % (e.__class__.__name__, e)

    def sample(self):
        d = self.dll
        out = {}
        try:
            u = _NvmlUtil()
            if d.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0:
                out["gpu"] = float(u.gpu)
                out["gpu_mem_busy"] = float(u.memory)
            e_u, e_p = ctypes.c_uint(), ctypes.c_uint()
            if d.nvmlDeviceGetEncoderUtilization(self.h, ctypes.byref(e_u),
                                                 ctypes.byref(e_p)) == 0:
                out["nvenc"] = float(e_u.value)
            d_u, d_p = ctypes.c_uint(), ctypes.c_uint()
            if d.nvmlDeviceGetDecoderUtilization(self.h, ctypes.byref(d_u),
                                                 ctypes.byref(d_p)) == 0:
                out["nvdec"] = float(d_u.value)
            m = _NvmlMem()
            if d.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0:
                out["vram"] = round(m.used / 1048576.0, 1)
            p = ctypes.c_uint()
            if d.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(p)) == 0:
                out["watt"] = round(p.value / 1000.0, 2)
            t = ctypes.c_uint()
            if d.nvmlDeviceGetTemperature(self.h, ctypes.c_uint(0), ctypes.byref(t)) == 0:
                out["temp"] = float(t.value)
        except Exception as e:                                   # noqa: BLE001
            out["err"] = str(e)[:80]
        return out


# ============================================================
# 采样器（在**本进程**里，不在后端进程里 —— 无探针）
# ============================================================
class Sampler(threading.Thread):
    def __init__(self, server_pid, period_ms=100):
        super().__init__(daemon=True)
        self.pid = int(server_pid)
        self.period = max(30, int(period_ms)) / 1000.0
        self.rows = []
        self.stop_flag = threading.Event()
        self.labels = {}          # pid -> ("py"|"dec"|"enc"|name)
        self._prev = {}           # pid -> cpu cum
        self._prev_t = None
        self._nv = Nvml()
        self._smisub = None

    # ---- 进程树 CPU ----
    def _tree(self):
        import psutil
        try:
            root = psutil.Process(self.pid)
        except Exception:                                        # noqa: BLE001
            return {}, 0, 0
        try:
            procs = [root] + root.children(recursive=True)
        except Exception:                                        # noqa: BLE001
            procs = [root]
        cur = {}
        for q in procs:
            try:
                c = q.cpu_times()
                cur[q.pid] = float(c.user) + float(c.system)
            except Exception:                                    # noqa: BLE001
                continue
            if q.pid not in self.labels:
                lab = "py"
                try:
                    nm = (q.name() or "").lower()
                    cl = " ".join(q.cmdline() or [])
                    if "ffmpeg" in nm:
                        lab = "enc" if "-c:v" in cl else "dec"
                    elif "python" in nm:
                        lab = "py"
                    else:
                        lab = nm or "?"
                except Exception:                                # noqa: BLE001
                    pass
                self.labels[q.pid] = lab
        return cur, len(cur), len(procs)

    def run(self):
        import psutil
        psutil.cpu_percent(None)
        # 和 LogWatch 用同一个零点：窗口（log 标记）和采样点必须能对起来
        self.t0 = CLOCK0 or time.perf_counter()
        while not self.stop_flag.is_set():
            tick = time.perf_counter()
            row = {"t": round(tick - self.t0, 3), "nproc": 0}
            row.update(self._nv.sample() if self._nv.ok else {})
            cur, n, _np = self._tree()
            row["nproc"] = n
            if self._prev_t is not None:
                dt = tick - self._prev_t
                if dt > 0.02:
                    per = {"py": 0.0, "dec": 0.0, "enc": 0.0, "other": 0.0}
                    tot = 0.0
                    for pid, cum in cur.items():
                        p0 = self._prev.get(pid)
                        if p0 is None:
                            continue
                        d = max(0.0, cum - p0) / dt
                        tot += d
                        per[self.labels.get(pid, "other")] = \
                            per.get(self.labels.get(pid, "other"), 0.0) + d
                    row["cores"] = round(tot, 2)
                    for k, v in per.items():
                        row["core_" + k] = round(v, 2)
            self._prev = dict(cur)
            self._prev_t = tick
            row["mach_pct"] = round(float(psutil.cpu_percent(None)), 1)
            self.rows.append(row)
            slp = self.period - (time.perf_counter() - tick)
            if slp > 0:
                time.sleep(slp)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)


class LogWatch(threading.Thread):
    """盯 <成片>.log 的两个标记行，给采样器一个**精确**的流水线窗口。

    起点 = 「快速导出 管道命令」那一行（_fused_run 里 t_all 之前最后一条日志），
    终点 = 「快速导出：流水线跑完」。两行都是 _vid_log 写 + flush 的，延迟 = 一次 write。
    """

    START = "管道命令"
    END = "流水线跑完"

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.t_start = None
        self.t_end = None
        self.stop_flag = threading.Event()
        self._off = 0
        self._buf = ""

    def run(self):
        t0 = CLOCK0
        while not self.stop_flag.is_set():
            try:
                if self.path.is_file():
                    sz = self.path.stat().st_size
                    if sz > self._off:
                        with open(str(self.path), "r", encoding="utf-8",
                                  errors="replace") as fh:
                            fh.seek(self._off)
                            self._buf += fh.read()
                            self._off = sz
                        now = time.perf_counter() - t0
                        if self.t_start is None and self.START in self._buf:
                            self.t_start = now
                        if self.t_end is None and self.END in self._buf:
                            self.t_end = now
                            return
            except OSError:
                pass
            time.sleep(0.02)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)


# ============================================================
# 后端
# ============================================================
def _rm_junction(p):
    try:
        os.rmdir(str(p))          # 目录联接用 rmdir 摘（不能 rmtree，会把真目录删掉）
    except OSError:
        pass


def prepare_sandbox(fresh=True):
    BOX.mkdir(parents=True, exist_ok=True)
    for d in (OUT, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    if fresh and WS.exists():
        for name in ("model", "_engines"):
            _rm_junction(WS / name)
        left = [n for n in ("model", "_engines") if (WS / n).exists()]
        if left:
            # 联接没摘掉就绝不能 rmtree：会把仓库里的真 model/ 删掉（数据无价）
            raise SystemExit("沙箱联接没摘掉（%s），拒绝清理：%s" % ("、".join(left), WS))
        shutil.rmtree(str(WS), ignore_errors=True)
    WS.mkdir(parents=True, exist_ok=True)
    for name in ("model", "_engines"):
        src = REPO / "workspace" / name
        if not src.is_dir():
            continue
        if not (WS / name).exists():
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / name), str(src)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit("建目录联接失败（%s）：%s" % (name, r.stdout))
    for name in SEED:
        src = REPO / "workspace" / name
        if src.is_file():
            shutil.copy2(str(src), str(WS / name))


def _kill_tree(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                       capture_output=True, timeout=60)
    except Exception:                                            # noqa: BLE001
        pass


def start_server(tag, extra_env=None, timeout=240):
    global BASE
    prepare_sandbox(fresh=True)
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK", "DFL_FUSED_BATCH_WAIT_MS",
              "DFL_FUSED_ENCODER", "DFL_COMPOSE_GPU", "DFL_COMPOSE_GPU_THREADS",
              "DFL_FUSED_CMP", "DFL_FUSED_PREP", "DFL_CV2_THREADS", "DFL_DET_EP",
              "DFL_COMPOSE_GPU_SEAMLESS", "DFL_FUSED_PIPE_PRIO", "DFL_FUSED_DECODE"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    logf = open(str(LOGS / ("server_%s.log" % tag)), "wb")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                                    # noqa: BLE001
                pass
            t = line.decode("utf-8", "replace")
            if port["v"] is None and t.startswith("PORT="):
                try:
                    port["v"] = int(t.strip().split("=", 1)[1])
                except Exception:                                # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("后端起不来，见 %s" % (LOGS / ("server_%s.log" % tag)))
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT=")
    BASE = "http://127.0.0.1:%d" % port["v"]
    while time.time() - t0 < timeout:
        st, _d = get("/api/merge/stage/status", timeout=30)
        if st == 200:
            return proc
        time.sleep(0.4)
    raise SystemExit("后端起来了但不应答")


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    time.sleep(1.2)


# ============================================================
# 一次导出
# ============================================================
def wait_export(timeout=1200.0):
    t0 = time.time()
    seen = False
    last = {}
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=120)
        cur = d or {}
        if cur.get("running"):
            seen = True
        last = cur
        if seen and not cur.get("running") and not cur.get("closing"):
            break
        time.sleep(0.3)
    return last


def has_x264_sei(p, head=400000):
    """成片里有没有 x264 自己的 SEI 串。

    <成片>.log 的「管道命令」那一行被 _vid_log 截到 300 字符，-c:v 那一段根本不在里面
    （曾经因此把 B/D 误标成 x264）。x264 会把 "x264 - core …" 写进 SEI，NVENC 不写，
    所以直接看字节 —— 便宜且决定性。
    """
    try:
        with open(str(p), "rb") as fh:
            blob = fh.read(int(head))
    except OSError:
        return None
    return b"x264 - core" in blob


def md5_of(p):
    h = hashlib.md5()
    with open(str(p), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_log_line(log, tag):
    for ln in (log or "").splitlines():
        if tag in ln:
            return ln
    return ""


def _win(rows, t0, t1):
    return [r for r in rows if t0 <= r["t"] <= t1]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def _p95(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[max(0, int(len(vals) * 0.95) - 1)], 2)


def run_round(arm, rnd, tag, frames, start_frame):
    global CLOCK0
    a = ARMS[arm]
    out_file = OUT / ("%s_r%d.mp4" % (tag, rnd))
    for old in OUT.glob("%s_r%d*.mp4" % (tag, rnd)):
        try:
            old.unlink()
        except OSError:
            pass
    body = {"video": str(SRC_VIDEO), "model": MODEL, "settings": dict(SETTINGS),
            "output_dir": str(out_file),
            "start_frame": int(start_frame),
            "end_frame": int(start_frame) + int(frames) - 1}
    if a["gpu"]:
        body["compose_gpu"] = 1
    CLOCK0 = time.perf_counter()
    log_path = OUT / ("%s_r%d_%04d-%04d.log"
                      % (tag, rnd, start_frame, start_frame + frames - 1))
    watch = LogWatch(log_path)
    watch.start()
    smp = Sampler(SERVER_PID)
    smp.start()
    t_post = time.time()
    st, d0 = post("/api/merge/export", body)
    if st != 200 or d0.get("ok") is False:
        smp.stop()
        watch.stop()
        raise RuntimeError("导出没起来：HTTP %s %s" % (st, d0.get("error") or d0))
    fin = wait_export()
    wall = time.time() - t_post
    smp.stop()
    watch.stop()

    vid = fin.get("vid") or {}
    stats = vid.get("stats") or {}
    per_stage = vid.get("per_stage_ms") or {}
    util = vid.get("stage_util_pct") or {}
    pipe = vid.get("pipeline") or {}
    produced = [p for p in OUT.glob("%s_r%d*.mp4" % (tag, rnd))]
    produced = sorted(produced)[-1] if produced else None
    log = ""
    if produced is not None:
        lp = produced.with_suffix(".log")
        if lp.is_file():
            log = lp.read_text("utf-8", "replace")

    rows = smp.rows
    t0 = watch.t_start if watch.t_start is not None else (rows[0]["t"] if rows else 0)
    t1 = watch.t_end if watch.t_end is not None else (rows[-1]["t"] if rows else 0)
    W = _win(rows, t0, t1)
    if not W:
        W = rows
    gpu = _mean([r.get("gpu") for r in W])
    nv = _mean([r.get("nvenc") for r in W])
    vram = max([r.get("vram") or 0 for r in W] or [0])
    rec = {
        "arm": arm, "round": rnd, "tag": tag, "frames": int(frames),
        "enc_env": a["enc"], "compose_gpu": a["gpu"],
        "fps": float(stats.get("fps") or 0.0),
        "sec": round(float(vid.get("done") or 0) / max(1e-6, float(stats.get("fps") or 1)),
                     2) if stats else None,
        "frames_done": int(vid.get("done") or 0),
        "per_stage_ms": {k: round(float(v), 2) for k, v in per_stage.items()},
        "util_pct": {k: float(v) for k, v in util.items()},
        "waits": {k: float(util.get(k) or 0) for k in
                  ("dec_wait", "det_wait", "gpu_wait", "cmp_wait", "enc_wait")},
        "threads": pipe.get("threads"), "compose_backend": pipe.get("compose_backend"),
        "decode_mode": pipe.get("decode_mode"), "frame_budget": vid.get("frame_budget"),
        "gpu_util_mean": gpu, "gpu_util_p95": _p95([r.get("gpu") for r in W]),
        "gpu_idle_lt10": round(100.0 * sum(1 for r in W if (r.get("gpu") or 0) < 10)
                               / max(1, len(W)), 1),
        "nvenc_util_mean": nv,
        "nvdec_util_mean": _mean([r.get("nvdec") for r in W]),
        "vram_peak_mb": round(vram, 1),
        "watt_mean": _mean([r.get("watt") for r in W]),
        "watt_max": max([r.get("watt") or 0 for r in W] or [0]),
        "temp_max": max([r.get("temp") or 0 for r in W] or [0]),
        "cores_tree": _mean([r.get("cores") for r in W]),
        "cores_py": _mean([r.get("core_py") for r in W]),
        "cores_dec_ff": _mean([r.get("core_dec") for r in W]),
        "cores_enc_ff": _mean([r.get("core_enc") for r in W]),
        "mach_pct": _mean([r.get("mach_pct") for r in W]),
        "n_samples": len(W), "win": [round(t0, 2), round(t1, 2)],
        "wall_s": round(wall, 2),
        "out": str(produced) if produced else "",
        "mb": round(produced.stat().st_size / 1048576.0, 2) if produced else None,
        "md5": md5_of(produced) if produced else "",
        "x264_sei": has_x264_sei(produced) if produced else None,
        "enc_line": _pick_log_line(log, "管道命令")[-260:],
        "eff_warn": _pick_log_line(log, "[eff-warn]")[:220],
        "det_line": _pick_log_line(log, "检测器")[:200],
        "stage_line": _pick_log_line(log, "profile（每帧 ms）")[:400],
        "phase": fin.get("phase"),
        "vram_plan_budget": (fin.get("vram_plan") or {}).get("frames"),
    }
    return rec


# ============================================================
# 汇总
# ============================================================
def med(recs, key, sub=None):
    vals = []
    for r in recs:
        v = (r.get(sub) or {}).get(key) if sub else r.get(key)
        if v is not None:
            vals.append(float(v))
    return round(statistics.median(vals), 2) if vals else None


def summarize(recs, arms):
    head("四臂（每臂 3 轮的中位；全程无探针）")
    cols = [("臂", 4, "l"), ("fps", 6, "r"), ("ms/帧", 6, "r"),
            ("decode", 7, "r"), ("det_prep", 8, "r"), ("detect", 7, "r"),
            ("dfm", 6, "r"), ("xseg", 6, "r"), ("compose", 8, "r"),
            ("write", 6, "r"), ("GPU%", 6, "r"), ("核", 6, "r"),
            ("py", 5, "r"), ("dec_ff", 7, "r"), ("enc_ff", 7, "r"),
            ("显存MB", 7, "r")]
    say("  ".join((h.ljust(w) if a == "l" else h.rjust(w)) for h, w, a in cols))
    say("-" * 130)
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            continue
        fps = med(rs, "fps")
        row = [arm, "%5.2f" % (fps or 0), "%5.1f" % (1000.0 / fps if fps else 0),
               "%6.1f" % (med(rs, "decode", "per_stage_ms") or 0),
               "%7.1f" % (med(rs, "det_prep", "per_stage_ms") or 0),
               "%6.1f" % (med(rs, "detect", "per_stage_ms") or 0),
               "%5.1f" % (med(rs, "dfm", "per_stage_ms") or 0),
               "%5.1f" % (med(rs, "xseg", "per_stage_ms") or 0),
               "%7.1f" % (med(rs, "compose", "per_stage_ms") or 0),
               "%5.1f" % (med(rs, "write", "per_stage_ms") or 0),
               "%5.1f" % (med(rs, "gpu_util_mean") or 0),
               "%5.2f" % (med(rs, "cores_tree") or 0),
               "%4.2f" % (med(rs, "cores_py") or 0),
               "%6.2f" % (med(rs, "cores_dec_ff") or 0),
               "%6.2f" % (med(rs, "cores_enc_ff") or 0),
               "%6.0f" % (med(rs, "vram_peak_mb") or 0)]
        say("  ".join(str(c).ljust(w) if a == "l" else str(c).rjust(w)
                      for (h, w, a), c in zip(cols, row)))
    say("")
    say("占用%（各段自己报告，等待/背压也在里面）")
    say("%-4s %8s %8s %8s %8s %8s %8s %8s %8s %8s" %
        ("臂", "decode", "det_prep", "detect", "dfm", "xseg", "compose", "write",
         "dec_wait", "enc_wait"))
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            continue
        say("%-4s %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f" %
            (arm,
             med(rs, "decode", "util_pct") or 0, med(rs, "det_prep", "util_pct") or 0,
             med(rs, "detect", "util_pct") or 0, med(rs, "dfm", "util_pct") or 0,
             med(rs, "xseg", "util_pct") or 0, med(rs, "compose", "util_pct") or 0,
             med(rs, "write", "util_pct") or 0, med(rs, "dec_wait", "util_pct") or 0,
             med(rs, "enc_wait", "util_pct") or 0))
    say("")
    say("%-4s %8s %8s %8s %8s %8s %8s %8s" %
        ("臂", "GPUp95", "GPU<10%", "NVENC%", "W均", "W峰", "温", "机%"))
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            continue
        say("%-4s %8.0f %8.0f %8.0f %8.1f %8.1f %8.0f %8.0f" %
            (arm, med(rs, "gpu_util_p95") or 0, med(rs, "gpu_idle_lt10") or 0,
             med(rs, "nvenc_util_mean") or 0, med(rs, "watt_mean") or 0,
             med(rs, "watt_max") or 0, med(rs, "temp_max") or 0,
             med(rs, "mach_pct") or 0))
    say("")
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if rs:
            say("  %s  %s" % (arm, ARMS[arm]["what"]))
            say("     md5: %s" % " | ".join((r["md5"] or "?")[:12] for r in rs))
            say("     编码: %s（按成片 x264 SEI 判：%s）" %
                ("x264" if rs[-1].get("x264_sei") else "nvenc",
                 "有" if rs[-1].get("x264_sei") else "无"))
            say("     合成: %s · 线程 %s · 解码 %s" %
                (rs[-1].get("compose_backend"), (rs[-1].get("threads") or {}).get("compose"),
                 rs[-1].get("decode_mode")))
            _ew = [r["eff_warn"] for r in rs if r.get("eff_warn")]
            if _ew:
                say("     [eff-warn] %s" % _ew[0])


def main(argv=None):
    global SERVER_PID
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--start-frame", type=int, default=200)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--period-ms", type=int, default=100)
    ap.add_argument("--json", default=str(BOX / "arm_ab.json"))
    ap.add_argument("--no-warm", action="store_true")
    a = ap.parse_args(argv)
    arms = [x.strip().upper() for x in str(a.arms).replace(" ", "").split(",") if x.strip()]

    if not SRC_VIDEO.is_file():
        raise SystemExit("素材不在：%s" % SRC_VIDEO)
    prepare_sandbox(fresh=True)
    say("沙箱：%s" % BOX)
    say("素材：%s（%d-%d 帧 · %d 帧）" %
        (SRC_VIDEO.name, a.start_frame, a.start_frame + a.frames - 1, a.frames))

    recs = []
    for rnd in range(1, a.rounds + 1):
        for enc in (None, "nvenc-h264"):
            pair = [x for x in arms
                    if ARMS[x]["enc"] == enc and x in arms]
            if not pair:
                continue
            tag = "x264" if enc is None else enc
            head("第 %d 轮 · 编码器=%s · 臂=%s" % (rnd, tag, "+".join(pair)))
            proc = None
            try:
                proc = start_server("%s_r%d" % (tag, rnd), ENC_ENV.get(enc) if enc else None)
                SERVER_PID = proc.pid
                if not a.no_warm:
                    w = run_round(pair[0], rnd, "WARM_%s" % tag, a.frames, a.start_frame)
                    say("  热身 %s：%.2f 帧/秒 · %s 合成 · GPU %.0f%%" %
                        (pair[0], w["fps"], w.get("compose_backend"), w["gpu_util_mean"] or 0))
                for arm in pair:
                    try:
                        r = run_round(arm, rnd, arm, a.frames, a.start_frame)
                    except Exception as e:                       # noqa: BLE001
                        say("  [FAIL] %s 第 %d 轮：%s: %s" %
                            (arm, rnd, e.__class__.__name__, e))
                        continue
                    recs.append(r)
                    say("  %s 第 %d 轮：%.2f 帧/秒 · 合成=%s · 解码 %s · GPU %.0f%% · "
                        "核 %.2f（py %.2f/dec %.2f/enc %.2f）· %.2f MB · md5 %s" %
                        (arm, rnd, r["fps"], r.get("compose_backend"), r.get("decode_mode"),
                         r["gpu_util_mean"] or 0, r["cores_tree"] or 0,
                         r["cores_py"] or 0, r["cores_dec_ff"] or 0,
                         r["cores_enc_ff"] or 0, r["mb"] or 0, (r["md5"] or "")[:10]))
            finally:
                stop_server(proc)

    Path(a.json).write_text(json.dumps({"meta": vars(a), "records": recs},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    got = [x for x in arms if any(r["arm"] == x for r in recs)]
    if recs:
        summarize(recs, got)
    say("")
    say("[json] %s" % a.json)
    return 0


SERVER_PID = 0

if __name__ == "__main__":
    raise SystemExit(main())
