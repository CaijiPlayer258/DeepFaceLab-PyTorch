# -*- coding: utf-8 -*-
r"""fused 导出的 GPU/CPU 利用率采样 + 攒批超时扫描（生产路径，不是原型）。

问题：「让 CPU 更多参与推理」是不是误述？真实收益是不是「把 GPU 空转填掉」？
要回答就得知道 GPU 到底空转多少、空转落在哪一段。这里做三件事：

  1. 起真后端（app/serve.py）→ POST /api/merge/export（fused，默认走流水线）→ 轮询终态；
  2. 整段采样 nvidia-smi（默认 100ms）与 CPU（psutil，缺了就退回 process CPU）；
  3. 把 nvidia-smi 的均值/p50/p95/空闲占比，和终态里的 per_stage_ms / stage_util_pct 拼起来。

--wait-ms 可以一次扫多档（0/5/10/20），对应 _FUSED_BATCH_WAIT_MS（DFL_FUSED_BATCH_WAIT_MS）。

    runtime\server\Scripts\python.exe app\tools\bench\fused_gpu_util.py --frames 120
    runtime\server\Scripts\python.exe app\tools\bench\fused_gpu_util.py --wait-ms 0 5 10 20

只测量，不改产品代码；产物写到 %%TEMP%%\dfl_gpu_util\。
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
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
WS = REPO / "workspace"
MODEL = "阿証証証証-416"     # 和 fused_export_accept 一样用模型名（不是路径）
OUTDIR = Path(os.environ.get("TEMP", ".")) / "dfl_gpu_util"
SETTINGS = {"blend": "seamless", "ct_mode": "mt", "mask_mode": 8,
            "mask_erode": 0, "mask_blur": 4}


def ffmpeg_exe():
    """PATH 里没有就用 imageio_ffmpeg 自带的那一份（仓库里就是这么找的）。"""
    import shutil as _sh
    exe = _sh.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                    # noqa: BLE001
        pass
    cand = REPO / "runtime" / "server" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
    return str(cand) if cand.is_file() else "ffmpeg"


def say(*a):
    print(*a, flush=True)


def _http():
    import urllib.request

    def post(path, body, timeout=60):
        req = urllib.request.Request(
            BASE + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8") or "{}")
        except Exception as e:                      # noqa: BLE001
            body = getattr(e, "read", lambda: b"{}")()
            try:
                return getattr(e, "code", 0), json.loads(body.decode("utf-8"))
            except Exception:
                return 0, {"error": str(e)}

    def get(path, timeout=30):
        try:
            with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8") or "{}")
        except Exception as e:                      # noqa: BLE001
            return 0, {"error": str(e)}

    return post, get


def start_server(extra_env=None):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK", "DFL_FUSED_BATCH_WAIT_MS"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            t = line.decode("utf-8", "replace")
            if port["v"] is None and t.startswith("PORT="):
                try:
                    port["v"] = int(t.strip().split("=", 1)[1])
                except Exception:                   # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端起不来")
        time.sleep(0.2)
    BASE = "http://127.0.0.1:%d" % port["v"]
    post, get = _http()
    while time.time() - t0 < 240:
        st, _ = get("/api/merge/stage/status", timeout=10)
        if st == 200:
            return proc
        time.sleep(0.4)
    raise SystemExit("后端不应答")


def stop_server(proc):
    if proc is None:
        return
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   capture_output=True)
    time.sleep(1.0)


# ---------------------------------------------------------------- 采样

class Sampler(threading.Thread):
    def __init__(self, period_ms=100):
        super().__init__(daemon=True)
        self.period = max(20, int(period_ms))
        self.gpu = []          # [(t, util%, mem_mb)]
        self.cpu = []          # [(t, cpu% of whole machine)]
        self.proc = []         # [(t, process cpu%)]
        self.t0 = None
        self.stop_flag = threading.Event()

    def run(self):
        try:
            import psutil
            p = psutil.Process()
            psutil.cpu_percent(None)
            p.cpu_percent(None)
        except Exception:                          # noqa: BLE001
            psutil = None
            p = None
        sub = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", str(self.period)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.t0 = time.time()
        th = threading.Thread(target=self._read_gpu, args=(sub,), daemon=True)
        th.start()
        while not self.stop_flag.is_set():
            t = time.time() - self.t0
            if psutil is not None:
                self.cpu.append((t, psutil.cpu_percent(None)))
                self.proc.append((t, p.cpu_percent(None)))
            time.sleep(self.period / 1000.0)
        try:
            sub.kill()
        except Exception:                          # noqa: BLE001
            pass

    def _read_gpu(self, sub):
        for line in iter(sub.stdout.readline, b""):
            t = time.time() - self.t0
            parts = [x.strip() for x in line.decode("utf-8", "replace").split(",")]
            if len(parts) >= 2:
                try:
                    self.gpu.append((t, float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)

    def summary(self):
        g = [x[1] for x in self.gpu]
        c = [x[1] for x in self.cpu]
        pc = [x[1] for x in self.proc]
        out = {}
        if g:
            gs = sorted(g)
            out["gpu"] = {
                "n": len(g), "mean": round(statistics.mean(g), 1),
                "p50": round(gs[len(gs) // 2], 1),
                "p95": round(gs[int(len(gs) * 0.95) - 1 if len(gs) > 1 else 0], 1),
                "max": round(max(g), 1),
                "idle_lt10_pct": round(100.0 * sum(1 for v in g if v < 10) / len(g), 1),
                "idle_lt30_pct": round(100.0 * sum(1 for v in g if v < 30) / len(g), 1),
                "series": [round(v, 0) for v in g],
            }
        if c:
            out["cpu_pct"] = round(statistics.mean(c), 1)
            out["cpu_series"] = [round(v, 0) for v in c]
        if pc:
            out["proc_cpu_pct"] = round(statistics.mean(pc), 1)
        return out


# ---------------------------------------------------------------- 一次导出

def run_once(clip, wait_ms, frames, period_ms, out_mp4):
    say("\n=== 导出：DFL_FUSED_BATCH_WAIT_MS=%s ===" % wait_ms)
    proc = start_server({"DFL_FUSED_BATCH_WAIT_MS": wait_ms})
    post, get = _http()
    try:
        body = {"video": str(clip), "model": str(MODEL), "settings": dict(SETTINGS),
                "output_dir": str(out_mp4), "frame_budget": 30,
                "start_frame": 1, "end_frame": frames}
        if out_mp4.exists():
            out_mp4.unlink()
        st, d = post("/api/merge/export", body)
        say("POST /api/merge/export -> HTTP %s fused=%s" % (st, d.get("fused")))
        if st != 200:
            say("body:", json.dumps(d, ensure_ascii=False)[:500])
            return {"error": d}
        smp = Sampler(period_ms)
        smp.start()
        t0 = time.time()
        seen = False
        last = {}
        while time.time() - t0 < 1800:
            _s, d2 = get("/api/merge/export/status", timeout=30)
            if d2.get("running"):
                seen = True
            last = d2
            if seen and not d2.get("running") and not d2.get("closing"):
                break
            time.sleep(0.2)
        wall = time.time() - t0
        smp.stop()
        res = smp.summary()
        res["wall_s"] = round(wall, 2)
        res["wait_ms"] = wait_ms
        for k in ("frames", "fps", "frame_fps", "frame_budget", "phase", "ok",
                  "per_stage_ms", "stage_util_pct", "seconds", "vram_plan"):
            if k in last:
                res[k] = last[k]
        return res
    finally:
        stop_server(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(WS / "9月29日 (5).mp4"))
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--period-ms", type=int, default=100)
    ap.add_argument("--wait-ms", nargs="*", type=str, default=["20"])
    ap.add_argument("--out", default=str(OUTDIR / "fused_gpu_util.json"))
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    clip = OUTDIR / "clip.mp4"
    if not clip.exists():
        say("切样片 …")
        subprocess.run([ffmpeg_exe(), "-y", "-ss", "3", "-t", "%d" % (a.frames / 30.0 + 1),
                        "-i", a.video, "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "copy",
                        "-movflags", "+faststart", str(clip)],
                       capture_output=True)
    if not clip.exists():
        raise SystemExit("样片切不出来：%s" % a.video)

    results = []
    for w in a.wait_ms:
        out_mp4 = OUTDIR / ("out_w%s.mp4" % w)
        try:
            results.append(run_once(clip, int(w), a.frames, a.period_ms, out_mp4))
        except Exception as e:                       # noqa: BLE001
            results.append({"wait_ms": w, "error": "%s: %s" % (type(e).__name__, e)})
        say("  ->", json.dumps({k: v for k, v in results[-1].items()
                               if k not in ("gpu", "cpu_series")},
                              ensure_ascii=False)[:600])
    Path(a.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    say("\n[ok]", a.out)


if __name__ == "__main__":
    main()
