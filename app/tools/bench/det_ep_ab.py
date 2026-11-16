# -*- coding: utf-8 -*-
r"""DFL_DET_EP 的 A/B 台架：检测放 GPU 队列 vs 放 CPU，端到端谁快。

为什么要有这个台架（而不是直接开那个开关）：检测段自己的 ms 只能说明「这一段多贵」，
说明不了「它占着 GPU 流时别人排了多久」。唯一的判据是**同一段片子、同一套参数**下的
端到端帧/秒 + 各段占用；而且必须**同一份代码、只翻一个环境变量**。

它量什么（每个配置 N 次，取中位）：
    fps        成片帧数 / 流水线墙钟（后端自己算的 stats.fps）
    GPU%       nvidia-smi 100ms 采样，取「已经出第一帧之后」的中位
    CPU%       整棵进程树的 CPU 时间 / 墙钟（psutil，含 ffmpeg 子进程）
    detect ms  per_stage_ms["detect"]（每帧；检测线程真正的忙时间）
    dfm ms     per_stage_ms["dfm"]（每帧；换脸前向 —— GPU 上最大的一笔）
    落后帧数    每 100ms 读一次 /api/merge/export/status 的 stages.ticks，算
                ticks.frames − ticks.<段> 的中位/p90（换脸/遮罩/合成各一列）

像素：额外的两次导出带 DFL_FUSED_DUMP=30,60,90,120，把 A/B 的 dump 目录逐文件比
（composed / pred / lm.json / xseg），报 max|Δ| 与「lm 是否逐字节相同」。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\det_ep_ab.py
    ... --runs 3 --frames 120 --configs A,B --model 阿証証証証-416
退出码：0 = 跑完（结论可能是「不值得」）；2 = 有配置没跑成。
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
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_det_ep_ab"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"

SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
DETECTOR = "tinymog"
VRAM_FREE = 5300        # 两个配置钉同一个显存数 → 同一个帧预算（否则比的是两种预算）
BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def say(m):
    print("       " + str(m), flush=True)


def head(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


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
            raise SystemExit("建目录联接失败（%s）：%s" % (name, r.stderr))
    say("沙箱：%s" % BOX)


def _child_pids(pid):
    try:
        return [int(x) for x in subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"ParentProcessId=%d\" | "
             "Select-Object -ExpandProperty ProcessId" % int(pid)],
            capture_output=True, text=True, timeout=30).stdout.split()
            if x.strip().isdigit()]
    except Exception:                                       # noqa: BLE001
        return []


def _kill_tree(pid, depth=0):
    if depth > 4:
        return
    for c in _child_pids(pid):
        _kill_tree(c, depth + 1)
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))], capture_output=True,
                       timeout=30)
    except Exception:                                       # noqa: BLE001
        pass


def start_server(tag="", extra_env=None):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_BUDGET", "DFL_VRAM_FREE_MB",
              "DFL_ALLOW_CPU_FALLBACK", "DFL_DET_EP", "DFL_DET_EP_THREADS",
              "DFL_FUSED_DUMP", "DFL_SESS_PLAN", "DFL_SESS_CPU", "DFL_SESS_GPU"):
        env.pop(k, None)
    env["DFL_FUSED_VRAM_FREE_MB"] = str(VRAM_FREE)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    logf = open(str(LOGDIR / "server.log"), "ab")
    logf.write(("\n===== start %s (%s) =====\n"
                % (tag, json.dumps(extra_env or {}, ensure_ascii=False))).encode("utf-8"))
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
            t = line.decode("utf-8", "replace")
            if port["v"] is None and t.startswith("PORT="):
                try:
                    port["v"] = int(t.strip().split("=", 1)[1])
                except Exception:                           # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % (LOGDIR / "server.log"))
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行")
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端[%s] %s（pid=%s）" % (tag, BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = http("GET", "/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 status 不应答")


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.0)


# ============================================================
# 采样：GPU%（nvidia-smi）/ CPU%（psutil 进程树积分）
# ============================================================
class Sampler(object):
    def __init__(self, pid):
        self.pid = int(pid)
        self.gpu = []          # [(t, util%)]
        self.cpu = []          # [(t, 累计 cpu 秒)]
        self.on = True
        self.proc = None
        self._th = []
        self._ps = None
        try:
            import psutil
            self._ps = psutil
        except Exception:                                   # noqa: BLE001
            self._ps = None

    def start(self):
        self._th.append(threading.Thread(target=self._gpu_loop, daemon=True))
        self._th.append(threading.Thread(target=self._cpu_loop, daemon=True))
        for t in self._th:
            t.start()
        return self

    def _gpu_loop(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-lms", "100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception:                                   # noqa: BLE001
            return
        for line in iter(self.proc.stdout.readline, b""):
            if not self.on:
                break
            try:
                v = float(line.decode("utf-8", "replace").strip().split(",")[0])
            except Exception:                               # noqa: BLE001
                continue
            self.gpu.append((time.time(), v))
        try:
            self.proc.kill()
        except Exception:                                   # noqa: BLE001
            pass

    def _tree_cpu(self):
        if self._ps is None:
            return 0.0
        try:
            p = self._ps.Process(self.pid)
            tot = 0.0
            for q in [p] + p.children(recursive=True):
                try:
                    t = q.cpu_times()
                    tot += float(t.user) + float(t.system)
                except Exception:                   # noqa: BLE001
                    pass
            return tot
        except Exception:                                   # noqa: BLE001
            return 0.0

    def _cpu_loop(self):
        while self.on:
            self.cpu.append((time.time(), self._tree_cpu()))
            time.sleep(0.2)

    def stop(self):
        self.on = False
        try:
            if self.proc is not None:
                self.proc.kill()
        except Exception:                                   # noqa: BLE001
            pass
        for t in self._th:
            t.join(timeout=2.0)
        g = [v for _t, v in self.gpu]
        cpu_pct = 0.0
        if len(self.cpu) >= 2:
            t0, c0 = self.cpu[0]
            t1, c1 = self.cpu[-1]
            if t1 > t0:
                cpu_pct = 100.0 * (c1 - c0) / (t1 - t0)
        return {"gpu": g, "cpu_pct": round(cpu_pct, 1),
                "cpu_samples": len(self.cpu), "gpu_samples": len(g)}


def _med(v):
    return round(float(statistics.median(v)), 2) if v else None


def _p90(v):
    if not v:
        return None
    s = sorted(v)
    return round(float(s[min(len(s) - 1, int(len(s) * 0.9))]), 2)


# ============================================================
# 一次导出
# ============================================================
def run_export(body, tag, want_dump=None):
    st, d = http("POST", "/api/merge/export", body)
    if st != 200 or d.get("ok") is False:
        raise SystemExit("[%s] 导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
    t_start = time.time()
    lags = {"dfm": [], "xseg": [], "compose": []}
    inflight = {"det": [], "dfm": [], "xseg": [], "compose": []}
    done_seq = []
    first_done_t = None
    seen = False
    last = {}
    same = 0
    while True:
        _s, cur = http("GET", "/api/merge/export/status", timeout=60)
        cur = cur or {}
        if cur.get("running"):
            seen = True
        v = cur.get("vid") or {}
        done = int(v.get("done") or 0)
        if done > 0 and first_done_t is None:
            first_done_t = time.time()
        stages = cur.get("stages") or {}
        if stages:
            tk = {k: int((stages.get(k) or {}).get("ticks") or 0) for k in stages}
            for k, key in (("dfm", "dfm"), ("xseg", "xseg"), ("compose", "compose")):
                if k in tk:
                    lags[key].append(int(tk.get("frames") or 0) - int(tk.get(k) or 0))
            for k in inflight:
                inflight[k].append(int((stages.get(k) or {}).get("inflight") or 0))
        done_seq.append(done)
        terminal = (cur.get("phase") in ("done", "cancelled", "failed")
                    and not cur.get("running"))
        same = (same + 1) if (terminal and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if terminal and ((seen and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.1)
    wall = time.time() - t_start
    vf = last.get("vid") or {}
    stats = vf.get("stats") or {}
    return {"tag": tag, "phase": last.get("phase"), "ok": last.get("ok"),
            "err": (last.get("error") or "")[:300],
            "fps": float(stats.get("fps") or 0.0), "wall": round(wall, 2),
            "frames": int(vf.get("done") or 0),
            "budget": vf.get("frame_budget"),
            "per_stage_ms": vf.get("per_stage_ms") or {},
            "util_pct": vf.get("stage_util_pct") or {},
            "lags": {k: {"med": _med(v), "p90": _p90(v), "max": (max(v) if v else None)}
                     for k, v in lags.items()},
            "inflight": {k: {"med": _med(v), "max": (max(v) if v else None)}
                         for k, v in inflight.items()},
            "done_first_at": (None if first_done_t is None else round(first_done_t - t_start, 2)),
            "out": str(vf.get("output") or body.get("output_dir") or ""),
            "log": (str(last.get("vid", {}).get("log_path") or "") or "")}


def log_text(out_mp4):
    p = Path(str(out_mp4)).with_suffix(".log")
    try:
        return p.read_text("utf-8", "ignore") if p.is_file() else ""
    except OSError:
        return ""


def dump_dir(out_mp4):
    root = Path(str(out_mp4)).with_suffix("")
    return Path(str(root) + ".dump")


def compare_dumps(a_dir, b_dir):
    """逐文件比两次导出的 dump（composed / pred / lm / xseg）。"""
    import numpy as np
    if not (a_dir.is_dir() and b_dir.is_dir()):
        return {"err": "dump 目录不存在：%s / %s" % (a_dir, b_dir)}
    res = {"files": 0, "pairs": [], "lm_same": 0, "lm_total": 0}
    names = sorted(p.name for p in a_dir.iterdir())
    for nm in names:
        pb = b_dir / nm
        if not pb.is_file():
            res["pairs"].append({"name": nm, "err": "B 缺这个文件"})
            continue
        res["files"] += 1
        if nm.endswith(".json"):
            res["lm_total"] += 1
            if (a_dir / nm).read_bytes() == pb.read_bytes():
                res["lm_same"] += 1
            else:
                res["pairs"].append({"name": nm, "text": True, "same": False})
            continue
        if nm.endswith(".txt"):
            res["pairs"].append({"name": nm, "text": True,
                                 "same": (a_dir / nm).read_bytes() == pb.read_bytes()})
            continue
        import cv2
        ia = cv2.imread(str(a_dir / nm), cv2.IMREAD_UNCHANGED)
        ib = cv2.imread(str(pb), cv2.IMREAD_UNCHANGED)
        if ia is None or ib is None or ia.shape != ib.shape:
            res["pairs"].append({"name": nm, "err": "读不出/形状不同"})
            continue
        dd = np.abs(ia.astype(np.int32) - ib.astype(np.int32))
        res["pairs"].append({"name": nm, "mean": float(dd.mean()),
                             "max": int(dd.max()), "n": int((dd > 0).sum()),
                             "px": int(dd.size)})
    return res


# ============================================================
# 主流程
# ============================================================
def clip_1080p(frames=120):
    import imageio_ffmpeg
    src = sorted((REPO / "workspace").glob("*.mp4"))
    if not src:
        raise SystemExit("workspace 里没有 mp4 当样片")
    out = MEDIA / ("clip%d.mp4" % frames)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # -frames:v 精确切 N 帧（-t 秒数会随源 fps 取整，129 帧那种数字没法做「同段 120 帧」）
    r = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-ss", "3",
                        "-i", str(src[0]), "-frames:v", str(int(frames)),
                        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("切样片失败：%s" % (r.stderr or "")[-300:])
    import cv2
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    say("样片 %s：%dx%d · %d 帧" % (out.name, w, h, n))
    if (w, h) != (1080, 1920):
        raise SystemExit("样片不是 1080x1920")
    return out, n


def one_run(tag, env, clip, n_frames, model, run_idx):
    """起一次后端 + 跑一次导出 + 收尾。交错对比要「每次都是冷启动」才公平。"""
    proc = start_server("%s#%d" % (tag, run_idx), env)
    try:
        out = OUT / ("%s_%d.mp4" % (tag, run_idx))
        if out.exists():
            out.unlink()
        body = {"video": str(clip), "model": model, "settings": dict(SETTINGS),
                "detector": DETECTOR, "output_dir": str(out),
                "start_frame": 1, "end_frame": int(n_frames)}
        smp = Sampler(proc.pid).start()
        t0 = time.time()
        r = run_export(body, "%s#%d" % (tag, run_idx))
        sm = smp.stop()
        r["sample"] = {"gpu_med": _med(sm["gpu"]), "gpu_p90": _p90(sm["gpu"]),
                       "cpu_pct": sm["cpu_pct"], "n_gpu": sm["gpu_samples"]}
        r["wall_total"] = round(time.time() - t0, 2)
        r["run"] = run_idx
        r["log_det"] = [ln.strip() for ln in log_text(r["out"]).splitlines()
                        if "检测 EP" in ln or "检测池" in ln][:4]
        say("%s#%d：%s · %.2f 帧/秒 · %d 帧 · %.2fs（检测 %.2f ms / dfm %.2f ms / "
            "xseg %.2f ms）· GPU %s%% / CPU %s%%"
            % (tag, run_idx, r["phase"], r["fps"], r["frames"], r["wall"],
               float((r["per_stage_ms"] or {}).get("detect") or 0.0),
               float((r["per_stage_ms"] or {}).get("dfm") or 0.0),
               float((r["per_stage_ms"] or {}).get("xseg") or 0.0),
               r["sample"]["gpu_med"], r["sample"]["cpu_pct"]))
        for ln in r["log_det"][:1]:
            say("   " + ln)
        return r
    finally:
        stop_server(proc)


def _agg(rows):
    good = [r for r in rows if r["phase"] == "done"]
    agg = {}
    if good:
        agg = {"fps": _med([r["fps"] for r in good]),
               "gpu": _med([r["sample"]["gpu_med"] for r in good]),
               "cpu": _med([r["sample"]["cpu_pct"] for r in good]),
               "budget": good[0]["budget"],
               "detect_ms": _med([float((r["per_stage_ms"] or {}).get("detect") or 0.0)
                                  for r in good]),
               "dfm_ms": _med([float((r["per_stage_ms"] or {}).get("dfm") or 0.0)
                               for r in good]),
               "xseg_ms": _med([float((r["per_stage_ms"] or {}).get("xseg") or 0.0)
                                for r in good]),
               "compose_ms": _med([float((r["per_stage_ms"] or {}).get("compose") or 0.0)
                                   for r in good]),
               "encode_ms": _med([float((r["per_stage_ms"] or {}).get("encode") or 0.0)
                                  for r in good]),
               "lag_dfm": _med([(r["lags"]["dfm"]["med"] or 0) for r in good]),
               "lag_xseg": _med([(r["lags"]["xseg"]["med"] or 0) for r in good]),
               "lag_compose": _med([(r["lags"]["compose"]["med"] or 0) for r in good]),
               "lag_dfm_p90": _med([(r["lags"]["dfm"]["p90"] or 0) for r in good]),
               "util": good[0]["util_pct"],
               "n_ok": len(good), "fps_all": [r["fps"] for r in good]}
    return agg


def one_config(tag, env, clip, n_frames, runs, model):
    """起一次后端，跑 runs 次导出（第一次含模型加载，单独标出来）。"""
    head("%s · DFL_DET_EP=%s" % (tag, env.get("DFL_DET_EP", "auto")))
    proc = start_server(tag, env)
    rows = []
    try:
        for i in range(runs):
            out = OUT / ("%s_%d.mp4" % (tag, i + 1))
            if out.exists():
                out.unlink()
            body = {"video": str(clip), "model": model, "settings": dict(SETTINGS),
                    "detector": DETECTOR, "output_dir": str(out),
                    "start_frame": 1, "end_frame": int(n_frames)}
            smp = Sampler(proc.pid).start()
            t0 = time.time()
            r = run_export(body, "%s#%d" % (tag, i + 1))
            sm = smp.stop()
            r["sample"] = {"gpu_med": _med(sm["gpu"]), "gpu_p90": _p90(sm["gpu"]),
                           "cpu_pct": sm["cpu_pct"], "n_gpu": sm["gpu_samples"]}
            r["wall_total"] = round(time.time() - t0, 2)
            r["run"] = i + 1
            r["log_det"] = [ln.strip() for ln in log_text(r["out"]).splitlines()
                            if "检测 EP" in ln or "检测池" in ln][:4]
            rows.append(r)
            say("%s#%d：%s · %.2f 帧/秒 · %d 帧 · %.2fs（检测 %.2f ms / dfm %.2f ms）"
                % (tag, i + 1, r["phase"], r["fps"], r["frames"], r["wall"],
                   float((r["per_stage_ms"] or {}).get("detect") or 0.0),
                   float((r["per_stage_ms"] or {}).get("dfm") or 0.0)))
            for ln in r["log_det"][:1]:
                say("   " + ln)
    finally:
        stop_server(proc)
    return {"tag": tag, "env": env, "runs": rows, "agg": _agg(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="DFL_DET_EP A/B（检测 GPU vs CPU）")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--model", default="阿証証証証-416")
    ap.add_argument("--configs", default="A,B")
    ap.add_argument("--dump", type=int, default=1, help="额外跑一次带 dump 的导出做像素对拍")
    ap.add_argument("--interleave", type=int, default=0,
                    help="1 = 交错跑（每轮 A/B 各一次、每次冷启动后端）：GPU 会热降频、"
                         "后台进程会变，串行跑「先 A 后 B」就会把时间趋势算进差异里。"
                         "本机实测两种跑法能差出 4 个百分点。")
    ap.add_argument("--json", default=str(Path(os.environ.get("TEMP") or ".") / "det_ep_ab.json"))
    args = ap.parse_args(argv)
    t_all = time.time()
    want = [c.strip().upper() for c in str(args.configs).split(",") if c.strip()]
    prepare_sandbox()
    clip, n = clip_1080p(args.frames)
    res = {"clip": str(clip), "frames": n, "model": args.model, "detector": DETECTOR,
           "settings": SETTINGS, "vram_free_mb": VRAM_FREE, "configs": {}}
    envs = {"A": {"DFL_DET_EP": "gpu"},
            "B": {"DFL_DET_EP": "cpu"},
            "AUTO": {"DFL_DET_EP": "auto"},
            # 线程数两端各来一档：2（给合成让路）与 4（用户口径的上限）
            "B2": {"DFL_DET_EP": "cpu", "DFL_DET_EP_THREADS": "2"},
            "B4": {"DFL_DET_EP": "cpu", "DFL_DET_EP_THREADS": "4"}}
    if args.interleave:
        rows = {t: [] for t in want}
        for k in range(args.runs):
            for tag in want:                    # 同一轮里 A/B 紧挨着跑：趋势两边一起承担
                rows[tag].append(one_run(tag, envs[tag], clip, n, args.model, k + 1))
        for tag in want:
            res["configs"][tag] = {"tag": tag, "env": envs[tag], "runs": rows[tag],
                                   "agg": _agg(rows[tag])}
        # 配对差（同一轮 A 与 B 的差，再取中位）—— 比「两个中位相减」抗漂移
        if "A" in want and "B" in want and len(rows["A"]) == len(rows["B"]):
            pairs = [(a["fps"], b["fps"]) for a, b in zip(rows["A"], rows["B"])
                     if a["phase"] == "done" and b["phase"] == "done"]
            if pairs:
                res["paired"] = {"pairs": [[round(a, 2), round(b, 2)] for a, b in pairs],
                                 "delta_pct_med": round(float(statistics.median(
                                     [100.0 * (b / a - 1.0) for a, b in pairs])), 2)}
    else:
        for tag in want:
            r = one_config(tag, envs[tag], clip, n, args.runs, args.model)
            res["configs"][tag] = r
    # ---- 像素对拍：A / B 各跑一次带 dump 的导出（帧预算/参数完全一样）----
    if args.dump and "A" in res["configs"] and "B" in res["configs"]:
        head("像素对拍：A/B 各一次带 dump 的导出（FUSED_DUMP=30,60,90,120）")
        dumped = {}
        for tag in ("A", "B"):
            _env = dict(envs[tag])
            _env["DFL_FUSED_DUMP"] = "30,60,90,120"
            proc = start_server("dump-" + tag, _env)
            try:
                out = OUT / ("dump_%s.mp4" % tag)
                shutil.rmtree(str(dump_dir(out)), ignore_errors=True)
                body = {"video": str(clip), "model": args.model, "settings": dict(SETTINGS),
                        "detector": DETECTOR, "output_dir": str(out),
                        "start_frame": 1, "end_frame": int(n)}
                r = run_export(body, "dump-" + tag)
                say("%s dump 导出：%s · %d 帧" % (tag, r["phase"], r["frames"]))
                dumped[tag] = dump_dir(out)
            finally:
                stop_server(proc)
        cmp = compare_dumps(dumped.get("A"), dumped.get("B"))
        res["dump"] = {"dir_a": str(dumped.get("A")), "dir_b": str(dumped.get("B")),
                       "compare": cmp}
        composed = [p for p in cmp.get("pairs", []) if p.get("name", "").endswith("_composed.png")]
        preds = [p for p in cmp.get("pairs", []) if "_pred.png" in p.get("name", "")]
        for p in composed:
            say("composed %s：mean=%.4f max=%d（不同像素 %d/%d）"
                % (p["name"], p.get("mean", -1), p.get("max", -1), p.get("n", -1),
                   p.get("px", -1)))
        if preds:
            say("pred 最大：mean=%.4f max=%d" % (max(p.get("mean", 0) for p in preds),
                                                 max(p.get("max", 0) for p in preds)))
        say("lm.json 逐字节相同：%d/%d" % (cmp.get("lm_same", -1), cmp.get("lm_total", -1)))
    Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    # ---- 汇总表 ----
    head("汇总（%d 次中位）" % args.runs)
    hdr = ("%-4s %7s %6s %6s %9s %8s %8s %8s %8s %s" %
           ("cfg", "fps", "GPU%", "CPU%", "detect ms", "dfm ms", "xseg ms", "合成 ms",
            "预算", "段占用%（detect/dfm/xseg/合成/编码）"))
    print(hdr)
    print("-" * len(hdr))
    base_fps = (res["configs"].get("A") or {}).get("agg", {}).get("fps")
    for tag, r in res["configs"].items():
        a = r["agg"]
        if not a:
            print("%-4s 没跑成" % tag)
            continue
        u = a.get("util") or {}
        print("%-4s %7.2f %6s %6s %9.2f %8.2f %8.2f %8.2f %8s %s / %s / %s / %s / %s%s"
              % (tag, a["fps"], a["gpu"], a["cpu"], a["detect_ms"], a["dfm_ms"],
                 a["xseg_ms"], a["compose_ms"], a["budget"],
                 u.get("detect"), u.get("dfm"), u.get("xseg"), u.get("compose"),
                 u.get("encode"),
                 ("" if not base_fps or tag == "A"
                  else "  · 相对 A %+.1f%%" % (100.0 * (a["fps"] / base_fps - 1.0)))))
    print("")
    print("落后帧数（ticks.frames − ticks.<段>，每个样本的中位再取配置中位）")
    for tag, r in res["configs"].items():
        a = r["agg"]
        if not a:
            continue
        print("  %-4s 换脸 %5s 帧（p90 %s）· 遮罩 %5s 帧 · 合成 %5s 帧"
              % (tag, a["lag_dfm"], a["lag_dfm_p90"], a["lag_xseg"], a["lag_compose"]))
    if res.get("paired"):
        p = res["paired"]
        print("")
        print("配对差（同轮 A→B，%d 对）：中位 %+.2f%%  %s"
              % (len(p["pairs"]), p["delta_pct_med"],
                 " · ".join("%.2f→%.2f" % (a, b) for a, b in p["pairs"])))
    if res.get("dump"):
        c = res["dump"]["compare"]
        print("")
        print("像素：composed / pred 全部文件见 JSON；lm.json 相同 %d/%d"
              % (c.get("lm_same", -1), c.get("lm_total", -1)))
    print("")
    print("用时 %.1fs · JSON %s · 沙箱 %s" % (time.time() - t_all, args.json, BOX))
    bad = [t for t, r in res["configs"].items() if not r["agg"]]
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
