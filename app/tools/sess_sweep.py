# -*- coding: utf-8 -*-
r"""异构 session 池的**扫描台**：在真流水线上把 (GPU, CPU, 策略) 逐格跑一遍。

为什么不是「算出来」：端到端帧率是**瓶颈段**决定的，任何一个非瓶颈段加容量都不会
提高 fps —— 这件事只能用同一条真片子、同一份 dfm、同一套参数跑出来才算数。所以
这里每一格都是**一次真的 /api/merge/export**，取 2~3 次的中位。

三档机器：
    low  —— 4 核 / 4GB 显存：DFL_SESS_LEVEL=low 让配比推导按低配的事实算，
            并用 cpu_affinity 把整个后端进程真摁到 4 个核上（不是嘴上说低配）
    mid  —— 本机真值（8 核 / 8GB 显存）
    high —— 32 核 / 24GB：DFL_SESS_LEVEL=high（推导）

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\sess_sweep.py --seconds 6 --reps 2
    runtime\server\Scripts\python.exe app\tools\sess_sweep.py --tiers mid --cells 1,0;1,1;1,2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
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
TEMP = Path(os.environ.get("TEMP") or ".")
BOX = TEMP / "dfl_sess_sweep"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"
DFM = "张元英"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}

BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

try:
    import psutil
except Exception:                                           # noqa: BLE001
    psutil = None


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


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ff(args):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败：%s" % (r.stderr or "")[-400:])


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
            raise SystemExit("建目录联接失败（%s）：%s" % (name, r.stdout))


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_FUSED_VRAM_FREE_MB", "DFL_VRAM_FREE_MB",
              "DFL_FUSED_BUDGET", "DFL_ALLOW_CPU_FALLBACK", "DFL_SESS_GPU",
              "DFL_SESS_CPU", "DFL_SESS_STRATEGY", "DFL_SESS_LEVEL"):
        env.pop(k, None)
    logf = open(str(SERVER_LOG), "ab")
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
            txt = line.decode("utf-8", "replace")
            if port["v"] is None and txt.startswith("PORT="):
                try:
                    port["v"] = int(txt.strip().split("=", 1)[1])
                except Exception:                           # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % SERVER_LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行")
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端：%s（pid=%s）" % (BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = http("GET", "/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端不应答")


def stop_server(proc):
    if proc is None:
        return
    try:
        for c in psutil.Process(proc.pid).children(recursive=True):
            try:
                c.kill()
            except Exception:                               # noqa: BLE001
                pass
    except Exception:                                       # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.0)


def set_affinity(proc, ncore):
    """把后端（以及它以后 fork 出来的孩子，Windows 上不继承，所以只对后端本身生效）
    摁到前 ncore 个核上 —— 模拟低配机的**真**算力，而不是嘴上说低配。"""
    if psutil is None or not ncore:
        return None
    try:
        p = psutil.Process(proc.pid)
        allc = p.cpu_affinity()
        p.cpu_affinity(list(allc[:int(ncore)]))
        return allc
    except Exception as e:                                  # noqa: BLE001
        say("affinity 设置失败：%s" % e)
        return None


def restore_affinity(proc, allc):
    if psutil is None or not allc:
        return
    try:
        psutil.Process(proc.pid).cpu_affinity(allc)
    except Exception:                                       # noqa: BLE001
        pass


def wait_export(timeout=1800.0):
    t0 = time.time()
    seen = False
    same = 0
    last = {}
    while time.time() - t0 < timeout:
        _s, d = http("GET", "/api/merge/export/status", timeout=60)
        cur = d or {}
        term = (cur.get("phase") in ("done", "cancelled", "failed")
                and not cur.get("running"))
        if cur.get("running"):
            seen = True
        same = (same + 1) if (term and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if term and ((seen and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.3)
    return last


def log_of(out_path):
    p = Path(str(out_path)).with_suffix(".log")
    try:
        return p.read_text("utf-8", "ignore") if p.is_file() else ""
    except OSError:
        return ""


def pick(log, tag):
    return [ln for ln in (log or "").splitlines() if tag in ln]


def one_run(clip, tag, cfg):
    out = OUT / ("%s.mp4" % tag)
    if out.exists():
        out.unlink()
    body = {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "output_dir": str(out)}
    if cfg.get("gpu") is not None:
        body["sess_gpu"] = int(cfg["gpu"])
    if cfg.get("cpu") is not None:
        body["sess_cpu"] = int(cfg["cpu"])
    if cfg.get("strategy"):
        body["sess_strategy"] = str(cfg["strategy"])
    if cfg.get("level"):
        body["sess_level"] = str(cfg["level"])
    peak = {"rss": 0.0}
    stop = threading.Event()

    def sample(pid):
        while not stop.is_set():
            try:
                tot = 0.0
                pr = psutil.Process(pid)
                for p in [pr] + pr.children(recursive=True):
                    tot += p.memory_info().rss
                peak["rss"] = max(peak["rss"], tot / 1048576.0)
            except Exception:                               # noqa: BLE001
                pass
            stop.wait(0.3)

    t0 = time.time()
    st, d = http("POST", "/api/merge/export", body)
    if st != 200 or (d or {}).get("ok") is False:
        return {"tag": tag, "cfg": cfg, "error": "HTTP %s %s" % (st, (d or {}).get("error"))}
    if psutil is not None:
        threading.Thread(target=sample, args=(SRV_PID[0],), daemon=True).start()
    last = wait_export()
    stop.set()
    wall = time.time() - t0
    log = log_of(out)
    fps = None
    for ln in pick(log, "流水线跑完"):
        m = re.search(r"([0-9.]+) 帧/秒", ln)
        if m:
            fps = float(m.group(1))
    util = {}
    for ln in pick(log, "各段占用"):
        for k, v in re.findall(r"([a-z_]+)=([0-9]+)%", ln):
            util[k] = int(v)
    plan = (pick(log, "检测池：") or [""])[0]
    bench = (pick(log, "检测池 基准") or [""])[0]
    return {"tag": tag, "cfg": dict(cfg), "fps": fps, "wall": round(wall, 2),
            "util": util, "plan": plan, "bench": bench,
            "peak_rss_mb": round(peak["rss"], 1), "phase": (last or {}).get("phase"),
            "error": "" if (last or {}).get("ok") else str((last or {}).get("error"))[:200]}


SRV_PID = [0]


def cells_for(tier):
    lv = {"low": "low", "mid": None, "high": "high"}.get(tier)
    base = []
    for g, c in [(1, 0), (1, 1), (1, 2), (1, 4), (2, 0), (0, 0)]:
        base.append({"gpu": g, "cpu": c, "strategy": "", "level": lv,
                     "name": "g%d_c%d_auto" % (g, c)})
    for s in ("shortq", "rr", "static"):
        base.append({"gpu": 1, "cpu": 2, "strategy": s, "level": lv,
                     "name": "g1_c2_%s" % s})
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description="异构 session 池扫描台")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tiers", default="low,mid,high")
    ap.add_argument("--cells", default="", help="只跑这几格，如 g1_c0_auto,g1_c1_auto")
    ap.add_argument("--json", default=str(BOX / "sweep.json"))
    args = ap.parse_args(argv)
    global DFM
    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / "clip.mp4"
    run_ff(["-ss", "3", "-t", str(args.seconds), "-i", src, "-c:v", "libx264", "-crf",
            "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", clip])
    say("样片 %s（%.0f 秒）" % (clip.name, args.seconds))
    only = [x.strip() for x in str(args.cells).split(";") if x.strip()]
    rows = []
    print("reps=%d · 每格取中位" % args.reps)
    for tier in [t.strip() for t in str(args.tiers).split(",") if t.strip()]:
        head("机器档位 %s" % tier)
        proc = start_server(tier)
        SRV_PID[0] = proc.pid
        allc = set_affinity(proc, 4 if tier == "low" else 0)
        try:
            for cell in cells_for(tier):
                if only and cell["name"] not in only:
                    continue
                got = []
                for rep in range(max(1, int(args.reps))):
                    r = one_run(clip, "%s_%s_r%d" % (tier, cell["name"], rep), cell)
                    if r.get("fps"):
                        got.append(r)
                    else:
                        say("[fail] %s r%d：%s" % (cell["name"], rep, r.get("error")))
                if not got:
                    continue
                med = dict(got[len(got) // 2])
                med["fps_list"] = [g["fps"] for g in got]
                med["fps"] = round(statistics.median([g["fps"] for g in got]), 2)
                med["tier"] = tier
                med["name"] = cell["name"]
                rows.append(med)
                print("  %-14s fps=%6.2f (%s) · detect=%s%% dfm=%s%% · 峰值 RSS %.0f MB · %s"
                      % (cell["name"], med["fps"],
                         "/".join("%.2f" % x for x in med["fps_list"]),
                         med["util"].get("detect"), med["util"].get("dfm"),
                         med["peak_rss_mb"], (med["plan"] or "")[:110]))
        finally:
            restore_affinity(proc, allc)
            stop_server(proc)
    try:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
        print("JSON：%s" % args.json)
    except Exception as e:                                  # noqa: BLE001
        say("写 JSON 失败：%s" % e)
    head("切片：detect 段占用 vs 帧率")
    for r in rows:
        print("  %-6s %-14s fps=%6.2f detect=%3s%% dfm=%3s%% compose=%3s%% CPU池=%s"
              % (r.get("tier"), r.get("name"), r.get("fps") or 0,
                 r["util"].get("detect"), r["util"].get("dfm"),
                 r["util"].get("compose"),
                 ("有" if "CPU" in (r.get("plan") or "") and "0×CPU" not in (r.get("plan") or "")
                  else "无")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
