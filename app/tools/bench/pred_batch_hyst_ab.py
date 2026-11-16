# -*- coding: utf-8 -*-
r"""不钉 batch 的端到端 A/B：DFL_PRED_BATCH_HYST=0（旧）vs =1（滞回）。

为什么要在真后端里量、而不是只调 _pred_pick_batch：批目标抖动只有在「同素材、
同模型、连续导出」这条路上才变成 fps 抖动；而 batch 的台阶是在**真实显存状态**下
量的（XSeg 会话、ORT arena 都还在），冷进程里量的数不代表它。

每个配置：起一次后端 → 连续 N 轮「同 start_frame..end_frame 的定时长导出」，
逐轮记 status 里的 vid.batch（真正的批目标）和 stats.fps。

跑法（仓库根，server venv）：
    runtime\server\Scripts\python.exe app\tools\bench\pred_batch_hyst_ab.py
    ... --rounds 10 --frames 150 --start 1200 --configs 0,1
    ... --configs 1            # 只跑一个配置
退出码：0 = 跑完；2 = 有一单没跑成。
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

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_pred_batch_ab"
WS = BOX / "ws"
OUT = BOX / "out"
LOGDIR = BOX / "log"

MODEL = "阿証証証証-416"
VIDEO = REPO / "workspace" / "9月29日 (5).mp4"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def say(m=""):
    try:
        print(m, flush=True)
    except UnicodeEncodeError:
        print(str(m).encode("utf-8", "replace").decode("utf-8", "replace"), flush=True)


def head(t):
    say("")
    say("=" * 78)
    say(t)
    say("=" * 78)


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
        except Exception:                                     # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


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
    for d in (WS, OUT, LOGDIR):
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
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"ParentProcessId=%d\" | "
             "Select-Object -ExpandProperty ProcessId" % int(pid)],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:                                         # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _kill_tree(pid, depth=0):
    if depth > 4:
        return
    for c in _child_pids(pid):
        _kill_tree(c, depth + 1)
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))], capture_output=True,
                       timeout=30)
    except Exception:                                         # noqa: BLE001
        pass


def start_server(tag, extra_env):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_BUDGET", "DFL_VRAM_FREE_MB",
              "DFL_ALLOW_CPU_FALLBACK", "DFL_DET_EP", "DFL_FUSED_DUMP",
              "DFL_PRED_BATCH_HYST", "DFL_FUSED_VRAM_FREE_MB", "DFL_DFM_TRT"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    logpath = LOGDIR / ("server_%s.log" % tag)
    logf = open(str(logpath), "wb")
    proc = subprocess.Popen([str(PY), "serve.py"], cwd=str(APP), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    port = {"v": None}

    def reader():
        for line in iter(proc.stdout.readline, b""):
            try:
                logf.write(line)
                logf.flush()
            except Exception:                                 # noqa: BLE001
                pass
            t = line.decode("utf-8", "replace")
            if port["v"] is None and t.startswith("PORT="):
                try:
                    port["v"] = int(t.strip().split("=", 1)[1])
                except Exception:                             # noqa: BLE001
                    pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while port["v"] is None and time.time() - t0 < 240:
        if proc.poll() is not None:
            raise SystemExit("后端启动失败，见 %s" % logpath)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % logpath)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端[%s]：%s（pid=%s）· log=%s" % (tag, BASE, proc.pid, logpath.name))
    while time.time() - t0 < 300:
        try:
            st, _d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc, logpath
        except Exception:                                     # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 status 不应答")


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                         # noqa: BLE001
        pass
    time.sleep(1.2)


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


_LADDER_RE = re.compile(r"batch 阶梯 (.+?) → 选 (\d+)")
_ROW_RE = re.compile(r"B(\d+)=([\d.]+)ms\((\d+\.\d+)ms/张\)")


def parse_ladder(line):
    """'B1=36.6ms(36.63ms/张) / B4=... → 选 8' → [{batch, ms, per_face_ms}, ...] + pick。"""
    m = _LADDER_RE.search(line or "")
    if not m:
        return [], None
    rows = [{"batch": int(b), "ms": float(ms), "per_face_ms": float(per)}
            for b, ms, per in _ROW_RE.findall(m.group(1))]
    return rows, int(m.group(2))


def vid_log_pick(out_mp4):
    """<成片>.log 里「dfm xxx 批大小 N」那一行的 N（生产日志的原文）。"""
    p = Path(str(out_mp4)).with_suffix(".log")
    try:
        txt = p.read_text("utf-8", "ignore")
    except OSError:
        return None, ""
    for ln in txt.splitlines():
        if "批大小" in ln:
            m = re.search(r"批大小 (\d+)", ln)
            if m:
                return int(m.group(1)), ln.strip()
    return None, ""


def ladder_lines(logpath):
    try:
        txt = Path(logpath).read_text("utf-8", "ignore")
    except OSError:
        return []
    return [ln.strip() for ln in txt.splitlines() if "batch 阶梯" in ln]


# ============================================================
# 一个配置：起后端 + N 轮导出
# ============================================================
def run_config(hyst, rounds, start_frame, frames, batch=0, fresh=False):
    env = {"DFL_PRED_BATCH_HYST": str(hyst)}
    head("配置 DFL_PRED_BATCH_HYST=%s · %d 轮 × %d 帧（不钉 batch）"
         % (hyst, rounds, frames))
    proc, logpath = start_server("hyst%s" % hyst, env)
    rows = []
    try:
        for i in range(int(rounds)):
            out = OUT / ("h%s_r%02d.mp4" % (hyst, i + 1))
            if out.exists():
                out.unlink()
            body = {"video": str(VIDEO), "model": MODEL, "settings": dict(SETTINGS),
                    "output_dir": str(out),
                    "start_frame": int(start_frame),
                    "end_frame": int(start_frame) + int(frames) - 1}
            if int(batch or 0) > 0:
                body["batch"] = int(batch)          # 钉死（对照用）
            t0 = time.time()
            st, d = post("/api/merge/export", body)
            if st != 200 or d.get("ok") is False:
                say("第 %d 轮没起来：HTTP %s %s" % (i + 1, st, (d or {}).get("error")))
                return {"hyst": hyst, "rows": rows, "failed": True,
                        "error": str((d or {}).get("error"))[:400]}
            fin = wait_export(1800.0)
            ph = fin.get("phase")
            v = fin.get("vid") or {}
            stats = v.get("stats") or {}
            fps = float(stats.get("fps") or 0.0)
            done = int(v.get("done") or 0)
            pick, pick_line = vid_log_pick(out)
            lads = ladder_lines(logpath)
            lad_rows, lad_pick = parse_ladder(lads[-1]) if lads else ([], None)
            if pick is None:
                pick = lad_pick
            rows.append({"i": i + 1, "phase": ph, "out": str(out),
                         "fps": round(fps, 3), "frames": done,
                         "batch": pick, "ladder": lad_rows, "ladder_line": lads[-1] if lads else "",
                         "per_stage_ms": v.get("per_stage_ms") or {},
                         "stage_util_pct": v.get("stage_util_pct") or {},
                         "wall_s": round(time.time() - t0, 1)})
            say("%2d  %s  %6.2f 帧/秒  %d 帧  批 %s  墙钟 %5.1fs  %s"
                % (i + 1, ph, fps, done, pick, time.time() - t0,
                   " ".join("B%d=%.1f" % (r["batch"], r["per_face_ms"]) for r in lad_rows)))
    finally:
        stop_server(proc)
    lads = ladder_lines(logpath)
    return {"hyst": hyst, "rows": rows, "ladders": lads, "log": str(logpath)}


def summarize(cfg):
    rows = [r for r in (cfg.get("rows") or []) if r.get("phase") == "done"]
    if not rows:
        return {"hyst": cfg.get("hyst"), "n": 0}
    f = [r["fps"] for r in rows]
    picks = [r["batch"] for r in rows]
    picks = [p if p is not None else -1 for p in picks]
    sw = sum(1 for a, b in zip(picks, picks[1:]) if a != b)
    counts = {}
    for p in picks:
        counts[str(p)] = counts.get(str(p), 0) + 1
    med = statistics.median(f)
    return {"hyst": cfg.get("hyst"), "n": len(rows),
            "fps": [round(v, 2) for v in f],
            "fps_median": round(med, 3), "fps_mean": round(statistics.fmean(f), 3),
            "fps_sd": round(statistics.pstdev(f), 3),
            "fps_min": round(min(f), 3), "fps_max": round(max(f), 3),
            "fps_range_pct": round(100.0 * (max(f) - min(f)) / med, 2),
            "fps_cv_pct": round(100.0 * statistics.pstdev(f) / statistics.fmean(f), 2),
            "picks": picks, "pick_counts": counts, "switches": sw,
            "jitter_pct": round(100.0 * sw / max(1, len(picks) - 1), 1),
            "batch_mean": round(statistics.fmean(picks), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="0,1")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--start", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=0, help=">0 = 钉死这个批大小（对照用）")
    ap.add_argument("--json", default=str(TEMP / "_pred_batch_hyst_ab.json"))
    args = ap.parse_args(argv)

    if not VIDEO.is_file():
        raise SystemExit("没有样片：%s" % VIDEO)
    prepare_sandbox()
    out = []
    for h in [x.strip() for x in args.configs.split(",") if x.strip()]:
        cfg = run_config(h, args.rounds, args.start, args.frames, batch=args.batch)
        cfg["summary"] = summarize(cfg)
        out.append(cfg)

    head("汇总")
    say("%-4s %-8s %-7s %-7s %-7s %-7s %-8s %-9s %s"
        % ("HYST", "帧/秒中位", "均值", "最小", "最大", "极差%", "cv%", "抖动率", "选择序列"))
    for cfg in out:
        s = cfg.get("summary") or {}
        if not s.get("n"):
            continue
        say("%-4s %-8.2f %-8.2f %-7.2f %-7.2f %-7.2f %-8.2f %-9.1f %s"
            % (cfg["hyst"], s["fps_median"], s["fps_mean"], s["fps_min"], s["fps_max"],
               s["fps_range_pct"], s["fps_cv_pct"], s["jitter_pct"],
               " / ".join(str(p) for p in s["picks"])))
    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    say("json：%s" % args.json)
    bad = [c for c in out if (c.get("summary") or {}).get("n", 0) < args.rounds]
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
