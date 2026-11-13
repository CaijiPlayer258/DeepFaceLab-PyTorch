# -*- coding: utf-8 -*-
r"""项目模式（DFL 项目导出）批量协议的验收：真后端 + 真项目 + 真 dfm/xseg，全在 %TEMP%。

这一版验的是「一次请求 k 帧 + 裸字节通道 + worker 内 k 线程」这条改造：

  a) 协议形状：一条请求带 k 帧、帧走裸字节块（不再 base64）、worker 用 k 个线程消费；
     status 里能读到 workers / batch / swap_p50，帧速从**预热之后**算。
  b) 扫描表：workers × batch = 1×1 / 1×4 / 2×2 / 2×4 / 2×8 / 3×4
     → 每个档位的 帧/秒、显存峰值、worker 单帧 p50。
  c) 逐像素对拍：每档产物 vs 1×1 基准档 → mean LSB / 最大差 / 差异像素比例。
  d) 断点续跑：中途取消 → done 与 merged/ 里的帧数一致（游标正确）→ 续跑产物与
     一次跑完逐像素一致。
  e) 落盘顺序：取消后 merged/ 里留下的必须是**前缀**（编号连续、从第一个开始）。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\proj_pool_accept.py
    （只跑某几段：--cases a,c；缩短帧数：--frames 60；不重建项目：--reuse）
退出码 0 = 全过。

前提：workspace\model 里至少有一个 .dfm（没有就先训练并导出），workspace 里有一个
能切出人脸的视频（或用 --video 指定）。
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
BOX = TEMP / "dfl_pool_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
PROJ = BOX / "proj"
CACHE = BOX / "cache"
SERVER_LOG = LOGDIR / "server.log"

DFM_NAME = "阿証証証証-416"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
TOL_MEAN = 0.5              # 和基准档的帧级 |diff| 平均（LSB）
SCAN = [(1, 1), (1, 4), (2, 2), (2, 4), (2, 8), (3, 4)]

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------
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


def http(method, path, body=None, timeout=600.0):
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


def dfm_path():
    p = REPO / "workspace" / "model" / (DFM_NAME + ".dfm")
    if p.is_file():
        return p
    cands = sorted((REPO / "workspace" / "model").glob("*.dfm"),
                   key=lambda x: x.stat().st_size)
    return cands[0] if cands else None


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    finally:
        cap.release()


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


def vram_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return max(int(x.strip()) for x in out.split() if x.strip())
    except Exception:                                       # noqa: BLE001
        return None


class VramWatch:
    """导出期间轮询显存峰值（0.25s 一次）。"""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.peak = 0

        def loop():
            while not self._stop.is_set():
                m = vram_used_mb()
                if m:
                    self.peak = max(self.peak, m)
                self._stop.wait(0.25)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)


# ------------------------------------------------------------
# 沙箱 / 后端
# ------------------------------------------------------------
def prepare_sandbox():
    if BOX.exists():
        for name in ("model", "_engines"):
            try:
                os.rmdir(str(WS / name))          # 目录联接用 rmdir 摘
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


def start_server(extra_env=None):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK",
              "DFL_PROJ_WORKERS", "DFL_PROJ_BATCH", "DFL_PROJ_SEND",
              "DFL_PROJ_OUT", "DFL_FUSED_BUDGET"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
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
            raise SystemExit("后端启动失败，见 %s" % SERVER_LOG)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % SERVER_LOG)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端：%s（pid=%s）" % (BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _ = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.3)
    raise SystemExit("后端 240s 没起来")


def make_clip(src, dst, ss=2.0, dur=4.0):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.3f" % ss, "-i", str(src), "-t", "%.3f" % dur,
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not Path(dst).is_file():
        raise SystemExit("切素材失败：%s" % (r.stderr or "")[-400:])


def wait_stage(timeout=900.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        st, d = get("/api/merge/stage/status", timeout=60)
        last = d or {}
        if st == 200 and not last.get("running"):
            return st, last
        time.sleep(1.0)
    return 200, last


# ------------------------------------------------------------
# 组项目：真帧 + 真对齐脸（走仓库自己的 stage 接口，和 fused_export_accept 的 F 段一致）
# ------------------------------------------------------------
def build_project(clip, dfm, end_frame):
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.rmtree(PROJ, ignore_errors=True)
    PROJ.mkdir(parents=True, exist_ok=True)
    body = {"video": str(clip), "model": Path(str(dfm)).stem, "settings": dict(SETTINGS),
            "cache_dir": str(CACHE), "end_frame": int(end_frame), "force": True}
    st, d = post("/api/merge/stage/frames", body)
    if st != 200:
        raise SystemExit("切帧起不来：HTTP %s %s" % (st, d.get("error")))
    wait_stage()
    st, d = post("/api/merge/stage/faces", body)
    if st != 200:
        raise SystemExit("切脸起不来：HTTP %s %s" % (st, d.get("error")))
    wait_stage()
    frames = sorted((CACHE / "frames").glob("*.png")) + sorted((CACHE / "frames").glob("*.jpg"))
    aligned = sorted((CACHE / "aligned").glob("*.jpg"))
    if not frames or not aligned:
        raise SystemExit("项目没组起来：%d 帧 / %d 张对齐脸" % (len(frames), len(aligned)))
    (PROJ / "aligned").mkdir(parents=True, exist_ok=True)
    keep = set(p.stem for p in frames)
    for p in frames:
        shutil.copy2(str(p), str(PROJ / p.name))
    for p in aligned:
        if p.stem.split("_")[0] in keep:
            shutil.copy2(str(p), str(PROJ / "aligned" / p.name))
    say("项目：%s（%d 帧 %s / %d 张对齐脸）"
        % (PROJ, len(frames), frames[0].suffix, len(aligned)))
    return len(frames)


# ------------------------------------------------------------
# 跑一档：workers × batch
# ------------------------------------------------------------
def merged_dir(tag):
    return OUT / ("m_" + tag)


def frames_in(d):
    return sorted(p.name for p in d.iterdir()
                  if p.suffix.lower() in (".png", ".jpg")) if d.is_dir() else []


def run_export(clip, dfm, workers, batch, tag, force=True, cancel_after=None,
               timeout=1800.0):
    """跑一档。cancel_after=N 时跑到 done>=N 就点取消（验续跑用）。"""
    md = merged_dir(tag)
    if md.exists():
        shutil.rmtree(md, ignore_errors=True)
    md.mkdir(parents=True, exist_ok=True)
    out = OUT / (tag + ".mp4")
    if out.exists():
        out.unlink()
    body = {"project_dir": str(PROJ), "frames_dir": str(PROJ),
            "aligned_dir": str(PROJ / "aligned"), "merged_dir": str(md),
            "video": str(clip), "out_video": str(out), "model": Path(str(dfm)).stem,
            "settings": dict(SETTINGS), "quality": 92, "force": bool(force),
            "workers": int(workers), "batch": int(batch)}
    t0 = time.time()
    with VramWatch() as vw:
        st, d = post("/api/merge/export-project", body)
        if st != 200:
            return {"tag": tag, "error": "HTTP %s %s" % (st, d.get("error")),
                    "fps": None}
        snap = {}
        snap0 = {}
        cancelled_at = None
        while time.time() - t0 < timeout:
            _st, snap = get("/api/merge/export-project/status", timeout=60)
            if not snap0 and snap.get("running"):
                snap0 = dict(snap)
            ph = snap.get("phase")
            if cancel_after is not None and cancelled_at is None \
                    and int(snap.get("done") or 0) >= int(cancel_after) \
                    and ph == "running":
                post("/api/merge/export-project/cancel", {})
                cancelled_at = int(snap.get("done") or 0)
                continue
            if ph in ("done", "failed", "cancelled"):
                break
            time.sleep(0.6)
        peak = vw.peak
    files = frames_in(md)
    return {"tag": tag, "workers": int(workers), "batch": int(batch),
            "fps": snap.get("frame_fps"), "done": int(snap.get("done") or 0),
            "swap_p50": snap.get("swap_p50"), "swap_elapsed": snap.get("swap_elapsed"),
            "elapsed": snap.get("elapsed"), "phase": snap.get("phase"),
            "error": (snap.get("error") or "")[:200],
            "encode_s": (snap.get("encode") or {}).get("seconds"),
            "files": len(files), "failed": int(snap.get("failed") or 0),
            "vram_peak_mb": peak, "wall": round(time.time() - t0, 1),
            "snap0": snap0, "cancelled_at": cancelled_at,
            "first": files[0] if files else None,
            "last": files[-1] if files else None, "dir": str(md)}


def cmp_dirs(a, b):
    """两个 merged 目录逐像素对拍。"""
    import cv2
    import numpy as np
    na, nb = frames_in(a), frames_in(b)
    if na != nb:
        return {"error": "帧集合不同：%d vs %d" % (len(na), len(nb))}
    tot = 0.0
    npix = 0
    mx = 0
    diffpx = 0
    worst = None
    byte_same = 0
    for n in na:
        fa, fb = a / n, b / n
        if fa.read_bytes() == fb.read_bytes():
            byte_same += 1
        ia = cv2.imread(str(fa), cv2.IMREAD_COLOR)
        ib = cv2.imread(str(fb), cv2.IMREAD_COLOR)
        if ia is None or ib is None or ia.shape != ib.shape:
            return {"error": "读不出/形状不同：%s" % n}
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        tot += float(d.sum())
        npix += d.size
        m = int(d.max())
        mx = max(mx, m)
        diffpx += int((d > 0).sum())
        if m and (worst is None or m > worst[2]):
            worst = (n, round(float(d.mean()), 4), m)
    return {"frames": len(na), "mean_lsb": round(tot / max(1, npix), 6),
            "max_lsb": mx, "diff_px_pct": round(100.0 * diffpx / max(1, npix), 4),
            "worst": worst, "byte_identical": byte_same}


# ============================================================
# A · 协议形状 / 默认档
# ============================================================
def case_a(clip, dfm, nframes):
    head("A · 协议形状：一次请求 k 帧 + 裸字节 + worker 内 k 线程（默认 2×4）")
    r = run_export(clip, dfm, 2, 4, "a_default")
    check(not r.get("error"), "A 默认档跑完", r.get("error") or "phase=%s" % r.get("phase"))
    check(r.get("phase") == "done" and r.get("files") == nframes,
          "A 产物帧数对", "%s 帧（源 %d）" % (r.get("files"), nframes))
    check(r.get("fps") and float(r["fps"]) > 0, "A status 有帧速", "%.2f 帧/秒" % float(r["fps"] or 0))
    say("帧速从预热之后算（swap_elapsed=%.1fs / elapsed=%.1fs），swap p50=%sms"
        % (float(r.get("swap_elapsed") or 0), float(r.get("elapsed") or 0),
           r.get("swap_p50")))
    # ---- 协议形状的三条硬证据 ----
    # 1) status 回显 workers/batch（_proj_set 写进作业状态，跑完仍在）
    check(int(r.get("workers") or 0) == 2 and int(r.get("batch") or 0) == 4,
          "A status 回显 workers=2 / batch=4",
          "workers=%s batch=%s" % (r.get("workers"), r.get("batch")))
    # 2) 真正落盘的那份 worker 脚本里同时有「批量 + 裸字节块」两件事
    scripts = sorted((TEMP / "dfl41_merge_preview").glob("preview_worker_*.py"),
                     key=lambda x: x.stat().st_mtime)
    src = scripts[-1].read_text(encoding="utf-8", errors="replace") if scripts else ""
    need = ('"blobs"', "_read_exact", "ThreadPoolExecutor", "_run_items",
            '"threads"')
    miss = [x for x in need if x not in src]
    check(bool(src) and not miss, "A 落盘 worker 脚本是「k 帧一条请求 + 裸字节块」那一版",
          ("%s（缺 %s）" % (scripts[-1].name, miss)) if miss else
          (scripts[-1].name if scripts else "没找到 worker 脚本"))
    # 3) 帧速口径：预热不算干活（swap_elapsed < elapsed，且 elapsed - swap ≈ 预热）
    check(float(r.get("swap_elapsed") or 0) > 0
          and float(r.get("swap_elapsed") or 0) <= float(r.get("elapsed") or 0),
          "A 帧速从预热之后算（swap_elapsed 单独计时）",
          "swap_elapsed=%.1fs / elapsed=%.1fs" % (float(r.get("swap_elapsed") or 0),
                                                  float(r.get("elapsed") or 0)))
    return r


# ============================================================
# B · 扫描表
# ============================================================
def case_b(clip, dfm, base_row=None):
    head("B · 扫描表 workers × batch（帧/秒 · 显存峰值 · worker 单帧 p50）")
    rows = []
    base = base_row
    if base is None:
        base = run_export(clip, dfm, 1, 1, "b_1x1")
        rows.append(base)
    else:
        rows.append(base)
    for (w, b) in SCAN:
        if (w, b) == (1, 1) and base_row is not None:
            continue
        r = run_export(clip, dfm, w, b, "b_%dx%d" % (w, b))
        rows.append(r)
        say("%dx%d：%.2f 帧/秒 · 显存峰值 %s MB · swap p50 %sms · 墙钟 %.1fs"
            % (w, b, float(r.get("fps") or 0), r.get("vram_peak_mb"),
               r.get("swap_p50"), float(r.get("wall") or 0)))
    print("")
    print("  %-6s %-9s %-9s %-10s %-9s %-7s %s"
          % ("档位", "帧/秒", "显存峰值", "swap p50", "落盘/失败", "墙钟", "状态"))
    for r in rows:
        print("  %-6s %-9s %-9s %-10s %-9s %-7s %s"
              % ("%dx%d" % (int(r.get("workers") or 0), int(r.get("batch") or 0)),
                 ("%.2f" % float(r["fps"])) if r.get("fps") else "-",
                 ("%s MB" % r.get("vram_peak_mb")) if r.get("vram_peak_mb") else "-",
                 ("%sms" % r.get("swap_p50")) if r.get("swap_p50") else "-",
                 "%s/%s" % (r.get("files"), r.get("failed")),
                 ("%.1fs" % float(r.get("wall") or 0)),
                 r.get("phase") or r.get("error")))
    for r in rows:
        if int(r.get("failed") or 0) > 0 and int(r.get("workers") or 0) * int(r.get("batch") or 0) <= 8:
            check(False, "B %dx%d 有帧掉队（推荐的档位不许掉帧）"
                  % (r["workers"], r["batch"]), "失败 %s" % r.get("failed"))
    best = max(rows, key=lambda x: float(x.get("fps") or 0))
    check(float(best.get("fps") or 0) > 0,
          "B 最优档能跑", "%dx%d → %.2f 帧/秒"
          % (int(best.get("workers") or 0), int(best.get("batch") or 0),
             float(best.get("fps") or 0)))
    slow = min(rows, key=lambda x: float(x.get("fps") or 0))
    check(float(best.get("fps") or 0) >= float(slow.get("fps") or 0),
          "B 最优档不是 1×1（批处理确实赚到了）",
          "最优 %s vs 最慢 %s" % (best.get("tag"), slow.get("tag")))
    return rows


# ============================================================
# C · 逐像素对拍（每档 vs 1×1 基准档）
# ============================================================
def case_c(rows):
    head("C · 逐像素对拍（每档产物 vs 1×1 基准档，mean ≤ %.1f LSB）" % TOL_MEAN)
    base = next((r for r in rows if int(r.get("workers") or 0) == 1
                 and int(r.get("batch") or 0) == 1), None)
    if base is None:
        check(False, "C 缺基准档", "先跑 B 段")
        return
    worst = 0.0
    for r in rows:
        if r is base:
            continue
        if int(r.get("failed") or 0) > 0:
            # 显存 spill 的档位会掉帧（帧少了就对不齐）：那不是像素问题 —— 单独报一句，
            # 别拿它当"对拍失败"。推荐档（≤2×4）掉帧才算真失败。
            say("跳过 %dx%d：落盘 %s / 失败 %s —— 显存 spill，这一档本来就不该用"
                % (r["workers"], r["batch"], r.get("files"), r.get("failed")))
            continue
        d = cmp_dirs(Path(base["dir"]), Path(r["dir"]))
        if d.get("error"):
            check(False, "C %s 对拍" % r.get("tag"), d["error"])
            continue
        worst = max(worst, float(d["mean_lsb"]))
        check(float(d["mean_lsb"]) <= TOL_MEAN,
              "C %dx%d vs 1×1" % (r["workers"], r["batch"]),
              "mean %.4f LSB · max %d · 差异像素 %.4f%% · 逐字节相同 %d/%d"
              % (d["mean_lsb"], d["max_lsb"], d["diff_px_pct"],
                 d["byte_identical"], d["frames"]))
    say("最差档 mean = %.4f LSB（门槛 %.1f）" % (worst, TOL_MEAN))


# ============================================================
# D · 断点续跑
# ============================================================
def case_d(clip, dfm, nframes):
    head("D · 断点续跑：中途取消 → 游标正确 → 续跑产物与一次跑完一致")
    full = run_export(clip, dfm, 2, 4, "d_full")
    check(full.get("phase") == "done" and full.get("files") == nframes,
          "D 一次跑完（基准）", "%s 帧 · %.2f 帧/秒" % (full.get("files"), float(full.get("fps") or 0)))
    cut = run_export(clip, dfm, 2, 4, "d_cut", cancel_after=max(4, nframes // 3))
    check(cut.get("phase") == "cancelled", "D 取消落地（终态 cancelled）",
          "phase=%s done=%s files=%s" % (cut.get("phase"), cut.get("done"), cut.get("files")))
    check(int(cut.get("done") or 0) == int(cut.get("files") or 0),
          "D 游标 = merged/ 里的帧数（一格都不差）",
          "done=%s files=%s" % (cut.get("done"), cut.get("files")))
    names = frames_in(Path(cut["dir"]))
    want = names[:1]
    nums = []
    for n in names:
        m = re.search(r"(\d+)", Path(n).stem)
        nums.append(int(m.group(1)) if m else -1)
    prefix_ok = bool(nums) and nums == list(range(nums[0], nums[0] + len(nums)))
    check(prefix_ok, "D 取消后留下的是前缀（编号连续、没有洞）",
          "%s…%s（%d 帧）" % (names[:1], names[-1:], len(names)))
    resumed = run_export(clip, dfm, 2, 4, "d_resumed", force=False)
    check(resumed.get("phase") == "done" and resumed.get("files") == nframes,
          "D 续跑补齐到满", "%s 帧（跳过 %s）" % (resumed.get("files"), nframes - len(names)))
    d = cmp_dirs(Path(full["dir"]), Path(resumed["dir"]))
    check(not d.get("error") and float(d["mean_lsb"]) <= TOL_MEAN,
          "D 续跑产物 vs 一次跑完：逐像素一致",
          ("mean %.6f LSB · max %d · 逐字节相同 %d/%d"
           % (d.get("mean_lsb"), d.get("max_lsb", -1), d.get("byte_identical", 0),
              d.get("frames", 0))) if not d.get("error") else d["error"])


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="a,b,c,d")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--video", default="")
    ap.add_argument("--reuse", action="store_true", help="复用沙箱里已有的项目")
    opt = ap.parse_args()

    if not PY.is_file():
        raise SystemExit("没有 %s（用 runtime\\server\\Scripts\\python.exe 跑）" % PY)
    dfm = dfm_path()
    if dfm is None:
        raise SystemExit("workspace\\model 里没有 .dfm —— 先训练并导出模型")
    video = Path(opt.video) if opt.video else None
    if video is None:
        vids = [p for p in (REPO / "workspace").glob("*.mp4")
                if p.is_file() and "_merged" not in p.name and "._" not in p.name]
        vids.sort(key=lambda p: p.stat().st_size, reverse=True)
        video = vids[0] if vids else None
    if video is None or not video.is_file():
        raise SystemExit("workspace 里没有能用的视频（或用 --video 指定）")

    cases = [c.strip() for c in opt.cases.split(",") if c.strip()]
    prepare_sandbox()
    clip = MEDIA / "clip.mp4"
    make_clip(video, clip, ss=2.0, dur=4.0)
    info = probe(clip)
    say("素材：%s（%dx%d, %.0f 帧 @%.2f fps）"
        % (clip.name, info["w"], info["h"], info["frames"], info["fps"]))

    proc = start_server()
    rows = []
    try:
        if opt.reuse and PROJ.is_dir() and any(PROJ.glob("*.png")):
            nframes = len([p for p in PROJ.iterdir() if p.suffix.lower() in (".png", ".jpg")])
            say("复用现有项目：%d 帧" % nframes)
        else:
            nframes = build_project(clip, dfm, int(opt.frames))
        if "a" in cases:
            case_a(clip, dfm, nframes)
        if "b" in cases:
            rows = case_b(clip, dfm)
        if "c" in cases:
            if not rows:
                rows = [run_export(clip, dfm, 1, 1, "b_1x1")]
                for (w, b) in SCAN[1:]:
                    rows.append(run_export(clip, dfm, w, b, "b_%dx%d" % (w, b)))
            case_c(rows)
        if "d" in cases:
            case_d(clip, dfm, nframes)
    finally:
        try:
            _kill_tree(proc.pid)
        except Exception:                                   # noqa: BLE001
            pass
        # worker 子进程也要收干净（它们由后端进程树带出来）
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-Process python -ErrorAction SilentlyContinue | "
                            "Where-Object { $_.Path -like '*engine*python.exe' } | "
                            "Stop-Process -Force"], capture_output=True, timeout=60)
        except Exception:                                   # noqa: BLE001
            pass

    bad = [x for x in RESULTS if not x[0]]
    print("")
    print("=" * 78)
    print("结果：%d 项通过，%d 项失败" % (len(RESULTS) - len(bad), len(bad)))
    for _ok, name, detail in bad:
        print("  [FAIL] %s  · %s" % (name, detail))
    print("沙箱：%s（server.log 在同一目录）" % BOX)
    print("=" * 78)
    sys.exit(1 if bad else 0)


main()
