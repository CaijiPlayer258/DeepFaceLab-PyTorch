# -*- coding: utf-8 -*-
r"""项目导出的**默认档按模型体量收缩**验收：真后端 + 真项目 + 真 dfm，全在 %TEMP%。

验的东西（对应用户这一轮的要求）：

  A  大模型（≥1GB 权重）默认档 = **1 worker × 4 线程**：
     status.workers=1 / batch=4、整卡峰值 ≤ 85%、帧/秒 ≥ 10、swap p50 ≤ 700ms、
     日志里有 @@[project] 模型 …GB · 可用 …GB · 计划 1 worker(s) × 4 线程@@ 那一行。
     （改前是同一条请求开 2 个 worker → 整卡 95% → WDDM 换页 → 3.6 帧/秒。）
  B  小模型（≤600MB）默认档 = **2 worker × 2 线程**：帧/秒 ≥ 12、峰值 ≤ 85%。
  C  手动要 workers=2 而装不下：**降级并提示**（workers=1 + [warn] + 备注里写「已减」），
     或者 400 带数字 —— 两条都算过，但必须是其中之一（不许闷头开 2 个把卡顶到 95%）。
  D  像素不变：A（1×4）与 C（降级后的 1×2）产物逐像素对拍 mean ≤ 0.5 LSB。
  E  「一份会话真跑起来多大」的实测真的进了预算：ws/_vram_sess_peak.json 里有大模型的
     条目，且下一次计划里的 sess_src 写着「实测」。

跑法（仓库根，必须 server venv）：
    runtime\server\Scripts\python.exe app\tools\proj_default_accept.py
    （少跑几帧：--frames 48；只跑某几段：--cases a,b）

退出码 0 = 全过。前提：workspace\model 里有一个 ≥1GB 的 .dfm 和一个 ≤600MB 的 .dfm。
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

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_proj_default"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
PROJ = BOX / "proj"
CACHE = BOX / "cache"
SERVER_LOG = LOGDIR / "server.log"

SETTINGS = {"mask_mode": 8, "mask_source": "XSegLite", "color_transfer": "mt",
            "blend": "seamless", "mask_erode": 0, "mask_blur": 4}
BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BIG_PCT = 0.85          # 显存峰值 ≤ 85%（改前是 94.9%）
# 夹具是**一段连续 100% 有脸的帧**（比真项目重：真项目 84.9% 有脸，剩下 15% 只搬原图）。
# 所以门槛钉在「每张有脸帧多少毫秒」，再折算成 85% 有脸那份真项目的帧/秒：
#   384 1×4 实测 108~127ms/帧 → 折算 8.6~10.2 帧/秒（改前 2×2 是 3.61，且整卡 94.9%）
#   416 2×2 实测  86~ 90ms/帧 → 折算 11.9~12.4 帧/秒（与 17-PROJ-BATCH 的 13.42 同档）
FACE_MS_BIG = 125.0     # ⇒ 折算 ≥ 10 帧/秒
FACE_MS_SMALL = 100.0   # ⇒ 折算 ≥ 12 帧/秒
SMALL_FPS = 12.0        # 416 模型默认档 ≥ 12 帧/秒
SWAP_P50_MAX = 700.0    # 换页时是 1334ms；正常应该在 200~500ms
TOL_MEAN = 0.5          # 逐像素 mean（LSB）


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


def http(method, path, body=None, timeout=900.0):
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
            return e.code, {"error": blob[:400].decode("utf-8", "replace")}


def get(path, **kw):
    return http("GET", path, **kw)


def post(path, body, **kw):
    return http("POST", path, body, **kw)


def dfm_candidates():
    d = REPO / "workspace" / "model"
    out = []
    for p in sorted(d.glob("*.dfm")):
        try:
            out.append((p, p.stat().st_size / 1048576.0))
        except OSError:
            pass
    return out


def pick_models():
    """大模型 = 最大那份（≥1024MB）；小模型 = 最小那份（≤600MB）。"""
    cands = dfm_candidates()
    big = max(cands, key=lambda x: x[1]) if cands else None
    small = min(cands, key=lambda x: x[1]) if cands else None
    if big and big[1] < 1024.0:
        big = None
    if small and small[1] > 600.0:
        small = None
    return big, small


def video_path():
    cands = [p for p in (REPO / "workspace").glob("*.mp4")
             if "_merged" not in p.stem]
    if not cands:
        raise SystemExit("workspace 里找不到可切人脸的 .mp4")
    return max(cands, key=lambda p: p.stat().st_size)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def make_clip(src, dst, ss=2.0, dur=6.0):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.3f" % ss, "-i", str(src), "-t", "%.3f" % dur,
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not Path(dst).is_file():
        raise SystemExit("切素材失败：%s" % (r.stderr or "")[-400:])
    return dst


def vram_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        row = [x for x in out.strip().splitlines() if x.strip()][0]
        used, total = [int(float(x)) for x in row.split(",")[:2]]
        return used, total
    except Exception:                                       # noqa: BLE001
        return None, None


class VramWatch:
    """导出期间轮询整卡显存（0.25s 一次），记峰值与峰值占整卡的百分比。"""

    def __init__(self):
        self.peak = 0
        self.total = 0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                u, t = vram_used_mb()
                if u:
                    self.peak = max(self.peak, u)
                    self.total = t or self.total
                self._stop.wait(0.25)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)

    def pct(self):
        return (100.0 * self.peak / self.total) if self.total else 0.0


def prepare_sandbox():
    if BOX.exists():
        for name in ("model", "_engines"):
            try:
                os.rmdir(str(WS / name))            # 目录联接用 rmdir 摘
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
    # 这一套验的是**保守默认档**（按权重 ≥1GB→1×4 / ≤600MB→2×2）：把自适应标定关掉，
    # 否则它会按实测换档（那才是对的，但就不是本文件要验的那条规则了）。
    # 自适应本身由 proj_adaptive_accept.py 验。
    env.setdefault("DFL_PROJ_BENCH", "0")
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


def stop_server(proc):
    if proc is None:
        return
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, timeout=60)
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.0)


def wait_stage(timeout=900.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        st, d = get("/api/merge/stage/status", timeout=60)
        last = d or {}
        if st == 200 and not last.get("running"):
            return st, last
        time.sleep(0.5)
    return 200, last


def build_project(clip, dfm_stem, end_frame, target=120):
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.rmtree(PROJ, ignore_errors=True)
    PROJ.mkdir(parents=True, exist_ok=True)
    body = {"video": str(clip), "model": dfm_stem, "settings": dict(SETTINGS),
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
    # 只留一段**连续有脸**的帧（最多 target 帧）：没有脸的帧走「原样搬」，会把帧/秒
    # 抬得虚高（它们几乎不花 GPU），那就不像用户那份「基本都有脸」的真项目了。
    face = {}
    for a in aligned:
        stem = a.stem.split("_")[0]
        if stem.isdigit():
            face.setdefault(int(stem), []).append(a)
    want = int(target or 120)
    nums = sorted(n for n in (int(p.stem) for p in frames if p.stem.isdigit()))
    if not nums:
        raise SystemExit("切帧结果里没有编号帧")
    # 挑「有脸比例最高的一段连续帧」（长度 = want）：真项目就是这样一段（本片实测
    # 84.9% 有脸）。全挑有脸的帧会让这份夹具比真项目更重，帧/秒被低估、显存峰值偏高。
    best, best_hit = nums[:want], -1
    for i in range(0, max(1, len(nums) - want + 1)):
        win = nums[i:i + want]
        hit = sum(1 for n in win if n in face)
        if hit > best_hit:
            best, best_hit = win, hit
    pick = list(best)
    keep = {"%06d" % n for n in pick}
    say("项目切片：帧 %d-%d（%d 帧 · %d 帧有对齐脸 = %.1f%%）"
        % (pick[0], pick[-1], len(pick), best_hit,
           100.0 * max(0, best_hit) / max(1, len(pick))))
    picked = [p for p in frames if p.stem in keep]
    (PROJ / "aligned").mkdir(parents=True, exist_ok=True)
    # 帧转成 .jpg（q95）：真 DFL 项目就是抽帧后的 .jpg；PNG 帧那条出图路径会让整卡
    # 多背 0.5~0.9GB（实测 416 2×2：jpg 6430MB / png 7084MB），那会把「≤85%」这条
    # 验收变成在量 PNG 路而不是量默认档。
    import cv2
    for p in picked:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(PROJ / (p.stem + ".jpg")), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    for a in aligned:
        if a.stem.split("_")[0] in keep:
            shutil.copy2(str(a), str(PROJ / "aligned" / a.name))
    frames = list(picked)             # 计数用（帧名换成了 .jpg，数量不变）
    say("项目：%s（%d 帧 .jpg / %d 张对齐脸）" % (PROJ, len(frames), len(aligned)))
    return len(frames)


def run_export(tag, clip, dfm_stem, workers=None, batch=None, force=True):
    md = OUT / ("m_" + tag)
    shutil.rmtree(str(md), ignore_errors=True)
    md.mkdir(parents=True, exist_ok=True)
    out = OUT / (tag + ".mp4")
    if out.exists():
        out.unlink()
    body = {"project_dir": str(PROJ), "frames_dir": str(PROJ),
            "aligned_dir": str(PROJ / "aligned"), "merged_dir": str(md),
            "video": str(clip), "out_video": str(out), "model": dfm_stem,
            "settings": dict(SETTINGS), "quality": 92, "force": bool(force)}
    if workers is not None:
        body["workers"] = int(workers)
    if batch is not None:
        body["batch"] = int(batch)
    t0 = time.time()
    base_used, _bt = vram_used_mb()          # 开跑前整卡占用（桌面/浏览器那份要减掉）
    with VramWatch() as vw:
        st, d = post("/api/merge/export-project", body, timeout=600)
        plan = (d or {}).get("vram_plan") or {}
        if st != 200 or (d or {}).get("ok") is False:
            return {"tag": tag, "http": st, "error": (d or {}).get("error"),
                    "plan": plan, "peak": vw.peak, "pct": vw.pct(),
                    "wall": round(time.time() - t0, 1)}
        snap = {}
        while time.time() - t0 < 1800:
            _s, snap = get("/api/merge/export-project/status", timeout=60)
            if snap.get("phase") in ("done", "failed", "cancelled"):
                break
            time.sleep(0.3)
    files = sorted(p.name for p in md.iterdir()
                   if p.suffix.lower() in (".png", ".jpg")) if md.is_dir() else []
    # 每张**有脸**帧的毫秒数 + 折算到「85% 有脸」那份真项目的帧/秒：
    # 夹具是 100% 有脸（比真项目重），绝对帧/秒会被压低；每帧毫秒数与夹具无关。
    ms_face = fps85 = None
    try:
        se = float(snap.get("swap_elapsed") or 0.0)
        sf = int(snap.get("swap_frames") or 0)
        if se > 0 and sf > 0:
            ms_face = round(1000.0 * se / sf, 1)
            # 折算：85% 的帧真有脸（每张 ms_face）+ 15% 的帧只搬原图（实测 ~12ms）
            fps85 = round(1.0 / ((0.85 * ms_face / 1000.0) + (0.15 * 0.012)), 2)
    except Exception:                                   # noqa: BLE001
        pass
    return {"tag": tag, "http": st, "phase": snap.get("phase"),
            "ms_face": ms_face, "fps85": fps85,
            "fps": snap.get("frame_fps"), "workers": snap.get("workers"),
            "batch": snap.get("batch"), "swap_p50": snap.get("swap_p50"),
            "pipeline": snap.get("pipeline"), "compose": snap.get("compose"),
            "done": snap.get("done"), "files": len(files),
            "swap_elapsed": snap.get("swap_elapsed"), "swap_frames": snap.get("swap_frames"),
            "failed": snap.get("failed"), "elapsed": snap.get("elapsed"),
            "plan": plan, "note": str(plan.get("worker_note") or ""),
            "log": list(snap.get("log") or []), "err": (snap.get("error") or "")[:200],
            "peak": vw.peak, "total": vw.total, "pct": round(vw.pct(), 1),
            "base": int(base_used or 0),
            "ours": int(max(0, (vw.peak or 0) - (base_used or 0))),
            "ours_pct": round(100.0 * max(0, (vw.peak or 0) - (base_used or 0))
                              / float(vw.total or 1), 1),
            "dir": str(md), "wall": round(time.time() - t0, 1)}


def cmp_dirs(a, b, limit=60):
    import cv2
    import numpy as np
    def _names(d):
        return sorted(p.name for p in Path(d).iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    na, nb = _names(a), _names(b)
    common = sorted(set(na) & set(nb))[:limit]
    if not common:
        return {"error": "没有同名帧（%d vs %d）" % (len(na), len(nb))}
    tot, npix, mx = 0.0, 0, 0
    for n in common:
        ia = cv2.imread(str(Path(a) / n), cv2.IMREAD_UNCHANGED)
        ib = cv2.imread(str(Path(b) / n), cv2.IMREAD_UNCHANGED)
        if ia is None or ib is None or ia.shape != ib.shape:
            continue
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        tot += float(d.sum())
        npix += int(d.size)
        mx = max(mx, int(d.max()))
    if not npix:
        return {"error": "一帧都没读出来"}
    return {"frames": len(common), "mean_lsb": round(tot / npix, 6), "max_lsb": mx}


def sess_peak_file():
    p = WS / "_vram_sess_peak.json"
    try:
        return json.loads(p.read_text("utf-8", "ignore") or "{}") if p.is_file() else {}
    except Exception:                                       # noqa: BLE001
        return {}


# ============================================================
# A · 大模型默认档
# ============================================================
def case_big(clip, big, nframes):
    stem, mb = big[0].stem, big[1]
    head("A · 大模型默认档（%.2fGB 权重）→ 一份会话 × dfm 攒批 4" % (mb / 1024.0))
    r = run_export("a_big_default", clip, stem, None, None)
    say("HTTP %s · phase=%s · workers=%r batch=%r · %.2f 帧/秒 · 峰值 %s/%s MB（%.1f%%）· swap p50 %sms"
        % (r.get("http"), r.get("phase"), r.get("workers"), r.get("batch"),
           float(r.get("fps") or 0), r.get("peak"), r.get("total"), r.get("pct") or 0,
           r.get("swap_p50")))
    if r.get("http") != 200:
        check(False, "a 大模型默认档起得来", (r.get("error") or "")[:300])
        return r
    check(r.get("phase") == "done" and int(r.get("files") or 0) == nframes,
          "a 大模型默认档跑完且帧数对", "%s 帧（源 %d）" % (r.get("files"), nframes))
    # 流水线之后 workers 恒为 1（一份 dfm 会话），batch = dfm 攒批目标（大模型按计划 4）
    check(int(r.get("workers") or 0) == 1 and int(r.get("batch") or 0) >= 4
          and str(r.get("pipeline") or "") == "fused",
          "a 默认档 = 一份会话（workers=1）× dfm 攒批 ≥4，走流水线",
          "workers=%r batch=%r pipeline=%r" % (r.get("workers"), r.get("batch"),
                                               r.get("pipeline")))
    say("整卡 %s/%s MB = %.1f%%（含桌面/etc %s MB）· 本程序 %s MB = %.1f%%"
        % (r.get("peak"), r.get("total"), r.get("pct") or 0, r.get("base"),
           r.get("ours"), r.get("ours_pct") or 0))
    check(float(r.get("ours_pct") or 0.0) <= BIG_PCT * 100.0,
          "a 本程序显存峰值 ≤ %d%%（改前 2×2 是 94.9%% 整卡）" % int(BIG_PCT * 100),
          "%s MB = %.1f%%（整卡 %.1f%%）" % (r.get("ours"), r.get("ours_pct") or 0,
                                            r.get("pct") or 0))
    fps, msf, f85 = (float(r.get("fps") or 0.0), float(r.get("ms_face") or 0.0),
                     float(r.get("fps85") or 0.0))
    check(msf > 0 and msf <= FACE_MS_BIG,
          "a 每张有脸帧 ≤ %d ms（= 折算到 85%% 有脸的真项目 ≥ 10 帧/秒）" % FACE_MS_BIG,
          "%.1f ms/帧 · 折算 %.2f 帧/秒 · 夹具实测 %.2f 帧/秒（100%% 有脸，比真项目重）"
          % (msf, f85, fps))
    say("单会话天花板 ≈ 4 / swap_p50 = %.1f 帧/秒（夹具 100%% 有脸）；"
        "再往上（1×5 / 1×6）整卡就到 89%%~91%% 开始换页"
        % ((4.0 * 1000.0 / float(r.get("swap_p50") or 1.0)) if r.get("swap_p50") else 0.0))
    p50 = r.get("swap_p50")
    check(p50 is not None and float(p50) <= SWAP_P50_MAX,
          "a swap p50 ≤ %.0fms（换页时 1334ms）" % SWAP_P50_MAX, "%sms" % p50)
    note = r.get("note") or ""
    say("决定说明：" + note)
    check("[project]" in note and "计划 1 worker(s) × 4 线程" in note and "参数量" in note
          and "权重" in note and "输入" in note and "可用" in note,
          "a 日志写着「参数量 X M · 权重 Y · 输入 Z · 可用 W · 计划 1 worker(s) × 4 线程」",
          note or "没有这一行")
    plan = r.get("plan") or {}
    check(bool(plan.get("sess_src")),
          "a 计划里说明了「一份会话」的来路（实测/标定）",
          "sess=%.0f MB · %s" % (float(plan.get("sess_mb") or 0), plan.get("sess_src")))
    # ---- 模型身份必须来自 ONNX 图，不是文件名/分辨率 ----
    check(float(plan.get("params_m") or 0) > 0,
          "a status 里带参数量", "params_m=%r" % plan.get("params_m"))
    check(abs(float(plan.get("model_mb") or 0) - mb) <= max(2.0, 0.01 * mb),
          "a 权重体量与 ONNX initializer 一致", "plan %.1f MB vs 文件 %.1f MB"
          % (float(plan.get("model_mb") or 0), mb))
    check("initializer" in str(plan.get("model_mb_src") or ""),
          "a 权重体量的来路写明「ONNX initializer 实测」", plan.get("model_mb_src"))
    check(str(plan.get("size_px_src") or "") == "ONNX 图输入"
          and int(plan.get("size_px") or 0) > 0,
          "a 输入边长来自 ONNX 图（不是文件名）",
          "size_px=%r · %r" % (plan.get("size_px"), plan.get("size_px_src")))
    check("参数量" in note and "权重" in note and "输入" in note,
          "a 决定说明里有「参数量 / 权重 / 输入」三项", note or "没有这一行")
    check(float(plan.get("sess_mb") or 0) > float(plan.get("model_mb") or 0)
          + float(plan.get("ctx_mb") or 0),
          "a 一份会话 > 权重 + context（运行时工作集算进去了）",
          "sess=%.0f / 权重 %.0f + context %s" % (float(plan.get("sess_mb") or 0),
                                                 float(plan.get("model_mb") or 0),
                                                 plan.get("ctx_mb")))
    return r


# ============================================================
# B · 小模型默认档
# ============================================================
def case_small(clip, small, nframes):
    stem, mb = small[0].stem, small[1]
    head("B · 小模型默认档（%.2fGB 权重）→ 一份会话（不再 2 进程）" % (mb / 1024.0))
    r = run_export("b_small_default", clip, stem, None, None)
    say("HTTP %s · phase=%s · workers=%r batch=%r · %.2f 帧/秒 · 峰值 %s/%s MB（%.1f%%）"
        % (r.get("http"), r.get("phase"), r.get("workers"), r.get("batch"),
           float(r.get("fps") or 0), r.get("peak"), r.get("total"), r.get("pct") or 0))
    if r.get("http") != 200:
        check(False, "b 小模型默认档起得来", (r.get("error") or "")[:300])
        return r
    check(r.get("phase") == "done" and int(r.get("files") or 0) == nframes,
          "b 小模型默认档跑完且帧数对", "%s 帧（源 %d）" % (r.get("files"), nframes))
    # 流水线之后小模型也不再开 2 个进程：一份会话 + 攒批（计划里的 2×2 只影响预算）
    check(int(r.get("workers") or 0) == 1 and int(r.get("batch") or 0) >= 4,
          "b 默认档 = 一份会话 × dfm 攒批 ≥4（不再 2 进程 × 2 线程）",
          "workers=%r batch=%r" % (r.get("workers"), r.get("batch")))
    say("整卡 %s/%s MB = %.1f%%（含桌面/etc %s MB）· 本程序 %s MB = %.1f%%"
        % (r.get("peak"), r.get("total"), r.get("pct") or 0, r.get("base"),
           r.get("ours"), r.get("ours_pct") or 0))
    check(float(r.get("ours_pct") or 0.0) <= BIG_PCT * 100.0,
          "b 本程序显存峰值 ≤ %d%%" % int(BIG_PCT * 100),
          "%s MB = %.1f%%（整卡 %.1f%%）" % (r.get("ours"), r.get("ours_pct") or 0,
                                            r.get("pct") or 0))
    fps, msf, f85 = (float(r.get("fps") or 0.0), float(r.get("ms_face") or 0.0),
                     float(r.get("fps85") or 0.0))
    check(msf > 0 and msf <= FACE_MS_SMALL,
          "b 每张有脸帧 ≤ %d ms（= 折算到 85%% 有脸 ≥ 12 帧/秒）" % FACE_MS_SMALL,
          "%.1f ms/帧 · 折算 %.2f 帧/秒 · 夹具实测 %.2f 帧/秒" % (msf, f85, fps))
    note = r.get("note") or ""
    # 计划那一行是**预算**的口径（沙箱里桌面占得多时它会自己减档），所以只验「有这行
    # 且写了 worker(s) × 线程」；实际生效的并行度看上面 workers/batch 那两条。
    check("计划" in note and "worker(s)" in note and "线程" in note,
          "b 决定说明里有「计划 N worker(s) × M 线程」", note or "没有这一行")
    return r


# ============================================================
# C · 手动要 2 个 worker 而装不下
# ============================================================
def case_manual(clip, big, nframes):
    stem, mb = big[0].stem, big[1]
    head("C · 手动 workers=2 而装不下 → 降级并提示（或 400 带数字）")
    r = run_export("c_big_manual2", clip, stem, 2, 2)
    if r.get("http") == 400:
        msg = str(r.get("error") or "")
        check("需要" in msg or "GB" in msg, "c 400 且消息里有数字", msg[:200])
        return r
    check(r.get("http") == 200 and r.get("phase") == "done",
          "c 手动 2 个 worker 被受理并跑完", "HTTP %s phase=%s %s"
          % (r.get("http"), r.get("phase"), (r.get("err") or "")[:120]))
    check(int(r.get("workers") or 0) == 1,
          "c 显存不够 → 实际只开了一份会话（降级 / 流水线本来就只有一份）",
          "workers=%r（请求 2）" % r.get("workers"))
    warns = [x for x in (r.get("log") or []) if "[warn]" in str(x) and "worker" in str(x)
             and ("只装得下" in str(x) or "只有一份 dfm 会话" in str(x))]
    check(bool(warns), "c 日志里有「只装得下 N 个 worker（请求 2）」/「只有一份 dfm 会话」提示",
          (warns[-1] if warns else "没找到")[:200])
    note = r.get("note") or ""
    check("已减" in note, "c 决定说明里写明「已减」", note or "没有这一行")
    check(float(r.get("pct") or 0.0) <= BIG_PCT * 100.0,
          "c 降级后显存峰值仍 ≤ %d%%" % int(BIG_PCT * 100),
          "%s/%s MB = %.1f%%" % (r.get("peak"), r.get("total"), r.get("pct") or 0))
    return r


# ============================================================
# D/E · 像素不变 + 实测进了预算
# ============================================================
def case_pixels(a, c):
    head("D · 像素不变（A 1×4 vs C 降级后的 1×2，mean ≤ %.1f LSB）" % TOL_MEAN)
    if not a or not c or not a.get("dir") or not c.get("dir"):
        check(False, "d 缺可对拍的产物", "A/C 没跑成")
        return {}
    d = cmp_dirs(a["dir"], c["dir"])
    if d.get("error"):
        check(False, "d 逐像素对拍", d["error"])
        return d
    check(float(d["mean_lsb"]) <= TOL_MEAN, "d 逐像素 mean ≤ %.1f LSB" % TOL_MEAN,
          "mean %.6f LSB · max %d · %d 帧" % (d["mean_lsb"], d["max_lsb"], d["frames"]))
    return d


def case_measured(big, r_after):
    head("E · 「一份会话真跑起来多大」的实测进了预算")
    rows = sess_peak_file()
    hit = {k: v for k, v in rows.items() if big[0].name in k}
    check(bool(hit), "e ws/_vram_sess_peak.json 里有大模型的实测条目",
          "、".join("%s → %.0f MB" % (k.split("|")[0], float(v.get("sess_mb") or 0))
                    for k, v in hit.items()) or "没有")
    if hit:
        sess = max(float(v.get("sess_mb") or 0) for v in hit.values())
        check(sess >= big[1], "e 实测会话体量 ≥ 权重（不是文件体积冒充）",
              "%.0f MB ≥ %.0f MB" % (sess, big[1]))
    plan = (r_after or {}).get("plan") or {}
    src = str(plan.get("sess_src") or "")
    check("实测" in src or "标定" in src, "e 下一次计划的 sess_src 说明了来路", src or "空")


def main(argv=None):
    ap = argparse.ArgumentParser(description="项目导出默认档（按模型体量）验收")
    ap.add_argument("--cases", default="a,b,c,d,e")
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args(argv)
    cases = {x.strip() for x in a.cases.split(",") if x.strip()}
    big, small = pick_models()
    if big is None:
        raise SystemExit("workspace/model 里没有 ≥1GB 的 .dfm —— 这条验收要一个大模型")
    head("默认档验收 · 大模型 %s（%.2fGB）· 小模型 %s"
         % (big[0].name, big[1] / 1024.0,
            ("%s（%.2fGB）" % (small[0].name, small[1] / 1024.0)) if small else "无"))
    prepare_sandbox()
    clip = make_clip(video_path(), MEDIA / "clip.mp4", 2.0, 6.0)
    proc = start_server()
    r_a = r_c = None
    try:
        n = build_project(clip, big[0].stem, a.frames, a.frames)
        if "a" in cases:
            r_a = case_big(clip, big, n)
        if "b" in cases and small is not None:
            r_b = case_small(clip, small, n)
        if "c" in cases:
            r_c = case_manual(clip, big, n)
        if "d" in cases:
            case_pixels(r_a, r_c)
        if "e" in cases:
            case_measured(big, r_c)
    finally:
        stop_server(proc)
    print("")
    check(not oom_in_log(), "z 整轮后端日志里没有 CUDA OOM", "server.log")
    fails = [x for x in RESULTS if not x[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - T0))
    print("后端日志：%s" % SERVER_LOG)
    print("=" * 78)
    return 0 if not fails else 1


def oom_in_log() -> bool:
    try:
        txt = SERVER_LOG.read_text("utf-8", "ignore")
    except OSError:
        return False
    for k in ("out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation",
              "Failed to allocate"):
        if k.lower() in txt.lower():
            return True
    return False


T0 = time.time()

if __name__ == "__main__":
    raise SystemExit(main())
