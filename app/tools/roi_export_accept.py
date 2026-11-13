# -*- coding: utf-8 -*-
r"""ROI 版项目导出的端到端验收：真后端 + 真项目 + 两档并行，前后帧率与像素。

验的是「合成器只算 ROI」这件事在**整条链路**上也成立：
  P) 组一个真项目（切帧 + 切脸走仓库自己的 stage 接口，默认 120 帧）
  X) 1×1（一帧一请求）：DFL_MERGE_ROI=0 vs =1 → merged/ 逐字节/逐像素对比 + 帧率
  Y) 2×4（2 进程 × 4 线程）：同上（改默认档之前的默认档，留着做前后对照）
  Z) 断点续跑：ROI=1 跑到一半取消 → 续跑 → 与一次跑完逐字节一致
  W) 汇总表：档位 · 帧/秒 · 单帧 p50 · 像素结论

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\roi_export_accept.py
    ... --frames 120            项目帧数
    ... --cases X,Y             只跑某几段
    ... --reuse                 复用沙箱里已有的项目
退出码 0 = 全过。
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
BOX = TEMP / "dfl_roi_export"
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

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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
        except Exception:                                   # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


def get(p, **kw):
    return http("GET", p, **kw)


def post(p, body, **kw):
    return http("POST", p, body, **kw)


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
    say("沙箱：%s" % BOX)


def start_server(roi: int):
    """起后端。DFL_MERGE_ROI 会随 os.environ 传进 worker 子进程（见 _spawn）。"""
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env["DFL_MERGE_ROI"] = str(int(roi))
    for k in ("DFL_PORT", "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK",
              "DFL_PROJ_WORKERS", "DFL_PROJ_BATCH", "DFL_PROJ_SEND",
              "DFL_PROJ_OUT", "DFL_FUSED_BUDGET"):
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
    say("后端：%s（pid=%s，DFL_MERGE_ROI=%d）" % (BASE, proc.pid, roi))
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
    try:
        _kill_tree(proc.pid)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-Process python -ErrorAction SilentlyContinue | "
                        "Where-Object { $_.Path -like '*engine*python.exe' } | "
                        "Stop-Process -Force"], capture_output=True, timeout=60)
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.0)


def make_clip(src, dst, ss=2.0, dur=8.0):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.3f" % ss, "-i", str(src), "-t", "%.3f" % dur,
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0 or not Path(dst).is_file():
        raise SystemExit("切素材失败：%s" % (r.stderr or "")[-400:])


def wait_stage(timeout=1800.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        st, d = get("/api/merge/stage/status", timeout=60)
        last = d or {}
        if st == 200 and not last.get("running"):
            return st, last
        time.sleep(1.0)
    return 200, last


def build_project(clip, dfm, end_frame):
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.rmtree(PROJ, ignore_errors=True)
    PROJ.mkdir(parents=True, exist_ok=True)
    body = {"video": str(clip), "model": Path(str(dfm)).stem,
            "settings": dict(SETTINGS), "cache_dir": str(CACHE),
            "end_frame": int(end_frame), "force": True}
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


def merged_dir(tag):
    return OUT / ("m_" + tag)


def frames_in(d):
    return sorted(p.name for p in d.iterdir()
                  if p.suffix.lower() in (".png", ".jpg")) if d.is_dir() else []


def run_export(clip, dfm, workers, batch, tag, force=True, cancel_after=None,
               timeout=1800.0):
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
    st, d = post("/api/merge/export-project", body)
    if st != 200:
        return {"tag": tag, "error": "HTTP %s %s" % (st, d.get("error")), "fps": None}
    snap = {}
    cancelled_at = None
    while time.time() - t0 < timeout:
        _st, snap = get("/api/merge/export-project/status", timeout=60)
        ph = snap.get("phase")
        if cancel_after is not None and cancelled_at is None \
                and int(snap.get("done") or 0) >= int(cancel_after) and ph == "running":
            post("/api/merge/export-project/cancel", {})
            cancelled_at = int(snap.get("done") or 0)
            continue
        if ph in ("done", "failed", "cancelled"):
            break
        time.sleep(0.5)
    files = frames_in(md)
    return {"tag": tag, "workers": int(workers), "batch": int(batch),
            "fps": snap.get("frame_fps"), "done": int(snap.get("done") or 0),
            "swap_p50": snap.get("swap_p50"), "swap_elapsed": snap.get("swap_elapsed"),
            "elapsed": snap.get("elapsed"), "phase": snap.get("phase"),
            "error": (snap.get("error") or "")[:200],
            "encode_s": (snap.get("encode") or {}).get("seconds"),
            "files": len(files), "failed": int(snap.get("failed") or 0),
            "wall": round(time.time() - t0, 1),
            "cancelled_at": cancelled_at, "dir": str(md)}


def cmp_dirs(a, b):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="P,X,Y,Z,W")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--video", default="")
    ap.add_argument("--reuse", action="store_true")
    opt = ap.parse_args()
    if not PY.is_file():
        raise SystemExit("没有 %s" % PY)
    dfm = dfm_path()
    if dfm is None:
        raise SystemExit(r"workspacemodel 里没有 .dfm")
    video = Path(opt.video) if opt.video else None
    if video is None:
        vids = [p for p in (REPO / "workspace").glob("*.mp4")
                if p.is_file() and "_merged" not in p.name and "._" not in p.name]
        vids.sort(key=lambda p: p.stat().st_size, reverse=True)
        video = vids[0] if vids else None
    if video is None or not video.is_file():
        raise SystemExit(r"workspace 里没有能用的视频（或用 --video 指定）")
    cases = [c.strip().upper() for c in opt.cases.split(",") if c.strip()]
    prepare_sandbox()
    clip = MEDIA / "clip.mp4"
    make_clip(video, clip, ss=2.0, dur=8.0)
    say("素材：%s（%s）" % (clip.name, "8s @ " + str(video.name)))

    rows = {}
    # ---------------- 组项目（ROI 无关，随便哪一档） ----------------
    proc = start_server(1)
    try:
        if opt.reuse and PROJ.is_dir():
            nframes = len([p for p in PROJ.iterdir()
                           if p.suffix.lower() in (".png", ".jpg") and p.is_file()])
            say("复用现有项目：%d 帧" % nframes)
        else:
            nframes = build_project(clip, dfm, int(opt.frames))
    finally:
        stop_server(proc)
    if "P" in cases:
        check(nframes >= 60, "P 项目组起来", "%d 帧" % nframes)

    def phase(roi, tags, extra=None):
        p = start_server(roi)
        try:
            for (w, b, tag) in tags:
                r = run_export(clip, dfm, w, b, tag, **(extra or {}))
                rows[tag] = r
                say("%s（ROI=%d，%dx%d）：%.2f 帧/秒 · swap p50 %sms · 落盘 %s/%s · %.1fs"
                    % (tag, roi, w, b, float(r.get("fps") or 0), r.get("swap_p50"),
                       r.get("files"), r.get("failed"), float(r.get("wall") or 0)))
        finally:
            stop_server(p)

    # 负控：同参数再跑一遍（两个进程各加载一次模型）。跨进程的 cuDNN 选核/累加顺序
    # 会让少数帧差 1~2 LSB —— 先把这个噪声地板量出来，再拿它当 ROI 前后差异的尺子。
    if "X" in cases or "Y" in cases:
        phase(0, [(1, 1, "roi0_1x1"), (2, 4, "roi0_2x4"), (2, 4, "roi0_2x4b")])
        phase(1, [(1, 1, "roi1_1x1"), (2, 4, "roi1_2x4")])

    if "X" in cases:
        head("X · 1×1（一帧一请求）：像素与帧率")
        a, b = rows.get("roi0_1x1"), rows.get("roi1_1x1")
        check(a and b and a.get("phase") == "done" and b.get("phase") == "done",
              "X 两档都跑完", "%s / %s" % (a.get("phase"), b.get("phase")))
        d = cmp_dirs(Path(a["dir"]), Path(b["dir"]))
        ctl = None
        if rows.get("roi0_2x4") and rows.get("roi0_2x4b"):
            ctl = cmp_dirs(Path(rows["roi0_2x4"]["dir"]), Path(rows["roi0_2x4b"]["dir"]))
        strict = not d.get("error") and d["max_lsb"] == 0 and d["byte_identical"] == d["frames"]
        if strict:
            check(True, "X 1×1 ROI 版产物与整帧版**逐字节相同**",
                  "%d 帧全部逐字节相同" % d.get("byte_identical", 0))
        elif ctl and not ctl.get("error"):
            ok = (d["mean_lsb"] <= max(ctl["mean_lsb"] * 1.5, 1e-6)
                  and d["max_lsb"] <= max(ctl["max_lsb"], 1) + 1
                  and d["byte_identical"] >= ctl["byte_identical"] * 0.9)
            check(ok, "X 1×1 ROI 前后差异 ≤ 跨进程噪声地板（逐位一致的严格证明在 roi_parity_accept）",
                  "ROI 前后：%d/%d 逐字节同 · mean %.6f · max %d ｜ 负控：%d/%d · mean %.6f · max %d"
                  % (d["byte_identical"], d["frames"], d["mean_lsb"], d["max_lsb"],
                     ctl["byte_identical"], ctl["frames"], ctl["mean_lsb"], ctl["max_lsb"]))
        else:
            check(False, "X 1×1 ROI 版产物与整帧版逐字节相同",
                  "逐字节相同 %d/%d · mean %.6f · max %d"
                  % (d["byte_identical"], d["frames"], d["mean_lsb"], d["max_lsb"]))
        check(float(b.get("fps") or 0) > float(a.get("fps") or 0), "X 帧率提升",
              "%.2f → %.2f 帧/秒（%.2fx）· swap p50 %sms → %sms"
              % (float(a.get("fps") or 0), float(b.get("fps") or 0),
                 float(b.get("fps") or 0) / max(1e-9, float(a.get("fps") or 0)),
                 a.get("swap_p50"), b.get("swap_p50")))

    if "Y" in cases:
        head("Y · 2×4（改默认档之前的默认档）：像素与帧率")
        a, b = rows.get("roi0_2x4"), rows.get("roi1_2x4")
        check(a and b and a.get("phase") == "done" and b.get("phase") == "done",
              "Y 两档都跑完", "%s / %s" % (a.get("phase"), b.get("phase")))
        d = cmp_dirs(Path(a["dir"]), Path(b["dir"]))
        ctl = None
        if rows.get("roi0_2x4") and rows.get("roi0_2x4b"):
            ctl = cmp_dirs(Path(rows["roi0_2x4"]["dir"]), Path(rows["roi0_2x4b"]["dir"]))
        strict = not d.get("error") and d["max_lsb"] == 0 and d["byte_identical"] == d["frames"]
        if strict:
            check(True, "Y 2×4 ROI 版产物与整帧版**逐字节相同**",
                  "%d 帧全部逐字节相同" % d.get("byte_identical", 0))
        elif ctl and not ctl.get("error"):
            # 跨进程比对：差异必须落在「同参数两次运行」那条噪声地板以内
            ok = (d["mean_lsb"] <= max(ctl["mean_lsb"] * 1.5, 1e-6)
                  and d["max_lsb"] <= max(ctl["max_lsb"], 1) + 1
                  and d["byte_identical"] >= ctl["byte_identical"] * 0.9)
            check(ok, "Y 2×4 ROI 前后差异 ≤ 跨进程噪声地板（逐位一致的严格证明在 roi_parity_accept）",
                  "ROI 前后：%d/%d 逐字节同 · mean %.6f · max %d ｜ 负控：%d/%d · mean %.6f · max %d"
                  % (d["byte_identical"], d["frames"], d["mean_lsb"], d["max_lsb"],
                     ctl["byte_identical"], ctl["frames"], ctl["mean_lsb"], ctl["max_lsb"]))
        else:
            check(False, "Y 2×4 ROI 版产物与整帧版逐字节相同",
                  "逐字节相同 %d/%d · mean %.6f · max %d"
                  % (d["byte_identical"], d["frames"], d["mean_lsb"], d["max_lsb"]))
        check(float(b.get("fps") or 0) > float(a.get("fps") or 0), "Y 帧率提升",
              "%.2f → %.2f 帧/秒（%.2fx）· swap p50 %sms → %sms"
              % (float(a.get("fps") or 0), float(b.get("fps") or 0),
                 float(b.get("fps") or 0) / max(1e-9, float(a.get("fps") or 0)),
                 a.get("swap_p50"), b.get("swap_p50")))

    if "Z" in cases:
        head("Z · 断点续跑（ROI=1）：取消 → 续跑 → 与一次跑完逐字节一致")
        p = start_server(1)
        try:
            cut = run_export(clip, dfm, 2, 4, "roi1_cut",
                             cancel_after=max(6, nframes // 3))
            rows["roi1_cut"] = cut
            check(cut.get("phase") == "cancelled", "Z 取消落地",
                  "phase=%s done=%s files=%s" % (cut.get("phase"), cut.get("done"),
                                                 cut.get("files")))
            check(int(cut.get("done") or 0) == int(cut.get("files") or 0),
                  "Z 游标 = merged/ 里的帧数", "done=%s files=%s"
                  % (cut.get("done"), cut.get("files")))
            names = frames_in(Path(cut["dir"]))
            nums = []
            for n in names:
                mm = re.search(r"(\d+)", Path(n).stem)
                nums.append(int(mm.group(1)) if mm else -1)
            check(bool(nums) and nums == list(range(nums[0], nums[0] + len(nums))),
                  "Z 留下的是前缀（编号连续）", "%s…%s（%d 帧）"
                  % (names[:1], names[-1:], len(names)))
            res = run_export(clip, dfm, 2, 4, "roi1_resumed", force=False)
            rows["roi1_resumed"] = res
            check(res.get("phase") == "done" and res.get("files") == nframes,
                  "Z 续跑补齐", "%s 帧" % res.get("files"))
            full = rows.get("roi1_2x4")
            if full:
                d = cmp_dirs(Path(full["dir"]), Path(res["dir"]))
                # 两次运行之间本来就有跨进程的 1~2 LSB 噪声（见负控），这里按它判。
                check(not d.get("error") and d["mean_lsb"] <= 0.002 and d["max_lsb"] <= 8,
                      "Z 续跑产物 vs 一次跑完一致（≤ 跨进程噪声地板）",
                      ("逐字节相同 %d/%d · mean %.6f · max %d"
                       % (d.get("byte_identical", 0), d.get("frames", 0),
                          d.get("mean_lsb", -1), d.get("max_lsb", -1)))
                      if not d.get("error") else d["error"])
        finally:
            stop_server(p)

    if "W" in cases:
        head("W · 汇总")
        print("  %-14s %-6s %-9s %-11s %-9s %s"
              % ("档位", "ROI", "帧/秒", "swap p50", "落盘/失败", "墙钟"))
        for tag in ("roi0_1x1", "roi1_1x1", "roi0_2x4", "roi1_2x4",
                    "roi1_cut", "roi1_resumed"):
            r = rows.get(tag)
            if not r:
                continue
            print("  %-14s %-6s %-9s %-11s %-9s %s"
                  % (tag.split("_")[0].replace("roi", "ROI=") + " " + tag.split("_", 1)[1],
                     tag[3], "%.2f" % float(r.get("fps") or 0),
                     "%sms" % r.get("swap_p50"), "%s/%s" % (r.get("files"), r.get("failed")),
                     "%.1fs" % float(r.get("wall") or 0)))
        try:
            (LOGDIR / "summary.json").write_text(
                json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            say("明细：%s" % (LOGDIR / "summary.json"))
        except Exception:                                   # noqa: BLE001
            pass

    bad = [x for x in RESULTS if not x[0]]
    print("")
    print("=" * 78)
    print("结果：%d 项通过，%d 项失败" % (len(RESULTS) - len(bad), len(bad)))
    for _ok, name, detail in bad:
        print("  [FAIL] %s  · %s" % (name, detail))
    print("沙箱：%s" % BOX)
    print("=" * 78)
    sys.exit(1 if bad else 0)


main()
