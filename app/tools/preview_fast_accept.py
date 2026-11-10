# -*- coding: utf-8 -*-
r"""单帧预览「一步式 + 零 cache」的验收：真后端 + 真视频（1080x1920）+ 现有 dfm，全在 %TEMP%。

口径（这一轮定的）：
  · 预览 = worker 里一次 MergeMaskedFace：dfm 推理 + 遮罩 + 融合，**内存出图**；
  · 预览路径不读、不写、不依赖 <视频>_cache/（没有 cache 也要一样跑）；
  · 导出仍走六阶段（见 stage6_accept.py），两条路算法等价但不是同一份代码，
    像素差允许存在：预览 vs 导出帧（都无损）mean 0.02~0.03；成片 vs 预览 ≈ 3.6~4
    （编码往返）—— 见 stage6_accept 的 d 段实测。

验的东西：
  a) 连续 10 帧预览（10 个不同的 t）：<视频>_cache/ 不存在；万一本来就在，
     文件数 / 每个文件的 mtime+size 一个都不许变。
  a2) 故意先摆一个「脏 cache」（frames/pred/mask 里塞不合法的小图 + meta.json）：
     同一张帧图分别放在 cache 外和 cache/frames 里各预览一次 —— 目录一个字不许动，
     两张图还要逐像素相同（证明预览完全不看 cache，不存在"有 cache 就复用"）。
  b) 每帧耗时（改前 vs 改后配成表）：--label 不同的两次跑，后一次自动和上一次比；
     热态目标 ≤ 400ms/帧。
  c) 响应体里没有 cache 路径 / cache 字段；后端日志、阶段日志里没有写 cache 的记录。
  d) 走的是一步式：timings 里必须有 swap_ms，不许有 compose_path / infer_calls /
     pred_cache_hits（那是六阶段 compose 那条路的打点）。

跑法（仓库根，先热后冷都行；两次 label 不同即可对比改前改后）：
    runtime\server\Scripts\python.exe app\tools\preview_fast_accept.py --label before
    runtime\server\Scripts\python.exe app\tools\preview_fast_accept.py --label after

退出码 0 = 全过。
"""
from __future__ import annotations

import base64
import hashlib
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
BOX = TEMP / "dfl_preview_fast_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay",
            "mask_erode": 0, "mask_blur": 4}
TARGET_MS = 400.0          # 热态单帧端到端目标
CAP_MS = 900.0             # 硬上限（超了就是没回到一步式）
N_FRAMES = 10

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
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-400:]))
    return r


def probe(path):
    import cv2
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True)
    txt = r.stderr.decode("utf-8", "ignore")
    dur = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    cap = cv2.VideoCapture(str(path))
    info = {"dur": dur, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    cap.release()
    return info


# ============================================================
# 沙箱 / 后端（和 stage6_accept 同一套写法）
# ============================================================
def prepare_sandbox():
    keep = BOX / "timings"
    saved = {}
    if keep.is_dir():                       # 保留历次 label 的耗时表（改前/改后要比）
        for p in keep.glob("timing_*.json"):
            saved[p.name] = p.read_bytes()
    if BOX.exists():
        try:
            os.rmdir(str(WS / "model"))
        except OSError:
            pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, LOGDIR, keep):
        d.mkdir(parents=True, exist_ok=True)
    for name, blob in saved.items():
        (keep / name).write_bytes(blob)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / "model"),
                        str(REPO / "workspace" / "model")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
    say("沙箱：%s" % BOX)


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
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))], capture_output=True, timeout=30)
    except Exception:                                       # noqa: BLE001
        pass


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)          # 不保留 cache：这轮就要它别出现
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
            st, _d = get("/api/merge/videos", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/videos 不应答，见 %s" % SERVER_LOG)


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.2)


# ============================================================
# cache 目录快照：不存在 / 存在但一个字节都没动
# ============================================================
def cache_snapshot(dirp: Path) -> dict:
    d = Path(dirp)
    if not d.is_dir():
        return {"exists": False, "files": 0, "items": []}
    items = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            st = p.stat()
            items.append([str(p.relative_to(d)), int(st.st_size), int(st.st_mtime_ns)])
    return {"exists": True, "files": len(items), "items": items}


def snap_diff(a: dict, b: dict) -> str:
    if not a["exists"] and not b["exists"]:
        return ""
    if a["exists"] != b["exists"]:
        return "存在性变了：%s → %s" % (a["exists"], b["exists"])
    if a["items"] == b["items"]:
        return ""
    old = {x[0]: x[1:] for x in a["items"]}
    new = {x[0]: x[1:] for x in b["items"]}
    added = sorted(set(new) - set(old))
    gone = sorted(set(old) - set(new))
    changed = sorted(k for k in (set(old) & set(new)) if old[k] != new[k])
    return "新增 %d %s · 少了 %d %s · 改了 %d %s" % (
        len(added), added[:3], len(gone), gone[:3], len(changed), changed[:3])


# ============================================================
# 预热：worker + 检测器 + DFM 全装好，再开始计时
# ============================================================
def warm(model=DFM, timeout=300.0):
    st, d = post("/api/merge/preview/warm",
                 {"models": [{"name": model, "faces": []}]}, timeout=120)
    if st != 200:
        raise SystemExit("预热接口失败：HTTP %s %s" % (st, str(d)[:200]))
    t0 = time.time()
    while time.time() - t0 < timeout:
        _s, s = get("/api/merge/preview/warm/status", timeout=60)
        if s.get("ready"):
            say("预热完成：%.1fs · worker=%s · 模型=%s"
                % (time.time() - t0, s.get("worker_pid"), s.get("model")))
            return s
        time.sleep(0.3)
    raise SystemExit("预热超时")


# ============================================================
# 主体：连续切 N 帧预览 + 计时 + 零 cache 断言
# ============================================================
def measure(video, info, label, n=N_FRAMES):
    cache = video.parent / (video.stem + "_cache")
    before = cache_snapshot(cache)
    check(not before["exists"], "a 起跑前 <视频>_cache/ 不存在（这一轮是干净沙箱）",
          str(cache) if before["exists"] else "确认不存在")

    # 预热那一枪不算：它要付 worker 启动 / 检测器 / dfm 首帧 kernel 的钱
    _st, _w = post("/api/merge/preview", {"video": str(video), "t": 0.2,
                                          "models": [{"name": DFM, "faces": []}],
                                          "settings": dict(SETTINGS)}, timeout=600)
    time.sleep(0.5)

    fps = info["fps"] or 30.0
    t0 = 1.0
    rows = []
    for i in range(n):
        t = round(t0 + i / fps, 4)
        tw = time.time()
        st, d = post("/api/merge/preview",
                     {"video": str(video), "t": t,
                      "models": [{"name": DFM, "faces": []}],
                      "settings": dict(SETTINGS)}, timeout=600)
        wall = (time.time() - tw) * 1000.0
        if st != 200 or not d.get("ok"):
            check(False, "b 第 %d 帧预览失败" % i, "HTTP %s · %s" % (st, str(d.get("error"))[:140]))
            continue
        _raw = base64.b64decode((d.get("png") or "").split(",", 1)[-1])
        rows.append({"i": i, "t": t, "ms": int(d.get("ms") or 0),
                     "wall_ms": round(wall, 1),
                     "timings": d.get("timings") or {},
                     "sha": hashlib.sha256(_raw).hexdigest(),
                     "bytes": len(_raw),
                     "keys": sorted(d.keys()),
                     "worker": (d.get("worker") or {}).get("script"),
                     "swapped": bool(d.get("swapped")),
                     "faces": int(d.get("swaps") or 0)})
    ok_rows = [r for r in rows if r["swapped"]]
    check(len(ok_rows) >= max(3, n - 1), "b %d 帧都真换了脸（不是回原帧）" % n,
          "%d / %d · faces=%s" % (len(ok_rows), n, [r["faces"] for r in rows]))

    # ---- a) cache 一次都没碰 ----
    after = cache_snapshot(cache)
    diff = snap_diff(before, after)
    check(diff == "", "a 预览全程没碰 <视频>_cache/（不存在 / 文件数+mtime+size 全不变）",
          diff or ("不存在" if not after["exists"] else "%d 个文件，未变" % after["files"]))

    # ---- c) 响应体里没有 cache 路径 / cache 字段 ----
    leak = []
    for r in rows:
        blob = json.dumps(r, ensure_ascii=False, default=str)
        for bad in ("_cache", "cache_dir", "frame_stem", "compose_path",
                    "pred_cache_hits", "infer_calls", str(cache)):
            if bad in blob:
                leak.append("第 %d 帧出现 %s" % (r["i"], bad))
    leaked_keys = sorted({k for r in rows for k in (r.get("keys") or []) if "cache" in k.lower()})
    check(not leak and not leaked_keys,
          "c 响应体里没有 cache 路径 / cache 字段（顶层键也不许带 cache）",
          ("；".join(leak[:3]) or ("键：" + ",".join(leaked_keys))) if (leak or leaked_keys) else "干净")

    # ---- d) 一步式：timings 有 swap_ms，没有六阶段 compose 的打点 ----
    bad = []
    for r in rows:
        tm = r["timings"] or {}
        if not tm.get("swap_ms"):
            bad.append("第 %d 帧没有 swap_ms" % r["i"])
        for k in ("compose_path", "pred_cache_hits", "mask_cache_hits", "infer_calls"):
            if k in tm:
                bad.append("第 %d 帧 timings 里有 %s（六阶段 compose 打点）" % (r["i"], k))
    check(not bad, "d 走的就是一步式 MergeMaskedFace（swap_ms 有 / compose 打点没有）",
          "；".join(bad[:3]) or "干净")

    # worker 的临时脚本只允许在 %TEMP% 里，不许落进素材目录 / cache 树
    scripts = sorted({str(r.get("worker") or "") for r in rows})
    ok_tmp = all(scripts) and all(str(BOX) not in s and "dfl41_merge_preview" in s
                                  for s in scripts)
    check(ok_tmp, "d worker 临时目录在 %TEMP%（不在 cache 目录树里）",
          scripts[0] if scripts else "没有 script 字段")

    # ---- b) 耗时 ----
    ms = sorted(r["ms"] for r in ok_rows)
    mean = sum(ms) / float(len(ms)) if ms else 0.0
    p50 = ms[len(ms) // 2] if ms else 0
    say("每帧端到端（服务端 ms）：%s" % [r["ms"] for r in rows])
    say("每帧端到端（客户端 wall ms）：%s" % [r["wall_ms"] for r in rows])
    say("分解（grab/detect/landmarks/swap）：%s"
        % [(r["timings"].get("grab_ms"), r["timings"].get("detect_ms"),
            r["timings"].get("landmarks_ms"), r["timings"].get("swap_ms"))
           for r in ok_rows])
    check(mean <= TARGET_MS, "b 热态平均每帧 ≤ %.0fms" % TARGET_MS,
          "mean=%.1f · p50=%d · max=%d" % (mean, p50, max(ms) if ms else 0))
    check(max(ms or [0]) <= CAP_MS, "b 没有单帧超过硬上限 %.0fms（真的一步式）" % CAP_MS,
          "max=%d" % (max(ms) if ms else 0))

    # ---- 阶段日志：没有写 cache 的记录 ----
    _s, st_status = get("/api/merge/stage/status", timeout=60)
    logtxt = "\n".join(str(x) for x in (st_status.get("log") or []))
    check("clip_cache" not in logtxt and str(cache) not in logtxt,
          "c 阶段日志里没有写 cache 的记录", "%d 行日志，没有 cache 路径" % len(logtxt.splitlines()))
    check(not st_status.get("running"), "c 预览期间六阶段一个都没被启动",
          "running=%s · next=%s" % (st_status.get("running"), st_status.get("next_stage")))

    return {"label": label, "mean_ms": round(mean, 1), "p50_ms": p50,
            "max_ms": max(ms) if ms else 0, "rows": rows,
            "cache": str(cache), "cache_existed": after["exists"]}


def dirty_cache_phase(video, info, clean_rows, n=3):
    """脏 cache 测试：预览只认那张帧图，不认它旁边有没有 <视频>_cache/。

    用 /api/merge/preview-frame 而不是 /preview：视频取帧在「离开再回到同一个 t」时
    不保证落在同一帧（实测 5 次里 3 次差一帧，是 cv2 重定位的噪声，和 cache 无关），
    那种噪声会盖掉要验的东西。静态帧这条链路的检测 / 推理 / 编码是确定性的（同帧连
    打两次字节完全相同，已实测），所以「读没读 cache」可以量成 0/1。

    摆的东西正好是老代码会去读的那些文件名：<cache>/frames/000001.png（真的帧图，
    故意放在 frames/ 里 —— 老代码会从它反推出 cache_dir）、pred/000001_00.png、
    mask/000001_00_xseg_prd.png、aligned/000001_00.jpg（假 JPEG）、meta.json。
    """
    import cv2
    import numpy as np

    cache = video.parent / (video.stem + "_cache")
    shutil.rmtree(str(cache), ignore_errors=True)
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    real = cache / "frames" / "000001.png"          # 真帧图，但放在 frames/ 里
    run_ff(["-ss", "2.0", "-i", video, "-frames:v", "1", str(real)], "取一帧当静态帧")
    outside = video.parent / "ref_000001.png"        # 同一份字节，放在 cache 之外
    shutil.copy2(str(real), str(outside))

    junk = np.zeros((256, 256, 3), np.uint8)
    junk[::2] = 255
    for sub in ("pred", "mask", "aligned"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache / "pred" / "000001_00.png"), junk)
    cv2.imwrite(str(cache / "pred" / "000001_00_prdm.png"), junk)
    cv2.imwrite(str(cache / "mask" / "000001_00_xseg_prd.png"), junk)
    cv2.imwrite(str(cache / "mask" / "000001_00_xseg_dst.png"), junk)
    (cache / "aligned" / "000001_00.jpg").write_bytes(b"not a jpeg")
    (cache / "meta.json").write_text('{"stages": {"frames": {"done": true}}}', "utf-8")

    before = cache_snapshot(cache)
    check(before["exists"] and before["files"] >= 7, "a2 摆好一个「脏 cache」（半成品文件）",
          "%d 个文件：%s" % (before["files"], [x[0] for x in before["items"]][:4]))

    def pv(fp):
        st, d = post("/api/merge/preview-frame",
                     {"frame_file": str(fp), "fmt": "png",
                      "models": [{"name": DFM, "faces": []}],
                      "settings": dict(SETTINGS)}, timeout=600)
        if st != 200 or not d.get("ok"):
            return None, d
        raw = base64.b64decode((d.get("png") or "").split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img, d

    def _diff(a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return float(d.mean()), int(d.max()), int((d > 0).any(2).sum())

    # 注意：比的是**像素**不是 sha —— 这条链路上 CPU 端的 2D106/推理有 ±1 的量化
    # 抖动（同帧连打两次实测 274/2073600 个像素差 1 级，位置就在脸框里），
    # 字节相同会误报。cache 真被读了的话差的是整张脸，不是 1 级。
    img_out, d_out = pv(outside)          # cache 之外的那份
    img_in, d_in = pv(real)               # cache/frames 里那份，周围全是脏 pred/mask
    ok_pair = img_out is not None and img_in is not None
    mean, mx, npx = _diff(img_out, img_in) if ok_pair else (999.0, 999, 0)
    check(ok_pair and mx <= 1,
          "a2 同一张帧图：放在 cache 外 / 放在脏 cache 的 frames/ 里，出图逐像素相同（≤1 级）",
          "mean=%.4f max=%d 差异像素=%d" % (mean, mx, npx))

    after = cache_snapshot(cache)
    diff = snap_diff(before, after)
    check(diff == "", "a2 预览没动脏 cache 一个字节（文件数 / mtime / size 全不变）",
          diff or "%d 个文件原样" % after["files"])

    # 把 pred/mask/aligned/meta 全拿掉，只留 frames/000001.png → 还是同一张图
    for sub in ("pred", "mask", "aligned"):
        shutil.rmtree(str(cache / sub), ignore_errors=True)
    try:
        (cache / "meta.json").unlink()
    except OSError:
        pass
    img2, d2 = pv(real)
    ok2 = img2 is not None and img_out is not None
    mean2, mx2, npx2 = _diff(img2, img_out) if ok2 else (999.0, 999, 0)
    check(ok2 and mx2 <= 1,
          "a2 拿掉 pred/mask 之后还是同一张图（≤1 级；预览不读它们，也不因缺它们而变）",
          "mean=%.4f max=%d 差异像素=%d" % (mean2, mx2, npx2))

    outs = [d for d in (d_out, d_in, d2) if d]
    tm = (outs[0].get("timings") or {}) if outs else {}
    shutil.rmtree(str(cache), ignore_errors=True)
    try:
        outside.unlink()
    except OSError:
        pass
    return {"files": after["files"], "px_max": mx,
            "swap_ms": tm.get("swap_ms")}



def load_timing(label):
    p = BOX / "timings" / ("timing_%s.json" % label)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:                                       # noqa: BLE001
        return None


def compare(cur, prev):
    if not prev:
        say("没有可对比的上一轮（--label 换一个再跑一次就有改前 vs 改后）")
        return
    prow = {r["i"]: r for r in (prev.get("rows") or [])}
    print("")
    print("改前（%s）vs 改后（%s）：" % (prev.get("label"), cur["label"]))
    print("  帧  改前 ms  改后 ms   差      swap_ms(改后)")
    deltas = []
    for r in cur["rows"]:
        o = prow.get(r["i"])
        if not o:
            continue
        d = r["ms"] - int(o["ms"])
        deltas.append(d)
        print("  %2d   %6d   %6d  %+6d    %s"
              % (r["i"], int(o["ms"]), r["ms"], d, (r["timings"] or {}).get("swap_ms")))
    if deltas:
        print("  平均：改前 %.1f ms → 改后 %.1f ms（%+.1f ms，%+.0f%%）"
              % (prev.get("mean_ms") or 0, cur["mean_ms"],
                 cur["mean_ms"] - (prev.get("mean_ms") or 0),
                 100.0 * (cur["mean_ms"] - (prev.get("mean_ms") or 0))
                 / max(1.0, prev.get("mean_ms") or 1.0)))
    check(cur["mean_ms"] < (prev.get("mean_ms") or 0),
          "b 改后比改前快", "%.1f ms → %.1f ms" % (prev.get("mean_ms") or 0, cur["mean_ms"]))


# ============================================================
# main
# ============================================================
def main():
    label = "after"
    n = N_FRAMES
    argv = sys.argv[1:]
    if "--label" in argv:
        label = str(argv[argv.index("--label") + 1])
    if "--frames" in argv:
        n = int(argv[argv.index("--frames") + 1])
    t_all = time.time()
    prepare_sandbox()
    src_video = [p for p in sorted((REPO / "workspace").glob("*.mp4"))
                 if "_merged" not in p.stem][0]
    video = MEDIA / "clip.mp4"
    run_ff(["-ss", "3", "-t", "3.0", "-i", src_video, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", video],
           "切样片")
    info = probe(video)
    say("样片：%s" % info)
    if info["w"] != 1080 or info["h"] != 1920:
        raise SystemExit("样片不是 1080x1920：%s" % info)

    proc = start_server("预览零 cache")
    try:
        warm()
        cur = measure(video, info, label, n=n)
        cur["dirty"] = dirty_cache_phase(video, info, cur["rows"])
        other = None
        for x in ("before", "after"):        # 找另一份 label 的耗时表来对比
            if x != label:
                other = other or load_timing(x)
        (BOX / "timings" / ("timing_%s.json" % label)).write_text(
            json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
        compare(cur, other)
    finally:
        stop_server(proc)

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs（label=%s）"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all, label))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
