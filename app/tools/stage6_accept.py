# -*- coding: utf-8 -*-
r"""六阶段分段导出的验收：真后端 + 真视频（1080x1920）+ 现有 dfm，全部在 %TEMP% 里跑。

口径（用户这一轮定的）：
  ① frames 切帧 → ② faces 切脸 → ③ pred 换脸(dfm 推理) → ④ mask 遮罩推理 →
  ⑤ merge 合成（**纯 CPU**）→ ⑥ encode 编码
  每步独立 API / 独立游标 / 可单跑可续跑；上一步没做完就 409 点名前一步。

验的东西：
  a) 六步顺序跑通，每步只产出自己目录的东西（文件清单为证）。
  b) ③ 只做 dfm 推理：pred/ 每张脸一个 PNG（含 dfm 自带 prdm/dstdm），不碰遮罩/合成；
     批处理证据（meta.stages.pred.batch）+ 张/秒。
  c) ④ 只做遮罩推理：mask/ 有 xseg_prd/xseg_dst（256×256 灰度），读回来断言非空、
     尺寸正确、遮罩均值在合理区间；来源=DFM 时不跑推理、但照样落盘 _dfm_prd/_dfm_dst。
  d) ⑤ 纯 CPU：推理调用计数 = 0、dfm session 数不变、显存不增长；
     五段 profile（prep/mask/ct/paste/blend/encode/write）+ 总 fps ≥ 20（1080p）。
  e) 409 闸门：缺 pred / 缺 mask / 缺 merge 都明确报错；改遮罩来源后 mask/ 失效要重跑。
  f) 续跑：③ 取消后从游标续；④ 取消后从游标续。
  g) 成片质量：抽 3 帧，成片 vs 单帧预览 mean ≤ 8。
  h) 与 4.0 逐像素对拍：同一帧同一参数，4.0 的 _phase_merge_worker（真跑，engine python）
     vs 我们的 compose —— mean ≤ 2 / max ≤ 8。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\stage6_accept.py

退出码 0 = 全过。
"""
from __future__ import annotations

import base64
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
ENGINE_PY = REPO / "runtime" / "engine" / "python.exe"
ENGINE = REPO.parent / "DeepFaceLab-PyTorch"      # 4.0（视频模式的语义基准）
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_stage6_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"
STAGES = ("frames", "faces", "pred", "mask", "merge", "encode")
TOL_VIDEO = 8          # 成片 vs 单帧预览
TOL_40 = 2             # 与 4.0 的逐像素 mean
MAX_40 = 8             # 与 4.0 的逐像素 max
FPS_MIN = 20.0

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
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"dur": dur, "audio": "Audio:" in txt, "frames": frames, "fps": fps, "w": w, "h": h}


def imread_any(path):
    import cv2
    import numpy as np
    try:
        with open(str(path), "rb") as fh:
            buf = np.frombuffer(fh.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                                       # noqa: BLE001
        return None


def read_gray(path):
    import cv2
    import numpy as np
    with open(str(path), "rb") as fh:
        buf = np.frombuffer(fh.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def decode_all(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
            out.append(img)
    finally:
        cap.release()
    return out


def diff(a, b):
    import numpy as np
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), float(np.percentile(d, 99)), int(d.max())


def read_png(data_url):
    import cv2
    import numpy as np
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# ============================================================
# 沙箱 / 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        try:
            os.rmdir(str(WS / "model"))
        except OSError:
            pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, OUT, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
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


def start_server(tag="", keep_cache=True):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)
    if keep_cache:
        env["DFL_STAGE_KEEP"] = "1"
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
    say("后端%s：%s（pid=%s）" % (tag, BASE, proc.pid))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/stage/status 不应答，见 %s" % SERVER_LOG)


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
# 阶段调用
# ============================================================
def run_stage(name, body, timeout=3600.0):
    st, d = post("/api/merge/stage/%s" % name, body)
    if st != 200 or d.get("ok") is False:
        return st, d, {}
    if d.get("skipped"):
        return st, d, d
    t0 = time.time()
    while time.time() - t0 < timeout:
        _s, s = get("/api/merge/stage/status", timeout=60)
        if not s.get("running"):
            return st, d, s
        time.sleep(0.4)
    return st, d, {}


def cache_of(video):
    v = Path(video)
    return v.parent / (v.stem + "_cache")


def imgs(dirp, exts=(".png", ".jpg", ".jpeg")):
    d = Path(dirp)
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts],
                  key=lambda p: p.name)


# ============================================================
# 案例 A：六步顺序跑通 + 每步只产出自己的目录
# ============================================================
def case_a(video):
    src = probe(video)
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM,
            "settings": {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay",
                         "mask_erode": 0, "mask_blur": 4}}
    shutil.rmtree(cache, ignore_errors=True)
    snap = {}
    shutil.rmtree(cache, ignore_errors=True)
    run_stage("frames", body)
    refs = {}
    for n in (4, 19, 44):
        fp = cache / "frames" / ("%06d.png" % (n + 1))
        if fp.is_file():
            refs[n] = OUT / ("ref_%03d.png" % (n + 1))
            shutil.copy2(str(fp), str(refs[n]))
    for name in ("faces", "pred", "mask", "merge", "encode"):
        st, d, fin = run_stage(name, body)
        snap[name] = fin
        meta = ((fin or {}).get("meta") or {}).get("stages", {}).get(name, {})
        check(st == 200 and fin.get("phase") == "done", "a 阶段 %s 完成" % name,
              "HTTP %s · %s" % (st, (fin.get("error") or "")[:100]))
        say("  %s：%s" % (name, {k: meta.get(k) for k in
                                ("done", "count", "seconds", "fps", "batch", "signature",
                                 "inference", "size")}))
    # 每阶段的耗时/速度要能一眼看到（status 的 stage_seconds / stage_fps）
    _s, sfin = get("/api/merge/stage/status")
    check(all(float(v or 0) > 0 for v in (sfin.get("stage_seconds") or {}).values()),
          "d status 里每阶段耗时都能看到", str({k: round(float(v), 2) for k, v in
                                            (sfin.get("stage_seconds") or {}).items()}))
    say("  每阶段耗时（秒）：%s" % {k: round(float(v), 2) for k, v in
                                (sfin.get("stage_seconds") or {}).items()})

    # ---- 每步只产出自己的目录 ----
    frames = imgs(cache / "frames")
    faces = imgs(cache / "aligned")
    preds = [p for p in imgs(cache / "pred") if not re.search(r"_(prdm|dstdm)$", p.stem)]
    masks = imgs(cache / "mask")
    check(len(frames) == src["frames"], "a frames/ = 源帧数",
          "%d / %d" % (len(frames), src["frames"]))
    check(len(faces) >= 1, "a aligned/ 有对齐脸", "%d 张" % len(faces))
    check(len(preds) == len(faces), "a pred/ 每张对齐脸一个换脸 PNG（一一对应）",
          "%d / %d" % (len(preds), len(faces)))
    check(all(p.stem == a.stem for p, a in zip(preds, faces)),
          "a pred 命名与 aligned 一一对应", preds[0].name if preds else "无")
    n_x = len([p for p in masks if p.stem.endswith("_xseg_prd")])
    n_xd = len([p for p in masks if p.stem.endswith("_xseg_dst")])
    n_d = len([p for p in masks if p.stem.endswith("_dfm_prd")])
    check(n_x == len(faces) and n_xd == len(faces),
          "a mask/ 有 xseg_prd + xseg_dst（每张脸各一张）", "%d / %d" % (n_x, n_xd))
    check(n_d == len(faces), "a mask/ 也留了 dfm 自带遮罩（mask_mode=9 要用）", "%d 张" % n_d)
    check(not (cache / "merged").exists(), "a 没有另开 merged/（就地覆盖 frames/）")

    # ---- b) ③ 的批处理证据 ----
    pmeta = ((snap.get("pred") or {}).get("meta") or {}).get("stages", {}).get("pred", {})
    check(int(pmeta.get("batch") or 0) >= 1, "b ③ 换脸批档由实测决定（>=1 即通过；本机 416 上 B4/B8 反而更慢，auto 只留 B1）",
          "batch=%s · %s 张/秒 · %s" % (pmeta.get("batch"), pmeta.get("faces_per_sec"),
                                        pmeta.get("sizes")))

    # ---- c) ④ 的遮罩产物读回断言 ----
    worst = None
    ok_sz = ok_nonempty = ok_range = 0
    for p in masks:
        if not (p.stem.endswith("_xseg_prd") or p.stem.endswith("_xseg_dst")):
            continue
        img = read_gray(p)
        if img is None:
            continue
        ok_sz += int(img.shape[:2] == (256, 256))
        ok_nonempty += int(p.stat().st_size > 0)
        m = float(img.mean()) / 255.0
        ok_range += int(0.01 <= m <= 0.99)
        worst = m if worst is None else min(worst, m)
    total = max(1, n_x + n_xd)
    check(ok_sz == total, "c 遮罩尺寸 = 256×256（读回来量的）", "%d / %d" % (ok_sz, total))
    check(ok_nonempty == total, "c 遮罩文件非空", "%d / %d" % (ok_nonempty, total))
    check(ok_range == total, "c 遮罩数值在 0..1 的合理区间（不是全黑/全白）",
          "最小均值 %.3f · %d / %d" % (worst or 0, ok_range, total))

    # ---- d) ⑤ 纯 CPU + profile + fps ----
    mmeta = ((snap.get("merge") or {}).get("meta") or {}).get("stages", {}).get("merge", {})
    prof = mmeta.get("profile") or {}
    per = prof.get("per_frame_ms") or {}
    say("⑤ profile（每帧 ms）：%s" % json.dumps(per, ensure_ascii=False))
    check(prof.get("inference_calls") == 0, "d 合成阶段推理调用 = 0（纯 CPU）",
          "inference_calls=%s" % prof.get("inference_calls"))
    check(int(prof.get("dfm_sessions_after") or 0) <= int(prof.get("dfm_sessions_before") or 0),
          "d 合成阶段没新建任何 dfm/xseg session",
          "%s → %s" % (prof.get("dfm_sessions_before"), prof.get("dfm_sessions_after")))
    g0, g1 = prof.get("gpu_mem_mb_before"), prof.get("gpu_mem_mb_after")
    grew = (int(g1) - int(g0)) if (g0 is not None and g1 is not None) else 0
    check(grew <= 200, "d 合成阶段显存不增长（±200MB 容差）", "%s → %s MB" % (g0, g1))
    check(float(prof.get("fps") or 0) >= FPS_MIN, "d 合成 fps ≥ %.0f（1080×1920）" % FPS_MIN,
          "%.2f 帧/秒 · %d 线程 · PNG level %s"
          % (float(prof.get("fps") or 0), int(prof.get("threads") or 0),
             prof.get("png_level")))
    for k in ("prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms", "encode_ms", "write_ms"):
        check(k in per, "d profile 有五段打点：%s" % k, per.get(k))

    # ---- 成片 ----
    out = Path(str((snap.get("encode") or {}).get("out_video") or ""))
    check(out.is_file() and out.stat().st_size > 0, "a 成片存在",
          "%s · %.2f MB" % (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    dst = probe(out)
    check(abs(dst["dur"] - src["dur"]) <= 0.3, "a 时长≈源",
          "源 %.2fs / 成片 %.2fs" % (src["dur"], dst["dur"]))
    check(abs(dst["frames"] - src["frames"]) <= 1, "a 帧数≈源",
          "源 %d / 成片 %d" % (src["frames"], dst["frames"]))
    check((cache / "meta.json").is_file(), "a meta.json 记着六个阶段",
          str(sorted((json.loads((cache / "meta.json").read_text("utf-8"))
                      .get("stages") or {}).keys())))
    return {"cache": cache, "out": out, "src": src, "settings": body["settings"],
            "profile": prof, "refs": refs}


# ============================================================
# 案例 B：409 闸门
# ============================================================
def case_b_gates(video):
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM,
            "settings": {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay"}}
    fresh = Path(str(video.parent / (Path(video).stem + "_gates_cache")))
    shutil.rmtree(fresh, ignore_errors=True)
    b = dict(body)
    b["cache_dir"] = str(fresh)
    # 空 cache 直接点合成
    st, d = post("/api/merge/stage/merge", b)
    check(st == 409, "e 没切帧就合成 → 409", "HTTP %s · %s" % (st, (d.get("error") or "")[:90]))
    run_stage("frames", b)
    st, d = post("/api/merge/stage/pred", b)
    check(st == 409, "e 没切脸就换脸 → 409", "HTTP %s · %s" % (st, (d.get("error") or "")[:90]))
    run_stage("faces", b)
    st, d = post("/api/merge/stage/mask", b)
    check(st == 409, "e 没换脸就遮罩 → 409", "HTTP %s · %s" % (st, (d.get("error") or "")[:90]))
    st, d = post("/api/merge/stage/merge", b)
    check(st == 409, "e 没换脸/没遮罩就合成 → 409", "HTTP %s · %s" % (st, (d.get("error") or "")[:90]))
    run_stage("pred", b)
    st, d = post("/api/merge/stage/merge", b)
    check(st == 409 and "遮罩" in str(d.get("error") or ""),
          "e 有换脸没遮罩就合成 → 409（点名先跑遮罩阶段）",
          "HTTP %s · %s" % (st, (d.get("error") or "")[:110]))
    run_stage("mask", b)
    st, d = post("/api/merge/stage/encode", b)
    check(st == 409, "e 没合成就编码 → 409", "HTTP %s · %s" % (st, (d.get("error") or "")[:90]))

    # ---- 遮罩来源改了 → mask/ 失效 ----
    b2 = dict(b)
    b2["settings"] = {"mask_mode": 4, "color_transfer": "mt", "blend": "overlay"}
    st, d = post("/api/merge/stage/merge", b2)
    check(st == 409 and "遮罩" in str(d.get("error") or ""),
          "e 中途改遮罩来源 → 合成 409（不许静默复用旧 mask/）",
          "HTTP %s · %s" % (st, (d.get("error") or "")[:110]))
    st, d = post("/api/merge/stage/mask", b2)
    check(st == 409, "e 中途改遮罩来源 → 遮罩阶段 409（要求 force 重跑）",
          "HTTP %s · %s" % (st, (d.get("error") or "")[:110]))
    b2["force"] = True
    st, d, fin = run_stage("mask", b2)
    check(st == 200 and (fin.get("phase") == "done"),
          "e force=true 重跑遮罩 → 成功（来源 DFM，不跑推理）",
          "inference=%s" % (((fin.get("meta") or {}).get("stages", {})
                             .get("mask", {}) or {}).get("inference")))
    mm = ((fin.get("meta") or {}).get("stages", {}).get("mask") or {})
    check(mm.get("inference") is False and mm.get("signature") == "dfm",
          "e DFM 来源的遮罩阶段不推理、但照常落盘", "signature=%s · count=%s"
          % (mm.get("signature"), mm.get("count")))
    n_dst = len([p for p in imgs(fresh / "mask") if p.stem.endswith("_dfm_dst")])
    check(n_dst >= 1, "e DFM 遮罩落盘（便于检查）", "%d 张 _dfm_dst" % n_dst)
    st, d, fin = run_stage("merge", b2)
    check(st == 200 and fin.get("phase") == "done", "e 换成 DFM 遮罩后合成正常",
          "fps=%s" % ((((fin.get("meta") or {}).get("stages", {}).get("merge") or {})
                       .get("profile") or {}).get("fps")))
    shutil.rmtree(fresh, ignore_errors=True)


# ============================================================
# 案例 C：③④ 取消后从游标续跑
# ============================================================
def case_c_resume(video):
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM,
            "settings": {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay"}}
    fresh = Path(str(video.parent / (Path(video).stem + "_resume_cache")))
    b = dict(body)
    b["cache_dir"] = str(fresh)
    shutil.rmtree(fresh, ignore_errors=True)
    run_stage("frames", b)
    run_stage("faces", b)

    for stage in ("pred", "mask"):
        st, d = post("/api/merge/stage/%s" % stage, b)
        seen = 0
        t0 = time.time()
        while time.time() - t0 < 600:
            _s, s = get("/api/merge/stage/status", timeout=60)
            if not s.get("running"):
                break
            if int(s.get("frames_done") or 0) >= 2:
                post("/api/merge/stage/cancel", {})
                seen = int(s.get("frames_done") or 0)
                break
            time.sleep(0.15)
        fin = {}
        t0 = time.time()
        while time.time() - t0 < 300:
            _s, s = get("/api/merge/stage/status", timeout=60)
            if not s.get("running"):
                fin = s
                break
            time.sleep(0.3)
        check(fin.get("phase") == "cancelled", "f %s 取消成功（等价断电）" % stage,
              "phase=%s · 停在第 %s 张" % (fin.get("phase"), seen))
        disk = json.loads((fresh / "meta.json").read_text("utf-8"))
        d_st = (disk.get("stages") or {}).get(stage) or {}
        cur = int(d_st.get("done_frames") or 0)
        check(cur > 0 and not d_st.get("done"), "f %s 游标落在中间" % stage,
              "done_frames=%s · done=%s" % (cur, d_st.get("done")))
        st, d, fin = run_stage(stage, b)
        newcur = int(((fin.get("meta") or {}).get("stages", {}).get(stage) or {})
                     .get("resumed_from") or 0)
        check(int(d_st.get("done_frames") or 0) == newcur,
              "f %s 从游标续跑（resumed_from = 取消时那一格）" % stage,
              "resumed_from=%s · 取消时 %s" % (newcur, cur))
        check((fin.get("meta") or {}).get("stages", {}).get(stage, {}).get("done") is True,
              "f %s 续跑后完成" % stage)
    shutil.rmtree(fresh, ignore_errors=True)


# ============================================================
# 案例 D：成片 vs 单帧预览（质量不回退）
# ============================================================
def case_d_quality(video, out, settings, src, cache=None, refs=None):
    if cache is not None and refs:
        import numpy as np
        worst_mean, worst_max = 0.0, 0
        for n, ref in sorted(refs.items()):
            # 预览走一步式 MergeMaskedFace（worker 里 dfm + 遮罩 + 融合一次算完，
            # 内存出图、不碰 <视频>_cache/），导出走六阶段（pred/mask 落盘 + 纯 CPU
            # compose_frame）。两条路算法等价但**不是同一条代码路径**，所以这里量的是
            # 「差值在容差内」而不是逐像素相同 —— 实测 mean 0.02~0.03 / max 20~33
            # （融合只差脸边那几级灰度；用户说的 ~3.7 是成片编码往返，见下面的 g 段）。
            # （以前这里靠 frame_stem+cache_dir 让预览读缓存里的 pred/mask，才凑出 0 差；
            #   那条缝已经拆掉，预览永远不依赖 cache 存在与否。）
            st, pv = post("/api/merge/preview-frame",
                          {"frame_file": str(ref), "fmt": "png",
                           "models": [{"name": DFM, "faces": []}],
                           "settings": settings}, timeout=300)
            got = read_png(pv.get("png") or "") if st == 200 else None
            disk = imread_any(cache / "frames" / ("%06d.png" % (n + 1)))
            if got is None or disk is None:
                check(False, "a 第 %d 帧 预览取不到" % (n + 1), str((pv or {}).get("error"))[:120])
                continue
            mean, p99, mx = diff(got, disk)
            worst_mean = max(worst_mean, mean)
            worst_max = max(worst_max, mx)
            tm = pv.get("timings") or {}
            check(mean <= TOL_VIDEO and bool(tm.get("swap_ms")) and "compose_path" not in tm,
                  "a 第 %d 帧 预览(一步式) vs 导出帧（容差 %d，两条代码路径）" % (n + 1, TOL_VIDEO),
                  "mean=%.2f p99=%.1f max=%d · swap_ms=%s" % (mean, p99, mx, tm.get("swap_ms")))
        check(worst_mean <= TOL_VIDEO,
              "a 三帧预览 vs 导出帧都在容差内（无损对拍实测 mean 0.02~0.03）",
              "worst mean=%.2f max=%d" % (worst_mean, worst_max))
    picked = [5, 20, 45]
    ex = decode_all(out)
    fps = src["fps"] or 30.0
    worst = 0.0
    for n in picked:
        if n >= len(ex):
            continue
        st, pv = post("/api/merge/preview",
                      {"video": str(video), "t": round(n / max(1.0, fps), 4),
                       "models": [{"name": DFM, "faces": []}], "settings": settings},
                      timeout=300)
        ref = read_png(pv.get("png") or "") if st == 200 else None
        if ref is None:
            check(False, "g 第 %d 帧取不到预览" % n, str((pv or {}).get("error"))[:120])
            continue
        mean, p99, mx = diff(ref, ex[n])
        worst = max(worst, mean)
        check(mean <= TOL_VIDEO, "g 第 %d 帧 成片 vs 单帧预览（容差 %d）" % (n, TOL_VIDEO),
              "mean=%.2f p99=%.1f max=%d" % (mean, p99, mx))
    check(worst <= TOL_VIDEO, "g 三帧里最差的一帧也在容差内", "worst mean=%.2f" % worst)


# ============================================================
# 案例 E：和 4.0 逐像素对拍（真跑 4.0 的 _phase_merge_worker）
# ============================================================
REF_DRIVER = r'''
# -*- coding: utf-8 -*-
"""在 4.0 的 python 里跑 MergeStudio.core.export_pipeline._phase_merge_worker。"""
import json, sqlite3, sys
from pathlib import Path
ENGINE = Path(sys.argv[1]); WORK = Path(sys.argv[2])
sys.path.insert(0, str(ENGINE))
import MergeStudio.core.export_pipeline as ep

cfg = json.loads((WORK / "cfg.json").read_text("utf-8"))
swap_dir = WORK / "swap"; mask_dir = WORK / "mask"
frames_dir = WORK / "frames"; out_dir = WORK / "out40"
out_dir.mkdir(parents=True, exist_ok=True)
db = WORK / "faces.sqlite"
if db.exists():
    db.unlink()
ep._init_db(db)
conn = sqlite3.connect(str(db))
for row in cfg["faces"]:
    ep._db_insert_face(conn, row["frame_idx"], row["face_idx"],
                       row["rect"], row["landmarks"], None, 416)
conn.commit()
conn.close()
ep._phase_merge_init(str(swap_dir), str(mask_dir), str(frames_dir), str(db),
                     [(r["frame_idx"], r["face_idx"], json.dumps(r["landmarks"]), "") for r in cfg["faces"]],
                     cfg["config"], str(out_dir), int(cfg["swap_sz"]))
done = []
for fr in cfg["frames"]:
    done.append(ep._phase_merge_worker(str(frames_dir / fr)))
print("OK", done)
'''


def case_e_vs40(cache, video, settings, src):
    """同一帧同一参数：4.0 的合成 vs 我们的合成，逐像素对比。"""
    if not (ENGINE / "MergeStudio" / "core" / "export_pipeline.py").is_file():
        say("找不到 4.0（%s）—— 这一段跳过" % ENGINE)
        return
    work = BOX / "ref40"
    shutil.rmtree(work, ignore_errors=True)
    for d in ("swap", "mask", "frames", "out40"):
        (work / d).mkdir(parents=True, exist_ok=True)
    frames = sorted((cache / "frames").glob("*.png"))
    aligned = sorted((cache / "aligned").glob("*.jpg"))
    sys.path.insert(0, str(APP))
    from server._align import read_dfljpg_meta

    cfg = {"mask_mode": int(settings.get("mask_mode", 6)),
           "color_transfer_mode": str(settings.get("color_transfer", "mt")),
           "mode": str(settings.get("blend", "overlay")),
           "erode_mask_modifier": int(settings.get("mask_erode", 0)),
           "blur_mask_modifier": int(settings.get("mask_blur", 0)),
           "hist_match_threshold": 255}
    picked = [4, 19, 44]
    picks = [fi for fi in picked if fi < len(frames)]
    rows, names = [], []
    import numpy as np
    import cv2
    # 原帧（合成前）+ 换脸图 + 遮罩，按 4.0 的命名铺开
    for fi in picks:
        fname = frames[fi].name
        idx = int(frames[fi].stem)
        aps = [a for a in aligned if a.stem.split("_")[0] == frames[fi].stem]
        if not aps:
            continue
        ap = aps[0]
        face_idx = int(ap.stem.split("_")[1])
        m = read_dfljpg_meta(ap) or {}
        lm = m.get("source_landmarks") or m.get("landmarks")
        shutil.copy2(str(frames[fi]), str(work / "frames" / fname))
        pred = imread_any(cache / "pred" / (ap.stem + ".png"))
        cv2.imwrite(str(work / "swap" / ("%08d_%d.jpg" % (idx, face_idx))), pred,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        for tag, name in (("_xseg_prd", "xseg_prd"), ("_xseg_dst", "xseg_dst")):
            src_p = cache / "mask" / (ap.stem + tag + ".png")
            if src_p.is_file():
                shutil.copy2(str(src_p), str(work / "mask" / ("%08d_%d_%s.png"
                                                             % (idx, face_idx, name))))
        rows.append({"frame_idx": idx, "face_idx": face_idx,
                     "rect": m.get("source_rect") or [0, 0, 0, 0],
                     "landmarks": lm})
        names.append((fi, fname, ap))
    if not names:
        say("没有可对拍的帧")
        return
    (work / "cfg.json").write_text(json.dumps({
        "config": cfg, "faces": rows, "swap_sz": int(cfg.get("size") or 416),
        "frames": [n[1] for n in names]}, ensure_ascii=False), "utf-8")
    drv = work / "ref_driver.py"
    drv.write_text(REF_DRIVER, "utf-8")
    r = subprocess.run([str(ENGINE_PY), str(drv), str(ENGINE), str(work)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "OK" not in (r.stdout or ""):
        check(False, "h 4.0 的 synthesize 没跑起来", (r.stdout or "")[-200:] + (r.stderr or "")[-400:])
        return
    say("4.0 跑完：%s" % (r.stdout or "").strip()[:120])

    # 我们这一侧：用同一份输入，roi=True / roi=False 各跑一遍
    import server.app  # noqa: F401
    from server import _merge_compose as MC
    from server.routes_merge import _stage_read_image
    MC.ensure_engine_path()
    worst = {"mean": 0.0, "max": 0}
    for fi, fname, ap in names:
        frame = _stage_read_image(cache / "frames" / fname)   # frames 已被就地覆盖！
        # 用 4.0 那份原始帧当输入（合成前）
        frame = imread_any(work / "frames" / fname)
        face = {"landmarks": np.asarray(rows[[n[1] for n in names].index(fname)]["landmarks"],
                                       np.float32),
                "pred": imread_any(cache / "pred" / (ap.stem + ".png")).astype(np.float32) / 255.0,
                "xseg_prd": (read_gray(cache / "mask" / (ap.stem + "_xseg_prd.png"))
                             .astype(np.float32) / 255.0),
                "xseg_dst": (read_gray(cache / "mask" / (ap.stem + "_xseg_dst.png"))
                             .astype(np.float32) / 255.0)}
        ours = MC.compose_frame(frame.copy(), [face], cfg, {})
        theirs = imread_any(work / "out40" / fname)
        if theirs is None:
            check(False, "h 4.0 没产出第 %d 帧" % (fi + 1), str(work / "out40"))
            continue
        # 4.0 的 swap 图是 JPEG(q100) 存的，它读回来的也是 JPEG —— 让两边输入一致
        jpg = imread_any(work / "swap" / ("%08d_%d.jpg" % (int(Path(fname).stem),
                                                           int(ap.stem.split("_")[1]))))
        face["pred"] = jpg.astype(np.float32) / 255.0
        ours = MC.compose_frame(frame.copy(), [face], cfg, {})
        mean, p99, mx = diff(theirs, ours)
        worst["mean"] = max(worst["mean"], mean)
        worst["max"] = max(worst["max"], mx)
        check(mean <= TOL_40 and mx <= MAX_40,
              "h 第 %d 帧 4.0 vs 我们（mean≤%d max≤%d）" % (fi + 1, TOL_40, MAX_40),
              "mean=%.3f p99=%.1f max=%d" % (mean, p99, mx))
    check(worst["mean"] <= TOL_40 and worst["max"] <= MAX_40,
          "h 三帧最差也在阈值内", "worst mean=%.3f max=%d" % (worst["mean"], worst["max"]))


# ============================================================
# main
# ============================================================
def main():
    t_all = time.time()
    prepare_sandbox()
    src_video = sorted((REPO / "workspace").glob("*.mp4"))[0]
    video = MEDIA / "clip.mp4"
    run_ff(["-ss", "3", "-t", "2.0", "-i", src_video, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", video],
           "切样片")
    info = probe(video)
    say("样片：%s" % info)
    if info["w"] != 1080 or info["h"] != 1920:
        raise SystemExit("样片不是 1080x1920：%s" % info)

    proc = start_server("六阶段")
    try:
        a = case_a(video)
        case_d_quality(video, a["out"], a["settings"], a["src"],
                       cache=a["cache"], refs=a["refs"])
        case_e_vs40(a["cache"], video, a["settings"], a["src"])
        case_b_gates(video)
        case_c_resume(video)
    finally:
        stop_server(proc)

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    if a.get("profile"):
        print("合成 profile：%s" % json.dumps(a["profile"].get("per_frame_ms"),
                                            ensure_ascii=False))
        print("合成 fps：%s（线程 %s · PNG level %s）"
              % (a["profile"].get("fps"), a["profile"].get("threads"),
                 a["profile"].get("png_level")))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
