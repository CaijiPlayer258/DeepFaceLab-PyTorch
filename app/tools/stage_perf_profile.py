# -*- coding: utf-8 -*-
r"""③换脸(pred) / ④遮罩(mask) 两阶段的逐段 profile + 后端判定（原生 TRT / ort-CUDA / ort-CPU）。

为什么要有这个工具：用户报「pred 5.8~6.2 帧/秒，mask 26 张/秒，而 XSegLite 单张推理
应该有 ~100 it/s」。要改就得先知道每一段各花多少毫秒，而且必须把「这次到底跑在哪个
后端」量出来（不猜）。这个脚本**不改生产代码**，只做三件事：

  1) bench  —— 纯推理微基准：dfm 的 batch 1/4/8、XSegLite 的 原生TRT(int8/bf16) /
               ort-CUDA(batch 1/N) / ort-CPU，外加 PNG 编码、PNG/JPEG 解码、warpAffine。
  2) stages —— 在 %TEMP% 沙箱里跑真后端（真视频 + 真模型）：frames → faces →
               pred → mask，每个阶段跑完从 meta.json 里把分成两半的耗时打出来。
  3) 两个都能只跑其中一个（--only bench / --only stages）。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\stage_perf_profile.py
    runtime\server\Scripts\python.exe app\tools\stage_perf_profile.py --only bench
    runtime\server\Scripts\python.exe app\tools\stage_perf_profile.py --only stages --seconds 6
    runtime\server\Scripts\python.exe app\tools\stage_perf_profile.py --stage pred --force --no-trt

退出码 0 = 跑完（不是断言）。
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
BOX = TEMP / "dfl_stage_perf"
WS = BOX / "ws"
MEDIA = BOX / "media"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"
DFM = "\u963f\u8a3c\u8a3c\u8a3c\u8a3c-416"

BASE = ""
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ------------------------------------------------------------------ 小工具
def say(msg=""):
    print(msg, flush=True)


def head(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


def table(rows, cols):
    """rows: [[c0, c1, ...]]；cols: [(标题, 宽度, 对齐)]"""
    line = "  ".join((h.ljust(w) if a == "<" else h.rjust(w)) for h, w, a in cols)
    say(line)
    say("-" * len(line))
    for r in rows:
        cells = []
        for (h, w, a), v in zip(cols, r):
            s = "" if v is None else str(v)
            cells.append(s.ljust(w) if a == "<" else s.rjust(w))
        say("  ".join(cells))


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
        except Exception:                                       # noqa: BLE001
            return e.code, {"error": blob[:300].decode("utf-8", "replace")}


# ------------------------------------------------------------------ bench
def bench():
    head("bench · 纯推理 / I-O 微基准（解释器：%s）" % sys.executable)
    import numpy as np
    import cv2
    import onnxruntime as ort

    sys.path.insert(0, str(APP))
    from server import app as A                              # noqa: F401  先建 router
    from server import routes_merge as rm

    rm._ensure_ort_dll_dirs()
    say("onnxruntime %s · providers=%s · device=%s"
        % (ort.__version__, ort.get_available_providers(), ort.get_device()))
    ws = Path(os.environ.get("DFL_WORKSPACE") or (REPO / "workspace"))
    say("workspace=%s" % ws)

    def fmt(v, n=2):
        return round(float(v), n)

    # ---- dfm 纯推理 ----
    dfm = ws / "model" / (DFM + ".dfm")
    head("dfm 纯推理：%s" % dfm.name)
    dfm_rows = []
    if not dfm.is_file():
        say("模型不存在，跳过 dfm 基准")
    else:
        sizes = {}
        for label, provs in (("ort-CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                             ("ort-CPU", ["CPUExecutionProvider"])):
            try:
                sess = ort.InferenceSession(str(dfm), providers=provs)
            except Exception as e:                          # noqa: BLE001
                say("%s 建会话失败：%r" % (label, e))
                continue
            active = sess.get_providers()
            shape = list(sess.get_inputs()[0].shape)
            nchw = len(shape) >= 4 and shape[1] in (1, 3)
            size = int(shape[2] if nchw else shape[1])
            name = sess.get_inputs()[0].name
            sizes[label] = size
            _batches = (1, 4, 8) if label == "ort-CUDA" else (1,)
            for b in _batches:
                x = np.zeros((b, size, size, 3), np.float32)
                if nchw:
                    x = np.transpose(x, (0, 3, 1, 2))
                for _ in range(2):
                    sess.run(None, {name: x})
                n = (10 if b == 1 else 6) if label == "ort-CUDA" else 3
                t0 = time.perf_counter()
                for _ in range(n):
                    sess.run(None, {name: x})
                dt = (time.perf_counter() - t0) * 1000.0
                dfm_rows.append([label + "(%s)" % active[0], "%dx%d" % (size, size),
                                 b, fmt(dt / n, 1), fmt(dt / n / b, 2),
                                 fmt(b / (dt / n / 1000.0), 1)])
                if label == "ort-CUDA" and b == 1:
                    break_all = False
            # CPU 只测 batch1，其余没意义
            if label == "ort-CPU":
                break
        # CPU 单张单测（上面循环里对 CPU 只跑了 b=1 就 break）
        try:
            sess = ort.InferenceSession(str(dfm), providers=["CPUExecutionProvider"])
            shape = list(sess.get_inputs()[0].shape)
            nchw = len(shape) >= 4 and shape[1] in (1, 3)
            size = int(shape[2] if nchw else shape[1])
            name = sess.get_inputs()[0].name
            x = np.zeros((1, size, size, 3), np.float32)
            if nchw:
                x = np.transpose(x, (0, 3, 1, 2))
            sess.run(None, {name: x})
            t0 = time.perf_counter()
            for _ in range(5):
                sess.run(None, {name: x})
            dt = (time.perf_counter() - t0) * 1000.0 / 5.0
            if not any(r[0].startswith("ort-CPU") for r in dfm_rows):
                dfm_rows.append(["ort-CPU(CPUExecutionProvider)", "%dx%d" % (size, size),
                                 1, fmt(dt, 1), fmt(dt, 2), fmt(1000.0 / dt, 1)])
        except Exception as e:                              # noqa: BLE001
            say("CPU 单张基准失败：%r" % (e,))
    if dfm_rows:
        table(dfm_rows, [("后端", 40, "<"), ("输入", 9, "<"), ("batch", 5, ">"),
                         ("一批 ms", 9, ">"), ("每张 ms", 9, ">"), ("张/秒", 8, ">")])

    # ---- XSeg 纯推理 ----
    head("XSegLite 纯推理（256x256）")
    onnx = ws / "model" / "XSegLite" / "xseglite.onnx"
    x_rows = []
    if not onnx.is_file():
        say("没有 %s，跳过" % onnx)
    else:
        face = (np.random.RandomState(0).rand(256, 256, 3) * 255).astype(np.uint8)

        def bench_masker(masker, label, iters=40):
            for _ in range(5):
                masker.mask(face, out_size=256)
            t0 = time.perf_counter()
            for _ in range(iters):
                masker.mask(face, out_size=256)
            dt = (time.perf_counter() - t0) * 1000.0 / iters
            info = getattr(masker, "info", {}) or {}
            return [label, info.get("backend") or getattr(masker, "backend", "?"),
                    Path(str(info.get("engine") or "")).name or "-", fmt(dt, 2),
                    fmt(1000.0 / dt, 1)]

        from server import _xseg as X
        from server import _engines as E
        info = {}
        try:
            from server._trt_session import TrtSession
            sess = E.try_engine(ws, "xseglite", onnx, precision=("int8", "bf16"),
                                use_graph=False, info=info)
        except Exception as e:                              # noqa: BLE001
            sess = None
            say("try_engine 抛异常：%r" % (e,))
        say("try_engine → backend=%s engine=%s reason=%s"
            % (info.get("backend"), info.get("engine"), str(info.get("reason"))[:160]))
        if sess is not None:
            def _trt_mask(sess_):
                inp = sess_.get_inputs()[0].name
                meta = dict(info)

                class _W(object):
                    info = meta

                    def mask(self, img, out_size=256):
                        x = img if img.shape[:2] == (256, 256) else cv2.resize(img, (256, 256))
                        blob = cv2.dnn.blobFromImage(x, 1.0 / 255.0, (256, 256), swapRB=True)
                        out = sess_.run(None, {inp: blob})[0]
                        return np.clip(np.squeeze(out).astype(np.float32), 0, 1)
                return _W()
            x_rows.append(bench_masker(_trt_mask(sess), "原生TRT"))
            # batch 能力
            try:
                say("原生TRT 引擎 max_batch=%s · use_graph 关" % sess.max_batch())
            except Exception:                               # noqa: BLE001
                pass
        # ort-CUDA 单张与批量
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            osess = ort.InferenceSession(str(onnx), sess_options=so,
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            say("XSeg ort 会话 providers=%s input=%s"
                % (osess.get_providers(), osess.get_inputs()[0].shape))

            class _O:
                info = {"backend": "ort-cuda", "engine": ""}

                def __init__(self, s):
                    self.s = s
                    self.name = s.get_inputs()[0].name

                def mask(self, img, out_size=256):
                    x = img if img.shape[:2] == (256, 256) else cv2.resize(img, (256, 256))
                    blob = cv2.dnn.blobFromImage(x, 1.0 / 255.0, (256, 256), swapRB=True)
                    out = self.s.run(None, {self.name: blob})[0]
                    return np.clip(np.squeeze(out).astype(np.float32), 0, 1)

            x_rows.append(bench_masker(_O(osess), "ort-CUDA"))
            # 批量：直接喂 blob 拼 batch
            name = osess.get_inputs()[0].name
            for b in (2, 4, 8, 16):
                blob = np.zeros((b, 3, 256, 256), np.float32)
                try:
                    osess.run(None, {name: blob})
                except Exception as e:                      # noqa: BLE001
                    x_rows.append(["ort-CUDA 批量 B%d" % b, "ort", "-", "失败", str(e)[:40]])
                    continue
                n = 10
                t0 = time.perf_counter()
                for _ in range(n):
                    osess.run(None, {name: blob})
                dt = (time.perf_counter() - t0) * 1000.0 / n
                x_rows.append(["ort-CUDA 批量 B%d" % b, "ort-cuda", "-",
                               fmt(dt / b, 2), fmt(b / (dt / 1000.0), 1)])
        except Exception as e:                              # noqa: BLE001
            say("ort-CUDA 基准失败：%r" % (e,))
        # ort-CPU
        try:
            csess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])

            class _C:
                info = {"backend": "ort-cpu", "engine": ""}

                def __init__(self, s):
                    self.s = s
                    self.name = s.get_inputs()[0].name

                def mask(self, img, out_size=256):
                    x = img if img.shape[:2] == (256, 256) else cv2.resize(img, (256, 256))
                    blob = cv2.dnn.blobFromImage(x, 1.0 / 255.0, (256, 256), swapRB=True)
                    out = self.s.run(None, {self.name: blob})[0]
                    return np.clip(np.squeeze(out).astype(np.float32), 0, 1)

            x_rows.append(bench_masker(_C(csess), "ort-CPU", iters=12))
        except Exception as e:                              # noqa: BLE001
            say("ort-CPU 基准失败：%r" % (e,))
    if x_rows:
        table(x_rows, [("路径", 22, "<"), ("后端", 10, "<"), ("引擎文件", 46, "<"),
                       ("每张 ms", 9, ">"), ("张/秒", 8, ">")])

    # ---- I/O：PNG 编码 / 解码 / warp ----
    head("I/O 微基准（这张机器上的绝对值）")
    io_rows = []
    color416 = (np.random.RandomState(1).rand(416, 416, 3) * 255).astype(np.uint8)
    gray416 = (np.random.RandomState(2).rand(416, 416) * 255).astype(np.uint8)
    gray256 = (np.random.RandomState(3).rand(256, 256) * 255).astype(np.uint8)
    for label, img, lvl, n in (("PNG 编码 416x416x3 level1", color416, 1, 60),
                               ("PNG 编码 416x416x3 level3", color416, 3, 60),
                               ("PNG 编码 416x416 灰度 level1", gray416, 1, 60),
                               ("PNG 编码 256x256 灰度 level1", gray256, 1, 120),
                               ("PNG 编码 256x256 灰度 level3", gray256, 3, 120)):
        for _ in range(5):
            cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), lvl])
        t0 = time.perf_counter()
        for _ in range(n):
            ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), lvl])
        dt = (time.perf_counter() - t0) * 1000.0 / n
        io_rows.append([label, fmt(dt, 2), "%d KB" % (len(buf) // 1024)])
    frame = (np.random.RandomState(4).rand(1920, 1080, 3) * 255).astype(np.uint8)
    ok, pngbuf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    ok, jpgbuf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    for label, buf in (("PNG 解码 1080x1920（%.1f MB）" % (len(pngbuf) / 1048576.0), pngbuf),
                       ("JPEG 解码 1080x1920", jpgbuf)):
        for _ in range(2):
            cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        for _ in range(10):
            cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        dt = (time.perf_counter() - t0) * 1000.0 / 10
        io_rows.append([label, fmt(dt, 2), "-"])
    f32 = frame.astype(np.float32) / 255.0
    mat = np.array([[0.15, 0, 100], [0, 0.15, 100]], np.float32)
    for label, src in (("warpAffine 1080p uint8→416", frame),
                       ("warpAffine 1080p float32→416", f32)):
        for _ in range(2):
            cv2.warpAffine(src, mat, (416, 416), flags=cv2.INTER_CUBIC)
        t0 = time.perf_counter()
        for _ in range(10):
            cv2.warpAffine(src, mat, (416, 416), flags=cv2.INTER_CUBIC)
        dt = (time.perf_counter() - t0) * 1000.0 / 10
        io_rows.append([label, fmt(dt, 2), "-"])
    t0 = time.perf_counter()
    for _ in range(10):
        x = f32 * 1.0
    io_rows.append(["astype(np.float32)/255 1080p", fmt((time.perf_counter() - t0) * 1000.0 / 10, 2), "-"])
    table(io_rows, [("操作", 40, "<"), ("ms", 9, ">"), ("备注", 12, ">")])


# ------------------------------------------------------------------ stages
def prepare_sandbox(link_engines=True):
    shot = []
    if BOX.exists():
        for p in (WS / "model", WS / "_engines"):
            if p.exists():
                subprocess.run(["cmd", "/c", "rmdir", str(p)], capture_output=True)
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    for name, target in (("model", REPO / "workspace" / "model"),
                         ("_engines", REPO / "workspace" / "_engines")):
        if name == "_engines" and not link_engines:
            continue
        if not target.exists():
            continue
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / name), str(target)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
        shot.append(name)
    say("沙箱：%s（联接 %s）" % (BOX, ",".join(shot)))


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
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
            if text.startswith("[stage]") or text.startswith("[TRT]"):
                say("  | " + text.rstrip())
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
            st, _d = http("GET", "/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但状态接口不应答")


def stop_server(proc):
    if proc is None:
        return
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(proc.pid))],
                   capture_output=True, timeout=30)
    time.sleep(1.0)


def run_stage(name, body, timeout=3600.0, watch=1.0, tag=""):
    st, d = http("POST", "/api/merge/stage/%s" % name, body)
    if st != 200 or d.get("ok") is False:
        return st, d, {}
    if d.get("skipped"):
        return st, d, d
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout:
        _s, s = http("GET", "/api/merge/stage/status", timeout=60)
        if not s.get("running"):
            return st, d, s
        el = time.time() - t0
        if watch and el - last >= watch:
            last = el
            say("   [%s %5.1fs] %s" % (tag or name, el, str(s.get("msg") or "")[:150]))
        time.sleep(0.3)
    return st, d, {}


def run_ff(args, what=""):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] \
        + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-400:]))


def stages(args):
    import cv2
    p = None
    prepare_sandbox(link_engines=not args.no_trt)
    src_video = sorted((REPO / "workspace").glob("*.mp4"))[0]
    video = MEDIA / "clip.mp4"
    t0 = args.start
    run_ff(["-ss", t0, "-t", args.seconds, "-i", src_video, "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", video],
           "切样片")
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    say("样片：%s 帧 %dx%d" % (n_frames, w, h))

    body = {"video": str(video), "model": DFM,
            "settings": {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay",
                         "mask_erode": 0, "mask_blur": 4}}
    for kv in (getattr(args, "knob", None) or []):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        body[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
        say("旋钮：%s=%s" % (k.strip(), v.strip()))
    cache = video.parent / (video.stem + "_cache")

    proc = start_server("profile")
    try:
        for name in ("frames", "faces"):
            st, d, fin = run_stage(name, body)
            if st != 200:
                raise SystemExit("阶段 %s 失败：%s" % (name, json.dumps(d, ensure_ascii=False)[:400]))
        stages_to_run = list(args.stage)
        if "mask" in stages_to_run and "pred" not in stages_to_run:
            stages_to_run.insert(0, "pred")      # 遮罩阶段要 pred/ 的产物
        for name in stages_to_run:
            meta0 = json.loads((cache / "meta.json").read_text("utf-8"))
            rng = (meta0.get("stages") or {}).get(name) or {}
            b = dict(body)
            b["force"] = bool(args.force)
            st, d, fin = run_stage(name, b)
            if st != 200:
                say("阶段 %s 失败：%s" % (name, json.dumps(d, ensure_ascii=False)[:600]))
                continue
            m = ((json.loads((cache / "meta.json").read_text("utf-8"))
                  .get("stages") or {}).get(name) or {})
            head("阶段 %s 结果" % name)
            keys = ("done", "count", "frames", "total", "faces_total", "seconds", "fps",
                    "faces_per_sec", "frame_fps", "batch", "batch_source", "readers",
                    "writers", "backend", "engine", "engine_file", "fallback_reason",
                    "mask_label", "infer_ms_per_face", "infer_seconds", "signature")
            for k in keys:
                if k in m:
                    say("  %-18s %s" % (k, m.get(k)))
            prof = m.get("profile") or {}
            if prof:
                say("  profile raw        %s" % json.dumps(prof, ensure_ascii=False))
            bm = m.get("batch_meta") or {}
            if bm.get("ladder"):
                say("  batch 阶梯         %s" % json.dumps(bm.get("ladder"), ensure_ascii=False))
    finally:
        stop_server(proc)
    say("")
    say("后端日志：%s" % SERVER_LOG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="both", choices=("bench", "stages", "both"))
    ap.add_argument("--stage", action="append", default=None,
                    help="只跑这几个阶段（pred / mask）")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--start", type=float, default=3.0)
    ap.add_argument("--force", action="store_true", default=True)
    ap.add_argument("--no-trt", action="store_true",
                    help="沙箱不联接 workspace/_engines（模拟没有 TRT 引擎的机器）")
    ap.add_argument("--knob", action="append", default=[],
                    help="给阶段请求体加一个键值（mask_readers=6 这类），可重复")
    args = ap.parse_args()
    args.stage = args.stage or ["pred", "mask"]
    if args.only in ("bench", "both"):
        bench()
    if args.only in ("stages", "both"):
        stages(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
