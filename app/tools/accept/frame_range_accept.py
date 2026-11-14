# -*- coding: utf-8 -*-
r"""帧范围的验收（方案 B）：真后端 + 真视频 + 现有 dfm，全部在 %TEMP% 里跑。

口径（这一轮定的）：
  · 切帧（frames）**仍然整段**（索引完整、成本极低），只在 meta 里记 range；
  · 后五个阶段（faces / pred / mask / merge / encode）**只处理 [start_frame, end_frame]**，
    1 起、闭区间；都不传 = 全片；只给 start = start..结尾；只给 end = 1..end；
  · 游标（done_frames）相对范围：0..N，N = end-start+1；
  · 成片时长 = N/fps，音轨用 -ss (start-1)/fps -t N/fps 从源视频切同一段；
  · 输出文件名带范围后缀（xxx_merged_0100-0200.mp4），全片不加；
  · 改范围 → 阶段指纹（range_key）变 → 旧游标作废/上游覆盖不足就 409。

五组验收（本脚本逐条断言并打印证据）：
  a) start=100,end=200：frames 仍=总帧数；faces/pred/mask/merge 只处理 101 帧；
     成片时长 = 101/fps ±1 帧；有音轨且起点与源视频第 100 帧对齐（交叉相关定位 + 抽帧比对）。
  b) 只给 start=100 → 处理到结尾，帧数 = 总数-99。
  c) 只给 end=200   → 处理 1..200。
  d) 非法：start=300,end=100 → 422；end>总帧数 → 422；start=0 → 422。
  e) 全片（不传范围）→ 与以前完全一致：文件名不带后缀、帧数/时长=全片、六阶段全过。
  f) 改范围后旧阶段失效：mask 要 1..50（pred 只跑过 100..200）→ 409 点名换脸；
     mask 改到 100..150 → 自动重跑（range_key 变、日志「范围变了」）；同范围再点 → 跳过。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\frame_range_accept.py
    # 只想看某一组：--only a,d

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
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_frame_range_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"
CLIP = MEDIA / "clip.mp4"

DFM = "阿証証証証-416"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "overlay",
            "mask_erode": 0, "mask_blur": 4}
SR = 16000                  # 音轨比对用的采样率
TONE_BLOCK = 0.25           # 合成音轨每 0.25s 换一个频率（错位就对不上）
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


# ============================================================
# HTTP
# ============================================================
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


def run_stage(name, body, timeout=3600.0):
    """跑一个阶段并等它结束：返回 (http 状态, 请求回包, 终态 status)。"""
    st, d = post("/api/merge/stage/%s" % name, body, timeout=120)
    if st != 200:
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


def stage_meta(status, name):
    return (((status or {}).get("meta") or {}).get("stages") or {}).get(name) or {}


# ============================================================
# ffmpeg / 视频 / 音频
# ============================================================
def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ff(args, what=""):
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-500:]))
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
    return {"dur": dur, "audio": "Audio:" in txt, "frames": frames, "fps": fps,
            "w": w, "h": h, "txt": txt}


def pcm(path, start=None, dur=None, sr=SR):
    """抽一段单声道 PCM（s16le），返回 np.int16 数组。不传 start 就是「从头、不 seek」。"""
    import numpy as np
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
    if start is not None:
        args += ["-ss", "%.6f" % float(start)]
    args += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if dur is not None:
        args += ["-t", "%.6f" % float(dur)]
    args += ["-f", "s16le", "-"]
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        return np.zeros(0, np.int16)
    return np.frombuffer(r.stdout, np.int16)


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


def imread_any(path):
    import cv2
    import numpy as np
    with open(str(path), "rb") as fh:
        buf = np.frombuffer(fh.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def diff(a, b):
    import numpy as np
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), int(d.max())


def make_tone(path, dur, sr=48000):
    """合成一段「每 0.25s 换一个频率」的音轨：错位 1 帧都能看出来。"""
    import numpy as np
    n = int(dur * sr)
    t = np.arange(n) / float(sr)
    block = np.floor(t / TONE_BLOCK).astype(np.int64)
    freq = 300.0 + 40.0 * block
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sr)
    sig = 0.35 * np.sin(phase)
    data = (np.clip(sig, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def best_lag(src, tpl):
    """tpl 在 src 里的最佳起点（样本数），FFT 互相关；顺带给出归一化相关值。"""
    import numpy as np
    src = np.asarray(src, np.float64)
    tpl = np.asarray(tpl, np.float64)
    if len(src) < len(tpl) + 8 or len(tpl) < 64:
        return 0, 0.0
    a = src - src.mean()
    b = tpl - tpl.mean()
    nfft = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b[::-1], nfft), nfft)
    corr = corr[len(b) - 1: len(a)]
    lag = int(np.argmax(corr))
    seg = a[lag: lag + len(b)]
    den = float(np.linalg.norm(seg) * np.linalg.norm(b))
    cc = float(np.dot(seg, b) / den) if den > 0 else 0.0
    return lag, cc


# ============================================================
# 沙箱 / 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
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


def start_server(tag="帧范围"):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env["DFL_STAGE_KEEP"] = "1"          # 验收要反复读 cache/，别让 encode 把它删了
    env.pop("DFL_PORT", None)
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

    import threading
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
# 小工具：文件清单 / 阶段体
# ============================================================
def imgs(dirp, exts=(".png", ".jpg", ".jpeg")):
    d = Path(dirp)
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts],
                  key=lambda p: p.name)


def frame_keys(paths):
    """000123_00.jpg -> 123（对齐脸用）；000123.png -> 123（帧图用）。"""
    out = []
    for p in paths:
        head = p.stem.split("_")[0]
        try:
            out.append(int(head, 10))
        except ValueError:
            pass
    return out


def cache_of(name):
    return MEDIA / (name + "_cache")


def body_of(video, cache, **extra):
    b = {"video": str(video), "model": DFM, "cache_dir": str(cache),
         "settings": dict(SETTINGS), "keep_cache": True}
    b.update(extra)
    return b


def six_stages(video, cache, rng=None, tag=""):
    """跑完整六阶段；返回每一步的终态 status。rng = (start, end) 或 None。"""
    body = body_of(video, cache)
    if rng:
        body["start_frame"] = int(rng[0])
        body["end_frame"] = int(rng[1])
    snap = {}
    for name in ("frames", "faces", "pred", "mask", "merge", "encode"):
        t0 = time.time()
        st, d, fin = run_stage(name, body)
        snap[name] = fin
        m = stage_meta(fin, name)
        ok = st == 200 and fin.get("phase") == "done"
        check(ok, "%s阶段 %s 完成" % (tag, name),
              "HTTP %s · %.1fs · %s" % (st, time.time() - t0, (fin.get("error") or "")[:120]))
        say("  %s：%s" % (name, {k: m.get(k) for k in
                                ("done", "done_frames", "count", "total", "range",
                                 "range_key", "seconds", "fps")}))
    return snap


def out_of(snap):
    return Path(str((snap.get("encode") or {}).get("out_video") or ""))


# ============================================================
# a) start=100, end=200
# ============================================================
def case_a(video, src):
    tag = "a"
    total = src["frames"]
    fps = src["fps"] or 30.0
    cache = cache_of("a")
    shutil.rmtree(cache, ignore_errors=True)
    snap = six_stages(video, cache, (100, 200), tag="a")
    n = 200 - 100 + 1

    frames = imgs(cache / "frames")
    aligned = imgs(cache / "aligned")
    preds = [p for p in imgs(cache / "pred") if not re.search(r"_(prdm|dstdm)$", p.stem)]
    masks = [p for p in imgs(cache / "mask") if p.stem.endswith("_xseg_prd")]
    fk = frame_keys(frames)
    ak = sorted(set(frame_keys(aligned)))

    # 1) 切帧仍然整段
    check(len(frames) == total, "a 切帧仍整段（frames/ = 总帧数）",
          "%d / %d" % (len(frames), total))
    check(min(fk) == 1 and max(fk) == total, "a frames 序号 1..总帧数（索引完整）",
          "%d..%d" % (min(fk), max(fk)))
    fmeta = stage_meta(snap["frames"], "frames")
    check((fmeta.get("range") or {}).get("start") == 1
          and (fmeta.get("range") or {}).get("end") == total,
          "a meta.stages.frames.range = 1..总帧数（记的是覆盖范围）", str(fmeta.get("range")))
    check((fmeta.get("range_requested") or {}).get("start") == 100
          and (fmeta.get("range_requested") or {}).get("end") == 200,
          "a meta.stages.frames.range_requested = 100..200", str(fmeta.get("range_requested")))

    # 2) 后四个阶段只处理 101 帧（文件数 / 帧号都要落在 100..200）
    fmd = stage_meta(snap["faces"], "faces")
    check(bool(ak) and min(ak) >= 100 and max(ak) <= 200,
          "a faces 没碰范围外的帧（aligned 帧号全在 100..200）",
          "%d 个帧号 · %d..%d" % (len(ak), ak[0] if ak else -1, ak[-1] if ak else -1))
    check(int(fmd.get("done_frames") or 0) == n and int(fmd.get("total") or 0) == n,
          "a faces 处理了 101 帧（其中 %d 帧检测到脸）" % len(ak),
          "done_frames=%s · total=%s · 有脸的帧=%d"
          % (fmd.get("done_frames"), fmd.get("total"), len(ak)))
    check(min(ak) == 100, "a faces 起点 = 第 100 帧（1..99 一张脸都没切）",
          "最小帧号 %d" % (min(ak) if ak else -1))
    check(len(preds) == len(aligned), "a pred 与 aligned 一一对应（只 101 帧的脸）",
          "%d / %d" % (len(preds), len(aligned)))
    pk = sorted(set(frame_keys(preds)))
    check(pk and pk[0] >= 100 and pk[-1] <= 200, "a pred 帧号全在 100..200",
          "%d..%d" % (pk[0] if pk else -1, pk[-1] if pk else -1))
    check(len(masks) == len(aligned), "a mask 只算 101 帧的脸", "%d / %d" %
          (len(masks), len(aligned)))
    mm = stage_meta(snap["mask"], "mask")
    check(int(mm.get("done_frames") or 0) == len(aligned) and int(mm.get("total") or 0) == len(aligned),
          "a mask 游标相对范围（done_frames = 范围内张数）",
          "%s / %s" % (mm.get("done_frames"), mm.get("total")))
    gm = stage_meta(snap["merge"], "merge")
    check(int(gm.get("count") or 0) == n or int(gm.get("frames") or 0) == n,
          "a merge 只覆盖 101 帧", "count=%s · frames=%s · swapped=%s · copied=%s" %
          (gm.get("count"), gm.get("frames"), gm.get("swapped"), gm.get("copied")))
    check(int((gm.get("profile") or {}).get("frames") or 0) == n,
          "a merge profile.frames = 101", (gm.get("profile") or {}).get("frames"))

    # 3) 成片：时长 = 101/fps ±1 帧 + 命名带后缀
    out = out_of(snap)
    check(out.is_file(), "a 成片存在", "%s · %.2f MB" %
          (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    check(out.stem.endswith("_0100-0200"), "a 成片名带范围后缀（不覆盖全片结果）", out.name)
    dst = probe(out)
    want = n / fps
    check(abs(dst["dur"] - want) <= 1.5 / fps, "a 成片时长 = 101/fps ±1 帧",
          "期望 %.4fs · 实际 %.4fs · fps %.3f" % (want, dst["dur"], fps))
    check(abs(dst["frames"] - n) <= 1, "a 成片帧数 = 101 ±1", "%d / %d" % (dst["frames"], n))
    em = stage_meta(snap["encode"], "encode")
    check(int(em.get("count") or 0) == n, "a encode 只编码 101 帧", em.get("count"))
    check((em.get("frame_range") or {}).get("start") == 100
          and (em.get("frame_range") or {}).get("end") == 200,
          "a encode 的 frame_range = 100..200", str(em.get("frame_range")))

    # 4) 音轨：存在 + 起点对齐（交叉相关定位到源视频的 t0）
    check(dst["audio"], "a 成片有音轨")
    t0 = (100 - 1) / fps
    src_pcm = pcm(video, 0.0, min(src["dur"], t0 + 4.0))
    prod_pcm = pcm(out, 0.0, 0.30)
    lag, cc = best_lag(src_pcm, prod_pcm[:int(0.25 * SR)])
    got = lag / float(SR)
    check(abs(got - t0) <= 1.5 / fps and cc >= 0.85,
          "a 音轨起点与源视频第 100 帧对齐（互相关定位）",
          "期望 %.4fs · 实测 %.4fs（差 %+.1f ms · 相关 %.3f · 容差 ±1 帧=%.1f ms）"
          % (t0, got, (got - t0) * 1000.0, cc, 1000.0 / fps))
    # 反证：把同一段音轨往错的位置对（t0-1s）相关必须明显更低
    wrong = max(0, int((t0 - 1.0) * SR))
    seg = src_pcm[wrong: wrong + int(0.25 * SR)].astype(float)
    tmpl = prod_pcm[:int(0.25 * SR)].astype(float)
    d1 = seg - seg.mean()
    d2 = tmpl - tmpl.mean()
    den = float((d1.dot(d1) ** 0.5) * (d2.dot(d2) ** 0.5)) or 1.0
    cc_wrong = float(d1.dot(d2) / den)
    check(cc - cc_wrong > 0.3, "a 反证：错位 1s 的音轨对不上（说明定位不是碰巧）",
          "对齐 %.3f vs 错位 %.3f" % (cc, cc_wrong))

    # 5) 视频起点对齐：成片第 1 帧 ≈ cache 里第 100 帧（已合成）
    prod0 = decode_all(out)[:1]
    ref = imread_any(cache / "frames" / "000100.png")
    if prod0 and ref is not None:
        mean, mx = diff(prod0[0], ref)
        check(mean <= 8.0, "a 成片第 1 帧 = cache 第 100 帧（画面起点对齐）",
              "mean=%.2f max=%d" % (mean, mx))
    else:
        check(False, "a 成片第 1 帧取不到", "%s" % (cache / "frames" / "000100.png"))

    # 6) meta / status 里的 range 与相对进度
    meta = json.loads((cache / "meta.json").read_text("utf-8"))
    check((meta.get("range") or {}).get("start") == 100
          and (meta.get("range") or {}).get("end") == 200,
          "a meta.range = 100..200", str(meta.get("range")))
    _s, stt = get("/api/merge/stage/status?cache_dir=%s" % str(cache))
    check((stt.get("range") or {}).get("start") == 100
          and int(stt.get("range_total") or 0) == n,
          "a status 给 range + 相对进度分母", "range=%s · range_total=%s" %
          (stt.get("range"), stt.get("range_total")))
    check(int(stt.get("frames_total") or 0) == n,
          "a status.frames_total = 101（相对范围，不是全片）", stt.get("frames_total"))
    sr_ranges = stt.get("stage_ranges") or {}
    check(all(int((sr_ranges.get(k) or {}).get("start") or 0) == 100
              for k in ("faces", "pred", "mask", "merge")),
          "a status.stage_ranges 五个阶段都是 100..200",
          json.dumps({k: sr_ranges.get(k) for k in ("faces", "pred", "mask", "merge")},
                     ensure_ascii=False))
    return {"cache": cache, "out": out, "snap": snap}


# ============================================================
# b) 只给 start=100 / c) 只给 end=200 / e) 全片
# ============================================================
def case_bc(kind, video, src):
    tag = kind
    total = src["frames"]
    fps = src["fps"] or 30.0
    cache = cache_of(tag)
    shutil.rmtree(cache, ignore_errors=True)
    body = body_of(video, cache)
    if kind == "b":
        body["start_frame"] = 100
        want = (100, total)
    else:
        body["end_frame"] = 200
        want = (1, 200)
    n = want[1] - want[0] + 1
    snap = {}
    for name in ("frames", "faces", "pred", "mask", "merge", "encode"):
        st, d, fin = run_stage(name, body)
        snap[name] = fin
        m = stage_meta(fin, name)
        check(st == 200 and fin.get("phase") == "done", "%s 阶段 %s 完成" % (tag, name),
              "HTTP %s · %s" % (st, (fin.get("error") or "")[:100]))
        say("  %s：%s" % (name, {k: m.get(k) for k in
                                ("done", "done_frames", "count", "total", "range", "seconds")}))
    aligned = imgs(cache / "aligned")
    ak = sorted(set(frame_keys(aligned)))
    fmd = stage_meta(snap["faces"], "faces")
    check(bool(ak) and min(ak) >= want[0] and max(ak) <= want[1],
          "%s faces 只处理 %d 帧（%d..%d，aligned 帧号不越界）" % (tag, n, want[0], want[1]),
          "%d 个帧号 · %d..%d" % (len(ak), ak[0] if ak else -1, ak[-1] if ak else -1))
    check(int(fmd.get("done_frames") or 0) == n and int(fmd.get("total") or 0) == n,
          "%s faces done_frames/total = %d" % (tag, n),
          "%s / %s · 有脸的帧 %d" % (fmd.get("done_frames"), fmd.get("total"), len(ak)))
    gm = stage_meta(snap["merge"], "merge")
    check(int(gm.get("frames") or 0) == n and int(gm.get("done_frames") or 0) == n,
          "%s merge 只覆盖 %d 帧" % (tag, n),
          "frames=%s · done_frames=%s · swapped=%s · copied=%s"
          % (gm.get("frames"), gm.get("done_frames"), gm.get("swapped"), gm.get("copied")))
    out = out_of(snap)
    dst = probe(out)
    check(abs(dst["dur"] - n / fps) <= 1.5 / fps, "%s 成片时长 = %d/fps ±1 帧" % (tag, n),
          "期望 %.4fs · 实际 %.4fs" % (n / fps, dst["dur"]))
    check(out.stem.endswith("_%04d-%04d" % want), "%s 成片名带范围后缀" % tag, out.name)
    return {"cache": cache, "out": out, "snap": snap, "want": want, "n": n}


def case_e_full(video, src):
    tag = "e"
    total = src["frames"]
    cache = cache_of("full")
    shutil.rmtree(cache, ignore_errors=True)
    snap = six_stages(video, cache, None, tag="e")
    frames = imgs(cache / "frames")
    check(len(frames) == total, "e 全片：frames/ = 总帧数", "%d / %d" % (len(frames), total))
    out = out_of(snap)
    check(out.name == Path(video).stem + "_merged.mp4", "e 全片成片名不带后缀（回归）", out.name)
    dst = probe(out)
    check(abs(dst["dur"] - src["dur"]) <= 0.3, "e 全片时长 ≈ 源",
          "源 %.3fs / 成片 %.3fs" % (src["dur"], dst["dur"]))
    check(abs(dst["frames"] - total) <= 1, "e 全片帧数 ≈ 总帧数",
          "%d / %d" % (dst["frames"], total))
    em = stage_meta(snap["encode"], "encode")
    check(int(em.get("count") or 0) == total, "e 全片 encode 编码了全部帧", em.get("count"))
    meta = json.loads((cache / "meta.json").read_text("utf-8"))
    rg = meta.get("range") or {}
    check(bool(rg.get("full")) and int(rg.get("start") or 0) == 1
          and int(rg.get("end") or 0) == total, "e 全片 meta.range = 1..总帧数（full）", str(rg))
    return {"cache": cache, "out": out, "snap": snap}


# ============================================================
# d) 非法范围 -> 422
# ============================================================
def case_d_invalid(video, src):
    total = src["frames"]
    cache = MEDIA / "d_cache"
    shutil.rmtree(cache, ignore_errors=True)
    body = body_of(video, cache)

    def bad(name, extra, why):
        b = dict(body)
        b.update(extra)
        st, d = post("/api/merge/stage/frames", b, timeout=120)
        ok = st == 422
        check(ok, "d %s → 422" % name, "HTTP %s · %s" %
              (st, (d.get("error") or d.get("detail") or "")[:160]))
        if ok:
            say("%s：%s" % (name, (d.get("error") or "")[:160]))
        return st

    bad("start=300,end=100", {"start_frame": 300, "end_frame": 100}, "范围反了")
    bad("end>总帧数", {"end_frame": total + 50}, "end 超界")
    bad("start=0", {"start_frame": 0}, "start<1")
    bad("start=总帧数+40", {"start_frame": total + 40}, "start 超界（start>end）")
    bad("start=abc", {"start_frame": "abc"}, "不是整数")
    # 422 之后 cache 里不该留下任何阶段产物
    check(not (cache / "frames").exists() or not imgs(cache / "frames"),
          "d 422 之后没有偷偷切帧", str(cache))


# ============================================================
# f) 改范围 -> 旧阶段失效 / 自动重跑
# ============================================================
def case_f_change(video, src, a):
    cache = a["cache"]
    body = body_of(video, cache)
    total = src["frames"]

    # 1) 上游只跑过 100..200：mask 要 1..50 -> 409，并点名是哪一步没覆盖到
    b = dict(body)
    b.update({"start_frame": 1, "end_frame": 50})
    st, d = post("/api/merge/stage/mask", b, timeout=120)
    err = (d.get("error") or "")
    check(st == 409 and ("切脸" in err or "换脸" in err) and "100..200" in err,
          "f mask 要 1..50（上游只跑过 100..200）→ 409 点名前一步 + 报出两段范围",
          "HTTP %s · %s" % (st, err[:160]))
    say(err[:200])

    # 2) mask 改到 100..150（pred 覆盖得到）-> 自动重跑，range_key 变、日志说清楚
    b = dict(body)
    b.update({"start_frame": 100, "end_frame": 150})
    st, d, fin = run_stage("mask", b)
    check(st == 200 and not d.get("skipped"), "f mask 改到 100..150 → 自动重跑（不跳过）",
          "HTTP %s · skipped=%s · range_changed=%s" %
          (st, d.get("skipped"), d.get("range_changed")))
    mm = stage_meta(fin, "mask")
    check(str(mm.get("range_key")) == "100-150", "f mask.range_key 变成 100-150",
          str(mm.get("range_key")))
    aligned = imgs(cache / "aligned")
    want_n = len([p for p in aligned if 100 <= frame_keys([p])[0] <= 150])
    check(int(mm.get("done_frames") or 0) == want_n,
          "f mask 游标按新范围重数（0..范围内张数）",
          "%s / %d" % (mm.get("done_frames"), want_n))
    log = "\n".join(fin.get("log") or [])
    check("范围变了" in log, "f 日志里明确写「范围变了」", [x for x in (fin.get("log") or [])
                                                       if "范围" in x][:4])

    # 3) 同范围再点一次 -> 跳过（幂等）
    st, d = post("/api/merge/stage/mask", b, timeout=120)
    check(st == 200 and bool(d.get("skipped")), "f 同范围再点 → 跳过（幂等）",
          "HTTP %s · skipped=%s" % (st, d.get("skipped")))

    # 4) 改回 100..200 -> 又一次自动重跑（pred/faces 覆盖得到，不该 409）
    b2 = dict(body)
    b2.update({"start_frame": 100, "end_frame": 200})
    st, d, fin2 = run_stage("mask", b2)
    check(st == 200 and not d.get("skipped"), "f mask 改回 100..200 → 又自动重跑",
          "HTTP %s · skipped=%s" % (st, d.get("skipped")))
    mm2 = stage_meta(fin2, "mask")
    want2 = len([p for p in aligned if 100 <= frame_keys([p])[0] <= 200])
    check(int(mm2.get("total") or 0) == want2 and str(mm2.get("range_key")) == "100-200",
          "f 回到 100..200：total=%d、range_key=100-200" % want2,
          "total=%s · key=%s" % (mm2.get("total"), mm2.get("range_key")))
    # 上游没被乱动：pred 的指纹还停在 100-200
    pkey = str((stage_meta(fin2, "pred") or {}).get("range_key") or "")
    check(pkey == "100-200", "f 只改了 mask 的范围，pred 没被重跑", "pred.range_key=%s" % pkey)

    # 5) encode 的成片不会被另一个范围覆盖（文件名不同）
    outs = sorted(p.name for p in MEDIA.glob("clip_merged*.mp4"))
    check(len(set(outs)) == len(outs) and any("_0100-0200" in o for o in outs),
          "f 不同范围写不同成片（不互相覆盖）", outs)


# ============================================================
# main
# ============================================================
def main():
    try:                       # Windows 控制台默认 GBK：中文日志别变成乱码
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑这几组（逗号分隔：a,b,c,d,e,f）")
    args = ap.parse_args()
    only = {x.strip().lower() for x in str(args.only or "").split(",") if x.strip()}
    want = (lambda k: (not only) or (k in only))

    t_all = time.time()
    prepare_sandbox()
    src_video = sorted((REPO / "workspace").glob("*.mp4"))[0]
    tone = MEDIA / "tone.wav"
    make_tone(tone, 7.0)
    run_ff(["-ss", "3", "-t", "7.0", "-i", str(src_video), "-i", str(tone),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", str(CLIP)], "切样片+合成音轨")
    src = probe(CLIP)
    say("样片：%s" % {k: src[k] for k in ("frames", "fps", "dur", "audio", "w", "h")})
    if src["w"] != 1080 or src["h"] != 1920:
        raise SystemExit("样片不是 1080x1920：%s" % src)
    if src["frames"] < 205:
        raise SystemExit("样片太短（%d 帧），a 组要 start=100/end=200" % src["frames"])

    proc = start_server()
    try:
        a = b = c = e = None
        if want("a"):
            a = case_a(CLIP, src)
        if want("b"):
            b = case_bc("b", CLIP, src)
        if want("c"):
            c = case_bc("c", CLIP, src)
        if want("d"):
            case_d_invalid(CLIP, src)
        if want("e"):
            e = case_e_full(CLIP, src)
        if want("f"):
            if a is None:
                say("f 需要 a 的 cache —— 先跑 a")
                a = case_a(CLIP, src)
            case_f_change(CLIP, src, a)
    finally:
        stop_server(proc)

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    print("样片：%d 帧 @ %.3f fps · 时长 %.3fs · 音轨 %s"
          % (src["frames"], src["fps"], src["dur"], src["audio"]))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
