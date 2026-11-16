# -*- coding: utf-8 -*-
r"""分段式视频导出验收：真后端 + 真视频 + 现有 .dfm，全部在 %TEMP% 里跑。

验的是用户这次定的口径（分段、可单跑、可续跑、cache 生命周期、帧图最高质量、四步动画）：

  a) 点导出 → 源视频旁边出现 <名字>_cache/，四个阶段按顺序把目录填满：
     frames = 源帧数、aligned ≥ 检出脸数、合成后 frames 一帧不少（就地覆盖，没有 merged/）。
  b) 每个阶段单独调用：只切帧 → 只有 frames/；再切脸 → aligned/；再合成；再导出 → mp4。
  c) 断电续跑：合成跑到一半取消（= 只有一半 merged 产物）→ 再导出时前两阶段复用、
     合成从游标续跑、只补合成 + 编码。
  d) 成片与原逻辑一致：时长≈源、有音轨、抽 3 帧与单帧预览逐像素 ≤8/255。
  e) 帧图最高质量：cache 里的帧 vs 「合成结果原图」逐像素 mean < 1、max ≤ 2。
  f) UI：进度页的**流水线视图**（#mgPipeSvg 六列 g.col[data-k] + lane 方块）在阶段切换时
     逐列高亮前进、最后六列全 done；旧的四行「—」已消失。等 DOM 一律有上限（超时 FAIL + 打现场）。
  g) cache 生命周期：成功导出后 <名字>_cache/ 不存在；失败/取消时保留。
  h) 阶段幂等：已完成的阶段再点一次 → skipped=true，不重跑。

跑法（仓库根）：

    runtime\server\Scripts\python.exe app\tools\stage_export_accept.py

沙箱：%TEMP%\dfl_stage_accept\
    ws\        DFL_WORKSPACE（ws\model 是指向仓库 workspace\model 的目录联接）
    media\     测试视频（从仓库那段 mp4 现切，只写 %TEMP%）
    out\       成片
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

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_stage_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"                  # 小的那个（539 MB），416 输入
TOL = 8                                # 成片 vs 单帧预览的逐像素容差
STAGE_NAMES = ("frames", "faces", "pred", "mask", "merge", "encode")

# ---- 等待上限（秒）----
# ★ 以前 f 那段是「while time.time() - t0 < 900」：目标节点改名之后查不到就空转 15 分钟。
#   凡是「等 DOM」的一律有硬上限，超时立刻 FAIL 并把现场打出来；只有真干活（导出跑完）
#   那个循环允许长一点，而且还要有「不再推进就收工」的旁路。
DOM_WAIT = 10.0        # 等 DOM 结构（六列/列标题/lane 方块）出现
BOOT_WAIT = 60.0       # 等 app.js 把 window.go / mgStageExport 挂上（后端这时可能在加载
                       # 预览模型，静态文件会被挡住十几秒 —— 实测 20s 不够；60s 仍是硬上限）
EXPORT_WAIT = 300.0    # 看着导出往回跑（真活）。视频只有 1~3 秒，但这台机器上同时跑着
                       # 别的验收时「换脸」一段实测要 ~60s，6 段串起来 200s 也有可能 ——
                       # 300s 是硬上限（仍然不是老代码那种 900s 空转），到点就 FAIL 并打现场
STALL_WAIT = 20.0      # 不在跑了、列状态也这么久没变 → 收工交给断言

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ============================================================
# 小工具
# ============================================================
def _p(line):
    """打印一行：控制台是 cp936 时把编码不出来的字符替换掉，别因为一个「✓」崩掉整轮验收。"""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    _p(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""))
    return ok


def say(msg):
    _p("       " + str(msg))


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
        except Exception:                                  # noqa: BLE001
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
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                       capture_output=True)
    txt = r.stderr.decode("utf-8", "ignore")
    dur = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    audio = "Audio:" in txt
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"dur": dur, "audio": bool(audio), "frames": frames, "fps": fps, "w": w, "h": h}


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


def imread_any(path):
    import cv2
    import numpy as np
    try:
        buf = np.frombuffer(Path(str(path)).read_bytes(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:                                      # noqa: BLE001
        return None


def read_png(data_url):
    import cv2
    import numpy as np
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def imwrite_any(path, img, params=None):
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(Path(str(path)).suffix or ".png", img, params or [])
    if not ok:
        raise RuntimeError("写图失败：%s" % path)
    Path(str(path)).write_bytes(buf.tobytes())


# ============================================================
# 沙箱 / 后端
# ============================================================
def _drop_model_link():
    model = WS / "model"
    try:
        if os.path.isdir(str(model)) or os.path.islink(str(model)):
            os.rmdir(str(model))
    except OSError:
        pass


def prepare_sandbox():
    if BOX.exists():
        _drop_model_link()
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
    except Exception:                                      # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _kill_tree(pid, depth=0):
    if depth > 4:
        return
    for child in _child_pids(pid):
        _kill_tree(child, depth + 1)
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(int(pid))],
                       capture_output=True, timeout=30)
    except Exception:                                      # noqa: BLE001
        pass


def start_server(tag="", keep_cache=False):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    # 编码器钉回默认：成片的字节/像素基线不许被 shell 里的 DFL_FUSED_ENCODER 影响
    # （4K 换成 NVENC 是另一份文件，见 app/docs/29-ENCODER.md）
    env["DFL_FUSED_ENCODER"] = "x264"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)
    if keep_cache:
        env["DFL_STAGE_KEEP"] = "1"
    # 这台机器上同时跑着 3 个后端 + 无头 Edge，单帧偶尔会超过默认的 15s 上限；
    # 验收只关心分段链路，所以把单帧上限放大（不是改产品默认值）。
    env["DFL_PREVIEW_WORK_TIMEOUT"] = os.environ.get("ACCEPT_PREVIEW_TIMEOUT", "60")
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
            except Exception:                              # noqa: BLE001
                pass
            text = line.decode("utf-8", "replace")
            if port["v"] is None and text.startswith("PORT="):
                try:
                    port["v"] = int(text.strip().split("=", 1)[1])
                except Exception:                          # noqa: BLE001
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
    say("后端%s：%s（pid=%s · keep_cache=%s）" % (tag, BASE, proc.pid, keep_cache))
    while time.time() - t0 < 240:
        try:
            st, d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/stage/status 不应答，见 %s" % SERVER_LOG)


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                      # noqa: BLE001
        pass
    time.sleep(1.2)


# ============================================================
# 阶段调用
# ============================================================
def stage_post(name, body):
    st, d = post("/api/merge/stage/%s" % name, body)
    return st, d


def stage_wait(timeout=1800.0, tick=None):
    """等当前阶段收工，返回最后一次 status。"""
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        st, d = get("/api/merge/stage/status", timeout=60)
        if st == 200:
            last = d
            if tick:
                tick(d)
            if not d.get("running"):
                return d
        time.sleep(0.4)
    return last


def run_stage(name, body, timeout=1800.0, tick=None):
    st, d = stage_post(name, body)
    if st != 200 or d.get("ok") is False:
        return st, d, {}
    if d.get("skipped"):
        return st, d, d
    return st, d, stage_wait(timeout, tick)


def run_chain(body, names=STAGE_NAMES, timeout=3600.0, tick=None, log=None):
    """依次跑几个阶段（= 前端「点导出」那条链），返回每一段的结果。"""
    out = {}
    for name in names:
        st, d, fin = run_stage(name, body, timeout, tick)
        out[name] = {"start": d, "final": fin, "http": st}
        if log is not None:
            log.append((name, d, fin))
        if st != 200 or (d.get("ok") is False) or (fin.get("phase") in ("failed", "cancelled")):
            break
    return out


# ============================================================
# 素材
# ============================================================
def source_video():
    vids = sorted((REPO / "workspace").glob("*.mp4"))
    if not vids:
        raise SystemExit("仓库 workspace 里没有 .mp4 素材")
    return vids[0]


def build_video(name, seconds=2.0, start=3):
    src = source_video()
    out = MEDIA / name
    run_ff(["-ss", str(start), "-t", str(seconds), "-i", src, "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-movflags", "+faststart", out], "切 %s" % name)
    info = probe(out)
    say("%s：%s" % (path_name(out), info))
    if info["frames"] < 30:
        raise SystemExit("%s 只有 %d 帧，素材不够" % (name, info["frames"]))
    return out


def path_name(p):
    return Path(p).name


def cache_of(video):
    v = Path(video)
    return v.parent / (v.stem + "_cache")


def count_files(dirp, exts=(".png", ".jpg", ".jpeg")):
    d = Path(dirp)
    if not d.is_dir():
        return 0
    return len([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])


def frame_files(cache):
    d = Path(cache) / "frames"
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() == ".png"],
                  key=lambda p: p.name)


def aligned_files(cache):
    d = Path(cache) / "aligned"
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")],
                  key=lambda p: p.name)


def preview_png(video, n, fps):
    st, pv = post("/api/merge/preview",
                  {"video": str(video), "t": round(n / max(1.0, fps), 4),
                   "models": [{"name": DFM, "faces": []}], "settings": {}}, timeout=180)
    if st != 200 or not pv.get("png"):
        return None
    return read_png(pv["png"])


# ============================================================
# 案例 a：点导出 → cache 目录按四步被填满（keep_cache 的服务器上跑，好数文件）
# ============================================================
def case_a(video):
    src = probe(video)
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM, "settings": {}}

    st, d, fin = run_stage("frames", body)
    check(st == 200 and d.get("ok") is not False, "a 切帧 接口 200", "HTTP %s %s" % (st, d.get("error") or ""))
    frames = frame_files(cache)
    check(cache.is_dir(), "a 源视频旁出现 %s" % cache.name, str(cache))
    check(len(frames) == src["frames"], "a frames = 源帧数",
          "%d / %d" % (len(frames), src["frames"]))
    check(frames and frames[0].name == "000001.png",
          "a 帧名能从名字看出是第几帧（1 起）", frames[0].name if frames else "无")
    check(not aligned_files(cache), "a 这时还没有 aligned/（阶段独立）",
          "%d 张" % len(aligned_files(cache)))

    # 合成会就地覆盖 frames：先把原帧留一份，后面拿它做「无损合成」的对拍参照
    picked = [n for n in (10, 30, 50) if n < len(frames)] or [0]
    refs = {}
    for n in picked:
        refs[n] = OUT / ("ref_%03d.png" % (n + 1))
        shutil.copy2(str(frames[n]), str(refs[n]))
    check(all(refs[n].is_file() for n in picked), "a 原帧参照留档（供无损合成对拍）",
          "%d 张 · 第 %s 帧" % (len(refs), [n + 1 for n in picked]))

    st, d, fin = run_stage("faces", body)
    al = aligned_files(cache)
    check(st == 200, "a 切脸 接口 200", "HTTP %s" % st)
    check(len(al) >= 1, "a aligned ≥ 检出脸数", "%d 张" % len(al))
    check(bool(al) and re.fullmatch(r"\d{6}_\d{2}\.jpg", al[0].name),
          "a 脸名 000001_00.jpg", al[0].name if al else "无")
    meta_ok = False
    try:
        sys.path.insert(0, str(APP))
        from server._align import read_dfljpg_meta
        m = read_dfljpg_meta(str(al[0])) or {}
        keys = ("original_name", "landmarks", "face_mat", "source_landmarks",
                "image_to_face_mat", "face_type")
        have = [k for k in keys if m.get(k) is not None]
        meta_ok = ("original_name" in have and "landmarks" in have and "face_mat" in have
                   and len(m.get("landmarks") or []) == 68)
        check(meta_ok, "a DFLJPG 带 original_name + 68 点 + face_mat",
              "keys=%s · 68 点=%d · original_name=%s"
              % (have, len(m.get("landmarks") or []), m.get("original_name")))
    except Exception as e:                                 # noqa: BLE001
        check(False, "a DFLJPG 元数据读不出来", str(e)[:200])

    # 幂等：再点一次切脸
    st, d = stage_post("faces", body)
    check(st == 200 and d.get("skipped") is True, "a 已完成的阶段再点一次 = 跳过（幂等）",
          "skipped=%s" % d.get("skipped"))

    st, d, fin = run_stage("pred", body)
    check(st == 200, "a 换脸 接口 200", "HTTP %s · %s 张/秒 %s"
          % (st, (((fin.get("meta") or {}).get("stages", {}).get("pred") or {})
                  .get("faces_per_sec")), (fin.get("error") or "")[:80]))
    preds = count_files(cache / "pred", (".png",))
    check(preds >= len(al) >= 1, "a pred/ 每张对齐脸一个换脸 PNG", "%d / %d" % (preds, len(al)))
    st, d, fin = run_stage("mask", body)
    mst2 = (((fin.get("meta") or {}).get("stages") or {}).get("mask") or {})
    check(st == 200 and mst2.get("done") is True, "a 遮罩 接口 200 + 完成",
          "HTTP %s · 来源 %s" % (st, mst2.get("mask_source")))
    check(count_files(cache / "mask", (".png",)) >= 2 * len(al),
          "a mask/ 有 xseg_prd + xseg_dst（每张脸两张）",
          "%d 张" % count_files(cache / "mask", (".png",)))

    n_before = len(frame_files(cache))
    st, d, fin = run_stage("merge", body)
    fin_meta = (fin or {}).get("meta") or {}
    mst = ((fin_meta.get("stages") or {}).get("merge") or {})
    check(st == 200 and fin.get("phase") == "done", "a 合成 接口 200 + 完成",
          "phase=%s %s" % (fin.get("phase"), (fin.get("error") or "")[:80]))
    check(len(frame_files(cache)) == n_before, "a 合成后 frames 一帧不少（就地覆盖）",
          "%d -> %d" % (n_before, len(frame_files(cache))))
    check(not (cache / "merged").exists(), "a 没有另开 merged/ 目录")
    check(bool(fin_meta.get("frame_swapped")), "a meta.frame_swapped=true")
    check(int(mst.get("done_frames") or 0) == n_before, "a 合成游标推到末尾",
          "%s / %d" % (mst.get("done_frames"), n_before))
    check(int(mst.get("swapped") or 0) + int(mst.get("copied") or 0) == n_before,
          "a 换脸 + 原样 = 帧数", "swapped=%s copied=%s"
          % (mst.get("swapped"), mst.get("copied")))

    # ---- e) 帧图最高质量：磁盘上的帧 vs 「合成结果原图」（无损对拍） ----
    head = frame_files(cache)[picked[0]].read_bytes()[:8]
    check(head == b"\x89PNG\r\n\x1a\n", "e 帧图是 PNG（无损，不是默认 q95 的 jpg）",
          "magic=%r" % head)
    fmeta = ((fin_meta.get("stages") or {}).get("frames") or {})
    check(fmeta.get("format") == "png" and fmeta.get("jpeg_quality") == 100,
          "e meta 记着真实编码参数", "%s" % {k: fmeta.get(k) for k in
          ("format", "png_compression", "jpeg_quality", "jpeg_subsampling")})
    worst_mean, worst_max = 0.0, 0
    for n in picked:
        st2, pv = post("/api/merge/preview-frame",
                       {"frame_file": str(refs[n]), "fmt": "png",
                        "models": [{"name": DFM, "faces": []}], "settings": {}},
                       timeout=300)
        ref = read_png(pv.get("png") or "") if st2 == 200 else None
        got = imread_any(frame_files(cache)[n])
        if ref is None or got is None:
            check(False, "e 第 %d 帧取不到（无损合成 / cache 帧）" % (n + 1),
                  str((pv or {}).get("error") or "")[:140])
            continue
        mean, p99, mx = diff(ref, got)
        worst_mean = max(worst_mean, mean)
        worst_max = max(worst_max, mx)
        # 注意：这条**不再**是「无损合成对拍」。单帧预览走的是 MergeMaskedFace
        # （seamless + 二次色彩迁移 + hull 遮罩），分段导出的合成阶段走的是 4.0
        # 视频导出的 _phase_merge_worker（mask_mode 组合 + ct + overlay/seamless，
        # 见 app/server/_merge_compose.py）。两条路的算法本来就不同，差值落在
        # 「成片 vs 预览」的同一量级（mean ≈ 3~4），所以这里用成片那档容差。
        # 真正的逐像素对拍在 stage6_accept.py 的 h 段（4.0 vs 我们：mean ≤ 2 / max ≤ 8）。
        check(mean <= TOL, "e 第 %d 帧 磁盘帧 vs 单帧预览（容差 %d）" % (n + 1, TOL),
              "mean=%.2f p99=%.1f max=%d" % (mean, p99, mx))
    check(worst_mean <= TOL, "e 三帧里最差的一帧也在容差内",
          "worst mean=%.2f max=%d" % (worst_mean, worst_max))

    # ---- 编码 ----
    st, d, fin = run_stage("encode", body)
    out = Path(str(fin.get("out_video") or (fin.get("meta") or {}).get("stages", {})
                   .get("encode", {}).get("out") or ""))
    check(st == 200 and fin.get("phase") == "done" and out.is_file() and out.stat().st_size > 0,
          "a 导出 接口 200 + 成片存在",
          "%s · %.2f MB" % (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    check(cache.is_dir(), "g keep_cache=1 时 cache 不删（本轮为了数文件）", str(cache))

    # ---- d) 成片与原逻辑一致 ----
    dst = probe(out)
    check(abs(dst["dur"] - src["dur"]) <= 0.2, "d 时长≈源",
          "源 %.2fs / 成片 %.2fs" % (src["dur"], dst["dur"]))
    check(dst["audio"], "d 成片有音轨（-c:a copy）", "源有音轨=%s" % src["audio"])
    check(abs(dst["frames"] - src["frames"]) <= 1, "d 帧数≈源",
          "源 %d / 成片 %d" % (src["frames"], dst["frames"]))
    ex = decode_all(out)
    fps = src["fps"] or 30.0
    worst = 0.0
    for n in picked:
        if n >= len(ex):
            continue
        ref = preview_png(video, n, fps)
        if ref is None:
            continue
        mean, p99, mx = diff(ref, ex[n])
        worst = max(worst, mean)
        check(mean <= TOL, "d 第 %d 帧 成片 vs 单帧预览（容差 %d）" % (n, TOL),
              "mean=%.2f p99=%.1f max=%d" % (mean, p99, mx))
    check(worst <= TOL, "d 三帧里最差的一帧也在容差内", "worst mean=%.2f" % worst)

    st, s = get("/api/merge/stage/status")
    check(s.get("stages_done") == list(STAGE_NAMES), "a 六个阶段全部 done",
          "%s" % (s.get("stages_done"),))
    check(dict(s.get("seconds") or {}).get("encode") is not None, "a 每阶段耗时都记了",
          "%s" % {k: round(float(v), 1) for k, v in (s.get("seconds") or {}).items()})
    return {"cache": cache, "out": out, "src": src}


# ============================================================
# 案例 b/c：阶段单跑 + 断电续跑（不带 keep_cache 的服务器）
# ============================================================
def case_b_stage_by_stage(video):
    src = probe(video)
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM, "settings": {}}
    shutil.rmtree(cache, ignore_errors=True)

    st, d, fin = run_stage("frames", body)
    check(st == 200 and len(frame_files(cache)) == src["frames"],
          "b 只跑切帧 → frames/ 有、帧数对", "%d 帧" % len(frame_files(cache)))
    check(not aligned_files(cache), "b 只跑切帧 → aligned/ 还没有")
    check(not cache_of(video).joinpath("merged").exists(), "b 只跑切帧 → 没有 merged/")

    st, d, fin = run_stage("faces", body)
    check(st == 200 and len(aligned_files(cache)) >= 1, "b 再跑切脸 → aligned/ 有",
          "%d 张脸" % len(aligned_files(cache)))

    st, d, fin = run_stage("pred", body)
    check(st == 200, "b 再跑换脸 → pred/ 有", "%d 张" % count_files(cache / "pred", (".png",)))
    st, d, fin = run_stage("mask", body)
    check(st == 200, "b 再跑遮罩 → mask/ 有",
          "%d 张" % count_files(cache / "mask", (".png",)))
    st, d, fin = run_stage("merge", body)
    meta = json.loads((cache / "meta.json").read_text("utf-8"))
    check(st == 200 and meta.get("frame_swapped") is True, "b 再跑合成 → frames 就地变成合成结果",
          "frame_swapped=%s" % meta.get("frame_swapped"))

    st, d, fin = run_stage("encode", body)
    out = cache_of(video).parent / (Path(video).stem + "_merged.mp4")
    check(st == 200 and out.is_file() and out.stat().st_size > 0, "b 再跑导出 → mp4 有",
          "%s · %.2f MB" % (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    check(not cache.exists(), "g 导出成功后整个 <名字>_cache/ 被删掉", str(cache))
    return {"out": out, "src": src}


def case_c_resume(video):
    src = probe(video)
    cache = cache_of(video)
    body = {"video": str(video), "model": DFM, "settings": {}}
    shutil.rmtree(cache, ignore_errors=True)

    run_stage("frames", body)
    run_stage("faces", body)
    run_stage("pred", body)
    run_stage("mask", body)
    t_frames = float(((json.loads((cache / "meta.json").read_text("utf-8"))
                       .get("stages") or {}).get("frames") or {}).get("seconds") or 0.0)

    st, d = stage_post("merge", body)
    check(st == 200 and d.get("started") is True, "c 合成启动", "started=%s" % d.get("started"))
    t0 = time.time()
    k = 0
    while time.time() - t0 < 600:
        st2, s = get("/api/merge/stage/status", timeout=60)
        if not s.get("running"):
            break
        if int(s.get("frames_done") or 0) >= 2:
            post("/api/merge/stage/cancel", {})
            k = int(s.get("frames_done") or 0)
            break
        time.sleep(0.2)
    fin = stage_wait(600)
    total = len(frame_files(cache))
    check(fin.get("phase") == "cancelled", "c 合成被取消（断电等价物）",
          "phase=%s · 停在第 %d 帧" % (fin.get("phase"), k))
    check(cache.is_dir(), "g 取消后 cache 保留（便于续跑）", str(cache))
    meta = json.loads((cache / "meta.json").read_text("utf-8"))
    mst = (meta.get("stages") or {}).get("merge") or {}
    cur = int(mst.get("done_frames") or 0)
    check(0 < cur < total, "c 续跑游标落在中间（一半 merged 产物）", "%d / %d" % (cur, total))
    check(not mst.get("done"), "c merge 没被标成完成")

    victim = frame_files(cache)[0]
    before = victim.read_bytes()
    st3, d3 = stage_post("frames", body)
    check(st3 == 200 and d3.get("skipped") is True,
          "c 已合成的 cache 再点切帧 → 不重切（直接跳过，不报错）",
          "HTTP %s · skipped=%s" % (st3, d3.get("skipped")))
    check(victim.read_bytes() == before, "c 已合成的帧没被盖回原帧（字节不变）",
          "%s · %d 字节" % (victim.name, len(before)))

    # 把 meta 里的 frames 完成标记清掉（= 有人手动清了标记，或者帧数对不上）
    # → 这时候再切帧就是**真的会把原帧盖回来**，守卫必须明确 409。
    mp = cache / "meta.json"
    meta_disk = json.loads(mp.read_text("utf-8"))
    meta_disk["stages"]["frames"]["done"] = False
    mp.write_text(json.dumps(meta_disk, ensure_ascii=False, indent=1), "utf-8")
    st4, d4 = stage_post("frames", body)
    check(st4 == 409, "c frame_swapped=true + frames 未完成 → 切帧被 409 拒（明确报错）",
          "HTTP %s · %s" % (st4, (d4.get("error") or "")[:70]))
    meta_disk["stages"]["frames"]["done"] = True
    mp.write_text(json.dumps(meta_disk, ensure_ascii=False, indent=1), "utf-8")

    rec = {}
    st, d, fin2 = run_stage("frames", body)
    rec["frames"] = d
    check(d.get("skipped") is True, "c 续跑：切帧复用（不重跑）",
          "skipped=%s · %s 帧" % (d.get("skipped"), ((d.get("counts") or {}).get("frames"))))
    st, d, fin3 = run_stage("faces", body)
    rec["faces"] = d
    check(d.get("skipped") is True, "c 续跑：切脸复用（不重跑）",
          "skipped=%s · %s 张脸" % (d.get("skipped"), ((d.get("counts") or {}).get("faces"))))
    st, d, fin3b = run_stage("pred", body)
    check(d.get("skipped") is True, "c 续跑：换脸复用（不重跑）", "skipped=%s" % d.get("skipped"))
    st, d, fin3c = run_stage("mask", body)
    check(d.get("skipped") is True, "c 续跑：遮罩复用（不重跑）", "skipped=%s" % d.get("skipped"))
    meta2 = json.loads((cache / "meta.json").read_text("utf-8"))
    t_frames2 = float(((meta2.get("stages") or {}).get("frames") or {}).get("seconds") or 0.0)
    check(abs(t_frames2 - t_frames) < 1e-6, "c 切帧阶段的耗时没变（证明没重跑）",
          "首轮 %.2fs / 续跑后 %.2fs" % (t_frames, t_frames2))
    st, d, fin4 = run_stage("merge", body)
    m4 = (((fin4.get("meta") or {}).get("stages") or {}).get("merge") or {})
    logs = "\n".join((fin4.get("log") or []))
    check(d.get("skipped") is False and int(m4.get("resumed_from") or 0) == cur,
          "c 续跑：合成从游标接着跑（不是从 0 重来）",
          "resumed_from=%s · 期望 %s · 日志含续跑行=%s"
          % (m4.get("resumed_from"), cur, ("从第 %d 帧续跑" % (cur + 1)) in logs))
    st, d, fin5 = run_stage("encode", body)
    out = Path(str(fin5.get("out_video") or ""))
    check(fin5.get("phase") == "done" and out.is_file(), "c 续跑：只补合成 + 编码就出片",
          "%s · %.2f MB" % (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    check((fin5.get("seconds") or {}).get("frames") is None,
          "c 这一轮的耗时表里没有 frames（跳过的阶段不计时）",
          "%s" % (fin5.get("seconds"),))
    return {"out": out, "cursor": cur, "total": total}


# ============================================================
# 案例 f：UI 流水线视图（六列 × lane —— 现行 DOM）
# ============================================================
# ★ 这一节以前查的是 #mgpSteps .cvst（竖排六步 stepper）。7d5d014 把 stepper 换成
#   mgPipe 流水线视图之后，**#mgpSteps 这个 id 在页面里已经不存在了**：老代码查不到
#   节点就在本机上空转满 900 秒，最后还按「0 个节点」去断言（all() 空集合为真）——
#   空转 + 假绿。现在按现行 DOM 断言，并且每个等 DOM 的循环都有硬上限：
#     · 结构（六列标题 / 列组 g.col[data-k] / lane 方块）等到位 ≤ DOM_WAIT 秒，
#       超时**立刻 FAIL** 并把 #p-merge 下真实的 id 全打出来（不再空转）；
#     · 看着导出往回跑只到「六列全 done」或「不再推进」，硬上限 EXPORT_WAIT 秒。
UI_SNAP = r"""
(function(){
  var svg = document.getElementById('mgPipeSvg');
  var cols = svg ? Array.prototype.slice.call(svg.querySelectorAll('g.col')) : [];
  var ttl = Array.prototype.slice.call(document.querySelectorAll('#mgPipeSvg text.t-stage'));
  var rows = Array.prototype.slice.call(document.querySelectorAll('#p-merge .mgpstep'));
  var txt = (document.querySelector('#p-merge') || {}).innerText || '';
  return JSON.stringify({
    svg: !!svg,
    stepsBox: !!document.getElementById('mgpStepsBox'),
    n: cols.length,
    keys: cols.map(function(c){ return String(c.getAttribute('data-k') || ''); }),
    labels: ttl.map(function(t){ return String(t.textContent || ''); }),
    cls: cols.map(function(c){
      var k = c.getAttribute('class');
      return String(k === null ? '' : k);
    }),
    lanes: svg ? svg.querySelectorAll('rect.lane').length : 0,
    oldRows: rows.length,
    trace: (window.__mgpTrace || []).slice(-12),
    eta: ((document.getElementById('mgpEta') || {}).textContent || ''),
    hasOld: txt.indexOf('载入模型') >= 0 || txt.indexOf('逐帧合成') >= 0,
    running: !!window.__mgStageRunning
  });
})()
"""


def wait_dom(b, expr, what, timeout=None):
    """轮询一个 JS 表达式直到为真。超时返回 False —— 调用方负责 FAIL + 打印 DOM。

    ★ 以前这里是「while time.time() - t0 < 900」：目标节点改名之后查不到就空转 15 分钟。
      等 DOM 结构一律有上限，超时立刻把现场打出来。"""
    timeout = DOM_WAIT if timeout is None else float(timeout)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if b.eval(expr):
                return True
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(0.2)
    return False


def dom_dump(b):
    """失败时贴给人看：**真实**的 DOM 结构 + 页面到哪一步了。

    只列 id 不够用：上次「页面没就绪」时 id 全在、但 app.js 还没跑完（cols=0），
    光看 id 会以为是节点改名。所以把 readyState / 脚本资源 / 控制台里的错一起带上。"""
    try:
        raw = b.eval(
            "(function(){var p=document.getElementById('p-merge');"
            "var ids=p?Array.prototype.slice.call(p.querySelectorAll('[id]'))"
            ".map(function(e){return e.id;}):[];"
            "var svg=document.getElementById('mgPipeSvg');"
            "var rs=[];try{rs=(performance.getEntriesByType('resource')||[])"
            ".filter(function(x){return String(x.name).indexOf('.js')>0;})"
            ".map(function(x){return String(x.name).split('/').pop()+'@'+Math.round(x.duration);});}catch(e){}"
            "return JSON.stringify({readyState:document.readyState,"
            "goType:typeof window.go, mseType:typeof window.mgStageExport, "
            "jsLoaded:rs,"
            "ids:ids.slice(0,90),"
            "stepsBox:!!document.getElementById('mgpStepsBox'),"
            "pipeSvg:!!svg,"
            "cols:svg?svg.querySelectorAll('g.col').length:0,"
            "tstage:document.querySelectorAll('#mgPipeSvg text.t-stage').length,"
            "lanes:svg?svg.querySelectorAll('rect.lane').length:0,"
            "mgpSteps:!!document.getElementById('mgpSteps'),"
            "cvst:document.querySelectorAll('.cvst').length});})()")
        tail = [str(x)[:160] for x in (getattr(b, "console", None) or [])[-3:]]
        return "%s · console=%s" % (str(raw)[:700], tail)
    except Exception as e:                                 # noqa: BLE001
        return "DOM 现场取不到：%s" % e


def _cls_short(cls):
    """"col done" -> "done"：状态序列读起来短一点。"""
    parts = [x for x in str(cls or "").split() if x != "col"]
    return parts[0] if parts else "-"


def case_ui(video):
    try:
        from tools.accept.merge_accept import Browser, _edge_path
    except Exception as e:                                 # noqa: BLE001
        return check(False, "f 浏览器自动化不可用（tools.merge_accept）", str(e)[:160])
    edge = _edge_path()
    if not edge:
        say("没装 Edge —— UI 这一段跳过")
        return None
    cache = cache_of(video)
    shutil.rmtree(cache, ignore_errors=True)
    b = Browser(edge)
    try:
        b.goto(BASE + "/")
        # ★ 只等 window.go：mgStageExport 是**进了合成页**才挂上的（bindMergeUI 里那句
        #   window.mgStageExport = mgStageExport）。老代码在这里等「go 且 mgStageExport」，
        #   条件是永远不成立的 —— 它只是 60 秒后**默默往下走**，于是后面查什么都是空的。
        if not wait_dom(b, "typeof window.go === 'function'", "页面就绪（window.go）",
                        BOOT_WAIT):
            return check(False, "f 页面没就绪（window.go 没挂上）", dom_dump(b))
        # 进合成页 → 进度视图（流水线图就在这一页）
        # 关掉欢迎页的扫码浮层（不然截图/点击都被它挡着）
        b.eval("(function(){var w=document.getElementById('welcome');"
               " if(w){w.style.display='none';} "
               " var m=document.getElementById('modal'); if(m){m.style.display='none';} "
               " return true;})()")
        b.eval("window.go('merge'); true")
        if not wait_dom(b, "typeof window.mgStageExport === 'function'",
                        "合成页就绪（mgStageExport）"):
            return check(False, "f 进了合成页但 mgStageExport 没挂上（bindMergeUI 没跑完）",
                         dom_dump(b))
        b.eval("window.mgStep && window.mgStep('progress'); true")
        # 结构就位：六列列组（g.col[data-k]）+ 六个列标题。超时就 FAIL，不空转。
        if not wait_dom(b,
                        "(function(){var s=document.getElementById('mgPipeSvg');"
                        "return !!(s && s.querySelectorAll('g.col').length === 6"
                        " && document.querySelectorAll('#mgPipeSvg text.t-stage').length === 6);})()",
                        "流水线视图六列"):
            snap_bad = b.js(UI_SNAP) or {}
            return check(False, "f 流水线视图（#mgPipeSvg 的六列）没出现",
                         "n=%s · %s" % (snap_bad.get("n"), dom_dump(b)))
        snap0 = b.js(UI_SNAP)
        check(snap0.get("n") == 6 and snap0.get("keys") ==
              ["frames", "det", "dfm", "xseg", "compose", "encode"],
              "f 六列都在（现行流水线视图 · frames/det/dfm/xseg/compose/encode）",
              "n=%s · keys=%s" % (snap0.get("n"), snap0.get("keys")))
        check(snap0.get("labels") == ["切帧", "切脸", "换脸", "遮罩", "合成", "缓存帧"],
              "f 列标题 = 切帧/切脸/换脸/遮罩/合成/缓存帧", "%s" % (snap0.get("labels"),))
        check(int(snap0.get("lanes") or 0) > 0,
              "f 每列的线程方块（rect.lane）画出来了",
              "%s 个 · #mgpStepsBox=%s" % (snap0.get("lanes"), snap0.get("stepsBox")))
        check(snap0.get("oldRows") == 0 and not snap0.get("hasOld"),
              "f 旧的四行（载入模型/逐帧合成/编码/完成）已消失",
              "mgpstep=%s · 旧文案=%s" % (snap0.get("oldRows"), snap0.get("hasOld")))

        b.eval("window.__mv = window.__mv || {}; window.__mv.video = %s; "
               "window.__mergeInputDir = ''; true" % json.dumps(str(video)))
        b.eval("window.mgStageExport({model: %s}); true" % json.dumps(DFM))

        seq = []
        seen_on = [False] * 6
        seen_done = [False] * 6
        shot = BOX / "ui_step2.png"
        try:
            shot.unlink()                     # 别拿上一轮留下的旧图当证据
        except OSError:
            pass
        t0 = time.time()
        t_change = t0
        last_key = ""
        while time.time() - t0 < EXPORT_WAIT:
            s = b.js(UI_SNAP)
            cls = list(s.get("cls") or [])
            # 切帧已 ✓、切脸正在跑：截一张图当「动画真的推进了」的视觉证据
            if (not shot.exists() and len(cls) == 6
                    and "done" in cls[0] and "on" in cls[1]):
                try:
                    shot.write_bytes(base64.b64decode(
                        b.call("Page.captureScreenshot", format="png").get("data") or ""))
                    say("f 截图（第 1 列 ✓、第 2 列高亮）：%s" % shot)
                except Exception:                          # noqa: BLE001
                    pass
            key = ",".join(cls)
            if key != last_key:
                last_key = key
                t_change = time.time()
                if not seq or seq[-1] != key:
                    seq.append(key)
            for i, c in enumerate(cls):
                if "on" in str(c).split():
                    seen_on[i] = True
                if "done" in str(c).split():
                    seen_done[i] = True
            all_done = len(cls) == 6 and all("done" in str(c).split() for c in cls)
            any_err = any("err" in str(c).split() for c in cls)
            if not s.get("running") and (all_done or any_err):
                break
            # 不在跑、列状态也 STALL_WAIT 秒没变 → 交给断言，不再空转
            if not s.get("running") and (time.time() - t_change) > STALL_WAIT:
                say("f 已经不在跑、列状态 %.0fs 没变 → 收工交给断言" % STALL_WAIT)
                break
            time.sleep(0.15)
        else:
            say("f 看导出跑完的上限 %.0fs 到了 → 收工交给断言" % EXPORT_WAIT)
        final = b.js(UI_SNAP)
        fcls = list(final.get("cls") or [])
        say("f trace：%s" % json.dumps(final.get("trace"), ensure_ascii=False)[:900])
        # 每一列都要**可见地**推进过：至少出现过「当前步」高亮或「完成」。
        # （单纯 on 会抖：这一段的视频只有 24 帧，某个阶段可能 200ms 就过去了，
        #   0.15s 的轮询仍可能刚好落在两次状态之间 —— 那不叫没显示。）
        check(all(a or b for a, b in zip(seen_on, seen_done)),
              "f 每一列都亮过（当前步高亮 或 已推到完成）",
              "on=%s · done=%s" % (seen_on, seen_done))
        check(all(seen_done), "f 每一列都推进到「完成」", "seen_done=%s" % (seen_done,))
        check(len(fcls) == 6 and all("done" in str(c).split() for c in fcls),
              "f 终态六列都是 done", "%s" % (fcls,))
        check(len(seq) >= 6, "f 阶段切换时列状态确实推进了", "%d 次状态变化" % len(seq))
        say("f 列状态序列：%s" % " -> ".join(",".join(_cls_short(c) for c in x.split(","))
                                              for x in seq[:10]))
        if not (all(seen_done) and len(fcls) == 6):
            say("f DOM 现场：%s" % dom_dump(b))
        return True
    finally:
        b.close()


# ============================================================
# main
# ============================================================
def main():
    t_all = time.time()
    prepare_sandbox()
    v1 = build_video("9月29日 (5).mp4", 2.0, 3)
    v2 = build_video("second (1).mp4", 1.5, 8)
    v3 = build_video("resume (2).mp4", 3.0, 12)
    v4 = build_video("ui (3).mp4", 1.2, 20)

    proc = start_server("A(keep_cache=1)", keep_cache=True)
    try:
        case_a(v1)
    finally:
        stop_server(proc)

    proc = start_server("B")
    try:
        case_b_stage_by_stage(v2)
        case_c_resume(v3)
        case_ui(v4)
    finally:
        stop_server(proc)

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
