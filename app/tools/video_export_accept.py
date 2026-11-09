# -*- coding: utf-8 -*-
r"""视频模式导出验收：真后端 + 真视频 + 现有 .dfm，全部在 %TEMP% 里跑。

验的是用户定下的四条（按这四条断言，不加别的口径）：

  a) 视频模式、**没有 DFL 项目、没有人脸库** → 点导出真的出 mp4：
     文件存在、时长≈源、有音轨、抽 3 帧和单帧预览逐像素一致（允许 ≤8/255），
     日志显示每一帧都走了 dfm。
  b) 视频模式、**有人脸库**（3 张某人脸）→ 每帧 faces[].dfm 按下标正确；
     不匹配的那张脸保持原像素（不兜给任何模型）。
  c) 进程重启 → 缓存计数 0；导出结束后再查 → 也是 0。
  d) 无人脸帧 → 导出时复制原帧（不跑 dfm），有耗时/日志为证。

跑法（仓库根）：

    runtime\server\Scripts\python.exe app\tools\video_export_accept.py

沙箱：%TEMP%\dfl_videxport_accept\
    ws\         DFL_WORKSPACE（ws\model 是指向仓库 workspace\model 的目录联接，
                .dfm 是真的，但一个字节都不会写进仓库）
    ws\merge\   默认人脸库（<工作区>/merge —— 后端 resolve_library("") 就是它）
    media\      测试视频（从仓库那段 mp4 现切，只写 %TEMP%）
    out\        成片
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
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_videxport_accept"
WS = BOX / "ws"
LIB = WS / "merge"                     # 默认库：resolve_library("") 落到这里
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"                   # 小一点的那个（539 MB），416 输入
GROUP = {"name": "张元英", "dfm": "张元英"}
THRESHOLD = 0.35
TOL = 8                                # 逐像素容差（0~255）

BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LAST_ERROR = ""


# ============================================================
# 小工具
# ============================================================
def check(ok, name, detail=""):
    ok = bool(ok)
    RESULTS.append((ok, name, str(detail)))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  · " + str(detail) if detail else ""),
          flush=True)
    return ok


def say(msg):
    print("       " + str(msg), flush=True)


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
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-400:]))
    return r


def probe(path):
    """(时长, 有没有音轨, 帧数, fps, 宽, 高) —— 时长/音轨问 ffmpeg，帧数问 OpenCV。"""
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
    return {"dur": dur, "audio": bool(audio), "frames": frames, "fps": fps,
            "w": w, "h": h}


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


def diff(a, b, box=None):
    """逐像素差：mean / p99 / max（box=[x,y,w,h] 时只比这一块）。"""
    import numpy as np
    if box is not None:
        x, y, w, h = [int(v) for v in box]
        a = a[max(0, y):y + h, max(0, x):x + w]
        b = b[max(0, y):y + h, max(0, x):x + w]
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), float(np.percentile(d, 99)), int(d.max())


def b64img(img, quality=95):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def imread_any(path):
    """读图：cv2.imread 在 Windows 上按 ANSI 解析路径，中文名一律 None（走 imdecode）。"""
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


# ============================================================
# 沙箱 / 后端
# ============================================================
def _drop_model_link():
    r"""摘掉 ws\model 这个目录联接（rmdir 只删链接，绝不碰仓库里的 .dfm）。"""
    model = WS / "model"
    try:
        if os.path.isdir(str(model)) or os.path.islink(str(model)):
            os.rmdir(str(model))
    except OSError:
        pass


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


def prepare_sandbox(reuse=False):
    if BOX.exists() and not reuse:
        _drop_model_link()
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, LIB, MEDIA, OUT, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    model = WS / "model"
    src = REPO / "workspace" / "model"
    if not model.exists():
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(model), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建目录联接失败：%s %s" % (r.stdout, r.stderr))
    say("沙箱：%s" % BOX)
    say("工作区：%s（model → %s）" % (WS, src))


def start_server(tag=""):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
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
    say("后端%s：%s（pid=%s）" % (tag, BASE, proc.pid))
    global LAST_ERROR
    while time.time() - t0 < 240:
        try:
            st, d = get("/api/merge/faces?project=", timeout=10)
            if st == 200 and d.get("ok"):
                return proc
            LAST_ERROR = "HTTP %s %s" % (st, str(d)[:200])
        except Exception as e:                             # noqa: BLE001
            LAST_ERROR = "%s: %s" % (type(e).__name__, e)
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/faces 不应答（%s），见 %s"
                     % (LAST_ERROR, SERVER_LOG))


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                      # noqa: BLE001
        pass
    time.sleep(1.0)


# ============================================================
# 素材
# ============================================================
def source_video():
    vids = sorted((REPO / "workspace").glob("*.mp4"))
    if not vids:
        raise SystemExit("仓库 workspace 里没有 .mp4 素材")
    return vids[0]


def build_face():
    """face.mp4：源片 t=3s 起 2 秒 @30fps，音轨原样带上（案例 a 用）。"""
    src = source_video()
    face = MEDIA / "face.mp4"
    run_ff(["-ss", "3", "-t", "2", "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", face],
           "切 face.mp4")
    frames = decode_all(face)
    if len(frames) < 60:
        raise SystemExit("face.mp4 只有 %d 帧，素材不够" % len(frames))
    say("face.mp4：%s" % probe(face))
    return {"face": face, "frames": frames}


def build_mixed(media):
    """mixed.mp4：0.5s 灰场 + 1s 人脸 + 0.5s 灰场（60 帧，案例 d 用）。"""
    import cv2
    import numpy as np
    frames = media["frames"]
    h, w = frames[0].shape[:2]
    seq = MEDIA / "seq_mixed"
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir(parents=True, exist_ok=True)
    gray = np.full((h, w, 3), 96, np.uint8)
    k = 0
    for _ in range(15):
        cv2.imwrite(str(seq / ("%04d.png" % k)), gray)
        k += 1
    for i in range(30):
        cv2.imwrite(str(seq / ("%04d.png" % k)), frames[i])
        k += 1
    for _ in range(15):
        cv2.imwrite(str(seq / ("%04d.png" % k)), gray)
        k += 1
    mixed = MEDIA / "mixed.mp4"
    run_ff(["-framerate", "30", "-i", seq / "%04d.png", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", mixed], "编 mixed.mp4")
    say("mixed.mp4：%s（前 15 + 后 15 帧是灰场，没有脸）" % probe(mixed))
    return {"mixed": mixed}


def build_two(media, patch_path, frac=0.22, name="two"):
    """two.mp4：45 帧「本人 + 另一个人」（把别人一张脸贴左上）。"""
    import cv2
    frames = media["frames"]
    patch = imread_any(patch_path)
    if patch is None:
        return None, None
    h, w = frames[0].shape[:2]
    size = max(200, int(min(h, w) * frac))
    pp = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    x, y = int(w * 0.06), int(h * 0.08)
    seq = MEDIA / ("seq_" + name)
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir(parents=True, exist_ok=True)
    for i in range(45):
        img = frames[i].copy()
        img[y:y + size, x:x + size] = pp
        cv2.imwrite(str(seq / ("%04d.png" % i)), img)
    clip = MEDIA / (name + ".mp4")
    run_ff(["-framerate", "30", "-i", seq / "%04d.png", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", clip], "编 %s.mp4" % name)
    return clip, [x, y, size, size]


def detects_two(clip) -> bool:
    """这份片子是不是每一帧都稳定检出 2 张脸（用例 b 的前提）。"""
    st, d = post("/api/merge/open", {"path": str(clip)})
    if st != 200:
        return False
    total = int(d.get("frames") or 0)
    for n in range(0, total):            # 每一帧都要 2 张脸，不抽样
        st, r = get("/api/merge/detect/%d?model=RetinaFace_10g" % n)
        if st != 200 or len(r.get("boxes") or []) != 2:
            return False
    return True


def patch_is_other(path):
    """这张脸是不是「库里谁都对不上」的那个人（用例 b 的前提）。

    借 /api/merge/faces/add 的判定：不指定 group、也不指定 dfm 时，后端拿这张脸
    的 embedding 和每个组的均值比 —— 不到阈值就 **什么都不写** 直接回
    reason=no_group_match。真的匹配上了说明是同一个人的另一张脸，把它挪进回收站，
    库回到 3 张，然后换下一个候选。
    """
    st, d = post("/api/merge/faces/add",
                 {"project": str(LIB), "frame": 0, "idx": 0,
                  "frame_path": str(path), "detector": "RetinaFace_10g"})
    if st != 200:
        return None, "HTTP %s %s" % (st, str(d.get("error") or "")[:120])
    if d.get("ok") is False and d.get("reason") == "no_group_match":
        return True, "similarity=%s < %.2f（库里谁都对不上）" % (d.get("similarity"), THRESHOLD)
    if d.get("ok"):
        pid = str(d.get("id") or "")
        if pid:
            post("/api/merge/faces/trash", {"project": str(LIB), "ids": [pid]})
        return False, "matched=True（是库里那个人的另一张脸）similarity=%s" % d.get("similarity")
    return None, str(d.get("reason") or d.get("error") or "判定失败")


def pick_two(media):
    """挑一张「另一个人」的素材：

      1) 45 帧每一帧都要稳定检出 2 张脸（贴上去得被认出来）；
      2) 这张脸要和库里那 3 张**都不像**（否则「不匹配就不换」就没得验）。

    不硬编码某一张：换台机器 data_src 里的素材就不一样了。按顺序试，第一个
    两条都满足的胜出。
    """
    cands = []
    aligned = sorted((REPO / "workspace" / "data_src" / "aligned").glob("*.jpg"))
    for k in (5, 40, 300, 120, 0, 1):
        if len(aligned) > k:
            cands.append(("aligned%d" % k, aligned[k]))
    photos = sorted(Path("D:/MyProject/示例图").glob("*.jpg"))
    for k in range(min(3, len(photos))):
        cands.append(("photo%d" % k, photos[k]))
    if not cands:
        raise SystemExit("找不到「另一个人」的素材（data_src/aligned、示例图）")
    for name, path in cands:
        other, why = patch_is_other(path)
        if not other:
            say("候选 %s 不是「另一个人」：%s" % (name, why))
            continue
        clip, paste = build_two(media, path, 0.22, name)
        if clip is None:
            say("候选 %s 读不出来：%s" % (name, path))
            continue
        if detects_two(clip):
            say("「另一个人」素材：%s（%s）· %s · 贴 %s" % (name, path, why, paste))
            return {"two": clip, "paste": paste, "patch": str(path)}
        say("候选 %s 有几帧只检出 1 张脸，换下一个" % name)
    raise SystemExit("找不到「既认得出、又和库里都不像」的素材")


# ============================================================
# 导出 + 状态
# ============================================================
def start_export(body, tag):
    st, d = post("/api/merge/export", body)
    check(st == 200 and d.get("ok"), "%s 导出请求被接受（视频模式，不需要项目）" % tag,
          "HTTP %s %s" % (st, str(d.get("error") or "")[:160]))
    return st == 200 and bool(d.get("ok")), d


def wait_export(tag, timeout=900.0, on_tick=None):
    t0 = time.time()
    last = ""
    final = {}
    while time.time() - t0 < timeout:
        st, s = get("/api/merge/export/status")
        if st != 200:
            time.sleep(1.0)
            continue
        msg = "%s%% %s" % (s.get("pct"), s.get("msg"))
        if msg != last:
            print("       [%s] %.1fs %s" % (tag, time.time() - t0, msg), flush=True)
            last = msg
        if on_tick:
            on_tick(s)
        if not s.get("running"):
            final = s
            break
        time.sleep(1.0)
    ok = bool(final.get("ok")) and final.get("phase") == "done"
    check(ok, "%s 导出跑完（ok/phase）" % tag,
          "phase=%s error=%s elapsed=%s" % (final.get("phase"),
                                            str(final.get("error") or "")[:160],
                                            final.get("elapsed")))
    return final


def cache_state():
    st, d = get("/api/merge/faces/cache")
    return (d if st == 200 else {"error": "HTTP %s" % st})


def clean_cache(d, tag):
    extra = d.get("extra") or {}
    ok = (int(d.get("profiles") or 0) == 0
          and all(int(v or 0) == 0 for v in extra.values())
          and not (d.get("files") or []))
    check(ok, "%s 缓存为 0（组均值/逐帧方案/磁盘文件）" % tag,
          "profiles=%s extra=%s files=%s"
          % (d.get("profiles"), extra, d.get("files")))


def log_lines(out_mp4):
    p = Path(str(out_mp4)).with_suffix(".log")
    if not p.is_file():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


# ============================================================
# 案例 a：无项目、无脸库
# ============================================================
def case_a(media):
    print("\n--- a) 视频模式 / 没有项目 / 没有人脸库 ---", flush=True)
    clip = media["face"]
    src = probe(clip)
    st, d = get("/api/merge/faces?project=")
    groups = (d.get("groups") or []) if st == 200 else []
    check(not groups, "前提：默认人脸库里没有任何组", "groups=%d" % len(groups))

    ok, started = start_export({"video": str(clip), "model": DFM, "settings": {},
                                "output_dir": str(OUT)}, "a")
    if not ok:
        return None
    final = wait_export("a")
    if not final.get("ok"):
        return None
    out = Path(str(final.get("out") or ""))
    check(out.is_file() and out.stat().st_size > 0, "a 成片存在且非空",
          "%s · %.2f MB" % (out.name, (out.stat().st_size / 1048576.0) if out.is_file() else 0))
    dst = probe(out)
    check(abs(dst["dur"] - src["dur"]) <= 0.15, "a 时长≈源",
          "源 %.2fs / 成片 %.2fs" % (src["dur"], dst["dur"]))
    check(dst["audio"], "a 成片有音轨（-c:a copy 搬的原音轨）",
          "源有音轨=%s" % src["audio"])
    check(dst["frames"] == src["frames"],
          "a 帧数与源一致", "源 %d / 成片 %d" % (src["frames"], dst["frames"]))

    vid = final.get("vid") or {}
    recs = vid.get("frames") or []
    stats = vid.get("stats") or {}
    check(len(recs) == src["frames"], "a 逐帧记录数 = 源帧数", "%d 条" % len(recs))
    all_dfm = all((r.get("faces") or [{}])[0].get("dfm") == DFM for r in recs if r.get("faces"))
    check(all_dfm and recs, "a 每帧的脸都用左栏选中的 dfm（%s）" % DFM,
          "有脸的帧 %d 条，全部 dfm=%s" % (sum(1 for r in recs if r.get("faces")), DFM))

    lines = log_lines(out)
    frame_lines = [ln for ln in lines if ln.startswith("[帧 ")]
    dfm_lines = [ln for ln in frame_lines if "dfm=" + DFM in ln]
    check(len(frame_lines) == src["frames"] and len(dfm_lines) == src["frames"],
          "a 日志逐帧留痕（每帧都走了 dfm）",
          "逐帧行 %d / 带 dfm 的行 %d / 共 %d 帧（日志 %s）"
          % (len(frame_lines), len(dfm_lines), src["frames"], out.with_suffix(".log").name))

    # ---- 抽 3 帧 vs 单帧预览 ----
    ex = decode_all(out)
    worst = 0.0
    for n in (5, 30, 55):
        st, pv = post("/api/merge/preview",
                      {"video": str(clip), "t": round(n / max(1.0, src["fps"]), 4),
                       "models": [{"name": DFM, "faces": []}],
                       "settings": {}}, timeout=120)
        if st != 200 or not pv.get("png"):
            check(False, "a 第 %d 帧单帧预览取不到" % n, str(pv.get("error") or "")[:160])
            continue
        prev = read_png(pv["png"])
        exp = ex[n] if n < len(ex) else None
        if exp is None:
            check(False, "a 成片里没有第 %d 帧" % n)
            continue
        mean, p99, mx = diff(prev, exp)
        # 对齐证明：相邻帧的差应该**明显更大**（否则说明比的是同一帧，白测）
        nb = [diff(prev, ex[j])[0] for j in (n - 1, n + 1) if 0 <= j < len(ex)]
        worst = max(worst, mean)
        check(mean <= TOL,
              "a 第 %d 帧 逐像素一致（成片 vs 单帧预览，容差 %d）" % (n, TOL),
              "mean=%.2f p99=%.1f max=%d · 相邻帧 mean=%s"
              % (mean, p99, mx, ["%.1f" % v for v in nb]))
    check(worst <= TOL, "a 三帧里最差的一帧也在容差内", "worst mean=%.2f" % worst)

    clean_cache(cache_state(), "a")
    say("a 真实速度：%.2f 帧/秒（%d 帧 / %.1f s）· 编码 %.2f s · 成片 %.2f MB"
        % (stats.get("out_fps") or 0, int(stats.get("frames") or 0),
           float(stats.get("elapsed") or 0), float(stats.get("encode_s") or 0),
           float(stats.get("mb") or 0)))
    return {"final": final, "stats": stats}


# ============================================================
# 案例 b：有人脸库
# ============================================================
def setup_library(media):
    import cv2
    st, d = post("/api/merge/faces/groups", {"project": str(LIB), "dfms": [GROUP["dfm"]]})
    ok = st == 200 and d.get("ok")
    check(ok, "b 建一个组（%s）" % GROUP["name"], str(d.get("error") or "")[:160])
    groups = d.get("groups") or []
    gid = ""
    for g in groups:
        if str(g.get("dfm") or "") == GROUP["dfm"]:
            gid = g.get("id") or ""
    if not gid:
        raise SystemExit("拿不到组 id：%s" % groups)
    idx_path = LIB / "_faces" / "index.json"
    raw = json.loads(idx_path.read_text(encoding="utf-8"))
    for g in raw.get("groups") or []:
        if g.get("id") == gid:
            g["name"] = GROUP["name"]
    idx_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    clip = media["face"]
    frames = media["frames"]
    added = 0
    for n in (0, 20, 40):
        st, d = post("/api/merge/faces/add",
                     {"project": str(LIB), "frame": n, "idx": 0,
                      "source": {"video": "accept_face", "frame": n},
                      "frame_b64": b64img(frames[n]), "group": gid})
        if st == 200 and d.get("ok"):
            added += 1
        else:
            say("入库失败（帧 %d）：%s" % (n, str(d.get("error") or d.get("reason"))[:160]))
    check(added == 3, "b 组里放进 3 张同一个人的脸", "成功 %d 张" % added)
    st, d = get("/api/merge/faces?project=" + urllib.parse.quote(str(LIB)))
    faces = d.get("faces") or []
    check(len(faces) == 3, "b 库里有 3 张脸", "faces=%d" % len(faces))
    return gid


def case_b(media):
    print("\n--- b) 视频模式 / 有人脸库（3 张脸）---", flush=True)
    clip = media["two"]
    src_frames = decode_all(clip)
    paste = media["paste"]

    peaks = {"profiles": 0, "auto_plan": 0}

    def tick(_s):
        c = cache_state()
        peaks["profiles"] = max(peaks["profiles"], int(c.get("profiles") or 0))
        peaks["auto_plan"] = max(peaks["auto_plan"],
                                 int((c.get("extra") or {}).get("auto_plan") or 0))

    ok, started = start_export({"video": str(clip), "model": GROUP["dfm"],
                                "detector": "RetinaFace_10g", "faces_project": "",
                                "settings": {}, "output_dir": str(OUT)}, "b")
    if not ok:
        return None
    final = wait_export("b", on_tick=tick)
    if not final.get("ok"):
        return None
    vid = final.get("vid") or {}
    recs = vid.get("frames") or []
    check(bool(recs), "b 有逐帧记录", "%d 条" % len(recs))

    st = {"match": 0, "unmatched": 0, "two": 0, "idx_ok": 0}
    match_box = None
    for r in recs:
        fs = r.get("faces") or []
        if not fs:
            continue
        if [f.get("idx") for f in fs] == list(range(len(fs))):
            st["idx_ok"] += 1
        if len(fs) >= 2:
            st["two"] += 1
        for f in fs:
            if f.get("matched") and f.get("dfm") == GROUP["dfm"]:
                st["match"] += 1
                if match_box is None:
                    match_box = f.get("box")
            elif not f.get("matched") and not f.get("dfm"):
                st["unmatched"] += 1
    check(st["two"] == len(recs) and st["idx_ok"] == len(recs),
          "b 每帧都检出 2 张脸、下标 = 检测顺序",
          "两脸帧 %d/%d · 下标对齐 %d/%d" % (st["two"], len(recs), st["idx_ok"], len(recs)))
    check(st["match"] == len(recs),
          "b 每帧里那个人都归到「%s」（>=%.2f）" % (GROUP["name"], THRESHOLD),
          "命中 %d 次 / %d 帧" % (st["match"], len(recs)))
    check(st["unmatched"] == len(recs),
          "b 另一个人的脸每帧都 matched=false 且 dfm 为空（不兜给任何模型）",
          "未匹配 %d 次 / %d 帧" % (st["unmatched"], len(recs)))
    check(peaks["profiles"] >= 1,
          "b 导出途中组均值缓存真的建起来了（导出结束才清）",
          "profiles 峰值 %d · auto_plan 峰值 %d" % (peaks["profiles"], peaks["auto_plan"]))

    # ---- 不匹配的脸保持原像素：拿贴上去那块和源帧比 ----
    out = Path(str(final.get("out") or ""))
    ex = decode_all(out)
    worst = 0.0
    for n in (0, 20, 44):
        if n >= len(ex) or n >= len(src_frames):
            continue
        mean, p99, mx = diff(src_frames[n], ex[n], box=paste)
        worst = max(worst, mean)
        check(mean <= TOL,
              "b 第 %d 帧：不匹配那张脸的区域保持原像素（只差编码噪声）" % n,
              "mean=%.2f p99=%.1f max=%d（区域 %s）" % (mean, p99, mx, paste))
    # 匹配那张脸必须真的被动过（否则「不匹配那张保持原像素」这条就说不通了）。
    # 注意：源片那个人和组里的脸是同一个人、组的 dfm 也是同一个人，所以换完画面
    # 本来就接近 —— 这里只判「不是逐像素不变」，不拿它当画质断言。
    if match_box:
        mean, p99, mx = diff(src_frames[0], ex[0], box=match_box)
        check(mean > 1.0, "b 对照：匹配那张脸确实被换掉了（不是原像素）",
              "换脸框 %s · mean=%.2f p99=%.1f max=%d" % (match_box, mean, p99, mx))

    clean_cache(final.get("vid", {}).get("cache") or {}, "b（job 自报）")
    clean_cache(cache_state(), "b（接口复查）")
    stats = vid.get("stats") or {}
    say("b 真实速度：%.2f 帧/秒（%d 帧 / %.1f s）"
        % (stats.get("out_fps") or 0, int(stats.get("frames") or 0),
           float(stats.get("elapsed") or 0)))
    return {"final": final, "stats": stats}


# ============================================================
# 案例 e：取消也算「导出结束」—— 缓存照样清
# ============================================================
def case_cancel(media):
    print("\n--- e) 取消导出：终态是 cancelled，缓存照样清 ---", flush=True)
    clip = media["face"]
    ok, _d = start_export({"video": str(clip), "model": GROUP["dfm"], "settings": {},
                           "output_dir": str(OUT)}, "取消")
    if not ok:
        return None
    # 等它真的开始逐帧跑（不然取消落在「加载模型」阶段，测不到逐帧收手）
    t0 = time.time()
    mid = {}
    while time.time() - t0 < 120:
        st, s = get("/api/merge/export/status")
        if s.get("vid", {}).get("done"):
            mid = cache_state()
            break
        if not s.get("running"):
            break
        time.sleep(0.3)
    check(bool(mid), "e 取消前已经跑过至少一帧（逐帧进度已在动）",
          "取消前缓存：profiles=%s extra=%s"
          % ((mid or {}).get("profiles"), (mid or {}).get("extra")))
    st, d = post("/api/merge/export/cancel", {})
    check(st == 200 and d.get("cancelled") is not False, "e 取消请求被接受",
          "HTTP %s %s" % (st, str(d.get("reason") or "")[:80]))
    # 等**线程真的收工**：取消时状态接口会先一步把 running 置 False
    # （merge_export_status 的既有语义：取消立刻对外不算 running），
    # 但那一步不代表 worker 收完尾。ok 变成非 None 才是终态写好了
    # —— 那时缓存已经清空、日志也已经落盘（见 _vid_worker 的 finally 顺序）。
    final = {}
    t1 = time.time()
    while time.time() - t1 < 180:
        st, s = get("/api/merge/export/status")
        final = s
        if s.get("ok") is not None and not s.get("running"):
            break
        time.sleep(0.3)
    check(final.get("phase") == "cancelled" and final.get("cancelled") is True
          and final.get("ok") is False,
          "e 终态是 cancelled（不是 done / 不是卡在 running）",
          "phase=%s cancelled=%s ok=%s msg=%s"
          % (final.get("phase"), final.get("cancelled"), final.get("ok"),
             str(final.get("msg") or "")[:40]))
    clean_cache(final.get("vid", {}).get("cache") or {}, "e（job 自报，取消也要清）")
    clean_cache(cache_state(), "e（接口复查）")
    txt = "\n".join(log_lines(final.get("out")))
    check("缓存已清空" in txt, "e 取消后日志里留了「缓存已清空」那一行",
          "日志 %s" % Path(str(final.get("out") or "")).with_suffix(".log").name)
    return {"final": final}


# ============================================================
# 案例 c：进程重启
# ============================================================
def case_c_restart():
    print("\n--- c) 进程重启 → 缓存 0 ---", flush=True)
    txt = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    hits = [ln for ln in txt.splitlines() if "[face-cache] 启动清理" in ln]
    check(bool(hits), "c serve 启动时调了 clear_face_group_cache()",
          hits[-1][:140] if hits else "日志里没有 [face-cache] 启动清理")
    d = cache_state()
    clean_cache(d, "c 重启后")
    return d


# ============================================================
# 案例 d：无人脸帧
# ============================================================
def case_d(media):
    print("\n--- d) 无人脸帧：复制原帧（不跑 dfm）---", flush=True)
    clip = media["mixed"]
    src = probe(clip)
    src_frames = decode_all(clip)
    ok, started = start_export({"video": str(clip), "model": GROUP["dfm"],
                                "settings": {}, "output_dir": str(OUT)}, "d")
    if not ok:
        return None
    final = wait_export("d")
    if not final.get("ok"):
        return None
    recs = (final.get("vid") or {}).get("frames") or []
    no_face = [r for r in recs if r.get("no_face")]
    face = [r for r in recs if not r.get("no_face")]
    check(len(no_face) == 30, "d 无人脸帧共 30 帧（灰场 15+15）", "%d 帧" % len(no_face))
    check(all(r.get("branch") == "copy" and not (r.get("faces") or [])
              and int(r.get("swapped") or 0) == 0 for r in no_face),
          "d 这些帧没跑 dfm（branch=copy、faces 为空、swapped=0）",
          "示例：frame=%s branch=%s faces=%s"
          % (no_face[0].get("frame") if no_face else None,
             no_face[0].get("branch") if no_face else None,
             no_face[0].get("faces") if no_face else None))

    lines = log_lines(final.get("out"))
    copy_lines = [ln for ln in lines if "无人脸 -> 原帧直出" in ln]
    check(len(copy_lines) == len(no_face),
          "d 日志逐帧写明「无人脸 -> 原帧直出（没跑 dfm）」", "%d 行" % len(copy_lines))

    ms_no = [float(r.get("ms") or 0) for r in no_face]
    ms_face = [float(r.get("ms") or 0) for r in face]
    avg_no = sum(ms_no) / len(ms_no) if ms_no else 0.0
    avg_face = sum(ms_face) / len(ms_face) if ms_face else 0.0
    check(avg_face > 0 and avg_no * 3 < avg_face,
          "d 耗时对比：无人脸帧远快于有脸帧（没跑 dfm）",
          "无人脸 %.1f ms/帧 · 有脸 %.1f ms/帧（%.1f 倍）"
          % (avg_no, avg_face, (avg_face / avg_no) if avg_no else 0))

    out = Path(str(final.get("out") or ""))
    ex = decode_all(out)
    worst = 0.0
    for n in (0, 50, 59):
        if n >= len(ex) or n >= len(src_frames):
            continue
        mean, p99, mx = diff(src_frames[n], ex[n])
        worst = max(worst, mean)
        check(mean <= TOL, "d 第 %d 帧（无人脸）与源帧逐像素一致" % n,
              "mean=%.2f p99=%.1f max=%d" % (mean, p99, mx))
    clean_cache(cache_state(), "d（导出结束后）")
    say("d 成片 %.2f 帧/秒 · %d 帧 · 编码 %.2f s"
        % ((final.get("vid") or {}).get("stats", {}).get("out_fps") or 0,
           (final.get("vid") or {}).get("stats", {}).get("frames") or 0,
           (final.get("vid") or {}).get("stats", {}).get("encode_s") or 0))
    return {"final": final}


# ============================================================
# main
# ============================================================
def main() -> int:
    print("仓库：%s" % REPO)
    print("Python：%s" % PY)
    prepare_sandbox()
    media = build_face()
    media.update(build_mixed(media))

    proc = start_server("#1")
    try:
        print("\n--- 前提：启动时缓存就是 0 ---", flush=True)
        clean_cache(cache_state(), "启动后（调用 clear_face_group_cache）")

        a = case_a(media)
        if a:
            setup_library(media)
            media.update(pick_two(media))       # 要建完库才判得出「谁不是库里那个人」
            b = case_b(media)
            case_cancel(media)
        else:
            check(False, "b/e 跳过（a 没过）")
    finally:
        stop_server(proc)

    proc2 = start_server("#2 (重启)")
    try:
        case_c_restart()
        d = case_d(media)
    finally:
        stop_server(proc2)

    total = len(RESULTS)
    bad = [r for r in RESULTS if not r[0]]
    print("\n================ 汇总 ================")
    for ok, name, detail in RESULTS:
        print(("PASS " if ok else "FAIL ") + name + ("  · " + detail if detail else ""))
    print("\n通过 %d / %d" % (total - len(bad), total))
    print("日志：%s" % SERVER_LOG)
    print("成片：%s" % OUT)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
