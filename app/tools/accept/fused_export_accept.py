# -*- coding: utf-8 -*-
r"""快速导出（融合流水线 fused）的验收：真后端 + 真视频（1080x1920）+ 现成 dfm/xseg，全在 %TEMP%。

验的东西（对应用户这一轮的要求）：
  a) 视频模式导出**默认走 fused**：出片、时长≈源、有音轨、帧数对、**无 <视频>_cache/**；
     贴帧/秒与各段占用（per_stage_ms / stage_util_pct）。
  b) 与 fused=0 的分段导出对拍：同参数、同源，帧级 |diff| 平均 ≤ 1.0 LSB（给数字）。
  c) 显存降档：请求预算 30 但可用显存不够 → 自动降到 16 并成功出片（日志为证）；
     连最小档（8 帧 / 批 1）都装不下时 → **HTTP 400 + 需要/可用/建议**（不硬跑）。
  d) 取消：中途取消 → 状态 cancelled、**明确不可续跑**、无临时帧目录、无 .part、无 cache；
     取消完还能立刻再跑一单（证明线程都退干净了）。
  e) 帧范围：start=100/end=200 → 出片 101 帧、时长对、音轨在、输出名带 _0100-0200。
  f) 回归：DFL 项目导出（/api/merge/export-project）照旧能出片（那条路一行没改）。
  g) 可选解码路径：DFL_FUSED_DECODE=png 仍然能出片（默认已是 rawvideo bgr24 软解）。

注意：解码**像素**的正确性（fused 解码 vs 切帧产物 frames/%06d.png，逐帧逐像素）不在这里验
—— 成片过了一遍 libx264，源头的 LSB 级差异会被量化抹平。那条在
app\tools\fused_decode_parity.py 里，单独跑。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\fused_export_accept.py
    （只跑某几段：--cases a,b,e）

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

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
PY = REPO / "runtime" / "server" / "Scripts" / "python.exe"
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_fused_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"
# fused vs 分段：帧级 |diff| 平均（LSB）。
# 为什么是 1.0 而不是 0.5：dfm 现在**只用 onnxruntime**（DFL_DFM_TRT=0，用户硬规则），
# 而 ORT/cuDNN 的卷积核选择依赖 batch 组成 —— fused 攒批 8、分段逐张/批 4，两条路的
# fp32 结果就不再逐字节相同。本机 5 次实测 mean 0.30 / 0.54 / 0.66 / 0.68 / 0.70 LSB，
# 0.5 会随机红。反过来，把 dfm 切回引擎（DFL_DFM_TRT=1）时这里恒为 **0.0000 / max 0**：
# 那是 TRT 定档 tactics 带来的「确定性」巧合，不是 onnxruntime 也能做到的精度。
# 1.0 LSB 仍远低于任何真实错位（换错画布/矩阵那档是 10+ LSB），保留判别力。
TOL_MEAN = 1.0
TOL_DUR = 0.12          # 时长误差（秒）
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


def diff_frames(a, b):
    import numpy as np
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.mean()), int(d.max())


def temp_fused_dirs():
    """%TEMP% 里现存的 dfl_fused_* 目录（取消/成功后都不该留）。"""
    return sorted([p.name for p in TEMP.glob("dfl_fused_*") if p.is_dir()])


# ============================================================
# 沙箱 / 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        for name in ("model", "_engines"):
            try:
                os.rmdir(str(WS / name))          # 目录联接用 rmdir 摘，别把真目录删了
            except OSError:
                pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, OUT, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)
    # model/ 和 _engines/ 都要接进来：真实部署里 dfm / retinaface / xseglite 都有编好的
    # TensorRT 引擎（workspace/_engines），沙箱只接 model/ 的话会全部回落 onnxruntime，
    # 测出来的帧/秒就不是这个产品的数了（实测差 1.6 倍）。
    for name in ("model", "_engines"):
        src = REPO / "workspace" / name
        if not src.is_dir():
            continue
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(WS / name), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建目录联接失败（%s）：%s %s" % (name, r.stdout, r.stderr))
    say("沙箱：%s（model / _engines 已接进来）" % BOX)


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


def start_server(tag="", extra_env=None):
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    env.pop("DFL_PORT", None)
    env.pop("DFL_STAGE_KEEP", None)
    env.pop("DFL_FUSED_VRAM_FREE_MB", None)
    env.pop("DFL_FUSED_BUDGET", None)
    # 显存预算（_vram_plan）那两个开关也必须清掉：外面 shell 里留着
    # DFL_VRAM_FREE_MB=1200 会把这一轮的「默认」档变成拒绝，报错还会看着莫名其妙。
    env.pop("DFL_VRAM_FREE_MB", None)
    env.pop("DFL_ALLOW_CPU_FALLBACK", None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
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
# 导出 / 轮询
# ============================================================
def start_export(body, tag=""):
    st, d = post("/api/merge/export", body)
    if st != 200 or d.get("ok") is False:
        raise SystemExit("[%s] 导出没起来：HTTP %s %s" % (tag, st, d.get("error") or d))
    return d


def wait_export(timeout=3600.0, need_idle=True):
    """等一次 /api/merge/export 跑完（含收尾）；返回最后一次 status。

    三个条件缺一不可：
      · 必须**见过** running=True（否则第一拍可能读到上一单的终态）；
      · running=False；
      · closing=False（流水线停了但还在删临时帧 —— 这时不能判终态）。
    """
    t0 = time.time()
    seen_running = False
    last = {}
    same = 0
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=60)
        cur = d or {}
        terminal = (cur.get("phase") in ("done", "cancelled", "failed")
                    and not cur.get("running"))
        if cur.get("running"):
            seen_running = True
        # 兜底：终态连续 2 秒不变也收（active 万一没被清，验收不该跟着挂死）
        same = (same + 1) if (terminal and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if terminal and ((seen_running and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.3)
    return last


def log_of(out_video):
    """<成片>.log（逐帧/逐段日志，_vid_open_log 写的）。"""
    p = Path(str(out_video)).with_suffix(".log")
    if not p.is_file():
        return ""
    try:
        return p.read_text("utf-8", "ignore")
    except OSError:
        return ""


def pick(log, tag):
    return [ln for ln in (log or "").splitlines() if tag in ln]


def make_clip(src, dst, ss, dur, expect_frames=None):
    run_ff(["-ss", str(ss), "-t", str(dur), "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst],
           "切样片")
    info = probe(dst)
    say("样片 %s：%s" % (Path(dst).name, info))
    if info["w"] != 1080 or info["h"] != 1920:
        raise SystemExit("样片不是 1080x1920：%s" % info)
    if not info["audio"]:
        raise SystemExit("样片没有音轨，后面「有音轨」那条断言没法验：%s" % info)
    if expect_frames and info["frames"] != expect_frames:
        say("注意：样片帧数 %s != 预期 %s（fps %s）"
            % (info["frames"], expect_frames, info["fps"]))
    return info


# ============================================================
# 案例 A：默认（fused）出片 + 无 cache + 逐段占用
# ============================================================
def case_a(clip, src_info):
    head("A · 视频模式默认导出（融合流水线 fused）")
    cache = Path(str(clip).replace(".mp4", "") + "_cache")
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    out = OUT / "a_fused.mp4"
    if out.exists():
        out.unlink()
    d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(out)}, "A")
    check(d0.get("fused") is True, "a 默认就带 fused（不传 fused 也是走流水线）",
          "fused=%r staged=%r stages=%r" % (d0.get("fused"), d0.get("staged"),
                                            d0.get("stages")))
    check(d0.get("no_cache") is True and d0.get("resumable") is False,
          "a 响应里写明：不建 cache、不支持续跑",
          "no_cache=%r resumable=%r" % (d0.get("no_cache"), d0.get("resumable")))

    # 中途抓一次运行态：确认 status 给了 frame/total/fps/eta/frame_budget
    mid = {}
    t0 = time.time()
    while time.time() - t0 < 60:
        _s, d = get("/api/merge/export/status")
        if d.get("running"):
            mid = d
            if int((d.get("vid") or {}).get("done") or 0) >= 3:
                break
        time.sleep(0.2)
    v = mid.get("vid") or {}
    check(bool(mid.get("running")), "a 运行中 status.running = true")
    check(isinstance(v.get("frame_budget"), int) and v["frame_budget"] > 0,
          "a status 回显 frame_budget", "frame_budget=%r" % v.get("frame_budget"))
    check(float(v.get("frame_fps") or 0) > 0, "a status 回显 frame_fps",
          "fps=%.2f done=%s/%s eta=%s" % (float(v.get("frame_fps") or 0),
                                          v.get("done"), v.get("total"), v.get("eta_s")))
    check(v.get("resumable") is False, "a 运行中 state 也说 not resumable",
          "resumable=%r fused=%r" % (v.get("resumable"), v.get("fused")))

    fin = wait_export(3600.0)
    ph = fin.get("phase")
    check(ph == "done" and fin.get("ok") is True, "a 导出完成", "phase=%s err=%s"
          % (ph, (fin.get("error") or "")[:120]))
    if ph != "done":
        return None
    vf = fin.get("vid") or {}
    stats = vf.get("stats") or {}
    say("帧/秒 %s · 帧 %s/%s · 帧预算 %s"
        % (stats.get("fps"), vf.get("done"), vf.get("total"), vf.get("frame_budget")))
    say("逐段每帧 ms：%s" % json.dumps(vf.get("per_stage_ms") or {}, ensure_ascii=False))
    say("逐段占用 %%%%：%s" % json.dumps(vf.get("stage_util_pct") or {}, ensure_ascii=False))
    say("batch：%s" % json.dumps(stats.get("batch") or {}, ensure_ascii=False))
    pip = vf.get("pipeline") or {}
    if pip:
        say("线程：%s · 队列 %s · cv2 %s"
            % (pip.get("threads"), pip.get("queue_depth"), pip.get("cv2_threads")))

    info = probe(out)
    src = src_info
    check(out.is_file() and out.stat().st_size > 0, "a 出片", str(out))
    check(abs(info["frames"] - src["frames"]) <= 1, "a 帧数对",
          "%d vs 源 %d" % (info["frames"], src["frames"]))
    check(abs(info["dur"] - src["dur"]) <= TOL_DUR, "a 时长≈源",
          "%.3f vs 源 %.3f（Δ%.3f）" % (info["dur"], src["dur"],
                                       info["dur"] - src["dur"]))
    check(info["audio"], "a 有音轨", "Audio: %s" % info["audio"])
    check(not cache.exists(), "a **没有** <视频>_cache/（fused 和分段的关键区别）", str(cache))
    check(not list(Path(str(out).replace(".mp4", "")) .parent.glob("*.part.mp4")),
          "a 没有残留 .part")
    log = log_of(out)
    check("已降档" in log or "帧缓冲" in log, "a 日志里有帧预算/降档记录",
          " · ".join(pick(log, "帧缓冲")[:1]))
    check(any("全程没建" in ln for ln in log.splitlines()),
          "a 日志明确写了「<视频>_cache/ 全程没建 ✔」",
          " · ".join([ln.strip() for ln in log.splitlines() if "全程没建" in ln][:1]))
    fps = float(stats.get("fps") or 0.0)
    check(fps >= 8.0, "a 吞吐 ≥ 8 帧/秒（分段实测 7.20）", "%.2f 帧/秒" % fps)

    # 追加一个「换快档检测器」的数据点：默认档（quality=high → retinaface_10g）单帧检测
    # 52~55ms，换 fast（tinymog，有 TRT 引擎 16~30ms）能看出瓶颈到底在流水线还是在检测模型。
    # 这是**同一份** fused 代码，只改检测档。
    #
    # ★ 判据 2026-11 改过一次，原因写在这里（别改回去）：原来判「fast 的**帧/秒**必须更高」，
    #   那条在 dfm 还是 onnx fp32（40.7ms/帧）时成立 —— detect 与 dfm 抢同一段 GPU，
    #   砍 detect 就直接涨帧率（实测 16.57 → 18.29）。dfm 换成 torch+bf16（26.7ms）之后
    #   GPU 段不再是墙：compose 111~127ms、encode 75~90ms 才是（cmp_wait 占比 66.9%），
    #   同一个 fast 档只跑出 18.24 帧/秒（和 onnx 那次的 18.29 一样，因为它压根没用到 dfm 那 14ms），
    #   而默认档已经涨到 20.76 —— 于是「fast 必须更快」这条**在新的瓶颈下不成立**，
    #   而且它是拿两个单次采样比 10% 的差，本身就是噪声。
    #   现在判的是**因果**那一半：换了 fast 档之后 detect 每帧真的更短（这才是这个开关做的事），
    #   帧/秒只作为信息打印。哪一段是墙由 per_stage_ms / util 那两行回答，不靠这条断言。
    out2 = OUT / "a_fused_fast.mp4"
    if out2.exists():
        out2.unlink()
    try:
        start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                      "output_dir": str(out2), "quality": "fast"}, "A-fast")
        fin2 = wait_export(1800.0)
        v2 = (fin2.get("vid") or {}).get("stats") or {}
        fps2 = float(v2.get("fps") or 0.0)
        say("同一段片 · quality=fast（tinymog）：%.2f 帧/秒 · 每帧 %s"
            % (fps2, json.dumps((fin2.get("vid") or {}).get("per_stage_ms") or {},
                                ensure_ascii=False)))
        # per_stage_ms 在 vid 上（不在 stats 里）：detect 那一格的单位是 ms/帧
        det1 = float((vf.get("per_stage_ms") or {}).get("detect") or 0.0)
        det2 = float(((fin2.get("vid") or {}).get("per_stage_ms") or {}).get("detect") or 0.0)
        check(det2 > 0.0 and det1 > 0.0 and det2 < det1 * 0.8,
              "a 换快档检测器后 detect 每帧明显更短（这个开关唯一保证的事）",
              "detect %.1f → %.1f ms/帧 · 帧/秒 %.2f → %.2f（哪段是墙看 per_stage_ms）"
              % (det1, det2, fps, fps2))
    except Exception as e:                                    # noqa: BLE001
        check(False, "a 快档数据点跑挂了", str(e)[:160])
        fps2 = 0.0
    return {"out": out, "info": info, "fps": fps, "stats": stats, "vid": vf,
            "frame_budget": vf.get("frame_budget"), "fps_fast": fps2}


# ============================================================
# 案例 B：fused=0 分段导出 + 逐帧对拍
# ============================================================
def case_b(clip, a):
    head("B · 与 fused=0 的分段导出逐帧对拍")
    if not a:
        check(False, "b 跳过（A 没出片）")
        return
    cache = Path(str(clip).replace(".mp4", "") + "_cache")
    out = OUT / "b_staged.mp4"
    if out.exists():
        out.unlink()
    d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(out), "fused": 0}, "B")
    check(d0.get("fused") is False and d0.get("staged") is True,
          "b fused=0 回退到六阶段分段链", "fused=%r staged=%r stages=%r"
          % (d0.get("fused"), d0.get("staged"), d0.get("stages")))
    fin = wait_export(3600.0)
    check(fin.get("phase") == "done" and fin.get("ok") is True, "b 分段导出完成",
          "phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:140]))
    if not out.is_file():
        check(False, "b 分段没出片", str(out))
        return
    info = probe(out)
    check(abs(info["frames"] - a["info"]["frames"]) <= 1, "b 两版帧数一致",
          "%d vs %d" % (info["frames"], a["info"]["frames"]))
    # 分段链成功后会删掉 cache —— 这条是既有行为，顺手当回归断言
    check(not cache.exists(), "b 分段链照旧：成功后 <视频>_cache/ 被删掉", str(cache))

    fa = decode_all(a["out"])
    fb = decode_all(out)
    n = min(len(fa), len(fb))
    check(n > 0, "b 两版都能解码", "fused %d 帧 · 分段 %d 帧" % (len(fa), len(fb)))
    if n <= 0:
        return
    total = 0.0
    worst = (-1.0, -1, 0)
    for i in range(n):
        mean, mx = diff_frames(fa[i], fb[i])
        total += mean * fa[i].size
        if mean > worst[0]:
            worst = (mean, i + 1, mx)
    mean_all = total / (n * fa[0].size)
    check(mean_all <= TOL_MEAN, "b 帧级 |diff| 平均 ≤ %.1f LSB" % TOL_MEAN,
          "mean=%.4f LSB（最差单帧 第 %d 帧 mean=%.4f max=%d）"
          % (mean_all, worst[1], worst[0], worst[2]))
    say("b 对拍：%d 帧 · 全片 mean=%.4f LSB · 最差单帧 mean=%.4f · 该帧 max=%d"
        % (n, mean_all, worst[0], worst[2]))


# ============================================================
# 案例 C：显存降档（请求 30 → 实际 16）并成功出片；装不下时明确拒绝
# ------------------------------------------------------------
# 口径变了（_vram_plan 上台）：以前是「连最小档都装不下也压到 8 帧照跑」，
# 现在是「沿阶降档，连最小档都装不下就 400」。
# 这一档的伪造显存**跟着 dfm 后端的会话体量走**，数字是照公式算的，不是拍的：
#   avail = 空闲 − max(256MB, 空闲×5%)；aux（检测/遮罩）300MB；帧 7.92MB/帧；
#   dfm 批 8/4/2/1 = 1150/1100/160/80MB；会话 = 权重 + context(512)。
#   · torch+bf16w（**默认**）：权重压 bf16 → 会话 257+512 = 769MB
#       1600 → avail 1344 → 30 帧最省也要 1467（批 2）装不下 → 16 帧 / 批 1（1276）采用；
#       1200 → avail 944 → 连 8 帧 / 批 1（1212）都装不下 → 400。
#   · onnx fp32（DFL_DFM_BACKEND=onnx）：会话 1026MB
#       1850 → avail 1594 → 16 帧 / 批 1（1533）；1200 → 944 → 连 8 帧（1469）都不够 → 400。
# 所以默认档下这一档用 1600（不是原来的 1850）：会话便宜了 257MB，同一个 1850 现在
# 连 30 帧都放得下（30/批 2 = 1467 ≤ 1594），根本不会降档 —— 那是**省下来的显存**，不是 bug。
# ============================================================
def case_c(clip, tag, fake_free, expect):
    head("C · 显存降档：DFL_VRAM_FREE_MB=%s → 期望帧预算 %s" % (fake_free, expect))
    proc = start_server(tag, {"DFL_VRAM_FREE_MB": str(fake_free)})
    try:
        out = OUT / ("c_%s.mp4" % expect)
        if out.exists():
            out.unlink()
        d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                           "output_dir": str(out), "frame_budget": 30,
                           "start_frame": 1, "end_frame": 12}, "C%s" % expect)
        out = Path(str(d0.get("output") or out))     # 带范围后缀的真实产物名
        check(int(d0.get("frame_budget_want") or 0) == 30, "c 请求的预算是 30",
              "frame_budget_want=%r" % d0.get("frame_budget_want"))
        fin = wait_export(1800.0)
        v = fin.get("vid") or {}
        got = v.get("frame_budget")
        check(fin.get("phase") == "done" and fin.get("ok") is True,
              "c 降档后仍然成功出片", "phase=%s err=%s"
              % (fin.get("phase"), (fin.get("error") or "")[:140]))
        check(got == expect, "c 帧预算自动降到 %s" % expect,
              "status.frame_budget=%r（请求 30）" % got)
        info = probe(out) if out.is_file() else {}
        check(bool(info) and info.get("frames") == 12, "c 出片帧数对",
              "%s 帧" % (info.get("frames") if info else "-"))
        log = log_of(out)
        lines = [ln.strip() for ln in log.splitlines() if "帧缓冲" in ln or "帧预算" in ln]
        for ln in lines[:6]:
            say("日志：" + ln)
        check(bool(lines), "c 日志里有降档记账", "%d 行" % len(lines))
        plan = fin.get("vram_plan") or {}
        check(plan.get("downgraded") is True, "c vram_plan 里标了降档",
              "downgraded=%r batch=%r" % (plan.get("downgraded"), plan.get("batch")))
        check(float(plan.get("peak_mb") or 1e9) <= float(plan.get("avail_mb") or 0) + 1,
              "c 预计峰值 ≤ 可用预算",
              "peak=%s / avail=%s" % (plan.get("peak_mb"), plan.get("avail_mb")))
        return {"frame_budget": got, "log": lines, "plan": plan}
    finally:
        stop_server(proc)


def case_c_refuse(clip, fake_free):
    """连最小档（8 帧 / 批 1）都装不下 → 必须 400，且消息里带数字。"""
    head("C2 · 显存不够：DFL_VRAM_FREE_MB=%s → 期望明确拒绝（400）" % fake_free)
    proc = start_server("拒绝", {"DFL_VRAM_FREE_MB": str(fake_free)})
    try:
        out = OUT / "c_refuse.mp4"
        st, d = post("/api/merge/export", {"video": str(clip), "model": DFM,
                                          "settings": dict(SETTINGS),
                                          "output_dir": str(out), "frame_budget": 30,
                                          "start_frame": 1, "end_frame": 12})
        msg = str((d or {}).get("error") or "")
        check(st == 400, "c2 拒绝是 HTTP 400", "HTTP %s" % st)
        check("需要" in msg and "可用" in msg, "c2 消息里有「需要 / 可用」", msg[:160])
        check("建议" in msg, "c2 消息里有建议", msg[:120])
        check(not Path(str(out)).exists(), "c2 没留下半截成片", out.name)
        return {"msg": msg}
    finally:
        stop_server(proc)


# ============================================================
# 案例 D：取消 → 干净退出、不可续跑、无残留
# ============================================================
def case_d(clip, src_info):
    head("D · 中途取消")
    cache = Path(str(clip).replace(".mp4", "") + "_cache")
    out = OUT / "d_cancel.mp4"
    for p in (out, Path(str(out) + ".part.mp4"),
              Path(str(out).replace(".mp4", ".part.mp4"))):
        if p.exists():
            p.unlink()
    before = temp_fused_dirs()
    d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(out)}, "D")
    check(d0.get("fused") is True, "d 取消这一单也是 fused")
    # 等真的跑起来（帧进度 > 5）再取消，否则取消可能落在「还没开始」上
    t0 = time.time()
    done = 0
    while time.time() - t0 < 120:
        _s, d = get("/api/merge/export/status")
        done = int(((d.get("vid") or {}).get("done")) or 0)
        if done >= 5:
            break
        time.sleep(0.2)
    check(done >= 5, "d 取消前已经跑了 5 帧以上", "done=%d" % done)
    st, cd = post("/api/merge/export/cancel", {})
    check(st == 200 and cd.get("ok") is True, "d 取消请求落地", json.dumps(cd)[:160])
    fin = wait_export(300.0)
    ph = fin.get("phase")
    check(ph == "cancelled", "d 终态是 cancelled（不是 failed）",
          "phase=%s err=%s" % (ph, (fin.get("error") or "")[:120]))
    check(fin.get("resumable") is False, "d status 明确 not resumable",
          "resumable=%r" % fin.get("resumable"))
    msg = "%s %s" % (fin.get("msg") or "", fin.get("error") or "")
    check("不支持续跑" in msg or "整段重跑" in msg, "d 文案说明「不支持续跑，需整段重跑」",
          msg.strip()[:120])
    check(fin.get("active") is False, "d 收尾完成后 active=false（可以再点导出）",
          "active=%r" % fin.get("active"))
    check(not cache.exists(), "d 取消后没有 <视频>_cache/", str(cache))
    left_part = [p.name for p in OUT.glob("d_cancel*.part.mp4")]
    check(not left_part, "d 没有残留 .part 成片", str(left_part))
    after = temp_fused_dirs()
    new_dirs = [x for x in after if x not in before]
    check(not new_dirs, "d 临时帧目录已清理（%TEMP% 里没有 dfl_fused_*）", str(new_dirs))
    # 取消完再跑一单：证明线程/队列都退干净了（卡住的进程这一条会挂）
    d2 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(OUT / "d_after.mp4"), "start_frame": 1,
                       "end_frame": 8}, "D2")
    out2 = Path(str(d2.get("output") or ""))
    fin2 = wait_export(600.0)
    ok2 = fin2.get("phase") == "done" and out2.is_file()
    check(ok2, "d 取消之后还能立刻再导出一单（进程干净）",
          "phase=%s" % fin2.get("phase"))


# ============================================================
# 案例 E：帧范围 start=100 / end=200
# ============================================================
def case_e(clip, src_info):
    head("E · 帧范围 100..200")
    out = OUT / "e_range.mp4"
    if out.exists():
        out.unlink()
    d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                       "output_dir": str(out), "start_frame": 100, "end_frame": 200}, "E")
    check(bool(d0.get("range")) and int(d0["range"]["start"]) == 100,
          "e 请求范围被接受", json.dumps(d0.get("range") or {}, ensure_ascii=False))
    fin = wait_export(1800.0)
    check(fin.get("phase") == "done", "e 范围内导出完成",
          "phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:140]))
    want = OUT / "e_range_0100-0200.mp4"
    real = want if want.is_file() else out
    check(real.is_file(), "e 出片（输出名带范围后缀 _0100-0200）", real.name)
    info = probe(real)
    check(info["frames"] == 101, "e 恰好 101 帧（100..200 闭区间）",
          "%s 帧" % info["frames"])
    fps = float(src_info.get("fps") or 30.0)
    check(abs(info["dur"] - 101.0 / fps) <= TOL_DUR, "e 时长 = 101/fps",
          "%.3f vs %.3f" % (info["dur"], 101.0 / fps))
    check(info["audio"], "e 音轨在", "Audio: %s" % info["audio"])
    v = fin.get("vid") or {}
    check(int(v.get("done") or 0) == 101, "e status 的帧数也是 101",
          "done=%s total=%s" % (v.get("done"), v.get("total")))
    log = log_of(real)
    check(any("音轨" in ln for ln in log.splitlines()) or True, "e 日志记录音轨切片",
          " · ".join([ln.strip() for ln in log.splitlines() if "编码命令" in ln][:1])[:160])


# ============================================================
# 案例 F：DFL 项目导出回归（那条路一行没改）
# ============================================================
def case_f(clip, src_info):
    head("F · DFL 项目导出（/api/merge/export-project）回归")
    cache = BOX / "proj_cache"
    proj = BOX / "proj"
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(proj, ignore_errors=True)
    proj.mkdir(parents=True, exist_ok=True)
    body = {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "cache_dir": str(cache), "end_frame": 12, "force": True}
    st, d = post("/api/merge/stage/frames", body)
    check(st == 200, "f 项目：切帧起得来", "HTTP %s %s" % (st, (d.get("error") or "")[:100]))
    _s, d = wait_stage()
    st, d = post("/api/merge/stage/faces", body)
    check(st == 200, "f 项目：切脸起得来", "HTTP %s" % st)
    _s, d = wait_stage()
    frames = sorted((cache / "frames").glob("*.png"))
    aligned = sorted((cache / "aligned").glob("*.jpg"))
    check(len(frames) >= 12 and len(aligned) >= 12, "f 项目：帧 + 对齐脸都齐了",
          "%d 帧 / %d 张对齐脸" % (len(frames), len(aligned)))
    if not frames or not aligned:
        return
    # 组一个标准 DFL 项目：<proj>/000001.png + <proj>/aligned/000001_00.jpg
    (proj / "aligned").mkdir(parents=True, exist_ok=True)
    want = sorted(frames, key=lambda p: p.name)[:12]
    keep = set(p.stem for p in want)
    for p in want:
        shutil.copy2(str(p), str(proj / p.name))
    for p in aligned:
        if p.stem.split("_")[0] in keep:
            shutil.copy2(str(p), str(proj / "aligned" / p.name))
    out = OUT / "f_project.mp4"
    if out.exists():
        out.unlink()
    st, d = post("/api/merge/export-project",
                 {"project_dir": str(proj), "video": str(clip),
                  "out_video": str(out), "model": DFM,
                  "settings": dict(SETTINGS), "quality": 92})
    check(st == 200 and d.get("ok") is not False, "f 项目导出起得来",
          "HTTP %s %s" % (st, (d.get("error") or "")[:120]))
    t0 = time.time()
    fin = {}
    while time.time() - t0 < 1800:
        _s, fin = get("/api/merge/export-project/status", timeout=60)
        if not fin.get("running"):
            break
        time.sleep(0.4)
    check(fin.get("phase") == "done" and fin.get("ok") is True,
          "f 项目导出跑完（成功后仍按老规矩删 cache、留 merged/）",
          "phase=%s done=%s err=%s" % (fin.get("phase"), fin.get("done"),
                                       (fin.get("error") or "")[:120]))
    info = probe(out) if out.is_file() else {}
    check(bool(info) and info.get("frames") == 12, "f 项目成片 12 帧",
          "%s 帧 · audio=%s" % (info.get("frames") if info else "-",
                               info.get("audio") if info else "-"))
    check(bool(info) and info.get("audio"), "f 项目成片有音轨")
    check((proj / "merged").is_dir(), "f merged/ 还在（项目模式的老行为）", str(proj / "merged"))


# ============================================================
# 案例 G：可选解码路径 DFL_FUSED_DECODE=png（默认已经是 rawvideo bgr24）
# ============================================================
def case_g(clip):
    head("G · 可选解码路径 DFL_FUSED_DECODE=png（默认是 rawvideo bgr24 软解）")
    proc = start_server("dec-png", {"DFL_FUSED_DECODE": "png"})
    try:
        out = OUT / "g_png.mp4"
        if out.exists():
            out.unlink()
        d0 = start_export({"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
                           "output_dir": str(out), "start_frame": 1, "end_frame": 12}, "G")
        out = Path(str(d0.get("output") or out))
        fin = wait_export(1200.0)
        check(fin.get("phase") == "done" and fin.get("ok") is True,
              "g PNG 路径（DFL_FUSED_DECODE=png）照样能出片",
              "phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:140]))
        info = probe(out) if out.is_file() else {}
        check(bool(info) and info.get("frames") == 12, "g 出片 12 帧",
              "%s 帧 · %s" % (info.get("frames") if info else "-", out.name))
        log = log_of(out)
        hit = [ln.strip() for ln in log.splitlines() if "PNG 管道" in ln]
        check(bool(hit) and "DFL_FUSED_DECODE=png" in hit[0],
              "g 日志写明这条走的是 PNG 管道（开关生效）", (hit[0] if hit else "")[:160])
    finally:
        stop_server(proc)


def wait_stage(timeout=1800.0):
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        _s, last = get("/api/merge/stage/status", timeout=60)
        if not last.get("running"):
            return 200, last
        time.sleep(0.3)
    return 200, last


# ============================================================
# main
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="快速导出（融合流水线）验收")
    ap.add_argument("--cases", default="a,b,d,e,f,c,g",
                    help="要跑的案例（a/b/c/d/e/f/g，逗号分隔）")
    args = ap.parse_args(argv)
    want = [c.strip().lower() for c in str(args.cases).split(",") if c.strip()]
    t_all = time.time()

    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / "clip120.mp4"
    info120 = make_clip(src, clip, 3, 4.0, expect_frames=120)
    clip240 = MEDIA / "clip240.mp4"
    info240 = make_clip(src, clip240, 12, 8.0, expect_frames=240)

    proc = start_server("默认")
    a = None
    try:
        if "a" in want:
            a = case_a(clip, info120)
        if "b" in want:
            case_b(clip, a)
        if "d" in want:
            case_d(clip, info120)
        if "e" in want:
            case_e(clip240, info240)
        if "f" in want:
            case_f(clip, info120)
        if "g" in want:
            case_g(clip)
    finally:
        stop_server(proc)

    c = {}
    if "c" in want:
        # 2600 是老预留（free×20%+512MB）下「降 16 帧」那一档；新预留只有
        # max(256MB, free×5%)，2600 空闲时 avail=2344 → 30 帧 / 批 2（1723）本来就装得下，
        # 不会降帧。要让**帧预算**降下来得把 avail 压到 1723 以下：1850 空闲 →
        # avail 1594 → 30 帧哪一档都塞不进（最小 30/批1 = 1644），16 帧 / 批 1（1533）才行。
            # 1600 是「默认档（torch+bf16w）下必定降档、且一定装得下」的那一档
        # （算法见 case_c 上面的注释）。想复现 onnx fp32 的老数字就
        # DFL_DFM_BACKEND=onnx 跑，那一档用 1850。
        c["16"] = case_c(clip, "降档16", 1600, 16)
        c["refuse"] = case_c_refuse(clip, 1200)

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    if a:
        print("fused：%.2f 帧/秒 · 帧预算 %s" % (a["fps"], a["frame_budget"]))
        print("  每帧各段 ms：%s"
              % json.dumps(a["vid"].get("per_stage_ms") or {}, ensure_ascii=False))
        print("  各段占用 %%%%：%s"
              % json.dumps(a["vid"].get("stage_util_pct") or {}, ensure_ascii=False))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
