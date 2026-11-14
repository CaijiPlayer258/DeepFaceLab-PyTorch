# -*- coding: utf-8 -*-
r"""显存预算（_vram_plan）验收：三条路 × 三档伪造显存 + CPU 回退，全在 %TEMP%。

验的东西（对应用户这一轮的要求）：
  A  fused 视频导出三档：DFL_VRAM_FREE_MB=1200 → **明确拒绝**（HTTP 400，消息带
     「需要 X GB / 可用 Y GB / 建议」）；2600 → **降档**后成功出片；8000 → 正常
     （帧预算 30 / dfm 批 8）。三档都不许出现 CUDA OOM（扫后端日志 + job.error）。
     注：预留改小之后（max(256MB, free×5%)），3000 空闲已经能跑满 30 帧 / 批 8
     （峰值 2714 ≤ avail 2744），所以「降档」那一档用的空闲值从 3000 挪到 2600。
  B  CPU 回退：DFL_ALLOW_CPU_FALLBACK=1 + 小显存 → 照样出片，status.on_cpu=true，
     日志里必须有「当前在 CPU 上跑」。
  C  预览：小显存 → 400 + 数字（不再半死不活）；正常显存 → 200 + vram_plan。
  D  项目导出：worker 数按预算（8000 → 3 / 3000 → 1 / 1200 → 拒绝），status 回显
     vram_plan.procs；拒绝同样是 400 + 数字。
  E  不在预算里的东西没被动过：拒绝时**没有**产生任何成片/临时目录。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\vram_plan_accept.py
    （只跑某几段：--cases a,b,c,d）

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
BOX = TEMP / "dfl_vram_accept"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
PROJ = BOX / "proj"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "阿証証証証-416"
SETTINGS = {"mask_mode": 8, "color_transfer": "mt", "blend": "seamless",
            "mask_erode": 0, "mask_blur": 4}
BASE = ""
RESULTS = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OOM_PAT = re.compile(r"out of memory|CUDA out of memory|cudaErrorMemoryAllocation|"
                     r"Failed to allocate|OOM", re.I)


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
        raise SystemExit("ffmpeg 失败（%s）：%s" % (what, (r.stderr or "")[-400:]))
    return r


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0)}
    finally:
        cap.release()


# ============================================================
# 沙箱 / 后端
# ============================================================
def prepare_sandbox():
    if BOX.exists():
        for name in ("model", "_engines"):
            try:
                os.rmdir(str(WS / name))          # 目录联接用 rmdir 摘
            except OSError:
                pass
        shutil.rmtree(BOX, ignore_errors=True)
    for d in (WS, MEDIA, OUT, LOGDIR, PROJ):
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
    """起一个后端；extra_env 就是这一档「伪造的显存」。"""
    global BASE
    env = dict(os.environ)
    env["DFL_WORKSPACE"] = str(WS)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK"):
        env.pop(k, None)
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


def log_mark():
    try:
        return SERVER_LOG.stat().st_size
    except OSError:
        return 0


def log_since(mark):
    try:
        with open(str(SERVER_LOG), "rb") as f:
            f.seek(int(mark))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def wait_export(timeout=1800.0):
    t0 = time.time()
    seen = False
    last = {}
    same = 0
    while time.time() - t0 < timeout:
        _s, d = get("/api/merge/export/status", timeout=60)
        cur = d or {}
        terminal = cur.get("phase") in ("done", "cancelled", "failed") and not cur.get("running")
        if cur.get("running"):
            seen = True
        same = (same + 1) if (terminal and cur.get("phase") == (last or {}).get("phase")) else 0
        last = cur
        if terminal and ((seen and not cur.get("active")) or same >= 6):
            break
        time.sleep(0.3)
    return last


def has_numbers(msg):
    """拒绝消息里必须有「需要 X GB / 可用 Y GB」这两个数字。"""
    up = re.search(r"需要\s*([0-9]+\.?[0-9]*)\s*GB", str(msg or ""))
    av = re.search(r"可用\s*([0-9]+\.?[0-9]*)\s*GB", str(msg or ""))
    return bool(up), bool(av), (up.group(1) if up else None), (av.group(1) if av else None)


# ============================================================
# A · fused 视频导出三档
# ============================================================
def export_body(clip, out):
    return {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "output_dir": str(out), "frame_budget": 30,
            "start_frame": 1, "end_frame": 12}


def case_refuse(clip):
    head("A1 · DFL_VRAM_FREE_MB=1200 → 明确拒绝（HTTP 400 + 数字）")
    proc = start_server("拒绝", {"DFL_VRAM_FREE_MB": "1200"})
    try:
        mark = log_mark()
        out = OUT / "a_refuse.mp4"
        t0 = time.time()
        st, d = post("/api/merge/export", export_body(clip, out))
        ms = int((time.time() - t0) * 1000)
        say("HTTP %s · %d ms · %s" % (st, ms, json.dumps(d, ensure_ascii=False)[:400]))
        check(st == 400, "a1 拒绝是 HTTP 400（不是 500、不是 200 后崩）", "HTTP %s" % st)
        msg = str(d.get("error") or d.get("detail") or "")
        has_need, has_avail, need_gb, avail_gb = has_numbers(msg)
        check(has_need and has_avail, "a1 消息里有「需要 X GB / 可用 Y GB」",
              "需要=%s GB · 可用=%s GB" % (need_gb, avail_gb))
        check("建议" in msg, "a1 消息里有建议", msg[:120])
        check(str(d.get("ok")).lower() == "false", "a1 响应体 ok=false", "ok=%r" % d.get("ok"))
        blob = log_since(mark)
        check(bool(re.search("显存不够|显存预算", blob)), "a1 后端日志里也留了账",
              "%d 字节" % len(blob))
        check(not OOM_PAT.search(blob), "a1 没有 CUDA OOM（拒绝发生在加载模型之前）",
              (OOM_PAT.search(blob).group(0) if OOM_PAT.search(blob) else "clean"))
        check(not Path(str(out)).exists(), "a1 没有产生成片", str(out.name))
        # 拒绝之后后端还得是活的（不能把进程搞崩）
        st2, d2 = get("/api/merge/export/status")
        check(st2 == 200 and d2.get("phase") in ("idle", "done", "failed"),
              "a1 后端拒绝后仍然健康", "phase=%r" % d2.get("phase"))
        return {"need_gb": need_gb, "avail_gb": avail_gb, "msg": msg}
    finally:
        stop_server(proc)


def case_downgrade(clip, fake_free, want_tier):
    """fake_free → 期望「降档后成功」或「正常」。"""
    head("%s · DFL_VRAM_FREE_MB=%s → %s"
         % ("A2" if want_tier == "down" else "A3", fake_free,
            "降档后成功出片" if want_tier == "down" else "正常出片（30 帧 / 批 8）"))
    proc = start_server(want_tier, {"DFL_VRAM_FREE_MB": str(fake_free)})
    try:
        mark = log_mark()
        out = OUT / ("a_%s.mp4" % want_tier)
        st, d0 = post("/api/merge/export", export_body(clip, out))
        check(st == 200 and d0.get("ok") is not False,
              "%s 起单成功（预算允许才起）" % want_tier,
              "HTTP %s %s" % (st, (d0.get("error") or "")[:140]))
        if st != 200:
            return None
        plan0 = d0.get("vram_plan") or {}
        say("起单计划：%s" % json.dumps({k: plan0.get(k) for k in
                                        ("ok", "frames", "batch", "peak_mb", "avail_mb",
                                         "downgraded", "free_mb", "sess_mb")},
                                       ensure_ascii=False))
        out_p = Path(str(d0.get("output") or out))
        fin = wait_export(1800.0)
        v = fin.get("vid") or {}
        plan = fin.get("vram_plan") or plan0
        check(fin.get("phase") == "done" and fin.get("ok") is True,
              "%s 出片成功" % want_tier,
              "phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:140]))
        check(bool(plan), "%s status 里有 vram_plan" % want_tier,
              json.dumps({k: plan.get(k) for k in ("ok", "frames", "batch", "peak_mb",
                                                   "avail_mb", "downgraded")},
                         ensure_ascii=False))
        if want_tier == "down":
            check(bool(plan.get("downgraded")) is True, "a2 计划里标了「已降档」",
                  "downgraded=%r" % plan.get("downgraded"))
            check(int(plan.get("batch") or 99) < 8, "a2 dfm 批被压下来了",
                  "batch=%r（请求 8）" % plan.get("batch"))
            check(float(plan.get("peak_mb") or 1e9) <= float(plan.get("avail_mb") or 0) + 1,
                  "a2 预计峰值 ≤ 可用预算",
                  "peak=%.0f / avail=%.0f" % (float(plan.get("peak_mb") or 0),
                                              float(plan.get("avail_mb") or 0)))
        else:
            check(bool(plan.get("downgraded")) is False, "a3 没降档",
                  "downgraded=%r" % plan.get("downgraded"))
            check(int(plan.get("frames") or 0) == 30 and int(plan.get("batch") or 0) == 8,
                  "a3 计划是 30 帧 / 批 8",
                  "frames=%r batch=%r" % (plan.get("frames"), plan.get("batch")))
        info = probe(out_p) if out_p.is_file() else {}
        check(bool(info) and int(info.get("frames") or 0) == 12,
              "%s 出片 12 帧" % want_tier,
              "%s 帧" % (info.get("frames") if info else "-"))
        blob = log_since(mark)
        check(not OOM_PAT.search(blob), "%s 全程没有 CUDA OOM" % want_tier,
              (OOM_PAT.search(blob).group(0) if OOM_PAT.search(blob) else "clean"))
        log = ""
        lp = Path(str(out_p)).with_suffix(".log")
        if lp.is_file():
            log = lp.read_text("utf-8", "ignore")
        hits = [ln.strip() for ln in log.splitlines() if "显存预算" in ln or "计划 帧预算" in ln]
        check(bool(hits), "%s 成片日志里有预算账" % want_tier,
              (hits[0][:160] if hits else ""))
        return {"plan": plan, "fps": (v.get("stats") or {}).get("fps")}
    finally:
        stop_server(proc)


# ============================================================
# B · CPU 回退
# ============================================================
def case_cpu(clip):
    head("B · DFL_ALLOW_CPU_FALLBACK=1 + 小显存 → CPU 出片 + 明确提示")
    proc = start_server("cpu", {"DFL_VRAM_FREE_MB": "1200", "DFL_ALLOW_CPU_FALLBACK": "1"})
    try:
        mark = log_mark()
        out = OUT / "b_cpu.mp4"
        st, d0 = post("/api/merge/export", export_body(clip, out))
        check(st == 200 and d0.get("ok") is not False, "b 小显存 + 开关打开 → 起单",
              "HTTP %s %s" % (st, (d0.get("error") or "")[:140]))
        if st != 200:
            return None
        plan0 = d0.get("vram_plan") or {}
        check(plan0.get("cpu") is True, "b 起单响应里已标 CPU",
              "cpu=%r reason=%s" % (plan0.get("cpu"), plan0.get("reason")))
        check(bool(d0.get("on_cpu")) is True, "b 顶层 on_cpu=true", "on_cpu=%r" % d0.get("on_cpu"))
        out_p = Path(str(d0.get("output") or out))
        fin = wait_export(3600.0)
        plan = fin.get("vram_plan") or plan0
        check(fin.get("phase") == "done" and fin.get("ok") is True, "b CPU 上也真的出片了",
              "phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:140]))
        check(bool(fin.get("on_cpu")) is True, "b status.on_cpu=true", "on_cpu=%r" % fin.get("on_cpu"))
        check(plan.get("cpu") is True, "b vram_plan.cpu=true", "cpu=%r" % plan.get("cpu"))
        check("CPU" in str(plan.get("msg") or ""), "b 计划消息里写明 CPU", (plan.get("msg") or "")[:140])
        info = probe(out_p) if out_p.is_file() else {}
        check(bool(info) and int(info.get("frames") or 0) == 12, "b 出片 12 帧",
              "%s 帧" % (info.get("frames") if info else "-"))
        blob = log_since(mark)
        check("当前在 CPU 上跑" in blob, "b 后端日志里有「当前在 CPU 上跑」",
              ([ln for ln in blob.splitlines() if "CPU 上跑" in ln][:1] or [""])[0][:160])
        log = ""
        lp = Path(str(out_p)).with_suffix(".log")
        if lp.is_file():
            log = lp.read_text("utf-8", "ignore")
        check("CPU" in log, "b 成片日志里也写了 CPU",
              ([ln for ln in log.splitlines() if "CPU" in ln][:1] or [""])[0][:160])
        check(not OOM_PAT.search(blob), "b 没有 CUDA OOM",
              (OOM_PAT.search(blob).group(0) if OOM_PAT.search(blob) else "clean"))
        return {"plan": plan}
    finally:
        stop_server(proc)


# ============================================================
# C · 预览
# ============================================================
def case_preview(clip):
    head("C · 预览：小显存 → 400 + 数字；正常显存 → 200 + vram_plan")
    body = {"video": str(clip), "t": 0.5, "model": DFM, "settings": dict(SETTINGS)}
    proc = start_server("预览小", {"DFL_VRAM_FREE_MB": "1200"})
    try:
        st, d = post("/api/merge/preview", body, timeout=600)
        say("HTTP %s · %s" % (st, json.dumps(d, ensure_ascii=False)[:300]))
        check(st == 400, "c1 预览在装不下时明确拒绝（400）", "HTTP %s" % st)
        msg = str(d.get("error") or "")
        has_need, has_avail, need_gb, avail_gb = has_numbers(msg)
        check(has_need and has_avail, "c1 预览拒绝消息里有数字",
              "需要=%s GB · 可用=%s GB" % (need_gb, avail_gb))
        st2, d2 = get("/api/merge/preview/status")
        plan = (d2 or {}).get("vram_plan") or {}
        check(st2 == 200 and plan.get("ok") is False,
              "c1 预览 status 里能看到这次拒绝的计划",
              "ok=%r need=%r avail=%r" % (plan.get("ok"), plan.get("need_mb"),
                                          plan.get("avail_mb")))
    finally:
        stop_server(proc)
    proc = start_server("预览正常", {"DFL_VRAM_FREE_MB": "8000"})
    try:
        st, d = post("/api/merge/preview", body, timeout=900)
        check(st == 200 and d.get("ok") is True, "c2 正常显存下预览照常出图",
              "HTTP %s %s" % (st, (d.get("error") or "")[:120]))
        plan = (d or {}).get("vram_plan") or {}
        check(plan.get("ok") is True and int(plan.get("procs") or 0) == 1,
              "c2 预览回包带 vram_plan（1 份会话）",
              "ok=%r sess=%.0f MB avail=%s MB" % (plan.get("ok"),
                                                  float(plan.get("sess_mb") or 0),
                                                  plan.get("avail_mb")))
        st2, d2 = get("/api/merge/preview/status")
        check(bool((d2 or {}).get("vram_plan")), "c2 preview/status 也有 vram_plan")
    finally:
        stop_server(proc)


# ============================================================
# D · 项目导出 worker 数
# ============================================================
def make_fake_project(clip):
    """最小项目：1 帧 + 1 张 aligned（够 _scan_first 过闸门；对拍不在这里验）。"""
    proj = PROJ / "proj1"
    aligned = proj / "aligned"
    aligned.mkdir(parents=True, exist_ok=True)
    run_ff(["-i", str(clip), "-frames:v", "1", proj / "000000.jpg"], "项目帧")
    shutil.copyfile(str(proj / "000000.jpg"), str(aligned / "000000_0.jpg"))
    return proj


def case_project(clip):
    # 预期 worker 数**按同一份计划反推**（avail/sess/aux/batch 全从响应里读），
    # 不再写死「3000 → 1」—— 那是老预留（free×20%+512MB）下的数；新预留只有
    # max(256MB, free×5%)，3000 空闲时 2 份会话（2×1026+300+80=2432）本就装得下。
    head("D · 项目导出：worker 数按预算（8000 → 3 / 3000 → 2 / 1200 → 拒绝）")
    proj = make_fake_project(clip)
    got = {}
    for tag, free in (("8000", 8000), ("3000", 3000), ("1200", 1200)):
        proc = start_server("项目" + tag, {"DFL_VRAM_FREE_MB": str(free)})
        try:
            body = {"project_dir": str(proj), "video": str(clip), "model": DFM,
                    "settings": dict(SETTINGS), "out_video": str(OUT / ("d_%s.mp4" % tag)),
                    "workers": 3}
            st, d = post("/api/merge/export-project", body, timeout=600)
            plan = (d or {}).get("vram_plan") or {}
            say("free=%s → HTTP %s · procs=%r · workers=%r"
                % (free, st, plan.get("procs"), (d or {}).get("workers")))
            got[tag] = {"st": st, "plan": plan, "workers": (d or {}).get("workers"),
                        "msg": (d or {}).get("error") or plan.get("msg")}
            if free >= 3000:
                sess = float(plan.get("sess_mb") or 0.0)
                aux = float(plan.get("aux_mb") or 0.0)
                bmb = float(plan.get("batch_mb") or 0.0)
                avail = float(plan.get("avail_mb") or 0.0)
                want = max(1, min(3, int((avail - aux - bmb) // max(1.0, sess))))
                check(st == 200 and int((d or {}).get("workers") or 0) == want,
                      "d %s → %d 个 worker" % (free, want),
                      "workers=%r procs=%r" % ((d or {}).get("workers"), plan.get("procs")))
                check(bool(plan.get("sess_mb")), "d %s vram_plan 有会话体量" % free,
                      "sess=%.0f MB avail=%s" % (float(plan.get("sess_mb") or 0),
                                                 plan.get("avail_mb")))
                post("/api/merge/export-project/cancel", {})
                time.sleep(0.6)
            else:
                check(st == 400, "d 1200 → 明确拒绝（400）", "HTTP %s" % st)
                has_need, has_avail, need_gb, avail_gb = has_numbers(
                    str((d or {}).get("error") or ""))
                check(has_need and has_avail, "d 1200 拒绝消息里有数字",
                      "需要=%s GB · 可用=%s GB" % (need_gb, avail_gb))
        finally:
            stop_server(proc)
    return got


# ============================================================
# E · 汇总断言
# ============================================================
BASELINE = {}


def case_asserts():
    head("E · 汇总断言")
    blob = ""
    try:
        blob = SERVER_LOG.read_text("utf-8", "ignore")
    except OSError:
        pass
    hits = OOM_PAT.search(blob)
    check(hits is None, "e 整轮后端日志里没有 CUDA OOM",
          (hits.group(0) if hits else "clean（%d 字节日志）" % len(blob)))
    fails_500 = len(re.findall(r'"error":\s*"服务端内部错误', blob))
    check(fails_500 == 0, "e 没有「服务端内部错误」500", "%d 次" % fails_500)
    # 临时帧目录只比「这一轮新增了几个」：%TEMP% 里本来就躺着一堆历史残留
    # （早前的验收脚本被 taskkill 掉之后留下的），拿总量断言只会一直红。
    before = set(BASELINE.get("fused_tmp") or [])
    left = sorted(p.name for p in TEMP.glob("dfl_fused_*")
                  if p.is_dir() and p.name not in before)
    check(len(left) == 0, "e 这一轮没留下临时帧目录", "%d 个新增：%s" % (len(left), left[:4]))


# ============================================================
# main
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="显存预算（_vram_plan）验收")
    ap.add_argument("--cases", default="a,b,c,d,e", help="要跑的案例（a/b/c/d/e）")
    args = ap.parse_args(argv)
    want = [c.strip().lower() for c in str(args.cases).split(",") if c.strip()]
    t_all = time.time()

    prepare_sandbox()
    BASELINE["fused_tmp"] = sorted(p.name for p in TEMP.glob("dfl_fused_*") if p.is_dir())
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / "clip12.mp4"
    run_ff(["-ss", "3", "-t", "0.4", "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", clip],
           "切样片")
    say("样片 %s：%s" % (clip.name, probe(clip)))

    if "a" in want:
        case_refuse(clip)
        # 降档那一档从 3000 改成 2600：新预留只有 max(256MB, free×5%)，3000 空闲时
        # 30 帧 / 批 8（峰值 2714MB）本来就装得下（avail 2744MB），不该再降档；
        # 2600 → avail 2344MB → 批 8（2714）与批 4（2663）都被否决 → 批 2（1723）装得上。
        case_downgrade(clip, 2600, "down")
        case_downgrade(clip, 8000, "normal")
    if "b" in want:
        case_cpu(clip)
    if "c" in want:
        case_preview(clip)
    if "d" in want:
        case_project(clip)
    if "e" in want:
        case_asserts()

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs"
          % (len(RESULTS) - len(fails), len(fails), time.time() - t_all))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("后端日志：%s" % SERVER_LOG)
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
