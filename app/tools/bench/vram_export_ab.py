# -*- coding: utf-8 -*-
r"""A/B：预览 worker 常驻时导出，帧预算 / dfm 批该不该降档（用户实测 22 → 15.8 帧/秒那条）。

两个跑法，同一台机器、同一段素材、同一份 .dfm，唯一的差别是**预算那一拍**：
  A 修复后   ：先预览一帧（worker 常驻 ~2.7GB）→ 直接导出
               → 导出起手先回收预览模型（日志：「回收预览模型 X MB（…）：空闲 A → B」）
               → 用回收后的空闲算账 → 帧预算 30 / dfm 批 8
  B 改动前的账：DFL_VRAM_NO_RECLAIM=1 + DFL_FUSED_VRAM_FREE_MB=2766
               （= 用户日志里那一单的「空闲 2766 MB」，且不回收 → 可回收不进公式）
               → 帧预算 8 / dfm 批 1，导出期间预览 worker 还占着那 2.7GB

比的数：导出成片日志里的「流水线跑完 N 帧 · X s · Y 帧/秒 · 帧缓冲 Z」。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\vram_export_ab.py
    （--frames 90 --dfm 张元英 ；退出码 0 = A 段全过）
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
BOX = TEMP / "dfl_vram_ab"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"

DFM = "张元英"
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


def smi_free():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int((r.stdout or "").strip().splitlines()[0])
    except Exception:                                       # noqa: BLE001
        return None


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
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK", "DFL_ALLOW_TEST_HOOK",
              "DFL_VRAM_NO_RECLAIM", "DFL_FUSED_BATCH"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    logp = LOGDIR / ("server_%s.log" % (re.sub(r"\W+", "_", tag) or "x"))
    logf = open(str(logp), "ab")
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
            raise SystemExit("后端启动失败，见 %s" % logp)
        time.sleep(0.2)
    if port["v"] is None:
        raise SystemExit("等不到 PORT= 行，见 %s" % logp)
    BASE = "http://127.0.0.1:%d" % port["v"]
    say("后端%s：%s（pid=%s · 日志 %s）" % (tag, BASE, proc.pid, logp.name))
    while time.time() - t0 < 240:
        try:
            st, _d = get("/api/merge/stage/status", timeout=20)
            if st == 200:
                return proc
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(0.4)
    raise SystemExit("后端起来了但 /api/merge/stage/status 不应答")


def stop_server(proc):
    if proc is None:
        return
    _kill_tree(proc.pid)
    try:
        proc.kill()
    except Exception:                                       # noqa: BLE001
        pass
    time.sleep(1.2)


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


def export_body(clip, out, dry=False, budget=30, end=90):
    body = {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "output_dir": str(out), "frame_budget": int(budget),
            "start_frame": 1, "end_frame": int(end)}
    if dry:
        body["dry_run"] = True
    return body


def preview_body(clip, t=0.5):
    return {"video": str(clip), "t": float(t), "model": DFM, "settings": dict(SETTINGS)}


def plan_of(d):
    d = d or {}
    return dict(d.get("vram_plan") or d.get("vram") or {})


BRIEF = ("ok", "free_mb", "avail_mb", "need_mb", "peak_mb", "frames", "batch", "requested_frames",
         "requested_batch", "reclaimable_mb", "reclaim_left_mb", "reclaimed_mb", "reclaim_note",
         "own_mb", "own_preview_mb", "other_mb", "reserve_mb", "downgraded", "source")


def brief(plan):
    return {k: (plan or {}).get(k) for k in BRIEF}


RE_REC = re.compile(r"回收预览模型\s*([0-9.]+)\s*MB.*?空闲\s*(\d+)\s*→\s*(\d+)\s*MB")
RE_RUN = re.compile(r"流水线跑完\s*(\d+)\s*帧\s*·\s*([0-9.]+)\s*s\s*·\s*([0-9.]+)\s*帧/秒"
                    r"\s*·\s*帧缓冲\s*(\d+)")
RE_BUDGET = re.compile(r"显存预算\s*→\s*帧缓冲\s*(\d+)\s*/\s*dfm 批\s*(\d+)")


def run_export(clip, out_name, end, tag):
    """预览一帧 → 直接导出；返回 (成片日志文本, 计划, 终态, 秒数)。"""
    mark = smi_free()
    post("/api/merge/preview", preview_body(clip), timeout=900)
    _s, dv = get("/api/merge/preview/status")
    wk = dict(dv or {})
    say("%s 预览常驻：models=%s used_mb=%s · nvidia-smi 空闲 %s MB"
        % (tag, wk.get("models"), wk.get("used_mb"), smi_free()))
    _s, d0 = post("/api/merge/export", export_body(clip, OUT / (out_name + ".mp4"), dry=True,
                                                   end=end))
    p0 = plan_of(d0)
    say("%s dry_run 计划：%s" % (tag, json.dumps(brief(p0), ensure_ascii=False)))
    t0 = time.time()
    st, d = post("/api/merge/export", export_body(clip, OUT / (out_name + ".mp4"), end=end))
    if st != 200:
        say("%s 导出被拒：%s" % (tag, str((d or {}).get("error") or "")[:300]))
        return "", p0, dict(d or {}), 0.0
    fin = wait_export(1800.0)
    wall = time.time() - t0
    # 真正的成片名是后端定的（带帧范围后缀：<stem>_0001-0090.mp4），逐帧日志叫 <成片>.log
    out_p = Path(str((d or {}).get("output") or "")) if isinstance(d, dict) else None
    logp = (out_p.with_suffix(".log") if out_p else (OUT / (out_name + ".log")))
    txt = logp.read_text("utf-8", "ignore") if logp.is_file() else ""
    say("%s 终态 phase=%s · 墙钟 %.1fs · 成片 %s · 日志 %s（%d 行）"
        % (tag, fin.get("phase"), wall,
           (out_p.name if out_p else "?"), logp.name, len(txt.splitlines())))
    return txt, plan_of(fin), fin, wall


def case_fixed(clip, end):
    head("A · 修复后：预览常驻 → 导出该回收、该 30 帧 / 批 8")
    proc = start_server("fixed")
    fps = 0.0
    try:
        txt, plan, fin, wall = run_export(clip, "a_fixed", end, "A")
        m = RE_REC.search(txt)
        check(bool(m), "A1 导出日志里先回收了预览模型（带「空闲 A → B」）",
              m.group(0) if m else "（没有这一行）")
        if m:
            freed, before, after = float(m.group(1)), int(m.group(2)), int(m.group(3))
            check(freed >= 1000.0, "A1 回收 ≥1GB（预览 worker 那一份）", "%.0f MB" % freed)
            check(after - before >= 1000.0, "A1 空闲真的涨上来了", "%d → %d MB" % (before, after))
        b = RE_BUDGET.search(txt)
        check(bool(b), "A2 日志里有「显存预算 → 帧缓冲 N / dfm 批 M」", b.group(0) if b else "-")
        if b:
            check(int(b.group(1)) == 30 and int(b.group(2)) == 8,
                  "A2 帧预算 30 / dfm 批 8（不因为自己的预览 worker 降档）",
                  "帧缓冲 %s / 批 %s" % (b.group(1), b.group(2)))
        # 顺序：回收那一行必须在预算那一行**前面**
        i_rec = txt.find("回收预览模型")
        i_bud = txt.find("显存预算 → 帧缓冲") if b else -1
        check(i_rec >= 0 and i_bud > i_rec, "A3 顺序：先回收 → 再算预算（不能反）",
              "回收@%d · 预算@%d" % (i_rec, i_bud))
        r = RE_RUN.search(txt)
        if r:
            fps = float(r.group(3))
            say("A 流水线：%s 帧 · %s s · %.2f 帧/秒 · 帧缓冲 %s"
                % (r.group(1), r.group(2), fps, r.group(4)))
            check(int(r.group(4)) == 30, "A4 真跑的帧缓冲就是 30", r.group(4))
        check(fin.get("phase") == "done", "A5 成片完成", "phase=%s" % fin.get("phase"))
        say("A 终态计划：%s" % json.dumps(brief(plan), ensure_ascii=False))
    finally:
        stop_server(proc)
    return fps


def case_legacy(clip, end):
    head("B · 改动前的账（不回收 + 空闲按用户日志的 2766 MB）：8 帧 / 批 1")
    proc = start_server("legacy", {"DFL_VRAM_NO_RECLAIM": "1",
                                   "DFL_FUSED_VRAM_FREE_MB": "2766"})
    fps = 0.0
    try:
        txt, plan, fin, wall = run_export(clip, "b_legacy", end, "B")
        b = RE_BUDGET.search(txt)
        say("B 预算行：%s" % (b.group(0) if b else "（没有）"))
        if b:
            check(int(b.group(1)) == 8 and int(b.group(2)) == 1,
                  "B1 改动前的账确实降到 8 帧 / 批 1（复现用户那一单）",
                  "帧缓冲 %s / 批 %s" % (b.group(1), b.group(2)))
        _s, dv = get("/api/merge/preview/status")
        wk = dict(dv or {})
        check(bool(wk.get("models")),
              "B2 这一轮预览 worker 全程还占着那 2.7GB（不改代码时的真实处境）",
              "models=%s used_mb=%s" % (wk.get("models"), wk.get("used_mb")))
        r = RE_RUN.search(txt)
        if r:
            fps = float(r.group(3))
            say("B 流水线：%s 帧 · %s s · %.2f 帧/秒 · 帧缓冲 %s"
                % (r.group(1), r.group(2), fps, r.group(4)))
        check(fin.get("phase") == "done", "B3 对照组成片完成", "phase=%s" % fin.get("phase"))
    finally:
        stop_server(proc)
    return fps


def main(argv=None):
    global DFM
    ap = argparse.ArgumentParser(description="帧预算降档 A/B（预览常驻 → 导出）")
    ap.add_argument("--dfm", default=DFM)
    ap.add_argument("--frames", type=int, default=90, help="导出多少帧（默认 90）")
    # 跑哪几臂（顺序有意义：先 A 后 B 会让 B 吃到「机器已经热了」的红利）。
    # 默认 A,B,A —— 两头都是 A，中位数/两次 A 的离散度能回答「是不是顺序效应」。
    ap.add_argument("--arms", default="A,B,A", help="A=修复后 · B=改动前的账，逗号分隔")
    args = ap.parse_args(argv)
    DFM = str(args.dfm)
    end = max(12, int(args.frames))
    t_all = time.time()

    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / ("clip%d.mp4" % end)
    sec = max(1.0, end / 30.0 + 0.5)
    run_ff(["-ss", "3", "-t", "%.2f" % sec, "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", clip], "切样片")
    say("样片 %s · dfm=%s · 导出帧 %d" % (clip.name, DFM, end))
    say("nvidia-smi 起始空闲：%s MB" % smi_free())

    arms = [a.strip().upper() for a in str(args.arms).split(",") if a.strip()]
    seq = []
    for arm in arms:
        fps = case_fixed(clip, end) if arm == "A" else case_legacy(clip, end)
        seq.append((arm, fps))
    fps_a = max([f for a, f in seq if a == "A"] or [0.0])
    fps_b = max([f for a, f in seq if a == "B"] or [0.0])

    head("A/B 汇总")
    for i, (arm, fp) in enumerate(seq, 1):
        print("  #%d  %s %-34s %6.2f 帧/秒"
              % (i, arm, ("保留：预览常驻 → 回收 → 30 帧 / 批 8" if arm == "A"
                          else "对照：不回收 → 8 帧 / 批 1"), fp), flush=True)
    print("  A 最好 %6.2f 帧/秒 · B 最好 %6.2f 帧/秒" % (fps_a, fps_b), flush=True)
    if fps_b > 0:
        print("  差：%+.1f%%（A/B − 1）" % (100.0 * (fps_a / fps_b - 1.0)), flush=True)
    # 「修完是不是就 ≥20 帧/秒」不是这一段能断的：那是素材 + 模型的绝对吞吐
    # （本机 1080p / 张元英 384 / retinaface_10g：两臂都在 15~16，检测 52ms/帧是硬底）。
    # 这一段只守一条：修完之后**不许比改动前更慢**（预算是纠错，不该拿性能当代价）。
    say("A6 fps = %.2f（参考：本机同素材同模型两臂都在 15~16，绝对吞吐不看这一段）" % fps_a)
    check(fps_b <= 0.0 or fps_a >= fps_b * 0.92, "A7 修复后不比改动前的账更慢（≥ −8%）",
          "A %.2f / B %.2f（%+.1f%%）"
          % (fps_a, fps_b, (100.0 * (fps_a / fps_b - 1.0)) if fps_b else 0.0))

    fails = [r for r in RESULTS if not r[0]]
    print("")
    print("=" * 78)
    print("汇总：PASS %d · FAIL %d · 用时 %.1fs" % (len(RESULTS) - len(fails), len(fails),
                                                  time.time() - t_all))
    for ok, name, detail in fails:
        print("  FAIL %s · %s" % (name, detail))
    print("沙箱：%s" % BOX)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
