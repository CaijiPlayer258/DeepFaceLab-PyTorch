# -*- coding: utf-8 -*-
r"""显存「可回收」验收：预览常驻过的模型不许把导出堵死，堵死时也必须说清是谁占的。

验的东西（对应这一轮用户报的真 bug：空闲 1387MB → 导出被拒，而他什么都没跑）：
  P0  探测校准：同一刻把我们的 free（torch 与 nvidia-smi 取小）和 nvidia-smi 原始行
      并排打出来；两者必须对得上（差 >256MB 就是探测口径坏了）。
  A   复现路径（用户那条）：预览一帧（worker 里 dfm 常驻）→ 直接点导出
      · A1 预览之后：worker 里确实常驻着模型（used_mb 有数）、空闲确实掉了；
      · A2 导出 dry_run：**计划必须 ok**（不再被拒），日志里有「回收预览模型 X MB」，
        plan.reclaimed_mb ≥ 1000MB，available 涨回来；
      · A3 真导出 12 帧：phase=done、成片存在、全程没有 CUDA OOM；
      · A4 导出结束后预览仍能用（懒加载回来），耗时打出来。
  B   别的程序占着：起一个 torch 子进程吃 2GB
      · B1 只占 2GB 时仍然允许导出（我们自己那 2GB 能回收，其它程序那 2GB 不归我们管）；
      · B2 占到真的放不下时 → 400，且消息**区分来源**：
        「本程序 X GB（可回收 Y）」+「其它程序 Z GB」，结构化字段 own_mb/other_mb 有数。
  C   status/日志：/api/merge/preview/status 有 used_mb；vram_plan 里有 own/other。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\vram_reclaim_accept.py
    （--cases p0,a,b ；--dfm 张元英 换模型）
退出码 0 = 全过。全程在 %TEMP% 里，不碰工作区产物。
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
BOX = TEMP / "dfl_vram_reclaim"
WS = BOX / "ws"
MEDIA = BOX / "media"
OUT = BOX / "out"
LOGDIR = BOX / "log"
SERVER_LOG = LOGDIR / "server.log"

DFM = "张元英"                      # 384 模型（文件名不带尺寸 → 加载后还会再判一次预算）
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


def smi():
    """nvidia-smi 的原始一行（排障第一现场）。"""
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,name,memory.total,memory.used,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return (r.stdout or "").strip()
    except Exception as e:                                  # noqa: BLE001
        return "ERR %s" % e


def smi_free():
    try:
        row = smi().splitlines()[0]
        return int(row.split(",")[4])
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
    for k in ("DFL_PORT", "DFL_STAGE_KEEP", "DFL_FUSED_VRAM_FREE_MB", "DFL_FUSED_BUDGET",
              "DFL_VRAM_FREE_MB", "DFL_ALLOW_CPU_FALLBACK", "DFL_ALLOW_TEST_HOOK"):
        env.pop(k, None)
    for k, v in (extra_env or {}).items():
        env[str(k)] = str(v)
    LOGDIR.mkdir(parents=True, exist_ok=True)
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


class Hog:
    """另一个进程（模拟「别的程序占着显存」）：torch 里真分配、真 hold 住。"""

    def __init__(self, mb):
        self.mb = int(mb)
        self.proc = None

    def __enter__(self):
        code = (
            "import torch, time\n"
            "n = int(%d * 1024 * 1024 // 4)\n"
            "x = torch.empty(n, dtype=torch.float32, device='cuda')\n"
            "x.zero_(); torch.cuda.synchronize()\n"
            "print('HOG_READY', flush=True)\n"
            "time.sleep(600)\n" % self.mb)
        self.proc = subprocess.Popen([str(PY), "-c", code], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        line = self.proc.stdout.readline()
        say("占显存的诱饵进程 pid=%s · %s MB · %s"
            % (self.proc.pid, self.mb, (line or "").strip()))
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            _kill_tree(self.proc.pid)
            try:
                self.proc.kill()
            except Exception:                               # noqa: BLE001
                pass
        time.sleep(1.5)
        return False


# ============================================================
# 请求体
# ============================================================
def export_body(clip, out, dry=False, budget=30):
    body = {"video": str(clip), "model": DFM, "settings": dict(SETTINGS),
            "output_dir": str(out), "frame_budget": int(budget),
            "start_frame": 1, "end_frame": 12}
    if dry:
        body["dry_run"] = True
    return body


def preview_body(clip, t=0.5):
    return {"video": str(clip), "t": float(t), "model": DFM, "settings": dict(SETTINGS)}


def plan_of(d):
    """dry_run 的响应把计划放在 vram_plan；拒绝（400）放在顶层 vram。"""
    d = d or {}
    return dict(d.get("vram_plan") or d.get("vram") or {})


def plan_brief(plan):
    keys = ("ok", "free_mb", "avail_mb", "need_mb", "peak_mb", "frames", "batch",
            "reclaimed_mb", "reclaimable_mb", "own_mb", "own_inproc_mb", "own_preview_mb",
            "other_mb", "resident_mb", "source")
    return {k: (plan or {}).get(k) for k in keys}


# ============================================================
# P0 · 探测校准
# ============================================================
def case_probe(clip):
    head("P0 · 探测校准：我们的 free vs nvidia-smi 原始行")
    proc = start_server("校准")
    try:
        say("nvidia-smi 原始行：%s" % smi())
        st, d = post("/api/merge/export", export_body(clip, OUT / "p0.mp4", dry=True))
        plan = plan_of(d)
        say("我们的计划：%s" % json.dumps(plan_brief(plan), ensure_ascii=False))
        ours = plan.get("free_mb")
        sf = smi_free()
        check(st == 200, "p0 dry_run 预检可用（HTTP 200）", "HTTP %s" % st)
        check(ours is not None and sf is not None and abs(int(ours) - int(sf)) <= 256,
              "p0 我们的 free 与 nvidia-smi 对得上（差 ≤ 256MB）",
              "ours=%s / smi=%s" % (ours, sf))
        check(str(plan.get("source") or "").startswith("min(torch"),
              "p0 source 写明了两个来源（torch 与 smi 取小）", plan.get("source"))
        check("torch" in str(plan.get("source") or "")
              and plan.get("free_torch_mb") is not None,
              "p0 计划里带 free_torch_mb（WDDM 上 torch 偏乐观，要能看见）",
              "torch=%s smi=%s" % (plan.get("free_torch_mb"), plan.get("free_smi_mb")))
    finally:
        stop_server(proc)


# ============================================================
# A · 复现：预览常驻 → 导出（用户的病）
# ============================================================
def case_preview_then_export(clip):
    head("A · 预览一帧（模型常驻）→ 直接点导出：必须成功 + 日志写明回收了多少")
    proc = start_server("回收")
    try:
        mark = log_mark()
        _st, d0 = post("/api/merge/export", export_body(clip, OUT / "a0.mp4", dry=True))
        p0 = plan_of(d0)
        say("A0 预览之前：%s" % json.dumps(plan_brief(p0), ensure_ascii=False))
        say("   nvidia-smi：%s" % smi())
        free_before = p0.get("free_mb")

        t0 = time.time()
        st, d = post("/api/merge/preview", preview_body(clip), timeout=900)
        prev_ms = (time.time() - t0) * 1000.0
        check(st == 200 and (d or {}).get("ok") is True,
              "a1 预览出图（第一次 = 冷加载）", "HTTP %s · %.0fms" % (st, prev_ms))
        _st2, dv = get("/api/merge/preview/status")
        wk = dict(dv or {})            # status 把 worker 的字段直接摊在顶层
        say("a1 预览 worker：models=%s used_mb=%s boot_ms=%s"
            % (wk.get("models"), wk.get("used_mb"), wk.get("boot_ms")))
        check(bool(wk.get("models")), "a1 预览 worker 里确实常驻着 dfm",
              "models=%s" % (wk.get("models"),))
        check(float(wk.get("used_mb") or 0.0) >= 500.0,
              "a1 worker 实测占着 ≥500MB（这也是「本程序占用」的一半）",
              "used_mb=%s" % wk.get("used_mb"))
        sf_now = smi_free()
        say("a1 常驻之后 nvidia-smi：%s" % smi())
        check(sf_now is not None and free_before is not None and sf_now < free_before,
              "a1 空闲显存被我们自己的 worker 吃掉了（这就是「我什么都没跑」的假象）",
              "空闲 %s → %s MB" % (free_before, sf_now))

        # ---- 关键：预览常驻着，直接点导出（dry_run 先看计划） ----
        _st3, d3 = post("/api/merge/export", export_body(clip, OUT / "a1.mp4", dry=True))
        p1 = plan_of(d3)
        say("A2 预览常驻时 dry_run：%s" % json.dumps(plan_brief(p1), ensure_ascii=False))
        check(_st3 == 200 and p1.get("ok") is True,
              "a2 预览常驻 → 导出预检**不再被拒**",
              "HTTP %s ok=%r msg=%s" % (_st3, p1.get("ok"),
                                        (p1.get("msg") or (d3 or {}).get("error") or "")[:160]))
        check(float(p1.get("reclaimed_mb") or 0.0) >= 1000.0,
              "a2 计划里记着「回收了多少」且 ≥1GB",
              "reclaimed_mb=%s" % p1.get("reclaimed_mb"))
        blob = log_since(mark)
        hits = [ln.strip() for ln in blob.splitlines() if "回收预览模型" in ln]
        check(bool(hits), "a2 后端日志有一行「回收预览模型 X MB … 空闲 A → B」",
              (hits[0][:200] if hits else ""))
        m = re.search(r"回收预览模型\s*([0-9.]+)\s*MB", hits[0] if hits else "")
        if m:
            say("a2 回收了 %.2f GB" % (float(m.group(1)) / 1024.0))
        say("nvidia-smi（回收后）：%s" % smi())

        # ---- 真导出 ----
        st4, d4 = post("/api/merge/export", export_body(clip, OUT / "a1.mp4"))
        check(st4 == 200 and (d4 or {}).get("ok") is not False,
              "a3 导出真的起来了（不是 400）",
              "HTTP %s %s" % (st4, ((d4 or {}).get("error") or "")[:160]))
        if st4 != 200:
            return
        out_p = Path(str((d4 or {}).get("output") or (OUT / "a1.mp4")))
        fin = wait_export(1800.0)
        plan = fin.get("vram_plan") or {}
        say("a3 终态：phase=%s err=%s" % (fin.get("phase"), (fin.get("error") or "")[:200]))
        say("a3 终态计划：%s" % json.dumps(plan_brief(plan), ensure_ascii=False))
        check(fin.get("phase") == "done" and fin.get("ok") is True,
              "a3 12 帧出片成功", "phase=%s" % fin.get("phase"))
        info = probe(out_p) if out_p.is_file() else {}
        check(bool(info) and int(info.get("frames") or 0) == 12,
              "a3 成片 12 帧", "%s 帧 · %s" % (info.get("frames") if info else "-", out_p.name))
        blob2 = log_since(mark)
        check(not OOM_PAT.search(blob2), "a3 全程没有 CUDA OOM",
              (OOM_PAT.search(blob2).group(0) if OOM_PAT.search(blob2) else "clean"))
        say("a3 导出之后 nvidia-smi：%s" % smi())

        # ---- 导出结束后预览仍能用（懒加载回来） ----
        _st5, dv2 = get("/api/merge/preview/status")
        wk2 = dict(dv2 or {})
        say("a4 导出之后 nvidia-smi：%s / worker used_mb=%s"
            % (smi(), wk2.get("used_mb")))
        check(not wk2.get("models"),
              "a4 导出结束后预览 worker 没有立刻重载（懒加载，别来回抖）",
              "models=%s used_mb=%s" % (wk2.get("models"), wk2.get("used_mb")))
        t0 = time.time()
        st6, d6 = post("/api/merge/preview", preview_body(clip, 0.2), timeout=900)
        warm_ms = (time.time() - t0) * 1000.0
        check(st6 == 200 and (d6 or {}).get("ok") is True,
              "a4 导出之后预览照样能出图（懒加载回来了）",
              "HTTP %s · %.0fms（冷加载那次 %.0fms）" % (st6, warm_ms, prev_ms))
        _st7, dv3 = get("/api/merge/preview/status")
        wk3 = dict(dv3 or {})
        say("a4 预览 worker：models=%s used_mb=%s spawns=%s"
            % (wk3.get("models"), wk3.get("used_mb"), wk3.get("spawns")))
        check(bool(wk3.get("models")) and int(wk3.get("spawns") or 0) <= 1,
              "a4 懒加载走的是同一个 worker（没有为了预览再起新进程）",
              "spawns=%s" % wk3.get("spawns"))
    finally:
        stop_server(proc)


# ============================================================
# B · 别的程序占着显存
# ============================================================
def case_foreign(clip):
    head("B · 别的程序占着显存：能回收自己的就照跑；真放不下才拒绝，且说清是谁")
    proc = start_server("外来占用")
    try:
        mark = log_mark()
        post("/api/merge/preview", preview_body(clip), timeout=900)
        _s, dv = get("/api/merge/preview/status")
        say("b0 预览常驻：models=%s used_mb=%s"
            % ((dv or {}).get("models"), (dv or {}).get("used_mb")))
        # 只占 1.2GB：我们自己的 2GB 收回来之后仍然装得下（这才是正常用户的情形）
        with Hog(1200):
            say("b1 nvidia-smi（诱饵 2GB）：%s" % smi())
            st, d = post("/api/merge/export", export_body(clip, OUT / "b1.mp4", dry=True))
            p = plan_of(d)
            say("b1 计划：%s" % json.dumps(plan_brief(p), ensure_ascii=False))
            check(st == 200 and p.get("ok") is True,
                  "b1 自己可回收 + 别人占 2GB → 仍然允许导出",
                  "HTTP %s ok=%r avail=%s need=%s" % (st, p.get("ok"), p.get("avail_mb"),
                                                      p.get("need_mb")))
            check(float(p.get("other_mb") or 0.0) >= 900.0,
                  "b1 计划里「其它程序」那一栏认得出那 1.2GB",
                  "other_mb=%s own_mb=%s" % (p.get("other_mb"), p.get("own_mb")))
        # 占到真的放不下：4.5GB
        with Hog(4600):
            say("b2 nvidia-smi（诱饵 4.5GB）：%s" % smi())
            st2, d2 = post("/api/merge/export", export_body(clip, OUT / "b2.mp4"))
            err = str((d2 or {}).get("error") or "")
            ext = d2 or {}
            p2 = ext.get("vram") or {}
            say("b2 HTTP %s · %s" % (st2, json.dumps(plan_brief(p2), ensure_ascii=False)))
            say("b2 消息：%s" % err[:400])
            check(st2 == 400, "b2 真放不下 → 明确拒绝（HTTP 400）", "HTTP %s" % st2)
            check(("本程序" in err) and ("其它程序" in err),
                  "b2 拒绝消息区分「本程序占用 / 其它程序占用」",
                  ("本程序" in err, "其它程序" in err))
            check(re.search(r"需要\s*[0-9.]+\s*GB", err) is not None
                  and re.search(r"可用\s*[0-9.]+\s*GB", err) is not None,
                  "b2 消息里仍有「需要 X GB / 可用 Y GB」（老口径不回退）", err[:120])
            check(ext.get("own_mb") is not None and ext.get("other_mb") is not None,
                  "b2 结构化字段带 own_mb / other_mb（前端不用抠中文）",
                  "own=%s other=%s" % (ext.get("own_mb"), ext.get("other_mb")))
            check(not Path(str(OUT / "b2.mp4")).exists(), "b2 没有产生成片",
                  "b2.mp4")
        blob = log_since(mark)
        check(not OOM_PAT.search(blob), "b 全程没有 CUDA OOM",
              (OOM_PAT.search(blob).group(0) if OOM_PAT.search(blob) else "clean"))
    finally:
        stop_server(proc)


# ============================================================
# main
# ============================================================
def main(argv=None):
    global DFM
    ap = argparse.ArgumentParser(description="显存可回收验收（预览常驻 → 导出）")
    ap.add_argument("--cases", default="p0,a,b", help="要跑的案例（p0/a/b）")
    ap.add_argument("--dfm", default=DFM, help="用哪个 dfm（默认 张元英 384）")
    args = ap.parse_args(argv)
    DFM = str(args.dfm)
    want = [c.strip().lower() for c in str(args.cases).split(",") if c.strip()]
    t_all = time.time()

    prepare_sandbox()
    src = sorted((REPO / "workspace").glob("*.mp4"))[0]
    clip = MEDIA / "clip12.mp4"
    run_ff(["-ss", "3", "-t", "0.4", "-i", src, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", clip],
           "切样片")
    say("样片 %s：%s · dfm=%s" % (clip.name, probe(clip), DFM))
    say("nvidia-smi 起始：%s" % smi())

    if "p0" in want:
        case_probe(clip)
    if "a" in want:
        case_preview_then_export(clip)
    if "b" in want:
        case_foreign(clip)

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
