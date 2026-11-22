# -*- coding: utf-8 -*-
"""视频切帧：用 ffmpeg 把视频按「帧率」或「每 N 帧」导出成图片序列。

只做一件事，所以不引引擎、不碰 DFL 的抽取流程：
  · 解码/缩放全交给 ffmpeg（-vf fps=N 或 select=not(mod(n\\,K))）
  · 进度用 ffmpeg 自己的 -progress pipe:1（frame/fps/speed/out_time_us），
    不用去数输出目录里的文件数，慢盘上那个数法既慢又抖。
  · 单任务单例：同时只跑一条，重复提交返回 409，和提取器同一口径。

输出命名是 image2 的 %06d.<ext>，和 DFL 的帧序列约定一致（合成器可直接读）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

WORKSPACE = Path(__file__).resolve().parents[3] / "workspace"

_LOCK = threading.Lock()
_JOB = {
    "running": False,
    "src": "", "dst": "", "mode": "", "value": 0,
    "msg": "", "error": "",
    "done": 0, "total": 0, "pct": 0.0,
    "fps": 0.0, "speed": "", "eta": "",
    "started": 0.0, "elapsed": 0.0,
    "cmd": "",
}
_PROC: subprocess.Popen | None = None
_THREAD: threading.Thread | None = None


def _ffmpeg() -> str:
    """找 ffmpeg：DFL_FFMPEG → imageio-ffmpeg → PATH。"""
    env = os.environ.get("DFL_FFMPEG", "").strip()
    if env and Path(env).exists():
        return env
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and Path(p).exists():
            return p
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise HTTPException(500, "找不到 ffmpeg：设置 DFL_FFMPEG，或把 ffmpeg 放进 PATH")


def _probe_fallback(src: Path, exe: str) -> dict:
    """没有 ffprobe 时用 `ffmpeg -i` 的输出兜底 —— imageio-ffmpeg 只带 ffmpeg 本体。

    没有这一步，duration 会是 0，进度条就永远停在 0%（只能等它突然跳到 100%）。
    """
    import re
    try:
        p = subprocess.run([exe, "-hide_banner", "-i", str(src)],
                           capture_output=True, text=True, errors="replace", timeout=30)
        txt = p.stderr or ""
    except Exception:
        return {"frames": 0, "duration": 0.0, "fps": 0.0, "w": None, "h": None}
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    fps = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
    if m:
        fps = float(m.group(1))
    w = h = None
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})", txt)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    return {"frames": int(dur * fps) if fps else 0, "duration": dur, "fps": fps, "w": w, "h": h}


def _ffprobe_info(src: Path) -> dict:
    """读时长/帧率/帧数。ffprobe 可能没有（imageio-ffmpeg 只带 ffmpeg），
    所以失败就退回用 ffmpeg -i 的输出解析，再失败就给 0（进度按已出帧数显示）。"""
    exe = _ffmpeg()
    probe = str(Path(exe).with_name("ffprobe.exe")) if exe.lower().endswith("ffmpeg.exe") else "ffprobe"
    if not Path(probe).exists() and shutil.which("ffprobe"):
        probe = shutil.which("ffprobe")
    args = [probe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,duration,r_frame_rate,width,height",
            "-of", "json", str(src)]
    try:
        out = subprocess.run(args, capture_output=True, timeout=30)
        d = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
        s = (d.get("streams") or [{}])[0]
        num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
        fr = (float(num) / float(den)) if float(den or 1) else 0.0
        out_info = {"frames": int(float(s.get("nb_frames") or 0)),
                    "duration": float(s.get("duration") or 0),
                    "fps": fr, "w": s.get("width"), "h": s.get("height")}
        # ffprobe 能跑但没给时长/帧数（mkv、剪辑点、mp4 缺 mvhd 都常见）→ 再兜一层
        if out_info["duration"] <= 0 and out_info["frames"] <= 0:
            return _probe_fallback(src, exe)
        return out_info
    except Exception:
        return _probe_fallback(src, exe)


def _reset(**kw) -> None:
    base = {"running": False, "src": "", "dst": "", "mode": "", "value": 0,
            "msg": "", "error": "", "done": 0, "total": 0, "pct": 0.0,
            "fps": 0.0, "speed": "", "eta": "", "started": time.time(),
            "elapsed": 0.0, "cmd": ""}
    base.update(kw)
    _JOB.update(base)


def _kill() -> None:
    global _PROC
    p = _PROC
    _PROC = None
    if p and p.poll() is None:
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=8)
        except Exception:
            pass


def _run(src: Path, dst: Path, mode: str, value: float, fmt: str, quality: int,
         t_from: float, t_to: float) -> None:
    """后台线程：起 ffmpeg、读 -progress、更新 _JOB。"""
    global _PROC
    try:
        info = _ffprobe_info(src)
        # 目标帧数：按帧率是 duration*fps，按每 N 帧是 frames/N
        span = (t_to - t_from) if t_to > t_from else info["duration"]
        dur_total = span if span > 0 else 0.0
        # 默认全切：不挂 -vf，源帧 1:1 全导出（-vsync 0 = passthrough）。
        # 帧率/间隔这些选项已经砍掉 —— 要抽帧的用「数据处理 → 过滤器」，
        # 这个工具只负责把视频原样拆成帧序列。
        src_frames = info["frames"] or int(info["duration"] * (info["fps"] or 0))
        if mode == "fps":                         # 兼容旧调用，保留但界面不再暴露
            expect = int(span * value)
            vf = "fps=%s" % ("%.6f" % value).rstrip("0").rstrip(".")
        elif mode == "every":
            expect = int(src_frames / max(1.0, value))
            vf = "select='not(mod(n\\,%d))'" % int(value)
        else:
            vf = ""
            expect = src_frames if t_to <= t_from else int(span * (info["fps"] or 0))
        if expect <= 0:
            expect = 0

        dst.mkdir(parents=True, exist_ok=True)
        ext = "png" if fmt == "png" else "jpg"
        pattern = str(dst / ("%06d." + ext))
        cmd = [_ffmpeg(), "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
               "-progress", "pipe:1", "-nostats"]
        if t_from > 0:
            cmd += ["-ss", "%.3f" % t_from]
        if t_to > t_from:
            cmd += ["-to", "%.3f" % t_to]
        cmd += ["-i", str(src)]
        if vf:
            cmd += ["-vf", vf]
        if ext == "jpg":
            # qscale 2 最好、31 最差；给个 1~31 的区间
            cmd += ["-q:v", str(max(1, min(31, int(quality))))]
        cmd += ["-vsync", "0", pattern]

        with _LOCK:
            _JOB["total"] = expect
            _JOB["cmd"] = " ".join(cmd)
            _JOB["msg"] = "正在切帧…"

        _PROC = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
        err_tail = []
        t0 = time.time()
        for line in _PROC.stdout:
            line = line.strip()
            if line.startswith("frame="):
                try:
                    done = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                with _LOCK:
                    _JOB["done"] = done
                    if expect:
                        _JOB["pct"] = min(99.0, done * 100.0 / expect)
                    _JOB["elapsed"] = time.time() - t0
                    if _JOB["elapsed"] > 0:
                        r = done / _JOB["elapsed"]
                        _JOB["fps"] = round(r, 1)
                        if expect and r > 0:
                            left = max(0.0, (expect - done) / r)
                            _JOB["eta"] = "%d:%02d" % (int(left) // 60, int(left) % 60)
            elif line.startswith("out_time_us="):
                # 帧数估不准（剪辑点、可变帧率）时，用已解码时长算进度更稳
                if dur_total > 0:
                    try:
                        us = int(line.split("=", 1)[1])
                        with _LOCK:
                            _JOB["pct"] = max(_JOB["pct"], min(99.0, us / 1e6 * 100.0 / dur_total))
                    except ValueError:
                        pass
            elif line.startswith("speed="):
                with _LOCK:
                    _JOB["speed"] = line.split("=", 1)[1].strip()
            elif line.startswith("progress=") and line.endswith("done"):
                break
        rc = _PROC.wait()
        try:
            err_tail = (_PROC.stderr.read() or "").strip().splitlines()[-4:]
        except Exception:
            pass
        with _LOCK:
            _JOB["elapsed"] = time.time() - t0
            if rc == 0:
                _JOB["done"] = _JOB["done"] or expect
                _JOB["pct"] = 100.0
                _JOB["msg"] = "完成：%d 帧" % _JOB["done"]
            else:
                _JOB["error"] = ("ffmpeg 退出码 %d" % rc) + (("：" + " / ".join(err_tail)) if err_tail else "")
                _JOB["msg"] = "失败"
    except Exception as e:
        with _LOCK:
            _JOB["error"] = str(e)[:400]
            _JOB["msg"] = "失败"
    finally:
        with _LOCK:
            _JOB["running"] = False
        _kill()


@router.post("/api/tools/frames")
def start_frames(payload: dict | None = None):
    """POST {src, dst, mode:'fps'|'every', value, fmt:'jpg'|'png', quality, from, to}"""
    global _THREAD
    payload = payload or {}
    src = Path(str(payload.get("src") or "").strip().strip('"'))
    dst_raw = str(payload.get("dst") or "").strip().strip('"')
    mode = str(payload.get("mode") or "all")   # all = 全切（默认）
    try:
        value = float(payload.get("value") or (2 if mode == "fps" else 1))
    except (TypeError, ValueError):
        raise HTTPException(400, "参数无效：value")
    value = max(value, 0.01)
    fmt = "png" if str(payload.get("fmt")) == "png" else "jpg"
    try:
        quality = int(payload.get("quality") or 2)
    except (TypeError, ValueError):
        quality = 2
    try:
        t_from = float(payload.get("from") or 0)
        t_to = float(payload.get("to") or 0)
    except (TypeError, ValueError):
        t_from, t_to = 0.0, 0.0

    if not str(src):
        raise HTTPException(400, "请选择源视频")
    if not src.exists():
        raise HTTPException(404, "源路径不存在：%s" % src)
    if src.is_dir():
        raise HTTPException(400, "源必须是视频文件（目录请用提取器）")
    dst = Path(dst_raw) if dst_raw else (WORKSPACE / "_frames" / src.stem)

    with _LOCK:
        if _JOB["running"]:
            raise HTTPException(409, "已经有一个切帧任务在跑")
        _reset(running=True, src=str(src), dst=str(dst), mode=mode, value=value,
               msg="准备中…")

    _THREAD = threading.Thread(
        target=_run, name="frames-job",
        args=(src, dst, mode, value, fmt, quality, t_from, t_to), daemon=True)
    _THREAD.start()
    return {"ok": True, "dst": str(dst)}


@router.get("/api/tools/frames/status")
def frames_status():
    with _LOCK:
        d = dict(_JOB)
    if d["running"] and d["started"]:
        d["elapsed"] = round(max(d["elapsed"], time.time() - d["started"]), 1)
    return d


@router.post("/api/tools/frames/stop")
def frames_stop():
    global _PROC
    with _LOCK:
        running = _JOB["running"]
    if not running:
        return {"ok": True, "running": False}
    _kill()
    with _LOCK:
        _JOB["running"] = False
        _JOB["msg"] = "已中断"
    return {"ok": True, "running": False}
