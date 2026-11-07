"""ffmpeg 硬件解码候选链：cuda → d3d11va → qsv → none。

背景（本机实测）：抽 200 帧 1080p HEVC，整条提取流水线 2.21s，其中 ffmpeg 解码
1.90s（86%）—— 检测搬上 TensorRT 之后，CPU 软解成了唯一瓶颈。所以给解码加一条
硬件加速候选链，按 独显 NVDEC → 核显（d3d11va / qsv）→ CPU 的顺序**真跑一遍**，
第一个通的就采用，结果一个进程只探一次。

几个刻意的选择：

1) 只传 -hwaccel X，**不传** -hwaccel_output_format。
   传了 -hwaccel_output_format cuda 帧会留在显存里，下游 np.frombuffer / cv2
   拿到的是 GPU 句柄，整条流水线（检测、对齐、JPEG 编码）都得改成 GPU 张量。
   这里只要「解码挪到显卡、帧照旧回 CPU」，让 ffmpeg 自己把帧下回内存再转 bgr24。

2) 探测要能识别「假成功」。
   ffmpeg 硬解初始化失败时经常**不报错**，只打一行 warning 就退回软解继续吐帧。
   只看退出码就会把软解当硬解缓存下来，越「优化」越慢。所以额外扫 stderr，见到
   "Failed setup for format ... hwaccel initialisation returned error" 这类字样
   一律判该级不可用，继续下一级。

3) 硬解要多挂一个 format=yuv420p（见 normalize_filter）。
   硬解下 hwdownload 给出来的是 NV12，swscale 的 NV12 -> BGR 和软解的 I420 -> BGR
   是两条不同的转换路径：Y/U/V 平面逐位相同，转成 BGR 后却能差 13/255（实测）。
   在 BGR 之前先 format=yuv420p 规范化一次，四组帧就逐位一致了（md5 相同）。

4) 任何一级失败都不抛异常。
   四级都不行就用 none；DFL_HWACCEL 写成不认识的值也退回 auto，绝不因此中断
   一次抽取。

环境变量：
    DFL_HWACCEL=cuda|d3d11va|qsv|none|auto     默认 auto
    DFL_HWACCEL_PROBE_FRAMES=8                  探测时真解几帧，调大更保守

强制某个级别时它排到队首，失败仍按顺序往下试 —— 「强制」只影响优先级，不影响
「绝不因解码不了而失败」这条底线。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ENV = "DFL_HWACCEL"
ORDER = ("cuda", "d3d11va", "qsv")          # 探测顺序：NVDEC → Windows 通用/核显 → Intel
NONE_WORDS = ("none", "off", "cpu", "0", "false", "no")

_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_PROBE_SECONDS = 3                          # 样本时长（秒）
try:
    _PROBE_FRAMES = max(1, int(os.environ.get("DFL_HWACCEL_PROBE_FRAMES") or 8))
except ValueError:
    _PROBE_FRAMES = 8
_PROBE_TIMEOUT = 25.0

# 出现这些字样 = 硬解其实没起来（ffmpeg 已退回软解），该级判不可用
_BAD_MARKERS = (
    "hwaccel initialisation returned error",
    "Failed setup for format",
    "No decoder surfaces left",
    "Device setup failed",
    "Could not open decoder",
    "Cannot load nvcuda",
    "Failed to initialise",
    "Error while decoding stream",
    "Invalid data found when processing input",
)

_lock = threading.Lock()
_state = {
    "done": False,      # 探测过了没
    "name": None,       # 最终采用的级别："cuda"/"d3d11va"/"qsv"/"none"
    "ms": 0,            # 选中那一级的探测耗时
    "choice": "",       # DFL_HWACCEL 原值
    "tried": [],        # [(level, ok, ms, why), ...]
    "fails": {},        # 实跑失败次数（探测过了但真解码挂了）
}


def _log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ---------------------------------------------------------------- 探测

def _sample_for(src):
    """3 秒小样本：真实视频优先（同编码同分辨率最可信），没有就现造一个 720p H.264。"""
    if src:
        try:
            p = Path(str(src))
            if p.is_file() and p.stat().st_size > 1024:
                return str(p)
        except OSError:
            pass
    d = Path(tempfile.gettempdir()) / "dfl_hwaccel_probe"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return ""
    out = d / "sample_720p_h264.mp4"
    if out.is_file() and out.stat().st_size > 4096:
        return str(out)
    exe = _exe()
    for enc in ("libx264", "mpeg4"):
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
               "-t", str(_PROBE_SECONDS), "-c:v", enc, "-pix_fmt", "yuv420p", str(out)]
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=_PROBE_TIMEOUT,
                               creationflags=_NO_WINDOW)
            if p.returncode == 0 and out.is_file() and out.stat().st_size > 4096:
                return str(out)
        except Exception:
            continue
    return ""


def _probe(level, sample):
    """拿 sample 的前 3 秒真跑一次 ffmpeg，取 rawvideo 通道。

    level 为空 = 纯软解（对照级）。判据：退出码 0、stdout 有真帧、stderr 没有
    「已退回软解」的痕迹。
    """
    exe = _exe()
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if level:
        cmd += ["-hwaccel", level]           # 只上硬解，不下 -hwaccel_output_format
    cmd += ["-t", str(_PROBE_SECONDS), "-i", sample]
    cmd += ["-frames:v", str(_PROBE_FRAMES),
            "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-f", "image2pipe", "-"]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=_PROBE_TIMEOUT,
                           creationflags=_NO_WINDOW)
    except Exception as e:                                  # 超时 / 起不来
        return False, (time.perf_counter() - t0) * 1000.0, "%s: %s" % (
            type(e).__name__, str(e)[:120])
    ms = (time.perf_counter() - t0) * 1000.0
    err = (p.stderr or b"").decode("utf-8", "ignore")
    if p.returncode != 0:
        lines = err.strip().splitlines()[-1:] or ["退出码 %s" % p.returncode]
        return False, ms, lines[0][:160]
    if not p.stdout:
        return False, ms, "输出为空"
    low = err.lower()
    for mk in _BAD_MARKERS:
        if mk.lower() in low:
            return False, ms, "已退回软解（%s）" % mk
    return True, ms, ""


def _chain(choice):
    """DFL_HWACCEL → 探测链。返回值 (chain, 备注)。"""
    c = (choice or "").strip().lower()
    if c in ("", "auto"):
        return list(ORDER), ""
    if c in NONE_WORDS:
        return [], ""
    if c in ORDER:
        # 强制只改优先级：这一级先试，失败照样往下走，最后兜底 none
        return [c] + [m for m in ORDER if m != c], ""
    return list(ORDER), "%s=%s 不认识，按 auto 探测" % (ENV, choice)


def _resolve_locked(src):
    choice = os.environ.get(ENV) or "auto"
    chain, note = _chain(choice)
    _state["choice"] = choice
    if note:
        _log("[decode] " + note)
    if not chain:
        _state.update(done=True, name="none", ms=0)
        _log("[decode] hwaccel=none（%s=%s 强制软解）" % (ENV, choice))
        return

    sample = _sample_for(src)
    if not sample:
        _state.update(done=True, name="none", ms=0)
        _log("[decode] hwaccel=none：%s 均不可用，且造不出探测样本" % "/".join(ORDER))
        return

    tried = []
    for level in chain:
        ok, ms, why = _probe(level, sample)
        tried.append((level, ok, round(ms, 1), why))
        if ok:
            _state.update(done=True, name=level, ms=round(ms), tried=tried)
            _log("[decode] hwaccel=%s（探测 %dms）" % (level, round(ms)))
            return
        _log("[decode] hwaccel=%s 探测失败（%dms）：%s" % (level, round(ms), why or "-"))
    _state.update(done=True, name="none", tried=tried)
    _log("[decode] hwaccel=none：%s 均不可用" % "/".join(ORDER))


def _resolve(src):
    with _lock:
        if not _state["done"]:
            _resolve_locked(src)


# ---------------------------------------------------------------- 对外

def input_params(src=None):
    """给 imageio_ffmpeg / ffmpeg 的输入侧参数。

    返回 ["-hwaccel","cuda"] 或 []（软解）。第一次调用会做一次探测（~0.2-0.5s），
    之后走缓存。
    """
    _resolve(src)
    name = _state["name"]
    return [] if name in (None, "none") else ["-hwaccel", name]


def current():
    return _state["name"] or "none"


NORM_FILTER = "format=yuv420p"


def normalize_filter():
    """硬解专用的收尾滤镜；软解返回空串。

    hwdownload 出来的是 NV12，swscale 的 NV12->BGR 与软解 I420->BGR 不是同一条
    路径，同一帧的 BGR 能差 13/255。先 format=yuv420p 摊平成平面 4:2:0 再转，
    两种解码出来的帧就逐位一致（实测 md5 相同）。
    """
    _resolve(None)
    return "" if _state["name"] in (None, "none") else NORM_FILTER


def report():
    return {"name": _state["name"], "probe_ms": _state["ms"],
            "choice": _state["choice"], "tried": list(_state["tried"]),
            "fails": dict(_state["fails"])}


def note_failure(level, why=None):
    """探测通了但真解码挂掉时调用：本次立刻换软解，同级别连挂 2 次就全局降级。"""
    if not level or level == "none":
        return
    with _lock:
        n = _state["fails"].get(level, 0) + 1
        _state["fails"][level] = n
        if n >= 2 and _state["name"] == level:
            _state["name"] = "none"
            _log("[decode] hwaccel=%s 实跑连续失败，本进程改用 none：%s"
                 % (level, str(why)[:120] if why else "-"))


def reset():
    """清掉缓存（测试用）。"""
    with _lock:
        _state.update(done=False, name=None, ms=0, choice="", tried=[], fails={})
