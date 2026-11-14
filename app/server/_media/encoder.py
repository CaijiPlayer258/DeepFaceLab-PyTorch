# -*- coding: utf-8 -*-
"""ffmpeg 封装：命令行拼装（_vid_cmd）、探视频（_vid_probe）+ 编码器/硬解表。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import time
from pathlib import Path
from .._common.http import (
    _BadInput,
)


# MergeStudio 的 -c:v 只认这些（其余名字 ffmpeg 会直接报 Unknown encoder）
_ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv",
             "h264_amf", "hevc_amf", "libx264", "libx265")
# 4.0 的 hwaccel 只有 auto / ""（关）两种语义，_merge_runner 也是这么折的
_HWACCEL = {"auto": "auto", "on": "auto", "1": "auto", "true": "auto",
            "off": "off", "none": "off", "0": "off", "false": "off", "": "off"}


def _encoder_val(raw) -> str:
    key = str(raw or "").strip()
    if not key:
        return "h264_nvenc"
    low = key.lower()
    if low not in _ENCODERS:
        raise _BadInput("「encoder」只认 %s，收到 %r" % (" / ".join(_ENCODERS), key))
    return low


def _hwaccel_val(raw) -> str:
    if raw is None:
        return "auto"
    key = str(raw).strip().lower()
    if key not in _HWACCEL:
        raise _BadInput("「hwaccel」只认 auto（开）/ off（关），收到 %r" % (raw,))
    return _HWACCEL[key]


def _vid_probe(video: Path) -> dict:
    """源视频的 帧数 / fps / 宽高（导出只认源视频，不猜）。

    ffmpeg 的 metadata 探测偶尔会抽风（文件正被别的进程摸一下、刚写入还没 flush 完），
    一次失败就 422 会让整条分段链莫名其妙断掉 —— 这里重试两次再说。
    """
    last = None
    frames = fps = 0
    w = h = 0
    for attempt in range(3):
        try:
            frames, fps, w, h = _app_mod._video_probe(video)
            last = None
            break
        except Exception as e:                              # noqa: BLE001
            last = e
            time.sleep(0.25 * (attempt + 1))
    if last is not None:
        raise _BadInput("读不出视频信息（%s）：%s" % (video.name, str(last)[:160]), 422)
    if not int(w or 0) or not int(h or 0):
        raise _BadInput("视频尺寸读不出来（%s）—— 换一个能解的片子" % video.name, 422)
    return {"frames": int(frames or 0), "fps": float(fps or 30.0),
            "w": int(w), "h": int(h)}



def _vid_cmd(pattern: str, dest: Path, video: Path, fps: float, img_start: int = 1,
             audio_ss: float = None, audio_t: float = None, limit: int = None) -> list:
    """一帧不差地构造导出用的 ffmpeg 命令（分段链和快速导出**共用这一份**）。

    ```
    ffmpeg -y ... -framerate FPS -start_number IMG_START -i <帧序列>
           [-ss AUDIO_SS -t AUDIO_T] -i <源视频>
           -map 0:v:0 -map 1:a?
           -vf pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv
           -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy [-frames:v LIMIT]
           [-movflags +faststart] -progress pipe:1 <dest>
    ```

    三条不能动的口径（改了就成片不一致）：
      · crf16 + yuv420p + out_range=tv：PNG 帧是 full range，不显式转 limited 会被标成 yuvj420p；
      · 音轨 -c:a copy：不重编码、**不带 -shortest**（带上去会按视频/音轨较短的一边截断，
        帧范围的片子就少一截音）；
      · 奇数宽高先 pad 一像素，否则 x264 直接拒。
    """
    pre_img = ["-start_number", str(int(img_start))]
    pre_aud = []
    if audio_ss is not None:
        pre_aud += ["-ss", "%.6f" % float(audio_ss)]
    if audio_t is not None:
        pre_aud += ["-t", "%.6f" % float(audio_t)]
    lim = ["-frames:v", str(int(limit))] if limit else []
    cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
           "-nostats", "-nostdin", "-framerate", "%.6g" % fps] + pre_img + \
          ["-i", str(pattern)] + pre_aud + ["-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "copy"] + lim
    if Path(dest).suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-progress", "pipe:1", str(dest)]
    return cmd


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 media_encoder 与 _app_mod
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod
