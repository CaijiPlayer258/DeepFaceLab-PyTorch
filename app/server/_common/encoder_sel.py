# -*- coding: utf-8 -*-
"""导出成片的编码器（DFL_FUSED_ENCODER）：默认 libx264 crf16；4K 可显式换 NVENC。

**为什么要有这个开关**（实测：RTX 5070 Laptop + 8940HX 16C/32T，2160×3840 真实实拍 80 帧，
口径与 pipeline_fused._fused_pipe_cmd 逐字一致，只换 -c:v 那一段）：

| 臂 | fps | ms/帧 | ffmpeg 占用 | 体积 |
|---|---|---|---|---|
| libx264 crf16（默认） | 17.0 | 59.0 | **14.98 / 16 物理核** | 30.4 Mbps |
| h264_nvenc CQ19     | 45.7 | 21.9 | 5.3 核 | 41.3 Mbps |
| hevc_nvenc CQ19     | 50.3 | 19.9 | 6.1 核 | 21.1 Mbps |

  · 4K 下 crf16 把整机吃满（94%），上游的检测 / dfm / 合成只剩 1 核，流水线被挤成串行 ——
    这不是"慢一点"，是抢占。1080p 只要 14.2 ms/帧，但同样占 14.8 核（已满，换 4K 是时间 ×4.2）。
  · 不编码、只走 raw 管道（24.9 MB/帧 + swscale）就已经 15.7 ms/帧（64 fps、4.7 核）——
    4K 的天花板不在编码器；换 NVENC 拿到的是「2.7~3.0× + 还回 10 个核」，不是无限快。
  · 画质：逐像素均差 0.80 LSB、99% 像素差 <1 LSB、>8 LSB 只占 0.02%，PSNR(vs x264) 45.6 dB、
    SSIM 0.991，抠图肉眼无差。**换编码器不是画质退步，是换一条压缩曲线。**

**三条纪律（改之前先看）**：
  1. 不设 DFL_FUSED_ENCODER = **一个字节都不许变**：默认分支返回的就是原来的
     ["-c:v", "libx264", "-crf", "16"]，与 4.1 的成片逐字节相同（accept 里对拍 argv + md5）。
  2. 换编码器 = **换 md5**。不是 bug，是"成片不再是同一份文件"。所以只能由用户显式选；
     auto 在 4K 下只**告警**，不自动切（要自动切得另开 DFL_FUSED_ENCODER_AUTO4K=1，默认关）。
  3. 两条出口（raw 管道 / PNG 落盘）必须调**同一个函数**选编码器，否则「两条路成片 md5 相同」
     这条不变量就没了（parity 用例在测它）。

**CQ 与 crf 不等价**（同素材实测体积，80 帧）：crf16 = 30.4 Mbps；h264_nvenc CQ19 = 41.3（+36%）、
CQ21 = 32.2（+6%）、CQ23 = 24.7（−18%）；hevc_nvenc CQ21 = 21.0（−31%）、CQ23 = 17.7。
默认 CQ 取 **21**：体积和 crf16 差 6%，是"换了编码器但文件还是原来那么大"的那一档。

**HEVC 的兼容面**：ffmpeg 的 mp4 默认打 `hev1`，Apple 系（QuickTime / Safari）只认 `hvc1`，
所以这里显式加 `-tag:v hvc1`。微信 / 老安卓的 H5 `<video>` 可能黑屏；实测软解 4K
（h264 177 fps / hevc 174 fps，因为体积只有 21 Mbps）解码不是瓶颈，瓶颈在"那个播放器认不认"。

用法（只在真的要用时才探一次，探不通**在起 ffmpeg 之前**就报错）：

    DFL_FUSED_ENCODER=auto|x264|nvenc-h264|nvenc-hevc     # 默认 auto = x264（现状）
    DFL_FUSED_ENCODER_CQ=21                               # 只对 nvenc 生效
    DFL_FUSED_ENCODER_AUTO4K=1                            # auto 在 >=6MP 时真自动切（默认 0）

文档：app/docs/29-ENCODER.md
"""
from __future__ import annotations

import os
import subprocess

from .http import _BadInput

ENCODER_ENV = "DFL_FUSED_ENCODER"
CQ_ENV = "DFL_FUSED_ENCODER_CQ"
AUTO4K_ENV = "DFL_FUSED_ENCODER_AUTO4K"

# >= 这个像素数算 4K：2160x3840 与 3840x2160 都是 8.29 MP（1080x1920 只有 2.07 MP）
NVENC_MIN_PX = 6_000_000
# 4K 告警文案里那句"吃掉 N/M 核"用的：实测 crf16 在 4K 下吃 14.98/16 物理核
_CORES_X264_4K = "15/16"

# 默认分支：**逐字节**就是 4.1 的那一段，别动（成片 md5 靠它）
X264_ARGS = ("-c:v", "libx264", "-crf", "16")
_NVENC = {"nvenc-h264": "h264_nvenc", "nvenc-hevc": "hevc_nvenc"}
_MODES = ("auto", "x264") + tuple(_NVENC)

# 预检缓存：enc -> (ok, stderr 尾巴)。一次进程只探一次（探一次 ~250ms，一单导出只该付一次）
_PROBE = {}


def _truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in ("1", "on", "true", "yes")


def mode() -> str:
    """这一单选的编码器口径（不解析 auto）。非法值当场报错 —— 那时还没起 ffmpeg，不亏。"""
    raw = str(os.environ.get(ENCODER_ENV) or "").strip().lower()
    if not raw:
        return "auto"
    if raw not in _MODES:
        raise _BadInput("「%s」只认 %s，收到 %r"
                        % (ENCODER_ENV, " / ".join(_MODES), raw))
    return raw


def cq() -> int:
    """nvenc 的 CQ（0~51）。默认 21 = 与 crf16 体积持平那一档。"""
    raw = str(os.environ.get(CQ_ENV) or "").strip()
    if not raw:
        return 21
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        raise _BadInput("「%s」要 0~51 的整数，收到 %r" % (CQ_ENV, raw))
    return max(0, min(51, v))


def auto4k_on() -> bool:
    return _truthy(AUTO4K_ENV)


def _px(w, h) -> int:
    try:
        return int(w) * int(h)
    except (TypeError, ValueError):
        return 0


def nvenc_probe(enc: str):
    """250ms 预检：这个编码器在不在、驱动认不认。返回 (ok, stderr 尾巴)。

    为什么值得付这 250ms：NVENC 不可用时 ffmpeg 是**在 -i 之后**才报错的（raw 管道那一路
    甚至要等第一帧写进去才知道），用户看到的是「导出到 30% 挂了」。在这里探完再起进程，
    失败就是一条干净的 4xx，不留半截 mp4、不留孤儿进程。
    """
    hit = _PROBE.get(enc)
    if hit is not None:
        return hit
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           # 256x256：NVENC 的最小尺寸比这小（64x64 会 "Frame Dimension less than the
           # minimum supported value"），探针自己不能先死在尺寸上
           "-f", "lavfi", "-i", "color=c=gray:s=256x256:d=0.04",
           "-frames:v", "1", "-c:v", enc, "-f", "null", "NUL"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        ok, err = (r.returncode == 0), r.stderr.decode("utf-8", "replace")[-400:]
    except Exception as e:                                    # noqa: BLE001
        ok, err = False, "%r" % (e,)
    _PROBE[enc] = (ok, err)
    return ok, err


def effective_mode(w=None, h=None) -> str:
    """解析 auto 之后**实际要跑**的编码器名（日志 / [eff] / accept 用）。

    w/h 未知时 auto 只能走 x264：判不出是不是 4K，就别替用户换编码器（换 md5 的代价
    比"没赶上 2.7 倍"大）。
    """
    want = mode()
    if want != "auto":
        return want
    # auto 的规则（2026-11 改）：**有 NVENC 就用 NVENC**，没有才落 x264。
    # 依据（同一素材 150 帧、同进程交替）：
    #   1080p：x264 20.22 帧/秒 → nvenc 23.44（+16%，且把 3.75 个核还给 decode/detect/compose）
    #   4K   ：x264 17.0 → nvenc 48.1（2.8 倍，x264 会吃掉 15/16 核）
    #   画质：vs 原始帧 PSNR x264 43.42 / h264-CQ19 43.67（NVENC 反而略高）；体积 CQ21 ≈ crf16（−5%）
    # 代价：成片 md5 变（换编码器必然），所以拿 md5 当基线的验收都把 DFL_FUSED_ENCODER=x264 钉死了。
    ok, _err = nvenc_probe("h264_nvenc")
    if ok:
        return "nvenc-h264"
    return "x264"


def encoder_args(w=None, h=None) -> list:
    """ffmpeg 的 -c:v 那一段。x264 分支 = 4.1 的原样，一个字节都不差。

    NVENC 不可用时**在这里**抛 _BadInput（调用点都在 Popen 之前），此时还没起任何进程。
    """
    want = effective_mode(w, h)
    if want == "x264":
        return list(X264_ARGS)
    enc = _NVENC[want]
    ok, err = nvenc_probe(enc)
    if not ok:
        raise _BadInput("NVENC 不可用（%s）：%s —— 取消 %s，或换一块支持 NVENC 的卡 / 更新驱动"
                        % (enc, (err or "?")[-200:], ENCODER_ENV))
    args = ["-c:v", enc, "-preset", "p5", "-tune", "hq", "-rc", "vbr",
            "-cq", str(cq()), "-b:v", "0"]
    if enc == "hevc_nvenc":
        args += ["-tag:v", "hvc1"]           # Apple 只认 hvc1；ffmpeg 默认打 hev1
    return args


def warn_text(w=None, h=None) -> str:
    """4K + auto（= x264）时那条人话告警；其余情况返回 ""。

    只告警不自动切的三条理由写在模块头（纪律 2）。文案三件事必须齐：**当前跑什么、
    代价多大、怎么改**（同 eff 那套 [eff-warn] 的口径，见 knobs.skip_text）。
    """
    px = _px(w, h)
    if px < NVENC_MIN_PX or mode() != "auto":
        return ""
    # 默认已经优先 NVENC；还落到 x264 只可能是这块卡没有 NVENC → 这才是要喊的情况。
    ok, _err = nvenc_probe("h264_nvenc")
    if ok:
        return ""
    return ("大分辨率素材（%dx%d = %.1f MP）但这块卡没有可用的 NVENC：退回 libx264 crf16，"
            "实测 4K 下 59 ms/帧、吃掉整机 %s 核，检测/dfm/合成只剩 1 核可用；"
            "换一块支持 NVENC 的卡或更新驱动可恢复（约 3 倍）"
            % (int(w or 0), int(h or 0), px / 1e6, _CORES_X264_4K))


def warn_patch(w=None, h=None):
    """告警的结构化版本（塞 status.eff_warnings[]，进度页顶部横幅读它）；没告警返回 None。"""
    text = warn_text(w, h)
    if not text:
        return None
    return {"code": "enc-4k", "field": "encoder", "level": "warn", "text": text}
