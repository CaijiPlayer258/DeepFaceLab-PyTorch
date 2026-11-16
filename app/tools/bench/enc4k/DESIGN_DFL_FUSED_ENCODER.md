# DFL_FUSED_ENCODER 落地设计（未改代码，只给补丁方案）

依据：`REPORT_4K_encoder.md`（同目录）（4K x264 = 59 ms/帧、15/16 核；nvenc-h264 CQ21 = 22 ms/帧、5.3 核、体积 +6%）。

## 1. 开关口径

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `DFL_FUSED_ENCODER` | `auto` / `x264` / `nvenc-h264` / `nvenc-hevc` | `auto` | `auto` = **现状 x264**，4K 只告警 |
| `DFL_FUSED_ENCODER_CQ` | 15~30 的整数 | `21` | 只对 nvenc 生效；21 是「体积≈crf16」的点 |
| `DFL_FUSED_ENCODER_AUTO4K` | `0`/`1` | `0` | 允许 auto 在 ≥6 MP 时真自动切；默认关 |

为什么 auto 不自动切：
- 换编码器 **改 md5**，`app/tools/accept/{fused_export_accept,parity_raw_png}.py` 的成片对拍会红；
- 改体积（CQ 档）与画质曲线，用户以为"只是变快"；
- hevc 有兼容面（`hev1` / H5 黑屏）。

所以 4K 只**告警一次**（日志 + 导出响应里的 `warnings`），文案带上实测数字，让用户自己决定。

## 2. 代码落点（按锚点，不按行号 —— 仓库有并发改动）

### (a) 新函数（放在 `pipeline_fused.py::_fused_pipe_cmd` 之前）

~~~python
_FUSED_ENCODER_ENV = "DFL_FUSED_ENCODER"          # auto | x264 | nvenc-h264 | nvenc-hevc
_FUSED_CQ_ENV      = "DFL_FUSED_ENCODER_CQ"       # nvenc 的 CQ，默认 21
_FUSED_AUTO4K_ENV  = "DFL_FUSED_ENCODER_AUTO4K"   # auto 在 >=6MP 时是否真自动切，默认 0
NVENC_MIN_PX = 6_000_000                          # 2160x3840 / 3840x2160 都 8.29 MP
_NVENC_NAMES = {"nvenc-h264": "h264_nvenc", "nvenc-hevc": "hevc_nvenc"}


def _nvenc_probe(enc: str):
    """250ms 预检：编码器在不在、驱动认不认。返回 (ok, stderr 尾巴)。"""
    r = subprocess.run(
        [str(_app_mod._ffmpeg_exe()), "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=gray:s=256x256:d=0.04", "-frames:v", "1", "-c:v", enc,
         "-f", "null", "NUL"], capture_output=True, timeout=20)
    return r.returncode == 0, r.stderr.decode("utf-8", "replace")[-400:]


def _fused_encoder_args(w, h, warn=None) -> list:
    """-c:v 那一段。x264 分支必须和 _vid_cmd / 历史成片逐字节一致。"""
    want = (os.environ.get(_FUSED_ENCODER_ENV) or "auto").strip().lower()
    px = int(w) * int(h)
    if want == "auto":
        big = px >= NVENC_MIN_PX
        if big and warn is not None:
            warn("4K 导出（%dx%d）：ffmpeg 走 libx264 crf16，实测 59ms/帧、占满 15/16 核。"
                 "换 NVENC 可到 22ms/帧（约 2.7x），但成片 md5/体积/画质都会变："
                 "DFL_FUSED_ENCODER=nvenc-h264（CQ 默认 21，体积 +6%%）/ nvenc-hevc"
                 "（体积 -31%%，需要播放器支持 HEVC）" % (w, h))
        if big and str(os.environ.get(_FUSED_AUTO4K_ENV) or "0").strip().lower() in ("1", "on", "true"):
            want = "nvenc-h264"
        else:
            want = "x264"
    if want == "x264":
        return ["-c:v", "libx264", "-crf", "16"]
    enc = _NVENC_NAMES.get(want)
    if enc is None:
        raise _BadInput("「DFL_FUSED_ENCODER」只认 auto / x264 / nvenc-h264 / nvenc-hevc，"
                        "收到 %r" % (want,))
    cq = str(int(os.environ.get(_FUSED_CQ_ENV) or 21))
    args = ["-c:v", enc, "-preset", "p5", "-tune", "hq", "-rc", "vbr",
            "-cq", cq, "-b:v", "0"]
    if enc == "hevc_nvenc":
        args += ["-tag:v", "hvc1"]          # Apple/QuickTime 只认 hvc1，不是 hev1
    ok, err = _nvenc_probe(enc)             # 250ms，只在真的要用时探
    if not ok:
        raise _BadInput("NVENC 不可用（%s）：%s —— 关掉 %s 或换显卡/驱动"
                        % (enc, err[-200:], _FUSED_ENCODER_ENV))
    return args
~~~

### (b) 接进两条路径（**必须两条一起改**）

1. `app/server/backend_merge/pipeline_fused.py::_fused_pipe_cmd`
   把硬编码的 `"-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",`
   换成 `*_fused_encoder_args(w, h, warn=_enc_warn), "-pix_fmt", "yuv420p",`
   （`w,h` 本来就是入参；`_enc_warn` 由调用点注入）

2. `app/server/_media/encoder.py::_vid_cmd`（PNG 落盘 / `DFL_FUSED_RAWPIPE=0` 那条路）
   同样换掉 `"-c:v", "libx264", "-crf", "16",`。
   **不改它就会破掉「两条路成片 md5 相同」这条既有不变量**（`parity_raw_png.py` 正在测它）。

### (c) 告警出口

- `pipeline_fused.py` 里 `raw_cmd = _fused_pipe_cmd(part, video, fps, _pw, _ph, ...)` 那处注入 warn 回调：先 `logger.warning`，再把文案塞进这一单的 `warnings`（`routes_export.py` 已把导出结果回给前端，加一条即可弹 toast）。
- 一单只告警一次，别按帧刷。

## 3. accept / 回归怎么改

- `fused_export_accept.py`、`parity_raw_png.py`、`stage_export_accept.py`：**显式** `os.environ["DFL_FUSED_ENCODER"]="x264"`，md5 基线不动。
- 新增 `enc_switch_accept.py`：
  1. `x264` 两轮 md5 相同（现状回归）；
  2. `nvenc-h264` 两轮 md5 相同（实测成立：本机/本驱动确定性）—— 断言**稳定性**，不是等于 x264；
  3. `nvenc-h264` 与 `x264` md5 **必须不同**（防止开关没生效却静默通过）；
  4. 4K 下 `auto` 不切、且 `warnings` 里有那行告警；
  5. 无 NVENC 的机器上 `DFL_FUSED_ENCODER=nvenc-*`：预检 250ms 内抛 `_BadInput`，不留半截 mp4。

## 4. 还没量、值得下一步做的

- `-profile:v high`：nvenc 默认 Main，x264 是 High（8x8 变换），可能吃掉一部分 +36% 体积；
- `-spatial_aq 1 -aq-strength 8`、`-preset p7`、`-multipass fullres`：给 4K 找 CQ21 以下更省的位置；
- 4K 真正的地板是管道本身（15.7 ms/帧 = 24.9 MB/帧 过匿名管道 + swscale）。要继续提速得动管道（共享内存 / 让 nvenc 直接吃 GPU 帧），不是换编码器。
