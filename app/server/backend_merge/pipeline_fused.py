# -*- coding: utf-8 -*-
"""融合流水线：解码 / 检测 / 换脸 / 合成 / 编码串成一条带队列的线。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations
import numpy as np

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from .._common.env import (
    _env_int,
)
from .._common.http import (
    _BadInput, _as_dict, _bool_val, _float_val, _int_val,
)
from .._common.proc import (
    _kill_proc_tree, _proc_mem_mb,
)
from ._shared import (
    _PRED_BATCH_CAP, _fmt_mb, _norm_path, _same_path, _settings_val, _trim_ws,
)


# ============================================================
# 视频模式 · 快速导出（融合流水线 fused）
# ------------------------------------------------------------
# 口径（用户定的，实测见 app/tools/bench/fused_pipeline_bench.py，同段 1080p 120 帧）：
#   分段六阶段 7.20 帧/秒 · 六阶段合一**串行** 6.10（0.85x）· 合一**流水线** 20.55（2.85x）。
# 这份生产实现的实测（app/tools/accept/fused_export_accept.py，同一个样片 1080x1920 120 帧）：
#   17.4~18.2 帧/秒（分段 7.20 → ≈2.5x）· quality=fast 20.9 帧/秒。
#   差的那一点全在检测器：retinaface_10g 单帧 47~50ms = 约 20 帧/秒的硬底（它跑在
#   onnxruntime 上，workspace/_engines 里的 retinaface TRT 引擎 app 层没接上）。
# 所以视频模式默认走流水线（fused=1）；分段链保留成 fused=0 的对照/排障路径。
# DFL 项目导出（/api/merge/export-project，iperov 语义、单模型、按 original_name 对帧）
# 一个字都没动。
#
# 这里**不是第二套算法**，每一段调的都是现成的那一个函数：
#   检测 _detect_boxes · 检测框→68 点→对齐 _align.*（和 ② 切脸同一串）·
#   归组 _auto_face_plan（经 _vid_models_for_frame）· dfm _dfm_run_batch（session 走 _dfm_session、
#   批大小走 _pred_pick_batch）· XSeg _stage_xseg().mask_batch · 合成 _merge_compose.compose_frame ·
#   编码 _vid_cmd/_vid_encode（和 ⑥ 导出同一条命令）。
# 差别只在**数据怎么走**：分段是「写出 PNG → 下一段读回来」，fused 是一帧的内存对象
# 从检测一路带到合成，中间不落一个文件 —— 所以 <视频>_cache/ 一个都不建（_fused_run 里断言）。
#
# 像素为什么能和分段对得上（每一条都是"照抄"，不是"差不多"）：
#   · 68 点/对齐：coarse_align_from_5 → 2D106 → landmarks68_in_source → align_face_68，
#     和 ② 切脸同一串；compose 用的 source_landmarks 就是 align_face_68 里那份 round(4)
#     的结果 —— DFLJPG 里存的也是它，所以 fused 和分段手里的 landmarks 逐位相同。
#   · dfm 输入：get_transform_mat(lm, S, WHOLE_FACE) + INTER_CUBIC warp → /255，
#     和 ③ 换脸同序（**uint8 整帧 warp 再 /255**，不是先转 float）。
#   · dfm 输出：np.clip(face*255).astype(uint8) 之后**才** /255 —— 就是 PNG 往返那一步，
#     只是不落盘；XSeg 的输入/输出同理（先 u8 再喂，遮罩也是 u8 量化后 /255）。
#   · dst 遮罩输入：分段的输入是 aligned/*.jpg（q100 4:4:4，仍然是有损的），所以这里也做
#     一次同样的 JPEG 往返再映射到 256 —— 省掉它 dst 遮罩就会差一点点。
#   · 合成：compose_frame(原帧, faces, 同一份 cfg)，连 mask_mode 都取自同一处。
#
# 不支持续跑：帧是一次性的内存对象，没有游标可存。取消 = 整段作废（临时帧当场删掉），
# 进度页会说清楚「快速导出不支持续跑，需整段重跑」；要续跑就切回分段（fused=0）。
# ============================================================
_FUSED_BUDGETS = (30, 16, 8)        # 帧缓冲预算阶梯（用户口径：30/16/8 自动降档）
_FUSED_QDEPTH = 24                  # 段间有界队列深度（原型同值；>= dfm batch 才凑得满）
_FUSED_RESERVE_MB = 1024            # 模型 + CUDA context 的固定占用（416 dfm 实测 ≈1GB）
_FUSED_VRAM_SAFETY = 0.80           # 探测量打八折才是敢用的预算
_FUSED_ENV_VRAM = "DFL_FUSED_VRAM_FREE_MB"     # 验收/排障：假装可用显存只有这么多
_FUSED_ENV_BUDGET = "DFL_FUSED_BUDGET"         # 验收/排障：钉住帧预算（不走显存探测）
_FUSED_ENV_DUMP = "DFL_FUSED_DUMP"             # 验收/排障：把这几帧的逐脸输入 dump 出来
# 为什么要这个开关：「fused 和分段像素一致」这件事只能靠**同样的输入**证明，光看成片
# 分不清差在 pred / 遮罩 / 几何 / 编码哪一层。DFL_FUSED_DUMP=41,80 时把那几帧的原帧、
# 每张脸的 pred / prd / dst / xseg_prd / xseg_dst / 68 点全写成文件（<成片>.dump/），
# 直接和分段 cache/ 里的同名产物逐位比。默认关（一个文件都不写）。

# ---- 攒批与线程数：B 配置的口径（桌面实测见 _attr_bench/REPORT.md 第 8 节）----
# 攒批目标：dfm 一次前向喂几张。这条链原来只是「pend 攒满 _pred_pick_batch 就送」，
# 而那个阶梯只看**空张量单发**的每张耗时（416：B1=30.0 / B4=33.8 / B8=31.5），
# 于是 416 被选到 4、更小的模型会被选到 1 —— 判断依据错了：进流水线之后真正的对手是
# 「GPU 空转等批」，不是单张能跑多快。这里另给一个显式目标（默认 8 = B 配置），
# 实际取 max(阶梯值, 目标)，再按 session 原生上限 与 _PRED_BATCH_CAP 截。
# 像素不受它影响：同一个 session、同一份权重、同一批算子，只是一次喂进去的**张数**不同
# （对拍证据见 _attr_bench/fused_ab_*.json 与 dump 对拍）。
_FUSED_BATCH_TARGET = max(1, min(int(_PRED_BATCH_CAP), _env_int("DFL_FUSED_BATCH", 8)))
# 攒批超时（ms）：从**这一批第一张脸进来**的时刻起算；超时就把手里这点先送出去。
# 为什么必须有：片子尾巴、以及多模型时「某一路的脸一直凑不满目标」这两种情况下，
# 已经进来的脸会一直压着不动 —— 那一帧也就一直进不了合成。0 = 关（一直攒到目标或 END）。
# 默认 20ms —— **这是量出来的，不是拍的**（_attr_bench/fused_ab_wait*.json，90 帧 ×3 次中位）：
#   超时 0（=一直攒满 8）15.62 帧/秒 · 250ms 15.82 · **20ms 16.79**（384 模型，同段同参）。
# 为什么「超时越短反而越快」：384 dfm 单张 32~38ms，而这条线的硬底是检测（retinaface_10g
# 50~55ms/帧），dfm 早就不在关键路径上了；batch 8 的代价是 GPU 段每 ~400ms 才动一次，
# 一次憋出 8 帧，把下游 compose/encode 从「连续小批」变成「一阵一阵」，实测 compose 的
# 墙钟从 108~121ms 拉到 126~161ms —— 吞吐反而掉。改成 20ms 之后平均批次 1~2 张，
# dfm 每张贵 3~4ms，但整条线多 10%。**所以「攒到 8」是上限，不是目标**：
# 上游真能喂满（>400 张脸/秒）时它照样攒满 8；平时它就是一条不憋气的流水线。
_FUSED_BATCH_WAIT_MS = max(0, _env_int("DFL_FUSED_BATCH_WAIT_MS", 20))
# 攒批超时改「自适应」：DFL_FUSED_BATCH_ADAPT（**默认 0 = 关，就是上面那版「一律等满超时」**；
# =1 打开下面这套规则。为什么默认关见本段末尾那组实测）。
# 为什么：20ms 超时是**延迟税**，不是攒批收益 —— 单脸/帧时下一张脸要 59ms 才来
# （17.06 帧/秒），等满 20ms 手里还是 1 张。416 单张 dfm 39.9ms + xseg，每张 41ms 干活
# + 20ms 空等 = 61ms/张 → 16.4 帧/秒，正好压在生产的 17.06 上；GPU 那 27~31% 的空闲
# 就是它（profile 里 gpu_wait/段空闲 11~16% 只是其中一段，其余摊在"批没到"上）。
# 规则（**只在 q_gpu 已经空了那一拍判**，绝不从队列里抢批，否则一批会被拆成两批）：
#   ① 投影攒不满：现有张数 + 到达率 × 剩余等待时间 < 目标 → 立刻送（why=starve）
#   ② GPU 段本来就空着（空闲 >20%，且已跑满 0.5s / 4 次调用）→ 立刻送（why=idle）
#   ③ 到点（why=timeout）—— 只有「看起来真能攒满」的时候才会等到这里
# 投影的到达率是**定窗**的（最近 1 秒来了几张），不是「张数/跨度」：一帧带 3 张脸时
# 跨度会被压成 0，算出来的到达率是假的。上游真能喂满（≥ 目标/窗口）时它照样会等满。
#
# ★ 默认 **0（关）** —— 实测在真素材上收益 ≈0，见下面这组数（150 帧 · 1080x1920 · 416 · tinymog，
#   4 配置 × 4 轮交错，中位）：
#     base (固定20ms)      18.62 帧/秒 · compose 107.3ms/帧 · 平均等批 1.2~3.2ms · 送出 {full 16, timeout 1}
#     adapt                18.58 帧/秒 · compose 109.2ms/帧 · 平均等批 0.5~1.0ms · 送出 {full 16, starve 1}
#   为什么这条在真素材上不成立：**q_gpu（深度 64）本身就是攒批器**。GPU 忙的时候脸在队列里
#   排着，pend 一取就是 7~8 张（实测批均值 7.4/8），根本走不到「等超时」那条路；走到的只有
#   17~19 批里的 1 批。而 GPU 段的空闲 30% 里，真正「等脸」只有 6%（profile 的 gpu_wait 3ms/帧），
#   剩下是 numpy 打包/量化（3.35ms/张 = 6.2% 帧预算，实测见 _cv2batch_ab/gpu_pack_cost.py）
#   和 _put 的开销 —— 那些不在这 20ms 里。
#   所以：留着实现与开关（want=1 时规则见下），**默认关**；要开就 DFL_FUSED_BATCH_ADAPT=1。
_FUSED_ENV_ADAPT = "DFL_FUSED_BATCH_ADAPT"
_FUSED_BATCH_ADAPT = str(os.environ.get(_FUSED_ENV_ADAPT) or "0").strip().lower() \
    not in ("0", "off", "false", "no")
_FUSED_ADAPT_IDLE_PCT = 20.0        # ② 的门槛：GPU 段空闲超过这个百分比就不再攒批
# cv2 内部线程数（进程级）：DFL_CV2_THREADS。**默认 0 = 这一单不去动它的设置**（OpenCV 自己的默认）。
# 0/1/4 三档都量过（孤立 compose 基准 vs 整条线，见 _fused_run 里 cv2_threads 那一段）：
# 孤立基准上「少给 cv2 线程」确实更好（4 → 1：容量 22.7 → 25.3，核 8.03 → 7.45；不设=26.6/7.38），
# 但**整条线**上 4 和 1 打平（18.62 vs 18.18，噪声内）—— 局部更快 ≠ 整条更快，所以默认不动。
_FUSED_ENV_CV2 = "DFL_CV2_THREADS"
_FUSED_CV2_DEFAULT = 0
# 对齐/预处理线程 与 合成线程（B 配置：4 / 8）。以前对齐是 min(4, cpu//3) —— 核少的机器
# 会自己掉到 1~2，同一个产品在两台机器上跑出两种口径。现在钉成常量，要改走环境变量。
_FUSED_PREP_DEFAULT = max(1, min(16, _env_int("DFL_FUSED_PREP", 4)))
_FUSED_CMP_DEFAULT = max(1, min(32, _env_int("DFL_FUSED_CMP", 8)))
# u8 量化的写法（DFL_FUSED_U8，**默认 1**）：
#   0 = 老口径 np.clip(x * 255.0, 0, 255).astype(np.uint8)，每张脸连着开三个 float32 临时量
#       （416×416×3 = 2MB 一个），全是刚 mmap 出来的新页 —— 第一次写就缺页，这部分比算术本身贵；
#   1 = 复用一条 float32 暂存（out= 原位三连）+ astype，只有最后的 uint8 是新的（1/4 体量）。
# 逐位相同（对拍见 experiments/enc_quant_diag/bench_quant.py：真实帧分布 + k/255 边界 + 越界样本，
# max=0），因为乘的是同一个 255.0、clip 只截两端、astype 就是 C 的截断。0 留着是为了随时退回老口径对照。
_FUSED_ENV_U8 = "DFL_FUSED_U8"
_FUSED_U8_DEFAULT = 1
# 「clip 恒等」这条路只对 dfm 的 face 输出成立：_dfm_run_batch 出参前已经 np.clip(face, 0, 1)，
# 再乘 255 的上限正好是 255.0（1.0×255 是精确值），所以那道 clip 一个像素都改不动。遮罩（_mask2d
# 不 clip）和 XSeg 输出想都别想，照旧 clip。


_U8_SCRATCH = {}


def _fused_q_u8(x, fast: int, clip: bool = True):
    """float 0..1 → uint8，**逐位等于** np.clip(x * 255.0, 0, 255).astype(np.uint8)。

    fast=0 就是那句话本身（老口径）。fast=1 只改「临时量从哪来」：复用一条按形状缓存的
    float32 暂存（out= 原位乘 → 原位 clip），再 astype 出 uint8 —— 算术顺序、dtype、
    舍入点一个都没动，省掉的是每帧两三个 2~4MB 的新页分配与缺页。

    clip=False 只对 **dfm 的 face 输出** 合法（见 _FUSED_U8_DEFAULT 上方那段）：
    _dfm_run_batch 已经 clip(0,1)，×255 上限正好 255.0 —— 恒等。别的地方传 False 就是改像素。

    只给 GPU 那一个线程用（_gpu()），所以缓存不用加锁；返回的 uint8 是**新数组**，
    不共享缓冲（这些数组要跟着 face 一路活到合成，共享会被下一张脸覆盖）。
    """
    if not fast or x.dtype != np.float32:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    buf = _U8_SCRATCH.get(x.shape)
    if buf is None:
        buf = np.empty(x.shape, np.float32)
        _U8_SCRATCH[x.shape] = buf
    np.multiply(x, 255.0, out=buf)
    if clip:
        np.clip(buf, 0.0, 255.0, out=buf)
    return buf.astype(np.uint8)


# ============================================================
# 成片怎么出去：rawvideo 管道（默认）还是 PNG 落盘
# ------------------------------------------------------------
# 老路径（DFL_FUSED_RAWPIPE=0）：编码 3 线程 cv2.imencode(level1+RLE) + 写 %TEMP%/*.png，
#   流水线跑完再让 ffmpeg 读这串 PNG 编成 mp4。实测（150 帧 · 1080×1920 · 416）：
#   imencode 92.3ms/帧（车道 CPU，65.7% 占用）+ 写盘 7.45，尾部 stage7 再串行 1.68s（11.2ms/帧）。
# 管道（默认）：每帧不做编码、也不落盘，单写者按帧号把 memoryview 塞进 ffmpeg stdin，
#   生产端 4.6ms/帧（tobytes 1.15 + write 3.42），车道占用掉到 ~3%，尾部只剩 x264 排空
#   0.76s（5.0ms/帧）—— 因为 x264 单独跑 22.2ms/帧 < 47ms 的生产节拍，它整个被流水线盖住了。
# 端到端：7.04s → 6.12s（21.3 → 24.5 帧/秒）。成片**逐位相同**（md5 相同，见 _fused_pipe_cmd）。
_FUSED_ENV_RAW = "DFL_FUSED_RAWPIPE"


def _fused_rawpipe_on() -> bool:
    """这一单走不走 raw 管道。默认走；=0 回老 PNG 落盘路（排障/A-B 用，两条都要能跑通）。"""
    return str(os.environ.get(_FUSED_ENV_RAW) or "1").strip().lower() \
        not in ("0", "off", "false", "no")


def _fused_pipe_cmd(part, video, fps, w, h, audio_ss=None, audio_t=None, limit=None) -> list:
    """rawvideo 版 _vid_cmd：同一个滤镜、同一个编码器、同一段音轨，只是帧从 pipe:0 来。

    三条口径必须和 _vid_cmd（app/server/_media/encoder.py）**一字不差**，否则成片就和老路径
    不一样了：crf16 + yuv420p + scale=out_range=tv；音轨 -c:a copy、**不带 -shortest**
    （带上会按较短的一边截断，帧范围的片子就少一截音）；奇数宽高先 pad 一像素。
    这里复制一份而不是参数化 _vid_cmd，是因为那个吃的是「帧序列文件模式」；改它的时候两边一起看，
    有 _accept 的成片 md5 对拍兜着。

    **format=rgb24 不能省**：rawvideo 声明 bgr24，而 scale 在 bgr24 输入下出来的像素和 PNG
    路径（png 解码器出的是 rgb24）**不是**逐位相同 —— 实测 12/12 帧 md5 全不同。中间插一层
    format=rgb24（ffmpeg 自己换通道，在它自己的线程里，约 1ms/帧）之后，成片文件 md5 与老路径
    **完全相同**（150 帧实测 8af9c0e8f81c / 321466B）。

    也**不给 -progress pipe:1**：那条进度走 stdout，没人读就会把 ffmpeg 堵在管道上；
    进度由写者线程逐帧上报（比解析 ffmpeg 的 frame= 更准，而且它本来就知道写到了第几帧）。

    **limit（-frames:v N）必须和 _vid_cmd 一样给**——这一条不是洁癖：实测（30 帧范围导出）
    不加它时音轨会多出**一个** AAC 包（43 → 44，成片 md5 因此不同、时长 1.00 → 1.02s）。
    原因：老路径有 -frames:v，视频一满 30 帧输出就收尾，那个还没来得及 mux 的音频包被丢掉；
    管道这条路没有 -frames:v，输出要等到 stdin EOF，音频已经把第 44 包 mux 进去了。
    加上之后两条路的包序列**逐字节相同**（30 视频包 + 43 音频包，包 md5 全等）。
    """
    pre_aud = []
    if audio_ss is not None:
        pre_aud += ["-ss", "%.6f" % float(audio_ss)]
    if audio_t is not None:
        pre_aud += ["-t", "%.6f" % float(audio_t)]
    cmd = [str(_app_mod._ffmpeg_exe()), "-y", "-hide_banner", "-loglevel", "error",
           "-nostats", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (int(w), int(h)),
           "-framerate", "%.6g" % float(fps), "-i", "pipe:0"] + pre_aud + \
          ["-i", str(video),
           "-map", "0:v:0", "-map", "1:a?",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=rgb24,scale=out_range=tv",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "copy"]
    if Path(str(part)).suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    if limit:
        cmd += ["-frames:v", str(int(limit))]
    cmd += [str(part)]
    return cmd


class _FusedRawWriter(threading.Thread):
    """rawvideo 管道的**唯一写者**：按帧号严格递增把帧塞进 ffmpeg 的 stdin。

    为什么必须单写者：编码有 3 条线程（乱序完成），而管道是**有序流** —— 谁先干完谁写，
    成片就串帧，而串帧肉眼几乎看不出来（几十毫秒的错位在 30fps 里像「抖了一下」），
    所以这里宁可断言也不静默：拿到的不是「下一个该写的帧号」就整单失败。

    为什么必须有一条 stderr 排空线程：ffmpeg 的 stderr 是管道，我们不读、它写满就**阻塞**；
    它一阻塞就不读 stdin，于是我们 write 也阻塞 —— 整条流水线抱死在两个管道上（很好复现）。

    背压：write 阻塞 = ffmpeg 比我们慢，那就是真的慢，等它，绝不丢帧（丢帧 = 成片少帧）。
    ffmpeg 提前退出（-vf 写坏 / 编码器不可用 / 磁盘满）：write 抛 BrokenPipeError →
    立刻停整条线，并把 stderr 尾巴交给调用方 → 用户拿到的是「为什么没出来」，不是一句「0 帧」。
    """

    def __init__(self, proc, total, sem, live, live_lock, ticks, stop, on_frame, on_progress):
        super().__init__(name="fused-raw", daemon=True)
        self.proc = proc
        self.total = int(total)
        self.sem = sem
        self.live = live
        self.live_lock = live_lock
        self.ticks = ticks
        self.stop = stop                 # 只用来让「上游」别再喂；写者自己照常把手里写完
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.cv = threading.Condition()
        self.store = {}
        self.expect = 1
        self.closing = False             # 上游说「不会再有帧了」
        # ★ 乱序缓冲**不设上限**：整条流水线的在飞帧数已经被帧预算（sem，默认 30）卡住了，
        #   这里再压一层「只看当前手里有多少帧」的小上限会**死锁**（实测：全片导出跑到第 140 帧
        #   时 store 里是 141..148、而 140 还排在 FIFO 的 q_enc 里，3 条编码线程全卡在 put 上，
        #   谁也取不到 140 → 整条线不动、主线程等线程退出等到天荒地老）。
        #   上界由 sem 保证：一帧从解码进来到写进管道才 release，所以 store ≤ 预算帧数（30×6.2MB）。
        self.depth = 0
        self.n = 0                       # 已写帧数
        self.tail = []
        self.rc = None
        self.err = ""
        self.seconds = 0.0
        self.write_max = 0.0
        self._dead = False

    # 上游：编码线程（可并发）
    def put(self, job, buf) -> bool:
        seq = int(job.n)
        with self.cv:
            while (self.depth > 0 and len(self.store) >= self.depth
                   and not self._dead and not self.closing):
                if self.stop.is_set():
                    return False
                self.cv.wait(0.05)
            if self._dead or self.stop.is_set():
                return False
            if seq in self.store:
                self.err = "同一帧被写了两遍（第 %d 帧）—— 帧号重复" % seq
                self._dead = True
                self.stop.set()
                return False
            if seq < self.expect:
                self.err = "第 %d 帧已经写出去了，又收到一次（帧号回退）" % seq
                self._dead = True
                self.stop.set()
                return False
            self.store[seq] = (job, buf)
            self.cv.notify_all()
            return True

    def start(self):                     # noqa: D102 - 先起 stderr 排空线程，再起写者
        threading.Thread(target=self._drain, name="fused-raw-err", daemon=True).start()
        super().start()

    def _drain(self):
        try:
            for ln in iter(self.proc.stderr.readline, b""):
                s = ln.decode("utf-8", "replace").rstrip()
                if s:
                    self.tail.append(s)
                    if len(self.tail) > 60:
                        del self.tail[:20]
        except Exception:                                  # noqa: BLE001
            pass

    def run(self):
        t0 = time.perf_counter()
        try:
            while True:
                with self.cv:
                    while self.expect not in self.store and not self.closing:
                        self.cv.wait(0.1)
                    item = self.store.pop(self.expect, None)
                    if item is None:
                        # closing 且下一帧不在手里：收尾（正常 = 全写完；异常 = 中间有洞）
                        if not self.store:
                            return
                        self.err = self.err or (
                            "帧序有洞：等第 %d 帧，手里最小的是第 %d 帧"
                            % (self.expect, min(self.store)))
                        self._dead = True
                        return
                job, buf = item
                if int(job.n) != self.expect:              # 双保险：写之前断言 seq 递增
                    self.err = "帧序错位：期望第 %d 帧，拿到第 %d 帧" % (self.expect, int(job.n))
                    self._dead = True
                    self.stop.set()
                    return
                t = time.perf_counter()
                try:
                    self.proc.stdin.write(buf)
                except (BrokenPipeError, ConnectionResetError, ValueError, OSError) as e:
                    self.err = ("ffmpeg 提前退出（第 %d 帧写不进去）：%s"
                                % (self.expect, str(e)[:160]))
                    self._dead = True
                    self.stop.set()
                    return
                dt = time.perf_counter() - t
                self.write_max = max(self.write_max, dt)
                self.ticks.add("write", dt)
                self.expect += 1
                self.n += 1
                try:
                    self.sem.release()
                except ValueError:
                    pass
                with self.live_lock:
                    self.live["n"] -= 1
                self.on_frame(job)
                self.on_progress(self.n)
        finally:
            self.seconds = time.perf_counter() - t0
            self._dead = True

    def finish(self, force: bool = False):
        """收尾：让写者把手里写完 → 关 stdin（ffmpeg 见 EOF 才会把最后一批编完）→ 等它退出。

        force=True（失败/取消）：不再等它排空，直接杀掉 —— 半截成片由调用方删。
        """
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        self.join(timeout=5.0 if force else 300.0)
        stuck = self.is_alive()
        # 关 stdin = 给 ffmpeg 一个 EOF：它才会把 lookahead 里剩下的帧编完（排空）。
        # 写者要是还卡在 write 里，这一下会让它拿到「write to closed file」并退出 ——
        # 比在这里干等一个可能永远不来的帧安全。
        try:
            self.proc.stdin.close()
        except Exception:                                  # noqa: BLE001
            pass
        if force or stuck:
            try:
                _kill_proc_tree(self.proc)
            except Exception:                              # noqa: BLE001
                pass
        try:
            self.rc = int(self.proc.wait(timeout=30 if (force or stuck) else 600))
        except Exception:                                  # noqa: BLE001
            try:
                _kill_proc_tree(self.proc)
            except Exception:                              # noqa: BLE001
                pass
            self.rc = -1
        return self.rc

    def tail_text(self, n: int = 12) -> str:
        return "\n".join((self.tail or [])[-int(n):])


def _fused_det_plan(det_name, w, h, over=None) -> dict:
    """检测池的配比：探测 → 微基准 → derive → EP 政策。over = 请求体覆盖（sess_gpu/sess_cpu/…）。"""
    b = _detect_bench(det_name, w, h)
    pol = _det_ep_resolve(det_name, w, h)
    gp = _sess.probe(*(_vram_free_mb()[:2]))
    # 验收/复现用：把「机器档位」换成低配/中配/高配的**事实**（不改真探测，
    # 只影响这一次推导 —— 生产路径上没人设 DFL_SESS_LEVEL，一行都不走）。
    lv = str((over or {}).get("level") or os.environ.get("DFL_SESS_LEVEL")
             or "").strip().lower()
    if lv:
        gp.update(_sess.level_cfg(lv))
        gp["level"] = lv
    # 检测器很小（retinaface 22 MB / tinymog 5 MB），一份 session 的显存 ≈ 权重 + 300MB
    # workspace/arena（日志里那个「检测/遮罩 300 MB/份」就是它）。
    mem_gpu = float(_vram_model_mb(det_name if str(det_name).endswith(".onnx")
                                   else "") or 0.0) + 300.0
    plan = _sess.derive("detect", gp, b.get("gpu_ms"), b.get("cpu_ms"),
                        mem_per_gpu_mb=mem_gpu, mem_per_cpu_mb=260.0,
                        stage_busy=_SESS_BUSY.get("detect"), over=over)
    plan["bench"] = b
    _det_ep_apply(plan, pol, over)
    return plan


def _fused_on(body) -> bool:
    """这一单走不走融合流水线。**默认走**（fused=1）；fused=0 回退六阶段分段链。"""
    if not isinstance(body, dict):
        return True
    return _bool_val(body.get("fused"), True)


def _fused_frame_mb(size_px: int) -> float:
    """一帧在飞时压在 GPU 上大约多少 MB：dfm 输入 + 输出脸 + prd + dst 四份 float32。

    计算在 _vram_frame_mb（三条路共用），这里只留名字给历史调用点。
    """
    return _vram_frame_mb(size_px)


def _fused_budget(want: int, free_mb, total_mb, size_px: int, model_mb: float = 0.0,
                  batch: int = 0, label: str = "快速导出",
                  resident: bool = False, force: bool = False,
                  reclaim: bool = False, reclaim_first: bool = False) -> dict:
    """fused 的帧预算 —— 现在就是 _vram_plan 的一层壳（老调用方读的键一个都不改）。

    和以前那一版（只按 30/16/8 降帧）的区别只有两点，但都是这次要治的病：
      · 把**模型体量**和 **dfm 批的开销**算进总需求（以前只算帧，模型 2GB 也当没看见）；
      · 连「最小档（8 帧 / 批 1）」都装不下时返回 ok=False —— 以前是压到 8 帧照跑，
        让 OOM 在流水线里炸；现在由调用方明确拒绝（或按 DFL_ALLOW_CPU_FALLBACK 走 CPU）。

    free_mb 传 None = 「按 _vram_probe 自己探」（env 伪造值优先）；调用方已经探过了就传进来。

    resident=True：这份会话**已经在当前进程里加载好了**（_fused_run 加载完 dfm 之后
    拿实测尺寸再判一次那一拍）。那一拍里会话的「权重 + context」已经在卡上，再算一遍
    就是重复计数 —— 384 模型实测「文件 1567MB / 建会话后整卡 +2.3GB」，重复计数会让
    一个本来跑得动的单子被判死（用户截图那一单就是这么被拒的）。
    """
    opts = {"mode": "frames", "size_px": int(size_px or _VRAM_BASE_PX),
            "frames": int(want or 0), "batch": int(batch or 0), "procs": 1,
            "label": label, "resident": bool(resident), "force": bool(force),
            # reclaim_first：先卸掉常驻预览 worker 的模型，再用**回收后**的空闲算账
            # —— 这就是「帧预算不该因为自己的预览 worker 占着而降到 8」那一条。
            "reclaim": bool(reclaim), "reclaim_first": bool(reclaim_first)}
    if free_mb is not None:
        opts["free_mb"] = int(free_mb)
        if total_mb is not None:
            opts["total_mb"] = int(total_mb)
    return _vram_plan(float(model_mb or 0.0), opts)


def _fused_cfg(body: dict) -> dict:
    """请求体 → 快速导出的 cfg。

    和 _stage_cfg 的差别只有一处：**不碰 <视频>_cache/** —— 不建目录、不读不写 meta.json。
    快速导出没有中间产物，也就没有阶段标记/续跑游标，meta 只在内存里当个空壳。
    """
    body = _as_dict(body)
    video = _norm_path(body.get("video") or body.get("source_video"))
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))

    probe = _vid_probe(video)
    rng = _stage_range_norm(body, int(probe.get("frames") or 0))
    settings = _settings_val(body.get("settings"))
    fps = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    dfm, model_name = _stage_dfm_pair(body, need=True)
    xseg = _xseg_of(body)
    out = _stage_out_path(video, rng, body)
    if _same_path(out, video):
        raise _BadInput("输出不能是源视频本身：%s" % out)

    snap = dict(_cache_snap())
    snap.update({"mode": "video", "path": str(video), "dir": "", "names": [], "srcdir": ""})
    library_raw = str(body.get("faces_project") or body.get("face_library")
                      or body.get("library") or "").strip()
    plan_payload = {"project": library_raw, "faces_project": library_raw}
    face_thr = _float_val("face_threshold", body.get("face_threshold"), 0.0, 1.0, None)
    if face_thr is not None:
        plan_payload["face_threshold"] = float(face_thr)

    models = []
    if body.get("models"):
        models = _models_of(body) or []
    if not models and dfm:
        models = [{"name": str(model_name), "dfm": str(dfm), "faces": [], "index": 0}]
    if not models:
        raise _BadInput("没有可用的 .dfm 模型：先在训练页把模型导出成 .dfm（%s）"
                        % (Path(WORKSPACE) / "model"), 409)
    if not model_name:
        model_name = str(models[0].get("name") or "")

    want = body.get("frame_budget")
    if want is None:
        want = _env_int(_FUSED_ENV_BUDGET, 0)
    _det_align, _det_align_src = _align_detector(body)   # 算一次，两个字段共用
    cfg = {
        "video": str(video), "out": str(out), "fps": float(fps or probe.get("fps") or 0.0),
        "probe": probe, "total": int(probe.get("frames") or 0),
        "range": rng, "range_key": str(rng["key"]), "range_total": int(rng["total"]),
        "range_full": bool(rng["full"]), "settings": dict(settings),
        # ★ 两个字段**永不打架**：都取 _align_detector(body)[0]。
        #   以前 cfg["detector"] = _detector_val(body["detector"])（只看 detector 键），
        #   cfg["det_align"] = _align_detector(body)（detector→quality 两级）——
        #   请求体只带 quality=high 时两者是 tinymog / retinaface_10g：回显一个、
        #   实际跑另一个，正是「界面写着 A，日志里是 B」的同一个形状。
        "detector": _det_align,
        "det_align": _det_align,
        "det_align_src": _det_align_src,
        "thresh": _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5),
        "align_thresh": _float_val("align_thresh", body.get("align_thresh"), 0.0, 1.0, 0.5),
        "dfm": str(dfm), "model": str(model_name), "xseg": str(xseg),
        "mask_mode": int((settings or {}).get("mask_mode", 6) or 0),
        "mask_src": _mask_src(settings, (settings or {}).get("mask_mode")),
        "config": dict(_merge_settings_to_config(settings)),
        "snap": snap, "plan_payload": plan_payload, "single_models": models,
        "use_library": True, "library": library_raw, "face_threshold": face_thr,
        "pred_batch": _int_val("batch", body.get("batch") or body.get("pred_batch"), 0, 64,
                               _PRED_BATCH_DEFAULT),
        # 合成线程默认 8（B 配置）；fused 有自己的默认，不再跟着 DFL_MERGE_WORKERS 走
        "merge_workers": _int_val(
            "merge_workers",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _FUSED_CMP_DEFAULT),
        "merge_cv2_threads": _int_val("merge_cv2_threads", body.get("merge_cv2_threads"),
                                      0, 64, _MERGE_CV2_THREADS_DEFAULT),
        "png_level": _int_val("png_compression", body.get("png_compression"), 0, 9,
                              _MERGE_PNG_LEVEL),
        # 帧图 JPEG 质量：固定 100（和分段链同一条口径，见 _STAGE_JPEG_QUALITY）
        "jpeg_quality": _int_val("jpeg_quality", body.get("jpeg_quality"), 1, 100,
                                 _STAGE_JPEG_QUALITY),
        # 对齐/预处理线程默认 4（B 配置），0 也让它是 4 —— 以前 0 落到 cpu//3，核少就掉档
        "pred_readers": _int_val("pred_readers", body.get("pred_readers"), 0, 16,
                                 _FUSED_PREP_DEFAULT),
        "pred_writers": _int_val("pred_writers", body.get("pred_writers"), 0, 16, 0),
        # 攒批：目标张数 + 超时 ms（都可以从请求体/环境变量压，排障用）
        "fused_batch": _int_val("fused_batch", body.get("fused_batch"), 1,
                                int(_PRED_BATCH_CAP), _FUSED_BATCH_TARGET),
        "fused_batch_wait_ms": _int_val("fused_batch_wait_ms",
                                        body.get("fused_batch_wait_ms"), 0, 60000,
                                        _FUSED_BATCH_WAIT_MS),
        "frame_budget_want": int(want or 0),
        # 异构 session 池的覆盖（0/空 = 自动推导，见 _fused_det_plan / server/_sess.py）
        "sess_gpu": (None if body.get("sess_gpu") in (None, "")
                     else _int_val("sess_gpu", body.get("sess_gpu"), 0, 8, 0)),
        "sess_cpu": (None if body.get("sess_cpu") in (None, "")
                     else _int_val("sess_cpu", body.get("sess_cpu"), 0, 16, 0)),
        "sess_strategy": (str(body.get("sess_strategy") or "").strip().lower() or None),
        # 验收/复现用：把「机器档位」按 low / mid / high 伪造（只影响配比推导）
        "sess_level": (str(body.get("sess_level") or "").strip().lower() or None),
        # fused 专有：没有 cache / meta / 阶段游标
        "fused": True, "cache": "", "meta": {}, "stages": ["fused"],
        "chain": True, "force": True, "keep_cache": False, "range_changed": False,
    }
    # 两个用户可调的性能旋钮（设置页「性能」区那两个下拉）：请求体 -> 进程 env，
    # 并把「这一单打算怎么跑」记进 cfg["knobs"]（[eff] 的两个新列就取它）。
    # 在这里就把 env 写下去的理由：下面的显存账（_vram_guard → _vram_plan）和 dfm 会话
    # 都按 dtype 算，晚一步写就会用上一单的档算这一单的账。
    _kn = _knobs.refine_compose(_knobs.resolve(body),
                                str((cfg.get("config") or {}).get("mode") or ""),
                                fused=True)
    # env 先写：下面的显存账（_vram_guard）按 dtype 算「权重砍半」那一项
    for _t in _knobs.apply(_kn):
        _knobs.warn_line(_t)
    cfg["knobs"] = _kn
    return cfg


def _fused_cv2_threads(cfg: dict) -> int:
    """fused 这条线给 OpenCV 留几条内部线程（**0 = 这一单不去动它**，用 OpenCV 自己的默认）。

    优先级：body.merge_cv2_threads / DFL_MERGE_CV2_THREADS（显式给了就用，还是那个老旋钮）
    → DFL_CV2_THREADS → 不动（默认）。

    默认为什么是「不动」而不是某个数字：三档都在真机上量过，**孤立基准和整条线给出
    相反的答案**（下面两组数是同一台机、同一份素材）：
      · 孤立 compose 基准（8 线程打满 · 真实 1080p 帧 · mask_mode8/seamless/mt · 5 轮中位）：
          cv2=4  22.7 帧/秒容量 · 8.03 核      cv2=1  25.3 帧/秒 · 7.45 核
          cv2=0（不动/OpenCV 默认 32） 26.6 帧/秒 · 7.38 核
        → 「少给 cv2 线程 = 同样的活少烧核、容量更高」在打满时成立。
      · 整条 fused（150 帧 416 tinymog，4 配置 × 4 轮交错中位）：
          cv2=4  18.62 帧/秒 · compose 107.3ms/帧
          cv2=1  18.18 帧/秒 · compose 125.8ms/帧   ← 整条**更慢**，compose 墙钟 +18ms
        → 生产 compose 只用到 ~25%，那 12% 容量换不成帧/秒；而 cv2 池退出并行会让
          每个合成任务变长（seamlessClone 300px 15.0 → 15.6ms、warp 0.94 → 3.04ms 这些
          单项差异在 25% 占用下被分摊掉，收益收不出来）。
    **结论：局部更快 ≠ 整条更快。** 默认保持「不动」（= 谁都不设时 OpenCV 自己的默认），
    要 A/B 就显式 DFL_CV2_THREADS=1 / 4。
    """
    raw = cfg.get("merge_cv2_threads")
    try:
        want = int(raw or 0)
    except (TypeError, ValueError):
        want = 0
    if want > 0:
        return min(32, want)
    env = str(os.environ.get(_FUSED_ENV_CV2) or "").strip()
    if env:
        try:
            return max(0, min(32, int(env, 10)))
        except ValueError:
            pass
    return max(0, min(32, int(_FUSED_CV2_DEFAULT)))


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


# 解码路径：**默认 raw**（-f rawvideo，软解）；DFL_FUSED_DECODE=png 退回 PNG 管道。
_FUSED_ENV_DECODE = "DFL_FUSED_DECODE"
# 检测输入（letterbox canvas）在 decode 线程先算：DFL_FUSED_DET_PREP=0 关（回老口径）
_FUSED_ENV_DET_PREP = "DFL_FUSED_DET_PREP"
# 解码线程只做 read() + ndarray：DFL_FUSED_READINTO=0 退回「read() 出 bytes 再 .copy()」。
# 1（默认）时 read 直接落进一个预分配环形缓冲（池 = 帧预算 + 4，见 _fused_budget/sem），
# 省掉每帧一次 6.2MB 的 bytes 分配 + 一次 6.2MB memcpy。像素不受影响（同一份字节、同一个
# ndarray 形状），实测 25.2→ 见 experiments/dec_lane/REPORT.md：隔离台架 101→119 帧/秒。
_FUSED_ENV_READINTO = "DFL_FUSED_READINTO"
# 解码那条 ffmpeg 的 -threads（-i 之前）。0/空 = auto（现状，= 逻辑核数）。
# 为什么可以动：解码是确定性的，-threads 1/2/4/8/16/auto 输出的 rawvideo 逐帧 sha1 完全相同
# （experiments/dec_lane/control.py），和 x264 那边「改 -threads 会换 md5」不是一回事。
# 为什么要动：auto 在 32 逻辑核上开一整池解码线程，和 x264/合成抢核；见同一份 REPORT 的核数列。
_FUSED_ENV_DEC_THREADS = "DFL_FUSED_DEC_THREADS"
_FUSED_RAW_MODES = ("raw", "rawvideo", "bgr24", "bgr48le")
# -i 的流信息里挑「pix_fmt, 宽x高」这一对；位深只从 pix_fmt 名字看
_FUSED_PIX_TOKEN = re.compile(r"Video:\s[^\n]*?,\s*([a-z][a-z0-9]+)(?:\([^)\n]*\))?,\s*\d+x\d+")
_FUSED_WIDE_TOKEN = re.compile(r"(10|12|16|48|64)(?:le|be)?$")


def _fused_src_bits(video: Path) -> int:
    """源视频**解码后**的位深（8 / 10 / 12 / 16）—— 决定 raw 管道进 bgr24 还是 bgr48le。

    为什么要问这一句：切帧阶段落盘的是 PNG，而 ffmpeg 的 png 编码器对 >8bit 的源会挑
    16bit（rgb48be）。所以「切帧产物」那一路是 16bit PNG 被 cv2 砍成 8bit（libpng 的
    chop = 取高字节，等价 >>8），这和 swscale 的 10→8 正确舍入**不是同一个数**：
    HEVC 10bit 实测 bgr24 vs frames/*.png mean 0.6038 / max 4 / 只有 5.16% 像素相同。
    改成 bgr48le 进来再 >>8 之后逐位相同（30 帧 mean 0.0000 / max 0 / 100%）。
    8bit 源仍然走 bgr24（多要一倍管道带宽没意义）。

    判不出来就当 8bit（= 原来的 bgr24 行为，一行都不变）。解析一次 -i 的流信息约 50ms，
    一次导出只调一次。
    """
    try:
        r = subprocess.run([str(_app_mod._ffmpeg_exe()), "-hide_banner", "-i", str(video)],
                           capture_output=True,
                           creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
        txt = (r.stderr or b"").decode("utf-8", "ignore")
    except Exception:                                       # noqa: BLE001
        return 8
    m = _FUSED_PIX_TOKEN.search(txt)
    if not m:
        return 8
    # yuv420p10le / gbrp12le / gray16le → 10 / 12 / 16；yuv420p / rgb24 / rgba → 8
    d = _FUSED_WIDE_TOKEN.search(m.group(1) or "")
    if not d:
        return 8
    return 16 if d.group(1) in ("48", "64") else int(d.group(1))


def _fused_open_frames(video: Path, start_i: int, size=None, mode="raw", bits=None,
                       threads=0):
    """开一路「ffmpeg → rawvideo 帧流」（默认）或「ffmpeg → PNG 帧流」，返回 (proc, 读一帧的函数)。

    默认 raw：`-f rawvideo -pix_fmt bgr24`（8bit 源）/`bgr48le -> >>8`（>8bit 源），
    软解。产物和切帧阶段落盘的 frames/%06d.png **逐位相同**（30 帧 mean 0.0000 / max 0 /
    100% 像素相同），每帧 25.2 → 8.9 ms。回归见 app/tools/accept/fused_decode_parity.py。

    DFL_FUSED_DECODE=png 保留 PNG 管道（ffmpeg 解 → 无损 PNG → 管道 → cv2.imdecode）：
    排障时用它，或者要「同一条 PNG 编解码路径」做对照时用它（约 25ms/帧）。

    **注意别用 -hwaccel cuda**：NVDEC 那条路把帧下载成 nv12 再转 BGR，色度是另一套
    插值，实测 mean 0.8895 / max 12（软解/NVDEC 的 YUV 平面本身逐位相同，差全在
    nv12→RGB 这一步）。**这条 0.89 与本改动无关**，容易误判成 raw 的锅。所以 raw 一律软解。

    >8bit 为什么是 bgr48le：见 _fused_src_bits（PNG 那一路是 16bit 截高字节，不是舍入）。

    旋转元数据（手机竖拍）：两条路都走 ffmpeg 的 autorotate，且 probe 给的是**显示方向**
    的宽高（imageio 读的就是 540x960 这种），所以 reshape 对得上 —— 90° 素材实测 100%
    相同。**-noautorotate 别加**：那会让 raw 变成未旋转的原始帧，和 PNG 产物对不上。

    start_i > 1 时用 ffmpeg 的 select 滤镜跳帧（0 基：保留 n >= start-1），跳过的帧不
    编码/不转换，比"读出来再丢"省得多；帧号仍然是绝对的（N 帧 = frames/NNNNNN.png）。
    """
    import cv2
    import numpy as np

    # ---- 三条解码路径 ----
    #   raw  · 8bit ：ffmpeg 解 → rawvideo bgr24  → 管道 → 直接 reshape。9 ms/帧。
    #   raw  · >8bit：ffmpeg 解 → rawvideo bgr48le → 管道 → <u2 reshape >>8。约 12 ms/帧。
    #   png  · 任意 ：ffmpeg 解 → PNG(无损) → 管道 → cv2.imdecode。25 ms/帧（可选）。
    # 为什么 raw 不改像素：两条路都是「同一个软解 → 同一个 swscale → BGR」，PNG 只是无损
    # 容器（8bit 源 rgb24 / >8bit 源 rgb48be，后者 cv2 取高字节 = 这里的 >>8）。
    # 这条口径由 app/tools/accept/fused_decode_parity.py 对着 frames/%06d.png 逐帧逐像素盯着。
    raw = str(mode or "raw").strip().lower() in _FUSED_RAW_MODES
    rf = (0, 0)
    deep = False
    if raw:
        if not size or int(size[0] or 0) <= 0 or int(size[1] or 0) <= 0:
            raise _BadInput("raw 解码要源视频的宽高，但 probe 里没有", 422)
        rf = (int(size[0]), int(size[1]))
        if bits is None:
            bits = _fused_src_bits(video)
        deep = int(bits) > 8

    ffmpeg = str(_app_mod._ffmpeg_exe())
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin"]
    if int(threads or 0) > 0:
        # 解码线程数（见 _FUSED_ENV_DEC_THREADS 的注释：逐帧 sha1 与 auto 相同）
        cmd += ["-threads", str(int(threads))]
    cmd += ["-i", str(video), "-map", "0:v:0", "-an"]
    if int(start_i) > 1:
        cmd += ["-vf", "select=gte(n\\,%d)" % (int(start_i) - 1)]
    # -vsync 0：原样搬每一帧（不按时间戳丢/补），帧数 = 选中的帧数
    cmd += ["-vsync", "0"]
    if raw:
        cmd += ["-f", "rawvideo", "-pix_fmt", ("bgr48le" if deep else "bgr24"), "pipe:1"]
    else:
        cmd += ["-f", "image2pipe", "-vcodec", "png",
                "-compression_level", "1", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=getattr(_app_mod, "_NO_WINDOW", 0))

    def _read_png(fh):
        """从管道里读一整张 PNG（按 chunk 读，8 字节签名 + 长度/类型/数据/CRC 到 IEND）。"""
        head = fh.read(8)
        if len(head) < 8 or head != _PNG_SIG:
            return None
        parts = [head]
        while True:
            ln = fh.read(4)
            if len(ln) < 4:
                return None
            typ = fh.read(4)
            if len(typ) < 4:
                return None
            size = int.from_bytes(ln, "big")
            body = fh.read(size) if size else b""
            crc = fh.read(4)
            if (size and len(body) < size) or len(crc) < 4:
                return None
            parts.append(ln)
            parts.append(typ)
            parts.append(body)
            parts.append(crc)
            if typ == b"IEND":
                return b"".join(parts)

    _mv = {}

    def _read_into(fh, arr):
        """把一整帧读进调用方给的 ndarray（readinto，不经过 bytes 中转）。"""
        e = _mv.get(id(arr))
        if e is None:
            # 顺手按住 arr，别让 id() 被回收后复用
            e = (memoryview(arr).cast("B"), arr)
            _mv[id(arr)] = e
        mv = e[0]
        n = 0
        total = len(mv)
        while n < total:
            k = fh.readinto(mv[n:])
            if not k:
                return False
            n += k
        return True

    def _next_frame_raw(buf=None):
        """rawvideo：定长读一帧 → (h,w,3) BGR uint8；流结束返回 None。

        >8bit 的源进的是 bgr48le（w*h*3*2 字节），这里 `>> 8` 取高字节 —— 和 16bit PNG
        被 libpng chop（= 取高字节）的结果逐位相同；用 swscale 的 10→8 舍入就对不上了。

        buf 给了且是 8bit 源：直接 readinto 那块内存（省一次 6.2MB 拷贝 + 一次 bytes 分配）。
        """
        bpp = 6 if deep else 3
        nb = rf[0] * rf[1] * bpp
        if buf is not None and not deep:
            if not _read_into(proc.stdout, buf):
                return None
            return buf
        buf = proc.stdout.read(nb)
        if not buf or len(buf) < nb:
            return None
        if deep:
            return (np.frombuffer(buf, "<u2").reshape(rf[1], rf[0], 3) >> 8).astype(np.uint8)
        return np.frombuffer(buf, np.uint8).reshape(rf[1], rf[0], 3).copy()

    def _next_frame(buf=None):
        """读下一帧 → BGR uint8；流结束返回 None。"""
        if raw:
            return _next_frame_raw(buf)
        blob = _read_png(proc.stdout)
        if not blob:
            return None
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        return img

    return proc, _next_frame


class _FusedFrame:
    """一帧在流水线里的全部状态（帧对象从头带到合成，合成之后源帧就放掉）。"""

    __slots__ = ("idx", "n", "frame", "boxes", "faces", "timings", "plan", "rec",
                 "det_ctx")

    def __init__(self, idx, n, frame):
        self.idx = int(idx)          # 源视频里的绝对帧号（1 起）
        self.n = int(n)              # 本段里的序号（1..total，PNG 文件名用它）
        self.frame = frame
        # 检测输入（decode 线程预先算好的 letterbox canvas + scale）；None = 检测线程自己做
        self.det_ctx = None
        self.boxes = None
        self.faces = None
        self.timings = {}
        self.plan = None
        self.rec = {"frame": int(idx), "detected": 0, "faces": 0, "swapped": 0,
                    "no_face": False, "dfm": "", "branch": ""}


def _fused_cancelled() -> bool:
    """取消：/export/cancel 和 /stage/cancel 都算（两条老取消路都能停掉快速导出）。"""
    if _vid_cancelled():
        return True
    with _STAGE_LOCK:
        return bool(_STAGE_JOB.get("cancelled"))


def _fused_dump(cfg: dict, job, composed) -> None:
    """DFL_FUSED_DUMP 点名的帧：把「合成前的每一份输入」落盘，和分段 cache/ 逐位比。

    只给排障/验收用（默认一个文件都不写）。
    """
    import cv2
    import numpy as np

    try:
        root = Path(str(cfg["out"])).with_suffix("")
        root = Path(str(root) + ".dump")
        root.mkdir(parents=True, exist_ok=True)
        tag = "%06d" % int(job.n)
        if composed is not None:
            cv2.imwrite(str(root / (tag + "_composed.png")), composed)
        for f in (job.faces or []):
            sub = tag + "_%02d" % int(f.get("k"))
            if f.get("pred_u8") is not None:
                cv2.imwrite(str(root / (sub + "_pred.png")), f["pred_u8"])
            for key, name in (("prd_u8", "prdm"), ("dstdfm_u8", "dstdm"),
                              ("xseg_prd_u8", "xseg_prd"), ("xseg_dst_u8", "xseg_dst")):
                arr = f.get(key)
                if arr is not None:
                    cv2.imwrite(str(root / (sub + "_" + name + ".png")), arr)
            if f.get("dst_u8") is not None:
                cv2.imwrite(str(root / (sub + "_dstin.png")), f["dst_u8"])
            (root / (sub + "_lm.json")).write_text(
                json.dumps(np.asarray(f["lm"]).tolist()), "utf-8")
            (root / (sub + "_dfm.txt")).write_text(
                "%s\n%s\n" % (Path(str(f.get("dfm") or "")).name, f.get("name") or ""),
                "utf-8")
    except Exception as e:                                     # noqa: BLE001
        _vid_log("快速导出：dump 失败（%s）" % str(e)[:160])


def _src_of(cfg: dict, keys, default, env_name: str) -> str:
    """这个数是从哪来的：环境变量 / 请求体 / 默认。**只用来打日志**（不改任何取值）。

    线程数这种东西一旦「日志写 8、实际跑 4」，后面所有排障都是错的 —— 所以每次导出都
    把「实际生效值 + 来源」写进 <成片>.log，改过 env 的人自己能看到。
    """
    env = str(os.environ.get(env_name) or "").strip()
    val = None
    for k in keys:
        v = cfg.get(k)
        if v is not None and not (isinstance(v, str) and not str(v).strip()):
            val = v
            break
    cur = default if val is None else val
    if env and str(env) == str(cur):
        return "env %s=%s" % (env_name, cur)
    if val is not None and str(val) != str(default):
        return "请求体 %s=%s" % (keys[0], cur)
    return "默认 %s" % cur


def _fused_run(cfg: dict) -> dict:
    """融合流水线：解码 → 检测 → 对齐/归组 → dfm → XSeg → 合成 → 编码。

    段间全是**有界**队列（depth=24）+ 一个**全局帧缓冲上限**（frame_budget 帧，
    1080p 一帧 6.22MB）；谁忙都不空转，谁也堆不起来。允许乱序完成：编码好的 PNG 字节
    进重排缓冲，写线程只按帧号顺序落盘；到写盘那一步才真正出现文件（问题 8 的断言：
    这些文件在 %TEMP% 里，**不是** <视频>_cache/）。
    """
    import queue
    import cv2
    import numpy as np
    from .compose import compose_frame, ensure_engine_path
    from .compose_gpu import (compose_frame_gpu, env_on as _compose_gpu_on,
                              ready as _compose_gpu_ready, last_error as _compose_gpu_err,
                              seamless_on as _compose_gpu_seamless_on)
    from .knobs import skip_text as _knob_skip_text
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    from .._align import (COARSE_SIZE, align_face_68, coarse_align_from_5,
                         landmarks68_in_source)

    video = Path(cfg["video"])
    out = Path(cfg["out"])
    mem0 = _proc_mem_mb()          # 流水线起点的本进程内存（收尾要和它比）
    rng = _stage_range(cfg)
    start_i, end_i, total = int(rng["start"]), int(rng["end"]), int(rng["total"])
    probe = cfg.get("probe") or {}
    fps = float(cfg.get("fps") or probe.get("fps") or 30.0)
    full = bool(rng["full"])
    rawpipe = _fused_rawpipe_on()      # 成片走 rawvideo 管道（默认）还是 PNG 落盘，见 _FUSED_ENV_RAW
    mask_src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int((cfg.get("settings") or {}).get("mask_mode", 6) or 0)
    need_xseg = str(mask_src.get("source") or "").lower() == "xseg"
    need_dst = bool(need_xseg and int(comp_cfg["mask_mode"]) in (7, 8, 9))
    # 兜底不再写死 "retinaface_10g"（重档、静默）：没有 det_align 就用 _DEFAULT_DETECTOR，
    # 而且**把这一跳写进 <成片>.log** —— 用户排障时看得到「为什么是 tinymog」。
    det_name = str(cfg.get("det_align") or "").strip()
    if not det_name:
        det_name = _DEFAULT_DETECTOR
        _vid_log("快速导出：cfg 里没有 det_align，用了默认检测器 %s（%s）"
                 % (det_name, _DEFAULT_DETECTOR))
    det_src = str(cfg.get("det_align_src") or "cfg.det_align")
    det_thresh = float(cfg.get("align_thresh") or 0.5)
    # 检测的 letterbox 在**哪儿**做 —— 三档，像素完全一样（同一个 _prep_ctx，同一张 canvas 的
    # 几何），区别只是「这一笔 0.6~6.4ms 记在谁头上」：
    #   decode（默认，=1）  decode 线程：它同时要 read 6.2MB，letterbox 排在同一线程里串起来
    #   detect（=0）        detect 线程：老口径，两段分在两处
    #   pool（=detect）     detect 线程 + 复用本线程画布（省每帧 1.2MB 的 np.full）
    # 三档逐位一致，回归见 app/tools/accept/det_prep_parity.py；A/B 的账见
    # experiments/dec_lane/REPORT.md（同进程交替、3 轮、探针/无探针两套）。
    _dp_raw = str(os.environ.get(_FUSED_ENV_DET_PREP) or "1").strip().lower()
    det_prep_on = _dp_raw not in ("0", "off", "false", "no")
    det_prep_pool = _dp_raw in ("detect", "pool", "detect_pool")
    png_level = int(cfg.get("png_level") if cfg.get("png_level") is not None else 1)
    # ---- 合成放在哪儿算：CPU（默认）还是 GPU（DFL_COMPOSE_GPU=1）----
    # GPU 那条路的每帧墙钟是 1~2ms 量级，一条线程就够（多开会挤同一个 CUDA stream，
    # 只会互相等）；顺带把 8 条合成线程的 CPU 占用整个还回去。
    # seamless 不进 GPU（那条要解 Poisson 方程，见 compose_gpu 的开头），继续走 CPU。
    gpu_cmp = {"on": False, "why": ""}
    if _compose_gpu_on():
        _mode0 = str(dict(cfg.get("config") or {}).get("mode") or "overlay").strip().lower()
        if "seamless" in _mode0 and not _compose_gpu_seamless_on():
            gpu_cmp["why"] = "mode=%s 且 DFL_COMPOSE_GPU_SEAMLESS=0，走 CPU" % _mode0
        elif _compose_gpu_ready():
            gpu_cmp["on"] = True
        else:
            gpu_cmp["why"] = "torch/cuda 起不来：%s" % (_compose_gpu_err() or "?")
    # GPU 合成开几条线程：**默认 4**（量出来的，150 帧同段，只差这个数）：
    #   overlay   1 条 14.72 帧/秒 → 4 条 21.04 帧/秒
    #   seamless  1 条 10.54 → 2 条 19.23 → 4 条 19.46
    # 原因：GPU 合成线程每次 .cpu() 都要等**整条 CUDA stream**（dfm/xseg/detect 也在上面），
    # 单线程时「墙钟 54~78ms/帧」里绝大部分是在排队，一条线程就把自己变成瓶颈了；
    # 多开几条把「等队列」和「真算」重叠起来。DFL_COMPOSE_GPU_THREADS 可压。
    if gpu_cmp["on"]:
        try:
            _n_gpu = max(1, min(8, int(str(os.environ.get("DFL_COMPOSE_GPU_THREADS")
                                           or "4").strip() or 4)))
        except (TypeError, ValueError):
            _n_gpu = 4
        n_cmp = _n_gpu
    else:
        n_cmp = max(1, int(_merge_worker_count(cfg)))
    # 回显给 status / 日志：这一单的合成到底在哪算的（用户/排障一眼看出 CPU 还是 GPU）
    cfg["_compose_backend"] = "gpu" if gpu_cmp["on"] else "cpu"
    cfg["_compose_threads"] = int(n_cmp)
    cfg["_compose_why"] = str(gpu_cmp.get("why") or "")
    # ★ 选了 gpu、实际跑 cpu → 必须喊出来（检测器那次事故的同类：界面写一套、日志跑另一套）。
    # 这一条只能在**这里**判：torch/cuda 起不起得来要真碰一次 torch，路由线程里碰会把
    # 一条 POST 卡好几秒，所以 [eff] 那行先按「打算怎么跑」写，跑之前的真实结论在这里补。
    # 文案三件事必须齐：选了 X、实际跑 Y、原因 Z（见 knobs.skip_text）。
    if str((cfg.get("knobs") or {}).get("compose_backend") or "") == "gpu" and not gpu_cmp["on"]:
        _kw = _knob_skip_text("合成后端", "gpu", "cpu",
                              gpu_cmp.get("why") or "torch/cuda 不可用")
        _knobs.warn_line(_kw)
        _vid_log("[eff-warn] " + _kw)
        with _merge_job_lock:
            _patch = {"code": "eff-skipped", "field": "compose_backend",
                      "level": "warn", "text": _kw}
            _merge_job["eff_warnings"] = list(_merge_job.get("eff_warnings") or []) + [_patch]
            _merge_job["warnings"] = list(_merge_job.get("warnings") or []) + [_kw]
            # 同一个 eff 对象就是 status.eff（_eff_publish 返回的那份，按引用存进来）：
            # 就地改成**实际跑的值**，界面抽屉里那一格才不会继续说 gpu。
            _e = _merge_job.get("eff")
            if isinstance(_e, dict):
                _e["compose_backend"] = "cpu"
                _e["compose_why"] = str(gpu_cmp.get("why") or "")
                _e["warnings"] = list(_e.get("warnings") or []) + [_patch]
        with _STAGE_LOCK:
            _STAGE_JOB["eff_warnings"] = list(_STAGE_JOB.get("eff_warnings") or []) + [_patch]
            _STAGE_JOB["warnings"] = list(_STAGE_JOB.get("warnings") or []) + [_kw]
            _e2 = _STAGE_JOB.get("eff")
            if isinstance(_e2, dict) and _e2 is not _merge_job.get("eff"):
                _e2["compose_backend"] = "cpu"
                _e2["compose_why"] = str(gpu_cmp.get("why") or "")
                _e2["warnings"] = list(_e2.get("warnings") or []) + [_patch]
        cfg.setdefault("knobs", {})["compose_backend"] = "cpu"
    # PNG 编码是这一条链上最贵的纯 CPU 活（1080p 实测 94ms/帧，cv2.imencode 会放 GIL，
    # 多线程是真并行：2 线程 ≈2 倍、4 线程 ≈3.6 倍）。默认 3：detect（52.8ms/帧）才是硬底，
    # 再往上加只会跟检测抢 CPU。
    n_enc = max(1, int(cfg.get("pred_writers") or 3))
    # 对齐/预处理线程：**默认 4**（B 配置）。以前是 min(4, cpu//3) —— 16 核机是 4、
    # 8 核机是 2，同一份代码在两台机器上跑出两种口径；现在钉成 _FUSED_PREP_DEFAULT。
    n_prep = max(1, int(cfg.get("pred_readers") or _FUSED_PREP_DEFAULT))
    # 攒批：目标张数 + 超时（秒）。实际目标 = max(阶梯值, 这个值)，但阶梯实测如果证明
    # 目标档每张更贵就退回阶梯那档 —— 见 _target_for 与 _FUSED_BATCH_TARGET 的注释。
    batch_target = max(1, min(int(_PRED_BATCH_CAP),
                              int(cfg.get("fused_batch") or _FUSED_BATCH_TARGET)))
    _wait_ms = cfg.get("fused_batch_wait_ms")
    if _wait_ms is None:
        _wait_ms = _FUSED_BATCH_WAIT_MS
    batch_wait = max(0.0, float(_wait_ms) / 1000.0)
    qd = int(_FUSED_QDEPTH)
    # cv2 的内部线程数（进程级）—— **默认 0 = 不动**（DFL_CV2_THREADS 可显式给；见
    # _fused_cv2_threads 的实测表）。老版本按 cpu//n_cmp 给 4 条，理由是「让检测的预处理
    # 并行起来」；现在检测是 tinymog 3.2ms/帧、letterbox 已经搬到 decode 线程，那条理由
    # 早就不成立，而 cv2=4 在孤立基准里确实比 1 差（容量 22.7 vs 25.3、核 8.03 vs 7.45）——
    # 但**整条线**上两者一样（18.62 vs 18.18，都在噪声里，见报告的表），所以默认回到
    # 「不设」（= OpenCV 自己的默认），谁也不改谁的池。
    cv2_threads = _fused_cv2_threads(cfg)
    # u8 量化的写法（DFL_FUSED_U8，默认 1 = 复用暂存；0 = 老口径，见 _FUSED_U8_DEFAULT）
    _u8_fast = 1 if str(os.environ.get(_FUSED_ENV_U8) or str(_FUSED_U8_DEFAULT)).strip() \
        not in ("0", "off", "false", "no") else 0

    cache_dir = Path(str(video.parent / (video.stem + "_cache")))
    cache_pre = cache_dir.is_dir()
    dump_frames = set()
    for _tok in str(os.environ.get(_FUSED_ENV_DUMP) or "").replace(";", ",").split(","):
        _tok = _tok.strip()
        if _tok.isdigit():
            dump_frames.add(int(_tok))

    # ---- 模型 / 显存 / 帧预算（起流水线之前，出问题要在这里就喊出来）----
    # 顺序要紧：**先算预算，再加载模型**。加载本身就是最可能 OOM 的那一步（一份 384
    # 会话 2.07GB，只有 2GB 空闲的卡上「加载」两个字就已经崩了），所以拒绝要在它之前。
    # 尺寸不加载也能知道（_dfm_size_hint 看文件名），加载后如果实测尺寸不一样再判一次。
    model_mb = _vram_model_mb(cfg["dfm"])
    size_hint = _dfm_size_hint(cfg["dfm"])
    want_frames = int(cfg.get("frame_budget_want") or 0)
    # ★ reclaim_first：**导出起手第一件事**是把常驻预览 worker 的模型卸掉，然后
    #   重新探显存、再算这一份预算。顺序反过来的代价是实测的 22 → 15.8 帧/秒：
    #   预览 worker 常驻 2733MB 时「空闲」只有 2766MB，帧预算被压到 8、dfm 批压到 1，
    #   而占着那 2733MB 的正是我们自己。
    bud = _fused_budget(want_frames, None, None, size_hint,
                        model_mb=model_mb, batch=int(batch_target),
                        force=bool(cfg.get("vram_force")),
                        reclaim=True, reclaim_first=True)
    _vram_apply(bud, "快速导出")
    for _ln in _vram_log_lines(bud, "快速导出"):
        _vid_log(_ln)
    lm = _landmarks_model()
    if lm is None:
        raise _BadInput("2D106 关键点模型不可用（weights/insightface_2d106.onnx）—— "
                        "没有 68 点就跑不了快速导出", 500)
    # 检测走哪个 EP（DFL_DET_EP=auto|gpu|cpu）—— 必须在**拿检测器**之前定，
    # 否则日志里那行「单帧 ms」量的是 GPU 那份、跑的是 CPU 那份，数字对不上账。
    det_pol = _det_ep_resolve(det_name, probe.get("w"), probe.get("h"))
    _vid_log("快速导出 %s" % _det_ep_line(det_pol))
    det = _detect_for(det_name, det_pol.get("ep") or "gpu")
    if det is None:                       # CPU 那一路起不来（极少数机器）→ 老实退回 GPU 并喊出来
        _vid_log("快速导出：检测器 %s EP 起不来 → 退回 GPU（%s）"
                 % (str(det_pol.get("ep")).upper(), det_name))
        det = _detector(det_name)
        det_pol["ep"] = "gpu"
        det_pol["fallback"] = True
    # 「用了哪个检测器 + 它到底多贵」必须在起流水线之前就落进 <成片>.log：
    # 界面上选的是 TinyMog、日志里出现 retinaface_10g 这种对不上账的问题，只有这里能复盘。
    _dp = _det_quick_profile(det, probe.get("w"), probe.get("h"))
    _vid_log("快速导出：检测器 %s（%s · %s）· 1080p 单帧 %s ms = letterbox %s + 前向 %s + "
             "NMS/解码 %s（零图 %d 次中位）"
             % (det_name, det_src, _dp.get("sess") or "?",
                _dp.get("full_ms"), _dp.get("prep_ms"), _dp.get("fwd_ms"),
                _dp.get("post_ms"), int(_dp.get("n") or 0)))
    if _dp.get("err"):
        _vid_log("快速导出：检测器基准没量成（%s）" % _dp["err"])
    elif det_name == "tinymog" and (_dp.get("full_ms") or 0.0) > 20.0:
        # 这条 warning 是给「以为在用 TinyMog」的人：tinymog 正常 2~6ms，20ms 以上只有三种可能
        _vid_log("快速导出：★ 检测器是 tinymog，单帧却 %.1f ms（正常 2~6ms）—— 查这三件事："
                 "有没有 TRT 引擎、是不是退回 onnxruntime、是不是在 CPU EP 上"
                 % float(_dp.get("full_ms") or 0.0))
    # 预算说不可行时**不加载** dfm —— 加载是要占显存的，先拒绝再别动显存
    if not bud.get("ok"):
        raise _BadInput(bud.get("msg") or _vram_refuse_msg(bud, "快速导出"), 400)
    ent0 = _dfm_session(str(cfg["dfm"]))
    size_px = int(ent0[2])
    if size_px != int(size_hint):
        _vid_log("快速导出：文件名给的尺寸 %d 与实测 %d 不一致 → 按实测重算预算"
                 % (int(size_hint), size_px))
        # **resident=True**：这一拍 dfm 已经在本进程里加载好了（size_px 是它自己报的），
        # 会话那 2GB 已经在卡上了 —— 再算一遍就是重复计数，会把可跑的单子判死。
        # 只判增量：检测/遮罩（检测器上面已经加载了）+ 帧缓冲 + dfm 批。
        bud = _fused_budget(want_frames, None, None, size_px,
                            model_mb=model_mb, batch=int(batch_target),
                            resident=True, force=bool(cfg.get("vram_force")),
                            reclaim=True, reclaim_first=False)
        _vram_apply(bud, "快速导出")
        for _ln in _vram_log_lines(bud, "快速导出"):
            _vid_log(_ln)
        if not bud.get("ok"):
            raise _BadInput(bud.get("msg") or _vram_refuse_msg(bud, "快速导出"), 400)
    budget = int(bud.get("frames") or bud.get("frame_budget") or _FUSED_BUDGETS[0])
    # 预算批 ≤ 攒批目标：显存只够批 2 时就别再按 8 攒（_target_for 还会按实测再降一次）
    _plan_batch = max(1, int(bud.get("batch") or batch_target))
    if _plan_batch < batch_target:
        _vid_log("快速导出：dfm 攒批 目标 %d → %d（显存预算）" % (batch_target, _plan_batch))
        batch_target = _plan_batch
    cfg["frame_budget"] = budget
    cfg["budget"] = bud
    cfg["vram_plan"] = bud
    xseg = None
    xs_label = ""
    if need_xseg:
        xseg, xs_res = _stage_xseg(mask_src.get("model"))
        xs_label = str(xs_res.get("label") or "")
        info = dict(getattr(xseg, "info", {}) or {})
        _vid_log("快速导出：遮罩 %s（%s）· 后端=%s%s"
                 % (xs_label or "-", Path(str(info.get("onnx") or "")).name or "-",
                    info.get("backend") or "?", ("(%s)" % info.get("precision"))
                    if info.get("precision") else ""))
    _vid_log("快速导出：dfm %s · %dx%d · %s · 检测器 %s thresh=%.2f（%s）"
             % (Path(str(cfg["dfm"])).name, size_px, size_px,
                (ent0[6] if len(ent0) > 6 else "?"), det_name, det_thresh, det_src))
    _vid_log("快速导出：显存预算 → 帧缓冲 %d / dfm 批 %d（请求 %d 帧 / 批 %d%s）· "
             "预计峰值 %s MB / 可用 %.0f MB"
             % (budget, int(bud.get("batch") or 0), int(bud.get("requested") or 0),
                int(bud.get("requested_batch") or 0),
                "，已降档" if bud.get("downgraded") else "，不用降档",
                ("%.0f" % bud["peak_mb"]) if bud.get("peak_mb") is not None else "装不下",
                float(bud.get("avail_mb") or 0.0)))
    if bud.get("cpu"):
        _vid_log("快速导出：★ 当前在 CPU 上跑 ★（显存装不下 + %s=1）—— 预计比 GPU 慢约 %.0f 倍"
                 % (_VRAM_ENV_CPU, _VRAM_CPU_SLOW))

    # raw 管道这条路**一个中间文件都不写**（连临时目录都不建）：帧从内存直接进 ffmpeg 的 stdin。
    # 只有老路（DFL_FUSED_RAWPIPE=0）才需要 %TEMP%/dfl_fused_* 那串 PNG。
    frames_dir = None if rawpipe else Path(tempfile.mkdtemp(prefix="dfl_fused_"))
    if frames_dir is not None:
        cfg["frames_dir"] = str(frames_dir)
    # 线程数一律把**实际生效的值**和它是从哪来的打出来：用户改过环境变量/请求体时，
    # 「日志里明明是 8，为什么只有 4」这种事不该再发生（这条日志是唯一能复盘的地方）。
    _vid_log("快速导出：线程 检测1/解码1/对齐%d/GPU1/合成%d/编码%d · 队列 %d · cv2 %s · "
             "检测预处理在解码线程 %s"
             % (n_prep, n_cmp, n_enc, qd,
                ("%d 线程（DFL_CV2_THREADS）" % cv2_threads) if cv2_threads
                else "不动（OpenCV 默认）", "开" if det_prep_on else "关"))
    _vid_log("快速导出：线程来源 对齐=%s 合成=%s（B 配置默认 4/8；DFL_FUSED_PREP / "
             "DFL_FUSED_CMP / body.pred_readers / body.merge_workers 可覆盖 · cpu %d 线程）"
             % (_src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT, "DFL_FUSED_PREP"),
                _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                        _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP"),
                int(os.cpu_count() or 0)))
    _vid_log("快速导出：攒批 目标 %d 张 / 超时 %dms / 模式 %s（%s · %s=0 可回滚）"
             % (batch_target, int(round(batch_wait * 1000.0)),
                "自适应" if _FUSED_BATCH_ADAPT else "固定超时",
                ("攒不满就立刻送、GPU 空着也立刻送，超时只当上限" if _FUSED_BATCH_ADAPT
                 else "一律等满超时（老口径）"), _FUSED_ENV_ADAPT))
    _vid_log("快速导出：合成在 %s（%s）· 线程 %d"
             % (("GPU（torch cuda）" if gpu_cmp["on"] else "CPU"),
                ("DFL_COMPOSE_GPU=1 · 只上传人脸框、只下载合成后的框"
                 if gpu_cmp["on"] else
                 ("DFL_COMPOSE_GPU=1 但%s" % gpu_cmp["why"] if gpu_cmp["why"]
                  else "默认（DFL_COMPOSE_GPU=1 可切 GPU）")),
                n_cmp))
    _vid_log("快速导出：合成后端回显 compose_backend=%s · 线程 %d%s"
             % (cfg["_compose_backend"], cfg["_compose_threads"],
                ("（%s）" % cfg["_compose_why"]) if cfg["_compose_why"] else ""))
    _vid_log("快速导出：u8 量化 %s（%s）"
             % ("复用 float 暂存 + astype" if _u8_fast else "老口径 np.clip(x*255).astype",
                "%s=%d" % (_FUSED_ENV_U8, _u8_fast)))
    _vid_log("快速导出：成片出口 %s（%s=%d；<视频>_cache/ 一个都不建）"
             % ("rawvideo 管道（不落中间帧）" if rawpipe else "PNG 落盘 → ffmpeg 读帧序列",
                _FUSED_ENV_RAW, 0 if rawpipe is False else 1))
    if frames_dir is not None:
        _vid_log("快速导出：临时帧目录 %s（成功/取消都删）" % frames_dir)

    END = object()
    ticks = _FusedTicks()
    stop = threading.Event()
    errors = []
    sem = threading.BoundedSemaphore(budget)
    q_dec = queue.Queue(maxsize=max(2, qd))
    q_det = queue.Queue(maxsize=max(2, qd))
    q_gpu = queue.Queue(maxsize=max(qd, 64))
    q_cmp = queue.Queue(maxsize=max(2, qd))
    q_enc = queue.Queue(maxsize=max(2, qd))
    cnt = {"prep": n_prep, "compose": n_cmp, "encode": n_enc}
    cnt_lock = threading.Lock()
    live = {"n": 0, "peak": 0}
    live_lock = threading.Lock()
    stat_lock = threading.Lock()
    bstat = {"calls": 0, "faces": 0, "max": 0, "hist": {}, "why": {}}
    rep = {"at": 0.0}
    got = {"frames": 0, "swapped": 0, "copied": 0, "no_face": 0, "faces": 0, "skipped": 0}

    def _fail(stage, exc):
        errors.append("%s: %r" % (stage, exc))
        stop.set()

    def _put(q, item, timeout=0.2):
        while not stop.is_set():
            try:
                q.put(item, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False

    def _get(q, timeout=0.2):
        while not stop.is_set():
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                continue
        return END

    def _poll(q, timeout=0.005):
        """非阻塞取一个：这一拍没活就返回 None，停止时返回 END。

        为什么 GPU 段需要它：攒批超时只能在「等下一张脸」的空档里去检查 —— 用 _get
        的无限等待，尾部那几张脸就会一直压在 pend 里等着凑满目标（见 _FUSED_BATCH_WAIT_MS）。
        """
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return END if stop.is_set() else None

    def _report(done, force=False):
        now = time.time()
        if not force and (now - rep["at"]) < 0.25:
            return
        rep["at"] = now
        el = max(0.001, now - t_all)
        rate = float(done) / el
        eta = int((total - done) / rate) if rate > 0 and 0 < done < total else None
        pct = int(round(_VID_PCT_SWAP * float(done) / total)) if total else 0
        _stage_progress(
            "fused", min(100, pct),
            "快速导出 %d/%d 帧（第 %d..%d 帧）· %.1f 帧/秒"
            % (done, total, start_i, end_i, rate),
            frames_done=int(done), frames_total=int(total),
            frame_fps=round(rate, 2), eta_s=eta,
            swapped=int(got["swapped"]), copied=int(got["copied"]),
            no_face=int(got["no_face"]))
        _vid_stats(done=int(done), total=int(total), fps=round(rate, 2), eta_s=eta,
                   frame_budget=budget, fused=True, peak_live=int(live["peak"]))
        # 每段的「每帧 ms」+ dfm 当前的攒批（界面 lane 旁边那个小字）；_report 自己
        # 就把频率压到 0.25s 一拍，所以这里不用再节流。
        try:
            _pm = ticks.dump(int(done))
            _STAGE_METER.note_ms({k: _pm.get(v) for k, v in _STAGE_MS_SRC.items()})
            _STAGE_METER.note_batch(
                {"dfm": int(batch_of.get(str(cfg["dfm"])) or batch_target)})
        except Exception:                                  # noqa: BLE001
            pass

    def _frame_done(job):
        """一帧「出去了」：老路 = PNG 已落盘，raw 路 = 已交给 ffmpeg。计数与上报只此一处。

        两条路抽成同一个函数是因为**它写在谁身上会变**：老路在编码线程里（3 条并发），
        raw 路在唯一写者里（按帧序）。数字必须还是同一口径，否则 status 里的帧速会一半大。
        """
        with stat_lock:
            got["frames"] += 1
            if job.rec.get("no_face"):
                got["copied"] += 1
                got["no_face"] += 1
            else:
                got["swapped"] += 1
            got["faces"] += int(job.rec.get("faces") or 0)
            done = got["frames"]
        _vid_record(job.rec)
        _report(done)

    # ---- 显存保护：dfm 批大小按 session 实测阶梯选（每个 dfm 一次）----
    batch_of = {}
    batch_meta = {}
    _batch_note = {}
    batch_lock = threading.Lock()

    def _batch_for(dfm_path):
        with batch_lock:
            hit = batch_of.get(dfm_path)
        if hit is not None:
            return hit
        ent = _dfm_session(dfm_path)
        b, bmeta = _pred_pick_batch({**cfg, "pred_batch": int(cfg.get("pred_batch") or 0)}, ent)
        with batch_lock:
            batch_of[dfm_path] = int(b)
            batch_meta[dfm_path] = dict(bmeta or {})
        _vid_log("快速导出：dfm %s 批大小 %d（%s）· %dx%d"
                 % (Path(dfm_path).name, int(b), bmeta.get("source"),
                    int(ent[2]), int(ent[2])))
        return int(b)

    # ---- ① 解码：ffmpeg 的 rawvideo 帧流 → 帧缓冲（信号量卡死上限，满了解码就停，天然背压）----
    # 默认 rawvideo（软解）：和切帧产物 frames/%06d.png 逐位相同，25.2→8.9 ms/帧
    # （见 _fused_open_frames 的注释 + app/tools/accept/fused_decode_parity.py 的对拍回归）。
    # DFL_FUSED_DECODE=png 可退回 PNG 管道（排障 / 想走「同一条 PNG 编解码路径」时用）。
    gen = {}
    dec_mode = str(os.environ.get(_FUSED_ENV_DECODE) or "raw").strip().lower()
    if dec_mode not in _FUSED_RAW_MODES:
        dec_mode = "png"                      # 只认显式非 raw 值（png）= 回落 PNG 管道
    src_bits = _fused_src_bits(video)        # 8 / 10 / 12 / 16：日志 + raw 选 bgr24 / bgr48le
    _vid_log("快速导出：解码 %s（%s）· 源 %sx%s / %sbit"
             % ("rawvideo %s（软解）" % ("bgr48le→>>8" if src_bits > 8 else "bgr24")
                if dec_mode != "png" else "PNG 管道",
                "默认；和切帧阶段落盘的 frames/%06d.png 逐位相同、25→9 ms/帧"
                if dec_mode != "png"
                else "DFL_FUSED_DECODE=png，可选；同一条 PNG 编解码路径，约 25 ms/帧",
                probe.get("w"), probe.get("h"), src_bits))

    readinto_on = str(os.environ.get(_FUSED_ENV_READINTO) or "1").strip().lower() \
        not in ("0", "off", "false", "no")
    dec_threads = 0
    try:
        dec_threads = max(0, int(str(os.environ.get(_FUSED_ENV_DEC_THREADS) or "0").strip()))
    except Exception:                                          # noqa: BLE001
        dec_threads = 0
    _vid_log("快速导出：解码线程 %s · readinto %s · 检测 letterbox 在 %s"
             % (("ffmpeg -threads %d" % dec_threads) if dec_threads else "auto（ffmpeg 默认）",
                "开（预分配环形缓冲，帧预算+4 槽）" if readinto_on else "关（read()+copy()）",
                "decode 线程" if det_prep_on else "detect 线程"))

    def _decode():
        proc = None
        # readinto 的环形缓冲：槽数 = 帧预算 + 4。为什么不担心被覆盖：在飞帧由 sem 卡在
        # budget 帧（decode 这里是「先读、后 acquire」，所以最多 budget+1 块同时活着），
        # 而 frame 引用在合成之后就放了（_compose 里 job.frame = None）。
        pool = []
        if readinto_on:
            try:
                _pw, _ph = int(probe.get("w") or 0), int(probe.get("h") or 0)
                if _pw > 0 and _ph > 0:
                    pool = [np.empty((_ph, _pw, 3), np.uint8)
                            for _ in range(int(budget) + 4)]
            except Exception:                                  # noqa: BLE001
                pool = []
        pi = 0
        try:
            proc, next_frame = _fused_open_frames(video, start_i,
                                                 size=(probe.get("w"), probe.get("h")),
                                                 mode=dec_mode, bits=(src_bits or None),
                                                 threads=dec_threads)
            gen["proc"] = proc
            n = start_i - 1
            while not stop.is_set():
                if n >= end_i:
                    break
                t_rd = time.perf_counter()
                if pool:
                    frame = next_frame(pool[pi])
                    if frame is not None:
                        pi = (pi + 1) % len(pool)
                else:
                    frame = next_frame()
                ticks.add("decode", time.perf_counter() - t_rd)
                if frame is None:
                    break
                n += 1
                t_wait = time.perf_counter()
                while not stop.is_set():
                    if sem.acquire(timeout=0.2):
                        break
                ticks.add("dec_wait", time.perf_counter() - t_wait)
                if stop.is_set():
                    break
                with live_lock:
                    live["n"] += 1
                    if live["n"] > live["peak"]:
                        live["peak"] = live["n"]
                job = _FusedFrame(n, n - start_i + 1, frame)
                if det_prep_on:
                    # 检测的 letterbox 在这条线程做：它只忙 15ms/帧（30%），有富余；
                    # 检测线程是硬底，把 0.7ms 的 resize + 填边搬走就是净赚。
                    # 逐位一致由「同一个 prep()」保证（见 _detect.py 的 _prep_ctx）。
                    t_prep = time.perf_counter()
                    job.det_ctx = _det_prep(det, frame)
                    if job.det_ctx is not None:
                        ticks.add("det_prep", time.perf_counter() - t_prep)
                _stage_tick("frames")            # 交接点：解码 → 检测队列
                if not _put(q_dec, job):
                    break
        except BaseException as e:                             # noqa: BLE001
            _fail("decode", e)
        finally:
            _put(q_dec, END)

    # ---- ② 检测：默认 1 路（检测器是共享对象，本来就不能并发）；异构池时 N 路 ----
    # 多路的由来：检测是这条链的硬底（实测 48.9 ms/帧 = 20.4 帧/秒，占用 96~98%），
    # 只靠一路 GPU 就把整条线钉死了。多开的那几路各拿**自己的** session（CPU EP 或
    # 第二份 CUDA），帧从 q_dec 里抢着取 —— 谁空谁拿，天然就是「最短队列优先」。
    # 这一路只做「前向 + NMS」：letterbox 已经在 decode 线程做完（见 _det_prep）。
    # 搬家只在「检测是瓶颈」时才值钱（retinaface：detect 占用 88%）；tinymog 下瓶颈
    # 已经移到 dfm/编码，端到端 0 收益 —— 别把它当提速手段。
    def _detect(k=0, det_obj=None, n_workers=1):
        try:
            while True:
                t_wait = time.perf_counter()
                job = _get(q_dec)
                ticks.add("det_wait", time.perf_counter() - t_wait)
                if job is END:
                    # 毒丸只放了一颗：谁吃到谁负责传下去（n_workers-1 颗），
                    # 最后各自退出 —— 不然其余几路会在 _get 里等到天荒地老。
                    for _i in range(max(0, int(n_workers) - 1)):
                        _put(q_dec, END)
                    for _i in range(n_prep):
                        _put(q_det, END)
                    break
                t = time.perf_counter()
                try:
                    if job.det_ctx is None and det_prep_pool and not det_prep_on:
                        # pool 档：letterbox 在本线程做，并复用本线程那张画布
                        # （prep → 前向 → 立刻 drop，所以复用安全；见 _prep_ctx 的注释）
                        tq = time.perf_counter()
                        _ctx = _det_prep(det if det_obj is None else det_obj, job.frame,
                                         reuse=True)
                        if _ctx is not None:
                            ticks.add("det_prep", time.perf_counter() - tq)
                            job.det_ctx = _ctx
                    if job.det_ctx is not None:
                        # 主路径：canvas 已经在 decode 线程做好，这里只剩前向 + NMS
                        boxes = (_detect_on_ctx(det_obj, job.det_ctx, det_thresh)
                                 if det_obj is not None
                                 else _detect_boxes_ctx(job.det_ctx, det_name, det_thresh))
                        job.det_ctx = None       # 1.2MB/帧：用完就放，别带到合成再扔
                    else:
                        boxes = (_detect_on(det_obj, job.frame, det_thresh)
                                 if det_obj is not None
                                 else _detect_boxes(job.frame, det_name, det_thresh))
                except Exception as e:                         # noqa: BLE001
                    _fail("detect", e)
                    for _i in range(n_prep):
                        _put(q_det, END)
                    break
                ticks.add("detect", time.perf_counter() - t)
                # 和 ② 切脸同一个次序（按 x1 排）：脸序号 = 这个顺序，compose 也按它贴
                job.boxes = sorted(list(boxes or []),
                                   key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
                job.rec["detected"] = len(job.boxes)
                _stage_tick("det")               # 交接点：检测 → 对齐/预处理
                _put(q_det, job)
        except BaseException as e:                             # noqa: BLE001
            _fail("detect", e)

    # ---- ③ 对齐/预处理 + 归组：68 点、dfm 输入、xseg 的 dst 画布、这一帧用哪个 dfm ----
    def _prep():
        try:
            while True:
                job = _get(q_det)
                if job is END:
                    return
                t = time.perf_counter()
                faces = []
                for k, b in enumerate(job.boxes or []):
                    box = b.get("box") or [0, 0, 0, 0]
                    try:
                        x1, y1, x2, y2 = [float(v) for v in box]
                    except (TypeError, ValueError):
                        continue
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    side = int(round(max(bw, bh) * 2.0))
                    side -= side % 2
                    kps = b.get("kps") or []
                    if len(kps) < 5 or side < 32:
                        continue
                    try:
                        cmat = coarse_align_from_5(kps)
                        coarse = cv2.warpAffine(job.frame, cmat,
                                                (COARSE_SIZE, COARSE_SIZE),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0))
                        lm68a = lm.landmarks68(coarse)
                        if lm68a is None:
                            continue
                        pts = landmarks68_in_source(lm68a, cmat)
                        crop, meta_f = align_face_68(
                            job.frame, np.asarray(pts, np.float64),
                            box=[int(round(x1)), int(round(y1)),
                                 int(round(x2)), int(round(y2))], size=side)
                    except Exception as e:                     # noqa: BLE001
                        got["skipped"] += 1
                        _vid_log("快速导出：第 %d 帧第 %d 张脸跳过：%s"
                                 % (job.idx, k, str(e)[:100]))
                        continue
                    if crop is None or not meta_f:
                        got["skipped"] += 1
                        _why = ((meta_f or {}).get("reject_reason")
                                if isinstance(meta_f, dict) else "")
                        if _why:
                            _vid_log("快速导出：第 %d 帧第 %d 张脸拒收：%s"
                                     % (job.idx, k, _why))
                        continue
                    # compose 用的 landmarks = DFLJPG 里那份（align_face_68 已 round(4)）
                    lm_f = np.asarray(meta_f["source_landmarks"], np.float32)
                    xseg_dst = None
                    if need_dst:
                        # 分段的 dst 输入是 aligned/*.jpg（q100 4:4:4，仍然有损）→ 照抄这一步
                        # 质量取同一个 cfg["jpeg_quality"]（默认 100）：两条路必须逐位一致，
                        # 所以这里不写死 100，免得以后改一处忘另一处。
                        from .._align import jpeg_encode_params
                        okj, buf = cv2.imencode(
                            ".jpg", crop,
                            jpeg_encode_params(int(cfg.get("jpeg_quality")
                                                   or _STAGE_JPEG_QUALITY), 0))
                        if okj:
                            dec = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                                               cv2.IMREAD_COLOR)
                            m = _stage_dst_face_mat(meta_f, 256)
                            if dec is not None and m is not None:
                                xseg_dst = cv2.warpAffine(dec, m, (256, 256),
                                                          flags=cv2.INTER_CUBIC)
                    faces.append({"k": int(k), "lm": lm_f, "meta": meta_f,
                                  "crop": crop, "dst_u8": xseg_dst,
                                  "dfm": "", "dfm_in": None, "pred_u8": None,
                                  "prd_u8": None, "dstdfm_u8": None,
                                  "xseg_prd_u8": None, "xseg_dst_u8": None})
                ticks.add("align", time.perf_counter() - t)
                # 归组（有人脸库才真的算 embedding；没库就是一次索引读）
                t = time.perf_counter()
                models, no_swap, plan = _vid_models_for_frame(cfg, job.frame, job.boxes,
                                                              job.idx)
                ticks.add("embed", time.perf_counter() - t)
                job.plan = plan
                if plan is not None and not models:
                    job.faces = []          # 有库但这帧一张都不像：整帧保持原像素
                    job.rec["branch"] = "library-none"
                elif not faces:
                    job.faces = []
                    job.rec["branch"] = "copy"
                else:
                    job.rec["branch"] = "library" if plan is not None else "single"
                    skip = set(int(i) for i in (no_swap or []))
                    use = []
                    for f in faces:
                        i = int(f["k"])
                        if i in skip:
                            continue        # 谁都不像的脸：保持原像素，不兜给任何模型
                        midx = _model_for_face(i, models)
                        m = (models[midx] if models else {}) or {}
                        f["dfm"] = str(m.get("dfm") or cfg["dfm"])
                        f["name"] = str(m.get("name") or cfg["model"])
                        # ③ 换脸那一刀的输入：同一条 get_transform_mat(lm, S, WHOLE_FACE)
                        # + INTER_CUBIC warp + **uint8 整帧 warp 再 /255**（和 ③ 逐字同序）。
                        # 模型尺寸按「这张脸归谁的 dfm」定（多模型时各不相同）。
                        try:
                            sz = int(_dfm_session(f["dfm"])[2])
                            fmat = get_transform_mat(np.asarray(f["lm"], np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            warped = cv2.warpAffine(job.frame, fmat, (sz, sz),
                                                    flags=cv2.INTER_CUBIC)
                            f["dfm_in"] = np.clip(warped.astype(np.float32) / 255.0, 0, 1)
                        except Exception as e:                 # noqa: BLE001
                            got["skipped"] += 1
                            _vid_log("快速导出：第 %d 帧第 %d 张脸 warp 失败：%s"
                                     % (job.idx, i, str(e)[:100]))
                            continue
                        use.append(f)
                    job.faces = use
                job.faces = job.faces or []
                if not job.faces:
                    _put(q_cmp, job)
                else:
                    # 每张脸都带上「这一帧一共几张」：GPU 段按 dfm 攒批，**可能先收到第 2 张、
                    # 后收到第 1 张**（不同 dfm 分开 flush）。只有生产者知道总数，收到的张数
                    # 凑齐才把这一帧交给合成 —— 否则 2 张脸的第一张一到就提前合成，
                    # 第二张的脸就丢了（这是原型里没暴露、但生产上必须堵的洞）。
                    nf = len(job.faces)
                    for f in job.faces:
                        _stage_tick("dfm")           # 交接点：对齐 → dfm/xseg（每张脸一次）
                        if not _put(q_gpu, (job, f, nf)):
                            break
        except BaseException as e:                             # noqa: BLE001
            _fail("prep", e)
        finally:
            with cnt_lock:
                cnt["prep"] -= 1
                last = cnt["prep"] == 0
            if last:
                _put(q_gpu, END)

    # ---- ④ dfm + ⑤ XSeg：按 dfm 攒批，一次前向；回填到各自的帧 ----
    oom_cap = {}                    # dfm_path -> OOM 退档后的硬上限（优先级最高）

    def _ladder_per_face(dfm_path) -> dict:
        """阶梯实测的 {batch: 每张 ms}（_pred_pick_batch 留下的那几行）。"""
        meta = batch_meta.get(dfm_path) or {}
        out = {}
        for row in (meta.get("ladder") or []):
            try:
                out[int(row.get("batch"))] = float(row.get("per_face_ms"))
            except (TypeError, ValueError, KeyError):
                continue
        return out

    def _target_for(dfm_path) -> int:
        """这一路 dfm 一次前向喂几张：攒批目标（默认 8），但**实测说不行就让开**。

        为什么不是硬写 8：攒批只在「GPU 是瓶颈」时值钱。416 的阶梯实测
        B1=29.1 / B4=28.7 / **B8=31.0~31.5**（每张），8 张比 4 张每张贵 5~11%；
        硬攒到 8 的端到端实测是 **-11%**（fused_ab_after_416：15.85 vs 17.82）。
        规则（三行）：
          · 阶梯没量到目标档            → 用目标（没数据就别猜）
          · 目标档每张 ≤ 最优 × 1.02    → 用目标（384：B8 就是最优 → 8）
          · 目标档每张 > 最优 × 1.02    → 用阶梯选的那档（416：用 4），日志说一次原因
        一句话：**目标 8 是「想攒到多少」的意图，不是「必须塞 8 张」的命令**。
        OOM 退档（oom_cap）优先级最高，压在所有规则之上。
        """
        b = int(_batch_for(dfm_path) or 1)
        cap_now = int(oom_cap.get(dfm_path) or _PRED_BATCH_CAP)
        tgt = max(b, int(batch_target))
        if tgt > b:
            per = _ladder_per_face(dfm_path)
            best = min(per.values()) if per else None
            at = per.get(tgt)
            if best and at is not None and at > best * 1.02:
                with batch_lock:
                    noted = bool(_batch_note.get(dfm_path))
                    _batch_note[dfm_path] = True
                if not noted:
                    _vid_log("快速导出：攒批目标 %d 张，但阶梯实测 B%d=%.1fms/张 比最优 "
                             "%.1fms 贵 %.0f%% → 按实测用 %d 张"
                             % (tgt, tgt, at, best, 100.0 * (at / best - 1.0), b))
                tgt = b
        return max(1, min(int(_PRED_BATCH_CAP), cap_now, tgt))

    def _gpu():
        pend = {}
        pend_frames = {}
        born = {}                       # dfm_path -> 「这一批第一张脸」进来的时刻
        arrive = []                     # 最近若干张脸的到达时刻（≈ 到达率，见 _due）
        t_gpu0 = time.perf_counter()    # GPU 段起点：「这一路空着多少」的分子

        def _arrival_rate(now) -> float:
            """最近 1 秒里来了几张脸（**定窗**，不用「张数/跨度」）。

            为什么不用跨度：一帧带 3 张脸时那 3 张的跨度接近 0，除出来的到达率是假的；
            定窗口径只会**低估**高速率（样本上限 96 张），估低了就早送 —— 早送永远不亏。
            """
            n = 0
            for t in reversed(arrive):
                if now - t > 1.0:
                    break
                n += 1
            return float(n)

        def _idle_pct(now) -> float:
            """GPU 段到现在空着的比例（%）：dfm/xseg 之外的时间都算空着。"""
            el = now - t_gpu0
            if el <= 0.0:
                return 0.0
            busy = float(ticks.raw("dfm")) + float(ticks.raw("xseg"))
            return max(0.0, 100.0 * (1.0 - busy / el))

        def _due(now):
            """q_gpu 已经空了的这一拍：哪几路该现在送、为什么（见 _FUSED_BATCH_ADAPT）。

            只在「队列空」时调用 —— 队列里还有脸时它们下一拍就进来，从队列里抢批会把
            一批拆成两批。手里有脸、外面一张都没有，等下去只有两种结果：真能攒满（等）
            或者攒不满（那 20ms 就是纯延迟税）。
            """
            out = []
            if not pend:
                return out
            rate = _arrival_rate(now)
            idle_pct = _idle_pct(now)
            idle_ok = bool(bstat["calls"] >= 4 and (now - t_gpu0) >= 0.5)
            for p in list(pend):
                part = pend.get(p) or []
                if not part:
                    continue
                tgt = _target_for(p)
                waited = now - float(born.get(p, now))
                left = max(0.0, batch_wait - waited)
                if len(part) + rate * left < tgt:
                    # ① 剩下的窗口里最多再来 rate*left 张，凑不满 → 现在送
                    out.append((p, "starve"))
                elif idle_ok and idle_pct >= _FUSED_ADAPT_IDLE_PCT:
                    # ② GPU 段本来就空着：攒批换不来吞吐，只是在给每张脸加延迟
                    out.append((p, "idle"))
                elif waited >= batch_wait:
                    out.append((p, "timeout"))
            return out

        def flush(only=None, why="full"):
            for dfm_path in (list(pend) if only is None else [only]):
                part = pend.get(dfm_path) or []
                if not part:
                    continue
                del pend[dfm_path]
                t_born = born.pop(dfm_path, None)
                if t_born is not None:
                    # 「这一批从第一张脸进来等到现在」= 这一批交的延迟税。自适应打开后
                    # 它应该只剩上游自己的流水线延迟，不再是那条 20ms 的超时。
                    bstat["wait_ms"] = float(bstat.get("wait_ms") or 0.0) + \
                        max(0.0, time.perf_counter() - float(t_born)) * 1000.0
                bstat["why"][why] = int(bstat["why"].get(why) or 0) + 1
                # 攒批目标同时决定「什么时候送」和「一次塞几张」：攒到 8 就一次前向 8 张
                # （这正是 B 配置要的东西）。显存不够时的退档走 oom_cap，优先级最高。
                step = _target_for(dfm_path)
                i = 0
                while i < len(part):
                    chunk = part[i:i + step]
                    i += step
                    faces = [c[1] for c in chunk]
                    sz = size_px_of(dfm_path)
                    arr = np.empty((len(faces), sz, sz, 3), np.float32)
                    for k, f in enumerate(faces):
                        if f.get("dfm_in") is None:
                            # 兜底：绝不让 None/NaN 进网络（早先漏算 warp 时就是靠这个
                            # 变成一片糊的——静默穿过 TRT 再从合成里冒出来，极难查）
                            raise _BadInput("第 %d 张脸没有 dfm 输入（warp 没做）"
                                            % int(f.get("k") or 0), 500)
                        arr[k] = f["dfm_in"]
                    t = time.perf_counter()
                    try:
                        outs = _dfm_run_batch(_dfm_session(dfm_path), arr)
                    except Exception as e:                     # noqa: BLE001
                        if not (_is_oom(e) and len(faces) > 1):
                            raise
                        h = max(1, len(faces) // 2)
                        _vid_log("快速导出：dfm batch=%d 显存不够（%s）→ 退到 %d 再试"
                                 % (len(faces), str(e)[:80], h))
                        with batch_lock:
                            batch_of[dfm_path] = max(1, min(int(batch_of.get(dfm_path) or h), h))
                        # 攒批目标也要跟着退，否则下一次照样攒到 8 张再撞 OOM
                        oom_cap[dfm_path] = max(1, min(int(oom_cap.get(dfm_path) or _PRED_BATCH_CAP), h))
                        outs = []
                        for j in range(0, len(faces), h):
                            outs.extend(_dfm_run_batch(_dfm_session(dfm_path),
                                                       arr[j:j + h]))
                    ticks.add("dfm", time.perf_counter() - t)
                    bstat["calls"] += 1
                    bstat["faces"] += len(faces)
                    bstat["max"] = max(bstat["max"], len(faces))
                    key = str(len(faces))
                    bstat["hist"][key] = int(bstat["hist"].get(key) or 0) + 1
                    # 量化单独记一笔（以前它摊在 dfm 与 xseg 之间的空档里，谁都不知道它多贵）：
                    # pred 的 clip 是恒等的（_dfm_run_batch 已 clip(0,1)），prd/dst 的遮罩不 clip 会改像素。
                    t_u8 = time.perf_counter()
                    for k, f in enumerate(faces):
                        prd, dst = outs[k][1], outs[k][2]
                        f["pred_u8"] = _fused_q_u8(outs[k][0], _u8_fast, clip=False)
                        f["prd_u8"] = (None if prd is None
                                       else _fused_q_u8(prd, _u8_fast))
                        f["dstdfm_u8"] = (None if dst is None
                                          else _fused_q_u8(dst, _u8_fast))
                    ticks.add("u8", time.perf_counter() - t_u8)
                    t = time.perf_counter()
                    if need_xseg:
                        # 和 ④ 遮罩阶段同一个口径：一张脸一次 mask_batch([pred, dst])
                        for f in faces:
                            imgs = [f["pred_u8"]]
                            if f.get("dst_u8") is not None:
                                imgs.append(f["dst_u8"])
                            try:
                                masks = xseg.mask_batch(imgs, out_size=256) or []
                            except Exception as e:             # noqa: BLE001
                                raise _BadInput("遮罩推理失败：%s" % (str(e)[:160]), 500)
                            t_um = time.perf_counter()
                            f["xseg_prd_u8"] = (
                                _fused_q_u8(masks[0], _u8_fast)
                                if len(masks) > 0 and masks[0] is not None else None)
                            f["xseg_dst_u8"] = (
                                _fused_q_u8(masks[1], _u8_fast)
                                if len(masks) > 1 and masks[1] is not None else None)
                            ticks.add("u8_mask", time.perf_counter() - t_um)  # ⊂ xseg 那一笔

                    ticks.add("xseg", time.perf_counter() - t)
                    for job, _f in chunk:
                        rec = pend_frames.get(id(job))
                        if rec is not None:
                            rec[0] -= 1
                            if rec[0] <= 0:
                                pend_frames.pop(id(job), None)
                                _stage_tick("xseg")      # 交接点：dfm/xseg → 合成
                                _put(q_cmp, job)

        def size_px_of(dfm_path):
            return int(_dfm_session(dfm_path)[2])

        try:
            while True:
                t_wait = time.perf_counter()
                item = _poll(q_gpu, 0.005)
                ticks.add("gpu_wait", time.perf_counter() - t_wait)
                if item is END:
                    flush(why="end")
                    for _i in range(n_cmp):
                        _put(q_cmp, END)
                    break
                if item is None:
                    # 这一拍没有新脸进来：手里这些要不要现在送出去？
                    #   · 自适应（默认）：投影攒不满 / GPU 本来就空着 → 立刻送，不等超时
                    #   · DFL_FUSED_BATCH_ADAPT=0 或超时=0：老口径，只在到点时才送
                    if not pend or batch_wait <= 0.0:
                        continue
                    now = time.perf_counter()
                    if _FUSED_BATCH_ADAPT:
                        for _p, _why in _due(now):
                            flush(_p, why=_why)
                    else:
                        for _p in list(pend):
                            if now - float(born.get(_p, now)) >= batch_wait:
                                flush(_p, why="timeout")
                    continue
                job, f, nf = item
                arrive.append(time.perf_counter())
                if len(arrive) > 192:
                    del arrive[:-96]
                if id(job) not in pend_frames:
                    # 生产者给的「这一帧一共几张脸」：收到的回填进度按它算（见 _prep 的说明）
                    pend_frames[id(job)] = [int(nf)]
                if not pend.get(f["dfm"]):
                    born[f["dfm"]] = time.perf_counter()
                pend.setdefault(f["dfm"], []).append((job, f))
                if len(pend[f["dfm"]]) >= _target_for(f["dfm"]):
                    flush(f["dfm"], why="full")
        except BaseException as e:                             # noqa: BLE001
            _fail("gpu", e)

    # ---- ⑤ 合成（CPU：compose_frame；GPU：compose_frame_gpu，DFL_COMPOSE_GPU=1）----
    def _compose_args_gpu(job):
        """GPU 入口直接吃 uint8 —— 省掉 CPU 上的 5 次 astype(/255)（实测 1.2ms/张脸）。"""
        return [{"landmarks": f["lm"], "pred": f["pred_u8"],
                 "prd_dfm": f.get("prd_u8"), "dst_dfm": f.get("dstdfm_u8"),
                 "xseg_prd": f.get("xseg_prd_u8"), "xseg_dst": f.get("xseg_dst_u8")}
                for f in (job.faces or [])]

    def _compose_args_cpu(job):
        args = []
        for f in (job.faces or []):
            prd = f.get("prd_u8")
            dst = f.get("dstdfm_u8")
            xp = f.get("xseg_prd_u8")
            xd = f.get("xseg_dst_u8")
            args.append({
                "landmarks": f["lm"],
                "pred": f["pred_u8"].astype(np.float32) / 255.0,
                "prd_dfm": (None if prd is None
                            else prd.astype(np.float32) / 255.0),
                "dst_dfm": (None if dst is None
                            else dst.astype(np.float32) / 255.0),
                "xseg_prd": (None if xp is None
                             else xp.astype(np.float32) / 255.0),
                "xseg_dst": (None if xd is None
                             else xd.astype(np.float32) / 255.0),
            })
        return args

    def _compose():
        try:
            while True:
                t_wait = time.perf_counter()
                job = _get(q_cmp)
                ticks.add("cmp_wait", time.perf_counter() - t_wait)
                if job is END:
                    return
                t = time.perf_counter()
                args = job.faces or []          # 只给下面的记账用（faces=张数）
                if gpu_cmp["on"]:
                    try:
                        out_frame = compose_frame_gpu(job.frame, _compose_args_gpu(job),
                                                      comp_cfg, job.timings)
                    except Exception as e:                     # noqa: BLE001
                        # 整段退回 CPU（不逐帧来回切：同一单里两套口径比慢更糟）。
                        # 回退后 n_cmp 已经是 1，会慢一些，但结果与「一开始就用 CPU」相同。
                        _vid_log("快速导出：GPU 合成失败（%s）→ 本单退回 CPU 合成"
                                 % str(e)[:200])
                        gpu_cmp["on"] = False
                        gpu_cmp["why"] = str(e)[:200]
                        out_frame = compose_frame(job.frame, _compose_args_cpu(job),
                                                  comp_cfg, job.timings)
                else:
                    try:
                        out_frame = compose_frame(job.frame, _compose_args_cpu(job),
                                                  comp_cfg, job.timings)
                    except Exception as e:                     # noqa: BLE001
                        raise _BadInput("第 %d 帧合成失败：%s" % (job.idx, str(e)[:160]), 500)
                ticks.add("compose", time.perf_counter() - t)
                if dump_frames and int(job.idx) in dump_frames:
                    _fused_dump(cfg, job, out_frame)
                job.rec["faces"] = len(args)
                job.rec["swapped"] = len(args)
                job.rec["no_face"] = not args
                if not args:
                    # 这一帧没有脸：换脸/遮罩都没碰它，合成也是原帧直通 → 走旁路那条线
                    _STAGE_METER.note_no_face()
                job.frame = None
                _stage_tick("compose")           # 交接点：合成 → PNG 编码
                _put(q_enc, (job, out_frame))
        except BaseException as e:                             # noqa: BLE001
            _fail("compose", e)
        finally:
            with cnt_lock:
                cnt["compose"] -= 1
                last = cnt["compose"] == 0
            if last:
                for _i in range(n_enc):
                    _put(q_enc, END)

    # ---- ⑥ PNG 编码 + 落盘（无损，level 同 ⑤ 合成阶段）----
    # 为什么**不需要**重排/单写者：帧图是 `%06d.png` 定长序列，文件名自带帧号，
    # 谁先写完都不影响 ffmpeg 读到的是第几帧（分段链的 merge 阶段也是多 worker 乱序写盘）。
    # 之前那版照抄原型留了个"唯一写者按帧号顺序"的环节，代价是：写线程死等最慢的那一帧，
    # 缓冲一满就把 encode 线程堵住、sem 不释放、解码停摆 —— 实测 detect 只有 61% 占用、
    # 整条线 10.75 帧/秒（detect 单独跑是 52.8ms/帧 = 18.9 帧/秒的硬底）。
    def _encode():
        try:
            while True:
                t_wait = time.perf_counter()
                item = _get(q_enc)
                ticks.add("enc_wait", time.perf_counter() - t_wait)
                if item is END:
                    return
                job, img = item
                if raw_w is not None:
                    # raw 管道：这里**不编码、不落盘**，只把这一帧交给唯一写者。
                    # 仍然留 3 条线程：memoryview/连续性这步可以并行（1.15ms/帧），
                    # 而 pipe.write 只有写者线程做（帧序不能乱，见 _FusedRawWriter）。
                    # sem/计数/进度都由写者那一侧收（帧真正出去的时刻才算出去）。
                    t = time.perf_counter()
                    buf = memoryview(img) if img.flags["C_CONTIGUOUS"] else                         memoryview(np.ascontiguousarray(img))
                    ticks.add("encode", time.perf_counter() - t)
                    _stage_tick("encode")        # 交接点：合成 → 排队等写管道
                    if not raw_w.put(job, buf.cast("B")):
                        return
                    continue
                t = time.perf_counter()
                data = _stage_imencode_png(img, png_level)
                ticks.add("encode", time.perf_counter() - t)
                _stage_tick("encode")            # 交接点：PNG 编码 → 写盘
                t = time.perf_counter()
                _stage_write_atomic(frames_dir / ("%06d.png" % job.n), data)
                ticks.add("write", time.perf_counter() - t)
                with live_lock:
                    live["n"] -= 1
                try:
                    sem.release()
                except ValueError:
                    pass
                _frame_done(job)
        except BaseException as e:                             # noqa: BLE001
            _fail("encode", e)

    # dfm 批大小阶梯（_pred_pick_batch）会真的跑几次空张量前向来量耗时：留在流水线里会让
    # 开头 1~2 秒的 GPU 被它占着（120 帧的片子那是 15% 的开销）。单模型这条最常见的路
    # 提前在这里跑掉 —— 既预热了 session，也让测速窗口干净。
    try:
        _batch_for(str(cfg["dfm"]))
    except Exception as e:                                     # noqa: BLE001
        _vid_log("快速导出：dfm 批大小探测失败（%s）→ 流水线里再试" % str(e)[:120])

    cv2_prev = 0
    cv2_set = 0
    if cv2_threads > 0:
        try:
            cv2_prev = int(cv2.getNumThreads())
            if cv2_prev != int(cv2_threads):
                cv2.setNumThreads(int(cv2_threads))
                cv2_set = int(cv2_threads)
        except Exception:                                      # noqa: BLE001
            cv2_prev, cv2_set = 0, 0

    # ---- 异构 session 池：探测 → 微基准 → 配比（见 server/_sess.py）----
    # 默认走池（n_det = plan.gpu + plan.cpu）；DFL_SESS_PLAN=0 退回单路老行为。
    det_sess = []
    det_plan = {}
    if _sess.enabled():
        try:
            det_plan = _fused_det_plan(det_name, probe.get("w"), probe.get("h"), {
                "gpu": cfg.get("sess_gpu"), "cpu": cfg.get("sess_cpu"),
                "strategy": cfg.get("sess_strategy"), "level": cfg.get("sess_level")})
            for _i in range(max(0, int(det_plan.get("gpu") or 0))):
                _o = _detect_for(det_name, "gpu" if _i == 0 else "gpu2")
                if _o is not None:
                    det_sess.append(("gpu", _o))
            for _j in range(max(0, int(det_plan.get("cpu") or 0))):
                _o = _detect_for(det_name, "cpu")
                if _o is not None:
                    det_sess.append(("cpu", _o))
            _vid_log("快速导出 检测池：%s" % _sess.plan_line(det_plan))
            _vid_log("快速导出 检测池 EP：%s"
                     % _det_ep_line(det_plan.get("det_ep") or det_pol))
            _vid_log("快速导出 检测池 基准：GPU %s ms / CPU %s ms（1080p 零图，2 张取中位）"
                     "· 实际起来 %d 路（GPU %d / CPU %d）"
                     % ((det_plan.get("bench") or {}).get("gpu_ms"),
                        (det_plan.get("bench") or {}).get("cpu_ms"), len(det_sess),
                        sum(1 for _e, _o in det_sess if _e.startswith("gpu")),
                        sum(1 for _e, _o in det_sess if _e == "cpu")))
        except Exception as e:                                 # noqa: BLE001
            det_sess = []
            _vid_log("快速导出 检测池：推导失败（%s）→ 退回单路" % str(e)[:160])
    n_det = len(det_sess) or 1

    # ---- 流水线实况：lane 数 = **实际**线程数（不是写死的）----
    # 合成池 n_cmp 条（t0..t7）、编码 n_enc 条（e0..e5）、dfm/xseg 各一个 session 就 1 条、
    # det 有几路异构 session 就有几条。xseg 不用（遮罩不来自模型）时给空列表 ——
    # 界面画成「不用」，不假装有一条在跑的 lane。
    _STAGE_METER.reset(lanes={
        "frames": ["解码"],
        "det": ["det#%d" % _i for _i in range(n_det)],
        "dfm": ["dfm#0"],
        "xseg": (["xse#0"] if need_xseg else []),
        "compose": ["t%d" % _i for _i in range(n_cmp)],
        "encode": ["e%d" % _i for _i in range(n_enc)],
    })
    # 每个阶段的「下游队列」= 列间那句「在飞 N」（inflight 直接读 qsize()）
    _STAGE_METER.bind({"frames": q_dec, "det": q_det, "dfm": q_gpu,
                       "xseg": q_cmp, "compose": q_enc})

    # ---- 成片出口：raw 管道要先起 ffmpeg，再起唯一写者 ----
    # 顺序要紧：ffmpeg 必须在**第一帧之前**就位（管道是阻塞流，写者一上来就写），
    # 而写者必须比编码线程先从下游起（和队列那条规则一样：先有消费者）。
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + ".part" + out.suffix)
    raw_w = None
    raw_proc = None
    raw_cmd = None
    t_raw0 = time.perf_counter()
    if rawpipe:
        _pw, _ph = int(probe.get("w") or 0), int(probe.get("h") or 0)
        if not _pw or not _ph:
            raise _BadInput("raw 管道要知道源视频宽高，但探测结果里没有（probe=%r）"
                            % (probe,), 500)
        plan_frames = max(1, int(end_i) - int(start_i) + 1)
        t_audio = ((start_i - 1) / fps) if fps > 0 else 0.0
        plan_dur = (plan_frames / fps) if fps > 0 else 0.0
        raw_cmd = _fused_pipe_cmd(part, video, fps, _pw, _ph,
                                  audio_ss=(None if full else t_audio),
                                  audio_t=(None if full else plan_dur),
                                  limit=(None if full else int(plan_frames)))
        _vid_log("快速导出：raw 管道 %dx%d @ %.6g 帧/秒 · %d 帧 · 音轨 %s"
                 % (_pw, _ph, fps, plan_frames,
                    "整段 copy" if full else "-ss %.4f -t %.4f" % (t_audio, plan_dur)))
        _vid_log("快速导出 管道命令：%s"
                 % " ".join('"%s"' % a if " " in str(a) else str(a) for a in raw_cmd))
        # ★ ffmpeg 必须**让路**：它在流水线跑的同时编（老路是跑完再编），默认优先级会跟检测、
        # 合成、dfm 抢同一批核。实测（150 帧 · 同进程交替 · 没让路那组）：raw 的 pipeline 比
        # png 慢 5~8%（中位 22.1 → 20.4 帧/秒 那条曲线），让路之后两者持平（5.87 vs 5.76 s）。
        # BELOW_NORMAL 只改调度、**不改码流**：x264 的输出跟线程数有关（实测 -threads 8 就换成
        # 另一份 md5），所以这里宁可用优先级换核，也不动 -threads —— 成片逐字节相同是硬约束。
        _prio = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) \
            if str(os.environ.get("DFL_FUSED_PIPE_PRIO") or "1").strip().lower() \
            not in ("0", "off", "false", "no") else 0
        raw_proc = subprocess.Popen(
            raw_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(_app_mod, "_NO_WINDOW", 0) | _prio)
        raw_w = _FusedRawWriter(raw_proc, plan_frames, sem, live, live_lock, ticks, stop,
                                _frame_done, _report)
        raw_w.start()

    ths = {}
    t_all = time.time()
    try:
        ths["decode"] = threading.Thread(target=_decode, name="fused-dec", daemon=True)
        if n_det <= 1:
            ths["detect"] = threading.Thread(target=_detect, name="fused-det", daemon=True)
        else:
            for _i, (_ep, _o) in enumerate(det_sess):
                ths["detect%d" % _i] = threading.Thread(
                    target=_detect, args=(_i, _o, n_det), name="fused-det%d-%s" % (_i, _ep),
                    daemon=True)
        for i in range(n_prep):
            ths["prep%d" % i] = threading.Thread(target=_prep, name="fused-prep%d" % i,
                                                 daemon=True)
        ths["gpu"] = threading.Thread(target=_gpu, name="fused-gpu", daemon=True)
        for i in range(n_cmp):
            ths["cmp%d" % i] = threading.Thread(target=_compose, name="fused-cmp%d" % i,
                                                daemon=True)
        for i in range(n_enc):
            ths["enc%d" % i] = threading.Thread(target=_encode, name="fused-enc%d" % i,
                                                daemon=True)
        # 从下游往上游起：先有消费者，生产者才不会一上来就撞满队列
        _det_keys = (["detect"] if n_det <= 1
                     else ["detect%d" % i for i in range(n_det)])
        for k in (["enc%d" % i for i in range(n_enc)]
                  + ["cmp%d" % i for i in range(n_cmp)] + ["gpu"]
                  + ["prep%d" % i for i in range(n_prep)] + _det_keys + ["decode"]):
            ths[k].start()
        # 全部线程退出 = 全片跑完；中途取消就置 stop 让所有段退出
        while True:
            if _fused_cancelled():
                stop.set()
                break
            if not any(t.is_alive() for t in ths.values()):
                break
            time.sleep(0.1)
    finally:
        wall = time.time() - t_all
        stop.set()
        for k, th in ths.items():
            try:
                th.join(timeout=10.0)
            except Exception:                                  # noqa: BLE001
                pass
        alive = [k for k, th in ths.items() if th.is_alive()]
        try:
            proc = gen.get("proc")
            if proc is not None:
                _kill_proc_tree(proc)
        except Exception:                                      # noqa: BLE001
            pass
        if cv2_set:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                                  # noqa: BLE001
                pass
        # raw 管道：成功、失败、取消**都要收**（绝不留一个在跑的 ffmpeg 进程）。
        # 正常收尾 = 等写者把手里写完 → 关 stdin（EOF）→ 等 x264 排空；失败/取消 = 直接杀。
        if raw_w is not None:
            _force = bool(errors) or bool(alive) or _fused_cancelled()
            try:
                raw_w.finish(force=_force)
            except Exception as e:                             # noqa: BLE001
                if not errors:
                    errors.append("raw 管道收尾: %r" % (e,))

    written = int(got["frames"] or 0)
    if _fused_cancelled():
        raise _StageCancel()
    if errors:
        raise _BadInput("快速导出失败（%s）" % (errors[0][:400],), 500)
    if alive:
        raise _BadInput("快速导出异常收尾：还有线程没退出（%s）" % "、".join(sorted(alive)), 500)
    if written <= 0:
        raise _BadInput("一帧都没解出来：%s" % video.name, 422)

    sec = round(time.time() - t_all, 2)
    rate = written / max(1e-6, sec)
    per_stage = ticks.dump(written)
    busy = {k: ticks.raw(k) for k in per_stage}
    util = {
        # 等待（背压）也算一列：某一段 100% 却整条线慢，看 *_wait 就知道谁在等谁
        "dec_wait": round(100.0 * busy.get("dec_wait", 0.0) / max(1e-6, wall), 1),
        "det_wait": round(100.0 * busy.get("det_wait", 0.0) / max(1e-6, wall), 1),
        "gpu_wait": round(100.0 * busy.get("gpu_wait", 0.0) / max(1e-6, wall), 1),
        "cmp_wait": round(100.0 * busy.get("cmp_wait", 0.0) / max(1e-6, wall * n_cmp), 1),
        "enc_wait": round(100.0 * busy.get("enc_wait", 0.0) / max(1e-6, wall * n_enc), 1),
        "decode": round(100.0 * busy.get("decode", 0.0) / max(1e-6, wall), 1),
        "detect": round(100.0 * busy.get("detect", 0.0) / max(1e-6, wall), 1),
        "det_prep": round(100.0 * busy.get("det_prep", 0.0) / max(1e-6, wall), 1),
        "align": round(100.0 * busy.get("align", 0.0) / max(1e-6, wall * n_prep), 1),
        "embed": round(100.0 * busy.get("embed", 0.0) / max(1e-6, wall * n_prep), 1),
        "dfm": round(100.0 * busy.get("dfm", 0.0) / max(1e-6, wall), 1),
        "xseg": round(100.0 * busy.get("xseg", 0.0) / max(1e-6, wall), 1),
        "compose": round(100.0 * busy.get("compose", 0.0) / max(1e-6, wall * n_cmp), 1),
        "encode": round(100.0 * busy.get("encode", 0.0) / max(1e-6, wall * n_enc), 1),
        "write": round(100.0 * busy.get("write", 0.0) / max(1e-6, wall), 1),
    }
    _vid_log("快速导出：流水线跑完 %d 帧 · %.2f s · %.2f 帧/秒 · 帧缓冲 %d（峰值在飞 %d 帧）"
             % (written, sec, rate, budget, int(live["peak"])))
    _vid_log("快速导出 profile（每帧 ms）：%s"
             % " · ".join("%s=%.2f" % (k, v) for k, v in per_stage.items()))
    _vid_log("快速导出 各段占用：%s"
             % " · ".join("%s=%.0f%%" % (k, v) for k, v in util.items()))
    _vid_log("快速导出 dfm 批次：调用 %d 次 / %d 张脸 / 平均 %.1f 张 · 最大 %d · 分布 %s"
             % (bstat["calls"], bstat["faces"],
                bstat["faces"] / max(1, bstat["calls"]), bstat["max"],
                dict(sorted(bstat["hist"].items(), key=lambda kv: int(kv[0])))))
    _vid_log("快速导出 dfm 攒批：目标 %d 张 / 超时 %dms / 模式 %s · 送出原因 %s"
             "（full=攒满 · starve=攒不满 · idle=GPU 空着 · timeout=到点 · end=收尾）"
             " · 平均等批 %.1f ms（这一批第一张脸进来 → 送出）"
             % (batch_target, int(round(batch_wait * 1000.0)),
                "自适应" if _FUSED_BATCH_ADAPT else "固定超时",
                dict(sorted((bstat.get("why") or {}).items())),
                float(bstat.get("wait_ms") or 0.0) / max(1, int(bstat["calls"]))))
    mem1 = _proc_mem_mb()
    _vid_log("快速导出 内存（本进程）：流水线前 RSS %s / 提交 %s MB → 流水线后 RSS %s / 提交 %s MB"
             " · 峰值 RSS %s MB（帧缓冲 %d 帧 × %.2f MB = %.0f MB 是设计内的常驻，"
             "ORT 建会话时那 1.5GB 权重要过一遍内存，是峰值的主要来源）"
             % (_fmt_mb(mem0.get("rss")), _fmt_mb(mem0.get("private")),
                _fmt_mb(mem1.get("rss")), _fmt_mb(mem1.get("private")),
                _fmt_mb(mem1.get("peak_rss")), budget, _fused_frame_mb(size_px),
                budget * _fused_frame_mb(size_px)))
    if written != total:
        _vid_log("快速导出：实解出 %d 帧（源信息说 %d 帧）→ 按实解帧数编码"
                 % (written, total))
    # 把这一轮实测的「段占用」留给下一轮的 derive()（这就是「按实测调配比」的回路）
    try:
        _SESS_BUSY["detect"] = round(float(util.get("detect") or 0.0) / 100.0, 3)
        _SESS_PLAN_LAST["detect"] = dict(det_plan or {})
        # 顺带学两件事（见 _sess.record_gain 的注释）：
        #   ① 1x1 / 1x2（多挂一路 CPU 检测 session）值不值；
        #   ② 0x1（检测整段走 CPU EP）值不值 —— 这是 DFL_DET_EP=auto 的门②。
        # 以前这行的参数少传了一个（把 rate 当 cpu_n、把 WORKSPACE 当 fps），
        # float(Path) 每次都抛 TypeError 被下面 except 吞掉 —— 于是 gain:detect
        # 永远写不进磁盘，「按实测学」这条回路其实一直没通电。
        _g = _sess.record_gain("detect", int((det_plan or {}).get("gpu") or 0),
                               int((det_plan or {}).get("cpu") or 0), rate, WORKSPACE)
        if _g.get("gain_pct") is not None:
            _vid_log("快速导出 检测池 实测增益：%s%%（基线 1×GPU %.2f → 本次 %s %.2f 帧/秒）→ "
                     "自动档%s"
                     % (_g.get("gain_pct"), float(_g.get("base") or 0),
                        _g.get("now_tag") or "-", float(rate),
                        ("会开 CPU session" if _g.get("auto_cpu_ok")
                         else "仍然只开 GPU（净增 <5%）")))
        if _g.get("cpu_ep_gain_pct") is not None:
            _vid_log("快速导出 检测 EP 实测：CPU %.2f 帧/秒 对基线 1×GPU %.2f = %+.2f%% → "
                     "DFL_DET_EP=auto 的门②%s（工作区 _sess_bench.json 的 gain:detect）"
                     % (float(_g.get("cpu_ep_fps") or 0), float(_g.get("base") or 0),
                        float(_g.get("cpu_ep_gain_pct") or 0),
                        ("已通过（≥+5%%），以后 auto 会选 CPU EP"
                         if _g.get("auto_cpu_ep_ok")
                         else "未通过（<+5%），auto 固定留在 GPU")))
    except Exception:                                       # noqa: BLE001
        pass
    _vid_stats(done=written, total=written, fps=round(rate, 2), eta_s=0,
               frame_budget=budget, fused=True, per_stage_ms=per_stage,
               stage_util_pct=util, batch=batch_of, batch_actual=dict(bstat),
               sess_plan={"detect": dict(det_plan or {}), "n_detect": int(n_det),
                          "sessions": [{"ep": _e} for _e, _o in det_sess]},
               pipeline={
                   "threads": {"decode": 1, "detect": 1, "prep": n_prep, "gpu": 1,
                               "compose": n_cmp, "encode": n_enc},
                   "threads_src": {
                       "prep": _src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT,
                                       "DFL_FUSED_PREP"),
                       "compose": _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                                          _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP")},
                   "batch_target": int(batch_target),
                   "batch_wait_ms": int(round(batch_wait * 1000.0)),
                   "batch_adapt": bool(_FUSED_BATCH_ADAPT),
                   "batch_wait_mean_ms": round(
                       float(bstat.get("wait_ms") or 0.0) / max(1, int(bstat["calls"])), 1),
                   "decode_mode": dec_mode,
                   "compose_backend": ("gpu" if gpu_cmp["on"] else "cpu"),
                   "queue_depth": qd, "budget": budget, "cv2_threads": cv2_threads})
    _stage_progress("fused", _VID_PCT_SWAP,
                    ("快速导出：%d 帧跑完（x264 已同步编完），正在排空" if raw_w is not None
                     else "快速导出：%d 帧跑完，开始编码（%d 帧 @ %.6g fps）")
                    % ((written,) if raw_w is not None else (written, written, fps)),
                    frames_done=written, frames_total=written,
                    frame_fps=round(rate, 2))

    # ---- ⑦ 编码 ----
    # raw 管道：x264 在流水线跑的时候就已经把片子编完了，这里只剩「排空 + 判退出码」；
    # 老路：此刻才串行跑 ffmpeg 读那串 PNG（实测 1.68s / 150 帧 = 11.2ms/帧，管道把这段省掉）。
    dur = (written / fps) if fps > 0 else 0.0
    t_audio = ((start_i - 1) / fps) if fps > 0 else 0.0
    if raw_w is not None:
        # ffmpeg 的退出码只能在**上面那几条 errors 之后**判：真正的根因（合成崩了、没解出帧）
        # 优先级更高，先报那个；这里再兜「ffmpeg 自己死了」这种情况，并且要带 stderr 尾巴 ——
        # 「成片没了，但日志里一句话都没有」是这个项目最不该再出现的排障体验。
        if int(raw_w.rc or 0) != 0:
            try:
                if part.is_file():
                    part.unlink()
            except OSError:
                pass
            # Windows 上 rc 是 DWORD：-22 (EINVAL) 会显示成 4294967274，读起来像另一个东西
            _rc = int(raw_w.rc or 0)
            _rc = _rc - 2 ** 32 if _rc > 2 ** 31 else _rc
            raise _BadInput(
                "成片编码失败（ffmpeg 退出码 %s，已写进管道 %d 帧）：%s"
                % (_rc, int(raw_w.n),
                   (raw_w.tail_text() or raw_w.err or "ffmpeg 没有输出")[-800:]), 500)
        if raw_w.err:
            raise _BadInput("raw 管道异常（成片作废）：%s" % str(raw_w.err)[:400], 500)
        _pipe_s = time.perf_counter() - t_raw0
        _drain = max(0.0, _pipe_s - raw_w.seconds)
        # encode_seconds 报的是**这一单为「编码」多花的墙钟** = 排空时间（x264 那部分是并发跑的，
        # 没有多花墙钟）。写者在线时长单独放在 res["pipe"]，日志两个都写出来，谁也不糊弄谁。
        enc = {"code": 0, "tail": raw_w.tail_text(), "seconds": round(_drain, 2),
               "part": str(part), "cmd": list(raw_cmd or [])}
        _vid_log("快速导出：raw 管道收尾 —— 写 %d 帧 · x264 与流水线并发（写者在线 %.2fs，"
                 "单次写最长 %.1f ms）· 收尾排空 %.2fs · 管道全程 %.2fs"
                 % (int(raw_w.n), raw_w.seconds, 1000.0 * raw_w.write_max, _drain, _pipe_s))
        if int(raw_w.n) != int(plan_frames):
            _vid_log("快速导出：★ 计划 %d 帧、实际写进管道 %d 帧（源比请求的少？）—— 成片按实际帧数收"
                     % (int(plan_frames), int(raw_w.n)))
    else:
        cmd = _vid_cmd(str(frames_dir / "%06d.png"), part, video, fps, img_start=1,
                       audio_ss=(None if full else t_audio),
                       audio_t=(None if full else dur),
                       limit=(None if full else int(written)))

        def _on_prog(done):
            pct = _VID_PCT_SWAP + (100 - _VID_PCT_SWAP) * min(1.0, done / max(1, written))
            _stage_progress("encode", pct, "编码 %d/%d 帧" % (done, written),
                            frames_done=min(done, written), frames_total=written)

        enc = _vid_encode(cmd, part, written, on_progress=_on_prog,
                          cancelled=_fused_cancelled)
        if enc["code"] != 0:
            try:
                if part.is_file():
                    part.unlink()
            except OSError:
                pass
            raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s"
                            % (enc["code"], (enc["tail"] or "")[-600:]), 500)
        _vid_log("快速导出 编码命令：%s"
                 % " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd))
    if not part.is_file() or part.stat().st_size <= 0:
        raise _BadInput("编码跑完了却没有产物：%s" % part, 500)
    os.replace(str(part), str(out))
    mb = round(out.stat().st_size / 1048576.0, 2)
    _vid_log("快速导出：编码完成 %s（%.2f MB · %d 帧 · %.3f s · %.4f s 墙钟）"
             % (out.name, mb, written, dur, enc["seconds"]))

    # ---- 断言：全程没碰 <视频>_cache/（这是 fused 和分段的关键区别）----
    if (not cache_pre) and cache_dir.exists():
        raise _BadInput("快速导出不该创建 %s，但它出现了 —— 说明有代码偷偷写了 cache"
                        % cache_dir, 500)
    _vid_log("快速导出：<视频>_cache/ %s" % ("本来就在（没动它）" if cache_pre else "全程没建 ✔"))

    return {"frames": written, "seconds": sec, "fps": round(rate, 2),
            "encode_seconds": enc["seconds"], "mb": mb, "duration": round(dur, 6),
            "out": str(out), "frame_budget": budget, "budget": bud,
            "per_stage_ms": per_stage, "stage_util_pct": util,
            "batch": dict(batch_of), "batch_actual": dict(bstat),
            "peak_live": int(live["peak"]),
            "cache_dir": str(cache_dir), "cache_created": False,
            "audio_ss": (None if full else round(t_audio, 6)),
            "range": {"start": start_i, "end": end_i, "total": total, "full": full},
            "frames_dir": ("" if frames_dir is None else str(frames_dir)),
            "sink": ("rawpipe" if raw_w is not None else "png"),
            "pipe": (None if raw_w is None else
                     {"frames": int(raw_w.n), "writer_seconds": round(raw_w.seconds, 2),
                      "write_max_ms": round(1000.0 * raw_w.write_max, 1),
                      "rc": int(raw_w.rc or 0), "cmd": list(raw_cmd or [])}),
            "threads": {"prep": n_prep, "compose": n_cmp, "encode": n_enc},
            "threads_src": {
                "prep": _src_of(cfg, ("pred_readers",), _FUSED_PREP_DEFAULT,
                                "DFL_FUSED_PREP"),
                "compose": _src_of(cfg, ("merge_workers", "merge_threads", "threads"),
                                   _FUSED_CMP_DEFAULT, "DFL_FUSED_CMP")},
            "batch_target": int(batch_target),
            "batch_wait_ms": int(round(batch_wait * 1000.0)),
            "decode_mode": dec_mode,
            # 解码腿的三个旋钮（见 _FUSED_ENV_* 的注释与 experiments/dec_lane/REPORT.md）
            "decode_threads": int(dec_threads or 0),
            "readinto": bool(readinto_on),
            "det_prep_where": ("decode" if det_prep_on
                               else ("detect-pool" if det_prep_pool else "detect")),
            "detector": str(det_name), "detector_src": str(det_src),
            "det_prep": bool(det_prep_on), "det_probe": dict(_dp or {})}


def _fused_worker(cfg: dict):
    """跑一次快速导出，并把终态落到两个 status（老 /export/status + /stage/status）。"""
    ok, err, cancelled = True, "", False
    t0 = time.time()
    mem0 = _proc_mem_mb()          # 导出前（起线程之前那一刻）：收尾要拿它做差
    frames_dir = ""
    try:
        _stage_progress("fused", 0, "快速导出启动…",
                        frames_done=0, frames_total=int(cfg.get("range_total") or 0))
        res = _fused_run(cfg) or {}
        frames_dir = str(res.get("frames_dir") or cfg.get("frames_dir") or "")
        with _merge_job_lock:
            _merge_job["encode"] = {"seconds": res.get("encode_seconds"), "code": 0,
                                    "out": str(res.get("out")), "mb": res.get("mb")}
        cfg["_fused_result"] = res
        _vid_log("快速导出完成：%s（%d 帧 · %.2f 帧/秒 · 编码 %.2fs）"
                 % (res.get("out"), res.get("frames"), res.get("fps"),
                    res.get("encode_seconds") or 0))
    except _StageCancel:
        ok, cancelled = False, True
        err = "已取消（快速导出不支持续跑，需整段重跑）"
    except _BadInput as e:
        ok, err = False, str(e.msg)[:1200]
    except Exception as e:                                     # noqa: BLE001
        traceback.print_exc()
        ok, err = False, "%s: %s" % (e.__class__.__name__, str(e)[:600])
    finally:
        # 流水线实况收单：不再涨，但**保留** lanes/ticks —— 进度页那张图在「完成」
        # 那一刻要还是满的（收单就清空的话，界面会当场塌成一排占位格）。
        _STAGE_METER.stop()
        # 强制尝试失败：先回滚现场（放掉本进程会话 + 重算当下的账 + 改写 err）再收尾 ——
        # 见 _vram_oom_rollback / _vram_forced_rollback。顺序反了用户看到的就是
        # 「导出失败」，而不是「确实不够，已回滚」。
        if (not ok) and (not cancelled) and cfg.get("vram_force"):
            _rb = _vram_forced_rollback("快速导出", _vram_model_mb(cfg.get("dfm")),
                                        {"mode": "frames",
                                         "frames": int(cfg.get("frame_budget_want") or 0),
                                         "batch": int(cfg.get("fused_batch") or 0),
                                         "procs": 1}, err)
            if _rb:
                cfg["vram_oom"] = _rb
                err = str(_rb.get("msg") or err)
        # active 保持 true 直到下面的收尾全部做完（删临时帧 / 清半截成片 / 写终态）。
        # 取消之后 /export/status 的 running 立刻变 false（既有语义，看门狗靠它），
        # 所以「能不能再点一次导出」要看 active —— 临时帧还没删完就再起一单，
        # 会在 %TEMP% 里留下一个没人管的 dfl_fused_* 目录。
        pass
        # 临时帧目录：成功、失败、取消**都删**（不存在续跑这回事）
        tmp = str(cfg.get("frames_dir") or frames_dir or "")
        removed = False
        if tmp:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                removed = not Path(tmp).exists()
            except OSError:
                removed = False
            _vid_log("快速导出：临时帧目录 %s（%s）" % (tmp, "已删除" if removed else "没删干净"))
        # 半截成片（.part）绝不能留着被当成结果
        try:
            op = Path(str(cfg.get("out") or ""))
            if op.name:
                p = op.with_name(op.stem + ".part" + op.suffix)
                if p.is_file():
                    p.unlink()
                    _vid_log("快速导出：删掉半截成片 %s" % p.name)
        except OSError:
            pass
        res = cfg.get("_fused_result") or {}
        cache_state = clear_face_group_cache(
            "快速导出完成" if ok else ("快速导出取消" if cancelled else "快速导出失败"))
        wall = round(time.time() - t0, 2)
        bud = (res.get("frame_budget") if res else None) or cfg.get("frame_budget")
        # 显示层用的是 res["budget"]（_fused_run 落实过的计划）；res 没有（失败得太早）就
        # 用 _start_video_export 闸门算的那一份 —— status 里始终有 vram_plan 可查
        vplan = dict((res.get("budget") if res else None) or cfg.get("vram_plan") or {})
        _oom = dict(cfg.get("vram_oom") or {})
        if _oom:
            # 强制尝试 OOM 之后：status 里的 vram_plan 换成**回滚后当下**的这一份，
            # 前端拿到它就能把「被拒态」的数字重画一遍（不是 OOM 之前那份旧账）。
            vplan = dict(_oom.get("plan") or vplan)
            vplan["forced_failed"] = True
        if vplan:
            vplan["cpu"] = bool(_vram_cpu_on())
            # 合成后端回显：前端 / 日志能直接看出这一单是 CPU 还是 GPU 合成的
            vplan["compose_backend"] = str(cfg.get("_compose_backend") or "cpu")
            vplan["compose_threads"] = int(cfg.get("_compose_threads") or 0)
            if cfg.get("_compose_why"):
                vplan["compose_why"] = str(cfg.get("_compose_why"))
        with _STAGE_LOCK:
            _STAGE_JOB.update({
                "running": False, "ok": bool(ok), "cancelled": bool(cancelled),
                "phase": ("done" if ok else ("cancelled" if cancelled else "failed")),
                "finished_at": time.time(), "error": "" if ok else err,
                "vram_oom": (_oom or None),
                "error_detail": "" if ok else "\n".join(
                    list(_STAGE_JOB.get("log") or [])[-30:]),
                "pct": (100 if ok else int(_STAGE_JOB.get("pct") or 0)),
                "stage_pct": (100 if ok else int(_STAGE_JOB.get("stage_pct") or 0)),
                "stage": "fused", "stage_index": -1,
                "msg": ("快速导出完成：%d 帧 · %.2f 帧/秒"
                        % (int(res.get("frames") or 0), float(res.get("fps") or 0.0))
                        if ok else ("已取消（快速导出不支持续跑）" if cancelled
                                    else "失败：" + err[:80])),
                "out_video": str(res.get("out") or cfg.get("out") or ""),
                "cache_dir": "", "cache_removed": removed, "cache": cache_state,
                "frame_budget": bud, "resumed": False, "seconds": {"fused": wall},
                "counts": {"frames": int(res.get("frames") or 0), "faces": 0,
                           "pred": 0, "masks": 0},
                "meta": {}, "vram_plan": vplan, "on_cpu": bool(_vram_cpu_on()),
                "compose_backend": str(cfg.get("_compose_backend") or "cpu"),
                "compose_threads": int(cfg.get("_compose_threads") or 0),
            })
        with _merge_job_lock:
            _merge_job.update({
                "running": False, "ok": bool(ok) and not cancelled,
                "cancelled": bool(cancelled), "finished_at": time.time(),
                "pct": (100 if ok else int(_merge_job.get("pct") or 0)),
                "pct_at_failure": (None if ok else int(_merge_job.get("pct") or 0)),
                "stage": (6 if ok else -1),
                "msg": ("合成完成" if ok else
                        ("已取消（快速导出不支持续跑，需整段重跑）" if cancelled else "合成失败")),
                "error": ("" if ok else err),
                "error_detail": ("" if ok else "\n".join(_log_tail(
                    list(_merge_job.get("log") or []), 30))[:4000]),
                "cache": cache_state, "frame_budget": bud, "active": False,
                "vram_plan": vplan, "on_cpu": bool(_vram_cpu_on()),
                "compose_backend": str(cfg.get("_compose_backend") or "cpu"),
                "compose_threads": int(cfg.get("_compose_threads") or 0),
                "vram_oom": (_oom or None),
            })
        _vid_log("快速导出结束：%s · %.2f s%s"
                 % ("成功" if ok else ("已取消" if cancelled else "失败"), wall,
                    ("（不支持续跑）" if cancelled else "")))
        # 收尾把本进程的 dfm/XSeg 会话放掉（成功/失败/取消都放）：留着它下一次预览
        # 会被自己的后端堵死 —— 见 _vram_release_inproc 的注释。预览 worker 不重载，
        # 下次预览再懒加载。
        try:
            _vram_release_inproc("快速导出结束")
        except Exception:                                   # noqa: BLE001
            pass
        # ---- 内存收尾（三步，顺序有讲究）----
        # ① 预览 worker 那份会话：导出结束后它就是纯占着的东西（显存 2.1~2.4GB /
        #    内存 1.3GB），而导出用的是本进程那份 —— 两份互不相干，必须分开还。
        #    DFL_FUSED_FREE_PREVIEW=0 可保留（下一次预览更快，代价是显存一直占着）。
        try:
            _rel = ({"did": False, "why": "DFL_FUSED_FREE_PREVIEW=0"}
                    if str(os.environ.get("DFL_FUSED_FREE_PREVIEW") or "1").strip().lower()
                    in ("0", "false", "no", "off") else _free_preview_sessions("快速导出结束"))
        except Exception as e:                              # noqa: BLE001
            _rel = {"did": False, "why": repr(e)}
        # ② 断完引用先收一遍垃圾：循环引用（闭包/异常栈）只有 gc 能收，光 drop 不够
        try:
            import gc as _gc
            _gc.collect()
            _gc.collect()
        except Exception:                                   # noqa: BLE001
            pass
        # ③ 先量「收尾后、回收前」—— 这才是**真的还剩下的**（对象都释放了，但堆把页
        #    攥在进程里）；再把工作集交给系统。两笔分开打，谁也不许拿回收冒充释放。
        mem1 = _proc_mem_mb()
        _vid_log("快速导出 内存（收尾后 · 回收前）：%s" % _mem_delta(mem0, mem1))
        try:
            _trim = _trim_ws("快速导出收尾")
        except Exception as e:                              # noqa: BLE001
            _trim = {"did": False, "why": repr(e)}
        mem2 = _proc_mem_mb()
        _vid_log("快速导出 内存（导出前后 · 最终）：%s" % _mem_delta(mem0, mem2))
        if (_trim or {}).get("did"):
            _vid_log("快速导出 内存：工作集回收 RSS %s → %s MB（提交 %s → %s MB）—— 回收只还"
                     "工作集，提交只随对象释放而降；两者不是一回事"
                     % (_fmt_mb((_trim.get("before") or {}).get("rss")),
                        _fmt_mb((_trim.get("after") or {}).get("rss")),
                        _fmt_mb((_trim.get("before") or {}).get("private")),
                        _fmt_mb((_trim.get("after") or {}).get("private"))))
        if (_rel or {}).get("did"):
            _vid_log("快速导出 内存：预览 worker 已卸下常驻 dfm（%d 个 · %.0f ms）—— 那是另开的"
                     "一份会话，导出用不到它"
                     % (int(_rel.get("unloaded") or 0), float(_rel.get("ms") or 0)))
        else:
            _vid_log("快速导出 内存：预览 worker 未卸（%s）" % str((_rel or {}).get("why") or "-")[:120])
        _vid_close_log()


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 fused 与 _app_mod、_sess、app、clear_face_group_cache、compose、media_encoder、media_frames、meter、pipeline_video、preview、stages、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from .. import _sess

from ..app import (
    WORKSPACE, _merge_job, _merge_job_lock, _merge_settings_to_config,
)

from .._face_group import clear_face_group_cache

from ._shared import _DEFAULT_DETECTOR

from .compose import (
    _cache_snap, _detect_boxes, _detector, _detector_val, _landmarks_model,
)

from .._media.encoder import (
    _vid_cmd, _vid_probe,
)

from .._media.frames import (
    _VIDEO_EXT,
)

from . import knobs as _knobs

from .meter import (
    _FusedTicks, _STAGE_METER, _STAGE_MS_SRC, _stage_tick,
)

from .pipeline_video import (
    _VID_PCT_SWAP, _log_tail, _vid_cancelled, _vid_close_log, _vid_encode, _vid_log,
    _vid_models_for_frame, _vid_record, _vid_stats,
)

from .preview import (
    _model_for_face, _models_of, _xseg_of,
)

from .pipeline_stages import (
    _MERGE_CV2_THREADS_DEFAULT, _MERGE_PNG_LEVEL, _PRED_BATCH_DEFAULT, _STAGE_JOB,
    _STAGE_JPEG_QUALITY, _STAGE_LOCK, _StageCancel, _align_detector, _dfm_run_batch,
    _dfm_session, _mask_src, _merge_worker_count, _stage_dfm_pair, _stage_dst_face_mat,
    _stage_imencode_png, _stage_out_path, _stage_progress, _stage_range, _stage_range_norm,
    _stage_write_atomic, _stage_xseg,
)

from .vram import (
    _SESS_BUSY, _SESS_PLAN_LAST, _VRAM_BASE_PX, _VRAM_CPU_SLOW, _VRAM_ENV_CPU, _det_ep_apply,
    _det_ep_line, _det_ep_resolve, _det_prep, _det_quick_profile, _detect_bench,
    _detect_boxes_ctx, _detect_for, _detect_on, _detect_on_ctx, _dfm_size_hint,
    _free_preview_sessions, _is_oom, _mem_delta, _pred_pick_batch, _vram_apply, _vram_cpu_on,
    _vram_forced_rollback, _vram_frame_mb, _vram_free_mb, _vram_log_lines, _vram_model_mb,
    _vram_plan, _vram_refuse_msg, _vram_release_inproc,
)
