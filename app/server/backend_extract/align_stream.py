# -*- coding: utf-8 -*-
"""边解码边对齐：视频 → 人脸图（DFLJPG），原帧只在内存里走一趟（原 frames.py:386-1040）。

这是**前端提取页**的落地实现（唯一会写 aligned/*.jpg 的那一条）。和另外两条"提取"的区别：
  · 快速导出（融合流水线）在 backend_merge/pipeline_fused.py 里内联做检测+对齐，不落盘；
  · 项目导出走 4.0 的 MergeStudio/core/export_pipeline.py（自带 detector，写 sqlite）。
三者共用 _vision/detect.py 与 _vision/align.py 的几何，别在这里再写一份 round/插值。

三级流水（每级一个线程）：reader 解码 → detector 检测 → writer 对齐+JPEG+DFLJPG。
稳态吞吐取三者最大值，不是三者之和。
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

from .._common import jobs as _jobs
from .._common.bootstrap import _NO_WINDOW
from .._common.jobs import JOB, _job_fail, _merge_cache, _merge_prep, _merge_prep_lock
from .detectors import _get_fast_detector, _get_heavy_detector, _get_landmarks106
from .media import _dfl_project_scan, _extract_frames, _imread_any, _open_video
from .resume import SRC_FP_NAME, _fp_diff, _note_resume_src, _read_src_fp, _src_fingerprint

# 落盘线程池：emit 段实测占 wall 的 81~85%（改前），是整条流水线最慢的一级。
# 里面 cv2.imencode 会释放 GIL，所以多线程真能并行（不是假并行）。
# 每批 _flush 结束时等本批写完，计数才 +，进度和续跑语义不变。
from concurrent.futures import ThreadPoolExecutor as _TPE
_WPOOL = _TPE(max_workers=4, thread_name_prefix="dfljpg")


def _align_stream(src: str, face_dir: Path, fps: float = 2.0, rotate: int = 0,
                  quality: str = "high", skip: int = 0, src_fps: float = 0.0,
                  strict_align: "bool | None" = None) -> int:
    """ffmpeg 边解码边对齐：原帧只在内存里走一趟，不落盘。

    三级流水线（每级一个线程，实测瓶颈在 H.264 解码 + 管道传输）：
        reader   解码 bgr24 rawvideo，扔进 q_frames
        detector 检测（GPU），扔进 q_detect
        writer   对齐 + JPEG 编码 + 写 DFLJPG
    稳态吞吐取三者最大值，而不是三者之和。

    尺寸：对齐边长 = 检测框长边 × 2，无上下限（越界部分填纯黑）。
    质量档只决定检测器：high=RetinaFace(det_10g)，fast=TinyMog。
    谁的档位谁负责 —— 不做跨检测器兜底（那会让 0 脸的帧跑两遍推理，
    既慢又让人误判"两个检测器一样快"）。

    strict_align（= --strict-align）：None 跟 DFL_ALIGN_GATE 走；True 强制 strict
    （5 点退化的脸真拒、计进 rejected_gate；rejected 仍是「外点 + 闸门」的总数），
    False 强制 warn。**默认 warn**：图照写，
    坏脸只记进 <face_dir>/_bad_frames.txt —— 既有行为一个字都不变。
    """
    import queue
    import threading
    import numpy as np
    import cv2
    from .._align import (COARSE_SIZE, align_mode, align_face, align_face_68,
                         append_bad_frame, coarse_align_from_5, gate_limits,
                         gate_mode, landmarks68_in_source, pose_from_kps,
                         write_dfljpg)

    # 谁的档位谁负责：high 只用 RetinaFace，fast 只用 TinyMog。
    # 以前会"0 脸就换另一个检测器兜底"，结果是同一批帧跑两次推理，
    # 速度被拖慢、还让人误以为两个检测器一样快。
    fast = _get_fast_detector() if quality != "high" else None
    heavy = _get_heavy_detector() if quality == "high" else None
    primary = heavy if quality == "high" else fast
    if primary is None:
        # 「模型没找到」也是失败：以前只改 msg 然后 return 0，前端收到的仍是「完成」。
        _job_fail("_align_stream/检测器",
                  RuntimeError("检测器不可用：%s 的模型没找到（weights/ 根下的 %s）"
                               % (("RetinaFace(det_10g)" if quality == "high" else "TinyMog"),
                                  "retinaface_10g.onnx" if quality == "high" else "tinymog.onnx")))
        return 0
    # TinyMog 走批量接口，需要提前把帧 letterbox 到 640；RetinaFace 自己会做
    need_canvas = bool(getattr(primary, "needs_canvas", False))

    lm106 = _get_landmarks106()

    # ---------------- 对齐模式 ----------------
    # DFL_ALIGN=68（默认）：5 点只算「宽松粗裁」→ 2D106 出 106 点 → 挑 68 → 回原图 →
    #                      align_face_68 做终裁（INTER_CUBIC）。落盘的是这一步。
    # DFL_ALIGN=5        ：老的 5 点仿射，只留作回退 / 对照，输出**没有**关键点。
    # 68 模式下 106 不可用时绝不退回 5 点：宁可不产出，也不写一张假的"对齐脸"。
    def _log(msg):
        """打印日志永不让抽取挂掉：控制台是 GBK 时，个别字符编不出来就降级。"""
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            enc = (getattr(sys.stdout, "encoding", "") or "utf-8")
            print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)

    _amode = align_mode()
    # 5 点退化闸门：warn（默认，只记清单）/ strict（真拒）/ off
    _gm = gate_mode()
    if strict_align is not None:
        _gm = "strict" if strict_align else "warn"
    # 本次运行的闸门档位与计数落进 JOB：/api/extract/status 要能读到，
    # 不然界面只有日志和清单两条路（自检脚本读 _LAST_EXTRACT，人只能翻文件）。
    with JOB.lock:
        JOB.gate = 0
        JOB.gate_mode = _gm if _amode == "68" else "off"
    if _gm != "off" and _amode == "68":
        _gs, _gi = gate_limits()
        _log("[align] 5 点退化闸门：%s · face_span<%.2f 且 iod_chip<%.2f"
             "（%s）"
             % (_gm, _gs, _gi,
                "真拒，计进 rejected_gate" if _gm == "strict"
                else "只记录到 _bad_frames.txt，图照写"))
    if _amode == "5":
        _log("[align] 警告：DFL_ALIGN=5，走 5 点仿射（旧路径，仅供回退/对照）。"
              "这条路上的输出没有 68 点关键点（landmarks_source=none），"
              "4.0 引擎的 SampleLoader / Sorter 读不到 landmarks —— 要训练就别用它。")
    elif lm106 is None:
        _log("[align] 失败风险：DFL_ALIGN=68 但 2D106 模型不可用"
              "（weights/insightface_2d106.onnx 不存在或加载失败）："
              "本次每一张脸都会跳过并计入 skipped，绝不退回 5 点。")
    else:
        _log("[align] 对齐模式：68 点仿射（5 点粗裁 %dpx ×%.2f → 2D106 → 终裁）"
              % (COARSE_SIZE, 1.30))

    face_dir.mkdir(parents=True, exist_ok=True)

    base_faces = len(list(face_dir.glob("*_*.jpg")))      # 目录里已有的对齐脸（续跑时不为 0）
    done_idx = set()
    for f in face_dir.glob("*_*.jpg"):
        head = f.stem.split("_")[0]
        if head.isdigit():
            done_idx.add(int(head))

    vf = []
    if rotate in (90, 180, 270):
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[rotate])
    # 必须始终带一个滤镜：imageio_ffmpeg 在 output_params 为空或只有 null 时会拼出
    # 坏命令并杀掉 ffmpeg（实测稳定复现 0 帧）。即便 fps == 源帧率，也显式写出来。
    vf.append("fps=%g" % max(0.01, float(fps)))

    # ---- 输入：视频文件 / 图片目录 / DFL 项目目录 ----
    # 这一点本来是支持的（本模块 import 里那三个 _dfl_project_scan / _extract_frames /
    # _imread_any 就是当时用的），后来收敛成只认视频，目录就被挡在门外了。
    # 目录模式不经过 ffmpeg：逐张读成 bgr24 字节，喂给同一条检测/对齐流水线。
    from .. import _hwaccel      # 视频模式用它探测硬解，结尾报状态也要读
    _src_p = Path(str(src))
    _dir_mode = _src_p.is_dir()
    if _dir_mode:
        _files = []
        try:
            # DFL 项目目录（有 aligned/）优先，其次才是普通图片目录
            _names, _srcdir, _kind = _dfl_project_scan(_src_p)
            if _names and _srcdir:
                _files = [Path(_srcdir) / _n for _n in _names]
        except Exception:
            _files = []
        if not _files:
            _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            try:
                _files = sorted(x for x in _src_p.iterdir()
                                if x.is_file() and x.suffix.lower() in _exts)
            except OSError:
                _files = []
        if not _files:
            _job_fail("_align_stream/输入",
                      RuntimeError("这个目录里没有图片：%s" % src))
            return 0
        def _as_bgr(_im):
            # 检测器吃的是 bgr24：灰度/带 alpha 的图必须先归一成 3 通道，
            # 否则 raw 长度不等于 W*H*3，_decode 会把每一帧都当坏帧丢掉（表现就是 0 张脸）。
            if _im is None:
                return None
            if _im.ndim == 2:
                return cv2.cvtColor(_im, cv2.COLOR_GRAY2BGR)
            if _im.shape[2] == 4:
                return cv2.cvtColor(_im, cv2.COLOR_BGRA2BGR)
            return _im

        _probe = _as_bgr(_imread_any(_files[0]))
        if _probe is None:
            _job_fail("_align_stream/输入",
                      RuntimeError("第一张图读不出来（损坏或不是图片）：%s" % _files[0]))
            return 0
        H, W = _probe.shape[:2]
        total_est = len(_files)
        dur = 0.0

        def _file_gen():
            for _f in _files[1:]:
                _im = _as_bgr(_imread_any(_f))
                if _im is None:
                    continue
                if _im.shape[0] != H or _im.shape[1] != W:
                    _im = cv2.resize(_im, (W, H), interpolation=cv2.INTER_AREA)
                yield _im.tobytes()

        gen = _file_gen()
        # ★ 目录/图片输入喂的是 **BGR**（W*H*3），视频输入喂的是 **NV12**（W*H*3/2）。
        #   _decode 里那句 `len(raw) != W*H*3//2` 只对视频成立 —— 目录模式下每一帧
        #   都被当成「长度不对」continue 掉，于是 1080p 图 6220800 字节 vs 期望的
        #   3110400，**一张脸都切不出来**，结尾还报「完成：解码 0 帧 · 0 张人脸」。
        #   routes.py 明确放行目录、UI 也提示可以拖目录，所以这条很容易踩。
        _dir_input = True
        first_frame = _probe.tobytes()
        _resume_n = len(done_idx)
        _diff = _note_resume_src(face_dir, src, W, H, 0.0, _resume_n)
    else:
        _dir_input = False
        in_params = _hwaccel.input_params(str(src))     # 触发（或命中）一次探测，下面复用
        norm = _hwaccel.normalize_filter()              # 硬解：先回 yuv420p 再转 BGR
        if norm:
            vf.append(norm)

        # 视频模式的 setup（滤镜/打开/分辨率/时长/续跑指纹）
        out_params = ["-vf", ",".join(vf)]
        # NV12：1080p 一帧 3.1MB（bgr24 是 6.2MB）。实测图传占 wall 的 84%，
        # 先把带宽砍一半；多出来的一次 cvtColor（实测 2.06ms）比少传 3MB 便宜。
        #
        # ★ 这里**是一次有意的质量让步**，写清楚免得以后有人以为是笔误：
        #   管道传 NV12 时，YUV→RGB 由 OpenCV 的 cvtColor 做（UV 平面用 bilinear
        #   上采样）；传 bgr24 时由 ffmpeg 的 swscale 做（bicubic）。两条路解出来的
        #   像素**不逐位相同**：本机 1080p 实测 mean 1.43 / max 19（G 通道最明显，
        #   因为它同时吃 U 和 V），换算成比例约 0.56%。差全在色度边缘（头发、嘴唇
        #   轮廓这类色彩突变处），平坦区域几乎不差。
        #
        #   为什么提取能接受：对齐图后面还要过一次 warpAffine(INTER_CUBIC)，那是比
        #   色度上采样大得多的重采样，0.56% 在这里被稀释掉了；而且图像是训练数据，
        #   不是最终产物。
        #   为什么合成器**不能**这么做：见 pipeline_fused.py:630 —— 它的产物要贴着
        #   原视频，frames/%06d.png 有 fused_decode_parity.py 逐位盯着（mean 0.0000
        #   / max 0），任何非零差都会挂回归。那边宁可慢 3.9ms/帧也强制软解 bgr24。
        #
        #   要退回逐位精确：把这里的 pix_fmt 换成 "bgr24" 即可（read 会慢回
        #   7.5ms/帧，整条 wall 多约 3.5ms/帧）。
        gen, meta, first_frame = _open_video(str(src), out_params, first=True,
                                            in_params=in_params, pix_fmt="nv12")
        W, H = meta.get("size") or (0, 0)
        if not W or not H:
            try: gen.close()
            except Exception: pass
            _job_fail("_align_stream/解码", RuntimeError("解不出分辨率，视频打不开或没有视频流：%s" % src))
            return 0
        dur = float(meta.get("duration") or 0)
        total_est = max(1, int(dur * max(0.01, float(fps)))) if dur else 1
        _resume_n = len(done_idx)
        _diff = _note_resume_src(face_dir, src, W, H, dur, _resume_n)


    if _resume_n and _diff:
        _log("[align] 续跑命中 %d 帧，源素材与上次不同（%s）—— 这些帧号可能属于另一段素材"
             % (_resume_n, _diff))

    S = 640
    JOB.begin(0, total_est, "边解码边对齐（%dx%d · %.2f fps · 不落盘）" % (W, H, fps))
    # JOB.begin 不会把 running 置 True，而 reader 线程用 JOB.running 判断是否继续。
    # 少了这一行，reader 第一次 next 前就 break → 永远 0 帧（实测踩过）。
    with JOB.lock:
        JOB.running = True

    PROF = bool(os.environ.get("DFL_EXTRACT_PROFILE"))
    # 每级"真正干活"与"等队列/等管道"的累计秒数，用来判断是不是伪并行
    cost = {"read": 0.0, "put1": 0.0, "get1": 0.0, "det": 0.0, "put2": 0.0,
            "get2": 0.0, "emit": 0.0,
            # 对齐内部再拆四段（只在 DFL_EXTRACT_PROFILE=1 时打印，但一直记账：
            # 单次 _t() 是几十纳秒，和一次 warp 比可以忽略）
            "coarse": 0.0, "lm106": 0.0, "final": 0.0, "jpeg": 0.0}
    qmax = {"q1": 0, "q2": 0}

    def _t():
        return time.perf_counter()

    # TinyMog 的 batch 维是动态的：攒一批一起推理，能把每帧的固定开销摊薄。
    # RetinaFace 的引擎是单图 profile，老老实实一张一张来。
    # TinyMog 引擎的优化 profile 是 batch 1..4，超过 4 会被 TRT 拒绝（实测 batch=8 直接报错）
    BATCH = min(8, int(os.environ.get("DFL_EXTRACT_BATCH") or (8 if need_canvas else 1)))

    # 队列容量：DFL_EXTRACT_Q（默认 6，且**不低于一整批**，否则批永远凑不满）。
    # 两种模式共用同一个值，否则对照不公平。
    # 有界就是背压：解码线程 put 满了就阻塞，常驻内存与视频长度无关。
    from .. import _pipeline
    q_cap = max(_pipeline.capacity(), BATCH)
    q_frames: "queue.Queue" = queue.Queue(maxsize=q_cap)
    q_detect: "queue.Queue" = queue.Queue(maxsize=q_cap)
    stop = threading.Event()
    # rejected        = 拒收总数（外点 + 闸门，保持老读法兼容）
    # rejected_outlier= 68 点外点检查判「检测异常」的张数
    # rejected_gate   = 闸门判坏且被 strict 真拒的张数
    # gate            = 5 点退化闸门判坏脸的张数（warn 下照样落盘，strict 下同一张也算 rejected_gate）
    # 为什么要拆：完成行原来把两笔账写成同一句「其中外点判异常 N」，而闸门拒收的
    # 恰恰是 resid 最小的那批（见 docs/30 §2）—— 混着写等于告诉人「是残差判的」。
    stats = {"seen": 0, "made": 0, "fallback": 0, "skipped": 0,
             "rejected": 0, "rejected_outlier": 0, "rejected_gate": 0, "gate": 0}
    # 帧速度（JOB.fps = 解到第几帧 / el）与 SSE 的 elapsed 共用 JOB.t0：
    # 前端算「张/秒」用的是 elapsed，两边各用各的 t0 会让两个速度一快一慢。
    t0 = JOB.t0 or time.time()
    lock = threading.Lock()

    def _canvas(img):
        h, w = img.shape[:2]
        sc = min(S / float(w), S / float(h))
        nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        cv = np.full((S, S, 3), 128, np.uint8)
        cv[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return cv, sc

    BATCH_106 = max(1, min(8, int(os.environ.get("DFL_EXTRACT_106_BATCH") or 8)))
    # 每项：68 模式 = {path, idx, img, coarse, mat, box, size, pose}
    #       5  模式 = {path, idx, crop, meta, pose}
    # 68 模式必须留着**原帧**引用：终裁是从原图重采样（不是把粗裁图放大），
    # 攒一批 = 最多攥住 BATCH_106 帧（1080p ≈ 6 MB/帧，峰值 ~50 MB），可接受。
    _pending = []
    _skip_dbg = {"n": 0}

    def _skip(idx, why):
        """一张脸没产出：计入 skipped + 日志留一行（前 20 条每张都打，之后每 200 条提醒）。

        为什么必须有：以前 106 跑挂会被静默吞掉，或者退化成"用插值点照样写一张"，
        两种都让「少了一张脸」在人眼里等于「这一帧没有脸」。
        """
        with lock:
            stats["skipped"] += 1
            n = stats["skipped"]
        if _skip_dbg["n"] < 20 or n % 200 == 0:
            _log("[align] 跳过：%s（第 %d 帧）" % (why, idx))
        _skip_dbg["n"] += 1

    _gate_dbg = {"n": 0}

    def _note_gate(idx, path, g):
        """5 点退化：把这张脸记进 <face_dir>/_bad_frames.txt（只记录，不拒、不删）。

        这笔账和外点检查分开记（stats["gate"]）：一个量「点跑飞」，一个量「脸在芯片里
        塌成一团」，两者互不包含 —— 已知坏帧恰恰是外点检查全放行的那批。
        """
        _name = Path(path).name if path else ("%06d" % idx)
        append_bad_frame(face_dir, _name, g)
        n = _gate_dbg["n"]
        if n < 20 or (n + 1) % 200 == 0:
            _log("[align] 5 点退化：%s（第 %d 帧）%s" % (_name, idx, g.get("reason") or ""))
        _gate_dbg["n"] = n + 1

    def _flush():
        """把攒下的脸一次性做 2D106，再统一落盘。

        68 模式：一批粗裁图 → 106 批出 106 点 → 按下标挑 68 → 反变换回原图 →
                align_face_68 终裁。
        5  模式：直接用 _emit 里已经裁好的图（旧行为，逐像素不变）。
        106 没结果的脸 = **不写文件**，计 skipped 并留日志；绝不拿插值点顶替。

        **计数就在这一层**：谁写成功一张，stats["made"] 立刻 +1，JOB.faces 跟着更新。
        以前只有调用方把 _flush/_emit 的返回值累加起来，而流水线模式里 emit 的返回值
        要等 _pipeline.run() 整段跑完才回到 _align_stream —— 于是整段运行期间
        stats["made"] 恒为 0，界面「已对齐人脸 0 张 / 人脸速度 0.0 张/秒」，
        磁盘上却明明在长文件（截图事故的根因）。
        """
        if not _pending:
            return 0
        items = list(_pending)
        _pending.clear()
        lms = [None] * len(items)
        if _amode == "68" and lm106 is not None:
            _t0 = _t()
            try:
                lms = lm106.landmarks68_batch([it["coarse"] for it in items])
            except Exception as e:          # noqa: BLE001 —— 整批失败就整批跳过
                _log("[align] 2D106 批量推理失败（本批 %d 张全部跳过）：%s"
                     % (len(items), e))
                lms = [None] * len(items)
            cost["lm106"] += _t() - _t0
        wrote = 0
        _jfuts = []
        for it, lm68 in zip(items, lms):
            if _amode == "68":
                if lm68 is None:
                    _skip(it["idx"], "拿不到 68 点（2D106 没有输出，不退回 5 点）")
                    continue
                try:
                    _tf = _t()
                    pts = landmarks68_in_source(lm68, it["mat"])
                    crop, meta_f = align_face_68(it["img"], pts, box=it["box"],
                                                 size=it["size"], gate=_gm)
                    cost["final"] += _t() - _tf
                except Exception as e:      # noqa: BLE001
                    _skip(it["idx"], "68 点反变换/仿射失败：%s" % e)
                    continue
                # 5 点退化闸门：判据在 align_face_68 里算完（O(1)），这里只记账/记账单。
                # strict 模式下面 crop 已经是 None，走下面那条 rejected 分支真拒掉。
                _g = (meta_f or {}).get("align_gate") if isinstance(meta_f, dict) else None
                if _g and _g.get("bad"):
                    with lock:
                        stats["gate"] += 1
                        _n_gate = stats["gate"]
                    _note_gate(it["idx"], it["path"], _g)
                    with JOB.lock:              # 锁序：先放 lock 再拿 JOB.lock
                        JOB.gate = _n_gate
                it["img"] = None            # 别把整帧攥在 _pending 里
                if crop is None:
                    # meta_f 此时是「拒收记录」：reject_reason / align_resid_px
                    _why = (meta_f or {}).get("reject_reason") if isinstance(meta_f, dict) else None
                    if _why:
                        with lock:
                            stats["rejected"] += 1
                            if _g and _g.get("bad"):
                                stats["rejected_gate"] += 1
                            else:
                                stats["rejected_outlier"] += 1
                    _skip(it["idx"], _why or "68 点仿射失败（点数/数值不合法）")
                    continue
            else:
                crop, meta_f = it["crop"], it["meta"]
            meta_f["source_filename"] = "%06d" % it["idx"]
            if it.get("pose"):
                meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = it["pose"]
            try:
                _tj = _t()
                # 丢给线程池：JPEG 4:4:4 编码 + pickle 元数据 + 写文件一起并行
                _jfuts.append(_WPOOL.submit(write_dfljpg, it["path"], crop, meta_f))
                cost["jpeg"] += _t() - _tj
                wrote += 1
                JOB.tick_face()             # 瞬时人脸速度用
                with lock:                  # 落盘即计数：跑的过程中就能看到数字在长
                    stats["made"] += 1
            except Exception as e:          # noqa: BLE001 —— 落盘失败也要留痕
                print("[extract] 写 %s 失败：%s" % (Path(it["path"]).name, e), flush=True)
                continue
        # 等本批的落盘真正写完（失败的单独记，不因为一张写坏就丢掉整批的计数）
        for _f in _jfuts:
            try:
                _f.result()
            except Exception as _e:      # noqa: BLE001
                print("[extract] 写盘失败：%s" % str(_e)[:160], flush=True)
        if wrote:
            # 锁序：先放掉 lock 再拿 JOB.lock（_decode 那边是 JOB.lock → 读 stats，反向嵌套会死锁）
            with JOB.lock:
                JOB.faces_new = stats["made"]
                JOB.faces = base_faces + stats["made"]
        return wrote

    def _emit(idx, img, faces):
        # 产出帧速：这里才是「一帧真正处理完」的时刻。reader 那边记的是
        # 「解到第几帧」，它有队列缓冲、会跑在前面（实测 89.5 vs 端到端 79.6）。
        JOB.tick_frame()
        n = 0
        for k, fc in enumerate(faces):
            if not JOB.running:
                break
            box = fc.get("box") or [0, 0, 0, 0]
            if min(box[2] - box[0], box[3] - box[1]) < 32:
                continue
            bw = max(1.0, float(box[2]) - float(box[0]))
            bh = max(1.0, float(box[3]) - float(box[1]))
            side = int(round(max(bw, bh) * 2.0))   # 2 倍框宽，不设上下限
            side -= side % 2
            kps = fc.get("kps") or []
            if len(kps) < 5:
                _skip(idx, "检测器没给 5 点")
                continue
            pose = None
            try:
                _y, _p, _r = pose_from_kps(kps, box)
                if _y is not None:
                    pose = (_y, _p, _r)
            except Exception:
                pose = None
            path = face_dir / ("%06d_%d.jpg" % (idx, k))
            if _amode == "68":
                # 粗裁：只为了喂 2D106，落盘的永远是下面 _flush 里的 68 点终裁图。
                # 画布直接出 192（2D106 的输入尺寸），视野外扩 1.3 倍 —— 三次 warp 里
                # 最便宜的一次，且不再让模型那层多 resize 一遍。
                try:
                    cmat = coarse_align_from_5(kps)
                except Exception:
                    cmat = None
                if cmat is None or not np.all(np.isfinite(cmat)):
                    _skip(idx, "5 点算不出粗对齐矩阵")
                    continue
                _tc = _t()
                coarse = cv2.warpAffine(img, cmat, (COARSE_SIZE, COARSE_SIZE),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                cost["coarse"] += _t() - _tc
                _pending.append({"path": path, "idx": idx, "img": img, "coarse": coarse,
                                 "mat": cmat, "box": box, "size": side, "pose": pose})
            else:
                crop, meta_f = align_face(img, kps, box=box, size=side)
                if crop is None:
                    _skip(idx, "5 点仿射失败")
                    continue
                _pending.append({"path": path, "idx": idx, "crop": crop,
                                 "meta": meta_f, "pose": pose})
            if len(_pending) >= BATCH_106:
                n += _flush()
        return n

    def _pipe_frames():
        """_open_video 预读的那一帧 + 管道剩下的帧，同一序号起点。"""
        if first_frame is not None:
            yield first_frame
        for raw in gen:
            yield raw

    def _decode(put):
        """解码一级：管道读一帧 → put((idx, frame))。流水线与回退路径共用这段。

        续跑（done_idx 里已有的帧）在这里就丢掉：它不需要检测、也没人写文件，
        早点丢省一个队列位；idx 仍按真正读到的帧号走，编号不会错位。
        """
        idx = 0
        it = iter(_pipe_frames())
        while True:
            t_a = _t()
            try:
                raw = next(it)          # ← 真正的管道读取（ffmpeg 解码 + 传输）
            except StopIteration:
                break
            cost["read"] += _t() - t_a
            if stop.is_set() or not JOB.running:
                break
            _want = (W * H * 3) if _dir_input else (W * H * 3 // 2)
            if len(raw) != _want:
                # 视频是 NV12（Y 平面 W*H + UV 交错 W*H/2），目录是 BGR 裸字节。
                # 长度对不上只可能是管道错位/半帧 —— 跳过，但记一笔（以前静默）。
                with lock:
                    stats["bad_len"] = int(stats.get("bad_len") or 0) + 1
                continue
            if idx not in done_idx:
                # NV12 → BGR 一次搞定（比管道里多传 3MB 划算，实测 2~3ms）
                if _dir_input:
                    frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
                else:
                    frame = cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(H * 3 // 2, W),
                                         cv2.COLOR_YUV2BGR_NV12)
                put((idx, frame))
            idx += 1
            with lock:
                stats["seen"] = idx
            if idx % 10 == 0:
                with JOB.lock:
                    JOB.done = idx
                    # 浮点百分比：整数只有 0..99 一百个值，速度曲线最多一百个点、
                    # 一段 17 秒的活儿里 X 轴会一大段一大段地空着。浮点之后每个
                    # 采样点都有自己的横坐标，曲线才连续。
                    JOB.pct = min(99.0, round(idx * 100.0 / max(1, total_est), 2))
                    el = max(0.01, time.time() - t0)
                    JOB.fps = idx / el
                    JOB.faces_new = stats["made"]
                    JOB.faces = base_faces + stats["made"]
                    JOB.msg = "解码 %d 帧 · 已出 %d 张 · %.1f 帧/秒" % (
                        idx, stats["made"], JOB.fps)

    fail_log = {"n": 0}

    def _warn_detect(idx, err):
        # 原来这里 except 一下就 append([]) —— TRT 偶发驱动错误会被静默吃成
        # 「这一帧没人脸」，输出比上一次少一张，且没有任何痕迹。留一行日志。
        fail_log["n"] += 1
        if fail_log["n"] <= 20:
            print("[extract] 检测失败（第 %d 帧，重试后仍失败）：%s" % (idx, err), flush=True)

    def _detect_batch(batch):
        """batch: list[(idx, img)] → list[list[face]]

        检测抛异常时先原样重试一次：TensorRT 在两个 CUDA 上下文并发执行时
        （检测模型 + 2D106）偶发 "enqueueV3 ... Cuda Driver" 驱动错误，
        重试一次就过了；只重试、不换算法，结果与一次跑通完全相同。
        """
        if not batch:
            return []
        if need_canvas:
            items = [(b[1], *_canvas(b[1])) for b in batch]
            try:
                res = primary.detect_batch(items)
            except Exception:
                try:
                    res = primary.detect_batch(items)
                except Exception as e2:                 # noqa: BLE001
                    _warn_detect(batch[0][0], e2)
                    res = []
            if len(res) != len(batch):
                res = [[] for _ in batch]
        else:
            res = []
            for idx, img in batch:
                try:
                    res.append(primary.detect(img))
                except Exception:
                    try:
                        res.append(primary.detect(img))
                    except Exception as e2:             # noqa: BLE001
                        _warn_detect(idx, e2)
                        res.append([])
        return res

    # ---------------------------------------------------------------- 预热
    # 两个模型各跑一次空输入，**在起任何 worker 之前**。原因：
    # _trt_session.TrtSession 第一次执行时会把这次执行捕获成 CUDA Graph，捕获是
    # 设备级操作 —— 如果此刻另一个线程正在推理（检测线程 vs 2D106），两边会互相踩，
    # 表现为偶发 "[TRT][E] enqueueV3 ... Cuda Driver"，被检测侧的 except 吞掉后
    # 就是「这次比上次少一张脸」。实测 21 次抽取里出现过 3 次，两种模式都有。
    # 预热把捕获挪到安静期：跑通一次以后再也没有捕获，只有 replay。
    try:
        if lm106 is not None:
            lm106.landmarks68_batch([np.zeros((512, 512, 3), np.uint8)] * BATCH_106)
    except Exception as e:                     # noqa: BLE001 —— 预热失败不影响主流程
        print("[extract] 2D106 预热失败（不影响抽取）：%s" % e, flush=True)
    try:
        _detect_batch([(-1, np.zeros((64, 64, 3), np.uint8))])
    except Exception as e:                     # noqa: BLE001
        print("[extract] 检测器预热失败（不影响抽取）：%s" % e, flush=True)

    # ---------------------------------------------------------------- 回退实现
    def _legacy_run():
        """DFL_EXTRACT_PIPELINE=0：改动前的三级三线程，原样留着做对照。

        检测线程不再过滤 done_idx —— 那些帧在 _decode 里就没进队列（见上）。
        """

        def _reader():
            def put(item):
                t_b = _t()
                q_frames.put(item)
                cost["put1"] += _t() - t_b
                if PROF:
                    qmax["q1"] = max(qmax["q1"], q_frames.qsize())
            try:
                _decode(put)
            finally:
                q_frames.put(None)

        def _detector():
            pending = []

            def flush():
                if not pending:
                    return
                batch = list(pending)
                pending.clear()
                t_d = _t()
                faces_list = _detect_batch(batch)
                cost["det"] += _t() - t_d
                t_p = _t()
                for item, faces in zip(batch, faces_list):
                    q_detect.put((item[0], item[1], faces))
                cost["put2"] += _t() - t_p
                if PROF:
                    qmax["q2"] = max(qmax["q2"], q_detect.qsize())

            while True:
                t_a = _t()
                item = q_frames.get()
                cost["get1"] += _t() - t_a
                if item is None:
                    flush()
                    q_detect.put(None)
                    return
                pending.append(item)
                if len(pending) < BATCH:
                    continue
                # 批次没满时看看队列里还有没有，有就一起吃掉（减少一次同步）
                while len(pending) < BATCH:
                    try:
                        nxt = q_frames.get_nowait()
                    except Exception:
                        break
                    if nxt is None:
                        q_frames.put(None)
                        break
                    pending.append(nxt)
                flush()

        def _writer():
            while True:
                t_a = _t()
                item = q_detect.get()
                cost["get2"] += _t() - t_a
                if item is None:
                    _flush()              # 收尾：把最后不满 8 张的一批也写掉（内部已计数）
                    return
                idx, img, faces = item
                t_e = _t()
                _emit(idx, img, faces)    # 计数在 _flush 里做，这里不要再累加
                cost["emit"] += _t() - t_e

        # 这三条线程以前是"裸的"：任一条抛异常就静默死掉，主线程还在 join 上等着。
        # 统一包一层：异常记进 job_errors + 落失败态 + 置 stop + 塞哨兵（别让下游线程
        # 卡在阻塞 get 上永远 join 不回来），收尾时抛回调用线程。
        job_errors = []

        def _guard(fn):
            def _run():
                try:
                    fn()
                except BaseException as e:      # noqa: BLE001 —— 线程顶层
                    job_errors.append(e)
                    _job_fail("_align_stream/" + fn.__name__, e)
                    stop.set()
                    for _q in (q_frames, q_detect):
                        try:
                            _q.put(None, timeout=1.0)
                        except Exception:
                            pass
            return _run

        threads = [threading.Thread(target=_guard(_reader), name="ext-read", daemon=True),
                   threading.Thread(target=_guard(_detector), name="ext-detect", daemon=True),
                   threading.Thread(target=_guard(_writer), name="ext-write", daemon=True)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if job_errors:
            raise job_errors[0]

    # ---------------------------------------------------------------- 主路径
    mode, cancelled, alive = "legacy", False, []
    try:
        if _pipeline.pipeline_enabled():
            res = _pipeline.run(
                decode=_decode,
                detect_batch=_detect_batch,
                emit=lambda item, faces: _emit(item[0], item[1], faces),
                batch=BATCH, q_in=q_cap, q_out=q_cap,
                should_stop=lambda: not JOB.running,
                prof=cost, name="ext",
            )
            # 不再 res["made"] 回填：_flush 落盘时就逐个计过数了（这里再累加会翻倍）。
            qmax["q1"] = max(qmax["q1"], int(res.get("q1max") or 0))
            qmax["q2"] = max(qmax["q2"], int(res.get("q2max") or 0))
            mode, cancelled, alive = "pipeline", bool(res["cancelled"]), list(res["alive"])
            _flush()                     # 收尾：最后不满 8 张的一批（内部已计数）
        else:
            _legacy_run()
    finally:
        stop.set()
        try: gen.close()
        except Exception: pass

    # ★ 断流对账。imageio 的读循环在 ffmpeg 中途死掉时是「读到 b'' → return」，
    #   **不抛异常**；_decode 收到 StopIteration 就当正常收尾。于是源片被截断、
    #   管道半路断、ffmpeg 被杀，全都表现为「进度 100% / 完成 / 帧数少于原片」，
    #   而训练和合成照常开始 —— 用户拿到的是一份缺了一截的数据集。
    #   拿 total_est（目录=文件数，视频=dur×fps）对账：差超过 2% 且至少 3 帧就判
    #   没跑完。取消不算（那是用户自己停的）。
    _short = (0 if cancelled else max(0, int(total_est) - int(stats["seen"])))
    _truncated = bool(_short >= max(3, int(total_est * 0.02 + 0.999)))
    if _truncated:
        _log("[align] ★ 解码提前结束：只读到 %d / %d 帧（差 %d）—— 源片可能被截断、"
             "管道中途断了，或 ffmpeg 被杀。这一份 aligned/ 不完整。"
             % (stats["seen"], total_est, _short))
    wall = max(1e-6, time.time() - t0)

    # 最近一次抽取的实测数字：给 benchmark / 自检脚本读，不依赖解析 stdout
    _jobs._LAST_EXTRACT = {
        "ok": not _truncated,
        "error": ("解码提前结束：读到 %d / %d 帧" % (stats["seen"], total_est))
                 if _truncated else "",
        "short": int(_short), "total_est": int(total_est),
        "align": _amode, "lm106": lm106 is not None,
        "mode": mode, "frames": stats["seen"], "faces": stats["made"],
        "skipped": stats["skipped"], "rejected": stats["rejected"],
        "rejected_outlier": stats["rejected_outlier"],
        "rejected_gate": stats["rejected_gate"],
        "gate": stats["gate"], "gate_mode": _gm,
        "wall": wall, "fps": stats["seen"] / wall,
        "cost": {k: cost[k] for k in ("read", "put1", "get1", "det", "put2", "get2", "emit",
                                      "coarse", "lm106", "final", "jpeg")},
        "queue": q_cap, "batch": BATCH, "batch106": BATCH_106,
        "q1max": qmax["q1"], "q2max": qmax["q2"],
        "cancelled": cancelled, "alive": alive,
        "hwaccel": _hwaccel.current(),
    }
    # 一行结论：模式 / 帧 / 脸 / 跳过 / 墙钟。自检脚本和人都直接看这行。
    _log("[align] %s：DFL_ALIGN=%s 帧=%d 脸=%d 跳过=%d%s%s 墙钟=%.2fs (%.1f 帧/秒)"
         % (("★ 没跑完" if _truncated else "完成"),
            _amode, stats["seen"], stats["made"], stats["skipped"],
            ("（其中外点判异常 %d）" % stats["rejected_outlier"])
            if stats["rejected_outlier"] else "",
            (("（其中 5 点退化 %d）" % stats["rejected_gate"]) if _gm == "strict"
             else ("（5 点退化 %d 张已记入 _bad_frames.txt，图仍写盘）" % stats["gate"]))
            if stats["gate"] else "",
            wall, stats["seen"] / wall))
    if _amode == "68" and stats["made"] == 0 and stats["skipped"]:
        _log("[align] 一张都没写出来：%d 张脸全部拿不到 68 点。"
             "检查 weights/insightface_2d106.onnx 与 TRT/ORT 是否可用"
             "（DFL_ALIGN=68 不会退回 5 点）。" % stats["skipped"])
    if PROF:
        try:
            print("[extract-profile] mode=%s q=%d batch=%d  wall=%.2fs  frames=%d  faces=%d  %.1f 帧/秒"
                  % (mode, q_cap, BATCH, wall, stats["seen"], stats["made"], stats["seen"] / wall))
            for k in ("read", "put1", "get1", "det", "put2", "get2", "emit",
                      "coarse", "lm106", "final", "jpeg"):
                print("  %-6s %7.3fs  %5.1f%% of wall" % (k, cost[k], cost[k] * 100.0 / wall))
            print("  queue max depth: q1=%d q2=%d" % (qmax["q1"], qmax["q2"]))
            if cancelled:
                print("  cancelled at frame %d（worker 残留：%s）" % (stats["seen"], alive or "无"))
        except Exception:
            pass

    with JOB.lock:
        JOB.running = False
        JOB.done = stats["seen"]
        JOB.total = max(stats["seen"], 1)
        JOB.pct = 100
        JOB.faces = base_faces + stats["made"]
        JOB.ok = True           # 只有真的跑到底才置 True
        JOB.error = ""
        JOB.msg = "完成：解码 %d 帧 · %d 张人脸%s → %s" % (
            stats["seen"], stats["made"],
            ("（跳过 %d 张：拿不到 68 点）" % stats["skipped"]) if stats["skipped"] else "",
            str(face_dir))
    return stats["made"]

