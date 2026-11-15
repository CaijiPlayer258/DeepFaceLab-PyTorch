# -*- coding: utf-8 -*-
"""抽帧 / 抽脸对齐：解码流水线 + 进度写 JOB（从 app/server/app.py 切出）。

  _frames_run / _frames_worker   抽帧（可选对齐），routes_trt.extract_start 的线程体
  _align_stream                  ffmpeg 边解边对齐，原帧只驻内存一次（性能关键路径）
  _merge_prepare_worker          视频 → 项目（000000.jpg + aligned/000000_0.jpg）
  _merge_frame_image             合成器按当前打开的 视频/项目 取一帧

搬家只动了两处：包内相对 import 的深度；_align_stream 末尾那句 global _LAST_EXTRACT
改成写 _common/jobs.py 里同一份状态（原来是 app 模块的全局，现在归 jobs 模块所有）。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import HTTPException

from .._common import jobs as _jobs
from .._common.bootstrap import _NO_WINDOW
from .._common.jobs import JOB, _job_fail, _merge_cache, _merge_prep, _merge_prep_lock
from .detectors import _get_fast_detector, _get_heavy_detector, _get_landmarks106
from .media import _dfl_project_scan, _extract_frames, _imread_any, _open_video


def _video_frame_jpeg(path, idx, maxw=900):
    """用 ffmpeg 直接抽帧并缩放成 JPEG。
    比 imageio 读全尺寸 4K 帧再 PIL 缩放快一个数量级（4K 竖屏实测 2.5s → 0.3s）。"""
    import subprocess
    import imageio_ffmpeg
    fps = float(_merge_cache.get("fps") or 30.0)
    t = max(0.0, float(idx) / fps) if fps else 0.0
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        exe, "-v", "error", "-ss", "%.3f" % t, "-i", str(path),
        "-frames:v", "1",
        "-vf", "scale=%d:-2" % int(maxw),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_NO_WINDOW)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError((out.stderr or b"").decode("utf-8", "ignore")[:200])
    return out.stdout

def _merge_prepare_worker(video: Path, project: Path):
    import threading as _th

    def _set(**kw):
        with _merge_prep_lock:
            _merge_prep.update(kw)

    try:
        project.mkdir(parents=True, exist_ok=True)
        aligned = project / "aligned"
        aligned.mkdir(parents=True, exist_ok=True)

        _set(stage="抽帧", pct=0, msg="分析视频…")
        try:
            import imageio_ffmpeg as _iff
            n, secs = _iff.count_frames_and_secs(str(video))
            total = int(n) if n else 0
            src_fps = (n / secs) if secs else 30.0
        except Exception:
            total, src_fps = 0, 30.0

        # 快路径：同一个视频已经准备过（帧数对得上、对齐脸也在），直接复用
        have = len(list(project.glob("*.jpg")))
        have_aligned = len(list(aligned.glob("*.jpg")))
        if total and have == total and have_aligned > 0:
            _set(frames=have, aligned=have_aligned, pct=100,
                 msg="复用已准备好的项目：%d 帧 · %d 张对齐脸" % (have, have_aligned))
            try:
                names, srcdir, kind = _dfl_project_scan(project)
                _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                     "names": names, "path": "", "frames": len(names),
                                     "fps": src_fps, "w": 0, "h": 0})
            except Exception:
                pass
            _set(running=False, ok=True, stage="完成")
            return

        def _on_frame(made):
            pct = int(35 * made / total) if total else 0
            _set(pct=min(35, pct), msg="抽帧 %d/%s" % (made, total or "?"))

        # 只有确认要重跑才清对齐脸（上面快路径会复用它们）
        for old in aligned.glob("*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
        got = _extract_frames(video, project, total, _on_frame)
        _set(frames=got, pct=35, msg="抽帧完成：%d 帧" % got)

        # 对齐：复用提取器的解码+检测+对齐流水线（它会自己开 JOB）
        _set(stage="对齐", msg="加载检测器…")
        holder = {}

        def _run_align():
            # _align_stream 自己不置 running —— 提取器那边是调用方置的。
            # 不置的话它内部那句 "if not JOB.running: break" 会立刻退出，产出 0 张脸。
            try:
                JOB.running = True
                holder["n"] = _align_stream(str(video), aligned, fps=src_fps,
                                            rotate=0, quality="high", skip=1,
                                            src_fps=src_fps)
            except Exception as e:                     # noqa: BLE001
                holder["err"] = e
            finally:
                JOB.running = False

        th = _th.Thread(target=_run_align, daemon=True)
        th.start()
        while th.is_alive():
            try:
                done = int(getattr(JOB, "done", 0) or 0)
                tot = int(getattr(JOB, "total", 0) or 0) or got or 1
                pct = 35 + int(63 * min(1.0, done / float(tot)))
                _set(pct=min(98, pct), msg="对齐 %d/%d" % (done, tot))
            except Exception:
                pass
            th.join(0.4)
        if "err" in holder:
            raise holder["err"]
        made = int(holder.get("n") or 0)
        _set(aligned=made, pct=100, msg="完成：%d 帧 · %d 张对齐脸" % (got, made))
        if not made:
            _set(ok=False, error="对齐没有产出（视频里没检出人脸？）",
                 msg="对齐失败：0 张脸")

        # 直接把准备好的项目打开，前端不用再点一次
        try:
            names, srcdir, kind = _dfl_project_scan(project)
            _merge_cache.update({"mode": "project", "dir": str(project), "srcdir": srcdir,
                                 "names": names, "path": "", "frames": len(names),
                                 "fps": src_fps, "w": 0, "h": 0})
        except Exception:
            pass
        _set(running=False, ok=True, stage="完成")
    except Exception as e:                             # noqa: BLE001
        _set(running=False, ok=False, error=str(e)[:300], msg="失败", stage="出错")

def _merge_frame_image(idx: int):
    """按当前打开的东西取一帧：视频模式解码，项目模式直接读图。

    返回 (图像, 总帧数)。合成器两种模式都要能检测/分析，所以抽出来共用。
    """
    import cv2
    total = int(_merge_cache.get("frames") or 0)
    if _merge_cache.get("mode") == "project":
        names = _merge_cache.get("names") or []
        srcdir = _merge_cache.get("srcdir") or ""
        if not names:
            raise HTTPException(409, "还没有打开项目")
        idx = max(0, min(int(idx), len(names) - 1))
        img = _imread_any(Path(srcdir) / names[idx])
        if img is None:
            raise HTTPException(500, "读不了这张图: " + names[idx])
        return img, len(names)

    path = _merge_cache.get("path")
    if not path:
        raise HTTPException(409, "还没有打开视频")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(500, "打不开视频")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "取不到第 %d 帧" % idx)
    return frame, total

def _frames_worker(src: str, dst: str, fps: float = 2.0, rotate: int = 0,
                   align: bool = True, quality: str = "high", size: int = 512,
                   skip: int = 0):
    """线程入口（routes_trt.extract_start 的 target）：异常一律落成失败态。

    这里必须是**唯一**的收口：worker 线程里抛出去的异常没人接，Python 只把它
    打到 stderr，JOB 还停在半吊子的 pct 上 —— 界面就成了「完成 0 帧」。
    """
    try:
        return _frames_run(src, dst, fps, rotate, align, quality, size, skip)
    except BaseException as e:                 # noqa: BLE001 —— 线程顶层，SystemExit 也收
        _job_fail("_frames_worker", e)
        raise

def _frames_run(src: str, dst: str, fps: float = 2.0, rotate: int = 0,
                align: bool = True, quality: str = "high", size: int = 512,
                skip: int = 0):
    """用 ffmpeg 抽帧到 dst，然后（可选）对齐成人脸。

    这一版修了三处：
      1) **先清空输出目录**。ffmpeg 的 frame_%06d 从 1 开始编号，不清的话上一个
         视频的残留帧会被当成本次素材 —— 换短一点的视频就会混进旧数据。
      2) fps 由**源视频真实帧率**换算（skip 帧抽 1 帧），而不是假设源是 24fps。
         以前 UI 传的是 24/skip，30fps 的源会被多抽、60fps 的源会被砍掉一半。
      3) 进度改用 ffmpeg -progress 的 frame= 计数，不再每 0.25 s 扫一遍目录
         （几万个文件时那次 glob 本身就很贵）。
    """
    import subprocess
    import imageio_ffmpeg

    out_dir = Path(dst)                                        # 人脸输出目录（用户填的那个）
    out_dir.mkdir(parents=True, exist_ok=True)

    # 对齐模式：原帧不落盘，ffmpeg 出帧 → 内存里检测 → 直接写对齐图
    if align:
        try:
            src_fps = 30.0
            n, secs = imageio_ffmpeg.count_frames_and_secs(str(src))
            if secs > 0:
                src_fps = n / secs
            if skip and skip > 0:
                fps = max(0.05, src_fps / float(skip))
            fps = min(float(fps), src_fps)
        except Exception:
            pass
        # 异常不在这里收：_frames_worker 是唯一收口（ok=False + error + traceback）。
        return _align_stream(str(src), out_dir, fps, rotate, quality, skip, src_fps)

    # 不对齐：保留原来的"抽成 jpg"路径（给只想拆帧的场景用）
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = out_dir.parent / (out_dir.name + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 增量抽帧：接着已有编号往后写，绝不覆盖、也不删除 ----
    # 用户的用法是"往同一个目录里一点点加素材"，清空会误删别人的数据。
    # 但 ffmpeg 默认从 1 开始编号，直接再抽一次会**覆盖**前 N 张 —— 所以要接着最大序号写。
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    start_no = 1
    if existing:
        try:
            start_no = max(int(p.stem.split("_")[-1]) for p in existing
                           if p.stem.split("_")[-1].isdigit()) + 1
        except Exception:
            start_no = len(existing) + 1

    # ---- 2) 源帧率 ----
    src_fps = 30.0
    total = 1
    try:
        n, secs = imageio_ffmpeg.count_frames_and_secs(str(src))
        if secs > 0:
            src_fps = n / secs
        if skip and skip > 0:
            fps = max(0.05, src_fps / float(skip))
        fps = min(float(fps), src_fps)
        total = max(1, int(secs * fps)) if secs else 1
    except Exception:
        pass

    with JOB.lock:
        JOB.running = True
        JOB.fps = 0.0
    # 阶段 0：ffmpeg 抽帧（进度用 -progress 的 frame= 计数驱动）
    JOB.begin(0, total,
              "正在抽帧（源 %.2f fps → 每 %.0f 帧取 1，输出 %.3f fps · 已有 %d 帧，从 %06d 接着写）"
              % (src_fps, skip, fps, len(existing), start_no))

    vf = []
    if rotate in (90, 180, 270):
        vf.append({90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}[rotate])
    vf.append("fps=%g" % fps)
    cmd = [exe, "-v", "error", "-i", str(src),
           "-vf", ",".join(vf),
           "-q:v", "2", "-progress", "pipe:1", "-nostats",
           "-start_number", str(start_no),
           str(frames_dir / "frame_%06d.jpg")]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=_NO_WINDOW, text=True, bufsize=1)
    t0 = JOB.t0 or time.time()      # 与 SSE 的 elapsed 同一个时钟
    made = 0
    try:
        for line in proc.stdout:
            if not JOB.running:
                proc.terminate()
                break
            line = line.strip()
            if line.startswith("frame="):
                try:
                    made = int(line.split("=", 1)[1])
                except Exception:
                    continue
                with JOB.lock:
                    JOB.done = made
                    JOB.pct = min(95, int(made * 100 / max(1, total))) if total else 0
                    el = max(0.01, time.time() - t0)
                    JOB.fps = made / el
                    JOB.msg = "本次新增 %d / 约 %d 帧 · %.1f 帧/秒" % (made, total, JOB.fps)
    finally:
        proc.wait()

    made = len(list(frames_dir.glob("frame_*.jpg")))

    with JOB.lock:
        JOB.running = False
        JOB.done = made
        JOB.total = max(total, made)
        JOB.pct = 100
        JOB.faces = 0
        JOB.ok = True           # 走到这里才是真的"跑完了"
        JOB.error = ""
        JOB.msg = "完成：%d 帧 → %s" % (made, str(frames_dir))
    return made

def _align_stream(src: str, face_dir: Path, fps: float = 2.0, rotate: int = 0,
                  quality: str = "high", skip: int = 0, src_fps: float = 0.0) -> int:
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
    """
    import queue
    import threading
    import numpy as np
    import cv2
    from .._align import (COARSE_SIZE, align_face, align_face_68, align_mode,
                         coarse_align_from_5, landmarks68_in_source,
                         pose_from_kps, write_dfljpg)

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

    from .. import _hwaccel
    in_params = _hwaccel.input_params(str(src))     # 触发（或命中）一次探测，下面复用
    norm = _hwaccel.normalize_filter()              # 硬解：先回 yuv420p 再转 BGR
    if norm:
        vf.append(norm)

    # vf 为空时不能传 -vf ""（ffmpeg 会报错），所以整段参数一起省掉
    # 不能传空列表：imageio_ffmpeg 在 output_params=[] 时会拼出坏命令并杀掉 ffmpeg（实测 0 帧）。
    # 没有滤镜就显式给个 null，等价于不处理。
    out_params = ["-vf", ",".join(vf)]
    # 解码走 _hwaccel 的候选链（cuda → d3d11va → qsv → none）：硬解起不来/吐不出帧
    # 就地换软解，绝不把异常抛给抽取流程。first=True 顺带验一帧，防「有头无帧」。
    gen, meta, first_frame = _open_video(str(src), out_params, first=True,
                                        in_params=in_params)
    W, H = meta.get("size") or (0, 0)
    if not W or not H:
        try: gen.close()
        except Exception: pass
        _job_fail("_align_stream/解码", RuntimeError("解不出分辨率，视频打不开或没有视频流：%s" % src))
        return 0
    dur = float(meta.get("duration") or 0)
    total_est = max(1, int(dur * max(0.01, float(fps)))) if dur else 1

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
    # rejected = 68 点外点检查判「检测异常」的张数（在 skipped 里再细分一项）
    stats = {"seen": 0, "made": 0, "fallback": 0, "skipped": 0, "rejected": 0}
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
        for it, lm68 in zip(items, lms):
            if _amode == "68":
                if lm68 is None:
                    _skip(it["idx"], "拿不到 68 点（2D106 没有输出，不退回 5 点）")
                    continue
                try:
                    _tf = _t()
                    pts = landmarks68_in_source(lm68, it["mat"])
                    crop, meta_f = align_face_68(it["img"], pts, box=it["box"], size=it["size"])
                    cost["final"] += _t() - _tf
                except Exception as e:      # noqa: BLE001
                    _skip(it["idx"], "68 点反变换/仿射失败：%s" % e)
                    continue
                it["img"] = None            # 别把整帧攥在 _pending 里
                if crop is None:
                    # meta_f 此时是「拒收记录」：reject_reason / align_resid_px
                    _why = (meta_f or {}).get("reject_reason") if isinstance(meta_f, dict) else None
                    if _why:
                        with lock:
                            stats["rejected"] += 1
                    _skip(it["idx"], _why or "68 点仿射失败（点数/数值不合法）")
                    continue
            else:
                crop, meta_f = it["crop"], it["meta"]
            meta_f["source_filename"] = "%06d" % it["idx"]
            if it.get("pose"):
                meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = it["pose"]
            try:
                _tj = _t()
                write_dfljpg(it["path"], crop, meta_f)
                cost["jpeg"] += _t() - _tj
                wrote += 1
                with lock:                  # 落盘即计数：跑的过程中就能看到数字在长
                    stats["made"] += 1
            except Exception as e:          # noqa: BLE001 —— 落盘失败也要留痕
                print("[extract] 写 %s 失败：%s" % (Path(it["path"]).name, e), flush=True)
                continue
        if wrote:
            # 锁序：先放掉 lock 再拿 JOB.lock（_decode 那边是 JOB.lock → 读 stats，反向嵌套会死锁）
            with JOB.lock:
                JOB.faces_new = stats["made"]
                JOB.faces = base_faces + stats["made"]
        return wrote

    def _emit(idx, img, faces):
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
            if len(raw) != W * H * 3:
                continue
            if idx not in done_idx:
                frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
                put((idx, frame))
            idx += 1
            with lock:
                stats["seen"] = idx
            if idx % 10 == 0:
                with JOB.lock:
                    JOB.done = idx
                    JOB.pct = min(99, int(idx * 100 / max(1, total_est)))
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

    wall = max(1e-6, time.time() - t0)
    # 最近一次抽取的实测数字：给 benchmark / 自检脚本读，不依赖解析 stdout
    _jobs._LAST_EXTRACT = {
        "ok": True, "error": "",
        "align": _amode, "lm106": lm106 is not None,
        "mode": mode, "frames": stats["seen"], "faces": stats["made"],
        "skipped": stats["skipped"], "rejected": stats["rejected"],
        "wall": wall, "fps": stats["seen"] / wall,
        "cost": {k: cost[k] for k in ("read", "put1", "get1", "det", "put2", "get2", "emit",
                                      "coarse", "lm106", "final", "jpeg")},
        "queue": q_cap, "batch": BATCH, "batch106": BATCH_106,
        "q1max": qmax["q1"], "q2max": qmax["q2"],
        "cancelled": cancelled, "alive": alive,
        "hwaccel": _hwaccel.current(),
    }
    # 一行结论：模式 / 帧 / 脸 / 跳过 / 墙钟。自检脚本和人都直接看这行。
    _log("[align] 完成：DFL_ALIGN=%s 帧=%d 脸=%d 跳过=%d%s 墙钟=%.2fs (%.1f 帧/秒)"
         % (_amode, stats["seen"], stats["made"], stats["skipped"],
            ("（其中外点判异常 %d）" % stats["rejected"]) if stats["rejected"] else "",
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
