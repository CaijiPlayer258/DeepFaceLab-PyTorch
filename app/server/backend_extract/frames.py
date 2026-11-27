# -*- coding: utf-8 -*-
"""抽帧（可选对齐）：/api/extract/start 的线程体。

本文件只做「视频 → 帧」和分发；三个真正的实现各自独立，别再把它们搅在一起：
  · .align_stream._align_stream   边解码边对齐，直接产出人脸 DFLJPG（提取页主路径）
  · .resume                       续跑指纹（换了源素材要能发现）
  · ../backend_merge/frames_prep  视频↔项目互转、合成器取帧（**合成用的**，不是提取）

老调用点（media.py / jobs.py / backend_merge/compose.py）仍然 import frames._align_stream，
所以这里保留一个转发，行为与拆分前完全一致。
"""
from __future__ import annotations

import time
from pathlib import Path

from .._common.bootstrap import _NO_WINDOW
from .._common.jobs import JOB, _job_fail

# 老名字继续可用（拆分只动位置，不动调用点）
from .align_stream import _align_stream                       # noqa: F401,E402
from .resume import SRC_FP_NAME, _fp_diff, _note_resume_src, _read_src_fp, _src_fingerprint  # noqa: F401,E402

def _frames_worker(src: str, dst: str, fps: float = 2.0, rotate: int = 0,
                   align: bool = True, quality: str = "high", size: int = 512,
                   skip: int = 0, strict_align: "bool | None" = None):
    """线程入口（routes_trt.extract_start 的 target）：异常一律落成失败态。

    这里必须是**唯一**的收口：worker 线程里抛出去的异常没人接，Python 只把它
    打到 stderr，JOB 还停在半吊子的 pct 上 —— 界面就成了「完成 0 帧」。
    """
    try:
        return _frames_run(src, dst, fps, rotate, align, quality, size, skip,
                           strict_align)
    except BaseException as e:                 # noqa: BLE001 —— 线程顶层，SystemExit 也收
        _job_fail("_frames_worker", e)
        raise

def _frames_run(src: str, dst: str, fps: float = 2.0, rotate: int = 0,
                align: bool = True, quality: str = "high", size: int = 512,
                skip: int = 0, strict_align: "bool | None" = None):
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
        n = _align_stream(str(src), out_dir, fps, rotate, quality, skip, src_fps,
                          strict_align)
        # 收尾必须在这里做：下面那段（149 行）只属于「抽帧写盘」分支，
        # align 分支直接 return 会把它整段跳过 —— JOB.running 永远 True，
        # 界面就停在 98%「正在提取」不动，看着像假死（实测 576 帧跑完不结束）。
        with JOB.lock:
            JOB.running = False
            JOB.done = JOB.done or JOB.total
            JOB.total = max(JOB.total, JOB.done)
            # ★ 不要无条件把失败/取消覆盖成「完成」。_align_stream 里有 4 个
            #   _job_fail 分支（检测器缺失 / 空目录 / 首图读不出 / 视频打不开）都只是
            #   `return 0`，以及取消路径会置 ok=False —— 而这里以前一律
            #   `ok=True / pct=100 / error="" / 完成：对齐 0 张人脸`，于是用户看到
            #   「红条一闪 → 完成 0 张脸」，只有 _LAST_EXTRACT.ok 能看出真相（界面不读）。
            if not JOB.ok:
                # 已经明确失败/取消：把状态定在失败上，别往下改成成功。
                JOB.pct = max(0, min(99, int(JOB.pct or 0)))
                if not JOB.error:
                    JOB.error = "提取没有正常完成"
                if not JOB.msg:
                    JOB.msg = "失败：%s" % JOB.error
                return n
            JOB.pct = 100
            JOB.ok = True
            JOB.error = ""
            JOB.msg = "完成：对齐 %d 张人脸 → %s" % (n or 0, str(out_dir))
        return n

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

