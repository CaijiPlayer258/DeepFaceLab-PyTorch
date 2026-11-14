# -*- coding: utf-8 -*-
"""帧来源：视频时长探测（_video_duration）+ 跳帧抓帧（_grab_*）+ 视频/图片扩展名表。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from .._common.http import (
    _BadInput,
)


# ---- 合成参数白名单 ----
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".m2v", ".vob", ".ogv", ".3gp"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---- 预览：取帧 / 关键点 / 模型 ----
_DUR_CACHE = {}
_DUR_LOCK = threading.Lock()


def _video_duration(p) -> float:
    """视频时长（秒）。按 大小+mtime 缓存：拖动时每帧都要问，不能每帧都去探。"""
    pp = Path(str(p))
    try:
        st = pp.stat()
    except OSError:
        return 0.0
    key = os.path.normcase(str(pp))
    sig = (int(st.st_size), int(st.st_mtime))
    with _DUR_LOCK:
        hit = _DUR_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    dur = 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(pp))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0 and n > 0:
                dur = n / fps
        finally:
            cap.release()
    except Exception:
        dur = 0.0
    if not dur:
        try:
            import imageio_ffmpeg
            _n, secs = imageio_ffmpeg.count_frames_and_secs(str(pp))
            dur = float(secs or 0.0)
        except Exception:
            dur = 0.0
    with _DUR_LOCK:
        _DUR_CACHE[key] = (sig, dur)
    return dur


# ---- 取帧：同帧复用 / 顺序解码缓存 / ffmpeg 快速 seek / cv2 回退 ----
#
# 先说清楚为什么不是「每次 ffmpeg -ss 一下」，下面是这台机器上量的（workspace 里那个
# 1080x1920 / 30fps / GOP=5s / 57MB 的 mp4）：
#   · ffmpeg 起一个进程本身就要 47ms（-version）、110ms（带 -i）——光启动就吃掉 50ms 的
#     预算，再叠 seek+解码，整次取帧实测 208~264ms（-ss 放 -i 前面已经是关键帧级定位了）；
#   · cv2 不换进程，复用常开句柄重新定位是 78~215ms，而且定位完句柄正好停在 t 上，
#     下一次「往后拖」的请求能接着顺序读（1.2ms/帧，完全不重新定位）。
# 所以冷定位默认走 cv2（_GRAB_FFMPEG_FIRST=False），ffmpeg 快速 seek 留着当回退；
# 想反过来「先 ffmpeg」把开关翻成 True 就行，两条路都实现在这里。
# 真正能进 50ms 的是前两条：同一帧直接复用（≈0ms）、前向 1.5s 内顺序解码（10~60ms）。
_GRAB_LOCK = threading.RLock()          # 常开句柄 + 缓存帧是共享可变状态（预览是串行的）
_GRAB_FWD_WINDOW = 1.5                  # 顺序解码窗口（秒）：超过就重新定位
_GRAB_FFMPEG_FIRST = False              # 冷定位先试 ffmpeg 快速 seek？（实测更慢，见上）
_GRAB_FFMPEG_TIMEOUT = 8.0              # 单次 ffmpeg 取帧上限（秒）
_GRAB_STATE = {"key": None, "cap": None, "fps": 0.0, "idx": -1,
               "frame": None, "pos": False}


def _grab_key(video):
    """缓存身份：路径 + 大小 + mtime。片子换了/改了，句柄和缓存帧一起作废。"""
    p = os.path.normcase(str(video))
    try:
        stt = os.stat(str(video))
        return (p, int(stt.st_size), int(stt.st_mtime))
    except OSError:
        return (p, 0, 0)


def _grab_close_locked():
    """丢掉常开句柄和缓存帧（换片时用；进程退出时句柄跟着回收）。"""
    cap = _GRAB_STATE.get("cap")
    _GRAB_STATE.update({"key": None, "cap": None, "fps": 0.0, "idx": -1,
                        "frame": None, "pos": False})
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _grab_index(t, fps):
    """t → 帧号。口径跟 cv2 的精确 seek 对齐：实测 set(POS_MSEC) 落的是 round(t*fps) 那一帧
    （7.21→216、7.23→217、7.25→218、20.598→618），四舍五入用 floor(x+0.5)，因为 t*fps 在
    浮点里常常贴着整数（7.2*30 = 215.99999999999997），直接 floor/ceil 都会差一帧。
    """
    import math
    x = max(0.0, float(t)) * float(fps)
    return int(math.floor(x + 0.5))


def _grab_ensure_locked(video):
    """按 key 准备常开句柄（换了片就重建）。cv2 打不开也不炸，回 (None, 30.0)。"""
    import cv2
    key = _grab_key(video)
    st = _GRAB_STATE
    cap = st.get("cap")
    if st.get("key") != key or cap is None or not cap.isOpened():
        _grab_close_locked()
        try:
            cap = cv2.VideoCapture(str(video))
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return None, 30.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0
        if not (1.0 <= fps <= 480.0):
            fps = 30.0
        st.update({"key": key, "cap": cap, "fps": fps, "idx": -1,
                   "frame": None, "pos": False})
    return st["cap"], float(st["fps"] or 30.0)


def _grab_via_ffmpeg(video, t):
    """ffmpeg 快速 seek 取一帧：-ss 放在 -i 前面是关键（关键帧级定位 + accurate_seek 到精确帧）。

    成功回 (frame, "")，失败回 (None, 原因) —— 原因要带进 _BadInput 里给前端看。
    """
    import cv2
    import numpy as np
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        return None, "没有 ffmpeg：%s" % (str(e)[:100])
    cmd = [exe, "-v", "error", "-nostdin", "-ss", "%.3f" % float(t), "-i", str(video),
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_GRAB_FFMPEG_TIMEOUT,
                             creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
    except Exception as e:
        return None, "ffmpeg 跑不起来：%s" % (str(e)[:120])
    if out.returncode != 0 or not out.stdout:
        why = (out.stderr or b"").decode("utf-8", "ignore").strip()
        return None, "ffmpeg：%s" % (why[:160] or "没有输出")
    try:
        frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return None, "ffmpeg 给的图解不开：%s" % (str(e)[:100])
    if frame is None or not int(getattr(frame, "size", 0)):
        return None, "ffmpeg 给的图解不开"
    return frame, ""


def _grab_frame_at(video, t: float, stop=None):
    """取 t 秒那一帧（BGR）。

    三条路，从便宜到贵：
      1) 同一帧：t 跟上一帧落在同一帧号上（前端 120ms 防抖会把同一个 t 请求两遍）→ 直接给缓存帧；
      2) 顺序解码：t 在上次之后 _GRAB_FWD_WINDOW 秒内 → 常开句柄 grab() 往前读，不再定位；
      3) 重新定位：ffmpeg 快速 seek（_GRAB_FFMPEG_FIRST 为真时先试）→ cv2 的
         set(CAP_PROP_POS_MSEC)（原实现的路子，只是复用常开句柄，省掉每次开文件的 6~9ms）。
    错误语义不变：取不出来 500、取消 499；成功时顺手把「上一帧」记进 _GRAB_STATE。
    """
    import cv2

    t = max(0.0, float(t))
    err = ""
    with _GRAB_LOCK:
        cap, fps = _grab_ensure_locked(video)
        st = _GRAB_STATE
        target = _grab_index(t, fps)

        # 1) 同一帧复用（同一个 t，或者两个 t 落在同一帧号上）
        if st["frame"] is not None and int(st["idx"]) == target:
            return st["frame"]

        # 2) 顺序解码：句柄就位 + 目标在前向窗口内
        need = target - int(st["idx"]) if int(st["idx"]) >= 0 else -1
        window = max(1, int(round(_GRAB_FWD_WINDOW * fps)))
        if st["pos"] and cap is not None and 0 < need <= window:
            if stop is not None and stop.is_set():
                raise _BadInput("预览已取消", 499)
            ok = True
            for _ in range(need):
                if not cap.grab():
                    ok = False
                    break
                st["idx"] = int(st["idx"]) + 1
            if ok:
                try:
                    ok, frame = cap.retrieve()
                except Exception:
                    ok, frame = False, None
                if ok and frame is not None and int(getattr(frame, "size", 0)):
                    st["frame"] = frame
                    st["pos"] = True
                    return frame
            err = "顺序解码没读到这一帧"
            # 句柄位置和缓存帧的帧号一起作废（否则下次同一个 t 会拿到错一帧的缓存）
            st["idx"] = -1
            st["pos"] = False
        if stop is not None and stop.is_set():
            raise _BadInput("预览已取消", 499)

        # 3) 重新定位：ffmpeg 快速 seek 与 cv2 seek 都试，谁先看开关
        if _GRAB_FFMPEG_FIRST:
            frame, why = _grab_via_ffmpeg(video, t)
            if frame is not None:
                # 句柄没跟着走：位置标脏，下一次前向请求老老实实重新定位（不会串帧）
                st.update({"frame": frame, "idx": target, "pos": False})
                return frame
            err = (err + "；" if err else "") + (why or "ffmpeg 取帧失败")

        if cap is None:
            err = (err + "；" if err else "") + "cv2 打不开这个文件"
        else:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
            except Exception as e:
                ok, frame = False, None
                err = (err + "；" if err else "") + "cv2 取帧失败：%s" % (str(e)[:120])
            if ok and frame is not None and int(getattr(frame, "size", 0)):
                try:                            # 帧号以句柄实际位置为准（read 之后指向下一帧）
                    posf = int(round(float(cap.get(cv2.CAP_PROP_POS_FRAMES))))
                except Exception:
                    posf = -1
                st.update({"frame": frame,
                           "idx": (posf - 1) if posf > 0 else target,
                           "pos": True})
                return frame
            err = (err + "；" if err else "") + "cv2 没读到这一帧"

        if not _GRAB_FFMPEG_FIRST:
            frame, why = _grab_via_ffmpeg(video, t)
            if frame is not None:
                st.update({"frame": frame, "idx": target, "pos": False})
                return frame
            err = (err + "；" if err else "") + (why or "ffmpeg 取帧失败")

    raise _BadInput("这一帧取不出来（t=%.3fs）：%s" % (t, err[:300]), 500)


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 media_frames 与 _app_mod
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod
