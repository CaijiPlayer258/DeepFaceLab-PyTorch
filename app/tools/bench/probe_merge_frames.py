# -*- coding: utf-8 -*-
r"""一个视频文件的「三个帧数」到底差多少 —— 合成工作台时间轴那个 35597 的取证工具。

用法（在仓库根执行）：
    .\runtime\unified\python.exe app\tools\bench\probe_merge_frames.py "D:\...\x.mp4"
    .\runtime\unified\python.exe app\tools\bench\probe_merge_frames.py "x.mp4" --full

它复刻后端两条互不相干的口径，并把差异摊开：

  ① /api/merge/open 的 frames        = backend_extract/media.py:22
     imageio_ffmpeg.count_frames_and_secs()  → ffmpeg -vf null 全解码，报的是
     **真正解出来的帧数**（不是 CAP_PROP_FRAME_COUNT），代价是整片解一遍。

  ② /api/merge/preview 的时长闸门    = _media/frames.py:32 _video_duration()
     cv2 CAP_PROP_FRAME_COUNT / CAP_PROP_FPS。routes.py:1291-1295 用它当上限：
         t > dur + 0.5      → 400「t=… 超出了视频长度」
         t = min(t, dur-0.05)
     这就是视频模式「拖过某个帧号以后画面再也不动」的那道墙。

  ③ _grab_frame_at 的帧号映射        = _media/frames.py:111 _grab_index()
     target = round(t * cv2_fps)，而 _source_frame(compose.py:1001) 传的是
     t = idx / open_fps —— 两个 fps 只要不一致，不同时间轴帧号就会落到同一个
     target 上（同一张缓存图）。

--full 会真的逐帧解一遍（cv2.read 到 EOF），慢，但这是唯一能判「谁在说谎」的数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def probe(path: str) -> dict:
    import cv2
    import imageio_ffmpeg

    out: dict = {"path": path, "bytes": os.stat(path).st_size}

    cap = cv2.VideoCapture(path)
    out["cv2_opened"] = bool(cap.isOpened())
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    out["cv2_fps"] = round(fps, 6)
    out["cv2_frame_count"] = int(n)
    # = _video_duration() 的返回值（fps>0 且 n>0 才算得出来）
    out["_video_duration_s"] = round(n / fps, 3) if (fps > 0 and n > 0) else 0.0
    cap.release()

    try:
        import imageio.v3 as iio
        meta = iio.immeta(path, plugin="FFMPEG")
        keep = ("fps", "duration", "codec", "audio_codec", "size", "rotate")
        out["iio_meta"] = {k: meta.get(k) for k in keep if k in meta}
        out["iio_fps"] = float(meta.get("fps") or 0.0)
        out["iio_duration_s"] = meta.get("duration")
    except Exception as e:                                   # noqa: BLE001
        out["iio_meta_err"] = repr(e)[:300]

    try:
        t0 = time.time()
        nf, secs = imageio_ffmpeg.count_frames_and_secs(path)
        out["open_frames"] = int(nf)          # ← /api/merge/open 回的 frames
        out["open_secs"] = float(secs)
        out["open_probe_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:                                   # noqa: BLE001
        out["open_probe_err"] = repr(e)[:300]
    return out


def full_decode(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    i, last, back = 0, -1.0, 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if ts <= last:
            back += 1
        last = ts
        i += 1
    cap.release()
    return {"decoded": i, "last_pos_ms": round(last, 3),
            "non_monotonic_ts": back, "decode_s": round(time.time() - t0, 2)}


def verdict(d: dict) -> list:
    out = []
    of = d.get("open_frames")
    cv = d.get("cv2_frame_count")
    dur = d.get("_video_duration_s")
    fps = d.get("cv2_fps") or 0.0
    iiod = d.get("iio_duration_s")
    if of and cv:
        out.append("open_frames=%s · cv2_frame_count=%s · 差 %+d"
                   % (of, cv, int(of) - int(cv)))
    if of and fps:
        out.append("open_frames / cv2_fps = %.3fs，而 _video_duration=%.3fs"
                   % (float(of) / fps, float(dur or 0.0)))
        if dur and abs(float(of) / fps - float(dur)) > 1.0:
            cliff = int(round(float(dur) * fps))
            out.append("★ 两者不一致：时间轴上第 %d 帧之后，"
                       "/api/merge/preview 会把 t 钳到 %.3fs（或直接 400）"
                       % (cliff, max(0.0, float(dur) - 0.05)))
    if iiod is not None and of and fps:
        out.append("iio_duration=%.3fs × cv2_fps = %.1f 帧（容器口径）"
                   % (float(iiod), float(iiod) * fps))
    if "decoded" in d.get("real", {}):
        out.append("逐帧解到 EOF = %d 帧" % d["real"]["decoded"])
        if of and d["real"]["decoded"] != int(of):
            out.append("★ 逐帧解(%d) ≠ open_frames(%d)：open 的 frames 就是「预估值」"
                       % (d["real"]["decoded"], int(of)))
    if not out:
        out.append("没测出来")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--full", action="store_true", help="逐帧解一遍（慢）")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.video):
        print("文件不存在：%s" % a.video)
        return 2
    d = probe(a.video)
    if a.full:
        d["real"] = full_decode(a.video)
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print("\n---- 判读 ----")
    for line in verdict(d):
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
