# -*- coding: utf-8 -*-
r"""PNG 编码档位表：level × 策略 × 线程的 ms / 体积，并验证「解码逐位不变」。

为什么单独一条：项目导出的 .png 帧是整条链上最贵的纯 CPU 活（改之前 1080×1920 单线程
134.8 ms/帧），而 PNG 是**无损**格式 —— 换 level/策略只改字节流，解码出来的像素
一个 bit 都不会变。这条工具就是把「省多少时间 / 体积怎么变 / 像素不变」一次量齐。

用法（仓库根，用 server 的 venv）：
    runtime\server\Scripts\python.exe app\tools\bench\png_encode_probe.py
    runtime\server\Scripts\python.exe app\tools\bench\png_encode_probe.py --src <某目录> --frames 8

输出（1080×1920 实测，2026-12，本机）：
    level 3 · 默认策略  134.8 ms/帧 · 1.241 MB     ← 项目导出改之前就是这个
    level 3 · Z_RLE      75.7 ms/帧 · 1.223 MB
    level 1 · 默认策略   78.2 ms/帧 · 1.362 MB
    level 1 · Z_RLE      72.2 ms/帧 · 1.223 MB     ← 现默认（又快又小）
    level 0              46.3 ms/帧 · 5.944 MB     ← 快但体积翻 4.8 倍
线程：Z_RLE 下 1/2/3/4/6/8 线程的墙钟 ≈ 67/33/22/17/12/9 ms/帧（近线性，cv2.imencode 放 GIL）。
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
DEFAULT_SRC = TEMP / "dfl_pool_accept" / "proj"


def load_frames(src: Path, n: int):
    import cv2
    import numpy as np
    imgs = []
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                imgs.append(im)
            if len(imgs) >= n:
                break
    if not imgs:                      # 没有素材就造一张：1080×1920 的渐变 + 噪声
        rng = np.random.default_rng(7)
        base = np.linspace(0, 255, 1080, dtype=np.float32)[None, :, None]
        im = np.repeat(base, 1920, axis=0).repeat(3, axis=2)
        im = np.clip(im + rng.integers(-6, 7, im.shape), 0, 255).astype(np.uint8)
        imgs = [im]
    return imgs


def encode(im, level, rle):
    import cv2
    params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(level)]
    k = getattr(cv2, "IMWRITE_PNG_STRATEGY", None)
    v = getattr(cv2, "IMWRITE_PNG_STRATEGY_RLE", None)
    if rle and k is not None and v is not None:
        params += [int(k), int(v)]
    ok, buf = cv2.imencode(".png", im, params)
    if not ok:
        raise RuntimeError("imencode 失败")
    return buf.tobytes()


def main():
    import cv2
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    imgs = load_frames(Path(a.src), max(1, int(a.frames)))
    say = lambda s: print(s, flush=True)
    say("素材 %d 帧 %s" % (len(imgs), imgs[0].shape))
    rows = []
    for level in (3, 2, 1, 0):
        for rle in (False, True):
            ts, sizes, ok_all = [], [], True
            for im in imgs:
                t = time.perf_counter()
                b = encode(im, level, rle)
                ts.append((time.perf_counter() - t) * 1000.0)
                sizes.append(len(b))
                back = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                ok_all = ok_all and np.array_equal(back, im)
            row = {"level": level, "strategy": ("rle" if rle else "default"),
                   "ms": round(statistics.mean(ts), 1),
                   "mb": round(statistics.mean(sizes) / 1048576.0, 3),
                   "pixels_identical": bool(ok_all)}
            rows.append(row)
            say("  level %d %-7s  %6.1f ms/帧  %5.3f MB  解码逐位相同=%s"
                % (level, row["strategy"], row["ms"], row["mb"], row["pixels_identical"]))
    say("线程扩展（level 1 + Z_RLE）：")
    for n in (1, 2, 3, 4, 6, 8):
        q = queue.Queue(maxsize=8)
        done = []
        def worker():
            while True:
                im = q.get()
                if im is None:
                    return
                t = time.perf_counter()
                encode(im, 1, True)
                done.append((time.perf_counter() - t) * 1000.0)
        ths = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
        for th in ths:
            th.start()
        seq = [imgs[i % len(imgs)] for i in range(24)]
        t0 = time.perf_counter()
        for im in seq:
            q.put(im)
        for _ in ths:
            q.put(None)
        for th in ths:
            th.join()
        wall = (time.perf_counter() - t0) * 1000.0 / len(seq)
        say("  %d 线程：墙钟 %.1f ms/帧 · 单帧 %.1f ms" % (n, wall, statistics.mean(done)))
    say(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
