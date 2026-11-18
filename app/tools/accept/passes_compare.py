# -*- coding: utf-8 -*-
r"""两条导出路径逐帧对拍：给两个 mp4（同一段、同参数），报逐帧差异。

用法：
    runtime\server\Scripts\python.exe app\tools\accept\passes_compare.py A.mp4 B.mp4 --frames 30

输出：每 N 帧一行的 均值/最大；末尾给汇总判定。
判据（写在脚本里，避免"看着差不多"）：
  · 均值 ≤ 2.0 LSB 且 最大 ≤ 40 LSB  → 等价（编码器差异量级）
  · 均值 ≤ 6.0                      → 接近（可接受，需要人看）
  · 其余                            → 不一致（几何/混合有问题）
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--frames", type=int, default=0, help="比对前 N 帧（0 = 全部）")
    a = ap.parse_args()
    import cv2
    import numpy as np

    ca, cb = cv2.VideoCapture(a.a), cv2.VideoCapture(a.b)
    n = 0; worst = 0; means = []
    while True:
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not (ok_a and ok_b):
            break
        if fa.shape != fb.shape:
            print("尺寸不同：%s vs %s" % (fa.shape, fb.shape))
            return 2
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        means.append(float(d.mean())); worst = max(worst, int(d.max()))
        n += 1
        if a.frames and n >= a.frames:
            break
    ca.release(); cb.release()
    if not n:
        print("没有任何可比帧")
        return 2
    mean = sum(means) / len(means)
    verdict = ("等价" if (mean <= 2.0 and worst <= 40) else
               "接近" if mean <= 6.0 else "不一致")
    print("%d 帧 · 逐帧均值 %.3f · 最差单帧最大差 %d · 判定：%s" % (n, mean, worst, verdict))
    return 0 if verdict != "不一致" else 1


if __name__ == "__main__":
    raise SystemExit(main())
