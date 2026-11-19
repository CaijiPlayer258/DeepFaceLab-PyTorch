# -*- coding: utf-8 -*-
r"""单脸对拍：dump（快速导出）× chunk（分趟）在同一帧上比 pred/遮罩/68 点。

为什么单独一个脚本：索引、帧号、脸的顺序三条路都可能不同，先断言再比，
不然算出来的 LSB 只是「两件不同的事」的差（这个坑我已经踩过两次）。

用法：
    python app/tools/accept/face_ab.py --dump <out>.face.npz --chunk chunks/x.npz --frame 90
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--max-shift", type=float, default=30.0)
    a = ap.parse_args()

    z = np.load(a.dump)
    zc = np.load(a.chunk)
    pre = "f%06d_" % int(a.frame)
    keys = [k for k in zc.files if k.startswith(pre) and k.endswith("_lm")]
    if not keys:
        print("chunk 里没有第 %d 帧" % a.frame)
        return 2
    fused = {i: np.asarray(z["f%d_landmarks" % i], np.float32)
             for i in range(8) if ("f%d_landmarks" % i) in z.files}
    if not fused:
        print("dump 里没有 landmarks")
        return 2
    ok = 0
    for k in keys:
        lm = np.asarray(zc[k], np.float32)
        c0 = lm.mean(axis=0)
        best, bd = None, 1e18
        for i, flm in fused.items():
            d = float(np.linalg.norm(flm.mean(axis=0) - c0))
            if d < bd:
                bd, best = d, i
        key = k[:-3]
        print("%s ↔ f%d · 中心距 %.1f px" % (key, best, bd))
        if bd > a.max_shift:
            print("  配对不可信（>%.0f px），拒绝给结论" % a.max_shift)
            continue
        for short, suf in (("pred", "_pred"), ("prd_dfm", "_mask"),
                           ("xseg_prd", "_xprd"), ("landmarks", "_lm")):
            fk, ck = "f%d_%s" % (best, short), key + suf
            if fk not in z.files or ck not in zc.files:
                continue
            x = np.asarray(z[fk], np.float32); y = np.asarray(zc[ck], np.float32)
            # 量纲要统一：dump 里是 0..1（compose 的入参），chunk 里存的是 u8 0..255。
            # 不归一化就会得到「均值差 164」这种假差异（这个坑刚踩过）。
            if float(y.max()) > 1.5 and float(x.max()) <= 1.5:
                y = y / 255.0
            if x.shape != y.shape:
                print("  %-9s 形状不同 %s vs %s" % (short, x.shape, y.shape)); continue
            d = np.abs(x - y)
            print("  %-9s 均值 %.3f · 最大 %.2f" % (short, float(d.mean()), float(d.max())))
        ok += 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
