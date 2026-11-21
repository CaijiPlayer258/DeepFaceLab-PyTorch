# -*- coding: utf-8 -*-
"""给报告出图：逐组合输出 + 一张「分辨率 × 方法」对照矩阵。"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app" / "engine"))

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

from app.tools.bench import inter_resample as ir                 # noqa: E402
import app.tools.bench.inter_study_run as R                      # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
MODEL, NAME = R.MODEL, R.NAME
RES0 = R.RES0
TARGETS = (320, 384, 448, 512)
METHODS = ("nearest", "bilinear", "bicubic", "area")


def label(img, text, h=22):
    pad = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(pad, text, (4, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([pad, img])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    cv2.imwrite(str(OUT / "input_416.png"), bgr0)
    enc, inter, dec = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    base = R.forward(enc, inter, dec, rgb0)
    base_u8 = (base * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "base416.png"), base_u8)

    cells = {}
    for res in TARGETS:
        for meth in METHODS:
            p = OUT / ("inter_%d_%s.pth" % (res, meth))
            if not p.exists():
                ir.convert(str(MODEL / (NAME + "_inter.pth")), RES0, res, meth, str(p))
            e2, i2, d2 = R.build(res, p)
            rgb2, _ = R.load_rgb(res)
            out = R.forward(e2, i2, d2, rgb2)
            up = cv2.resize((out * 255).astype(np.uint8), (RES0, RES0),
                            interpolation=cv2.INTER_LINEAR)
            cells[(res, meth)] = up
            cv2.imwrite(str(OUT / ("out_%d_%s.png" % (res, meth))), up)
            print("done", res, meth, flush=True)

    # 矩阵：行=分辨率，列=方法；角落放 baseline
    cell = 208
    def small(im):
        return cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
    rows = []
    head = [label(np.zeros((cell, cell, 3), np.uint8), "method →")] + \
           [label(np.zeros((cell, cell, 3), np.uint8), m) for m in METHODS]
    rows.append(np.hstack(head))
    for res in TARGETS:
        line = [label(np.zeros((cell, cell, 3), np.uint8), "%d" % res)]
        for m in METHODS:
            line.append(label(small(cells[(res, m)]), ""))
        rows.append(np.hstack(line))
    grid = np.vstack(rows)
    # 顶部放 input / baseline
    blank = label(np.zeros((cell, cell, 3), np.uint8), "")
    top = np.hstack([label(small(bgr0), "input 416"),
                     label(small(base_u8), "baseline 416"), blank, blank, blank])
    cv2.imwrite(str(OUT / "grid_matrix.png"), np.vstack([top, grid]))
    print("matrix ok", flush=True)


if __name__ == "__main__":
    main()
