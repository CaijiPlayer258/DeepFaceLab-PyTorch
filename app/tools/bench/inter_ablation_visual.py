# -*- coding: utf-8 -*-
"""出图：按张量分组的消融（3 个分辨率 × 4 种组合）。"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app" / "engine"))

import cv2                                                      # noqa: E402
import numpy as np                                              # noqa: E402

import app.tools.bench.inter_study_run as R                      # noqa: E402
from app.tools.bench.xseg_mask import mask_of, masked, structure_metrics   # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
TMP = OUT / "cand_ab"
COMBOS = [("d1pinv_d2lanczos", "d1=pinv  d2=lanczos"),
          ("d1pinv_d2blend", "d1=pinv  d2=blend"),
          ("d1blend_d2lanczos", "d1=blend  d2=lanczos"),
          ("d1blend_d2blend", "d1=blend  d2=blend")]
RES = (384, 448, 512)


def main():
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    for res in RES:
        rgb2, _ = R.load_rgb(res)
        for key, label in COMBOS:
            pth = TMP / ("%d_%s.pth" % (res, key))
            if not pth.exists():
                print("missing", pth, flush=True)
                continue
            e, i, d = R.build(res, pth)
            o = R.forward(e, i, d, rgb2)
            img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0),
                             interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(TMP / ("%d_%s.png" % (res, key))), img)
            met = masked(base, img.astype("float32") / 255.0, m0)
            print("%-4d %-20s psnr=%.2f ssim=%.4f" % (res, label, met["psnr"], met["ssim"]), flush=True)


if __name__ == "__main__":
    main()
