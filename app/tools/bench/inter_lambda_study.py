# -*- coding: utf-8 -*-
"""288（比值 1.44）：λ 需要随比值放大 —— 扫一遍。

假设：Tikhonov 的 λ 应该随 L 的条件数增长，粗略 λ ∝ (比值−1)²。
这里只测 288，看 blend 能否被救回来。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app" / "engine"))

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

import app.tools.bench.inter_study_run as R                      # noqa: E402
import app.tools.bench.inter_adv as A                            # noqa: E402
import app.tools.bench.inter_resample as ir                      # noqa: E402
from app.tools.bench.xseg_mask import mask_of, masked, structure_metrics   # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
TMP = OUT / "cand128"
RES = 288


def main():
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(R.MODEL / (R.NAME + "_inter.pth"))
    lan = A.resample_inter(raw, R.RES0, RES, "lanczos", mode="interp")
    rows = []
    for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
        pin = A.resample_inter(raw, R.RES0, RES, "area", mode="pinv", lam=lam)
        for tag, arrs in (("pinv_area", pin),
                          ("blend", [(0.5 * x + 0.5 * y).astype(np.float32)
                                     for x, y in zip(pin, lan)])):
            pth = TMP / ("%d_lam%g_%s.pth" % (RES, lam, tag))
            ir.write_pth(pth, arrs)
            e, i, d = R.build(RES, pth)
            rgb2, _ = R.load_rgb(RES)
            o = R.forward(e, i, d, rgb2)
            img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0),
                             interpolation=cv2.INTER_LINEAR)
            met = masked(base, img.astype(np.float32) / 255.0, m0)
            st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
            rows.append(dict(res=RES, lam=lam, method=tag,
                             **{k: float(v) for k, v in met.items()},
                             **{k: float(v) for k, v in st.items()}))
            print("lam=%-5g %-10s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f"
                  % (lam, tag, met["psnr"], met["ssim"], met["mae"], st["ncc"]), flush=True)
            if tag == "blend":
                cv2.imwrite(str(TMP / ("%d_lam%g_blend.png" % (RES, lam))), img)
    (OUT / "results_lambda288.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
