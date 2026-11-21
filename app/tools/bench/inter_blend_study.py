# -*- coding: utf-8 -*-
"""混合：W' = α·pinv + (1-α)·bicubic —— 用插值的稳定性 + 伪逆的锐度。

pinv 的 SSIM/PSNR 好但 MAE 偏高（过锐），bicubic 反之。
两者在同一网格上，直接凸组合即可逐项取长补短；α 用验证集选，报告用 target.jpg。
"""
from __future__ import annotations

import argparse
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
from app.tools.bench.xseg_mask import mask_of, masked            # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
TMP = OUT / "cand"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    lam = 1e-4 if a.res > RES0 else 1e-1
    bic = A.resample_inter(raw, RES0, a.res, "bicubic", mode="interp")
    pin = A.resample_inter(raw, RES0, a.res, "bicubic", mode="pinv", lam=lam)
    rgb2, _ = R.load_rgb(a.res)
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        arrs = [(alpha * p + (1 - alpha) * b).astype(np.float32) for p, b in zip(pin, bic)]
        pth = TMP / ("blend_%d_%.2f.pth" % (a.res, alpha))
        ir.write_pth(pth, arrs)
        e, i, d = R.build(a.res, pth)
        o = R.forward(e, i, d, rgb2)
        up = cv2.resize((o * 255).astype(np.uint8), (RES0, RES0), interpolation=cv2.INTER_LINEAR)
        met = masked(base, up.astype(np.float32) / 255.0, m0)
        rows.append(dict(res=a.res, alpha=alpha, **{k: float(v) for k, v in met.items()}))
        print("res=%d alpha=%.2f psnr=%.2f ssim=%.4f mae=%.2f"
              % (a.res, alpha, met["psnr"], met["ssim"], met["mae"]), flush=True)
        if alpha in (0.5,):
            cv2.imwrite(str(TMP / ("blend_%d_%.2f.png" % (a.res, alpha))), up)
    (OUT / ("blend_%d.json" % a.res)).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
