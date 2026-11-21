# -*- coding: utf-8 -*-
"""新 blend（dense1=pinv, dense2/bias=lanczos）在六档分辨率上的统一评测 + 出图。"""
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
TMP = OUT / "cand_final"
RES = (288, 320, 384, 448, 512, 544)


def resample(raw, res, d1_kind, d2_kind):
    up = res > R.RES0
    lam = 1e-4 if up else 1e-1
    d1 = A.resample_inter(raw, R.RES0, res, d1_kind, mode="pinv", lam=lam) if d1_kind else None
    d2 = A.resample_inter(raw, R.RES0, res, d2_kind, mode="interp")
    return [d1[0], d1[1], d2[2], d2[3]]


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(R.MODEL / (R.NAME + "_inter.pth"))
    rows = []
    for res in RES:
        up = res > R.RES0
        d1_kind = "lanczos" if up else "area"     # pinv 的基座核
        arrs = resample(raw, res, d1_kind, "lanczos")
        pth = TMP / ("blend_%d.pth" % res)
        ir.write_pth(pth, arrs)
        e, i, d = R.build(res, pth)
        rgb2, _ = R.load_rgb(res)
        o = R.forward(e, i, d, rgb2)
        img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(TMP / ("blend_%d.png" % res)), img)
        met = masked(base, img.astype(np.float32) / 255.0, m0)
        st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
        rows.append(dict(res=res, psnr=float(met["psnr"]), ssim=float(met["ssim"]),
                         mae=float(met["mae"]), ncc=float(st["ncc"])))
        print("%-4d blend(d1=pinv,d2=lanczos) psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f"
              % (res, met["psnr"], met["ssim"], met["mae"], st["ncc"]), flush=True)
    (OUT / "results_blend_final.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
