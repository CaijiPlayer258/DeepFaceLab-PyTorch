# -*- coding: utf-8 -*-
"""±128 的极端比值：288（降）与 544（升）。

网格：416→13；288→9（丢 52% 维度），544→17（不丢维度）。
候选：nearest / bicubic / lanczos / pinv(lanczos) / pinv(area) / blend(old) / blend(new)
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
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0
TARGETS = (288, 544)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    rows = []
    for res in TARGETS:
        up = res > RES0
        lam = 1e-4 if up else 1e-1
        v = {}
        v["nearest"] = A.resample_inter(raw, RES0, res, "nearest", mode="interp")
        v["bicubic"] = A.resample_inter(raw, RES0, res, "bicubic", mode="interp")
        v["lanczos"] = A.resample_inter(raw, RES0, res, "lanczos", mode="interp")
        v["pinv_lanczos"] = A.resample_inter(raw, RES0, res, "lanczos", mode="pinv", lam=lam)
        v["pinv_area"] = A.resample_inter(raw, RES0, res, "area", mode="pinv", lam=lam)
        bc = A.resample_inter(raw, RES0, res, "bicubic", mode="interp")
        pb = A.resample_inter(raw, RES0, res, "bicubic", mode="pinv", lam=lam)
        v["blend_old"] = [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pb, bc)]
        la = A.resample_inter(raw, RES0, res, "lanczos", mode="interp")
        pinv_kind = "lanczos" if up else "area"
        pn = A.resample_inter(raw, RES0, res, pinv_kind, mode="pinv", lam=lam)
        v["blend_new"] = [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pn, la)]
        rgb2, _ = R.load_rgb(res)
        for name, arrs in v.items():
            pth = TMP / ("%d_%s.pth" % (res, name))
            ir.write_pth(pth, arrs)
            e, i, d = R.build(res, pth)
            o = R.forward(e, i, d, rgb2)
            img = cv2.resize((o * 255).astype(np.uint8), (RES0, RES0), interpolation=cv2.INTER_LINEAR)
            met = masked(base, img.astype(np.float32) / 255.0, m0)
            st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
            rows.append(dict(res=res, method=name, **{k: float(x) for k, x in met.items()},
                             **{k: float(x) for k, x in st.items()}))
            print("%-4d %-12s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f hf=%.2f"
                  % (res, name, met["psnr"], met["ssim"], met["mae"], st["ncc"], st["hf_ratio"]),
                  flush=True)
            cv2.imwrite(str(TMP / ("%d_%s.png" % (res, name))), img)
    (OUT / "results_extreme.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
