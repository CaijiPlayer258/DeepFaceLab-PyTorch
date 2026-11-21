# -*- coding: utf-8 -*-
"""交叉组合：dense1 与 dense2/bias 各取自己的最优算子。"""
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
TMP = OUT / "cand_ab"


def parts(kind, res):
    raw = R._read_pth(R.MODEL / (R.NAME + "_inter.pth"))
    up = res > R.RES0
    lam = 1e-4 if up else 1e-1
    if kind == "blend":
        la = A.resample_inter(raw, R.RES0, res, "lanczos", mode="interp")
        pk = "lanczos" if up else "area"
        pn = A.resample_inter(raw, R.RES0, res, pk, mode="pinv", lam=lam)
        return [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pn, la)]
    if kind == "pinv":
        pk = "lanczos" if up else "area"
        return A.resample_inter(raw, R.RES0, res, pk, mode="pinv", lam=lam)
    return A.resample_inter(raw, R.RES0, res, kind, mode="interp")


def main():
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    rows = []
    for res in (384, 448, 512):
        rgb2, _ = R.load_rgb(res)
        cache = {k: parts(k, res) for k in ("bicubic", "lanczos", "pinv", "blend")}
        combos = [("d1=pinv d2=lanczos", "pinv", "lanczos"),
                  ("d1=pinv d2=blend", "pinv", "blend"),
                  ("d1=blend d2=lanczos", "blend", "lanczos"),
                  ("d1=blend d2=blend", "blend", "blend")]
        for label, a, b in combos:
            pa, pb = cache[a], cache[b]
            arrs = [pa[0], pa[1], pb[2], pb[3]]
            pth = TMP / ("%d_%s.pth" % (res, label.replace(" ", "_").replace("=", "")))
            ir.write_pth(pth, arrs)
            e, i, d = R.build(res, pth)
            o = R.forward(e, i, d, rgb2)
            img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0), interpolation=cv2.INTER_LINEAR)
            met = masked(base, img.astype(np.float32) / 255.0, m0)
            st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
            rows.append(dict(res=res, combo=label, psnr=float(met["psnr"]), ssim=float(met["ssim"]),
                             mae=float(met["mae"]), ncc=float(st["ncc"])))
            print("%-4d %-22s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f"
                  % (res, label, met["psnr"], met["ssim"], met["mae"], st["ncc"]), flush=True)
    (OUT / "results_ablation_combo.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
