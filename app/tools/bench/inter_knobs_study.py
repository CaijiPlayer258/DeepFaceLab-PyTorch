# -*- coding: utf-8 -*-
"""确认「d1=pinv / d2=lanczos」是否真的最优：三个剩余旋钮的补测。

1) 降采样时 dense2 用 lanczos 还是 area（抗混叠）
2) 升采样时 dense1 的 λ
3) dense2 是否值得来一点点逆滤波（α 小的部分逆滤波）
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
TMP = OUT / "cand_knobs"


def build(raw, res, d1_kind, d1_lam, d2_arr):
    d1 = A.resample_inter(raw, R.RES0, res, d1_kind, mode="pinv", lam=d1_lam)
    return [d1[0], d1[1], d2_arr[2], d2_arr[3]]


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(R.MODEL / (R.NAME + "_inter.pth"))
    rows = []

    def run(label, res, arrs):
        pth = TMP / ("%d_%s.pth" % (res, label.replace(" ", "_").replace("=", "-")))
        ir.write_pth(pth, arrs)
        e, i, d = R.build(res, pth)
        rgb2, _ = R.load_rgb(res)
        o = R.forward(e, i, d, rgb2)
        img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0), interpolation=cv2.INTER_LINEAR)
        met = masked(base, img.astype(np.float32) / 255.0, m0)
        st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
        rows.append(dict(res=res, label=label, psnr=float(met["psnr"]), ssim=float(met["ssim"]),
                         mae=float(met["mae"]), ncc=float(st["ncc"])))
        print("%-4d %-28s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f"
              % (res, label, met["psnr"], met["ssim"], met["mae"], st["ncc"]), flush=True)

    # 1) 降采样：dense2 用 lanczos / area / bicubic
    for res in (320, 384):
        d1_kind = "area"
        for k in ("lanczos", "area", "bicubic"):
            d2 = A.resample_inter(raw, R.RES0, res, k, mode="interp")
            run("d2=" + k, res, build(raw, res, d1_kind, 1e-1, d2))

    # 2) 升采样：dense1 的 λ
    for res in (448, 512):
        d2 = A.resample_inter(raw, R.RES0, res, "lanczos", mode="interp")
        for lam in (1e-5, 1e-4, 1e-3, 1e-2):
            run("d1lam=%g" % lam, res, build(raw, res, "lanczos", lam, d2))

    # 3) dense2 加一点逆滤波（α）？
    d2_lan = None
    for res in (448, 512):
        d2_lan = A.resample_inter(raw, R.RES0, res, "lanczos", mode="interp")
        d2_pin = A.resample_inter(raw, R.RES0, res, "lanczos", mode="pinv", lam=1e-4)
        for a in (0.0, 0.15, 0.3):
            mix = [(a * p + (1 - a) * l).astype(np.float32) for p, l in zip(d2_pin, d2_lan)]
            run("d2a=%g" % a, res, build(raw, res, "lanczos", 1e-4, mix))

    (OUT / "results_knobs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
