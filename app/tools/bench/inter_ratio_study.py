# -*- coding: utf-8 -*-
"""第二轮研究：核 × 逆滤波 × 抗混叠，以及「结构/锐度」分离指标。

候选（每个目标分辨率）：
  nearest / bilinear / bicubic / lanczos            —— 纯插值
  pinv(bicubic) / pinv(lanczos)                     —— 逆滤波，换核
  pinv(area)                                        —— 抗混叠预滤波（box）+ 逆滤波（仅降采样有意义）
  blend(bicubic) 0.5 / blend(lanczos) 0.5           —— 部分逆滤波
指标：遮罩内 PSNR / SSIM / MAE + 结构 NCC（低频）+ 高频能量比（锐度）
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
TMP = OUT / "cand2"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0
RES_LIST = (320, 384, 448, 512)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    rows = []
    for res in RES_LIST:
        lam = 1e-4 if res > RES0 else 1e-1
        variants = {}
        for kind in ("nearest", "bilinear", "bicubic", "lanczos"):
            variants[kind] = A.resample_inter(raw, RES0, res, kind, mode="interp")
        for kind in ("bicubic", "lanczos", "area"):
            variants["pinv_" + kind] = A.resample_inter(raw, RES0, res, kind,
                                                        mode="pinv", lam=lam)
        for kind in ("bicubic", "lanczos"):
            bic = A.resample_inter(raw, RES0, res, kind, mode="interp")
            pin = A.resample_inter(raw, RES0, res, kind, mode="pinv", lam=lam)
            variants["blend_" + kind] = [(0.5 * x + 0.5 * y).astype(np.float32)
                                         for x, y in zip(pin, bic)]
        rgb2, _ = R.load_rgb(res)
        for name, arrs in variants.items():
            pth = TMP / ("%d_%s.pth" % (res, name))
            ir.write_pth(pth, arrs)
            e, i, d = R.build(res, pth)
            o = R.forward(e, i, d, rgb2)
            up = cv2.resize((o * 255).astype(np.uint8), (RES0, RES0),
                            interpolation=cv2.INTER_LINEAR)
            met = masked(base, up.astype(np.float32) / 255.0, m0)
            st = structure_metrics(base, up.astype(np.float32) / 255.0, m0)
            row = dict(res=res, method=name, **{k: float(v) for k, v in met.items()},
                       **{k: float(v) for k, v in st.items()})
            rows.append(row)
            print("%-4d %-14s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f hf=%.2f"
                  % (res, name, met["psnr"], met["ssim"], met["mae"], st["ncc"], st["hf_ratio"]),
                  flush=True)
            if name in ("lanczos", "pinv_lanczos", "blend_lanczos", "pinv_area"):
                cv2.imwrite(str(TMP / ("%d_%s.png" % (res, name))), up)
    (OUT / "results_ratio.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
