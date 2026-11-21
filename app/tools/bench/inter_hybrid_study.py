# -*- coding: utf-8 -*-
"""第三轮：把「抗混叠」和「换核」合并成一个候选。

观察（第二轮）：
  · 纯插值里 lanczos-3 全面优于 bicubic（448: 25.41 vs 23.41）
  · 降采样时 pinv(area)（box 预滤波 + 逆滤波）MAE/NCC 明显最好（320: 16.5 / 0.781）
  · 所以「基座核」和「逆滤波核」不必是同一个：
        升采样：基座 lanczos，逆滤波 lanczos（无混叠问题）
        降采样：基座 lanczos，逆滤波 area  （area 自带抗混叠）
    再 0.5/0.5 混合。
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
TMP = OUT / "cand3"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    rows = []
    for res in (320, 384, 448, 512):
        up = res > RES0
        lam = 1e-4 if up else 1e-1
        base_kind = "lanczos"
        inv_kind = "lanczos" if up else "area"
        bic = A.resample_inter(raw, RES0, res, base_kind, mode="interp")
        pin = A.resample_inter(raw, RES0, res, inv_kind, mode="pinv", lam=lam)
        cands = {
            "hybrid_blend": [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pin, bic)],
            "hybrid_pinv": pin,
        }
        # 对照：全 lanczos 的 blend
        pin_l = A.resample_inter(raw, RES0, res, "lanczos", mode="pinv", lam=lam)
        cands["blend_lanczos"] = [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pin_l, bic)]
        rgb2, _ = R.load_rgb(res)
        for name, arrs in cands.items():
            pth = TMP / ("%d_%s.pth" % (res, name))
            ir.write_pth(pth, arrs)
            e, i, d = R.build(res, pth)
            o = R.forward(e, i, d, rgb2)
            up_img = cv2.resize((o * 255).astype(np.uint8), (RES0, RES0),
                                interpolation=cv2.INTER_LINEAR)
            met = masked(base, up_img.astype(np.float32) / 255.0, m0)
            st = structure_metrics(base, up_img.astype(np.float32) / 255.0, m0)
            rows.append(dict(res=res, method=name,
                             **{k: float(v) for k, v in met.items()},
                             **{k: float(v) for k, v in st.items()}))
            print("%-4d %-14s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f hf=%.2f"
                  % (res, name, met["psnr"], met["ssim"], met["mae"], st["ncc"], st["hf_ratio"]),
                  flush=True)
            if name == "hybrid_blend":
                cv2.imwrite(str(TMP / ("%d_%s.png" % (res, name))), up_img)
    (OUT / "results_hybrid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
