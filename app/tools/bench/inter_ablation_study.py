# -*- coding: utf-8 -*-
"""消融：算法选择到底作用在 dense1 还是 dense2/bias？

在 448 上把两者解耦：
  A 组：dense1 换方法，dense2/bias 固定 bicubic
  B 组：dense1 固定 blend，dense2/bias 换方法
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
TMP = OUT / "cand_ab"
RES = 448


def parts(kind):
    """返回 (dense1_w, dense1_b, dense2_w, dense2_b) 按 kind 重采样。"""
    raw = R._read_pth(R.MODEL / (R.NAME + "_inter.pth"))
    if kind == "blend":
        la = A.resample_inter(raw, R.RES0, RES, "lanczos", mode="interp")
        pn = A.resample_inter(raw, R.RES0, RES, "lanczos", mode="pinv", lam=1e-4)
        return [(0.5 * x + 0.5 * y).astype(np.float32) for x, y in zip(pn, la)]
    if kind == "pinv":
        return A.resample_inter(raw, R.RES0, RES, "lanczos", mode="pinv", lam=1e-4)
    return A.resample_inter(raw, R.RES0, RES, kind, mode="interp")


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(R.RES0)
    m0 = mask_of(bgr0)
    e0, i0, d0 = R.build(R.RES0, R.MODEL / (R.NAME + "_inter.pth"))
    base = R.forward(e0, i0, d0, rgb0)
    rgb2, _ = R.load_rgb(RES)
    cache = {k: parts(k) for k in ("bicubic", "lanczos", "pinv", "blend")}
    ref = cache["bicubic"]
    rows = []
    cases = []
    for k in ("nearest", "bicubic", "lanczos", "pinv", "blend"):
        cases.append((f"A dense1={k}", "dense1", k))
        cases.append((f"B dense2={k}", "dense2", k))
    for label, which, k in cases:
        if k not in cache:
            cache[k] = parts(k)
        v = cache[k]
        if which == "dense1":                 # 只换 dense1，dense2/bias 用 bicubic
            arrs = [v[0], v[1], ref[2], ref[3]]
        else:                                  # 只换 dense2/bias，dense1 用 blend
            b = cache["blend"]
            arrs = [b[0], b[1], v[2], v[3]]
        pth = TMP / ("448_%s.pth" % label.replace(" ", "_"))
        ir.write_pth(pth, arrs)
        e, i, d = R.build(RES, pth)
        o = R.forward(e, i, d, rgb2)
        img = cv2.resize((o * 255).astype(np.uint8), (R.RES0, R.RES0), interpolation=cv2.INTER_LINEAR)
        met = masked(base, img.astype(np.float32) / 255.0, m0)
        st = structure_metrics(base, img.astype(np.float32) / 255.0, m0)
        rows.append(dict(case=label, psnr=float(met["psnr"]), ssim=float(met["ssim"]),
                         mae=float(met["mae"]), ncc=float(st["ncc"])))
        print("%-18s psnr=%5.2f ssim=%.4f mae=%5.2f ncc=%.4f"
              % (label, met["psnr"], met["ssim"], met["mae"], st["ncc"]), flush=True)
    (OUT / "results_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
