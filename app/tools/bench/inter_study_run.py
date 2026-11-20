# -*- coding: utf-8 -*-
"""inter 分辨率实验：用仓库自带的 leras 前向（不自己实现网络）。

步骤：初始化 leras → 建 Encoder/Inter/Decoder → load_weights(.pth)
      → 416 baseline → 320/384/448/512 × 4 方法 → 与 baseline 比 → 出图 + JSON
用法：python app/tools/bench/inter_study_run.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ENGINE = REPO / "app" / "engine"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(REPO))

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402

MODEL = REPO / "workspace" / "model"
NAME = "New416_SAEHD"
TARGET = REPO / "app" / "docs" / "notes" / "target.jpg"
OUT = REPO / "app" / "docs" / "notes" / "inter_study"
RES0 = 416
DOWNS = (320, 384)
UPS = (448, 512)
METHODS = ("nearest", "bilinear", "bicubic", "area")


def _init_leras():
    errs = []
    try:
        from core.interact import interact as io
        io.initialize_main_env()
    except Exception as e:                                      # noqa: BLE001
        errs.append("io:%r" % (e,))
    import core.leras.layers                     # noqa: F401  ← 注册 ModelBase/Conv2D/Dense
    import core.leras.archis                     # noqa: F401
    from core.leras.nn import nn as NS           # 命名空间对象（不是模块）
    for kwargs in ({}, {"device_config": None}):
        try:
            NS.initialize(**kwargs)
        except Exception as e:                                  # noqa: BLE001
            errs.append("init%r:%r" % (kwargs, e))
        if hasattr(NS, "ModelBase"):
            return NS, errs
    return NS, errs


def build(res, inter_path):
    import torch
    lnn, errs = _init_leras()
    if not hasattr(lnn, "ModelBase"):
        raise SystemExit("leras 初始化失败：%s" % errs[-2:])
    from core.leras.archis.DeepFakeArchi import DeepFakeArchi
    archi = DeepFakeArchi(res, opts="udt")
    low = res // 32
    enc = archi.Encoder(in_ch=3, e_ch=80)
    inter = archi.Inter(in_ch=80 * 8 * low * low, ae_ch=320, ae_out_ch=320)
    dec = archi.Decoder(in_ch=320, d_ch=80, d_mask_ch=26)
    enc.load_weights(str(MODEL / (NAME + "_encoder.pth")))
    inter.load_weights(str(inter_path))
    dec.load_weights(str(MODEL / (NAME + "_decoder_src.pth")))
    for m in (enc, inter, dec):
        m.eval()
    return enc, inter, dec


def forward(enc, inter, dec, rgb):
    import torch
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        z = inter(enc(x))
        out = dec(z)
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o[0].permute(1, 2, 0).clamp(0, 1).numpy()


def load_rgb(res):
    img = cv2.imread(str(TARGET), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, img


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim(a, b):
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m1 = cv2.GaussianBlur(ga, (11, 11), 1.5); m2 = cv2.GaussianBlur(gb, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - m1 * m1
    s2 = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - m2 * m2
    s12 = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - m1 * m2
    return float((((2 * m1 * m2 + C1) * (2 * s12 + C2)) /
                  ((m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2))).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from app.tools.bench import inter_resample as ir
    rgb0, bgr0 = load_rgb(RES0)
    enc, inter, dec = build(RES0, MODEL / (NAME + "_inter.pth"))
    base = forward(enc, inter, dec, rgb0)
    print("baseline ok, mean=%.4f" % base.mean(), flush=True)
    cv2.imwrite(str(OUT / "base416.png"), cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    rows, tiles = [], [("baseline 416", cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))]
    for res in DOWNS + UPS:
        for meth in METHODS:
            p = OUT / ("inter_%d_%s.pth" % (res, meth))
            ir.convert(str(MODEL / (NAME + "_inter.pth")), RES0, res, meth, str(p))
            e2, i2, d2 = build(res, p)
            rgb2, _b = load_rgb(res)
            out = forward(e2, i2, d2, rgb2)
            up = cv2.resize((out * 255).astype(np.uint8), (RES0, RES0),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            row = dict(res=res, method=meth, psnr=psnr(base, up), ssim=ssim(base, up),
                       mae=float(np.abs(base - up).mean() * 255.0))
            rows.append(row)
            print("%-4d %-9s PSNR=%.2f SSIM=%.4f MAE=%.2f"
                  % (res, meth, row["psnr"], row["ssim"], row["mae"]), flush=True)
            if meth in ("bilinear", "bicubic"):
                tiles.append(("%s %d" % (meth, res),
                              cv2.cvtColor((up * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    h = w = RES0 // 2
    tiles = [(lab, cv2.resize(im, (w, h))) for lab, im in tiles]
    cols, n = 3, len(tiles)
    grid = np.zeros((((n + cols - 1) // cols) * (h + 26), cols * w, 3), np.uint8)
    for i, (lab, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * (h + 26):r * (h + 26) + h, c * w:c * w + w] = im
        cv2.putText(grid, lab, (c * w + 6, r * (h + 26) + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "compare_grid.png"), grid)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
