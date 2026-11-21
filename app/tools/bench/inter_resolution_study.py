# -*- coding: utf-8 -*-
"""inter 换分辨率实验（完整版）。

流程：
  1. target.jpg → Haar 找最大脸 → 外扩成方形 → 416 对齐式裁切（face crop）
  2. 原模型（416）前向 → baseline
  3. inter 换到 {256,320,384,512} × {nearest,bilinear,bicubic,area}，同权重前向，
     输出统一升回 416 与 baseline 比（PSNR / SSIM / MAE）
  4. 另做一组「网格用错」的对照（把 t 模型的 enc 网格当成 res/16），量化后缀差异
  5. 出对比图 + JSON 结果
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from app.tools.bench import inter_resample as ir     # noqa: E402
from app.tools.bench import saehd_torch as st        # noqa: E402

MODEL = REPO / "workspace" / "model"
TARGET = REPO / "app" / "docs" / "notes" / "target.jpg"
OUT = REPO / "app" / "docs" / "notes" / "inter_study"
RES0 = 416
METHODS = ("nearest", "bilinear", "bicubic", "area")
TARGETS = (256, 320, 384, 512)


def face_crop(path, res=RES0):
    """target.jpg 本身**就是已对齐的人脸**，这里只做缩放，不做任何检测/裁剪。"""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("读不到 %s" % path)
    if img.shape[0] != res or img.shape[1] != res:
        img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, img, "as-is"


def to_t(x):
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0


def forward(enc, inter, dec, rgb, device):
    with torch.no_grad():
        z = inter(enc(to_t(rgb).to(device)))
        out, _m = dec(z)
    return out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim(a, b):
    import cv2
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu1 = cv2.GaussianBlur(ga, k, 1.5)
    mu2 = cv2.GaussianBlur(gb, k, 1.5)
    s1 = cv2.GaussianBlur(ga * ga, k, 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(gb * gb, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(ga * gb, k, 1.5) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float((num / den).mean())


def build_at(res, inter_path, device):
    """按目标分辨率重建模型；encoder/decoder 权重与分辨率无关，inter 用转好的。"""
    enc = st.Encoder()
    st.assign(enc, st.read_pth(MODEL / "New416_SAEHD_encoder.pth"), "encoder")
    dec = st.Decoder()
    st.assign(dec, st.read_pth(MODEL / "New416_SAEHD_decoder_src.pth"), "decoder_src")
    inter = st.Inter(res)
    st.assign(inter, st.read_pth(inter_path), "inter@%d" % res)
    return enc.to(device).eval(), inter.to(device).eval(), dec.to(device).eval()


def main():
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    rgb, crop_bgr, how = face_crop(TARGET)
    print("crop:", how, crop_bgr.shape, flush=True)
    cv2.imwrite(str(OUT / "face_416.png"), crop_bgr)

    t0 = time.time()
    enc, inter, dec = build_at(RES0, MODEL / "New416_SAEHD_inter.pth", device)
    base = forward(enc, inter, dec, rgb, device)
    print("baseline done %.1fs" % (time.time() - t0), flush=True)
    cv2.imwrite(str(OUT / "out_base416.png"),
                cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    rows = []
    tiles = [("baseline 416", cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))]
    for res in TARGETS:
        for meth in METHODS:
            pth = OUT / ("inter_%d_%s.pth" % (res, meth))
            ir.convert(str(MODEL / "New416_SAEHD_inter.pth"), RES0, res, meth, str(pth))
            e2, i2, d2 = build_at(res, pth, device)
            rgb2 = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                              (res, res), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            out = forward(e2, i2, d2, rgb2, device)
            up = cv2.resize((out * 255).astype(np.uint8), (RES0, RES0),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            row = {"res": res, "method": meth, "psnr": psnr(base, up),
                   "ssim": ssim(base, up), "mae": float(np.abs(base - up).mean() * 255.0)}
            rows.append(row)
            print("res=%-4d %-9s PSNR=%.2f SSIM=%.4f MAE=%.2f"
                  % (res, meth, row["psnr"], row["ssim"], row["mae"]), flush=True)
            if meth in ("bilinear", "bicubic"):
                tiles.append(("%s %d" % (meth, res),
                              cv2.cvtColor((up * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

    # 网格用错（把 t 模型当 res/16 解析）—— 只做 320/bilinear 一例，量化后缀差异
    raw = ir.read_pth(MODEL / "New416_SAEHD_inter.pth")
    d1w, d1b, d2w, d2b = raw
    wrong = {"res": 320, "method": "bilinear_wronggrid", "psnr": None, "ssim": None, "mae": None}
    try:
        M = ir._grid(13, 10, "bilinear")            # 416/32=13 → 320/32=10，正确
        Mw = ir._grid(26, 20, "bilinear")            # 误当 res/16 用
        b1 = d1w.reshape(26, 26, 640, 320)
        t = np.tensordot(Mw, b1, axes=(1, 0))
        t = np.einsum("xy,yzca->xzca", Mw, t.transpose(1, 0, 2, 3))
        w1 = np.ascontiguousarray(t).reshape(-1, 320)
        p = OUT / "inter_320_wronggrid.pth"
        ir.write_pth(p, [w1.astype(np.float32), d1b, d2w, d2b])
        e2, i2, d2 = build_at(320, p, device)
        rgb2 = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (320, 320)).astype(np.float32) / 255.0
        out = forward(e2, i2, d2, rgb2, device)
        up = cv2.resize((out * 255).astype(np.uint8), (RES0, RES0)).astype(np.float32) / 255.0
        wrong.update({"psnr": psnr(base, up), "ssim": ssim(base, up),
                      "mae": float(np.abs(base - up).mean() * 255.0)})
        print("WRONGGRID PSNR=%.2f SSIM=%.4f" % (wrong["psnr"], wrong["ssim"]), flush=True)
    except Exception as e:                                     # noqa: BLE001
        print("wronggrid 失败：", e, flush=True)

    (OUT / "results.json").write_text(json.dumps({"rows": rows, "wronggrid": wrong}, indent=2),
                                     encoding="utf-8")

    # 拼接对比图（每格标注）
    h = w = RES0 // 2
    tiles = [(lab, cv2.resize(im, (w, h))) for lab, im in tiles]
    cols = 3
    rowsn = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rowsn * (h + 26), cols * w, 3), np.uint8)
    for i, (lab, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * (h + 26):r * (h + 26) + h, c * w:c * w + w] = im
        cv2.putText(grid, lab, (c * w + 6, r * (h + 26) + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "compare_grid.png"), grid)
    print("OUT:", OUT, flush=True)


if __name__ == "__main__":
    main()
