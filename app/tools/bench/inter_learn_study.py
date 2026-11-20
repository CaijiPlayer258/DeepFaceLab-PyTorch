# -*- coding: utf-8 -*-
"""给 inter 换分辨率找一个「更好的线性映射」：直接在端到端损失上优化 dense1。

动机：前三步（频响 → 保泛函伪逆 → 数据驱动修正）都是在**潜空间**上做代理目标。
但下游是固定的 decoder，真正的目标函数是**输出图像**的差。既然 encoder/decoder 与分辨率无关、
且 dense1 只是一个线性层，那就可以把 dense1 当成可训练参数，用反向传播把这个线性映射学出来：

    W' = argmin  Σ_i  L_mask( Dec( W'ᵀ f'_i ) , Dec( Wᵀ f_i ) )
    初始化 W' = pinv（伪逆解），只学一个残差 Δ

用法：
  python app/tools/bench/inter_learn_study.py --res 448 --steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app" / "engine"))

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402
import torch                                                    # noqa: E402

import app.tools.bench.inter_study_run as R                      # noqa: E402
import app.tools.bench.inter_adv as A                            # noqa: E402
import app.tools.bench.inter_resample as ir                      # noqa: E402
from app.tools.bench.xseg_mask import mask_of, masked            # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
TMP = OUT / "cand"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0


def frames(n, res=RES0):
    vp = sorted((REPO / "workspace").glob("*.mp4"))[0]
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(0, total - 2), n).astype(int)
    out = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = min(h, w)
        c = f[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        out.append(cv2.resize(c, (res, res), interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def to_t(bgr, dev):
    return torch.from_numpy(np.ascontiguousarray(bgr)).permute(2, 0, 1).float().to(dev)[None] / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    enc, inter, dec = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    # 训练集输出（416 原模型），frozen
    train = frames(a.n)
    tgts = []
    with torch.no_grad():
        for im in train:
            tgts.append(dec(inter(enc(to_t(im, dev))))[0].detach())
    print("targets ready", len(tgts), flush=True)

    # 目标分辨率的模型：inter 用 pinv 初始化，只有 dense1 可训练
    base = A.resample_inter(raw, RES0, a.res, "bicubic", mode="pinv", lam=a.lam)
    p_init = TMP / ("learn_init_%d.pth" % a.res)
    TMP.mkdir(parents=True, exist_ok=True)
    ir.write_pth(p_init, base)
    e2, i2, d2 = R.build(a.res, p_init)
    w1 = i2.dense1.weight
    w1.requires_grad_(True)
    print("param", tuple(w1.shape), "requires_grad", w1.requires_grad, flush=True)

    opt = torch.optim.Adam([w1], lr=a.lr)
    low = a.res // 32
    feats = []
    with torch.no_grad():
        for im in train:
            imr = cv2.resize(im, (a.res, a.res), interpolation=cv2.INTER_AREA)
            feats.append(e2(to_t(imr, dev)).detach())
    opt.zero_grad()
    t0 = time.time()
    for step in range(a.steps):
        idx = np.random.randint(0, len(feats), a.batch)
        loss = 0.0
        for i in idx:
            out = d2(i2(feats[i]))[0]
            tgt = tgts[i]
            if out.shape[-1] != tgt.shape[-1]:
                out = torch.nn.functional.interpolate(out, size=tgt.shape[-2:], mode="bilinear",
                                                      align_corners=False)
            loss = loss + torch.nn.functional.l1_loss(out, tgt) + 0.5 * torch.nn.functional.mse_loss(out, tgt)
        loss = loss / len(idx)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 20 == 0:
            print("step %4d  l1=%.5f  %.1fs" % (step, float(loss), time.time() - t0), flush=True)
        if time.time() - t0 > 600:
            print("time budget reached", flush=True)
            break

    learned = [w1.detach().cpu().numpy().T.astype(np.float32), base[1], base[2], base[3]]
    p_out = TMP / ("learned_%d.pth" % a.res)
    ir.write_pth(p_out, learned)

    # 评估 target.jpg（未参与训练）
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    e3, i3, d3 = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    baseline = R.forward(e3, i3, d3, rgb0)
    cv2.imwrite(str(OUT / "base416.png"), (baseline * 255).astype(np.uint8))
    rgb2, _ = R.load_rgb(a.res)
    e4, i4, d4 = R.build(a.res, p_out)
    out = R.forward(e4, i4, d4, rgb2)
    up = cv2.resize((out * 255).astype(np.uint8), (RES0, RES0), interpolation=cv2.INTER_LINEAR)
    met = masked(baseline, up.astype(np.float32) / 255.0, m0)
    bic = TMP / ("ref_bic_%d.pth" % a.res)
    ir.write_pth(bic, A.resample_inter(raw, RES0, a.res, "bicubic", mode="interp"))
    e5, i5, d5 = R.build(a.res, bic)
    out_b = R.forward(e5, i5, d5, rgb2)
    up_b = cv2.resize((out_b * 255).astype(np.uint8), (RES0, RES0), interpolation=cv2.INTER_LINEAR)
    met_b = masked(baseline, up_b.astype(np.float32) / 255.0, m0)

    res = dict(res=a.res, steps=a.steps, learned=met, bicubic=met_b)
    print("LEARNED ", met, flush=True)
    print("BICUBIC ", met_b, flush=True)
    cv2.imwrite(str(TMP / ("learned_%d.png" % a.res)), up)
    cv2.imwrite(str(TMP / ("bicubic_ref_%d.png" % a.res)), up_b)
    (OUT / ("learn_%d.json" % a.res)).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
