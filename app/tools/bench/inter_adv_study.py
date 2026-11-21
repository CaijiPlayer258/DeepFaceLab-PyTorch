# -*- coding: utf-8 -*-
"""bicubic vs 伪逆 vs 数据驱动最小二乘：全部在 XSeg 遮罩内比较。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "app" / "engine"))

import numpy as np                                              # noqa: E402
import cv2                                                      # noqa: E402
import torch                                                    # noqa: E402

import app.tools.bench.inter_study_run as R                      # noqa: E402
import app.tools.bench.inter_adv as A
import app.tools.bench.inter_resample as ir                            # noqa: E402
from app.tools.bench.xseg_mask import mask_of, masked            # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
TMP = OUT / "cand"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0
RES_LIST = (320, 384, 448, 512)
N_FRAMES = 96


def video_path():
    vs = sorted((REPO / "workspace").glob("*.mp4"))
    return vs[0] if vs else None


def collect_features(enc, res_list):
    """从真实视频取帧，取 416 与各目标分辨率下的 encoder 特征。"""
    vp = video_path()
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(0, total - 2), N_FRAMES).astype(int) if total else range(30)
    dev = next(enc.parameters()).device
    F, Fp = [], {r: [] for r in res_list}
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = frame[y0:y0 + s, x0:x0 + s]

        def feat(res):
            im = cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1).to(dev)
            with torch.no_grad():
                return enc(x.unsqueeze(0))[0].detach().cpu().numpy().reshape(-1)

        F.append(feat(RES0))
        for r in res_list:
            Fp[r].append(feat(r))
        if k % 20 == 0:
            print("  feat %d/%d" % (k, len(idxs)), flush=True)
    cap.release()
    F = np.asarray(F, np.float32)
    return F, {r: np.asarray(v, np.float32) for r, v in Fp.items()}


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    rgb0, bgr0 = R.load_rgb(RES0)
    m0 = mask_of(bgr0)
    enc, inter, dec = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    dev = next(enc.parameters()).device
    x0 = torch.from_numpy(np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        z_old = inter(enc(x0))[0].detach().cpu().numpy()
    base = R.forward(enc, inter, dec, rgb0)
    cv2.imwrite(str(OUT / "base416.png"), (base * 255).astype(np.uint8))

    print("collect features…", flush=True)
    F, Fp = collect_features(enc, RES_LIST)
    print("features", F.shape, {r: v.shape for r, v in Fp.items()}, flush=True)
    np.savez_compressed(str(TMP / "features.npz"), F=F, **{("F%d" % r): v for r, v in Fp.items()})

    raw = R._read_pth(MODEL / (NAME + "_inter.pth"))
    rows = []
    for res in RES_LIST:
        m = res // 32
        cands = {}
        for meth in ("nearest", "bilinear", "bicubic", "area"):
            cands[meth] = A.resample_inter(raw, RES0, res, meth, mode="interp")
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            cands["pinv@%g" % lam] = A.resample_inter(raw, RES0, res, "bicubic",
                                                      mode="pinv", lam=lam)
        bic = A.resample_inter(raw, RES0, res, "bicubic", mode="interp")
        for lam in (1e-2, 1e-1, 1.0):
            w = A.ls_correct(bic[0], raw[0], F, Fp[res], lam)
            cands["ls@%g" % lam] = [w, bic[1], bic[2], bic[3]]

        rgb2, _ = R.load_rgb(res)
        for name, arrs in cands.items():
            p = TMP / ("%d_%s.pth" % (res, name.replace("@", "_")))
            ir.write_pth(p, arrs)
            e2, i2, d2 = R.build(res, p)
            o = R.forward(e2, i2, d2, rgb2)
            up = cv2.resize((o * 255).astype(np.uint8), (RES0, RES0),
                            interpolation=cv2.INTER_LINEAR)
            met = masked(base, up.astype(np.float32) / 255.0, m0)
            xr = torch.from_numpy(np.ascontiguousarray(rgb2)).permute(2, 0, 1).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                z2 = i2(e2(xr))[0].detach().cpu().numpy()
            zb = np.stack([cv2.resize(c, (13, 13), interpolation=cv2.INTER_LINEAR) for c in z2], 0)
            cos = float((zb.ravel() @ z_old.ravel()) /
                        (np.linalg.norm(zb) * np.linalg.norm(z_old) + 1e-12))
            rows.append(dict(res=res, method=name, cos=cos, **met))
            print("%-4d %-12s psnr=%5.2f ssim=%.4f mae=%5.2f cos=%.4f"
                  % (res, name, met["psnr"], met["ssim"], met["mae"], cos), flush=True)
            if name in ("bicubic", "pinv@0.01", "ls@0.1"):
                cv2.imwrite(str(TMP / ("%d_%s.png" % (res, name.replace("@", "_")))), up)

    (OUT / "results_adv.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
