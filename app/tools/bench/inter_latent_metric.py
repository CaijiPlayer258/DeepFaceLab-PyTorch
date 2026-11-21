# -*- coding: utf-8 -*-
"""潜空间口径：inter 换分辨率后「映射」保留得怎么样（不训练也能比）。

做法：416 编码器出 13×13×640 → 空间缩到 low=res/32 得到新分辨率的特征 →
      过新 dense1/dense2 得到 z_new（320×low×low）→ 插值回 13×13 与 z_old 比。
这样把 decoder 的饱和放大排除掉，只比「inter 这一层有没有换对」。
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
import torch                                                    # noqa: E402

from app.tools.bench import inter_resample as ir                 # noqa: E402
import app.tools.bench.inter_study_run as R                      # noqa: E402

OUT = REPO / "app" / "docs" / "notes" / "inter_study"
MODEL, NAME, RES0 = R.MODEL, R.NAME, R.RES0
TARGETS = (320, 384, 448, 512)
METHODS = ("nearest", "bilinear", "bicubic", "area")


def inter_only(inter_path, res):
    """只建 inter（dense1/dense2），权重按文件顺序灌入。"""
    from core.leras.archis.DeepFakeArchi import DeepFakeArchi
    lnn = R._init_leras()
    archi = DeepFakeArchi(res, opts="udt")
    low = res // 32
    mod = archi.Inter(in_ch=80 * 8 * low * low, ae_ch=320, ae_out_ch=320, name="inter")
    arrs = R._read_pth(inter_path)
    ws = mod.get_weights()
    for t, a in zip(ws, arrs):
        v = a
        if tuple(v.shape) != tuple(t.shape):
            v = v.T
        with torch.no_grad():
            t.copy_(torch.from_numpy(np.ascontiguousarray(v)))
    mod.eval()
    return mod


def main():
    rgb0, _ = R.load_rgb(RES0)
    enc, inter, dec = R.build(RES0, MODEL / (NAME + "_inter.pth"))
    dev = next(enc.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(rgb0)).permute(2, 0, 1).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        e = enc(x)                       # (1, N)
        z_old = inter(e)                 # (1, 320, 13, 13)  可能经过 reshape_4D
    z_old = z_old[0].detach().cpu().numpy()
    if z_old.ndim == 3 and z_old.shape[0] == 320:
        pass
    else:                                # 若返回 (1,320,13,13) 已处理；否则兜底
        z_old = z_old.reshape(320, 13, 13)
    feat = e[0].detach().cpu().numpy().reshape(640, 13, 13)
    rows = []
    for res in TARGETS:
        low = res // 32
        f_new = np.stack([cv2.resize(ch, (low, low), interpolation=cv2.INTER_AREA)
                          for ch in feat], 0).reshape(-1)
        for meth in METHODS:
            p = OUT / ("inter_%d_%s.pth" % (res, meth))
            if not p.exists():
                ir.convert(str(MODEL / (NAME + "_inter.pth")), RES0, res, meth, str(p))
            mod = inter_only(p, res)
            with torch.no_grad():
                z_new = mod(torch.from_numpy(f_new).unsqueeze(0).float().to(next(mod.parameters()).device))
            z_new = (z_new[0] if z_new.ndim == 4 else z_new.reshape(320, low, low)).detach().cpu().numpy()
            back = np.stack([cv2.resize(ch, (13, 13), interpolation=cv2.INTER_LINEAR)
                             for ch in z_new], 0)
            diff = float(np.abs(back - z_old).mean())
            cos = float((back.ravel() @ z_old.ravel()) /
                        (np.linalg.norm(back) * np.linalg.norm(z_old) + 1e-12))
            rows.append(dict(res=res, method=meth, latent_mae=diff, latent_cos=cos))
            print("%-4d %-9s latentMAE=%.4f cos=%.4f" % (res, meth, diff, cos), flush=True)
    (OUT / "latent_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
