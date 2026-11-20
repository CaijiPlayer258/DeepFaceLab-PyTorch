# -*- coding: utf-8 -*-
"""inter 换分辨率实验：baseline → 各算法 → 与 baseline 比对。

用法：
  python app/tools/bench/inter_resolution_study.py --probe-norm
      用 decoder_dst 重建同一张脸，判断输入归一化是 [0,1] 还是 [-1,1]
  python app/tools/bench/inter_resolution_study.py --run --res 320,384,512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from app.tools.bench import inter_resample as ir          # noqa: E402
from app.tools.bench import saehd_torch as st            # noqa: E402

MODEL = REPO / "workspace" / "model"
TARGET = REPO / "app" / "docs" / "notes" / "target.jpg"
OUT = REPO / "app" / "docs" / "notes" / "inter_study"


def load_rgb(path, res):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("读不到 %s" % path)
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, img


def to_tensor(rgb, mode):
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    if mode == "pm1":
        x = x * 2.0 - 1.0
    return x


def from_tensor(t, mode):
    x = t.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def run_once(enc, inter, dec, rgb, mode="pm1", device="cpu"):
    x = to_tensor(rgb, mode).to(device)
    with torch.no_grad():
        z = inter(enc(x))
        out, mask = dec(z)
    return out, mask


def probe_norm(device):
    enc, inter, dec_src, _ = st.load_models(str(REPO))
    dec_dst = st.Decoder()
    st.assign(dec_dst, st.read_pth(MODEL / "New416_SAEHD_decoder_dst.pth"), "decoder_dst")
    dec_dst.eval()
    enc, inter, dec_dst = enc.to(device), inter.to(device), dec_dst.to(device)
    rgb, _ = load_rgb(TARGET, st.RES)
    for mode in ("01", "pm1"):
        out, _m = run_once(enc, inter, dec_dst, rgb, mode, device)
        rec = from_tensor(out, mode).astype(np.float32)
        tgt = (rgb * 255.0)
        mse = float(((rec - tgt) ** 2).mean())
        print("norm=%-4s decoder_dst 重建 MSE=%.1f  (越小说明归一化越对)" % (mode, mse), flush=True)
    return enc, inter, dec_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-norm", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if a.probe_norm:
        probe_norm(device)
    if a.run:
        pass


if __name__ == "__main__":
    main()
