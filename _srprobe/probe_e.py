
# -*- coding: utf-8 -*-
"""Probe E: 如果 worker 真的回传"对齐脸遮罩"，compose 现在这套 warp 能对吗？
   对比：按 compose 现在硬编码的 512 vs 按 dfm 真实 input_size(416)。"""
import os, sys, json, importlib.util
import numpy as np, cv2
ROOT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1"
ENG = os.path.join(ROOT, "app", "engine"); sys.path.insert(0, ENG)
from MergeStudio.core.merger import MergeMaskedFace
from MergeStudio.core.config import MergerConfigMasked
ALIGN = importlib.util.spec_from_file_location("dflalign", os.path.join(ROOT, "app", "server", "_vision", "align.py"))
AL = importlib.util.module_from_spec(ALIGN); ALIGN.loader.exec_module(AL)
OUT = os.path.join(ROOT, "_srprobe")
meta = json.load(open(os.path.join(OUT, "a_meta.json")))
H, W, IS = meta["H"], meta["W"], meta["input_size"]
frame = cv2.imread(os.path.join(OUT, "a1_frame.png"))
truth = (np.load(os.path.join(OUT, "a_fmask_truth.npy")) > 127).astype(np.float32)
lm = np.load(os.path.join(OUT, "a_lm.npy"))

cfg = MergerConfigMasked(face_type="whole_face", mode="overlay", mask_mode=6)
cap = {}
def _p(x):
    h, w = x.shape[:2]
    m = np.zeros((h, w), np.float32); cv2.ellipse(m, (w//2, h//2), (w//2-20, h//2-20), 0, 0, 360, 1.0, -1)
    return x.copy(), m, m
def _xs(x):
    h, w = x.shape[:2]
    m = np.zeros((h, w), np.float32); cv2.ellipse(m, (w//2, h//2), (w//2-8, h//2-8), 0, 0, 360, 1.0, -1)
    return m
MergeMaskedFace(frame.astype(np.float32)/255.0, lm, cfg, _p, xseg_256_extract_func=_xs,
                capture=cap, input_size=IS)
eff = np.clip(np.asarray(cap["effective_mask"], np.float32).squeeze(), 0, 1)
print("capture['effective_mask'] shape =", eff.shape, "（= dfm input_size，不是 512）")

x1, y1 = lm[:,0].min(), lm[:,1].min(); x2, y2 = lm[:,0].max(), lm[:,1].max()
px, py = (x2-x1)*.08, (y2-y1)*.08
bx, by = int(x1-px), int(y1-py); bw, bh = int(x2-x1+2*px), int(y2-y1+2*py)
bx = max(0, min(bx, W-bw)); by = max(0, min(by, H-bh))

def iou(a, b):
    a = a > 0.5; b = b > 0.5
    return float((a & b).sum()) / max(1, int((a | b).sum()))

for tag, out_size in (("compose 现状：硬编码 512", 512), ("按 per_face.size=416", IS)):
    mat = AL.align_fit(np.asarray(lm, np.float64), out_size)[0]
    mk = cv2.resize(eff, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    inv = cv2.invertAffineTransform(np.asarray(mat, np.float64))
    mf = cv2.warpAffine(mk, inv, (W, H), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = np.clip(mf[by:by+bh, bx:bx+bw], 0, 1)
    a = np.zeros((H, W), np.float32); a[by:by+bh, bx:bx+bw] = m
    hard = np.maximum(np.abs(np.diff(a, axis=1))[:-1, :], np.abs(np.diff(a, axis=0))[:, :-1])
    print("  %-26s 对齐遮罩 → warp 回画面：IoU(真值) = %.3f   |Δα|>0.6 像素 %d  mean %.3f"
          % (tag, iou(a, truth), int((hard > 0.6).sum()), float(a.mean())))
