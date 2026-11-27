
# -*- coding: utf-8 -*-
"""Probe E2: 分清楚两件事 —— (1) warp 矩阵对不对（框内比） (2) 真遮罩有多少落在检测框外。"""
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
truthf = np.load(os.path.join(OUT, "a_fmask_truth.npy")).astype(np.float32) / 255.0
lm = np.load(os.path.join(OUT, "a_lm.npy"))
cfg = MergerConfigMasked(face_type="whole_face", mode="overlay", mask_mode=6)
cap = {}
def _p(x):
    h, w = x.shape[:2]; m = np.zeros((h,w), np.float32)
    cv2.ellipse(m, (w//2,h//2), (w//2-20,h//2-20), 0,0,360,1.0,-1); return x.copy(), m, m
def _xs(x):
    h, w = x.shape[:2]; m = np.zeros((h,w), np.float32)
    cv2.ellipse(m, (w//2,h//2), (w//2-8,h//2-8), 0,0,360,1.0,-1); return m
MergeMaskedFace(frame.astype(np.float32)/255.0, lm, cfg, _p, xseg_256_extract_func=_xs, capture=cap, input_size=IS)
eff = np.clip(np.asarray(cap["effective_mask"], np.float32).squeeze(), 0, 1)

x1,y1 = lm[:,0].min(), lm[:,1].min(); x2,y2 = lm[:,0].max(), lm[:,1].max()
px,py = (x2-x1)*.08, (y2-y1)*.08
bx,by = int(x1-px), int(y1-py); bw,bh = int(x2-x1+2*px), int(y2-y1+2*py)
bx = max(0,min(bx,W-bw)); by = max(0,min(by,H-bh))
print("检测框(landmark bbox+8%%) = %dx%d @ (%d,%d)" % (bw,bh,bx,by))
ys, xs = np.nonzero(truthf > 0.02)
print("真遮罩 bbox = %dx%d @ (%d,%d)  ← 比检测框大得多"
      % (xs.max()-xs.min()+1, ys.max()-ys.min()+1, xs.min(), ys.min()))
inside = truthf[by:by+bh, bx:bx+bw]
tot = float((truthf > 0.5).sum())
lost = float(((truthf > 0.5).sum()) - ((inside > 0.5).sum()))
print("真遮罩里落在检测框外的面积占比 = %.1f%%  → 现状这几块**永远不超分**（且边界=框的直线）"
      % (lost / max(tot, 1) * 100))

# 框内比：warp 矩阵到底对不对
mat = AL.align_fit(np.asarray(lm, np.float64), IS)[0]
mk = cv2.resize(eff, (IS, IS), interpolation=cv2.INTER_LINEAR)
inv = cv2.invertAffineTransform(np.asarray(mat, np.float64))
mf = cv2.warpAffine(mk, inv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
w = np.clip(mf[by:by+bh, bx:bx+bw], 0, 1)
def iou(a, b):
    a = a > 0.5; b = b > 0.5
    return float((a & b).sum()) / max(1, int((a | b).sum()))
print("框内 IoU(warp 回来的对齐遮罩 vs 真值) = %.3f  ← 矩阵若对得上应该≈1.0" % iou(w, inside))
print("框内 mean: warp %.3f  真值 %.3f" % (float(w.mean()), float(inside.mean())))
# 偏差方向
cy_w = np.argwhere(w > 0.5).mean(axis=0) if (w > 0.5).any() else None
cy_t = np.argwhere(inside > 0.5).mean(axis=0) if (inside > 0.5).any() else None
print("框内质心: warp %s  真值 %s  (dy,dx)=%s" % (
    np.round(cy_w,1) if cy_w is not None else None,
    np.round(cy_t,1) if cy_t is not None else None,
    np.round(cy_w-cy_t,1) if (cy_w is not None and cy_t is not None) else None))
