
# -*- coding: utf-8 -*-
"""Probe B: 照 compose.py L1736-1790 复现预览那条路的 alpha，和真值(帧坐标遮罩)比。"""
import os, sys, json, importlib.util
import numpy as np, cv2
OUT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\_srprobe"
ALIGN_PY = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\server\_vision\align.py"
spec = importlib.util.spec_from_file_location("dflalign", ALIGN_PY)
AL = importlib.util.module_from_spec(spec); spec.loader.exec_module(AL)

meta = json.load(open(os.path.join(OUT, "a_meta.json")))
H, W = meta["H"], meta["W"]
frame = cv2.imread(os.path.join(OUT, "a1_frame.png"))
truth8 = np.load(os.path.join(OUT, "a_fmask_truth.npy"))
truth = truth8.astype(np.float32) / 255.0
lm = np.load(os.path.join(OUT, "a_lm.npy"))
b1, b2 = lm[:, 0].min(), lm[:, 0].max()
c1, c2 = lm[:, 1].min(), lm[:, 1].max()
px, py = (b2 - b1) * 0.08, (c2 - c1) * 0.08
bx1, by1, bx2, by2 = b1 - px, c1 - py, b2 + px, c2 + py
print("detect box = (%.0f, %.0f) - (%.0f, %.0f)  %.0fx%.0f" % (bx1, by1, bx2, by2, bx2-bx1, by2-by1))

# ================= ① worker: 帧坐标遮罩 → 256² (preview_worker.py L421-432) =========
mk256 = cv2.resize(truth8, (256, 256), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(OUT, "b1_worker_256.png"), mk256)
print("[worker L425] 帧坐标 (%d,%d) → 256x256 等比? %s  (纵 %.3f, 横 %.3f)"
      % (truth8.shape[0], truth8.shape[1],
         truth8.shape[0] / truth8.shape[1] == 1.0, 256.0/H, 256.0/W))

# ================= ② compose: L1758-1788 原样复现 ============================
_x, _y = int(round(bx1)), int(round(by1))
_bw, _bh = int(round(bx2 - bx1)), int(round(by2 - by1))
_x = max(0, min(_x, W - _bw)); _y = max(0, min(_y, H - _bh))
_mat = AL.align_fit(np.asarray(lm, np.float64), 512)[0]
_mk512 = cv2.resize(mk256.astype(np.float32)/255.0, (512, 512), interpolation=cv2.INTER_LINEAR)
_inv = cv2.invertAffineTransform(np.asarray(_mat, np.float64))
_mf = cv2.warpAffine(_mk512, _inv, (W, H), flags=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
preview_m = np.clip(_mf[_y:_y+_bh, _x:_x+_bw], 0.0, 1.0)
print("[compose L1778] 折回画面后 box 内 alpha: max=%.3f  >0.5 占比 %.4f  mean %.3f"
      % (float(preview_m.max()), float((preview_m>0.5).mean()), float(preview_m.mean())))
_sig = max(1.0, min(_bw, _bh) * 0.02)
preview_m = cv2.GaussianBlur(preview_m, (0, 0), _sig)
print("[compose L1788] GaussianBlur sigma=%.1f px (在 %dx%d 上)" % (_sig, _bw, _bh))

# 全帧 alpha（box 外恒 0 —— L1789 只把 box 那块交给 apply_roi）
alpha_prev = np.zeros((H, W), np.float32)
alpha_prev[_y:_y+_bh, _x:_x+_bw] = preview_m
np.save(os.path.join(OUT, "b_alpha_preview.npy"), alpha_prev)
cv2.imwrite(os.path.join(OUT, "b2_alpha_preview.png"), (alpha_prev*255).astype(np.uint8))

# ================= ③ 应得的 alpha = 真遮罩（同一张脸，同一坐标系） ==============
alpha_true = truth.copy()

# ================= ④ 差多少 =================================================
t_hi = truth > 0.9
def rep(name, a):
    iou = float(((a>0.5)&(truth>0.5)).sum()) / max(1, int(((a>0.5)|(truth>0.5)).sum()))
    inside_lost = float((a[t_hi] < 0.1).mean())     # 脸内本该 SR 却完全没做
    outside_gain = float((a[~(truth>0.02)] > 0.5).mean())  # 脸外居然做了 SR
    print("%-18s IoU=%5.3f  脸内漏做=%5.1f%%  脸外误做=%5.1f%%  mean=%.3f"
          % (name, iou, inside_lost*100, outside_gain*100, float(a.mean())))
rep("预览(现状)", alpha_prev)
rep("应得(真遮罩)", alpha_true)
# 调试那条路的椭圆（在 416 对齐画布上，等价于帧里以脸中心为心的椭圆）
ell = np.zeros((H, W), np.float32)
cv2.ellipse(ell, (int((bx1+bx2)/2), int((by1+by2)/2)), (int(_bw/2)-2, int(_bh/2)-2), 0, 0, 360, 1.0, -1)
ell = cv2.GaussianBlur(ell, (0,0), max(1.0, min(_bw,_bh)*0.02))
rep("调试(椭圆)", ell)

# ================= ⑤ 硬边界检测：alpha 的台阶在多长的一条线上 =================
gx = np.abs(np.diff(alpha_prev, axis=1))
gy = np.abs(np.diff(alpha_prev, axis=0))
step = np.maximum(gx[:-1, :], gy[:, :-1])
hard = step > 0.6          # 2px 内 1→0/0→1
n_hard = int(hard.sum())
ys, xs = np.nonzero(hard)
print("\n[硬边界] |Δα|>0.6/px 的像素数 = %d" % n_hard)
if n_hard:
    print("         落在 x∈[%d,%d] y∈[%d,%d]（%d 条水平边 / %d 条竖直边）"
          % (xs.min(), xs.max(), ys.min(), ys.max(),
             int((np.abs((ys)%H)).size and n_hard), n_hard))
    hx = np.bincount(xs, minlength=W); hy = np.bincount(ys, minlength=H)
    print("         最密集的 3 条竖直边 x =", np.argsort(hx)[-3:][::-1].tolist(),
          " 最密集的 3 条水平边 y =", np.argsort(hy)[-3:][::-1].tolist())
    print("         box 边界 x=[%d, %d] y=[%d, %d]" % (_x, _x+_bw, _y, _y+_bh))

# 参考：真遮罩的硬边界数量
gg = np.maximum(np.abs(np.diff(alpha_true, axis=1))[:-1, :], np.abs(np.diff(alpha_true, axis=0))[:, :-1])
print("[硬边界] 真遮罩 |Δα|>0.6 的像素数 =", int((gg > 0.6).sum()))

# ================= ⑥ 可视化 =================================================
def overlay(base, a, tag):
    img = base.copy().astype(np.float32)
    sharp = cv2.detailEnhance(base, sigma_s=20, sigma_r=0.25)
    m = a[..., None]
    out = img*(1-m) + sharp.astype(np.float32)*m
    vis = np.clip(out, 0, 255).astype(np.uint8)
    cv2.rectangle(vis, (_x, _y), (_x+_bw, _y+_bh), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, tag), vis)
    return vis
overlay(frame, alpha_prev, "b3_sim_preview.png")
overlay(frame, alpha_true, "b4_sim_true.png")
print("\nsaved", OUT)
