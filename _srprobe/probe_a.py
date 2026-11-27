
# -*- coding: utf-8 -*-
"""Probe A: 遮罩坐标系。MergeMaskedFace 回传的 _fmask 到底是哪个坐标系？"""
import os, sys, json
import numpy as np, cv2
ENG = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app\engine"
sys.path.insert(0, ENG)
from MergeStudio.core.merger import MergeMaskedFace, build_debug_panels
from MergeStudio.core.config import MergerConfigMasked
from facelib import LandmarksProcessor, FaceType

OUT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1\_srprobe"
os.makedirs(OUT, exist_ok=True)

H, W = 1080, 1920
frame = np.zeros((H, W, 3), np.uint8)
frame[:] = (40, 55, 70)
# 一张"脸"：椭圆 + 五官色块，位置 (600,300)-(1000,700)，400x400
cx, cy, R = 800, 500, 200
cv2.ellipse(frame, (cx, cy), (R-20, R-10), 0, 0, 360, (150, 165, 185), -1)
cv2.ellipse(frame, (cx-70, cy-40), (34, 18), 0, 0, 360, (60, 60, 70), -1)
cv2.ellipse(frame, (cx+70, cy-40), (34, 18), 0, 0, 360, (60, 60, 70), -1)
cv2.ellipse(frame, (cx, cy+40), (16, 40), 0, 0, 360, (120, 130, 150), -1)
cv2.ellipse(frame, (cx, cy+130), (60, 22), 0, 0, 360, (70, 80, 130), -1)
# 背景里加高频纹理，方便看出超分边界
rng = np.random.default_rng(0)
frame[..., 0] += rng.integers(0, 12, (H, W), dtype=np.uint8)

# ---- 合成一套 68 点（帧坐标）：先用 5 点式模板铺，再取 get_transform_mat 的逆把模板换到帧空间 ----
# 拿一个 draft：把 UNIT 模板按脸框摆
unit = []
# jaw 0..16, brow 17..26, nose 27..35, eyes 36..47, mouth 48..67
import math
for i in range(17):
    a = math.pi * (1.0 - i / 16.0)
    unit.append((0.5 + 0.36 * math.cos(a), 0.42 + 0.40 * math.sin(a)))
for i in range(5):
    unit.append((0.28 + 0.09 * i, 0.30))
for i in range(5):
    unit.append((0.64 + 0.09 * i, 0.30))
for i in range(9):
    unit.append((0.50 - 0.02 + 0.005 * i, 0.44 + 0.055 * i) if False else (0.50, 0.44 + 0.05 * i))
for i in range(6):
    t = math.pi * (0.0 - 1.0 + i / 5.0)
    unit.append((0.50 - 0.13 * abs(math.cos(t)) - 0.0, 0.44))
for i in range(6):
    unit.append((0.50 + 0.13 * abs(math.cos(math.pi * i / 5.0)), 0.44))
for i in range(20):
    a = 2 * math.pi * i / 20.0
    unit.append((0.50 + 0.16 * math.cos(a), 0.72 + 0.075 * math.sin(a)))
unit = np.array(unit[:68], np.float64)
assert unit.shape == (68, 2), unit.shape
# 单位 → 帧：脸框 400 见方放在 (600,300)
lm = np.stack([600 + unit[:, 0] * 400, 300 + unit[:, 1] * 400], axis=1).astype(np.float64)

cfg = MergerConfigMasked(face_type="whole_face", mode="overlay", mask_mode=6,
                         color_transfer_mode=1, erode_mask_modifier=0,
                         blur_mask_modifier=0, show_debug=False)

INPUT_SIZE = 416          # DF-UDT416 的 input_size
cap = {}
def _predictor(x):
    # 桩：返回原图 + 两张遮罩（warped dst 脸的 hull 用不到，给个椭圆）
    h, w = x.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 20, h // 2 - 20), 0, 0, 360, 1.0, -1)
    return x.copy(), m, m

def _xs(x):
    h, w = x.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 8, h // 2 - 8), 0, 0, 360, 1.0, -1)
    return m

fo, fmask = MergeMaskedFace(frame.astype(np.float32) / 255.0, lm, cfg, _predictor,
                            xseg_256_extract_func=_xs, capture=cap, input_size=INPUT_SIZE)
print(">>> MergeMaskedFace 返回: fo.shape=%s  _fmask.shape=%s dtype=%s" % (fo.shape, np.asarray(fmask).shape, np.asarray(fmask).dtype))
f = np.asarray(fmask, np.float32)
if f.ndim == 3:
    f = f[..., 0]
ys, xs = np.nonzero(f > 0.5)
print(">>> _fmask 帧空间 bbox: x %d..%d  y %d..%d  (帧 %dx%d)" % (xs.min(), xs.max(), ys.min(), ys.max(), W, H))
print(">>> _fmask 覆盖率 %.4f  峰值 %.3f" % (float((f > 0.02).mean()), float(f.max())))

# 关键证据：如果 _fmask 是"对齐脸坐标系"，它应该是 INPUT_SIZE 方阵；
# 如果是帧坐标，它就是 HxW 且 bbox 正好落在脸上。
print(">>> 是对齐脸坐标系吗（形状==(416,416)）？", np.asarray(fmask).shape[:2] == (INPUT_SIZE, INPUT_SIZE))
print(">>> capture 里的 effective_mask（真·对齐脸坐标系遮罩）形状 =", np.asarray(cap.get("effective_mask")).shape)

cv2.imwrite(os.path.join(OUT, "a1_frame.png"), frame)
cv2.imwrite(os.path.join(OUT, "a2_fmask_truth.png"), (np.clip(f, 0, 1) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(OUT, "a3_effective_mask_416.png"),
            (np.clip(np.asarray(cap["effective_mask"], np.float32).squeeze(), 0, 1) * 255).astype(np.uint8))
np.save(os.path.join(OUT, "a_fmask_truth.npy"), f)
np.save(os.path.join(OUT, "a_lm.npy"), lm)
json.dump({"H": H, "W": W, "input_size": INPUT_SIZE}, open(os.path.join(OUT, "a_meta.json"), "w"))
print("saved to", OUT)
