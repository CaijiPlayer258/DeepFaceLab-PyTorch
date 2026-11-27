
# -*- coding: utf-8 -*-
"""Probe D: 在"用户看到的同一个人脸尺寸"上比两条路的锐度/风格；并量化预览的两次额外重采样。"""
import os, sys, glob, importlib.util
import numpy as np, cv2

os.environ["DFL_SR"] = "100"
os.environ["DFL_SR_BACKEND"] = "onnx"
ROOT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1"
os.environ["DFL_SR_MODEL"] = os.path.join(ROOT, "weights", "gpen_bfr_512.onnx")
for _d in (os.path.join(ROOT, "runtime", "unified", "Lib", "site-packages", "torch", "lib"),):
    if os.path.isdir(_d):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
        try: os.add_dll_directory(_d)
        except Exception: pass
sys.path.insert(0, os.path.join(ROOT, "app"))
from server.backend_merge import facesr
ALIGN = importlib.util.spec_from_file_location("dflalign", os.path.join(ROOT, "app", "server", "_vision", "align.py"))
AL = importlib.util.module_from_spec(ALIGN); ALIGN.loader.exec_module(AL)
OUT = os.path.join(ROOT, "_srprobe")

A = cv2.imread(sorted(glob.glob(os.path.join(ROOT, "workspace", "data_dst", "aligned", "*.jpg")))[3])

def ellipse_m(h, w, inset=2, sig_frac=0.02):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w//2, h//2), (max(1, w//2-inset), max(1, h//2-inset)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), max(1.0, min(w, h)*sig_frac))

def g(x): return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
def lapvar(x): return float(cv2.Laplacian(g(x), cv2.CV_32F, ksize=3).var())
def grms(x):
    q = g(x)
    gx = cv2.Sobel(q, cv2.CV_32F, 1, 0, ksize=3) * .125
    gy = cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3) * .125
    return float(np.sqrt((gx**2 + gy**2).mean()))
def rs(img, s): return cv2.resize(img, (max(1, int(round(img.shape[1]*s))), max(1, int(round(img.shape[0]*s)))), interpolation=cv2.INTER_AREA)
def jpg(img, q=88):
    ok, b = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR) if ok else img

# ---------- 路 D：调试格 ----------
panel = cv2.resize(A, (256, 256), interpolation=cv2.INTER_AREA)
mD = ellipse_m(256, 256)
outD = facesr.apply_roi(panel.copy(), mD, 1.0)

# ---------- 路 P：预览（帧空间 box） ----------
u2c = AL._affine_from_unit_square(np.array([[1.,0.,0.],[0.,1.,0.]]), 512)
c68 = (u2c @ np.hstack([AL.UNIT_68, np.ones((68,1))]).T).T[:, :2]
x1,y1 = c68[:,0].min(), c68[:,1].min(); x2,y2 = c68[:,0].max(), c68[:,1].max()
px,py = (x2-x1)*.08, (y2-y1)*.08
roi = A[int(max(0,y1-py)):int(min(511,y2+py)), int(max(0,x1-px)):int(min(511,x2+px))]
rh, rw = roi.shape[:2]
mP = ellipse_m(rh, rw)
outP = facesr.apply_roi(roi.copy(), mP, 1.0)

# ---------- 归一到"同一个脸宽"再看观感 ----------
# whole_face 512 画布里，68 点的脸宽就是 (x2-x1)；所以脸宽占比可知
face_frac_align = (x2-x1)/512.0
FACE_DISPLAY = 160.0            # 两条路都放到 160px 脸宽再比
sD = FACE_DISPLAY/(face_frac_align*256.0)
sP = FACE_DISPLAY/((x2-x1)/float(rw)*rw)     # = FACE_DISPLAY/(x2-x1) 针对帧里那块的尺寸
sP2 = FACE_DISPLAY/((x2-x1)*1.0)             # 帧空间 ROI 里脸宽 ≈ (x2-x1)
D_base, D_sr = rs(panel, sD), rs(outD, sD)
P_base, P_sr = rs(roi, sP2), rs(outP, sP2)
P_base_j, P_sr_j = jpg(P_base), jpg(P_sr)
# 预览实际还多缩了一道 maxw（1920→900 = 0.469 相对帧），这里把 ROI 那一步也加上
P_base_show = jpg(rs(rs(roi, 0.469), FACE_DISPLAY/((x2-x1)*0.469)))
P_sr_show   = jpg(rs(rs(outP, 0.469), FACE_DISPLAY/((x2-x1)*0.469)))

def row(tag, b, s):
    print("  %-26s 无SR: lapVar %7.0f  gradRMS %5.2f   →  有SR: lapVar %7.0f  gradRMS %5.2f  (SR 增益 %+5.1f%% / %+5.1f%%)"
          % (tag, lapvar(b), grms(b), lapvar(s), grms(s),
             (lapvar(s)/lapvar(b)-1)*100, (grms(s)/grms(b)-1)*100))

print("\n=== 同一张脸、同一 GPEN-512、同一 alpha=1.00、同一显示脸宽 %.0fpx ===" % FACE_DISPLAY)
row("D 调试格 256²→显示", D_base, D_sr)
row("P 预览 帧空间→显示", P_base, P_sr)
row("P 预览 含 maxw 缩放", P_base_show, P_sr_show)

# SR 的"风格"差异：两张 SR 图互相比 + 与各自 baseline 的差
_n = (min(D_sr.shape[0], P_sr.shape[0]), min(D_sr.shape[1], P_sr.shape[1]))
def _fit(x): return cv2.resize(x, (_n[1], _n[0]), interpolation=cv2.INTER_AREA)
d1 = np.abs(_fit(D_sr).astype(np.float32) - _fit(P_sr).astype(np.float32)).mean()
d0 = np.abs(_fit(D_base).astype(np.float32) - _fit(P_base).astype(np.float32)).mean()
print("\n  两张 baseline 平均像素差 %.2f / 两张 SR 平均像素差 %.2f" % (d0, d1))
# 皮肤平滑签名：平坦区(局部标准差小)的梯度能量变化 = GPEN 的"磨皮"程度
def plastic(b, s):
    gb, gs = g(b), g(s)
    m = np.abs(cv2.Laplacian(gb, cv2.CV_32F, ksize=3)) < 6.0     # 平坦区掩膜
    if m.sum() < 200: return float('nan')
    lb = np.abs(cv2.Laplacian(gb, cv2.CV_32F, ksize=3))[m].mean()
    ls = np.abs(cv2.Laplacian(gs, cv2.CV_32F, ksize=3))[m].mean()
    return float(ls / max(lb, 1e-6) - 1.0) * 100
print("  皮肤平坦区梯度变化(磨皮/加纹): 调试 %+.1f%%   预览 %+.1f%%" % (plastic(D_base, D_sr), plastic(P_base, P_sr)))

# GPEN 的"改图量"（SR-baseline 残差能量 = 风格签名）
rD = np.abs(D_sr.astype(np.float32) - D_base.astype(np.float32))
rP = np.abs(P_sr.astype(np.float32) - P_base.astype(np.float32))
print("  GPEN 改动量: 调试 mean|Δ| %5.2f (p99 %.0f)   预览 mean|Δ| %5.2f (p99 %.0f)"
      % (rD.mean(), np.percentile(rD, 99), rP.mean(), np.percentile(rP, 99)))

# ---------- 各向异性：同一个 crop，等比 pad 成方形 vs 直接拉伸 ----------
side = max(rw, rh)
sq = np.zeros((side, side, 3), np.uint8)
ox, oy = (side-rw)//2, (side-rh)//2
sq[oy:oy+rh, ox:ox+rw] = roi
outSq = facesr.apply_roi(sq.copy(), ellipse_m(side, side), 1.0)[oy:oy+rh, ox:ox+rw]
aS = rs(outSq, sP2); P_b = rs(roi, sP2)
print("\n  同样的 crop：等比(方形 pad) vs 直接拉伸到 512")
print("    等比     lapVar %7.0f gradRMS %5.2f" % (lapvar(aS), grms(aS)))
print("    拉伸     lapVar %7.0f gradRMS %5.2f   (差 %.1f%% / %.1f%%)"
      % (lapvar(P_sr), grms(P_sr), (lapvar(P_sr)/lapvar(aS)-1)*100, (grms(P_sr)/grms(aS)-1)*100))

# 拼一张对比图
pad = np.full((256, 8, 3), 255, np.uint8)
def fit(img, n=256): return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
row1 = np.hstack([fit(A), pad, fit(panel), pad, fit(outD), pad, fit(D_sr)])
row2 = np.hstack([fit(roi), pad, fit(outP), pad, fit(P_sr), pad, fit(P_sr_show)])
cv2.imwrite(os.path.join(OUT, "d_compare.png"), np.vstack([row1, row2]))
cv2.imwrite(os.path.join(OUT, "d_outD.png"), outD)
cv2.imwrite(os.path.join(OUT, "d_outP.png"), outP)
print("\nsaved d_compare.png")