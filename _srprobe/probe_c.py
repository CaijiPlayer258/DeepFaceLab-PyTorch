
# -*- coding: utf-8 -*-
"""Probe C: 同一个 GPEN、同一个 alpha，两条路的输入尺寸/内容不同 → 观感差多少（实测）。"""
import os, sys, glob, importlib.util
import numpy as np, cv2

os.environ["DFL_SR"] = "100"          # 滑块 100 → size()=512, strength()=1.0
os.environ["DFL_SR_BACKEND"] = "onnx"
ROOT = r"D:\MyProject\DeepFaceLab-PyTorch-4.1"
os.environ["DFL_SR_MODEL"] = os.path.join(ROOT, "weights", "gpen_bfr_512.onnx")
sys.path.insert(0, os.path.join(ROOT, "app"))
# 和 preview_worker 一样先把 torch/lib 塞进 DLL 搜索路径，CUDA EP 才加载得起来
for _d in (os.path.join(ROOT, "runtime", "unified", "Lib", "site-packages", "torch", "lib"),):
    if os.path.isdir(_d):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_d)
        except Exception:
            pass
from server.backend_merge import facesr

print("[facesr] level=%d size=%d strength=%.2f" % (facesr.level(), facesr.size(), facesr.strength()))

ALIGN = importlib.util.spec_from_file_location(
    "dflalign", os.path.join(ROOT, "app", "server", "_vision", "align.py"))
AL = importlib.util.module_from_spec(ALIGN); ALIGN.loader.exec_module(AL)

OUT = os.path.join(ROOT, "_srprobe")
os.makedirs(OUT, exist_ok=True)

# ---- 一张真实的 whole_face 512 对齐脸（= 六宫格里 merged/dfm-pred 的那种图）----
A = cv2.imread(sorted(glob.glob(os.path.join(ROOT, "workspace", "data_dst", "aligned", "*.jpg")))[3])
print("[input] aligned face:", A.shape)

# =============== 路 D：调试六宫格（compose.py L2038-2051 + merger.build_debug_panels）======
panel = cv2.resize(A, (256, 256), interpolation=cv2.INTER_AREA)      # _face_img: INTER_AREA
ph, pw = panel.shape[:2]
mD = np.zeros((ph, pw), np.float32)
cv2.ellipse(mD, (pw//2, ph//2), (max(1, pw//2-2), max(1, ph//2-2)), 0, 0, 360, 1.0, -1)
sigD = max(1.0, min(pw, ph) * 0.02)
mD = cv2.GaussianBlur(mD, (0, 0), sigD)
print("\n[D 调试] 输入 %dx%d  椭圆遮罩 sigma=%.2f  → apply_roi 内部再裁到 mask bbox"
      % (pw, ph, sigD))
ys, xs = np.nonzero(mD > 0.02)
cropD = panel[int(ys.min()):int(ys.max())+1, int(xs.min()):int(xs.max())+1]
print("[D 调试] apply_roi 里的 crop = %dx%d → resize 到 (%d,%d)：横 %.2fx 纵 %.2fx（等比 %s）"
      % (cropD.shape[1], cropD.shape[0], facesr.size(), facesr.size(),
         facesr.size()/cropD.shape[1], facesr.size()/cropD.shape[0],
         abs(facesr.size()/cropD.shape[1] - facesr.size()/cropD.shape[0]) < 0.02))
outD = facesr.apply_roi(panel.copy(), mD, facesr.strength())

# =============== 路 P：预览（compose.py L1736-1790，帧空间 box）================
# 帧空间检测框在 512 对齐图里对应的那块：68 点 bbox（+8%）
unit2chip = AL._affine_from_unit_square(np.array([[1., 0., 0.], [0., 1., 0.]]), 512)
chip68 = (unit2chip @ np.hstack([AL.UNIT_68, np.ones((68, 1))]).T).T[:, :2]
x1, y1 = chip68[:, 0].min(), chip68[:, 1].min()
x2, y2 = chip68[:, 0].max(), chip68[:, 1].max()
padx, pady = (x2-x1)*0.08, (y2-y1)*0.08
bx1, by1 = int(max(0, x1-padx)), int(max(0, y1-pady))
bx2, by2 = int(min(511, x2+padx)), int(min(511, y2+pady))
roi = A[by1:by2, bx1:bx2]
rh, rw = roi.shape[:2]
print("\n[P 预览] 帧空间 box ROI = %dx%d（就是画面里检测框那块；68 点 bbox+8%%）" % (rw, rh))
mP = np.zeros((rh, rw), np.float32)
cv2.ellipse(mP, (rw//2, rh//2), (max(1, rw//2-2), max(1, rh//2-2)), 0, 0, 360, 1.0, -1)
mP = cv2.GaussianBlur(mP, (0, 0), max(1.0, min(rw, rh)*0.02))
print("[P 预览] apply_roi 内部：crop %dx%d → resize (%d,%d)：横 %.2fx 纵 %.2fx（等比 %s）"
      % (rw, rh, facesr.size(), facesr.size(), facesr.size()/rw, facesr.size()/rh,
         abs(facesr.size()/rw - facesr.size()/rh) < 0.02))
outP = facesr.apply_roi(roi.copy(), mP, facesr.strength())

# 预览之后还有两道：整帧缩到 maxw=900（1920 宽的帧 → 0.469）+ JPEG q88
FRAME_W, MAXW = 1920, 900
s = min(1.0, MAXW / float(FRAME_W))
outP_show = cv2.resize(outP, (max(1, int(rw*s)), max(1, int(rh*s))), interpolation=cv2.INTER_AREA)
ok, enc = cv2.imencode(".jpg", outP_show, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
outP_show_jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else outP_show
print("[P 预览] SR 之后还被缩了 %.3f（maxw 900 / 帧宽 1920）+ JPEG q88：SR 那块 %dx%d → %dx%d"
      % (s, rw, rh, outP_show.shape[1], outP_show.shape[0]))

# =============== 统一到"用户实际看到的尺寸"再比锐度 ======================
def to_disp(img, face_frac_from, face_frac_to):
    return cv2.resize(img, None, fx=face_frac_to/face_frac_from, fy=face_frac_to/face_frac_from,
                      interpolation=cv2.INTER_AREA)

def grey(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)

def sharp(x):
    g = grey(x)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    return float(lap.var()), float(np.sqrt((gx**2 + gy**2).mean()))

# 用户看到的脸宽：预览面板 900 宽、脸占画面约 400/1920 → ~187 px；调试格 256 px
FACE_D_PX = 256.0
FACE_P_PX = 187.0
D0 = to_disp(panel, 256.0, FACE_D_PX)           # 无 SR 的调试格（面板基准）
D1 = to_disp(outD, 256.0, FACE_D_PX)
P0 = cv2.resize(roi, (int(round(rw*s)), int(round(rh*s))), interpolation=cv2.INTER_AREA)  # 无 SR 的预览
P1 = outP_show_jpg
print("\n== 送到 GPEN 的输入 ==")
print("  调试: %d² 对齐脸 panel（整脸居中，padding 0.40 的 whole_face 画布）→ crop %d²  等比"
      % (256, cropD.shape[0]))
print("  预览: %dx%d 画面检测框（含头发/背景四角、横竖拉伸 %.2fx/%.2fx）" % (rw, rh, 512.0/rw, 512.0/rh))
print("\n== 用户实际看到的锐度（同一张脸，同一 alpha=1.0）==")
for tag, base, sr in (("调试六宫格", D0, D1), ("预览", P0, P1)):
    lv0, gr0 = sharp(base); lv1, gr1 = sharp(sr)
    print("  %-10s 无SR  拉普拉斯方差 %7.1f  梯度RMS %5.2f" % (tag, lv0, gr0))
    print("  %-10s 有SR  拉普拉斯方差 %7.1f  梯度RMS %5.2f   → +%.0f%% / +%.0f%%"
          % ("", lv1, gr1, (lv1/max(lv0,1e-6)-1)*100, (gr1/max(gr0,1e-6)-1)*100))

# 把两条路的 SR 结果放大到同一尺寸，肉眼看风格差
bigD = cv2.resize(outD, (512, 512), interpolation=cv2.INTER_NEAREST)
bigP = cv2.resize(outP, (512, 512), interpolation=cv2.INTER_NEAREST)
cmp = np.hstack([A, cv2.resize(panel, (512,512), interpolation=cv2.INTER_NEAREST), bigD, bigP])
cv2.imwrite(os.path.join(OUT, "c1_compare_A_panel_D_P.png"), cmp)
cv2.imwrite(os.path.join(OUT, "c2_outD_debug.png"), outD)
cv2.imwrite(os.path.join(OUT, "c3_outP_preview.png"), outP)
cv2.imwrite(os.path.join(OUT, "c4_outP_display.png"), outP_show_jpg)
np.save(os.path.join(OUT, "c_metrics.npy"), np.array([rw, rh, s]))
print("\nsaved c1..c4 to", OUT)