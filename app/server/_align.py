"""DFL 对齐内核：检测到的 5 点 → 68 点 → 512 对齐图 → DFLJPG。

4.1 原来只做「框中心裁正方形」，写出去的是裸 JPEG。4.0 的 SampleLoader 用
DFLIMG.has_data() 判断，没有 APP15 段的一律当「不是 DFL 图片」丢掉，所以那样
产出的对齐集在 4.0 里根本不能用。这里把两件事补齐：

1) 相似变换对齐 —— 用检测到的 5 点拟合到 68 点模板的对应 5 点，再按 DFL
   get_transform_mat() 的同一套 padding / 中心偏移算仿射矩阵，warp 到 512。
2) DFLJPG 写入 —— JPEG 里按 DFL 的格式插 APP15 段，内容是 pickle 的 dict
   （landmarks / source_landmarks / source_rect / image_to_face_mat / face_type）。

TEMPLATE_68 是经验模板：从 4.0 现有 data_src/aligned 里随机取 1200 张
whole_face 对齐脸的 landmarks 求均值得到，坐标就是 DFL 512 对齐空间里人脸
该在的位置，不是拍脑袋编的。UNIT_68 是同一份模板反推回「单位方形」空间的
结果（其中 33 个 DFL 点被钉成 facelib 的 landmarks_2D_new）—— 5 点路径用它，
68 点的**外点检查**也用它。padding 只算一次。

终裁矩阵本身（align_mat_from_68）现在**只拟合 33 点**，与 facelib 逐位等价，
见下面 2026-12 那段；68 点保留，但只当证人。

────────────────────────────────────────────────────────────────────────
2026-11 起：**5 点只做几何，68 点只从模型来**
────────────────────────────────────────────────────────────────────────
这里原来有个「5 点插值出 68 点」的函数（lm68_from_5，2026-11 **整个删掉**）——
它把检测到的 5 点按模板相似变换插值成 68 个点，看着像那么回事。删它的原因
是踩过一次真实的坑 ——

  提取器当时先用 5 点裁剪，再对**已裁好的图**跑 2D106 拿 68 点；106 一旦失败，
  代码就退回那个插值函数的结果，把它**当成真值写进 DFLJPG 的 dat**。
  那些数是"按模板摆出来的"，不是这张脸的：留出集上模板插值 10.6 px（512 空间），
  模型直出的点小一个量级，而且**永远不会自己说"我错了"**——更糟的是 dat 里看
  起来**一切正常**
  —— landmarks 齐全、形状 68×2、排序工具照样能算 yaw/pitch/roll，没有任何
  一处会报错。也就是说：假数据不会暴露自己，只会安静地训练坏、排序乱。

所以现在的规矩是硬性的，别再往回改：

  1) 5 点唯一的用途是**求一个粗对齐矩阵**（给 2D106 准备输入），绝不产出 68 点；
  2) 68 点只能来自 2D106 的真实输出（landmarks68_batch：106 点按固定下标挑 68 个，
     见 _landmarks106.landmark106to68），失败就**明确跳过**，
     并记进 skipped + 日志，不写文件、不写估算值；
  3) 元数据里带 landmarks_source：'model68'（来自模型）/ 'none'（没有点）。
     看到 'none' 就说明这张图没有可信关键点，别再拿它算掩码或排序。

────────────────────────────────────────────────────────────────────────
2026-12 起：**终裁矩阵与 iperov 的 get_transform_mat 逐位等价**
────────────────────────────────────────────────────────────────────────
以前 align_mat_from_68 拿**全部 68 点**去拟合 UNIT_68（33 点 facelib 模板 + 35 点
经验模板）。真实人脸不是那 35 个点的均值：下颌/下巴落在单位方形外面
（x∈[-0.12,1.05]、y∈[0.02,1.16]），最小二乘被它们拽着走。1203 张真实对齐脸实测，
与 facelib 的 WHOLE_FACE 矩阵相比：缩放中位 **+0.36%**、旋转中位 **+2.53°**
（p05..p95 −4.9°..+4.9°）、画布四角平均差 **34.2 px**（最大 149.6 px）。

现在只拟合 **33 点**（17..48 + 54）到 facelib 的 landmarks_2D_new —— 和
facelib.LandmarksProcessor.get_transform_mat 用的是同一份模板、同一串几何代码，
所以同一组 68 点算出来的矩阵与它逐位相同（1203 张实测：zoom 1.00000、
rot 0.0000°、四角位移 0.000 px）。**模板只从 facelib 取，不手抄**。

68 点仍然全用上，但**只做外点检查**：拟出 T 后把 68 个点都映射进单位空间，与
UNIT_68 比残差（换算成芯片像素），超过阈值就整张脸判 rejected、写日志、不进
aligned。理由：33 点定几何，68 点当证人 —— 拟合权重越少越不容易被坏点带偏，
而"多出来的 35 个点对不对"恰好是检测跑飞的最好证据。阈值见 outlier_limit
（DFL_ALIGN_OUTLIER_PX）。

注意：这个残差**和姿态强相关**（同一个模板贴侧脸当然贴不上）。1203 张实测
|yaw|<15° 时 p50 55 px、|yaw|>45° 时 p50 168 px（芯片像素）。所以默认阈值按
"灾难性失败"来定（0.55 × 画布边长，见 OUTLIER_FRACTION），不是按 p99 定 ——
实测 resid/S 最大 0.370，按 p99（0.34）定等于把侧脸全扔掉。
"""
from __future__ import annotations

import io
import os
import pickle
import struct
import threading

from pathlib import Path

import numpy as np

# 512x512 对齐空间（whole_face）里的 68 点经验模板
TEMPLATE_68 = np.array([
    [105.257, 206.724], [106.436, 245.065], [111.383, 281.135], [119.033, 315.074],
    [132.562, 350.413], [153.951, 379.906], [178.972, 401.578], [209.278, 420.109],
    [250.766, 430.961], [289.974, 419.777], [316.838, 400.404], [339.049, 378.873],
    [358.908, 349.447], [372.550, 314.272], [380.681, 280.344], [386.405, 244.806],
    [387.842, 207.726], [139.488, 172.741], [156.277, 159.222], [177.128, 154.829],
    [198.050, 157.702], [217.324, 164.172], [292.028, 163.896], [309.841, 158.367],
    [329.137, 156.583], [348.272, 161.370], [362.825, 173.657], [255.681, 208.733],
    [255.998, 232.241], [256.406, 254.200], [256.980, 272.792], [229.761, 295.180],
    [241.181, 296.846], [254.698, 299.167], [266.894, 298.150], [277.669, 294.485],
    [164.103, 204.413], [179.650, 195.837], [197.577, 196.174], [214.156, 206.886],
    [198.777, 211.880], [179.002, 211.536], [292.409, 206.903], [309.457, 196.106],
    [327.028, 195.859], [340.732, 205.011], [326.546, 211.523], [307.670, 211.236],
    [206.558, 342.784], [223.748, 331.696], [242.613, 324.300], [254.920, 326.718],
    [266.423, 324.513], [283.299, 331.924], [298.769, 343.977], [282.238, 358.576],
    [268.512, 365.502], [253.312, 366.716], [237.536, 365.044], [222.412, 356.805],
    [210.413, 341.844], [239.446, 337.723], [253.936, 337.221], [267.516, 337.846],
    [293.185, 342.974], [266.977, 348.086], [253.278, 349.008], [238.415, 348.208],
], dtype=np.float64)

SIZE = 512
PADDING = 0.40          # FaceType.WHOLE_FACE
WHOLE_FACE_SHIFT = 0.07  # DFL 整脸对齐把中心往下推 7%


# 5 点对应的 3D 参考点，取自 DFL 自己的 landmarks_68_3D 的 [36, 45, 30, 48, 54]
# （左右眼外角 / 鼻尖 / 左右嘴角），mm，y 向下、z 朝外。
# 关键：这 5 点必须在**原帧坐标系**里用 —— 对齐之后旋转已经被抹平，解不出姿态。
FACE3D_5 = np.array([
    [-46.0473, -37.4714,   7.0380],
    [ 45.1708, -38.0865,   4.8942],
    [  1.2268,  -8.4145, -36.9481],
    [-28.9163,  28.6127,  -2.2403],
    [ 28.7944,  28.0799,  -3.2174],
], dtype=np.float64)

# 与 4.0 estimate_pitch_yaw_roll 对齐的线性校准（400 张真实 src 拟合）
_YAW_K, _YAW_B = 0.6315, -2.96
_PITCH_K, _PITCH_B = 0.6833, 0.84


def pose_from_kps(kps5, box):
    """原帧 5 点 → (yaw, pitch, roll) 角度制；失败返回 (None, None, None)。

    这是"预对齐"那一步的副产品：5 点本来就带着头姿，只要在对齐**之前**解。
    实测（400 张真实对齐脸，用它们的 source 5 点）与 4.0 estimate_pitch_yaw_roll 相比：
        yaw     相关 0.945，校准后残差 std 7.3°
        -pitch  相关 0.889，校准后残差 std 3.7°
    注意 4.0 的那个函数返回的是 (-pitch, yaw, roll)，这里按常规的 (yaw, pitch, roll) 存。
    """
    import cv2
    import math
    try:
        pts = np.asarray(kps5, dtype=np.float64).reshape(5, 2)
        x1, y1, x2, y2 = [float(v) for v in box]
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        f = 2.0 * max(w, h)
        cam = np.array([[f, 0, (x1 + x2) / 2.0], [0, f, (y1 + y2) / 2.0], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(FACE3D_5, pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None, None, None
        R, _ = cv2.Rodrigues(rvec)
        sy = math.hypot(R[0, 0], R[1, 0])
        if sy < 1e-6:
            x = math.atan2(-R[1, 2], R[1, 1]); y = math.atan2(-R[2, 0], sy); z = 0.0
        else:
            x = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(-R[2, 0], sy); z = math.atan2(R[1, 0], R[0, 0])
        neg_pitch = math.degrees(-x)
        yaw = math.degrees(y)
        roll = math.degrees(z)
        yaw_c = _YAW_K * yaw + _YAW_B
        neg_pitch_c = _PITCH_K * neg_pitch + _PITCH_B
        half = 90.0
        return (round(float(np.clip(yaw_c, -half, half)), 2),
                round(float(np.clip(-neg_pitch_c, -half, half)), 2),
                round(float(np.clip(roll, -half, half)), 2))
    except Exception:
        return None, None, None


def encode_xseg_mask(mask) -> bytes | None:
    """把 (H,W) 的 0..1 遮罩压成 DFL 的存储格式（PNG/JPEG 字节，≤ 50000 字节）。

    与 DFLJPG.set_xseg_mask 一致：先 PNG，超长再逐级降 JPEG 质量。
    """
    import cv2
    if mask is None:
        return None
    img = np.clip(np.asarray(mask, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    if ok and len(buf) <= 50000:
        return buf.tobytes()
    for q in range(100, -1, -5):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok and len(buf) <= 50000:
            return buf.tobytes()
    return None


def umeyama(src, dst, estimate_scale=True):
    """相似变换最小二乘（Shinji Umeyama, PAMI 1991），返回 3x3 齐次矩阵。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


def transform_points(points, mat, invert=False):
    import cv2
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.invertAffineTransform(np.asarray(mat, dtype=np.float64)) if invert else mat
    return cv2.transform(pts, np.asarray(m, dtype=np.float64)).reshape(-1, 2)


def _affine_from_unit_square(mat_unit, output_size=SIZE, padding=PADDING,
                             whole_face_shift=WHOLE_FACE_SHIFT):
    """单位方形 → 输出图的仿射矩阵（DFL get_transform_mat 的几何部分）。

    mat_unit 是「原图 → 单位方形」的相似变换；输出图把这个单位方形按 padding
    外扩，再把三个角点映到 (0,0)/(S,0)/(S,S)。
    """
    import cv2
    output_size = int(output_size)
    corners = np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    g_p = transform_points(corners, mat_unit, invert=True)      # 单位方形 → 原图
    g_c = g_p[4].astype(np.float64)

    tb = (g_p[2] - g_p[0]).astype(np.float64)
    tb /= max(1e-9, np.linalg.norm(tb))
    bt = (g_p[1] - g_p[3]).astype(np.float64)
    bt /= max(1e-9, np.linalg.norm(bt))

    mod = float(np.linalg.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))

    if whole_face_shift:
        v = (g_p[0] - g_p[3]).astype(np.float64)
        v_len = float(np.linalg.norm(v))
        if v_len > 1e-9:
            g_c = g_c + (v / v_len) * v_len * float(whole_face_shift)

    l_t = np.float32([g_c - tb * mod, g_c + bt * mod, g_c + tb * mod])
    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    return cv2.getAffineTransform(l_t, pts2)


_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# DFL 自己的 33 点单位模板 = facelib.LandmarksProcessor.landmarks_2D_new。
#
# **只从 facelib 取，不手抄**：这份模板是"唯一真相"，抄一份就等于多一个可能
# 漂移的副本 —— facelib 改成 6 位有效数字，我们这边的副本却停在 5 位，芯片就会
# 悄悄挪 0.01 px，而且没有任何报错。以前这里确实是一份手抄的 float64 字面量
# （和 facelib 的 float32 差 2.9e-08），现在直接 import。
#
# facelib 在 app/engine 下；正常运行时 _merge_compose.ensure_engine_path() 已经
# 把它 append 进 sys.path。单独 import（脚本/测试）时这里再兜一次 —— append 不
# insert，绝不抢同名模块。
_TPL_2D_NEW = None          # 懒加载缓存，见 landmarks_2d_new()
_REAL_IDX = list(range(17, 49)) + [54]       # facelib 认的 33 个下标：17..48 + 54


def landmarks_2d_new():
    """facelib 的 33 点单位模板（float64 视图），首次调用时 import 并缓存。"""
    global _TPL_2D_NEW
    if _TPL_2D_NEW is not None:
        return _TPL_2D_NEW
    try:
        from facelib.LandmarksProcessor import landmarks_2D_new as _tpl
    except Exception:                       # noqa: BLE001 —— 引擎不在 sys.path 上
        import sys
        root = Path(__file__).resolve().parents[1] / "engine"
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from facelib.LandmarksProcessor import landmarks_2D_new as _tpl
    _TPL_2D_NEW = np.asarray(_tpl, dtype=np.float64).copy()
    if _TPL_2D_NEW.shape != (33, 2):
        raise RuntimeError("facelib landmarks_2D_new 形状异常：%r" % (_TPL_2D_NEW.shape,))
    return _TPL_2D_NEW


def _build_unit_68():
    """68 点单位模板：DFL 认得的 33 点用 DFL 原值，缺的点用经验模板补。

    经验模板来自真实对齐脸，它和 DFL 的 33 点模板之间差一个相似变换，
    先用这 33 点把两者对齐，再把剩下 35 个点搬进 DFL 的单位空间。
    """
    tpl = landmarks_2d_new()
    emp_unit = transform_points(
        TEMPLATE_68, _affine_from_unit_square(_IDENTITY, SIZE), invert=True)
    T = umeyama(emp_unit[_REAL_IDX], tpl, True)
    mapped = transform_points(emp_unit, T[0:2])
    mapped[_REAL_IDX] = tpl
    return mapped.astype(np.float64)


UNIT_68 = _build_unit_68()


def template_5(template_68=None):
    """68 点模板里对应的 5 点：双眼中心、鼻尖、左右嘴角。"""
    t = UNIT_68 if template_68 is None else np.asarray(template_68, dtype=np.float64)
    return np.stack([
        t[36:42].mean(axis=0),
        t[42:48].mean(axis=0),
        t[30],
        t[48],
        t[54],
    ]).astype(np.float64)


def _ordered_5(kps5):
    src5 = np.asarray(kps5, dtype=np.float64).reshape(5, 2).copy()
    if src5[1, 0] < src5[0, 0]:
        src5[[0, 1]] = src5[[1, 0]]
    if src5[4, 0] < src5[3, 0]:
        src5[[3, 4]] = src5[[4, 3]]
    return src5


def align_mat_from_5(kps5, output_size=SIZE, padding=PADDING,
                     whole_face_shift=WHOLE_FACE_SHIFT):
    """5 点（原图坐标）→ 原图到对齐图的仿射矩阵。"""
    mat_unit = umeyama(_ordered_5(kps5), template_5(), True)[0:2]
    return _affine_from_unit_square(mat_unit, output_size, padding, whole_face_shift)


# ---------------- 终裁矩阵：拟合 33 点，与 facelib 逐位等价 ----------------
#
# 两道外点阈值，各管一种失败：
#
# ① 绝对（芯片像素）：DFL_ALIGN_OUTLIER_PX。默认 0.55 × 画布边长。
#    为什么不是实测 p99：残差和 |yaw| 强相关（见文件头），p99 挑出来的全是侧脸
#    （实测 resid/S 最大 0.370，p99 就是 0.34），0.55 留 1.5 倍余量，只打
#    「点跑到画布外」级别的失败 —— 检测整体跑飞、点到别的脸上、背景上。
#
# ② 相对（同一张脸内部）：DFL_ALIGN_OUTLIER_REL = max 残差 / 该脸残差中位数。
#    姿态会把**所有**点的残差一起抬高，比值不动；只有「某几个点单独跑掉」才会
#    把比值顶上去 —— 这才是「外点」的原意。2424 张真实脸实测最大 10.60，
#    默认 12 不误伤，同时能抓住「3 个下颌点推出 2 个单位」这种局部崩坏（比值 ~10
#    到 20，看画布）。
OUTLIER_FRACTION = 0.55
OUTLIER_REL = 12.0


def outlier_limit(output_size=SIZE):
    """当前生效的外点阈值（芯片像素）；0 = 关闭。

    DFL_ALIGN_OUTLIER_PX：
      off / 0 / none       关闭
      auto（默认）/ 空      OUTLIER_FRACTION × output_size
      数字                 绝对像素
    """
    raw = (os.environ.get("DFL_ALIGN_OUTLIER_PX") or "auto").strip().lower()
    if raw in ("0", "off", "none", "no", "false", "disable"):
        return 0.0
    if raw in ("", "auto", "default", "on", "true"):
        return float(output_size) * OUTLIER_FRACTION
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(output_size) * OUTLIER_FRACTION


def outlier_rel_limit():
    """相对外点阈值（max/median）；0 = 关闭。DFL_ALIGN_OUTLIER_REL。"""
    raw = (os.environ.get("DFL_ALIGN_OUTLIER_REL") or "auto").strip().lower()
    if raw in ("0", "off", "none", "no", "false", "disable"):
        return 0.0
    if raw in ("", "auto", "default", "on", "true"):
        return float(OUTLIER_REL)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(OUTLIER_REL)


def align_fit(kps68, output_size=SIZE, padding=PADDING,
              whole_face_shift=WHOLE_FACE_SHIFT):
    """68 点 → (mat, T, resid_px, px_per_unit)；算不出来返回 (None, None, None, None)。

    mat        原图 → 对齐图的仿射矩阵（= iperov get_transform_mat 的同一张）
    T          原图 → 单位方形的相似变换（3×3）
    resid_px   68 个点各自的残差（芯片像素）：点映射进单位空间后与 UNIT_68 比，
               再乘「芯片像素 / 单位」。33 个拟合点这里是拟合残差，其余 35 个是
               「这张脸和平均脸差多少」—— 外点检查看的是这一列的最大值。
    """
    src = np.asarray(kps68, dtype=np.float64)
    if src.ndim != 2 or src.shape != (68, 2) or not np.all(np.isfinite(src)):
        return None, None, None, None

    tpl = landmarks_2d_new()
    T = umeyama(src[_REAL_IDX], tpl, True)          # 只拟合 facelib 认的 33 点
    if not np.all(np.isfinite(T)):
        return None, None, None, None
    mat = _affine_from_unit_square(T[0:2], output_size, padding, whole_face_shift)
    if not np.all(np.isfinite(mat)):
        return None, None, None, None

    # 单位 → 芯片 = mat（原图→芯片）∘ T⁻¹（单位→原图）；px_per_unit 就是它的缩放
    try:
        unit2chip = np.vstack([mat, [0.0, 0.0, 1.0]]) @ np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return None, None, None, None
    px_per_unit = float(np.hypot(unit2chip[0, 0], unit2chip[1, 0]))
    act_unit = transform_points(src, T[0:2])                  # 实际点：原图 → 单位（T 是仿射）
    resid = np.linalg.norm(act_unit - UNIT_68, axis=1) * px_per_unit
    if not np.all(np.isfinite(resid)):
        return None, None, None, None
    return mat, T, resid, px_per_unit


def align_mat_from_68(kps68, output_size=SIZE, padding=PADDING,
                     whole_face_shift=WHOLE_FACE_SHIFT):
    """68 点（原图坐标）→ 原图到对齐图的仿射矩阵；**与 facelib 逐位等价**。

    拟合只用 33 点（17..48 + 54）对 facelib 的 landmarks_2D_new —— 和
    facelib.LandmarksProcessor.get_transform_mat 完全同一份模板、同一串几何代码
    （padding 0.40 / 整脸下移 7% / 三对角点求仿射都没动）。所以同一组 68 点，
    这里算出来的矩阵和 iperov 的 WHOLE_FACE 矩阵是同一个东西（1203 张实测
    zoom 1.00000 / rot 0.0000° / 四角 0.000 px）。

    剩下的 35 个点不参与拟合，只给 align_face_68 做外点检查 —— 为什么以前用
    全部 68 点会错，见文件头。

    拿不到 68 个点（数量不对 / 有 NaN / 退化）返回 None —— 调用方必须报错，
    **不许退回 5 点**。
    """
    mat, _T, _resid, _ppu = align_fit(kps68, output_size, padding, whole_face_shift)
    return mat


def align_face_68(img_bgr, kps68, box=None, size=SIZE, interpolation=None):
    """68 点仿射裁一张对齐脸。返回 (对齐图, 元数据 dict)；失败返回 (None, meta)。

    人脸库（extract / add 两条路）统一走这里：插值默认 INTER_CUBIC —— 和合成链路
    （MergeMaskedFace 那条路）保持一致，比 5 点 + INTER_LINEAR 的边缘更干净。
    border 仍按 DFL 的规矩填纯黑。

    **外点检查**：几何只用 33 点拟合（= facelib 逐位等价），68 个点全映射进单位
    空间与 UNIT_68 比残差。max 残差超过 outlier_limit(size) 就判这张脸检测异常：
    不返回图，返回的 meta 里 rejected=True + reject_reason + align_resid_px，
    调用方据此跳过并记账（不进 aligned）。阈值可用 DFL_ALIGN_OUTLIER_PX 调。
    """
    import cv2
    size = int(size)
    if interpolation is None:
        interpolation = cv2.INTER_CUBIC
    mat, _T, resid, _ppu = align_fit(kps68, size)
    if mat is None:
        return None, None

    resid_max = float(resid.max())
    resid_med = float(np.median(resid))
    resid_rel = resid_max / max(1e-6, resid_med)
    limit = outlier_limit(size)
    limit_rel = outlier_rel_limit()
    bad = ((limit > 0 and resid_max > limit)
           or (limit_rel > 0 and resid_rel > limit_rel))
    if bad:
        reason = ("68 点残差 %.1f px / 中位 %.1f px = %.1f 倍（阈值 %.1f px / %.1f 倍）"
                  % (resid_max, resid_med, resid_rel, limit, limit_rel))
        _say("[align] rejected：%s" % reason)
        return None, {
            "face_type": "whole_face",
            "align": "68",
            "landmarks_source": "none",
            "rejected": True,
            "reject_reason": reason,
            "align_resid_px": round(resid_max, 3),
            "align_resid_rel": round(resid_rel, 3),
            "align_resid_limit_px": round(limit, 3),
            "align_resid_limit_rel": round(limit_rel, 3),
        }

    aligned = cv2.warpAffine(img_bgr, mat, (size, size),
                             flags=int(interpolation),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if aligned is None:
        return None, None

    pts = np.asarray(kps68, dtype=np.float64).round(4)
    lm_aligned = transform_points(pts, mat).astype(np.float64)

    h, w = img_bgr.shape[:2]
    if box is None:
        box = [float(pts[:, 0].min()), float(pts[:, 1].min()),
               float(pts[:, 0].max()), float(pts[:, 1].max())]
    rect = [int(max(0, box[0])), int(max(0, box[1])),
            int(min(w, box[2])), int(min(h, box[3]))]

    meta = {
        "face_type": "whole_face",
        "landmarks": np.round(lm_aligned, 4).tolist(),
        "source_landmarks": pts.tolist(),
        "source_rect": rect,
        "image_to_face_mat": np.asarray(mat, dtype=np.float64).tolist(),
        "align": "68",
        "landmarks_source": "model68",     # 这 68 个点是 2D106 的输出（按下标挑的）
        # 外点检查的实际读数（芯片像素 / 相对倍数）：写进元数据，重切/排查不用再算
        "align_resid_px": round(resid_max, 3),
        "align_resid_rel": round(resid_rel, 3),
    }
    return aligned, meta


# ---------------- 5 点：只做几何，只用于给 2D106 准备输入 ----------------

ALIGN_MODES = ("68", "5")


def align_mode() -> str:
    """DFL_ALIGN=68|5，默认 68。

    68 = 唯一的主路径（68 点仿射，点来自 2D106）；
     5 = 临时回退 / 对照用：只按 5 点仿射裁剪，**元数据里没有 landmarks**。
     """
    v = (os.environ.get("DFL_ALIGN") or "68").strip().lower()
    return "5" if v in ("5", "5pt", "5-pt", "five") else "68"


# 粗裁画布边长。2D106 的输入本来就是 192（InsightFace106.landmarks68_batch 会把
# 输入 resize 到 192），所以粗裁直接出 192：三次 warp 里最便宜的一次，
# 而且不用再让模型那层多缩一次。
COARSE_SIZE = 192
# 粗裁视野 = 终裁视野 × 1.3。侧脸时 5 点算出来的紧框会把额头/下巴/耳朵切掉，
# 2D106 看到切边的脸，出的点会整体往画面中心缩；多给 30% 余量就没这问题。
# 这不会污染最终构图：终裁是「33 点 → facelib 的 landmarks_2D_new」的相似变换拟合，
# 模板把构图钉死了，粗裁多看到多少、少看到多少，都会被那一步吸收掉（缩放/平移都在
# 相似变换里）。粗裁只影响 2D106 看得清不清楚，不进入几何。
COARSE_EXPAND = 1.30


def _zoom_about_center(mat, out_size, factor):
    """仿射矩阵的视野放大 factor 倍：同一个输出中心，看进去更多，输出画布不变。"""
    c = float(out_size) / 2.0
    s = 1.0 / float(factor)
    Z = np.array([[s, 0.0, c * (1.0 - s)],
                  [0.0, s, c * (1.0 - s)],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    M = np.vstack([np.asarray(mat, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (Z @ M)[0:2]


def coarse_align_from_5(kps5, out_size=COARSE_SIZE, expand=COARSE_EXPAND):
    """5 点 → 「宽松粗裁」的仿射矩阵（原图 → 粗裁图）。粗裁图**永不落盘**。

    几何上就是老 5 点裁剪的同一套模板 / padding / 整脸中心偏移，只改两件事：
      1) 画布缩到 192 —— 2D106 内部也要缩到 192，多出来的像素没人用；
      2) 视野再放大 expand 倍 —— 让 106 看到整张脸（含被 5 点紧框切掉的边缘）。
    """
    base = align_mat_from_5(kps5, output_size=out_size)
    return _zoom_about_center(base, out_size, expand)


def landmarks68_in_source(lm68_coarse, mat_coarse):
    """粗裁图坐标的 68 点 → 原图坐标。拿不到就抛，不做任何估算兜底。"""
    src = transform_points(np.asarray(lm68_coarse, dtype=np.float64).reshape(-1, 2),
                           np.asarray(mat_coarse, dtype=np.float64), invert=True)
    if src.shape != (68, 2) or not np.all(np.isfinite(src)):
        raise ValueError("68 点反变换失败（数量/数值不合法）")
    return src


def align_face(img_bgr, kps5, box=None, size=SIZE):
    """5 点仿射裁一张对齐脸。返回 (对齐图, 元数据 dict)；失败返回 (None, None)。

    这条只给两处用：
      * DFL_ALIGN=5 的回退/对照路径（旧行为，逐像素与 4.1 早期一致）；
      * 给人脸库/合成链路当 2D106 的输入（先粗对一次，再让模型出点）。

    **它不产出任何关键点。** 以前这里会把 5 点插值成 68 点写进 meta，结果那
    68 个数被当成真值写进 dat —— 见文件头的说明。
    现在 meta 里没有 landmarks，只有 landmarks_source="none"。
    """
    import cv2
    size = int(size)
    mat = align_mat_from_5(kps5, size)
    if not np.all(np.isfinite(mat)):
        return None, None
    # 越界部分填纯黑（DFL 的做法）。以前用 BORDER_REPLICATE，边缘会被拉出一圈
    # 拉伸的"拖影"，看着脏，还会污染训练（模型学到边框拖影）。
    aligned = cv2.warpAffine(img_bgr, mat, (size, size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    pts5 = _ordered_5(kps5)
    h, w = img_bgr.shape[:2]
    if box is None:
        box = [float(pts5[:, 0].min()), float(pts5[:, 1].min()),
               float(pts5[:, 0].max()), float(pts5[:, 1].max())]
    rect = [int(max(0, box[0])), int(max(0, box[1])),
            int(min(w, box[2])), int(min(h, box[3]))]

    meta = {
        "face_type": "whole_face",
        "source_rect": rect,
        "image_to_face_mat": np.asarray(mat, dtype=np.float64).tolist(),
        "align": "5",
        "landmarks_source": "none",        # 5 点不产生 68 点，这里没有可信关键点
    }
    return aligned, meta


def dfljpg_bytes(jpeg_data: bytes, meta: dict) -> bytes:
    """把 APP15 段插进 JPEG：SOI/APPn 之后、SOS 之前，与 DFL 的 DFLJPG 一致。"""
    data = bytes(jpeg_data)
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a jpeg")

    clean = {k: v for k, v in meta.items() if v is not None}
    pkl = pickle.dumps(clean, protocol=2)
    pkl = pkl.replace(b"numpy._core", b"numpy.core")   # numpy 2.x → 1.x 兼容
    if len(pkl) + 2 > 0xFFFF:
        raise ValueError("DFLJPG metadata too large")
    app15 = b"\xff\xef" + struct.pack(">H", len(pkl) + 2) + pkl

    sos = data.find(b"\xff\xda")
    if sos <= 0:
        return data + app15
    return data[:sos] + app15 + data[sos:]


# ---------------- JPEG 4:4:4：一次探测，之后不再解释 ----------------
#
# 对齐脸是整个流水线的原始素材：色度在这里被抽稀，后面没有任何一步能长回来。
# 4:2:0 把 Cb/Cr 各砍掉一半（每 2x2 只留一份），先糊的就是嘴唇、眼白边缘和肤色过渡。
#
# cv2 5.0 换了这条参数的写法：IMWRITE_JPEG_SAMPLING_FACTOR 只认
# ImwriteJPEGSamplingFactorParams 枚举（IMWRITE_JPEG_SAMPLING_FACTOR_444 之类），
# 老写法 `[..., IMWRITE_JPEG_SAMPLING_FACTOR, 0]` 会被**拒绝**：
#
#   [ WARN:0@0.053] grfmt_jpeg.cpp:809 cv::JpegEncoder::write The value(0) for
#   IMWRITE_JPEG_SAMPLING_FACTOR must be one of ImwriteJPEGSamplingFactorParams.
#   It is fallbacked to IMWRITE_JPEG_SAMPLING_FACTOR_420.
#
# 关键在于 imencode 照样返回 True、照样落盘 —— 静默变成 4:2:0。老代码那句
# `except AttributeError` 兜不住它（常量在，值是错的），于是「切脸传了
# subsampling=0」在 cv2 5.0 上从来没生效过：实测带不带这个参数编出来的字节
# **一模一样**（同一张 256² 图都是 78070 字节，且 SOF 头是 (2,2)(1,1)(1,1)）。
#
# 所以现在的规矩是：
#   1) 不硬编码枚举数字 —— 按 cv2 有没有这两个常量探测；
#   2) 不信参数，**按落盘字节的 SOF 头核对**真的 4:4:4 才算数；
#   3) 探测一次、缓存；之后不再刷日志。实在拿不到就 WARN 一次，并说清后果。
_JPEG444_LOCK = threading.Lock()
_JPEG444 = None          # 探测结果 dict；None = 还没探过（见 jpeg444_probe）


def _say(msg):
    """打一条日志，永不让抽取挂掉（控制台是 GBK 时个别字符编不出来就降级）。"""
    import sys
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", "") or "utf-8")
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


def jpeg_sof_sampling(blob):
    """JPEG 字节 → [(h, v), ...]（Y/Cb/Cr 的采样因子）；不是 JPEG / 读不到返回 None。

    这是**文件里的事实**，不是参数意图：4:4:4 = 三个分量都 (1,1)；
    4:2:0 = 第一个 (2,2)、后两个 (1,1)。只看 SOFn 头，不解像素。
    """
    data = bytes(blob or b"")
    if data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:                       # 填充字节
            i += 1
            continue
        if m == 0x01 or m == 0xD8 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9:                       # 到 EOI 还没见到 SOF
            return None
        seglen = (data[i + 2] << 8) | data[i + 3]
        if seglen < 2:
            return None
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            nf = data[i + 9] if i + 9 < n else 0
            p = i + 10
            comps = []
            for _ in range(nf):
                if p + 1 >= n:
                    return None
                hv = data[p + 1]
                comps.append((hv >> 4, hv & 0x0F))
                p += 3
            return comps or None
        if m == 0xDA:                       # SOS 之后是熵编码数据
            return None
        i += 2 + seglen
    return None


def jpeg444_probe(force: bool = False) -> dict:
    """探测一次：这台机器的 cv2 怎么写才**真的**产出 4:4:4（结果缓存在模块里）。

    返回 dict：
      cv2     —— cv2.__version__（日志和报告都要能写清是哪一版）
      path    —— "enum444"（IMWRITE_JPEG_SAMPLING_FACTOR + _444 枚举，cv2 5.x/4.5.3+）
                 "int0"    （老 cv2：常量在、只认裸整数 0）
                 "none"    （两条都不行，库默认 4:2:0，已 WARN 过）
      sf      —— 要追加进 imencode 的 [(id, value), ...]；空列表 = 不带采样参数
      ok      —— True 表示「编出来的字节真的是 4:4:4」（按 SOF 头核对）
      detail  —— 证据：哪条候选、编出来的分量采样因子

    只探一次：第二条脸开始就是查缓存，不会再打日志，也不会重复编码。
    """
    global _JPEG444
    with _JPEG444_LOCK:
        if _JPEG444 is not None and not force:
            return _JPEG444
        import cv2
        res = {"cv2": "?", "path": "none", "sf": [], "ok": False, "detail": ""}
        try:
            res["cv2"] = str(cv2.__version__)
        except Exception:                              # noqa: BLE001
            pass
        # 16x16 样板图：采样因子写在文件头，与内容无关；但用彩色图能顺带确认编码
        # 走的是普通 JPEG 路径（不是被某个分支截走）。
        tile = np.zeros((16, 16, 3), np.uint8)
        tile[:, :8] = (255, 0, 0)
        tile[:, 8:] = (0, 0, 255)
        q = int(getattr(cv2, "IMWRITE_JPEG_QUALITY", 1))
        fid = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        cands = []
        if fid is not None and getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None) is not None:
            cands.append(("enum444",
                          [(int(fid), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444))]))
        if fid is not None:
            cands.append(("int0", [(int(fid), 0)]))
        notes = []
        for name, sf in cands:
            trial = [(q, 95)] + [(int(k), int(v)) for k, v in sf]
            flat = [int(x) for pair in trial for x in pair]
            try:
                ok, enc = cv2.imencode(".jpg", tile, flat)
            except Exception as e:                     # noqa: BLE001 —— 老 cv2 可能直接抛
                notes.append("%s 抛异常（%s）" % (name, str(e).splitlines()[0][:60]))
                continue
            if not ok or enc is None:
                notes.append("%s 返回失败" % name)
                continue
            comps = jpeg_sof_sampling(enc.tobytes())
            if comps and len(comps) == 3 and all(tuple(c) == (1, 1) for c in comps):
                res.update(path=name, sf=sf, ok=True,
                           detail="%s 编码后 SOF 采样因子 %s" % (name, comps))
                break
            notes.append("%s 编出来是 %s" % (name, comps))
        if res["ok"]:
            _say("[align] JPEG 4:4:4 探测：cv2 %s → %s（%s）"
                 % (res["cv2"], res["path"], res["detail"]))
        else:
            _say("[align] 警告：这台机器的 OpenCV 写不出 4:4:4（%s）。"
                 "对齐脸只能按库默认 4:2:0 落盘 —— 色度被抽稀（每 2x2 只留一份 Cb/Cr），"
                 "嘴唇/眼白边缘和肤色过渡的细节会永久丢失，后面训练再补不回来。"
                 "本条只提示一次。" % ("；".join(notes) or "cv2 没有 SAMPLING_FACTOR 常量"))
        _JPEG444 = res
        return res


def _subsampling_params(subsampling):
    """老式整数 subsampling → 当前 cv2 真认的 [(id, value), ...]；[] = 不带采样参数。

    0=4:4:4 / 1=4:2:2 / 2=4:2:0（老 cv2 的 IMWRITE_JPEG_SAMPLING_FACTOR 就是这套整数，
    cv2 5.0 换成枚举后裸数字会被拒并静默降级）。只有 0 需要探测 + 验证，
    其余按枚举名查，查不到就不带参数（不猜）。
    """
    import cv2
    v = int(subsampling)
    if v == 0:
        probe = jpeg444_probe()
        return [tuple(p) for p in probe["sf"]] if probe["ok"] else []
    name = {1: "IMWRITE_JPEG_SAMPLING_FACTOR_422",
            2: "IMWRITE_JPEG_SAMPLING_FACTOR_420"}.get(v)
    fid = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
    enum = getattr(cv2, name, None) if name else None
    if fid is None:
        return []
    if enum is not None:
        return [(int(fid), int(enum))]
    return [(int(fid), v)]                    # 老 cv2：裸整数就是这套语义


def jpeg_encode_params(quality: int = 95, subsampling=0) -> list:
    """imencode 参数表：质量 + （真能生效的）采样因子写法。subsampling=None = 不带。"""
    import cv2
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if subsampling is not None:
        params += [int(x) for pair in _subsampling_params(subsampling) for x in pair]
    return params


def write_dfljpg(path, img_bgr, meta: dict, quality: int = 95, subsampling=0):
    """写一张 DFLJPG（JPEG + APP15 元数据）。

    默认 **4:4:4**（subsampling=0）：对齐脸是原始素材，色度在这里丢了就再长不回来。
    4.1 以前默认是库默认的 4:2:0，切脸阶段虽然写了 subsampling=0，但在 cv2 5.0 上
    那个 0 被拒绝、静默落成 4:2:0（见上面那段注释）—— 于是「最高质量」一直是假的。

    subsampling=None → 一个采样参数都不带，库默认 4:2:0：只给对照 / 复现老产物用。
    「传了 0」不等于「落盘是 4:4:4」：真正生效的写法由 jpeg444_probe() 一次探测决定。
    """
    import cv2
    params = jpeg_encode_params(quality, subsampling)
    ok, enc = cv2.imencode(".jpg", img_bgr, params)
    if not ok:
        raise RuntimeError("JPEG 编码失败: %s" % path)
    blob = dfljpg_bytes(enc.tobytes(), meta)
    with open(str(path), "wb") as f:
        f.write(blob)
    return len(blob)


# ---------------- 批量应用遮罩（提取之后单独一步，与 DFL 的 extract → mask 流程一致） ----------------

# DFL 的 APP15 段是 pickle —— 直接 loads 等于让一张 jpg 执行任意代码。
# 数据集里混进外部图片时这就是个真实的入口，所以只放行数据类对象。
#
# ("_codecs","encode") 是老漏的那条：pickle protocol≤2 没有 BINBYTES，bytes 一律存成
#   _codecs.encode(str(obj,'latin1'), 'latin1')
# 而 xseg_mask 恰好就是 bytes → 整个 data_src/aligned（1203 张，实测 1203/1203 都在
# 旧段里带 xseg_mask）用这份白名单一张都解不开，read_dfljpg_meta 对所有带遮罩的图
# 都返回 None。空 bytes 走的是另一条路（pickle 流里写的是 py2 名字 __builtin__.bytes），
# 所以两条都放行；解码侧 _codecs.decode 同理，留着对称。
_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("builtins", "bytes"),
    ("__builtin__", "bytes"),
    ("_codecs", "encode"),
    ("_codecs", "decode"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError("blocked global: %s.%s" % (module, name))


def safe_loads(blob: bytes):
    return _SafeUnpickler(io.BytesIO(blob)).load()


_HEAD_BYTES = 256 * 1024      # 头部段（EXIF / APP15）够用
_TAIL_BYTES = 256 * 1024      # DFL 的 APP15 也可能挂在 EOI 之前（带 xseg 遮罩时更大）
_EOI = bytes((0xFF, 0xD9))
_APP15 = bytes((0xFF, 0xEF))
_SOI = bytes((0xFF, 0xD8))


def _read_head_tail(path):
    """只读首尾两块，避免为了几十 KB 元数据把整张图搬进内存。"""
    import os as _os
    p_ = Path(path)
    try:
        size = _os.path.getsize(p_)
    except OSError:
        return b'', 0
    if size <= _HEAD_BYTES + _TAIL_BYTES:
        try:
            return p_.read_bytes(), 0
        except OSError:
            return b'', 0
    try:
        with open(p_, 'rb') as f:
            head = f.read(_HEAD_BYTES)
            f.seek(size - _TAIL_BYTES)
            tail = f.read(_TAIL_BYTES)
    except OSError:
        return b'', 0
    return head + tail, len(head)


def _unpickle_dict(payload):
    """APP15 的 payload → dict；不是 pickle / 解不开 / 解出来不是 dict 都返回 None。"""
    if not payload or payload[:1] != b"\x80":      # protocol 2 的 PROTO 开头
        return None
    try:
        obj = safe_loads(bytes(payload))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def split_jpeg(data):
    """→ (header, tail)：header 到 SOS 之前为止，tail 从 SOS 起到文件尾。

    DFLJPG.load_raw 从头一段一段走到 SOS，然后直接跳到 EOI 划出压缩数据段 ——
    所以 **APP15 只有落在 SOS 之前的头部才读得到**，挂在 EOI 前面等于没写。
    """
    data = bytes(data)
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        m = data[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:      # 无长度字段的标记
            i += 2
            continue
        if m == 0xDA:                                   # SOS
            return data[:i], data[i:]
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return data, b""


def _head_segments(header):
    """头部段 → [(marker, 原始字节)]，顺序保持原样。

    marker 为 None 表示「认不出长度字段、剩下原样带走」。SOI / TEM / RSTn 这些
    零长度标记单独保留：老写法只认带长度字段的段，碰到它们会把后面的段长读错，
    把扫描数据复制一遍，文件变大还解不开。
    """
    out, i, n = [], 2, len(header)
    while i < n:
        if header[i] != 0xFF or i + 1 >= n:
            out.append((None, bytes(header[i:])))
            break
        m = header[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            out.append((m, bytes(header[i:i + 2])))
            i += 2
            continue
        if i + 4 > n:
            out.append((None, bytes(header[i:])))
            break
        end = i + 2 + struct.unpack(">H", header[i + 2:i + 4])[0]
        if end > n or end <= i:
            out.append((None, bytes(header[i:])))
            break
        out.append((m, bytes(header[i:end])))
        i = end
    return out


def find_app15_header(data):
    """头部（SOS 之前）那份 APP15 的 dict —— 引擎 DFLJPG.load 只认这一份。"""
    data = bytes(data)
    if not data.startswith(_SOI):
        return None
    header, _tail = split_jpeg(data)
    for m, seg in _head_segments(header):
        if m == 0xEF:
            return _unpickle_dict(seg[4:])
    return None


def find_app15_legacy(data):
    """挂在 EOI 之前的那份 APP15 的 dict —— 旧写法（本文件的 rewrite_dfljpg_meta
    在 4.1 之前）留下的死重。

    DFLJPG.load_raw 遇到 SOS 直接跳到 EOI，所以这份引擎永远看不到：训练一直用的是
    「没有遮罩」的头部那份。这里读出来只为两件事：迁移（搬到头部）+ 前端回显。

    判据必须三条同时成立：正好顶到 EOI、payload 是 pickle 的 PROTO 开头(0x80)、
    而且真能解出 dict —— 只按「长度正好凑到 EOI」判断会撞上压缩数据里随机的
    FF EF（实测能把一张 40 KB 的脸图砍掉 8 KB 扫描数据，图直接废掉）。
    """
    data = bytes(data)
    if not data.startswith(_SOI):
        return None
    eoi = data.rfind(_EOI)
    if eoi <= 0:
        return None
    pos = data.rfind(_APP15, 0, eoi)
    while pos > 0:
        if pos + 4 <= eoi:
            seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            end = pos + 2 + seg_len
            if end == eoi:
                obj = _unpickle_dict(data[pos + 4:end])
                if isinstance(obj, dict):
                    return obj
        pos = data.rfind(_APP15, 0, pos)
    return None


def _strip_legacy_app15(tail):
    """摘掉 tail 里顶到 EOI 的那段 legacy APP15；没有就原样返回。"""
    if not tail:
        return tail
    eoi = tail.rfind(_EOI)
    if eoi <= 0:
        return tail
    pos = tail.rfind(_APP15, 0, eoi)
    while pos > 0:
        if pos + 4 <= eoi:
            seg_len = struct.unpack(">H", tail[pos + 2:pos + 4])[0]
            end = pos + 2 + seg_len
            if end == eoi and isinstance(_unpickle_dict(tail[pos + 4:end]), dict):
                return tail[:pos] + tail[eoi:]
        pos = tail.rfind(_APP15, 0, pos)
    return tail


def read_dfljpg_metas(path):
    """→ (头部 APP15 dict | None, EOI 前 legacy APP15 dict | None)。

    只读首尾各 256KB；头部段跨过 256KB 窗口才整读兜底（异常文件才走到）。
    """
    data, off = _read_head_tail(path)
    if not data or not data.startswith(_SOI):
        return None, None
    head = find_app15_header(data)
    legacy = find_app15_legacy(data)
    if off and head is None:
        try:
            whole = Path(path).read_bytes()
        except OSError:
            return head, legacy
        if len(whole) > len(data):
            head = find_app15_header(whole)
            if legacy is None:
                legacy = find_app15_legacy(whole)
    return head, legacy


def read_dfljpg_meta(path):
    """读出 DFLJPG 的元数据 dict；没有就返回 None。

    头部（引擎读的那份）优先，其次 EOI 前那份 legacy —— 整个 data_src/aligned 都是
    「头部没遮罩 + EOI 前带 xseg_mask」的双份布局，只看头部等于把遮罩全丢掉。
    """
    head, legacy = read_dfljpg_metas(path)
    if head is not None:
        return head
    return legacy


def encode_app15_segment(meta):
    """meta → 完整的 APP15 段（FF EF + 段长 + pickle payload）。"""
    clean = {k: v for k, v in dict(meta or {}).items() if v is not None}
    pkl = pickle.dumps(clean, protocol=2)
    pkl = pkl.replace(b"numpy._core", b"numpy.core")     # 先替换，再算长度
    if len(pkl) + 2 > 0xFFFF:
        raise ValueError("DFLJPG metadata too large")
    return _APP15 + struct.pack(">H", len(pkl) + 2) + pkl


def write_app15_head(data, meta):
    """把 meta 写成**头部** APP15，返回新文件的字节。纯函数，不落盘。

    1) 在 SOS 处切开 header / tail；
    2) 头部旧的 APP15 丢掉，新段插在最后一个 APP* 段之后（DFLJPG.dump 同款位置）；
    3) tail 里顶到 EOI 的那段 legacy APP15 摘掉 —— 写的时候不再产生，
       但读的时候仍然认（整个 aligned 目录的历史文件都还带着它）。
    """
    data = bytes(data)
    if not data.startswith(_SOI):
        raise ValueError("not a jpeg")
    header, tail = split_jpeg(data)
    body, insert_at = [], 0
    for m, seg in _head_segments(header):
        if m == 0xEF:                                   # 旧头部 APP15：丢掉
            continue
        body.append(seg)
        if m is not None and (m & 0xF0) == 0xE0:        # APP0..APP15
            insert_at = len(body)                       # 插在最后一个 APP* 之后
    tail = _strip_legacy_app15(tail)
    seg = encode_app15_segment(meta)
    return _SOI + b"".join(body[:insert_at]) + seg + b"".join(body[insert_at:]) + tail


def rewrite_dfljpg_meta(path, meta, merge=True):
    """把 APP15 换成新的 dict，**写在头部**（引擎唯一读得到的位置），返回字节数。

    merge=True（默认）：先拿已有的 legacy + 头部 dict 当底，再用 meta 覆盖 ——
    调用方没提到的键（face_type / landmarks / seg_ie_polys / pitch …）全部保留；
    merge=False：只用 meta 重写。

    历史坑，全在这儿记着：
    1) APP15 曾经追加在 EOI 之前 —— DFLJPG.load_raw 遇到 SOS 就跳到 EOI，引擎压根
       读不到：整个 aligned 目录的遮罩一直是死的，_mask_worker「补遮罩」写进去的同样
       读不到；
    2) 白名单漏了 _codecs.encode，protocol≤2 的 bytes 一律解不开；
    3) 更早的写法插在 SOI 之后并重排段序 —— 遇到 JPEG 填充字节(0xFF 0xFF)会把段长
       读错，扫描数据被复制一遍，文件变大且无法解码；
    4) 先算 pickle 长度再替换 numpy._core→numpy.core，长度差 1 字节，解码器读坏。
    """
    data = Path(path).read_bytes()
    if not data.startswith(_SOI):
        raise ValueError("not a jpeg")
    out_meta = {}
    if merge:
        legacy, head = find_app15_legacy(data), find_app15_header(data)
        for d in (legacy, head):                        # 头部优先
            if isinstance(d, dict):
                out_meta.update(d)
    out_meta.update(dict(meta or {}))
    out = write_app15_head(data, out_meta)
    Path(path).write_bytes(out)
    return len(out)
