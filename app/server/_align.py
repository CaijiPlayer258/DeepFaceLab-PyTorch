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
结果 —— 所有拟合都在单位空间里做，padding 只算一次。
"""
from __future__ import annotations

import io
import pickle
import struct

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

# DFL 自己的 33 点单位模板（facelib/LandmarksProcessor.landmarks_2D_new）：
# 索引 17..48 + 54，其余点它没有，需要从经验模板补。
_TPL_2D_NEW = np.array([
    [0.000213256, 0.106454], [0.0752622, 0.038915], [0.18113, 0.0187482],
    [0.29077, 0.0344891], [0.393397, 0.0773906], [0.586856, 0.0773906],
    [0.689483, 0.0344891], [0.799124, 0.0187482], [0.904991, 0.038915],
    [0.98004, 0.106454], [0.490127, 0.203352], [0.490127, 0.307009],
    [0.490127, 0.409805], [0.490127, 0.515625], [0.36688, 0.587326],
    [0.426036, 0.609345], [0.490127, 0.628106], [0.554217, 0.609345],
    [0.613373, 0.587326], [0.121737, 0.216423], [0.187122, 0.178758],
    [0.265825, 0.179852], [0.334606, 0.231733], [0.260918, 0.245099],
    [0.182743, 0.244077], [0.645647, 0.231733], [0.714428, 0.179852],
    [0.793132, 0.178758], [0.858516, 0.216423], [0.79751, 0.244077],
    [0.719335, 0.245099], [0.254149, 0.780233], [0.726104, 0.780233],
], dtype=np.float64)

_REAL_IDX = list(range(17, 49)) + [54]


def _build_unit_68():
    """68 点单位模板：DFL 认得的 33 点用 DFL 原值，缺的点用经验模板补。

    经验模板来自真实对齐脸，它和 DFL 的 33 点模板之间差一个相似变换，
    先用这 33 点把两者对齐，再把剩下 35 个点搬进 DFL 的单位空间。
    """
    emp_unit = transform_points(
        TEMPLATE_68, _affine_from_unit_square(_IDENTITY, SIZE), invert=True)
    T = umeyama(emp_unit[_REAL_IDX], _TPL_2D_NEW, True)
    mapped = transform_points(emp_unit, T[0:2])
    mapped[_REAL_IDX] = _TPL_2D_NEW
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


def landmarks68_from_5(kps5):
    """检测到的 5 点 → 原图坐标系的 68 点（其余点由模板相似变换带出来）。"""
    mat_unit = umeyama(_ordered_5(kps5), template_5(), True)[0:2]
    return transform_points(UNIT_68, mat_unit, invert=True).astype(np.float64)


def align_face(img_bgr, kps5, box=None, size=SIZE):
    """裁一张对齐脸。返回 (对齐图, 元数据 dict)；失败返回 (None, None)。"""
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

    lm68_img = landmarks68_from_5(kps5)
    lm68_aligned = transform_points(lm68_img, mat).astype(np.float64)

    h, w = img_bgr.shape[:2]
    if box is None:
        box = [float(lm68_img[:, 0].min()), float(lm68_img[:, 1].min()),
               float(lm68_img[:, 0].max()), float(lm68_img[:, 1].max())]
    rect = [int(max(0, box[0])), int(max(0, box[1])),
            int(min(w, box[2])), int(min(h, box[3]))]

    meta = {
        "face_type": "whole_face",
        "landmarks": np.round(lm68_aligned, 4).tolist(),
        "source_landmarks": np.round(lm68_img, 4).tolist(),
        "source_rect": rect,
        "image_to_face_mat": np.asarray(mat, dtype=np.float64).tolist(),
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


def write_dfljpg(path, img_bgr, meta: dict, quality: int = 95):
    import cv2
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
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
