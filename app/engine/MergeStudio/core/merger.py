"""
Face swap merge engine.
Transplanted from merger/MergeMasked.py - preserving exact flow.
"""
from pathlib import Path
import os
import cv2
import numpy as np
import traceback
from facelib import FaceType, LandmarksProcessor
from core import imagelib

_model_input_size = 256  # Overridden by model_loader

# ----------------------------------------------------------------------------
# ROI（只算脸所在的那一块）
#
# 一帧 1080p 的 float32 是 25MB，MergeMaskedFace 原来要整帧来回搬十来趟
# （两次合成、两次 clip、u8 转换、遮罩阈值/包围盒…），实测这些「纯内存带宽」
# 的算子占单帧 ~150ms。而遮罩支撑只占整帧 ~2%（包围盒 ~3%），遮罩外一个像素
# 都不会变 —— 所以整块整块地算是在白搬。
#
# 开关：环境变量 DFL_MERGE_ROI=0 退回整帧版（逐位对拍/出问题时逃生）。
# 逐位一致的依据见 _compose_roi_standard 的注释；对拍脚本见
# app/tools/roi_parity_accept.py。
# ----------------------------------------------------------------------------
_ROI_ENV = "DFL_MERGE_ROI"


def _roi_enabled():
    v = os.environ.get(_ROI_ENV)
    if v is None:
        return True
    return str(v).strip().lower() not in ("0", "false", "off", "no", "")


def _quad_box(mat, size, shape, margin=8):
    """S×S 画布落在原帧里的那块四边形 → 外接矩形，外扩 margin、裁到帧内。

    mat 是 get_transform_mat 的结果（**帧 → 画布**）：warpAffine(img_bgr, mat, (S,S))
    对每个画布像素 (u,v) 取帧上的 inv(mat)·[u,v]，所以画布四角在帧上的位置要用
    inv(mat) 求。返回 (x0, y0, x1, y1) 或 None（脸整个在帧外）。

    margin 是给三次插值留的余量：插值要读样本点周围 ±2px，画布边上的 dst 像素会
    读到四边形外面 —— 整帧算法里那些像素是真的，切片里也必须是。
    """
    h, w = shape[:2]
    m = cv2.invertAffineTransform(np.asarray(mat, np.float64))
    c = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], np.float64)
    xs = m[0, 0] * c[:, 0] + m[0, 1] * c[:, 1] + m[0, 2]
    ys = m[1, 0] * c[:, 0] + m[1, 1] * c[:, 1] + m[1, 2]
    x0 = int(np.floor(float(xs.min()))) - margin
    x1 = int(np.ceil(float(xs.max()))) + margin
    y0 = int(np.floor(float(ys.min()))) - margin
    y1 = int(np.ceil(float(ys.max()))) + margin
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


#: ROI 快路的计数器（排障用；正常跑就是几个整数加法，多线程下数值可能偏小 ——
#: 它只用来回答「这一帧到底走没走快点」，不做记账）
ROI_STATS = {"frames": 0, "work_none": 0, "big": 0,
             "roi_taken": 0, "roi_false": 0}


def _box_slice(box):
    """"(x0, y0, x1, y1)" → 行列切片 (slice_y, slice_x)。"""
    return (slice(box[1], box[3]), slice(box[0], box[2]))


def _union_box(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _roi_gate(cfg, frame_img):
    """这一帧 / 这组参数能不能走 ROI。不满足就老路（整帧）跑。"""
    if not _roi_enabled():
        return False
    if getattr(frame_img, "dtype", None) != np.uint8:
        return False
    if "raw" in str(cfg.mode):                  # raw-* 走的是另一套输出
        return False
    # 这三个会**改写 img_bgr / out_img 的整帧内容**（medianBlur、整帧重采样、
    # 整图调色），ROI 语义不成立（它们读的就是整帧），老路跑。
    if cfg.image_denoise_power != 0 or cfg.bicubic_degrade_power != 0             or cfg.color_degrade_power != 0:
        return False
    return True


def _build_debug_vis(prd_face_bgr, dst_face_bgr, wrk_face_mask_a_0,
                     prd_face_mask_a_0=None, prd_face_dst_mask_a_0=None, output_size=256):
    """Build debug visualization: 2 rows x 3 cols of output_size images."""
    def colorize_mask(mk):
        m_8u = (np.clip(mk, 0, 1) * 255).astype(np.uint8)
        if m_8u.ndim == 2:
            return cv2.applyColorMap(m_8u, cv2.COLORMAP_VIRIDIS)
        return cv2.applyColorMap(m_8u[:, :, 0], cv2.COLORMAP_VIRIDIS) if m_8u.ndim == 3 else m_8u

    vis = np.zeros((output_size * 2 + 30, output_size * 3 + 30, 3), dtype=np.uint8)
    vis[:] = 32
    gap = 10
    os = output_size
    # Row 1: masks (use model/XSeg masks, fallback to wrk)
    mask_items = [
        ('pred', prd_face_mask_a_0 if prd_face_mask_a_0 is not None else wrk_face_mask_a_0),
        ('dst', prd_face_dst_mask_a_0 if prd_face_dst_mask_a_0 is not None else wrk_face_mask_a_0),
        ('prd*dst', wrk_face_mask_a_0),
    ]
    for i, (name, mk) in enumerate(mask_items):
        x = gap + i * (os + gap)
        if mk is not None and hasattr(mk, 'shape') and mk.ndim >= 2 and mk.size > 0:
            mk_2d = mk.squeeze() if mk.ndim > 2 else mk
            if mk_2d.shape[:2] != (os, os):
                mk_2d = cv2.resize(mk_2d, (os, os), interpolation=cv2.INTER_CUBIC)
            if mk_2d.ndim == 2 and mk_2d.size > 0:
                colored = colorize_mask(mk_2d)
                vis[gap:gap + os, x:x + os] = colored
                cv2.putText(vis, name, (x + 2, gap + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    # Row 2: RGB images
    y2 = gap + os + gap
    prd_rgb = cv2.resize(prd_face_bgr, (os, os)) if prd_face_bgr.shape[:2] != (os, os) else prd_face_bgr[:os, :os]
    dst_rgb = cv2.resize(dst_face_bgr, (os, os)) if dst_face_bgr.shape[:2] != (os, os) else dst_face_bgr[:os, :os]
    if wrk_face_mask_a_0 is not None:
        mk_2d = wrk_face_mask_a_0.squeeze()
        if mk_2d.shape[:2] != (os, os):
            mk_2d = cv2.resize(mk_2d, (os, os), interpolation=cv2.INTER_CUBIC)
        mk_3ch = np.stack([mk_2d] * 3, axis=-1)
    else:
        mk_3ch = np.ones((os, os, 3))
    merged = prd_rgb * mk_3ch + dst_rgb * (1 - mk_3ch)
    for j, (name, img) in enumerate([('pred RGB', prd_rgb), ('dst RGB', dst_rgb), ('merged', merged)]):
        x = gap + j * (os + gap)
        vis[y2:y2 + os, x:x + os] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.putText(vis, name, (x + 2, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return vis


#: 单脸调试六宫格的格子顺序：3 列 × 2 行、行优先（前端按这个顺序排版）。
#: 第 1 行 = 三张遮罩，第 2 行 = 三张脸图 —— 和 _build_debug_vis() 的老排版一致，
#: 只是这里把六格拆开单独发，前端才能自己排版、点开放大。
DEBUG_PANELS = (
    ("mask-prd", "mask(dfm-pred)"),
    ("mask-dst", "mask(dst)"),
    ("mask-both", "merged(mask(dfm-pred), mask(dst))"),
    ("dfm-pred", "dfm-pred"),
    ("dst-original", "dst-original"),
    ("merged", "merged"),
)


def build_debug_panels(capture, size=256):
    """MergeMaskedFace(capture=…) 攒下来的中间量 → 六张**等尺寸**的 BGR 图。

    返回 [(key, label, img), …]，顺序 = DEBUG_PANELS。
    缺哪一项就给一张同尺寸黑图：六格永远都有图，前端不用判空、也不用占位。
    """
    cap = capture or {}
    side = max(64, int(size or 256))

    def _fit_mask(mk):
        if mk is None:
            return None
        m = np.asarray(mk, np.float32)
        while m.ndim > 2:
            m = m[..., 0]
        m = np.clip(m, 0, 1)
        if m.shape[:2] != (side, side):
            m = cv2.resize(m, (side, side), interpolation=cv2.INTER_CUBIC)
        return m

    def _mask_img(mk):
        m = _fit_mask(mk)
        if m is None:
            return np.zeros((side, side, 3), np.uint8)
        g = np.clip(m * 255.0, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    def _face_img(im):
        if im is None:
            return np.zeros((side, side, 3), np.uint8)
        a = np.asarray(im, np.float32)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.shape[-1] > 3:
            a = a[:, :, :3]
        if a.shape[-1] == 3 and float(a.max(initial=0.0)) > 1.5:   # 0-255 的旧图
            a = a / 255.0
        a = np.clip(a, 0, 1)
        if a.shape[:2] != (side, side):
            a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
        return np.clip(a * 255.0, 0, 255).astype(np.uint8)

    prd_m = _fit_mask(cap.get("prd_face_mask"))
    dst_m = _fit_mask(cap.get("dst_face_mask"))
    eff_m = _fit_mask(cap.get("effective_mask"))
    both = (prd_m * dst_m) if (prd_m is not None and dst_m is not None) else None

    merged = cap.get("merged_face")
    if merged is None:
        # 拿不到「真合成图」就按老公式自己混一张：prd 用**实际生效**的遮罩盖回 dst
        prd_f, dst_f = cap.get("prd_face_bgr"), cap.get("dst_face_bgr")
        if prd_f is not None and dst_f is not None:
            m3 = eff_m if eff_m is not None else np.ones((side, side), np.float32)
            mask3 = np.stack([m3] * 3, axis=-1)
            merged = prd_f * mask3 + dst_f * (1.0 - mask3)

    imgs = {
        "mask-prd": _mask_img(prd_m),
        "mask-dst": _mask_img(dst_m),
        "mask-both": _mask_img(both),
        "dfm-pred": _face_img(cap.get("prd_face_bgr")),
        "dst-original": _face_img(cap.get("dst_face_bgr")),
        "merged": _face_img(merged),
    }
    for key, _label in DEBUG_PANELS:
        imgs.setdefault(key, np.zeros((side, side, 3), np.uint8))
    return [(key, label, imgs[key]) for key, label in DEBUG_PANELS]


def _ct_table(mask_area):
    """色彩迁移实现表（key = color_transfer_mode）。整帧版与 ROI 版共用。"""
    return {
        1: lambda s, t: imagelib.reinhard_color_transfer(s, t, target_mask=mask_area, source_mask=mask_area),
        2: lambda s, t: imagelib.linear_color_transfer(s, t),
        3: lambda s, t: imagelib.color_transfer_mkl(s, t),
        4: lambda s, t: imagelib.color_transfer_mkl(s * mask_area, t * mask_area),
        5: lambda s, t: imagelib.color_transfer_idt(s, t),
        6: lambda s, t: imagelib.color_transfer_idt(s * mask_area, t * mask_area),
        7: lambda s, t: imagelib.color_transfer_sot(s * mask_area, t * mask_area, steps=10, batch_size=30),
        8: lambda s, t: imagelib.color_transfer_mix(s * mask_area, t * mask_area),
        # 9/10 与训练预览侧对齐（config.ctm_dict 同步）
        9: lambda s, t: imagelib.match_tone_np(s, t, src_mask=mask_area, trg_mask=mask_area),
        10: lambda s, t: imagelib.color_transfer_lut(s, t),
    }  # ct_functions


def _compose_roi_standard(frame_img, img_bgr, img_face_mask_full, prd_face_bgr,
                          dst_face_bgr, wrk_face_mask_a, face_mat, face_output_mat,
                          img_size, output_size, cfg, work, capture):
    """标准模式（overlay / hist-match / seamless / seamless-hist-match）的 ROI 实现。

    语义与 MergeMaskedFace 里那段整帧代码**逐位相同**，只算 work 那一块
    （= 脸画布四边形 + 8px，实测占整帧 4~8%）。逐位一致的依据：

    1. 遮罩支撑外 img_face_mask_full 恒为 0，合成式就退化成 img_bgr*1 + x*0；
       而 u8 → f32(/255) → *255 → u8 这一趟对 0..255 全部 256 个值**逐位无损**
       （探针实测 256/256 相同，整帧 74.6M 像素 0 不一致），所以遮罩外直接沿用
       原帧像素，与整帧算出来的结果一模一样。
    2. 所有切片都是同一块 float32 内存的视图：逐元素算子（乘加 / clip / 阈值）
       在切片上算与整帧算是同 dtype、同运算顺序 → 逐位相同。
    3. seamlessClone 仍然吃整帧输入、吐整帧输出：它的 Poisson 解只读遮罩包围盒
       那一块、盒外输出恒等于 dst（探针实测盒外 max|diff| = 0，1024 个像素级
       一致），但**输入尺寸会改边界条件**（裁成小图实测差 3~7 LSB、10% 像素），
       所以不裁 —— 省掉的是它那三张整帧 uint8 输入/输出的生成（约 12ms → 2ms）。
    4. 两个 warpAffine（遮罩反贴 / 脸反贴）仍按整帧坐标做：把 ROI 平移折进 M 的
       平移项会让坐标的定点舍入（1/32 px）变位，实测 44%~79% 的像素差 1e-5 量级
       —— 只有 3.7ms，不值得为它冒逐位不一致的险。

    返回 (out_img, out_merging_mask_a, final_u8, final_mask_u8)。
    返回 False 表示「遮罩支撑太小」：整帧版在这一点上不会给 out_img 赋值，
    会抛 UnboundLocalError 被外层 except 兜住 → 原样返回帧 + 全零遮罩。
    """
    H, W = img_size[1], img_size[0]
    wy, wx = work[1], work[0]
    # img_face_mask_full 是 (H, W, 1)；合成式要那个多出来的一维来广播，
    # 遮罩自己的运算（阈值/包围盒/u8）用 2D 视图 —— 都是同一块内存的视图。
    mask2 = img_face_mask_full[..., 0]
    m_w2 = mask2[_box_slice(work)]
    m_w = m_w2[..., None]
    ib_w = img_bgr[_box_slice(work)]                  # 视图

    # ---- ① 区域判定（原 L343-347；work 外遮罩恒为 0，所以在 work 里扫等价）----
    ge01 = m_w2 >= 0.1
    if not bool(ge01.any()):
        return False
    rows = np.flatnonzero(ge01.any(axis=1))
    cols = np.flatnonzero(ge01.any(axis=0))
    if min(int(rows[-1]) - int(rows[0]), int(cols[-1]) - int(cols[0])) < 4:
        return False
    del ge01

    # ---- ② 遮罩支撑的包围盒（真正会改像素的地方）----
    gt0 = m_w2 > 0
    r0 = np.flatnonzero(gt0.any(axis=1))
    c0 = np.flatnonzero(gt0.any(axis=0))
    del gt0
    ry0, ry1 = wy + int(r0[0]), wy + int(r0[-1]) + 1
    rx0, rx1 = wx + int(c0[0]), wx + int(c0[-1]) + 1
    m_r2 = mask2[ry0:ry1, rx0:rx1]
    m_r = m_r2[..., None]
    ib_r = img_bgr[ry0:ry1, rx0:rx1]
    r_loc = (slice(ry0 - wy, ry1 - wy), slice(rx0 - wx, rx1 - wx))

    mask_area = wrk_face_mask_a.copy()
    mask_area[mask_area > 0] = 1.0

    # ---- ③ 色彩迁移（output_size 画布，与整帧版同一份代码）----
    ct_functions = _ct_table(mask_area)
    fn = ct_functions.get(cfg.color_transfer_mode)
    if fn:
        try:
            prd_face_bgr = np.clip(fn(prd_face_bgr, dst_face_bgr), 0, 1)
        except Exception:
            pass

    # Hist-match（画布上的活，整脸）
    if cfg.mode == 'hist-match':
        hist_mask = np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
        if cfg.masked_hist_match:
            hist_mask *= mask_area
        white = (1.0 - hist_mask) * np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
        hm1 = prd_face_bgr * hist_mask + white
        hm1[hm1 > 1.0] = 1.0
        hm2 = dst_face_bgr * hist_mask + white
        hm2[hm2 > 1.0] = 1.0
        prd_face_bgr = imagelib.color_hist_match(hm1, hm2, cfg.hist_match_threshold).astype(np.float32)

    # ---- ④ seamless 用的二值遮罩（只在遮罩包围盒里做；盒外恒 0）----
    seam_r = None
    if 'seamless' in cfg.mode:
        for i in range(1, 10):
            a = m_r2 > i / 10.0
            if not bool(a.any()):
                continue
            seam = m_r2.copy()
            seam[a] = 1.0
            seam[seam <= i / 10.0] = 0.0
            seam_r = seam
            break

    # ---- ⑤ 脸反贴（整帧坐标，见上面第 4 条）----
    warp_full = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr),
                               flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    w_wrk = np.clip(warp_full[_box_slice(work)], 0, 1)
    del warp_full

    # ---- ⑥ seamlessClone：整帧输入/输出，只把「会被读到的」填对 ----
    # src 的遮罩盒外 == img_bgr（out_img 在盒外就是原帧），dst 就是原帧本身
    # （原代码 frame_img.astype(np.uint8) 是一份内容相同的拷贝）。
    cloned = False
    sc_r = None
    if 'seamless' in cfg.mode and seam_r is not None:
        try:
            seam_u8_r = (seam_r * 255).astype(np.uint8)
            l, t, w, h = cv2.boundingRect(seam_u8_r)
            # center 与整帧版逐字节相同：整数平移可交换（int(l+w/2)+rx0 == int((l+rx0)+w/2)）
            center = (int(l + w / 2) + rx0, int(t + h / 2) + ry0)
            mask_u8 = np.zeros((H, W), np.uint8)
            mask_u8[ry0:ry1, rx0:rx1] = seam_u8_r
            src_u8 = frame_img.copy()
            src_u8[_box_slice(work)] = (w_wrk * 255).astype(np.uint8)
            sc = cv2.seamlessClone(src_u8, frame_img, mask_u8, center, cv2.NORMAL_CLONE)
            sc_r = sc[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0
            cloned = True
        except Exception:
            cloned = False

    # ---- ⑦ 第一次合成（原 L416）：work 整块都要对（w6 会按画布读它）----
    if cloned:
        cur = ib_w * (1 - m_w) + w_wrk * m_w        # 遮罩外 == img_bgr，与整帧版相同
        cur[r_loc] = ib_r * (1 - m_r) + sc_r * m_r  # 遮罩内换成 seamless 的结果
    else:
        cur = ib_w * (1 - m_w) + w_wrk * m_w
    out_img = np.empty_like(img_bgr)                # 整帧形状、惰性页；只填 work
    out_img[_box_slice(work)] = cur

    # ---- ⑧ 后处理（原 L419-478）：w6 按画布读 out_img ----
    # 注：image_denoise_power / bicubic_degrade_power / color_degrade_power 三段会改写
    # 整帧 img_bgr / out_img，ROI 语义不成立 —— _roi_gate 已经把这三项非 0 的情况挡在
    # ROI 路径之外（那时走整帧版），所以这里不用重复。
    mp = cfg.motion_blur_power / 100.0
    needs_pp = ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
               cfg.mode == 'seamless-hist-match' or mp != 0 or \
               cfg.blursharpen_amount != 0 or cfg.image_denoise_power != 0 or \
               cfg.bicubic_degrade_power != 0

    if needs_pp:
        out_face_bgr_pp = cv2.warpAffine(out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)

        if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
            try:
                fn_pp = ct_functions.get(cfg.color_transfer_mode)
                if fn_pp:
                    out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
            except Exception:
                pass
                try:
                    out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
                except Exception:
                    pass

        if cfg.mode == 'seamless-hist-match':
            out_face_bgr_pp = imagelib.color_hist_match(out_face_bgr_pp, dst_face_bgr, cfg.hist_match_threshold)

        if mp != 0:
            k = int(mp * 100)
            if k >= 1:
                k = np.clip(k + 1, 2, 50)
                kernel = np.zeros((k, k))
                kernel[:, k // 2] = 1.0 / k
                out_face_bgr_pp = cv2.filter2D(out_face_bgr_pp, -1, kernel)

        if cfg.blursharpen_amount != 0:
            amt = cfg.blursharpen_amount / 100.0
            if cfg.sharpen_mode == 1:
                blurred = cv2.blur(out_face_bgr_pp, (3, 3))
            else:
                blurred = cv2.GaussianBlur(out_face_bgr_pp, (0, 0), 1.0)
            out_face_bgr_pp = cv2.addWeighted(out_face_bgr_pp, 1.0 + amt, blurred, -amt, 0)
            out_face_bgr_pp = np.clip(out_face_bgr_pp, 0, 1)

        new_out = cv2.warpAffine(out_face_bgr_pp, face_mat, img_size, np.empty_like(img_bgr),
                                 flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        # 原 L478 等价：有 capture 时整块 work（capture 的 merged_face 会按画布读它），
        # 否则只算遮罩盒（盒外一定等于 img_bgr）。
        b_box = work if capture is not None else (rx0, ry0, rx1, ry1)
        bs = _box_slice(b_box)
        out_img[bs] = np.clip(img_bgr[bs] * (1 - img_face_mask_full[bs])
                              + new_out[bs] * img_face_mask_full[bs], 0, 1)

    # ---- ⑨ 出图（原 L514/L515）----
    final = frame_img.copy()          # 遮罩盒外：整帧版算出来就是原帧像素（见第 1 条）
    final[ry0:ry1, rx0:rx1] = np.clip(out_img[ry0:ry1, rx0:rx1] * 255, 0, 255).astype(np.uint8)
    final_mask = np.zeros((H, W), np.uint8)
    final_mask[ry0:ry1, rx0:rx1] = np.clip(m_r2 * 255, 0, 255).astype(np.uint8)
    # prd_face_bgr 也要带回去：色彩迁移 / hist-match 是就地把画布换了一张，
    # 整帧版是同一个变量在函数体里被覆盖，capture 的「dfm-pred」格与 debug 视图都看它。
    return out_img, img_face_mask_full, final, final_mask, prd_face_bgr


def MergeMaskedFace(frame_img, face_landmarks, cfg, predictor_func=None,
                    face_enhancer_func=None, xseg_256_extract_func=None,
                    capture=None, input_size=None):
    """
    Merge a single face. Matches original MergeMasked.py flow exactly.
    Everything works in 0-1 float range. Predictor returns [face, prd_mask, dst_mask].

    capture（可选）= 一个 dict：把这次合成的中间量塞进去（单脸六宫格调试用）。
    它**只写不读**，不影响任何一个像素 —— 不传的时候行为和老版本逐字节一致。
    键：dst_face_bgr / prd_face_bgr / prd_face_mask / dst_face_mask /
        effective_mask（真正贴回去用的那张，erode+blur 之后）/ merged_face。
    """
    try:
        if isinstance(cfg.face_type, str):
            ft_map = {'half_face': FaceType.HALF, 'midfull_face': FaceType.MID_FULL,
                      'full_face': FaceType.FULL, 'whole_face': FaceType.WHOLE_FACE, 'head': FaceType.HEAD}
            cfg.face_type = ft_map.get(cfg.face_type, FaceType.WHOLE_FACE)

        # input_size 显式传进来就以它为准：批处理里同一进程内多帧并行，
        # 模块级 _model_input_size 是共享可变状态（两个线程先后写它，后写的那份会让
        # 另一帧按错的边长 warp —— 多模型、输入尺寸不同时必现）。不传 = 老行为。
        input_size = _model_input_size if input_size is None else int(input_size)
        output_size = input_size
        if cfg.super_resolution_power != 0:
            output_size *= 4

        # Normalize frame to 0-1
        use_roi = _roi_gate(cfg, frame_img)
        if use_roi:
            # 先只要一块**整帧形状**的空缓冲：矩阵是按整帧坐标写的，取样原点必须
            # 是帧原点，所以缓冲不能缩成小图；但只填「脸画布 + 余量」那一块，
            # 没碰过的页根本不会被换进来（np.empty 是惰性的）。
            img_bgr = np.empty(frame_img.shape, np.float32)
        else:
            img_bgr = frame_img.astype(np.float32) / 255.0
        img_size = (frame_img.shape[1], frame_img.shape[0])

        # Hull mask on full frame
        img_face_mask_a = LandmarksProcessor.get_image_hull_mask(img_bgr.shape, face_landmarks)

        # Get transform mats
        face_mat = LandmarksProcessor.get_transform_mat(face_landmarks, output_size, face_type=cfg.face_type)
        scale = 1.0 + 0.01 * cfg.output_face_scale
        face_output_mat = LandmarksProcessor.get_transform_mat(face_landmarks, output_size, face_type=cfg.face_type, scale=scale)

        # ROI 工作区 = 两张画布（seamless 前/后）在原帧里的四边形并集。
        # 这块之外 img_bgr 一个字节都不会被读到：所有整帧算子都只作用于它。
        work = None
        if use_roi:
            work = _union_box(_quad_box(face_mat, output_size, img_bgr.shape),
                              _quad_box(face_output_mat, output_size, img_bgr.shape))
            if work is None:
                ROI_STATS["work_none"] += 1
            if work is None or (work[2] - work[0]) * (work[3] - work[1]) * 5 > \
                    img_bgr.shape[0] * img_bgr.shape[1] * 3:
                # 脸占了大半帧（或整张在帧外）：没有收益，退回整帧语义
                use_roi = False
                work = None
                ROI_STATS["big"] += 1
                img_bgr = frame_img.astype(np.float32) / 255.0
            else:
                img_bgr[_box_slice(work)] = frame_img[_box_slice(work)].astype(np.float32) / 255.0

        # Extract face + mask
        dst_face_bgr = cv2.warpAffine(img_bgr, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
        dst_face_bgr = np.clip(dst_face_bgr, 0, 1)

        dst_face_mask_a_0 = cv2.warpAffine(img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
        dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

        # Predict
        predictor_input_bgr = cv2.resize(dst_face_bgr, (input_size, input_size))
        if predictor_func is not None:
            result = predictor_func(predictor_input_bgr)
            if isinstance(result, tuple):
                predicted = list(result)
            else:
                predicted = [result, None, None]
        else:
            predicted = [dst_face_bgr, None, None]

        prd_face_bgr = np.clip(predicted[0], 0, 1.0) if predicted[0] is not None else dst_face_bgr
        prd_face_mask_a_0 = np.clip(predicted[1], 0, 1.0) if len(predicted) > 1 and predicted[1] is not None else None
        prd_face_dst_mask_a_0 = np.clip(predicted[2], 0, 1.0) if len(predicted) > 2 and predicted[2] is not None else None

        # Super resolution
        if cfg.super_resolution_power != 0 and face_enhancer_func is not None:
            enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
            mod = cfg.super_resolution_power / 100.0
            prd_face_bgr = cv2.resize(prd_face_bgr, (output_size, output_size)) * (1.0 - mod) + enhanced * mod
            prd_face_bgr = np.clip(prd_face_bgr, 0, 1)
            if prd_face_mask_a_0 is not None:
                prd_face_mask_a_0 = cv2.resize(prd_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            if prd_face_dst_mask_a_0 is not None:
                prd_face_dst_mask_a_0 = cv2.resize(prd_face_dst_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        # When seg_mode is XSeg/XSegLite, replace model masks with XSeg results
        seg = getattr(cfg, 'seg_mode', 'model')
        if seg in ('xseg', 'xseglite') and xseg_256_extract_func is not None:
            prd_face_xseg = cv2.resize(prd_face_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            xseg_result_prd = xseg_256_extract_func(prd_face_xseg)
            X_prd = cv2.resize(xseg_result_prd, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            xseg_mat = LandmarksProcessor.get_transform_mat(face_landmarks, 256, face_type=cfg.face_type)
            dst_xseg = cv2.warpAffine(img_bgr, xseg_mat, (256, 256), flags=cv2.INTER_CUBIC)
            xseg_result_dst = xseg_256_extract_func(dst_xseg)
            X_dst = cv2.resize(xseg_result_dst, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            #print(f"[Mask] XSeg({seg}): prd mean={X_prd.mean():.3f} max={X_prd.max():.3f}, dst mean={X_dst.mean():.3f} max={X_dst.max():.3f}")
            #print(f"[Mask] XSeg dst_xseg: shape={dst_xseg.shape} range=[{dst_xseg.min():.3f},{dst_xseg.max():.3f}] mean={dst_xseg.mean():.3f}")
            #print(f"[Mask] XSeg result: prd range=[{xseg_result_prd.min():.3f},{xseg_result_prd.max():.3f}] mean={xseg_result_prd.mean():.3f}, "
            #      f"dst range=[{xseg_result_dst.min():.3f},{xseg_result_dst.max():.3f}] mean={xseg_result_dst.mean():.3f}")
            # Override model masks with XSeg masks
            prd_face_mask_a_0 = np.clip(X_prd, 0, 1)
            prd_face_dst_mask_a_0 = np.clip(X_dst, 0, 1)
        elif seg == 'model' and prd_face_mask_a_0 is not None and prd_face_dst_mask_a_0 is not None:
            pass  # DFM masks already set above

        # 单脸调试：两张模型遮罩 + 对齐后的原脸（就在这儿留，之后 wrk 遮罩会把它们盖掉概念上）
        if capture is not None:
            capture["dst_face_bgr"] = dst_face_bgr
            capture["prd_face_mask"] = prd_face_mask_a_0
            capture["dst_face_mask"] = prd_face_dst_mask_a_0

        # Build working mask
        if cfg.mask_mode == 0:  # full
            wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 1:  # dst
            wrk_face_mask_a_0 = cv2.resize(dst_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        elif cfg.mask_mode == 2:  # learned-prd
            wrk_face_mask_a_0 = prd_face_mask_a_0 if prd_face_mask_a_0 is not None else np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 3:  # learned-dst
            wrk_face_mask_a_0 = prd_face_dst_mask_a_0 if prd_face_dst_mask_a_0 is not None else np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 4:  # learned-prd*learned-dst
            if prd_face_mask_a_0 is not None and prd_face_dst_mask_a_0 is not None:
                wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0
            else:
                wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 5:  # learned-prd+learned-dst
            mask = np.zeros_like(dst_face_mask_a_0)
            if prd_face_mask_a_0 is not None: mask += prd_face_mask_a_0
            if prd_face_dst_mask_a_0 is not None: mask += prd_face_dst_mask_a_0
            wrk_face_mask_a_0 = np.clip(mask, 0, 1)
        elif cfg.mask_mode >= 6:  # XSeg modes (old style, kept for compat)
            if cfg.mask_mode in (6, 8, 9) and xseg_256_extract_func is not None:
                prd_face_xseg = cv2.resize(prd_face_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
                prd_xseg_mask = xseg_256_extract_func(prd_face_xseg)
                X_prd = cv2.resize(prd_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            else:
                X_prd = None
            if cfg.mask_mode >= 7 and xseg_256_extract_func is not None:
                xseg_mat = LandmarksProcessor.get_transform_mat(face_landmarks, 256, face_type=cfg.face_type)
                dst_xseg = cv2.warpAffine(img_bgr, xseg_mat, (256, 256), flags=cv2.INTER_CUBIC)
                dst_xseg_mask = xseg_256_extract_func(dst_xseg)
                X_dst = cv2.resize(dst_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            else:
                X_dst = None

            if cfg.mask_mode == 6:
                wrk_face_mask_a_0 = X_prd if X_prd is not None else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 7:
                wrk_face_mask_a_0 = X_dst if X_dst is not None else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 8:
                wrk_face_mask_a_0 = X_prd * X_dst if (X_prd is not None and X_dst is not None) else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 9:
                mask = np.ones_like(dst_face_mask_a_0)
                if prd_face_mask_a_0 is not None: mask *= prd_face_mask_a_0
                if prd_face_dst_mask_a_0 is not None: mask *= prd_face_dst_mask_a_0
                if X_prd is not None: mask *= X_prd
                if X_dst is not None: mask *= X_dst
                wrk_face_mask_a_0 = mask
        else:
            wrk_face_mask_a_0 = dst_face_mask_a_0

        #print(f"[Mask] Final mask (seg={seg}, mode={cfg.mask_mode}): mean={wrk_face_mask_a_0.mean():.3f}, max={wrk_face_mask_a_0.max():.3f}")
        wrk_face_mask_a_0[wrk_face_mask_a_0 < (1.0 / 255.0)] = 0.0

        # Process mask (erode/blur) in padded space
        if 'raw' not in cfg.mode:
            wrk_face_mask_a_0 = np.pad(wrk_face_mask_a_0, input_size)
            ero = cfg.erode_mask_modifier
            blur = cfg.blur_mask_modifier
            if ero > 0:
                wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero)), iterations=1)
            elif ero < 0:
                wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-ero, -ero)), iterations=1)
            clip_size = input_size + blur // 2
            wrk_face_mask_a_0[:clip_size, :] = 0
            wrk_face_mask_a_0[-clip_size:, :] = 0
            wrk_face_mask_a_0[:, :clip_size] = 0
            wrk_face_mask_a_0[:, -clip_size:] = 0
            if blur > 0:
                blur = blur + (1 - blur % 2)
                wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur), 0)
            wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size, input_size:-input_size]
            wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

        # Warp mask to full frame and resize to output_size
        img_face_mask_full = cv2.warpAffine(
            wrk_face_mask_a_0, face_output_mat, img_size,
            np.zeros(img_bgr.shape[:2], dtype=np.float32),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
        )[..., None]
        if work is not None:
            # ROI：只在 work 里 clip/清零。work 外 warpAffine 的 dst 是 zeros、
            # 取样点落在画布外 → 那些像素本来就是 0，clip 和 <1/255 清零都是恒等。
            _mb = img_face_mask_full[_box_slice(work)]
            np.clip(_mb, 0, 1, out=_mb)
            _mb[_mb < (1.0 / 255.0)] = 0.0
            del _mb
        else:
            img_face_mask_full = np.clip(img_face_mask_full, 0, 1)
            img_face_mask_full[img_face_mask_full < (1.0 / 255.0)] = 0.0

        if wrk_face_mask_a_0.shape[0] != output_size:
            wrk_face_mask_a_0 = cv2.resize(wrk_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        wrk_face_mask_a = wrk_face_mask_a_0[..., None]
        # 真正贴回去用的那一张（erode/blur/裁剪之后）—— 六宫格第 3、6 格看的就是它
        if capture is not None:
            capture["effective_mask"] = wrk_face_mask_a_0

        _final_u8 = None          # 只有 ROI 快路会填（见下）
        _final_mask_u8 = None
        if 'raw' in cfg.mode:
            if cfg.mode == 'raw-rgb':
                out_face = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                out_mask = cv2.warpAffine(np.ones_like(prd_face_bgr), face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                out_img = img_bgr * (1 - out_mask) + out_face * out_mask
                out_merging_mask_a = img_face_mask_full
            elif cfg.mode == 'raw-predict':
                out_img = prd_face_bgr
                out_merging_mask_a = wrk_face_mask_a
            out_img = np.clip(out_img, 0, 1)
        else:
            # Standard mode
            # ------------------------------------------------------------------
            # ROI 快路：只算脸所在的那一块（逐位一致，见 _compose_roi_standard）
            # DFL_MERGE_ROI=0 / 条件不满足 → 走下面的整帧实现（那是参考实现）
            # ------------------------------------------------------------------
            _roi_res = None
            if use_roi and work is not None:
                _roi_res = _compose_roi_standard(
                    frame_img, img_bgr, img_face_mask_full, prd_face_bgr,
                    dst_face_bgr, wrk_face_mask_a, face_mat, face_output_mat,
                    img_size, output_size, cfg, work, capture)
                ROI_STATS["frames"] += 1
                if _roi_res is not False:
                    ROI_STATS["roi_taken"] += 1
                if _roi_res is False:
                    # 遮罩支撑太小：整帧版在这一点上不会给 out_img 赋值，
                    # 会抛 UnboundLocalError 被外层 except 兜住 → 原样返回帧 + 全零遮罩。
                    # ROI 版把同一个结果直接给出来（少一条 traceback 噪音）。
                    ROI_STATS["roi_false"] += 1
                    return frame_img, np.zeros(frame_img.shape[:2], dtype=np.uint8)
            if _roi_res is not None:
                out_img, out_merging_mask_a, _final_u8, _final_mask_u8, prd_face_bgr = _roi_res
            else:
                # ↓↓↓ 整帧参考实现（一个字都没改，只整体右移一层）↓↓↓
                maxregion = np.argwhere(img_face_mask_full >= 0.1)
                if maxregion.size != 0:
                    miny, minx = maxregion.min(axis=0)[:2]
                    maxy, maxx = maxregion.max(axis=0)[:2]
                    if min(maxy - miny, maxx - minx) >= 4:
                        mask_area = wrk_face_mask_a.copy()
                        mask_area[mask_area > 0] = 1.0

                        # Color transfer functions（整帧版与 ROI 版共用同一张表）
                        ct_functions = _ct_table(mask_area)
                        fn = ct_functions.get(cfg.color_transfer_mode)
                        if fn:
                            try:
                                prd_face_bgr = np.clip(fn(prd_face_bgr, dst_face_bgr), 0, 1)
                            except Exception:
                                pass

                        # Hist-match
                        if cfg.mode == 'hist-match':
                            hist_mask = np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
                            if cfg.masked_hist_match:
                                hist_mask *= mask_area
                            white = (1.0 - hist_mask) * np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
                            hm1 = prd_face_bgr * hist_mask + white
                            hm1[hm1 > 1.0] = 1.0
                            hm2 = dst_face_bgr * hist_mask + white
                            hm2[hm2 > 1.0] = 1.0
                            prd_face_bgr = imagelib.color_hist_match(hm1, hm2, cfg.hist_match_threshold).astype(np.float32)

                        # Seamless mask
                        img_face_seamless_mask = None
                        if 'seamless' in cfg.mode:
                            for i in range(1, 10):
                                a = img_face_mask_full > i / 10.0
                                if len(np.argwhere(a)) == 0:
                                    continue
                                img_face_seamless_mask = img_face_mask_full.copy()
                                img_face_seamless_mask[a] = 1.0
                                img_face_seamless_mask[img_face_seamless_mask <= i / 10.0] = 0.0
                                break

                        # Warp predicted face to full frame
                        out_img = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr),
                                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                        out_img = np.clip(out_img, 0, 1)

                        # Seamless clone
                        if 'seamless' in cfg.mode and img_face_seamless_mask is not None:
                            try:
                                l, t, w, h = cv2.boundingRect((img_face_seamless_mask * 255).astype(np.uint8))
                                center = (int(l + w / 2), int(t + h / 2))
                                out_img = cv2.seamlessClone(
                                    (out_img * 255).astype(np.uint8),
                                    frame_img.astype(np.uint8),
                                    (img_face_seamless_mask * 255).astype(np.uint8),
                                    center, cv2.NORMAL_CLONE
                                ).astype(np.float32) / 255.0
                            except Exception:
                                pass

                        # Composite with mask
                        out_img = img_bgr * (1 - img_face_mask_full) + out_img * img_face_mask_full

                        # Post-processing (on warped-back face)
                        mp = cfg.motion_blur_power / 100.0
                        needs_pp = ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
                                   cfg.mode == 'seamless-hist-match' or mp != 0 or \
                                   cfg.blursharpen_amount != 0 or cfg.image_denoise_power != 0 or \
                                   cfg.bicubic_degrade_power != 0

                        if needs_pp:
                            out_face_bgr_pp = cv2.warpAffine(out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)

                            if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                                try:
                                    fn_pp = ct_functions.get(cfg.color_transfer_mode)
                                    if fn_pp:
                                        out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
                                except Exception:
                                    pass
                                    try:
                                        out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
                                    except Exception:
                                        pass

                            if cfg.mode == 'seamless-hist-match':
                                out_face_bgr_pp = imagelib.color_hist_match(out_face_bgr_pp, dst_face_bgr, cfg.hist_match_threshold)

                            if mp != 0:
                                k = int(mp * 100)
                                if k >= 1:
                                    k = np.clip(k + 1, 2, 50)
                                    kernel = np.zeros((k, k))
                                    kernel[:, k // 2] = 1.0 / k
                                    out_face_bgr_pp = cv2.filter2D(out_face_bgr_pp, -1, kernel)

                            if cfg.blursharpen_amount != 0:
                                amt = cfg.blursharpen_amount / 100.0
                                if cfg.sharpen_mode == 1:
                                    blurred = cv2.blur(out_face_bgr_pp, (3, 3))
                                else:
                                    blurred = cv2.GaussianBlur(out_face_bgr_pp, (0, 0), 1.0)
                                out_face_bgr_pp = cv2.addWeighted(out_face_bgr_pp, 1.0 + amt, blurred, -amt, 0)
                                out_face_bgr_pp = np.clip(out_face_bgr_pp, 0, 1)

                            if cfg.image_denoise_power != 0:
                                n = cfg.image_denoise_power
                                while n > 0:
                                    denoised = cv2.medianBlur(frame_img.astype(np.float32) / 255.0, 5)
                                    if int(n / 100) != 0:
                                        img_bgr = denoised
                                    else:
                                        p = (n % 100) / 100.0
                                        img_bgr = img_bgr * (1.0 - p) + denoised * p
                                    n = max(n - 10, 0)

                            if cfg.bicubic_degrade_power != 0:
                                p = 1.0 - cfg.bicubic_degrade_power / 101.0
                                down = cv2.resize(img_bgr, (int(img_size[0] * p), int(img_size[1] * p)), interpolation=cv2.INTER_CUBIC)
                                img_bgr = cv2.resize(down, img_size, interpolation=cv2.INTER_CUBIC)

                            new_out = cv2.warpAffine(out_face_bgr_pp, face_mat, img_size, np.empty_like(img_bgr),
                                                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                            out_img = np.clip(img_bgr * (1 - img_face_mask_full) + new_out * img_face_mask_full, 0, 1)

                        if cfg.color_degrade_power != 0:
                            reduced = imagelib.reduce_colors(out_img, 256)
                            if cfg.color_degrade_power == 100:
                                out_img = reduced
                            else:
                                alpha = cfg.color_degrade_power / 100.0
                                out_img = out_img * (1.0 - alpha) + reduced * alpha

                out_merging_mask_a = img_face_mask_full

        if out_img is None:
            out_img = img_bgr.copy()

        # 单脸调试：预测脸（色彩迁移之后）与「真的合成出来的那张脸」。
        # merged_face 是把整帧结果按 face_mat 反变换回正脸 —— 和用户看到的像素同一份，
        # 所以「脸看起来发蒙」时第 6 格能直接指出是遮罩、色彩迁移还是 seamless。
        if capture is not None:
            try:
                capture["prd_face_bgr"] = prd_face_bgr
                if out_img.shape[:2] == img_bgr.shape[:2]:
                    capture["merged_face"] = cv2.warpAffine(
                        np.clip(out_img, 0, 1), face_mat, (output_size, output_size),
                        flags=cv2.INTER_CUBIC)
                else:
                    capture["merged_face"] = out_img
            except Exception:                                # noqa: BLE001
                pass

        # Debug visualization (replaces normal output)
        if cfg.mode == 'debug' or getattr(cfg, 'show_debug', False):
            vis = _build_debug_vis(prd_face_bgr, dst_face_bgr, wrk_face_mask_a_0,
                                   prd_face_mask_a_0, prd_face_dst_mask_a_0, output_size)
            return vis.astype(np.uint8), np.zeros((1, 1), dtype=np.uint8)

        if _final_u8 is not None:
            # ROI 版已经把出图那一趟（整帧 float32 *255 + u8）在遮罩盒里做完了：
            # 盒外按定义等于原帧像素（u8 → f32(/255) → *255 → u8 对 0..255 逐位无损），
            # 所以这里直接用它，不再整帧搬一遍。
            final, final_mask = _final_u8, _final_mask_u8
        else:
            final = np.clip(out_img * 255, 0, 255).astype(np.uint8)
            final_mask = np.clip(out_merging_mask_a * 255 if out_merging_mask_a is not None else 0, 0, 255).astype(np.uint8)
        return final, final_mask.squeeze()

    except Exception as e:
        traceback.print_exc()
        return frame_img, np.zeros(frame_img.shape[:2], dtype=np.uint8)


def MergeMasked(frame_info, cfg, predictor_func=None, face_enhancer_func=None,
                xseg_256_extract_func=None):
    """Merge all faces in a frame. Returns RGBA."""
    frame_img = cv2.imread(str(frame_info.filepath))
    if frame_img is None:
        return None
    out_img = None
    full_mask = np.zeros(frame_img.shape[:2], dtype=np.float32)
    for landmarks in frame_info.landmarks_list:
        face_img, face_mask = MergeMaskedFace(
            frame_img, landmarks, cfg, predictor_func,
            face_enhancer_func, xseg_256_extract_func)
        if out_img is None:
            out_img = face_img.astype(np.float32)
            full_mask = face_mask.astype(np.float32)
        else:
            m = (face_mask.astype(np.float32) / 255.0) * (1.0 - full_mask / 255.0)
            m3 = np.stack([m, m, m], axis=-1)
            out_img = out_img * (1 - m3) + face_img.astype(np.float32) * m3
            full_mask = np.maximum(full_mask, face_mask.astype(np.float32))
    if out_img is None:
        out_img = frame_img.astype(np.float32)
    rgba = np.concatenate([
        np.clip(out_img, 0, 255).astype(np.uint8),
        np.clip(full_mask, 0, 255).astype(np.uint8)[:, :, None]
    ], axis=-1)
    return rgba


def MergeFaceAvatar(predictor_func, predictor_input_shape, cfg,
                    prev_temporal_frame_infos, frame_info, next_temporal_frame_infos):
    """FaceAvatar merge with temporal context."""
    try:
        frame_img = cv2.imread(str(frame_info.filepath))
        if frame_img is None:
            return None
        prev_imgs = []
        for fi in prev_temporal_frame_infos:
            img = cv2.imread(str(fi.filepath))
            if img is not None:
                prev_imgs.append(img)
        next_imgs = []
        for fi in next_temporal_frame_infos:
            img = cv2.imread(str(fi.filepath))
            if img is not None:
                next_imgs.append(img)
        if not frame_info.landmarks_list:
            return frame_img
        landmarks = frame_info.landmarks_list[0]
        mat = LandmarksProcessor.get_transform_mat(landmarks, 256, FaceType.FULL)
        face_img = cv2.warpAffine(frame_img, mat, (256, 256), flags=cv2.INTER_LANCZOS4)
        pred = predictor_func(face_img)
        out_img = pred.astype(np.float32) if pred is not None else face_img.astype(np.float32)
        if cfg.sharpen_mode != 0 and cfg.blursharpen_amount != 0:
            amount = cfg.blursharpen_amount / 100.0
            if cfg.sharpen_mode == 1:
                blurred = cv2.blur(out_img, (3, 3))
            else:
                blurred = cv2.GaussianBlur(out_img, (0, 0), 1.0)
            out_img = cv2.addWeighted(out_img, 1.0 + amount, blurred, -amount, 0)
            out_img = np.clip(out_img, 0, 255)
        if cfg.add_source_image:
            h, w = frame_img.shape[:2]
            out_img = np.concatenate([frame_img.astype(np.float32), out_img], axis=1)
            out_img = cv2.resize(out_img, (w, h))
        return np.clip(out_img, 0, 255).astype(np.uint8)
    except Exception as e:
        traceback.print_exc()
        if frame_info.filepath:
            return cv2.imread(str(frame_info.filepath))
        return None
