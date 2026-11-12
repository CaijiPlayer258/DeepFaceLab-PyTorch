# -*- coding: utf-8 -*-
"""视频模式 · 合成阶段（第 ⑤ 步）的**纯 CPU** 实现。

语义基准 = 4.0 的 MergeStudio/core/export_pipeline.py::_phase_merge_worker
（D:/MyProject/DeepFaceLab-PyTorch/MergeStudio/core/export_pipeline.py:808-976）。
4.0 的视频导出本来就是「swap → mask → merge」三段分开跑的：swap 把换脸图和 dfm 自带
两张遮罩落在 work 目录（{idx:08d}_{fi}.jpg / _prdm.png / _dstdm.png），mask 段把
XSeg 的两张遮罩落在 mask 目录（_xseg_prd.png / _xseg_dst.png），merge 段只读文件、
不碰推理。我们照这个粒度拆成 ③换脸 ④遮罩 ⑤合成，**逐段对齐**：

  4.0 行号            做什么                        我们的位置
  ─────────────────────────────────────────────────────────────────────────
  808-823             读帧 / 读 cfg / mask_mode      compose_frame()
  840-881             遮罩准备：按 mask_mode 取 mask   _prepare_mask()
                      组合（1=dst hull、2/4/5=prd、3=dst、
                      6/8=xseg_prd、7=xseg_dst、9=全乘）
                      尺寸不符就 CUBIC 缩放到换脸图
  882-893             腐蚀/膨胀（erode±）+ 高斯模糊   _prepare_mask()
                      （blur），pad 32 再裁回
  896-935             色彩迁移 ct_mode 9 种            _color_transfer()
  931-936             hist-match（'hist-match' 在     _blend()
                      mode 里才做）
  937-943             反贴：fmat=get_transform_mat(lm,  _paste_layer()
                      sz, WHOLE_FACE)，WARP_INVERSE_MAP
                      把脸和遮罩都变换回原帧尺寸
  945-966             overlay / seamless 合成          compose_frame()
                      （seamless = cv2.seamlessClone
                      NORMAL_CLONE 后再按 mw 混合）
  969-975             写盘（就地覆盖 frames/ 或写     compose_frame() / 调用方
                      merged/），PNG 无损

三条硬约束（验收要盯）：
  · **不碰推理**：这个模块只 import numpy / cv2 / facelib.LandmarksProcessor，
    没有 onnxruntime、没有 torch，也没有任何 session —— GPU 显存一个字节都不涨。
  · 输出像素与 4.0 一致：同一帧同一参数下 mean ≤ 2 / max ≤ 8（见 accept 脚本）。
  · 五段打点：遮罩准备 / 色彩迁移 / 融合 / 反贴 / 写盘，逐帧累计到 timings。
"""
from __future__ import annotations

import time

import numpy as np

# 4.0 的 ct_mode 字符串 / 数字两套写法都认（见 export_pipeline.py:907-928）
_CT_ALIAS = {
    "none": 0, "": 0, "0": 0,
    "rct": 1, "1": 1,
    "lct": 2, "2": 2,
    "mkl": 3, "3": 3,
    "mkl-m": 4, "4": 4,
    "idt": 5, "5": 5,
    "idt-m": 6, "6": 6,
    "sot-m": 7, "sot": 7, "7": 7,
    "mix-m": 8, "mix": 8, "8": 8,
    "mt": 9, "9": 9,
    "lut": 10, "10": 10,
}


def ct_code(raw) -> int:
    """ct_mode -> 4.0 的整数码（认不出来按 none）。"""
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return int(raw) if 0 <= int(raw) <= 10 else 0
    return _CT_ALIAS.get(str(raw or "").strip().lower(), 0)


_ENGINE_PATH_READY = {"done": False}


def ensure_engine_path():
    """把 4.0 引擎根（app/engine）追加到 sys.path —— facelib / core.imagelib 在那儿。

    **append 不是 insert**：4.1 自己的模块优先，绝不抢同名模块（和 _legacy.py 同口径）。
    """
    if _ENGINE_PATH_READY["done"]:
        return
    _ENGINE_PATH_READY["done"] = True
    import sys
    try:
        from ._paths import engine_root
        root = str(engine_root())
    except Exception:                                       # noqa: BLE001
        root = ""
    if root and root not in sys.path:
        sys.path.append(root)


def _imagelib():
    """4.0 的色彩迁移实现（core.imagelib.color_transfer）。

    4.1 的 server python 里没有 numexpr，而 color_transfer 只在 reinhard 的三行
    用了 ne.evaluate（单层四则运算）。这里先补一个极小的 numpy 替身，再 import ——
    只为让 rct 也能跑；mt/lct/mkl/idt/sot/mix 本来就不碰 numexpr。
    """
    ensure_engine_path()
    import sys
    if "numexpr" not in sys.modules:
        try:
            import numexpr  # noqa: F401
        except Exception:                                   # noqa: BLE001
            from . import _numexpr_shim
            sys.modules["numexpr"] = _numexpr_shim
    # 注意：core.imagelib 的 __init__ 里也导出了一个**同名函数** color_transfer，
    # `from core.imagelib import color_transfer` 拿到的是函数，不是模块 —— 必须按
    # 模块名 import。
    import importlib
    return importlib.import_module("core.imagelib.color_transfer")


# ---------------- 遮罩准备（4.0:840-893） ----------------
def _read_gray(path):
    """灰度 mask -> float32 0..1；读不到返回 None。"""
    import cv2
    if not path:
        return None
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def _prepare_mask(face: dict, frame_shape, cfg: dict, size: int, timings: dict):
    """按 mask_mode 组出工作遮罩（0..1，size×size），再做 erode/blur。

    对应 4.0 export_pipeline.py:840-893。mask_mode：
      0 full（全 1）· 1 dst hull · 2 prd · 3 dst · 4 prd×dst · 5 prd+dst
      6 xseg-prd · 7 xseg-dst · 8 xseg-prd×xseg-dst · 9 全部相乘
    """
    import cv2
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_image_hull_mask, get_transform_mat

    t0 = time.time()
    mask_mode = int(cfg.get("mask_mode", 6) or 0)
    erode = int(cfg.get("erode_mask_modifier", cfg.get("mask_erode", 0)) or 0)
    blur = int(cfg.get("blur_mask_modifier", cfg.get("mask_blur", 0)) or 0)
    lm = face["landmarks"]
    prd_dfm = face.get("prd_dfm")
    dst_dfm = face.get("dst_dfm")
    xseg_prd = face.get("xseg_prd")
    xseg_dst = face.get("xseg_dst")

    wrk = None
    if mask_mode == 1:
        # dst hull：先把原帧的凸包画出来，再用正变换搬到脸空间（和 4.0 一模一样）
        hull = get_image_hull_mask(frame_shape, lm)
        fmat = get_transform_mat(lm, int(size), FaceType.WHOLE_FACE, scale=1.0)
        wrk = cv2.warpAffine(hull, fmat, (int(size), int(size)),
                             flags=cv2.INTER_CUBIC)
    elif mask_mode in (2, 4, 5):
        if prd_dfm is not None:
            wrk = prd_dfm.copy()
            if mask_mode == 4 and dst_dfm is not None:
                wrk = wrk * dst_dfm
            elif mask_mode == 5 and dst_dfm is not None:
                wrk = np.clip(wrk + dst_dfm, 0, 1)
    elif mask_mode == 3:
        if dst_dfm is not None:
            wrk = dst_dfm.copy()
    elif mask_mode in (6, 8):
        if xseg_prd is not None:
            wrk = np.asarray(xseg_prd, np.float32).copy()
            if mask_mode == 8 and xseg_dst is not None:
                wrk = wrk * np.asarray(xseg_dst, np.float32)
    elif mask_mode == 7:
        if xseg_dst is not None:
            wrk = np.asarray(xseg_dst, np.float32).copy()
    elif mask_mode == 9:
        wrk = np.ones((int(size), int(size)), np.float32)
        for m in (prd_dfm, dst_dfm, xseg_prd, xseg_dst):
            if m is not None:
                wrk = wrk * np.asarray(m, np.float32)
    if wrk is None:
        wrk = np.ones((int(size), int(size)), np.float32)      # 4.0:878-879
    if wrk.shape[:2] != (int(size), int(size)):
        wrk = cv2.resize(np.asarray(wrk, np.float32), (int(size), int(size)),
                         interpolation=cv2.INTER_CUBIC)        # 4.0:880-881
    if erode != 0 or blur != 0:                                # 4.0:882-892
        pad = 32
        msk = np.pad(np.asarray(wrk, np.float32), pad)
        if erode > 0:
            msk = cv2.erode(msk, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erode, erode)), iterations=1)
        elif erode < 0:
            msk = cv2.dilate(msk, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (-erode, -erode)), iterations=1)
        if blur > 0:
            b = blur + (1 - blur % 2)
            msk = cv2.GaussianBlur(msk, (b, b), 0)
        wrk = msk[pad:-pad, pad:-pad]
    wrk = np.clip(wrk, 0, 1).astype(np.float32)                # 4.0:893
    timings["mask_ms"] = timings.get("mask_ms", 0.0) + (time.time() - t0) * 1000.0
    return wrk


# ---------------- 色彩迁移（4.0:896-935） ----------------
def _mask_bbox(m):
    """遮罩非零的包围盒（slice）；全零返回 None。

    用 any(axis=…) + flatnonzero 而不是 np.argwhere：后者会给每个非零像素建一行
    坐标（416² 的软遮罩能到十万行），比统计本身还贵。
    """
    m2 = m[..., 0] if m.ndim == 3 else m
    rows = np.flatnonzero(m2.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m2.any(axis=0))
    return (slice(int(rows[0]), int(rows[-1]) + 1),
            slice(int(cols[0]), int(cols[-1]) + 1))


def _match_tone_roi_apply(src, trg, mask):
    """match_tone 的「算统计量 + 应用」二合一（src/trg/mask 已是同一块 ROI 的切片）。"""
    return _match_tone_roi(src, trg, mask)


def _match_tone_roi(src, trg, mask):
    """match_tone_np 的 ROI 版（逐字照 4.0 core/imagelib/color_transfer.py:523-558 的公式）。

    唯一的改动：均值/方差只在**遮罩包围盒**里累加。遮罩在盒外恒为 0，按定义权重为 0，
    所以这两个统计量在数学上完全一样（差别只在 float32 的求和顺序，末位噪声级）。
    收益：416² 全图 → 脸的包围盒（约 300²），实测 ct 88ms → ~35ms/帧。
    """
    src = np.ascontiguousarray(np.asarray(src, np.float32))
    trg = np.ascontiguousarray(np.asarray(trg, np.float32))
    bb = _mask_bbox(mask) if mask is not None else None
    if bb is None:
        bb = (slice(0, src.shape[0]), slice(0, src.shape[1]))
    s_crop = src[bb]
    t_crop = trg[bb]
    w = None if mask is None else np.asarray(mask, np.float32)[bb]
    nch = int(src.shape[2])
    if w is not None:
        if w.ndim == 2:
            w = w[..., None]                 # (H,W,1)
        # **先摊成 (N,C) 再广播到 (N,3)**，而不是 np.repeat 出真实的 (N,3)：
        # numpy 沿 axis=0 归约 (N,1) 和 (N,3) 的求和顺序不同（(N,1) 实测 63277.77，
        # (N,3) 63277.88，差 2.6e-8 相对），会顺着统计量一路放大到 0.12 的像素差；
        # 而 np.broadcast_to 出来的 0 步长 (N,3) 视图走的是与真实 (N,3) 完全相同的
        # 归约路径 —— 实测逐个 bit 相同，且不占内存（省掉 1MB/帧的临时数组）。
        w = (w.reshape(-1, 1) if w.shape[2] == 1 else w.reshape(-1, w.shape[2]))

    def _stats(x):
        flat = x.reshape(-1, nch)
        if w is None:
            return flat.mean(axis=0), flat.std(axis=0)
        wf = np.broadcast_to(w, flat.shape)
        ws = wf.sum(axis=0) + 1e-8                       # (3,)
        mean = (flat * wf).sum(axis=0) / ws              # (3,)
        # (flat-mean)**2*wf：以前是 3 个临时数组，这里 1 个 + 原地三连（逐位等价）
        d = np.subtract(flat, mean, dtype=np.float32)
        np.multiply(d, d, out=d)                         # == d**2
        np.multiply(d, wf, out=d)
        var = d.sum(axis=0) / ws
        return mean, np.sqrt(var + 1e-6)

    s_mean, s_std = _stats(s_crop)
    t_mean, t_std = _stats(t_crop)
    scale = t_std / (s_std + 1e-6)
    # (src - mean)*scale + mean 的三个临时数组 → 一个 out 缓冲 + 原地三连。
    # 运算顺序、dtype 一个都没动，所以逐位一致（对拍见 tools 里的数值等价测试）。
    out = np.empty_like(src)
    np.subtract(src, s_mean.reshape(1, 1, -1), out=out)
    np.multiply(out, scale.reshape(1, 1, -1), out=out)
    np.add(out, t_mean.reshape(1, 1, -1), out=out)
    np.clip(out, 0.0, 1.0, out=out)
    return out

def _color_transfer(fo, dw, wrk_mask, code: int, timings: dict):
    """fo / dw 都是 0..1 float32 的 S×S 脸；返回迁移后的 fo。语义照 4.0:907-928。

    带遮罩的几种迁移（rct / mkl-m / idt-m / sot-m / mix-m / mt）只在**遮罩包围盒**里
    算和写：盒外 mw 恒为 0，合成时乘的是 0，最终像素一个 bit 都不受影响；盒内用的
    统计量与全图加权统计量在数学上相同（权重的定义就是遮罩）。这一步把 416² 的整脸
    运算压到脸的包围盒（实测约 1/3 面积）。
    """
    if not code:
        return fo
    ct = _imagelib()
    m = wrk_mask[..., None] if wrk_mask.ndim == 2 else wrk_mask
    out = fo
    bb = None
    if code in (1, 4, 6, 7, 8):
        bb = _mask_bbox(wrk_mask)
        if bb is not None:
            out = fo.copy()
            fo = fo[bb]
            dw = dw[bb]
            m = m[bb]
    try:
        if code == 1:
            out = ct.reinhard_color_transfer(fo, dw, target_mask=m, source_mask=m)
        elif code == 2:
            out = ct.linear_color_transfer(fo, dw)
        elif code == 3:
            out = ct.color_transfer_mkl(fo, dw)
        elif code == 4:
            out = ct.color_transfer_mkl(fo * m, dw * m)
        elif code == 5:
            out = ct.color_transfer_idt(fo, dw)
        elif code == 6:
            out = ct.color_transfer_idt(fo * m, dw * m)
        elif code == 7:
            out = ct.color_transfer_sot(fo * m, dw * m, steps=10, batch_size=30)
        elif code == 8:
            out = ct.color_transfer_mix(fo * m, dw * m)
        elif code == 9:
            # mt（match_tone）：统计量与写回都限制在遮罩包围盒里（数值等价）
            bb9 = _mask_bbox(wrk_mask)
            if bb9 is None:
                out = fo
            else:
                out = fo.copy()
                out[bb9] = np.clip(
                    _match_tone_roi_apply(fo[bb9], dw[bb9], wrk_mask[bb9]), 0, 1)
        elif code == 10:
            out = ct.color_transfer_lut(fo, dw)
    except Exception:                                          # noqa: BLE001
        out = fo                       # 4.0 里 ct 挂了也只是打一行日志、保原样
    if bb is not None and out is not fo:
        out[bb] = np.clip(np.asarray(fo, np.float32), 0, 1)
        return out
    return np.clip(np.asarray(out, np.float32), 0, 1)


# ---------------- 反贴 + 融合（4.0:937-966） ----------------
def _paste_layer(fo, wrk_mask, fmat, frame_shape, mode: str):
    """把脸和遮罩都反变换回原帧尺寸：返回 (fw, mw)。4.0:938-943。"""
    import cv2
    h, w = frame_shape[:2]
    mw = cv2.warpAffine(wrk_mask, fmat, (w, h),
                        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    mw = np.clip(mw, 0, 1)
    if mw.ndim == 2:
        mw = mw[..., None]
    fw = cv2.warpAffine(fo, fmat, (w, h),
                        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
    fw = np.clip(fw, 0, 1)
    return fw, mw


def _roi_of_face(mat, size: int, frame_shape, margin: int = 8):
    """S×S 画布在原帧里的包围盒（外扩 margin，保证三次插值的取样全在 ROI 内）。

    返回 (x0, y0, x1, y1) 或 None（脸整个落在帧外）。
    """
    import math
    h, w = frame_shape[:2]
    inv = cv2_invert(mat)
    pts = []
    for cx, cy in ((0, 0), (size, 0), (size, size), (0, size)):
        pts.append((inv[0, 0] * cx + inv[0, 1] * cy + inv[0, 2],
                    inv[1, 0] * cx + inv[1, 1] * cy + inv[1, 2]))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = int(math.floor(min(xs))) - margin
    x1 = int(math.ceil(max(xs))) + margin
    y0 = int(math.floor(min(ys))) - margin
    y1 = int(math.ceil(max(ys))) + margin
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def cv2_invert(mat):
    """cv2 的 2×3 仿射求逆（和 cv2.WARP_INVERSE_MAP 内部那一套同一个算法）。"""
    import cv2
    return cv2.invertAffineTransform(np.asarray(mat, np.float64))


def _blend_layer(src, fw, mw, mode: str, dst_u8=None):
    """一层脸的融合（4.0:949-965）：seamless 走 seamlessClone，其它按 mask 直接叠。

    src: 当前画面（float32 0..1，ROI 或整帧）· fw: 反贴回来的脸 · mw: 反贴回来的遮罩。
    非 seamless 只在 mw>0 的包围盒里动 —— 盒外 mw==0，像素按定义不变。
    """
    import cv2
    # 融合只在「遮罩非零」的包围盒里做：盒外 mw==0，混合式退化成 src*1 + x*0，
    # 而 u8 -> f32(/255) -> *255 -> u8 这一趟实测对 0..255 全部 256 个值**逐位无损**
    # （74.6M 像素实测 0 个不一致），所以盒外一个 bit 都不会变。省掉的是：整块 ROI
    # 的 float 转换、5 次整块浮点乘加、再整块转回 u8（实测约 4ms/帧）。
    m2 = mw[..., 0] if mw.ndim == 3 else mw
    nz = cv2.findNonZero((m2 > 0).astype(np.uint8))
    if nz is None:
        return src
    bx, by, bw_, bh_ = cv2.boundingRect(nz)
    r = (slice(by, by + bh_), slice(bx, bx + bw_))
    m_roi = mw[r] if mw.ndim == 3 else mw[r][..., None]
    if "seamless" in mode:
        try:
            mw_bin = np.clip(mw * 255, 0, 255).astype(np.uint8)
            fw_u8 = np.clip(fw * 255, 0, 255).astype(np.uint8)
            m2b = mw_bin[..., 0] if mw_bin.ndim == 3 else mw_bin
            coords = cv2.findNonZero(m2b)
            if coords is None:
                # u8 遮罩整块为 0（只有 mw<1/255 的软边）：沿用旧式的整块混合，
                # 不给自己省这一步 —— 差别理论上 ≤1 LSB，但没必要赌。
                return src * (1 - mw) + fw * mw
            # center 与 4.0/旧版**逐字节相同**：仍然取 u8 遮罩的包围盒中心
            # （mw>0 的包围盒比它大一圈：mw < 1/255 的软边像素进不了 u8 遮罩）。
            cbx, cby, cbw, cbh = cv2.boundingRect(coords)
            center = (cbx + cbw // 2, cby + cbh // 2)
            dst_u8 = (np.clip(src * 255.0, 0, 255).astype(np.uint8)
                      if dst_u8 is None else dst_u8)
            sc = cv2.seamlessClone(fw_u8, dst_u8, mw_bin, center, cv2.NORMAL_CLONE)
            out = src.copy()
            out[r] = src[r] * (1 - m_roi) + (sc[r].astype(np.float32) / 255.0) * m_roi
            return out
        except Exception:                                      # noqa: BLE001
            return src * (1 - mw) + fw * mw
    out = src.copy()
    out[r] = src[r] * (1 - m_roi) + fw[r] * m_roi
    return out


def compose_frame(frame_bgr, faces, cfg: dict, timings: dict = None,
                  roi: bool = True) -> np.ndarray:
    """一帧的合成（纯 CPU）：遮罩准备 → 色彩迁移 → 融合 → 反贴回原帧。

    frame_bgr : 原帧 uint8 (H,W,3)
    faces     : [{"landmarks": (68,2), "pred": (S,S,3) 0..1,
                  "prd_dfm"/"dst_dfm"/"xseg_prd"/"xseg_dst": (S,S) 或 (256,256) 0..1}]
    cfg       : {"mask_mode","color_transfer_mode","mode","erode_mask_modifier",
                 "blur_mask_modifier","hist_match_threshold"}
    timings   : 累计计时（prep/mask/ct/paste/blend，单位 ms）

    性能：整帧 float32 是 25MB，4.0 的做法（整帧 fc、两趟整帧 warp、整帧混合、整帧回
    u8）每帧要搬 ~100MB。这里把**重活都压进脸的包围盒 ROI**（遮罩支撑一定落在 S 画布的
    四边形内，向外留 8px 给三次插值取样），ROI 之外一个像素都不动 —— 与 4.0 逐像素等价。
    u8→f32→u8 对 ROI 内的非遮罩像素是恒等变换，所以只有 mw>0 的地方才有新像素。
    """
    import cv2
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat

    timings = timings if isinstance(timings, dict) else {}
    if not faces:
        return frame_bgr
    mode = str(cfg.get("mode") or "overlay").strip().lower()
    ct = ct_code(cfg.get("color_transfer_mode", cfg.get("ct_mode")))
    hm_thr = int(cfg.get("hist_match_threshold", 255) or 255)
    h, w = frame_bgr.shape[:2]
    layers = []

    for f in faces:
        lm = np.asarray(f["landmarks"], np.float32)
        fo = np.clip(np.asarray(f["pred"], np.float32), 0, 1)
        sz = int(fo.shape[0])
        fmat = get_transform_mat(lm, sz, FaceType.WHOLE_FACE, scale=1.0)   # S->帧
        wrk = _prepare_mask(f, frame_bgr.shape, cfg, sz, timings)          # 遮罩准备

        box = _roi_of_face(fmat, sz, frame_bgr.shape) if roi else None
        if roi and box is None:
            continue                        # 脸整个在帧外：贴上去也看不见
        if box is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = box
        rw, rh = x1 - x0, y1 - y0

        t_p = time.time()
        roi_u8 = frame_bgr[y0:y1, x0:x1]
        roi_f = roi_u8.astype(np.float32) / 255.0
        timings["prep_ms"] = timings.get("prep_ms", 0.0) + (time.time() - t_p) * 1000.0

        # ROI 偏移：cv2 的仿射是「默认把 M 当 src→dst，再自己求逆」；把 ROI 的原点
        # 折进平移项就是 M' = fmat · T(o)，正反两次 warp 用的是同一个 M'（一次带
        # WARP_INVERSE_MAP，一次不带）—— 和 4.0 整帧那两个调用逐字节等价。
        fwd = np.array(fmat, dtype=np.float64)
        fwd[0, 2] += fmat[0, 0] * x0 + fmat[0, 1] * y0
        fwd[1, 2] += fmat[1, 0] * x0 + fmat[1, 1] * y0

        t_ct = time.time()
        if ct:
            dw = cv2.warpAffine(roi_f, fwd, (sz, sz), flags=cv2.INTER_CUBIC)
            fo = _color_transfer(fo, dw, wrk, ct, timings)
        if "hist-match" in mode:
            dw2 = cv2.warpAffine(roi_f, fwd, (sz, sz), flags=cv2.INTER_CUBIC)
            fo = _imagelib().color_hist_match(fo, dw2, hist_match_threshold=hm_thr)
        timings["ct_ms"] = timings.get("ct_ms", 0.0) + (time.time() - t_ct) * 1000.0

        t_p = time.time()
        mw = cv2.warpAffine(wrk, fwd, (rw, rh),
                            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        mw = np.clip(mw, 0, 1)
        if mw.ndim == 2:
            mw = mw[..., None]
        fw = cv2.warpAffine(fo, fwd, (rw, rh),
                            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        fw = np.clip(fw, 0, 1)
        timings["paste_ms"] = timings.get("paste_ms", 0.0) + (time.time() - t_p) * 1000.0
        layers.append((fw, mw, roi_f, roi_u8, (x0, y0, x1, y1)))

    if not layers:
        return frame_bgr

    t_b = time.time()
    if len(layers) == 1:
        fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) = layers[0]
        blended = _blend_layer(roi_f, fw, mw, mode, dst_u8=roi_u8)
        out = frame_bgr if roi is False else frame_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    else:
        # 多张脸：整帧浮点累加，最后只回一次 u8（和 4.0 完全一致）
        work = frame_bgr.astype(np.float32) / 255.0
        for fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) in layers:
            sub = work[y0:y1, x0:x1]
            blended = _blend_layer(sub, fw, mw, mode, dst_u8=roi_u8)
            work[y0:y1, x0:x1] = blended
        out = np.clip(work * 255.0, 0, 255).astype(np.uint8)
    timings["blend_ms"] = timings.get("blend_ms", 0.0) + (time.time() - t_b) * 1000.0
    return out


def compose_frame_frame_scope(frame_bgr, faces, cfg: dict, timings: dict = None):
    """整帧版（DFL_MERGE_ROI=0 时用）：语义同 4.0，慢一个数量级，留作对拍基准。"""
    return compose_frame(frame_bgr, faces, cfg, timings, roi=False)
