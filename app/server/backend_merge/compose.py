# -*- coding: utf-8 -*-
"""合成粘合 + 合成核。（上半是 _merge_compose.py 原身：compose_frame / ensure_engine_path；下半从 routes_merge 切出来：素材缓存 / 取帧 / 调试视图 / 自动人脸分配 / 检测器）

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations
import numpy as np

import base64 as _b64
import collections
import json
import os
import threading
import time
import traceback
from pathlib import Path
from fastapi import HTTPException
from .._common.http import (
    _BadInput, _as_dict, _bool_val, _float_val, _int_val, _ok,
)
from .._face_group import register_cache_clearer
from ._shared import (
    _DEFAULT_DETECTOR, _DETECTORS, _DET_ALIAS_API, _REMOVED_DETECTORS, _settings_val,
)


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
        from .._paths import engine_root
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
            from .. import _numexpr_shim
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


def _prepare_mask(face: dict, frame_shape, cfg: dict, size: int, timings: dict,
                  fmat=None):
    """按 mask_mode 组出工作遮罩（0..1，size×size），再做 erode/blur。

    对应 4.0 export_pipeline.py:840-893。mask_mode：
      0 full（全 1）· 1 dst hull · 2 prd · 3 dst · 4 prd×dst · 5 prd+dst
      6 xseg-prd · 7 xseg-dst · 8 xseg-prd×xseg-dst · 9 全部相乘

    fmat=None（视频/项目 facelib 那条路）时按 68 点算 facelib WHOLE_FACE 矩阵；
    aligned 原生合成把 image_to_face_mat 传进来（mask_mode==1 的 dst hull 要用它
    把原帧的凸包搬到**对齐画布**上，不然又变成两张画布混用）。
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
    if mask_mode == 1 and lm is not None:
        # dst hull：先把原帧的凸包画出来，再用正变换搬到脸空间（和 4.0 一模一样）
        hull = get_image_hull_mask(frame_shape, lm)
        m1 = fmat
        if m1 is None:
            m1 = get_transform_mat(lm, int(size), FaceType.WHOLE_FACE, scale=1.0)
        wrk = cv2.warpAffine(hull, m1, (int(size), int(size)),
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



def _apply_sr_layers(out, layers, timings):
    """合成完成之后，在**帧空间**逐脸跑一次超分（GFPEN/GPEN 那类盲修复）。

    为什么在帧空间：芯片 224~416，模型 512 → 在芯片上做等于多一次重采样（先糊再加锐）。
    这里拿的是 ROI 的原生像素，只重采样一次；混回权重 = 该脸的遮罩 × 档位强度。

    DFL_SR=off（默认）时一个像素都不碰 —— 这条必须保证，mode B 的 md5 基线靠它。
    """
    try:
        from . import facesr
        if not facesr.enabled():
            return out
        a = facesr.strength()
        t0 = time.time()
        for fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) in layers:
            roi = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = facesr.apply_roi(roi, mw, a, timings)
        timings["sr_ms"] = timings.get("sr_ms", 0.0) + (time.time() - t0) * 1000.0
    except Exception as e:                                  # noqa: BLE001
        # 超分失败不该把整单合成带崩：告警一次，本帧按原样走
        print("[sr] 跳过（%s: %s）" % (e.__class__.__name__, str(e)[:160]), flush=True)
    return out

_PRED_SHAPE_SEEN = {}


def _pred_shape_complain(where: str, shape, extra: str = "") -> None:
    """pred / pred_u8 不是方形时喊一次（带调用栈），之后同处不再重复。

    只在**真的出问题**时打印，正常路径零成本（一次 ndim/shape 比较）。
    """
    key = str(where)
    if _PRED_SHAPE_SEEN.get(key):
        return
    _PRED_SHAPE_SEEN[key] = True
    import traceback as _tb
    print("[compose] %s 拿到非方形的 pred：shape=%s%s —— 这份数据不该出现在这里，"
          " 下面是它被送进来时的调用栈" % (key, tuple(shape), (" · " + extra) if extra else ""),
          flush=True)
    _tb.print_stack(file=__import__("sys").stdout)

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
        # 尺寸自检：pred 必须是**方形**（模型输出 S×S×3）。不是的话 sz 会按高度取，
        # 而后面的 warp/混合全按方形假设走 —— 最后在 out[y0:y1, x0:x1] = blended 处
        # 抛 could not broadcast（实测见过 177x131 对 147x71，两个方向比例还不一样）。
        # 这里**第一次**遇到就打出形状 + 完整调用栈：知道是谁把非方形的东西当 pred 送进来了。
        if fo.ndim != 3 or fo.shape[0] != fo.shape[1]:
            _pred_shape_complain("compose_frame", fo.shape)
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
        # 尺寸守卫：seamlessClone 或者某条回退路径算出来的 blended 偶尔会和 ROI 差一圈
        # （实测见过 177×131 对 147×71 —— 差得不是一像素，是两套坐标系混了），
        # 直接 `out[y0:y1, x0:x1] = blended` 会抛 could not broadcast，整单导出失败。
        # 这里就近重采样到 ROI 尺寸：不改变「贴哪儿」，只把尺寸对齐。
        _bh, _bw = blended.shape[:2]
        _rh, _rw = (y1 - y0), (x1 - x0)
        if (_bh, _bw) != (_rh, _rw):
            import cv2 as _cv2
            blended = _cv2.resize(np.asarray(blended, np.float32), (_rw, _rh),
                                  interpolation=_cv2.INTER_LINEAR)
            timings["shape_fix"] = timings.get("shape_fix", 0) + 1
            if not timings.get("_shape_warned"):
                timings["_shape_warned"] = True
                print("[compose] 尺寸不一致：blended=%dx%d ROI=%dx%d"
                      " （已就近重采样，帧不会被跳过；把这一行发出来能定位是 pred 还是 mask 跑偏）"
                      % (_bw, _bh, _rw, _rh), flush=True)
        out[y0:y1, x0:x1] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    else:
        # 多张脸。4.0 的做法是「整帧 float32 累加，最后回一次 u8」——
        # 4K 上这一步实测 190 ms 起（astype 22 + 混合 193 + clip/astype 46），每帧都做。
        # 关键观察：只要这些脸的 ROI 两两不相交，逐块混合 == 整帧混合（逐字节相同），
        # 因为每个像素只被一张脸碰过。相交时才退回整帧路径（保 4.0 的舍入）。
        boxes = [(x0, y0, x1, y1) for (_fw, _mw, _rf, _ru, (x0, y0, x1, y1)) in layers]
        _disjoint = True
        for i in range(len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                bx0, by0, bx1, by1 = boxes[j]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    _disjoint = False
                    break
            if not _disjoint:
                break
        if _disjoint:
            out = frame_bgr.copy() if roi is False else frame_bgr
            for fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) in layers:
                blended = _blend_layer(roi_f, fw, mw, mode, dst_u8=roi_u8)
                out[y0:y1, x0:x1] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        else:
            # ROI 相交：这才是必须整帧浮点的那种情况（4.0 的舍入语义）
            work = frame_bgr.astype(np.float32) / 255.0
            for fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) in layers:
                sub = work[y0:y1, x0:x1]
                blended = _blend_layer(sub, fw, mw, mode, dst_u8=roi_u8)
                work[y0:y1, x0:x1] = blended
            out = np.clip(work * 255.0, 0, 255).astype(np.uint8)
    timings["blend_ms"] = timings.get("blend_ms", 0.0) + (time.time() - t_b) * 1000.0
    out = _apply_sr_layers(out, layers, timings)
    return out


# ---------------- aligned 原生画布合成（项目模式默认） ----------------
def compose_aligned_frame(frame_bgr, faces, cfg: dict, timings: dict = None,
                          roi: bool = True) -> np.ndarray:
    """一帧的合成（纯 CPU）——**aligned 原生画布**：贴回只用 image_to_face_mat。

    和 :func:`compose_frame` 的唯一区别就是「画布」：

      compose_frame         fmat = get_transform_mat(lm68, S, WHOLE_FACE)   ← facelib 画布
      compose_aligned_frame fmat = 调用方给的 image_to_face_mat            ← aligned 画布

    整条链（dfm 输入 / 遮罩 / 色彩迁移 / 反贴）都在**同一个** S×S 画布上走，所以
    这里**一个 68 点都不用**：不需要 get_transform_mat，也没有「68 点 → WHOLE_FACE
    矩阵」这一步。

    **2026-12 更正**：aligned 画布和 facelib 画布现在是**同一张**。align_mat_from_68
    已经改成只拟合 33 点、模板直接从 facelib 取，与 get_transform_mat 逐位等价
    （1203 张实测 zoom 1.00000 / rot 0.0000° / 四角 0.000 px）。以前这里写的
    「放大约 13%」是错的：13% 是分布的**尾部**（p95），中位只有 +0.4%，主因是
    3° 旋转（中位 2.5°，p05..p95 −4.9°..+4.9°）加姿态相关的缩放，根因是
    **拟合点数**（68 点 vs 33 点）—— 多出来的下颌/下巴点落在单位方形外面，
    把最小二乘拽偏。详见 app/docs/23-ALIGNED-COMPOSE.md §2.2 与
    app/docs/24-ALIGN-WHOLEFACE.md。旧芯片（重切之前切的）仍然差 3°/0.4%，
    所以「混用两张画布」在**没重切过的老数据**上还是那句：错位几十像素。

    faces : [{"mat": 原帧→画布 的 2×3（**已把画布尺寸折进 pred 的 S**）,
              "pred": (S,S,3) 0..1,
              "dw": (S,S,3) 0..1 可选 —— 画布上的原帧脸（色彩迁移的参考）。
                    给了就不再把原帧 ROI 重新 warp 到画布（那是画布空间的重复采样）；
                    不给则退回 compose_frame 的做法。
              "landmarks": (68,2) 或 None（只有 mask_mode==1 的 dst hull 要它）,
              "prd_dfm"/"dst_dfm"/"xseg_prd"/"xseg_dst": (S,S) 或 (256,256) 0..1}]
    cfg   : 同 compose_frame
    timings: 同 compose_frame

    反贴语义（**别把两次求逆叠在一起**）：cv2.WARP_INVERSE_MAP 的含义是「M 是
    dst→src」，而 image_to_face_mat 正是 原帧→画布（= 贴回时的 dst→src），所以正确
    写法是 warpAffine(canvas, mat, (W,H), WARP_INVERSE_MAP)。先 invertAffineTransform
    再 WARP_INVERSE_MAP 等于恒等，会把脸贴到别处（实测差 82 LSB）。
    """
    import cv2

    timings = timings if isinstance(timings, dict) else {}
    if not faces:
        return frame_bgr
    mode = str(cfg.get("mode") or "overlay").strip().lower()
    ct = ct_code(cfg.get("color_transfer_mode", cfg.get("ct_mode")))
    hm_thr = int(cfg.get("hist_match_threshold", 255) or 255)
    h, w = frame_bgr.shape[:2]
    layers = []

    for f in faces:
        mat = f.get("mat")
        if mat is None:
            raise ValueError("compose_aligned_frame 需要 mat（image_to_face_mat）")
        m = np.asarray(mat, np.float64)
        if m.shape != (2, 3) or not np.all(np.isfinite(m)):
            raise ValueError("mat 不是合法的 2x3 仿射矩阵：%r" % (np.shape(mat),))
        fo = np.clip(np.asarray(f["pred"], np.float32), 0, 1)
        sz = int(fo.shape[0])
        wrk = _prepare_mask(f, frame_bgr.shape, cfg, sz, timings, fmat=m)   # 遮罩准备

        box = _roi_of_face(m, sz, frame_bgr.shape) if roi else None
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

        # ROI 偏移折进平移项：M' = M · T(o)（ROI 原点 o 折进去以后，正反两次 warp
        # 用的是同一个 M'，与整帧调用逐字节等价 —— 和 compose_frame 同一套做法）。
        fwd = np.array(m, dtype=np.float64)
        fwd[0, 2] += m[0, 0] * x0 + m[0, 1] * y0
        fwd[1, 2] += m[1, 0] * x0 + m[1, 1] * y0

        t_ct = time.time()
        if ct or "hist-match" in mode:
            dw = f.get("dw")
            if dw is None:
                dw = cv2.warpAffine(roi_f, fwd, (sz, sz), flags=cv2.INTER_CUBIC)
            else:
                dw = np.asarray(dw, np.float32)
                if dw.shape[:2] != (sz, sz):
                    dw = cv2.resize(dw, (sz, sz), interpolation=cv2.INTER_CUBIC)
            if ct:
                fo = _color_transfer(fo, dw, wrk, ct, timings)
            if "hist-match" in mode:
                fo = _imagelib().color_hist_match(fo, dw, hist_match_threshold=hm_thr)
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
        # 多张脸：整帧浮点累加，最后只回一次 u8（和 compose_frame/4.0 一致）
        work = frame_bgr.astype(np.float32) / 255.0
        for fw, mw, roi_f, roi_u8, (x0, y0, x1, y1) in layers:
            sub = work[y0:y1, x0:x1]
            blended = _blend_layer(sub, fw, mw, mode, dst_u8=roi_u8)
            _bh, _bw = blended.shape[:2]
            if (_bh, _bw) != (y1 - y0, x1 - x0):
                import cv2 as _cv2
                blended = _cv2.resize(np.asarray(blended, np.float32),
                                      (x1 - x0, y1 - y0), interpolation=_cv2.INTER_LINEAR)
                timings["shape_fix"] = timings.get("shape_fix", 0) + 1
            work[y0:y1, x0:x1] = blended
        out = np.clip(work * 255.0, 0, 255).astype(np.uint8)
    timings["blend_ms"] = timings.get("blend_ms", 0.0) + (time.time() - t_b) * 1000.0
    out = _apply_sr_layers(out, layers, timings)
    return out


# ============================================================
# 缓存快照（_merge_cache 是 app.py 里的全局字典，多请求共用）
# ============================================================
_cache_lock = threading.RLock()
_det_load_lock = threading.Lock()      # 检测器懒加载：别让并发请求各建一份
_det_run_lock = threading.Lock()       # det.thresh + det.detect 是共享对象上的两步操作


def _cache_snap() -> dict:
    """取一份快照。

    dict.update() 本身在 C 层是一口气做完的，但"读 mode、再读 names、再读 srcdir"
    是三次独立读，中间被另一次 /open 插进来就会拼出不存在的东西。所以每请求只取
    一次快照，之后全部读快照。
    """
    with _cache_lock:
        snap = dict(_merge_cache)
    snap.setdefault("mode", "")
    snap.setdefault("names", [])
    snap.setdefault("path", "")
    snap.setdefault("srcdir", "")
    snap.setdefault("dir", "")
    snap.setdefault("frames", 0)
    snap.setdefault("fps", 0)
    return snap


def _cache_set(**kw):
    with _cache_lock:
        _merge_cache.update(kw)


def _cache_mode(snap: dict) -> str:
    mode = str(snap.get("mode") or "")
    if mode:
        return mode
    if snap.get("names"):
        return "project"                     # 兼容这次改动之前写进去的缓存
    if snap.get("path"):
        return "video"
    return ""


def _scan_first(dirp, exts, limit=1):
    """目录里前 limit 个符合扩展名的文件名（排序后）。"""
    out = []
    try:
        with os.scandir(dirp) as it:
            for e in it:
                try:
                    if e.is_file() and Path(e.name).suffix.lower() in exts:
                        out.append(e.name)
                        if len(out) >= limit * 4:
                            break
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(out)[:limit]


def _has_dfl_meta(dirp, tries=8) -> bool:
    """抽样看这个目录里的图带不带 DFLJPG 元数据（landmarks）。

    合成完全靠 source_filename / landmarks 定位人脸，没有元数据的对齐集必然在
    子进程里以"没有任何对齐脸能对上帧"收场 —— 在这儿就说清楚。
    """
    for n in _scan_first(dirp, {".jpg", ".jpeg"}, tries):
        try:
            if _dfljpg_dict(Path(dirp) / n).get("landmarks") is not None:
                return True
        except Exception:
            continue
    return False


def _check_cache_ready(snap: dict):
    """缓存还在不在：项目目录 / 视频文件被删了就说"过期"，别让下游 500。"""
    mode = _cache_mode(snap)
    if mode == "project":
        srcdir = str(snap.get("srcdir") or "")
        if not srcdir or not Path(srcdir).is_dir():
            _cache_set(stale=True)
            raise _BadInput("项目目录不在了：%s —— 重新打开一次" % (srcdir or "?"), 410)
        return mode
    if mode == "video":
        path = str(snap.get("path") or "")
        if not path or not Path(path).is_file():
            _cache_set(stale=True)
            raise _BadInput("视频文件不见了：%s —— 可能被移动/删除了，重新打开一次"
                            % (path or "?"), 410)
        return mode
    raise _BadInput("还没有打开素材（先 /api/merge/open 或 /api/merge/prepare）", 409)


def _check_idx(snap: dict, idx: int) -> int:
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    mode = _cache_mode(snap)
    if mode == "project":
        total = len(snap.get("names") or [])
    else:
        total = int(snap.get("frames") or 0)
    if total and idx >= total:
        raise _BadInput("帧号越界：%d，这里只有 %d 帧（0~%d）" % (idx, total, total - 1), 404)
    return idx


# ============================================================
# 带参数的整帧合成：GET/POST /api/merge/frame/{idx} 与 /api/merge/merged/{idx}
# ------------------------------------------------------------
# 用户报的 BUG：「右侧改参数，画面不动」。
# 根因不是 worker 不认参数 —— /api/merge/preview 一直是通的 —— 而是这两个接口
# **根本没把参数带进合成链路**：
#   · /frame  原样把源帧编成 JPEG 就返回（settings 一个字节都没用上）
#   · /merged 从「上一次导出的成片」里抠一帧（那是导出那一刻的参数，改滑块当然不动）
# 现在两条都走下面 _frame_payload()：帧 → 检测 → 68 点 → worker 换脸/遮罩/合成
# （和 /preview 完全同一份 _preview_job），settings / model / models 逐项透传，
# 且在**同一帧同一模型**下只改一项，像素就必须变。
#
# 兼容：
#   · 不带 settings = 用当前默认值（ct_mode=mt、blend=seamless、detector=tinymog），
#     和显式传同一组默认参数**逐字节一致**（同一份 _settings_val → 同一个 cfg）
#   · raw=1（或 compose=0）= 老行为：原样返回源帧（界面「原图 / 检测」两视图用）
#   · 没有可用 .dfm 时不报错，退化成原帧并在 note 里说明（老调用不会突然 4xx）
# ============================================================
_COMPOSE_CACHE = collections.OrderedDict()      # 整帧合成结果（含 data URL）
_COMPOSE_CACHE_LOCK = threading.Lock()
_COMPOSE_CACHE_MAX = 6                          # 拖滑块来回切参数时命中率最高的几个


def _source_frame(snap: dict, mode: str, idx: int):
    """按当前打开的东西取第 idx 帧的**原图**（BGR，不缩放）。返回 (图像, 显示名)。

    视频模式走 _grab_frame_at：它复用常开句柄 + 上一帧缓存，比每次重开
    cv2.VideoCapture / 起 ffmpeg 都快，而且和 /preview 用的是同一条取帧路径
    （同一帧走 /preview 和走 /frame，合成输入是同一份像素）。
    """
    if mode == "project":
        names = snap.get("names") or []
        if not names:
            raise _BadInput("还没有打开项目", 409)
        fp = Path(str(snap.get("srcdir"))) / names[idx]
        if not fp.is_file():
            raise _BadInput("图片不见了：%s" % fp, 404)
        img = _imread_any(fp)
        if img is None:
            raise _BadInput("这张图解不开（不是有效图片？）：%s" % fp.name, 422)
        return img, names[idx]

    path = str(snap.get("path") or "")
    if not path:
        raise _BadInput("还没有打开视频", 409)
    try:
        fps = float(snap.get("fps") or 0.0) or 30.0
    except (TypeError, ValueError):
        fps = 30.0
    try:
        img = _grab_frame_at(path, float(idx) / fps, None)
    except _BadInput:
        raise
    except Exception as e:                     # noqa: BLE001
        raise _BadInput("取帧失败（ffmpeg）：%s" % str(e)[:200], 500)
    if img is None:
        raise _BadInput("这一帧读不出来：%d" % idx, 422)
    return img, os.path.basename(path)


def _encode_frame(img, maxw: int, fmt: str, quality: int) -> bytes:
    """整帧 → 字节。先按 maxw 缩（接口一直只用缩略图给界面看），再编码。"""
    import cv2
    h0, w0 = img.shape[:2]
    if w0 > maxw:
        img = cv2.resize(img, (maxw, int(h0 * maxw / w0)), interpolation=cv2.INTER_AREA)
    if str(fmt or "jpg").strip().lower() == "png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise _BadInput("这一帧编码失败", 500)
    return buf.tobytes()


def _source_ident(snap: dict, mode: str, name: str) -> list:
    """「这一帧属于哪份素材」的身份段（合成缓存 / 自动分配缓存共用同一份口径）。"""
    if mode == "project":
        return ["project", str(snap.get("srcdir")), str(name)]
    return ["video"] + list(_grab_key(str(snap.get("path") or "")))


def _sr_level_of(payload: dict) -> int:
    """payload.sr → 0..100（和 knobs._sr_level 同一套解析；缓存身份要用它）。"""
    try:
        from .knobs import _sr_level as _lv
        return int(_lv((payload or {}).get("sr")))
    except Exception:                                      # noqa: BLE001
        return 0


def _compose_sig(snap: dict, mode: str, idx: int, name: str, cfg: dict, dfm: str,
                 models, xseg: str, detector: str, thresh: float,
                 fmt: str, quality: int, maxw: int, no_swap=None,
                 debug: bool = False, sr: int = 0) -> str:
    """合成结果的身份：**参数一个都不能漏**，漏了就是"改了参数画面不动"的老 bug。

    debug 也是身份的一部分：同一帧同一组参数，「合成」和「调试」是两张不同的图，
    缓存键不分家的话切视图会拿到上一张（看着像"点了没反应"）。
    """
    ident = _source_ident(snap, mode, name)
    try:
        tail = json.dumps([int(idx), cfg, str(dfm), str(xseg), str(detector),
                           round(float(thresh), 4),
                           [[m["name"], m["dfm"], list(m["faces"])] for m in (models or [])],
                           str(fmt), int(quality), int(maxw),
                           sorted(int(x) for x in (no_swap or ())),
                           bool(debug), int(sr)],
                          sort_keys=True, default=str)
    except Exception:                          # noqa: BLE001
        tail = repr([idx, cfg, dfm, xseg, detector, thresh, fmt, quality, maxw])
    return json.dumps(ident, default=str) + "|" + tail


def _compose_cached(sig: str):
    if not sig:
        return None
    with _COMPOSE_CACHE_LOCK:
        hit = _COMPOSE_CACHE.get(sig)
        if hit is not None:
            _COMPOSE_CACHE.move_to_end(sig)
    return dict(hit) if hit is not None else None


def _compose_store(sig: str, resp: dict):
    if not sig:
        return
    with _COMPOSE_CACHE_LOCK:
        _COMPOSE_CACHE[sig] = dict(resp)
        _COMPOSE_CACHE.move_to_end(sig)
        while len(_COMPOSE_CACHE) > _COMPOSE_CACHE_MAX:
            _COMPOSE_CACHE.popitem(last=False)


def _merge_query_into(payload: dict, **kw):
    """query 参数并进 payload：**body 优先**（GET 没 body 时全靠它）。"""
    for k, v in kw.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if k in payload and payload[k] not in (None, ""):
            continue
        payload[k] = v


# ============================================================
# 自动分配：人脸 → 组 → dfm（请求没给 models 时）
# ------------------------------------------------------------
# 产品语义（用户定的）：组里的人脸 = 那个组的 dfm。所以合成时「哪张脸用哪个 dfm」
# 不该由左栏选中的那一个模型决定，而应该由**这张脸更像哪个组里的人脸**决定：
#   每张检测到的脸 → ArcFace embedding → 和每个组的均值向量比余弦 → 取最大且 >= 阈值。
# 匹配不上（< 阈值）的脸**不换**（保持原像素），绝不兜给"第 0 个模型"——
# 那等于把别人的人脸换上去，比不换更糟。
#
# 只读不写：这里绝不动 index.json，也不改 .npy（补算走 _face_group 的自己那条路）。
# ============================================================
_AUTO_PLAN_CACHE = collections.OrderedDict()   # 帧身份 → 自动分配结果（拖滑块时不用重算）
_AUTO_PLAN_MAX = 6
# 这份方案里存着「每张脸和各组均值的相似度」= embedding 派生的结果，所以它和组均值
# 属于同一类缓存：进程启动清一次、每次导出结束清一次（见 _face_group.clear_face_group_cache）。
try:
    register_cache_clearer(_AUTO_PLAN_CACHE.clear, size=_AUTO_PLAN_CACHE.__len__,
                           name="auto_plan")
except Exception:                                   # noqa: BLE001 - 登记失败不影响合成
    pass
_AUTO_EMB_LIMIT = 40          # 一次自动分配最多临时补算多少张老脸的特征


def _auto_cache_get(key: str):
    with _COMPOSE_CACHE_LOCK:
        hit = _AUTO_PLAN_CACHE.get(key)
        if hit is not None:
            _AUTO_PLAN_CACHE.move_to_end(key)
    return dict(hit) if hit is not None else None


def _auto_cache_put(key: str, plan: dict):
    if not key:
        return
    with _COMPOSE_CACHE_LOCK:
        _AUTO_PLAN_CACHE[key] = dict(plan)
        _AUTO_PLAN_CACHE.move_to_end(key)
        while len(_AUTO_PLAN_CACHE) > _AUTO_PLAN_MAX:
            _AUTO_PLAN_CACHE.popitem(last=False)


def _auto_library(payload: dict):
    """请求 → 人脸库目录（project > faces_project > 默认库）。拿不到返回 None。

    自动分配是锦上添花：库路径非法/越界就当"没有库"，照旧走单模型老路径，
    绝不因为一个可选的 project 字段把整帧打成 400。
    """
    from .. import _faces as _store
    raw = payload.get("project")
    if raw in (None, ""):
        raw = payload.get("faces_project")
    try:
        return _store.resolve_library(str(raw or "").strip())
    except Exception:                                     # noqa: BLE001
        return None


def _dfm_by_name(name: str):
    """模型名 → workspace/model/<name>.dfm；不存在返回 None（组可能已 orphan）。"""
    nm = str(name or "").strip()
    if nm.lower().endswith(".dfm"):
        nm = nm[:-4]
    if not nm or "/" in nm or "\\" in nm or ".." in nm:
        return None
    path = Path(WORKSPACE) / "model" / (nm + ".dfm")
    return path if path.is_file() else None


def _detect_boxes(frame, detector: str, thresh: float) -> list:
    """整帧检测：和 _preview_job 里那一步**完全同一条**（同一把锁、同一个 thresh）。"""
    det = _detector(detector)
    with _det_run_lock:
        det.thresh = float(thresh)
        try:
            return det.detect(frame) or []
        except Exception as e:                            # noqa: BLE001
            raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)


def _face_vector_from_frame(frame, det_box, box_xyxy):
    """这一帧里那张脸的特征：68 点 → 512 仿射裁 → ArcFace。拿不到返回 None。"""
    from .. import _face_group as _fg
    lm = _frame_landmarks(frame, det_box, box_xyxy)
    if lm is None:
        return None
    try:
        import numpy as np
        from .._align import align_face_68
        crop, _meta = align_face_68(frame, np.asarray(lm, dtype=np.float64),
                                    box_xyxy, 512)
        if crop is None:
            return None
        return _fg.embed_crop(crop)
    except Exception:                                     # noqa: BLE001
        return None


def _auto_face_plan(payload: dict, frame, snap: dict, mode: str, name: str,
                    idx: int, detector: str, thresh: float, boxes=None):
    """按人脸库自动决定每张脸用哪个 dfm。返回 None = 这次不走自动。

    boxes 可选：调用方**已经为这一帧检测过一次**时把结果递进来，这里就不再检测
    第二遍（那一份检测同时还要用来判「这一帧到底有没有脸」，重检一遍纯属浪费）。
    boxes=None 仍然自己检测，老调用方（/api/merge/face-debug）行为一个字都不变。

    None 的三种情形（都退回老的单模型路径，行为一个字都不变）：
      · 库不存在 / 库里的组都没有 dfm；
      · 组里一张有特征的脸都没有（模型缺失、老库还没补算成功）；
      · 这一帧一张脸都没检测到。
    返回体：
      models    [{name, dfm(路径), faces:[脸序号]}]  —— 直接喂给 _preview_multi
      faces     [{idx, group, dfm, similarity, matched, auto:true}]  —— 给前端显示
      unmatched [脸序号]  —— 谁都不像：调用方把它并进 no_swap，保持原像素
      boxes     这次检测的结果（_preview_job 复用它，不再检测第二遍）
    """
    from .. import _face_group as _fg
    from .. import _faces as _store

    library = _auto_library(payload)
    if library is None:
        return None
    try:
        index = _store.load_index(library)
    except _store.FacesError:
        return None
    if not any(str(g.get("dfm") or "").strip() for g in index["groups"]):
        return None

    threshold = _float_val("face_threshold",
                           payload.get("face_threshold",
                                       payload.get("threshold")),
                           0.0, 1.0, _fg.DEFAULT_THRESHOLD)
    try:
        sig = _fg.library_signature(library)
    except Exception:                                     # noqa: BLE001
        sig = str(library)
    key = json.dumps([_source_ident(snap, mode, name), int(idx),
                      round(float(threshold), 4), sig], sort_keys=True, default=str)
    hit = _auto_cache_get(key)
    if hit is not None:
        return hit

    profiles = _fg.group_profiles(library, index, compute_limit=_AUTO_EMB_LIMIT)
    if not profiles:
        return None

    if boxes is None:
        boxes = _detect_boxes(frame, detector, thresh)
    boxes = list(boxes or [])
    if not boxes:
        return None

    faces, by_dfm, unmatched = [], {}, []
    for i, b in enumerate(boxes[:_PREVIEW_MAX_SWAPS]):
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
        except Exception:                                 # noqa: BLE001
            unmatched.append(int(i))
            continue
        vec = _face_vector_from_frame(frame, b, [x1, y1, x2, y2])
        best = _fg.match_vector(vec, profiles, threshold) if vec is not None else None
        matched = bool(best and best.get("matched") and best.get("dfm"))
        path = _dfm_by_name(best["dfm"]) if matched else None
        if path is None:                     # 组还在但 .dfm 没了（orphan）→ 当没匹配
            matched = False
        if matched:
            by_dfm.setdefault(best["dfm"], {"path": path, "faces": []})["faces"].append(int(i))
        else:
            unmatched.append(int(i))
        faces.append({"idx": int(i), "face": int(i),
                      "group": (best.get("group") or "") if best else "",
                      "group_name": (best.get("name") or "") if best else "",
                      "dfm": best["dfm"] if matched else "",
                      "model": best["dfm"] if matched else "",
                      "similarity": (best.get("similarity") if best else None),
                      "matched": bool(matched), "auto": True})

    models = []
    for pos, (nm, info) in enumerate(by_dfm.items()):
        models.append({"name": str(nm), "dfm": str(info["path"]),
                       "faces": sorted(info["faces"]), "index": int(pos)})
    plan = {"models": models, "faces": faces, "unmatched": sorted(set(unmatched)),
            "boxes": boxes, "threshold": float(threshold), "library": str(library),
            "profiles": len(profiles), "detected": len(boxes),
            "matched": sum(1 for f in faces if f["matched"])}
    _auto_cache_put(key, plan)
    return plan


def _auto_faces_merged(worker_faces, plan: dict) -> list:
    """worker 真贴了哪几张 × 自动判定的目标 → 对外那一份 faces（顺序 = 脸序号）。"""
    by_face = {}
    for f in worker_faces or []:
        try:
            by_face[int(f.get("face"))] = f
        except (TypeError, ValueError):
            continue
    out = []
    for item in plan.get("faces") or []:
        rec = dict(item)
        hit = by_face.get(int(item["idx"])) or {}
        if hit.get("box"):
            rec["box"] = hit["box"]
        rec["applied"] = bool(hit)
        out.append(rec)
    return out


def _manual_faces(worker_faces, library=None) -> list:
    """手动指定（models 里点名）时的 faces：auto=false，group 由 dfm 反查。"""
    index = None
    if library is not None:
        try:
            from .. import _faces as _store
            index = _store.load_index(library)
        except Exception:                                 # noqa: BLE001
            index = None
    from .. import _face_group as _fg
    out = []
    for f in worker_faces or []:
        try:
            i = int(f.get("face"))
        except (TypeError, ValueError):
            continue
        # 和自动分配那条路对齐：dfm 记**模型名**（路径另放 dfm_path），
        # 前端/验收拿名字直接比，不用先 basename。
        dfm_path = str(f.get("dfm") or "")
        dfm_name = str(f.get("model") or "") or Path(dfm_path).stem
        out.append({"idx": i, "face": i,
                    "group": _fg.group_id_for_dfm(index, dfm_name) if index else "",
                    "dfm": dfm_name, "model": dfm_name, "dfm_path": dfm_path,
                    "similarity": None, "matched": True, "auto": False,
                    "box": f.get("box")})
    return out



# ---- 调试视图（视图级，独立于融合方式）------------------------------------
# 调试图 = 合成结果 + 4 类调试信息，一个像素都不改回合成链路：
#   1) 每张脸的「遮罩区域」：整块检测框淡绿覆盖（算法认为的脸在哪儿）
#   2) 68 点凸包边界：黄线，真正贴回去的多边形边界（不是矩形）
#   3) 检测框 + 序号：绿框 = 换过了；红框 + keep = 这张没换（no_swap / 点框切的原图）
#   4) 顶部 HUD：帧号 + 生效参数（blend / mask / ct / erode / blur）
# 以前 debug 是塞在 settings.mode 里的一个「假融合方式」（引擎 _MODE_EFFECTIVE 不认，
# 选了也按 overlay 出片）。现在它只由 payload.debug=1 触发，融合方式原样保留。
_DEBUG_TINT_BGR = (60, 200, 120)
_DEBUG_EDGE_BGR = (0, 210, 255)
_DEBUG_BOX_BGR = (90, 220, 120)
_DEBUG_NOSWAP_BGR = (70, 70, 235)
_DEBUG_HUD_BGR = (24, 26, 30)
_DEBUG_HUD_TXT = (120, 240, 150)


def _debug_view_image(out, frame, boxes, no_swap, settings, idx, total):
    """合成结果 → 调试图。返回 (图, 画了几张脸)。

    boxes 是**原帧坐标**里的检测框（_detect_boxes 的格式）；out 可能已经缩到
    maxw，这里按 frame 的尺寸算缩放比，框和边界才不会飘。
    """
    import cv2
    import numpy as np

    if out is None or getattr(out, "ndim", 0) != 3:
        return out, 0
    img = np.ascontiguousarray(out).copy()
    h, w = img.shape[:2]
    try:
        fh, fw = frame.shape[:2]
    except Exception:                                   # noqa: BLE001
        fh, fw = h, w
    sx = (w / float(fw)) if fw else 1.0
    sy = (h / float(fh)) if fh else 1.0

    def _rect_of(b):
        try:
            x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
        except Exception:                               # noqa: BLE001
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        X1 = max(0, min(w - 1, int(round(x1 * sx))))
        Y1 = max(0, min(h - 1, int(round(y1 * sy))))
        X2 = max(X1 + 1, min(w - 1, int(round(x2 * sx))))
        Y2 = max(Y1 + 1, min(h - 1, int(round(y2 * sy))))
        return X1, Y1, X2, Y2

    no_swap = {int(x) for x in (no_swap or ())}
    rects = []
    for i, b in enumerate(boxes or []):
        r = _rect_of(b or {})
        if r is not None:
            rects.append((int(i), b, r))

    # ---- 1) 遮罩区域：整块框淡绿（先整体混一次，比逐框 addWeighted 快也干净）----
    if rects:
        region = img.copy()
        for _i, _b, (X1, Y1, X2, Y2) in rects:
            cv2.rectangle(region, (X1, Y1), (X2, Y2), _DEBUG_TINT_BGR, -1)
        img = cv2.addWeighted(img, 0.88, region, 0.12, 0)

    # ---- 2/3) 68 点凸包边界 + 检测框 ----
    for i, b, (X1, Y1, X2, Y2) in rects:
        try:
            lm = _frame_landmarks(frame, b, [float(v) for v in ((b or {}).get("box") or ())])
        except Exception:                               # noqa: BLE001
            lm = None
        if lm is not None and len(lm) >= 5:
            try:
                pts = np.asarray([[float(p[0]) * sx, float(p[1]) * sy] for p in lm],
                                 dtype=np.float32)
                hull = cv2.convexHull(pts.reshape(-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [hull], True, _DEBUG_EDGE_BGR, 2, cv2.LINE_AA)
            except Exception:                           # noqa: BLE001
                pass
        kept = i in no_swap
        col = _DEBUG_NOSWAP_BGR if kept else _DEBUG_BOX_BGR
        cv2.rectangle(img, (X1, Y1), (X2, Y2), col, 2, cv2.LINE_AA)
        tag = "#%d%s" % (i, " keep" if kept else "")
        ty = (Y1 - 6) if Y1 >= 18 else (Y1 + 16)
        cv2.putText(img, tag, (X1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (12, 12, 12), 3, cv2.LINE_AA)
        cv2.putText(img, tag, (X1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)

    # ---- 4) HUD：帧号 + 生效参数（全是 ASCII，cv2 的 Hershey 字体只能画 ASCII）----
    def _v(key, dflt="-"):
        v = (settings or {}).get(key) if isinstance(settings, dict) else None
        return dflt if v is None or v == "" else str(v)

    hud = "DEBUG  idx=%s/%s  blend=%s  mask=%s  ct=%s  erode=%s  blur=%s  faces=%d" % (
        idx, (total if total not in (None, "") else "?"), _v("blend"), _v("mask_mode"),
        _v("ct_mode"), _v("mask_erode"), _v("mask_blur"), len(rects))
    scale = max(0.38, min(0.62, w / 1700.0))
    bar = int(round(24 * scale)) + 8
    cv2.rectangle(img, (0, 0), (w, bar), _DEBUG_HUD_BGR, -1)
    cv2.putText(img, hud, (8, bar - int(round(7 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                scale, _DEBUG_HUD_TXT, 1, cv2.LINE_AA)
    cv2.rectangle(img, (1, 1), (w - 2, h - 2), _DEBUG_EDGE_BGR, 2, cv2.LINE_AA)
    return img, len(rects)


def _frame_payload(idx: int, payload: dict, default_w: int = 900) -> dict:
    """第 idx 帧 → 按 payload 里的合成参数出图（data URL）。

    payload 认的键（全部可选）：
        w / fmt / quality / raw / compose / debug
        settings: {ct_mode, blend, mask_mode, mask_erode, mask_blur, …}
        model: "<dfm 名>"        models: [{"name","dfm","faces"}]
        detector / thresh / xseg_onnx
    raw=1（或 compose=0）= 原图视图，不合成；debug=1 = 调试视图（合成 + 调试叠加）。
    settings.blend 只认 4 个融合方式（overlay / hist-match / seamless /
    seamless-hist-match），其它值一律 400 —— 视图不进 settings。

    **两个短路（都是原帧，不碰 worker）**：
      · 没点名模型（models / model / dfm 一个都没有）→ no_model=true，不做
        「挑第一个 .dfm」那种代用户决定的事，响应 note="没有选择模型，返回原帧"；
      · 这一帧一张脸都没检出 → no_faces=true，跳过自动分配（它还要再检测一次）。
    三个布尔字段的语义固定为：
        有人脸 + 有模型 → has_face=true,  swapped=true,  no_faces=false, no_model=false
        有人脸 + 无模型 → has_face=true,  swapped=false, no_faces=false, no_model=true
        无人脸         → has_face=false, swapped=false, no_faces=true
    检测整条链路只做一次（调用方这份 boxes 直接交给 worker 和自动分配复用）。
    """
    # 预览也要吃「人脸超分」滑块：以前只有导出会写 env，滑到 512·α1.00 预览却看不出
    # 任何变化（用户报的「前端合成没有 GPEN 效果」）。payload.sr = 滑块值 0..100。
    try:
        from .knobs import apply_sr as _apply_sr
        _apply_sr(payload.get("sr") if isinstance(payload, dict) else None)
    except Exception:                                      # noqa: BLE001
        pass

    t_start = time.time()
    payload = _as_dict(payload)
    maxw = max(200, min(int(payload.get("w") or default_w or 900), 1600))
    if idx < 0:                          # 参数错先说参数错，别被“没打开素材”盖住
        raise _BadInput("帧号不能是负数：%d" % idx)

    snap = _cache_snap()
    mode = _check_cache_ready(snap)
    idx = _check_idx(snap, idx)
    total = int(snap.get("frames") or 0) or len(snap.get("names") or [])

    fmt = "png" if str(payload.get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    quality = _int_val("quality", payload.get("quality"), 1, 100,
                       88 if fmt == "jpg" else 92)
    raw_only = _bool_val(payload.get("raw"), False) or (
        "compose" in payload and not _bool_val(payload.get("compose"), True))
    # 视图级的「调试」：独立参数（不是融合方式）。raw 优先级更高 —— 原图就是原图，
    # 一个调试像素都不该有。
    debug_view = (not raw_only) and _bool_val(payload.get("debug"), False)

    t_grab = time.time()
    frame, name = _source_frame(snap, mode, idx)
    grab_ms = round((time.time() - t_grab) * 1000.0, 1)

    # ---- 1) raw：老行为，一个字节都不动（原图视图要它）----
    if raw_only:
        buf = _encode_frame(frame, maxw, fmt, quality)
        mime = "image/png" if fmt == "png" else "image/jpeg"
        return _ok(idx=idx, total=total, name=name, raw=True, composed=False,
                   fmt=fmt, bytes=len(buf), ms=int(round((time.time() - t_start) * 1000.0)),
                   png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))

    # ---- 2) 参数（缺省 = 当前默认值，和显式给默认值走同一条路）----
    settings = _settings_val(payload.get("settings"))
    cfg = _merge_settings_to_config(settings)
    # 检测器：顶层 detector 优先，其次 settings.detector（界面把它塞在 settings 里）
    detector = _detector_val(payload.get("detector") or settings.get("detector"))
    thresh = _float_val("thresh", payload.get("thresh"), 0.0, 1.0, 0.5)
    models = _models_of(payload)
    # ---- 2a) 请求里到底有没有点名模型？三种写法都算：models 数组 / model 名 / dfm 路径 ----
    # 一个都没有 = **用户没选模型**。以前这里会掉进 _dfm_of 的老兜底：悄悄拿
    # workspace/model 里的第一个 .dfm 就合成，用户没选模型却看到一张换过的脸，
    # 而且谁也不知道用的是哪个模型。现在这种情况直接回原帧（见下面 2d / 2e）。
    asked_model = bool(models) or bool(str(payload.get("model") or "").strip()) \
        or bool(str(payload.get("dfm") or "").strip())
    # 自动分配只对「没点名模型」的请求有意义：点名了就以手动为准，一个字节都不参与。
    auto_on = (not models) and _bool_val(payload.get("auto_faces",
                                                     payload.get("auto")), True)

    # 点名"不换"的脸（前端「不换脸」勾选）。没给 = 老行为（未声明的脸兜给第 0 个模型）。
    no_swap = _face_ids_val("no_swap_faces",
                            payload.get("no_swap_faces",
                                        payload.get("skip_faces",
                                                    settings.get("no_swap_faces"))))
    dfm, model_name = None, ""

    def _short_circuit(img, note, *, has_face, no_model, no_faces, detect_ms=0.0,
                       dbg_boxes=None, extra=None):
        """不合成的那两条路（no_model / no_faces）→ **还是原帧**，只是换个编码。

        不碰 worker、不跑 dfm、不重检测：这里只剩「画调试信息（可选）+ 编码」，
        所以耗时和 raw=1 那一条是同一量级。三个字段按语义写死：
        has_face / no_model / no_faces，前端不用再靠 note 猜。
        """
        t_enc = time.time()
        out_img, dbg_n = img, 0
        if debug_view and dbg_boxes:
            # 没有模型也能看调试视图：框 / 遮罩区域 / HUD 全都在，只是没有换脸结果。
            out_img, dbg_n = _debug_view_image(out_img, frame, list(dbg_boxes), [],
                                               settings, idx, total)
        buf = _encode_frame(out_img, maxw, fmt, quality)
        enc_ms = round((time.time() - t_enc) * 1000.0, 1)
        mime = "image/png" if fmt == "png" else "image/jpeg"
        resp = _ok(idx=int(idx), total=int(total), name=name, raw=True,
                   composed=False, swapped=False, has_face=bool(has_face),
                   no_model=bool(no_model), no_faces=bool(no_faces), faces=[],
                   fmt=fmt, bytes=len(buf),
                   ms=int(round((time.time() - t_start) * 1000.0)),
                   timings={"grab_ms": grab_ms,
                            "detect_ms": round(float(detect_ms), 1),
                            "encode_ms": enc_ms, "swap_ms": None, "worker": 0},
                   detector=detector, thresh=float(thresh), settings=settings,
                   debug=bool(debug_view), debug_faces=int(dbg_n), note=note,
                   png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))
        if model_name:
            resp["model"] = model_name
            resp["dfm"] = str(dfm)
        if extra:
            resp.update(extra)
        return resp

    def _cache_hit(hit):
        # 缓存里放的是共享 dict：拷一份再改 ms / cached，别把缓存本身写脏
        hit = dict(hit)
        hit["cached"] = True
        hit["ms"] = int(round((time.time() - t_start) * 1000.0))
        tm = dict(hit.get("timings") or {})
        tm["cache_hit"] = 1
        hit["timings"] = tm
        return hit

    # ---- 2b) 给了 models 数组：先解析 dfm（不存在就照实 404），先试一次缓存 ----
    # 只有显式 models 能在检测**之前**查缓存：别的写法下自动分配还要参与，
    # 结果和"单模型全帧换"不是同一张图（sig 分不开），必须先跑完自动分配。
    if models:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
        xseg = _xseg_of(payload)
        sig = _compose_sig(snap, mode, idx, name, cfg, str(dfm), models, xseg,
                           detector, float(thresh), fmt, int(quality), int(maxw),
                           no_swap, debug=debug_view,
                           sr=_sr_level_of(payload))
        hit = _compose_cached(sig)
        if hit is not None:
            return _cache_hit(hit)
    else:
        xseg, sig = "", ""

    # ---- 2c) 检测**一次**：自动分配 / 「这一帧有没有脸」/ worker 换脸 共用这一份 ----
    t_det = time.time()
    boxes = list(_detect_boxes(frame, detector, thresh))
    detect_ms = round((time.time() - t_det) * 1000.0, 1)

    # ---- 2d) 一张脸都没检出 → 原帧（no_faces）：不自动分配、不进 worker ----
    # 以前这里照样往下走：_auto_face_plan 内部再检测一遍，然后整帧编码回一张
    # 和原图一模一样的图。现在最贵的一步只剩「检测 + 编码」。
    if not boxes:
        return _short_circuit(frame, "这一帧没检出人脸：返回原帧",
                              has_face=False, no_model=not asked_model,
                              no_faces=True, detect_ms=detect_ms)

    # ---- 2e) 没给 models 数组 → 按人脸库自动分配（复用上面那一份检测结果）----
    # 和以前一样：**只有 models 数组为空才走自动**（左栏那个 model 只是"库判不出来
    # 时"的老兜底，不参与自动判定）。
    auto_plan = None
    if not models and auto_on:
        auto_plan = _auto_face_plan(payload, frame, snap, mode, name, int(idx),
                                    detector, thresh, boxes=boxes)
        if auto_plan is not None:
            models = auto_plan["models"] or None

    auto_extra = None
    if auto_plan is not None:
        # 前端要照这份清单显示每张脸的相似度（auto=true = 判定来自人脸库）
        auto_extra = {"auto": True, "faces": list(auto_plan["faces"]),
                      "face_threshold": float(auto_plan["threshold"]),
                      "face_library": str(auto_plan.get("library") or ""),
                      "auto_matched": int(auto_plan.get("matched") or 0),
                      "auto_detected": int(auto_plan.get("detected") or 0)}

    if not models:
        if auto_plan is not None:
            # 自动分配判定过、但一张脸都没匹配上：**不能退回"默认模型全帧换"**，
            # 那等于把不相干的人换上去。整帧保持原像素，判定结果照样带出去。
            return _short_circuit(
                frame, "没有可用的 .dfm 模型：只回原帧（自动分配：这一帧没有和任何人脸组"
                       "匹配上的脸）", has_face=True, no_model=not asked_model,
                no_faces=False, detect_ms=detect_ms, dbg_boxes=boxes, extra=auto_extra)
        if not asked_model:
            # 请求里没有任何模型信息 = 用户没选模型：**绝不替用户挑一个 .dfm** ——
            # 那等于用了一个他不知道的模型，还把别人的脸换上去。
            return _short_circuit(frame, "没有选择模型，返回原帧",
                                  has_face=True, no_model=True, no_faces=False,
                                  detect_ms=detect_ms, dbg_boxes=boxes)
        # 点名了 model / dfm：按老规矩解析（不存在就 404），然后正常合成。
        dfm, model_name = _dfm_of(payload)
        xseg = _xseg_of(payload)
    else:
        dfm, model_name = Path(models[0]["dfm"]), models[0]["name"]
        xseg = _xseg_of(payload)

    if auto_plan is not None:
        # 谁都不像的脸：并进 no_swap，保持原像素（不换 > 换错）
        no_swap = sorted(set(int(x) for x in no_swap)
                         | set(int(x) for x in (auto_plan.get("unmatched") or [])))

    # ---- 3) 命中缓存就直接回（同一帧 + 同一组参数，拖滑块回退时立刻出图）----
    if not sig:
        sig = _compose_sig(snap, mode, idx, name, cfg, str(dfm), models, xseg,
                           detector, float(thresh), fmt, int(quality), int(maxw),
                           no_swap, debug=debug_view,
                           sr=_sr_level_of(payload))
        hit = _compose_cached(sig)
        if hit is not None:
            return _cache_hit(hit)

    # ---- 4) 真合成：和 /preview、/preview-frame 共用 _preview_job ----
    stop = threading.Event()
    state = {"stage": "start"}
    # fmt=png 让 worker 直接给无损的中间结果（逐参数对比要看真实像素差，
    # 不能被 JPEG 量化噪声盖住）；jpg 就用 worker 编好的那一份，不再二次压缩。
    want_png = (fmt == "png")
    res = _preview_job({"frame": frame, "models": models, "dfm": dfm, "xseg": xseg,
                        "config": cfg, "detector": detector, "thresh": float(thresh),
                        "stop": stop, "state": state,
                        "fmt": "png" if want_png else "jpg",
                        "quality": int(quality),
                        "no_swap": list(no_swap),
                        # 这一帧唯一的那次检测（2c）直接交给 worker：它不再检测第二遍，
                        # 自动分配也是拿这份 boxes 算的 embedding —— 全链路只检测一次。
                        "boxes": list(boxes),
                        # png 请求要的是"能逐像素对比"的结果 → 帧无损发给 worker
                        "send_lossless": bool(want_png)})
    img = res.get("img") or res.get("jpg") or b""
    if not img:
        raise _BadInput("这一帧没有产出图", 500)
    import cv2
    import numpy as np
    out = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    # ---- 人脸超分（在**最终这张图**上重新检测一次，按检测框做）----
    # 为什么不用上游的 box：worker 回的图可能被缩过，而 box 是原帧坐标 ——
    # 换算错一步，增强就会跑到画面别处（用户报的「不在脸上」）。
    # 直接在当前这张图上检一次（~35ms），坐标天然对齐，代价可以接受。
    try:
        from . import facesr as _sr
        _lv = _sr_level_of(payload)
        if _lv > 0 and _sr.enabled():
            from .knobs import apply_sr as _apply_sr
            _apply_sr(_lv)
            _bs = _detect_boxes(out, detector, thresh) or []
            for _b in _bs:
                try:
                    _x, _y, _bw, _bh = [int(round(float(v))) for v in (_b.get("box") or _b)]
                except Exception:                              # noqa: BLE001
                    continue
                if _bw <= 8 or _bh <= 8:
                    continue
                _x = max(0, min(_x, out.shape[1] - _bw))
                _y = max(0, min(_y, out.shape[0] - _bh))
                _m = np.zeros((_bh, _bw), np.float32)
                cv2.ellipse(_m, (_bw // 2, _bh // 2),
                            (max(1, _bw // 2 - 2), max(1, _bh // 2 - 2)), 0, 0, 360, 1.0, -1)
                _m = cv2.GaussianBlur(_m, (0, 0), max(1.0, min(_bw, _bh) * 0.03))
                out[_y:_y + _bh, _x:_x + _bw] = _sr.apply_roi(
                    out[_y:_y + _bh, _x:_x + _bw], _m, _sr.strength())
            if not want_png:
                img = _encode_frame(out, maxw, fmt, quality)
    except Exception as _e:                                    # noqa: BLE001
        print("[sr] 预览跳过（%s: %s）" % (_e.__class__.__name__, str(_e)[:120]), flush=True)
        raise _BadInput("合成结果解不开", 500)
    # 先缩到 maxw 再画调试信息：字号/线宽按最终展示尺寸给，图缩一半时框线才不会细成一条。
    resized = False
    if out.shape[1] > maxw:
        out = cv2.resize(out, (maxw, max(1, int(out.shape[0] * maxw / out.shape[1]))),
                         interpolation=cv2.INTER_AREA)
        resized = True
    dbg_faces = 0
    if debug_view:
        out, dbg_faces = _debug_view_image(out, frame, boxes, no_swap,
                                           settings, idx, total)
    # 什么时候要重编：png（无损）· 刚缩过图 · 画了调试信息（不重编等于白画）。
    # 老逻辑是「jpg 且没超出 maxw 就复用 worker 编好的那一份」—— 现在 out 可能
    # 已经被我们缩过/画过，必须显式重编，不能再用 worker 那张原图。
    if want_png or resized or debug_view:
        img = _encode_frame(out, maxw, fmt, quality)
    mime = "image/png" if fmt == "png" else "image/jpeg"

    # timings 只记一次检测：worker 那份是 reuse（它没再检测），detect_ms 换成 2c 实测的数字
    timings = dict(res.get("timings") or {})
    timings["grab_ms"] = grab_ms
    timings["detect_ms"] = detect_ms
    timings["detect_reused"] = 1             # worker 用的是 2c 那一份，不是它自己检的
    resp = _ok(idx=int(idx), total=int(total), name=name, composed=True,
               swapped=bool(res.get("swapped")), has_face=bool(res.get("has_face")),
               no_model=False, no_faces=bool(res.get("no_faces")),
               box=res.get("box"), fmt=fmt, bytes=len(img),
               ms=int(round((time.time() - t_start) * 1000.0)),
               model=model_name, dfm=str(dfm), detector=detector,
               thresh=float(thresh), settings=settings, effective=cfg,
               mask_mode=int(cfg.get("mask_mode") or 0),
               mode=cfg.get("mode"), ct_mode=cfg.get("color_transfer_mode"),
               erode_mask=cfg.get("erode_mask_modifier"),
               blur_mask=cfg.get("blur_mask_modifier"),
               xseg=xseg, timings=timings,
               no_swap_faces=list(no_swap), skipped=list(res.get("skipped") or []),
               cached=False, note=res.get("note") or "",
               debug=bool(debug_view), debug_faces=int(dbg_faces),
               png="data:%s;base64,%s" % (mime, _b64.b64encode(img).decode("ascii")))
    if models:
        resp["models"] = [{"name": m["name"], "dfm": m["dfm"],
                           "faces": list(m["faces"])} for m in models]
        resp["models_used"] = list(res.get("models_used") or [])
        if auto_plan is not None:
            # 自动分配：faces = 每张检测到的脸 + 它被判定给了哪个组/dfm/相似度
            resp["faces"] = _auto_faces_merged(res.get("faces"), auto_plan)
            resp["auto"] = True
            resp["face_threshold"] = float(auto_plan["threshold"])
            resp["face_library"] = str(auto_plan.get("library") or "")
            resp["auto_matched"] = int(auto_plan.get("matched") or 0)
            resp["auto_detected"] = int(auto_plan.get("detected") or 0)
        else:
            # 手动指定（chip 上有值）：auto=false，group 由 dfm 反查
            resp["faces"] = _manual_faces(res.get("faces"), _auto_library(payload))
            resp["auto"] = False
        resp["swaps"] = int(res.get("swaps") or 0)
        # 多人素材排查用：一张脸没换时，这三组数对不上就能立刻定位是"没检测到"、
        # "没分配上"还是"worker 没换"。DFL_DEBUG_FACES=1 打开。
        if os.environ.get("DFL_DEBUG_FACES"):
            _det = len(boxes or [])
            _want = sum(len(m.get("faces") or []) for m in (models or []))
            _got = len(res.get("faces") or [])
            print("[faces] frame=%s 检测=%d 分配=%d 换=%d · models=%s · got=%s"
                  % (idx, _det, _want, int(res.get("swaps") or 0),
                     [(m.get("name"), list(m.get("faces") or [])) for m in (models or [])],
                     [(f.get("idx"), f.get("dfm_name") or f.get("model") or "") for f in (res.get("faces") or [])]),
                  flush=True)
    _compose_store(sig, resp)
    return resp


def _body_idx(payload: dict) -> int:
    """帧号从 body 里取：idx > frame > n（三种写法前端都在用）。"""
    for k in ("idx", "frame", "n", "index", "frame_idx"):
        if payload.get(k) not in (None, ""):
            return _int_val("idx", payload.get(k), 0, None, 0)
    raise _BadInput("缺少帧号：body 里给 idx（也认 frame / n），或者用 /api/merge/frame/{idx}")


def _frame_body(idx: int, body, **query) -> dict:
    """POST body + query 合成一个 payload（body 优先）。"""
    payload = dict(_as_dict(body))
    _merge_query_into(payload, **query)
    return payload


# ============================================================
# 单脸调试：GET / POST /api/merge/face-debug
# ------------------------------------------------------------
# 一张脸的「六宫格」证据链：两张模型遮罩、它们的合并、预测脸、对齐后的原脸、
# 以及真的合成出来的那张脸。前端自己排版（3 列 × 2 行），所以这里返回**六张
# 独立图**而不是拼好的一整张 —— 点开放大、单格存图都靠这个。
#   第 1 行：(1) mask(dfm-pred)  (2) mask(dst)  (3) merged(mask(dfm-pred), mask(dst))
#   第 2 行：(4) dfm-pred        (5) dst-original (6) merged
# 图从哪来：**复用预览 worker**（同一份 MergeMaskedFace、同一批常驻 DFM），
# 只多跑一次单脸合成 + 把中间量截下来，不重新加载模型、不另起进程。
# 参数语义和 /api/merge/frame 完全一致（settings / models / detector …），
# 唯一的例外：**idx 是「第几张脸」**，帧号只认 frame / n（见 _debug_face_idx）。
# ============================================================
_FACE_DBG_CACHE = collections.OrderedDict()     # 同一帧同一张脸 + 同一组参数 → 六格图
_FACE_DBG_MAX = 8
_FACE_DBG_LOCK = threading.Lock()


def _debug_face_idx(payload: dict) -> int:
    """这张脸是这一帧里的第几张（检测顺序，从 0 起）。

    帧号只认 frame / n / frame_idx：本接口的 idx 按契约是**人脸序号**，
    和老接口的「idx = 帧号」正好相反，所以这里绝不走 _body_idx()。
    """
    for k in ("face", "face_idx", "face_index", "idx", "i"):
        if payload.get(k) not in (None, ""):
            return _int_val("face", payload.get(k), 0, 63)
    raise _BadInput("缺少人脸序号：给 face（也认 face_idx / idx）—— 这一帧里第几张脸，从 0 起")


def _debug_frame_idx(payload: dict) -> int:
    for k in ("frame", "n", "frame_idx"):
        if payload.get(k) not in (None, ""):
            return _int_val("frame", payload.get(k), 0, None)
    return 0


def _face_debug_payload(payload: dict) -> dict:
    """一张脸 → 六张图（data URL）。"""
    t_start = time.time()
    payload = _as_dict(payload)
    panel_size = max(96, min(int(payload.get("panel_size") or 256), 512))

    snap = _cache_snap()
    mode = _check_cache_ready(snap)
    idx = _check_idx(snap, _debug_frame_idx(payload))
    total = int(snap.get("frames") or 0) or len(snap.get("names") or [])
    face_i = _debug_face_idx(payload)
    frame, name = _source_frame(snap, mode, idx)

    settings = _settings_val(payload.get("settings"))
    cfg = _merge_settings_to_config(settings)
    detector = _detector_val(payload.get("detector") or settings.get("detector"))
    thresh = _float_val("thresh", payload.get("thresh"), 0.0, 1.0, 0.5)

    # ---- 这一帧的脸都在哪：有 models 就照 models 分配，没有就按人脸库自动判 ----
    models = _models_of(payload)
    auto_plan = None
    if not models:
        auto_plan = _auto_face_plan(payload, frame, snap, mode, name, int(idx),
                                    detector, thresh)
        if auto_plan is not None:
            models = auto_plan["models"] or None
    boxes = list((auto_plan or {}).get("boxes") or [])
    if not boxes:
        try:
            boxes = list(_detect_boxes(frame, detector, thresh))
        except Exception:                                 # noqa: BLE001
            boxes = []
    if not boxes:
        raise _BadInput("这一帧没检测到人脸 —— 换一帧再点调试", 404)
    if face_i >= len(boxes):
        raise _BadInput("这一帧只有 %d 张脸，要的是第 %d 张（从 0 起）"
                        % (len(boxes), face_i), 404)

    # ---- 这张脸用哪个 dfm：models 点名 → 自动判定 → 显式 dfm/model ----
    dfm, model_name = None, ""
    if models:
        mi = _model_for_face(face_i, models)
        m = models[mi]
        # 只在「这个模型确实管这张脸」时才认（faces 空 = 管所有脸；点名了但没这张 = 不管）
        named = (not (m.get("faces") or [])) or (int(face_i) in (m.get("faces") or []))
        if named:
            dfm, model_name = Path(m["dfm"]), str(m["name"])
    if dfm is None:
        try:
            dfm, model_name = _dfm_of(payload)
        except _BadInput:
            dfm, model_name = None, ""
    if dfm is None:
        raise _BadInput("第 %d 张脸没有可用的 .dfm（谁的组都不像）—— 先在框上的 chip 里"
                        "给它选一个模型，再看六宫格" % face_i, 409)

    b = boxes[face_i]
    try:
        x1, y1, x2, y2 = [float(v) for v in (b.get("box") or ())]
    except Exception:                                     # noqa: BLE001
        raise _BadInput("第 %d 张脸的检测框不合法" % face_i, 500)
    lm = _frame_landmarks(frame, b, [x1, y1, x2, y2])
    if lm is None:
        raise _BadInput("第 %d 张脸算不出 68 点 —— 换一帧再试" % face_i, 409)
    box_wh = [int(round(x1)), int(round(y1)),
              int(round(max(0.0, x2 - x1))), int(round(max(0.0, y2 - y1)))]

    xseg = _xseg_of(payload)
    key = json.dumps([_source_ident(snap, mode, name), int(idx), int(face_i),
                      str(dfm), cfg, xseg, int(panel_size), _sr_level_of(payload)], sort_keys=True, default=str)
    with _FACE_DBG_LOCK:
        hit = _FACE_DBG_CACHE.get(key)
        if hit is not None:
            _FACE_DBG_CACHE.move_to_end(key)
            out = dict(hit)
            out["cached"] = True
            out["ms"] = int(round((time.time() - t_start) * 1000.0))
            return out

    import numpy as np
    # 帧走裸字节块（PNG，无损）：六宫格要的是"脸发蒙时最直接的证据"，缩放/有损编码
    # 都会把证据毁掉，所以这里仍旧无损，只是不再 base64。
    _enc, _blob, _h, _w = _send_frame_bytes(frame, {"send_lossless": True})
    resp = _WORKER.call({
        "cmd": "face_debug",
        "dfm": str(dfm),
        "landmarks": np.asarray(lm, dtype=np.float64).round(2).tolist(),
        "box": list(box_wh),
        "config": dict(cfg),
        "xseg": xseg,
        "panel_size": int(panel_size),
    }, [_blob])
    if not resp.get("ok"):
        raise _BadInput("六宫格出图失败：%s" % (resp.get("error") or "预览 worker 没给原因"), 500)

    panels = []
    for item in resp.get("panels") or []:
        raw = item.get("png") or ""
        if not raw:
            continue
        panels.append({"key": str(item.get("key") or ""),
                       "label": str(item.get("label") or ""),
                       "w": int(item.get("w") or panel_size),
                       "h": int(item.get("h") or panel_size),
                       "png": "data:image/png;base64," + raw})
    # 最后一格「merged」按滑块过一遍超分：调试页要看的就是「GPEN 之后」的样子。
    # 这一格是**脸本身**（panel_size），不是整帧，所以直接按整块 + 椭圆遮罩做。
    try:
        from .knobs import apply_sr as _apply_sr
        _lv = int(_apply_sr(_sr_level_of(payload)))
        if _lv > 0:
            import base64 as _b64d
            import cv2 as _cv2
            from . import facesr as _sr
            for _pn in panels:
                if str(_pn.get("label") or "").strip() != "merged":
                    continue
                _raw = _pn["png"].split(",", 1)[-1]
                _img = _cv2.imdecode(np.frombuffer(_b64d.b64decode(_raw), np.uint8),
                                    _cv2.IMREAD_COLOR)
                if _img is None:
                    continue
                _h, _w = _img.shape[:2]
                _m = np.zeros((_h, _w), np.float32)
                _cv2.ellipse(_m, (_w // 2, _h // 2), (max(1, _w // 2 - 2), max(1, _h // 2 - 2)),
                            0, 0, 360, 1.0, -1)
                _m = _cv2.GaussianBlur(_m, (0, 0), max(1.0, min(_w, _h) * 0.02))
                _img = _sr.apply_roi(_img, _m, _sr.strength())
                _ok2, _buf = _cv2.imencode(".png", _img)
                if _ok2:
                    _pn["png"] = "data:image/png;base64," + _b64d.b64encode(_buf).decode("ascii")
    except Exception as _e:                                # noqa: BLE001
        print("[sr] 调试页跳过（%s: %s）" % (_e.__class__.__name__, str(_e)[:120]), flush=True)
    if len(panels) != 6:
        raise _BadInput("六宫格只出了 %d 张（要 6 张）—— worker 版本对不上？" % len(panels), 500)

    out = _ok(frame=int(idx), total=int(total), name=name, size=panel_size,
              face={"idx": int(face_i), "box": box_wh,
                    "score": (b.get("score") if isinstance(b, dict) else None),
                    "model": model_name, "dfm": str(dfm)},
              panels=panels, stats=resp.get("stats") or {},
              settings=settings, detector=detector, xseg=xseg,
              model=model_name, dfm=str(dfm), auto=bool(auto_plan is not None),
              ms=int(round((time.time() - t_start) * 1000.0)), cached=False)
    with _FACE_DBG_LOCK:
        _FACE_DBG_CACHE[key] = out
        while len(_FACE_DBG_CACHE) > _FACE_DBG_MAX:
            _FACE_DBG_CACHE.popitem(last=False)
    return out


def _merged_from_export(idx: int, maxw: int, payload: dict = None) -> dict:
    """老行为：从「最近一次导出的成片」里抠一帧（没有打开的素材时才走这条）。"""
    with _merge_job_lock:
        last_out = str(_merge_job.get("out") or "").strip()
    cands = []
    if last_out:
        cands.append(Path(last_out))
    snap = _cache_snap()
    d = str(snap.get("dir") or "").strip()
    if d:                                    # 重启之后 _merge_job 是空的 → 按默认命名找
        cands.append(Path(d + "_merged.mp4"))
        cands.append(Path(d) / "merged.mp4")
    out = next((p for p in cands if p and p.is_file()), None)
    if out is None:
        raise _BadInput("还没有导出的成片 —— 先点一次导出（或者先打开素材再看合成）", 409)

    import cv2
    cap = cv2.VideoCapture(str(out))
    if not cap.isOpened():
        raise _BadInput("打不开成片：%s" % out, 500)
    total = 0
    try:
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total = 0
        if total and idx >= total:
            raise _BadInput("帧号越界：%d，成片只有 %d 帧（0~%d）"
                            % (idx, total, total - 1), 404)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        good, frame = cap.read()
        if not good or frame is None:
            raise _BadInput("成片里没有第 %d 帧" % idx, 404)
    finally:
        cap.release()

    fmt = "png" if str((payload or {}).get("fmt") or "jpg").strip().lower() == "png" else "jpg"
    buf = _encode_frame(frame, maxw, fmt, 88)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return _ok(idx=int(idx), total=total, out=str(out), source="export",
               composed=False, fmt=fmt, bytes=len(buf),
               png="data:%s;base64,%s" % (mime, _b64.b64encode(buf).decode("ascii")))


def _frame_image(idx: int):
    """给检测 / 分析取一帧：先做形位检查，别让 app._merge_frame_image 的 500 冒出来。"""
    if idx < 0:
        raise _BadInput("帧号不能是负数：%d" % idx)
    snap = _cache_snap()
    _check_cache_ready(snap)
    idx = _check_idx(snap, idx)
    if _cache_mode(snap) == "project":
        fp = Path(str(snap.get("srcdir"))) / (snap.get("names") or [])[idx]
        if not fp.is_file():
            raise _BadInput("图片不见了：%s" % fp, 404)
    try:
        frame, total = _merge_frame_image(idx)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("取帧失败：%s" % str(e)[:200], 500)
    if frame is None:
        raise _BadInput("这一帧读不出来：%d" % idx, 422)
    return frame, total


def _detector_val(raw) -> str:
    """检测器入参 → 规范名（tinymog / retinaface_10g）；缺省 = tinymog。

    只有白名单里的两个名字能过。retinaface_500m / det_500m **明确 400**：
    这个模型不存在（4.1 已移除），以前它靠别名的兜底也能"成功"，属于静默回退。
    """
    key = str(raw or "").strip()
    low = key.lower().replace("-", "_")
    if low.endswith(".onnx"):
        low = low[:-len(".onnx")]
    if not low:
        return _DEFAULT_DETECTOR
    if low in _REMOVED_DETECTORS:
        raise _BadInput("检测器 %s 不存在（4.1 已移除 RetinaFace_500m），可选：%s"
                        % (key, " / ".join(_DETECTORS)))
    canon = _DET_ALIAS_API.get(low)
    if canon is None:
        raise _BadInput("未知检测器 %r，只认：%s" % (key, " / ".join(_DETECTORS)))
    return canon


def _detector(name=None):
    key = _detector_val(name)
    had = _det_cached(key)            # 已经在父进程里 = 不再占新显存，也不用量
    _own_f0 = None if had else _vram_measure_begin()
    try:
        with _det_load_lock:              # 加载要几百毫秒，别让并发请求各建一份
            det = _get_detector(key)
    except HTTPException as e:
        raise _BadInput(str(e.detail), e.status_code)
    except Exception as e:
        raise _BadInput("检测器加载失败（%s）：%s" % (key, str(e)[:160]), 500)
    if _own_f0 is not None:
        _vram_measure_end("detector:" + key, _own_f0)
    return det


def _with_kps68(frame, boxes):
    # 给每个检测框补一条 kps68（原图坐标，68 行 2 列，两位小数）。
    #
    # 为什么在这里补：人脸库要的是 68 点仿射（5 点抖动大，旋转/缩放不稳），
    # 而前端手上只有检测框。让检测回包把 68 点直接带出去，前端点「＋」时
    # 就是纯搬运 —— extract 和 add 拿到的是同一份数，像素差才是 0。
    import numpy as np
    out = []
    for b in (boxes or []):
        try:
            item = dict(b)
        except Exception:
            out.append(b)
            continue
        lm = None
        try:
            lm = _frame_landmarks(frame, b, b.get("box"))
        except Exception:
            lm = None
        if lm is not None:
            try:
                item["kps68"] = np.asarray(lm, dtype=np.float64).round(2).tolist()
            except Exception:
                pass
        out.append(item)
    return out


_LM106 = {"m": None, "tried": False}


def _landmarks_model():
    """2D106 关键点模型（68 点由它回归出来），本进程只加载一次。

    **不传给 app._get_landmarks106()**：那个会把 workspace/engine_name 一起传下去，
    _engines.try_engine 会命中预编译的 TensorRT 引擎，_trt_session.run 走
    torch.cuda.CUDAGraph.replay —— CUDA 图是在「捕获它的那个线程」的流上建的，
    预览是每个请求一个新线程，跨线程 replay 直接把整个进程打崩（实测 0xC0000005
    access violation，第一次请求没事、第二次就死）。这里用纯 onnxruntime + CPU：
    CPU 实测 4~5ms/张，够快，也不会碰 CUDA 图。
    """
    if _LM106["tried"]:
        return _LM106["m"]
    _LM106["tried"] = True
    try:
        from .._landmarks106 import InsightFace106
        from .._paths import weights_dir
        # 平铺布局：<weights>/insightface_2d106.onnx；老的分目录布局作兜底
        for base in (weights_dir(), Path(__file__).resolve().parents[3] / "weights"):
            for rel in ("insightface_2d106.onnx",
                        "onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(base) / rel
                if cand.is_file():
                    _LM106["m"] = InsightFace106(str(cand),
                                                 providers=["CPUExecutionProvider"])
                    return _LM106["m"]
    except Exception:
        traceback.print_exc()
    _LM106["m"] = None
    return None


def _frame_landmarks(frame, det_box, box_xyxy):
    """检测框/5 点 → 原图坐标的 68 点。**只认 2D106 真的回归出来的点。**

    跟 _align_stream（抽帧+对齐那条路）同一个套路：5 点先粗对齐一次（只为了给
    模型一个像样的输入），2D106 回归 68 点，再用 image_to_face_mat 反变换回原图。

    模型缺失 / 跑挂时返回 None，由调用方报「68 点算不出来」。以前这里有个
    "2D106 不可用就退回 5 点插值"的分支（合成链路默认走它），已经删掉：那等于
    把插值估算当成检测结果，误差大 4 倍却完全看不出来（见 _align.py 文件头）。
    """
    import numpy as np

    kps = det_box.get("kps") or []
    lm68 = None
    if len(kps) >= 5:
        try:
            from .._align import align_face, transform_points
            aligned, meta = align_face(frame, np.array(kps, dtype=np.float64), box_xyxy)
            if aligned is not None and meta:
                lm = _landmarks_model()
                if lm is not None:
                    lm68a = lm.landmarks68(aligned)
                    if lm68a is not None:
                        lm68 = transform_points(
                            np.asarray(lm68a, dtype=np.float64),
                            np.array(meta["image_to_face_mat"], dtype=np.float64),
                            invert=True)
        except Exception:
            lm68 = None
    if lm68 is None:
        return None
    arr = np.asarray(lm68, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr)):
        return None
    return arr


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 compose 与 app、media_frames、preview、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from ..app import (
    WORKSPACE, _dfljpg_dict, _get_detector, _imread_any, _merge_cache, _merge_frame_image,
    _merge_job, _merge_job_lock, _merge_settings_to_config,
)

from .._media.frames import (
    _grab_frame_at, _grab_key,
)

from .preview import (
    _PREVIEW_MAX_SWAPS, _WORKER, _det_cached, _dfm_of, _face_ids_val, _model_for_face,
    _models_of, _preview_job, _send_frame_bytes, _xseg_of,
)

from .vram import (
    _vram_measure_begin, _vram_measure_end,
)
