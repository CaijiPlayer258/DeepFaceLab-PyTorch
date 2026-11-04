# -*- coding: utf-8 -*-
"""错脸检测（ErrFaceFilter 的 xgboost 模型）。

原理：模型吃的是 **68 点 × 2 = 136 维**，判断「这张脸的landmark 几何是否像正常脸」。
要点（本地实测得出，别改）：

1) 特征必须映射到 DFL 的 512 坐标系：landmarks × (512 / 图片高度)；不缩放会误判 70%+
2) 必须用**真实检测到的 68 点**（我们这边是 insightface 2D106 → 68 映射，
   提取时设 DFL_LANDMARKS106=1）。如果用的是「5 点推出来的模板 68 点」，
   几何方差≈0，模型永远给 0 分，等于没有判别力。
3) 所以扫描前先做一次「是否为模板点」的判定，模板点直接报不可用，不产生假阳性。
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

from ._align import read_dfljpg_meta

_APP = Path(__file__).resolve().parent.parent
_CAND_MODEL_DIRS = [
    Path(os.environ.get("DFL_WEIGHTS_DIR") or (_APP.parent / "weights")) / "errface",
    _APP / "engine" / "modelhub" / "errface",     # 迁移前的位置
    _APP / "modelhub" / "errface",                # 更早的位置
]
_MODEL_DIR = next((d for d in _CAND_MODEL_DIRS if d.is_dir()), _CAND_MODEL_DIRS[0])
# 统一命名：errface_xgb.ubj（旧名 CL.ubj / CL.model 只作兼容回退）
_MODEL = _MODEL_DIR / "errface_xgb.ubj"
_LEGACY = [_MODEL_DIR / "CL.ubj", _MODEL_DIR / "CL.model"]
_BOOSTER = None
_LOAD_TRIED = False
_LM106 = False        # False=未尝试, None=不可用, 否则为模型对象


def model_path() -> Path:
    return _MODEL


def _resolved() -> Path | None:
    if _MODEL.is_file():
        return _MODEL
    for p in _LEGACY:
        if p.is_file():
            return p
    return None


def available() -> bool:
    return _resolved() is not None


def _booster():
    global _BOOSTER, _LOAD_TRIED
    if _BOOSTER is not None or _LOAD_TRIED:
        return _BOOSTER
    _LOAD_TRIED = True
    path = _resolved()
    if path is None:
        return None
    try:
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(path))
        _BOOSTER = b
    except Exception:
        _BOOSTER = None
    return _BOOSTER


def _img_h(path: Path) -> int:
    """只读 JPEG 头部拿高度，不解码整张图。"""
    with open(path, "rb") as f:
        data = f.read(4096)
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = (data[i + 5] << 8) | data[i + 6]
            return int(h)
        i += 2 + seg_len
    return 0


def _features(path: Path, why: list | None = None):
    """landmarks → 136 维特征；不合格的返回 None。

    注意：这里完全信任图片里内嵌的 metadata。如果图片被外部改过（缩放、裁剪、
    换成别的图而没更新 landmarks），算出来的几何就是错的，模型会给出错误的判断。
    所以下面加了几道体检，不合格的宁可跳过也不猜。
    """
    def fail(reason):
        if why is not None:
            why.append(reason)
        return None

    meta = read_dfljpg_meta(path)
    if not isinstance(meta, dict):
        return fail("没有 DFL 元数据")
    lm = meta.get("landmarks")
    src = meta.get("source_landmarks")
    mat = meta.get("image_to_face_mat")
    if lm is None and src is not None and mat is not None:
        # 有些数据集省了 landmarks 字段，但留着原始点和变换矩阵 —— 能算回来
        try:
            T = np.asarray(mat, dtype=np.float64)[:2, :3]
            P = np.asarray(src, dtype=np.float64)
            lm = (P @ T[:, :2].T + T[:, 2]).tolist()
        except Exception:
            lm = None
        if lm is not None and why is not None:
            why.append("(由 source_landmarks 复原)")
    if lm is None:
        return fail("元数据里没有 landmarks")
    # 自洽校验：landmarks 必须等于 image_to_face_mat × source_landmarks
    # （DFL 原版就是这么算出来的，实测 1204 张误差 0.000px）
    if src is not None and mat is not None and not (why and why[0].startswith("(")):
        try:
            T = np.asarray(mat, dtype=np.float64)[:2, :3]
            P = np.asarray(src, dtype=np.float64)
            pred = P @ T[:, :2].T + T[:, 2]
            dev = float(np.abs(pred - np.asarray(lm, dtype=np.float64)).max())
            if dev > 2.0:
                return fail("元数据自相矛盾（landmarks 与变换矩阵差 %.1fpx）——文件被改过" % dev)
        except Exception:
            pass
    h = _img_h(path)
    if h <= 0:
        return fail("读不出图片高度")
    a = np.asarray(lm, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        return fail("landmarks 形状异常")
    if a.shape[0] != 68:
        return fail("landmark 不是 68 点（%d 点）" % a.shape[0])
    if not np.isfinite(a).all():
        return fail("landmarks 含非法数值")
    a = a * (512.0 / h)          # 映射到 DFL 的 512 坐标系
    # 体检：正常对齐脸的关键点应当落在画面内，越界说明 scale 对不上（图被缩放过）
    if a.min() < -40 or a.max() > 552:
        span = float(a.max() - a.min())
        return fail("关键点越界（跨度 %.0f，图高 %d）——图片可能被缩放过但元数据没更新" % (span, h))
    span = a.max(axis=0) - a.min(axis=0)
    if span[0] < 40 or span[1] < 40:
        return fail("关键点几乎重合，几何退化")
    if not (0.5 <= span[1] / max(1e-6, span[0]) <= 2.0):
        return fail("关键点长宽比异常")
    # 体检二：DFL 对齐脸的关键点重心一定在画面中心附近（实测 1204 张：p99 = 38px）
    c = a.mean(axis=0)
    off = float(np.hypot(c[0] - 256.0, c[1] - 256.0))
    if off > 60:
        return fail("关键点偏心 %.0fpx —— 元数据与图片对不上（位移会让模型直接判成错脸）" % off)
    vspan = float(a[:, 1].max() - a[:, 1].min())
    if not (170 <= vspan <= 400):
        return fail("脸高异常 %.0fpx —— 缩放档位和元数据不一致" % vspan)
    return a.reshape(-1)


def shape_variance(paths, limit=60) -> float:
    """抽若干张，算「人眼间距/脸高」的相对标准差。

    模板 68 点 ≈ 0.002；真实检测点 ≈ 0.02~0.3。用来判断数据集能不能用这个模型。
    """
    vals = []
    for p in paths[:limit]:
        f = _features(p)
        if f is None:
            continue
        pts = f.reshape(-1, 2)
        eye = np.linalg.norm(pts[36] - pts[45])
        vert = np.linalg.norm(pts[8] - pts[27])
        if vert > 1e-6:
            vals.append(eye / vert)
    if len(vals) < 5:
        return -1.0
    return float(np.std(vals) / max(1e-6, abs(np.mean(vals))))


def _landmark_model():
    """2D106 模型：找不到就返回 None（数据集缺元数据时用来现场补点）。"""
    global _LM106
    if _LM106 is not False:
        return _LM106 or None
    _LM106 = None
    try:
        from ._landmarks106 import InsightFace106
        from ._ortcuda import _candidate_roots
        for root in _candidate_roots():
            for rel in ("modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx",
                        "DeepFaceLab-PyTorch/modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(root) / rel
                if cand.is_file():
                    _LM106 = InsightFace106(str(cand), engine_name="insightface_2d106")
                    return _LM106
    except Exception:
        _LM106 = None
    return None


def _estimate68(path: Path):
    """读图现场跑 2D106 → 68 点（512 空间）。模型不可用返回 None。"""
    m = _landmark_model()
    if m is None:
        return None
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        out = m.landmarks68_batch([img])
        a = np.asarray(out[0], dtype=np.float64)
        # landmarks68_batch 的输出是「输入图像像素坐标」（它假设输入是 512 图），
        # 这里统一映射到 DFL 的 512 坐标系，和 _features() 的结果同源可比
        return a * (512.0 / float(img.shape[0]))
    except Exception:
        return None


def scan(directory, threshold=0.5, limit=0, percent=0.0, progress=None, estimate=True):
    """扫描一个对齐脸目录，返回被判为「错脸」的文件。"""
    booster = _booster()
    if booster is None:
        return {"ok": False, "error": "模型不可用：缺 %s（需要 xgboost）" % _MODEL.name}

    files = sorted(glob.glob(os.path.join(str(directory), "*.jpg")))
    if limit:
        files = files[: int(limit)]
    if not files:
        return {"ok": False, "error": "目录里没有 jpg"}

    var = shape_variance(files)
    synthetic = var >= 0 and var < 0.01
    if synthetic:
        return {
            "ok": False,
            "synthetic": True,
            "variance": round(var, 5),
            "n": len(files),
            "error": "这个数据集的 68 点是模板生成的（形状方差 %.4f），几何恒定，模型无法判别。"
                     "重新提取时设 DFL_LANDMARKS106=1，用 2D106 模型产出真实关键点。" % var,
        }

    feats, keep, skipped = [], [], []
    estimated = 0
    for p in files:
        why = []
        f = _features(Path(p), why)
        if f is None:
            # 没有元数据 / 元数据坏了：现场跑 2D106 补 68 点（只对对齐脸有意义）
            est = None
            bad = bool(why) and why[0].startswith("元数据自相矛盾")
            if estimate and not bad and len(feats) < 4000:
                est = _estimate68(Path(p))
            if est is not None:
                sp = est.max(axis=0) - est.min(axis=0)
                c = est.mean(axis=0)
                if np.isfinite(est).all() and sp.min() >= 40 and np.hypot(c[0] - 256, c[1] - 256) <= 60:
                    feats.append(est.reshape(-1))
                    keep.append(p)
                    estimated += 1
                    continue
                why = ["现场补点的几何不合理（这张可能不是对齐脸）"]
            skipped.append({"file": os.path.basename(p), "why": why[0] if why else "未知"})
            continue
        feats.append(f)
        keep.append(p)
    if not feats:
        return {"ok": False, "skipped": skipped[:20], "skipped_n": len(skipped),
                "estimated_n": estimated,
                "error": "没有可用的 landmark 数据：%d 张里没有一张能拿到有效的 68 点" % len(files)}

    import xgboost as xgb
    X = np.stack(feats).astype(np.int64)
    probs = booster.predict(xgb.DMatrix(X))

    # percent>0：不看绝对阈值，直接取最差的这一批（和「模糊」过滤器的分位语义一致）
    if percent and float(percent) > 0:
        cut = float(np.percentile(probs, 100.0 - min(50.0, float(percent))))
        thr = max(float(threshold), cut) if False else cut
    else:
        thr = float(threshold)

    hits = []
    for p, pr in zip(keep, probs.tolist()):
        if pr >= thr:
            hits.append({"file": os.path.basename(p), "prob": round(float(pr), 4)})
    hits.sort(key=lambda x: -x["prob"])

    lo = int((probs < 0.2).sum()); mid = int(((probs >= 0.2) & (probs < 0.5)).sum())
    hi = int((probs >= 0.5).sum())
    return {
        "ok": True,
        "n": len(keep),
        "variance": round(var, 5),
        "threshold": round(float(thr), 4),
        "hits": hits,
        "counts": {"good": lo, "suspect": mid, "bad": hi},
        "skipped": skipped[:20],
        "skipped_n": len(skipped),
        "estimated_n": estimated,
        "stats": {
            "p50": round(float(np.median(probs)), 4),
            "p90": round(float(np.percentile(probs, 90)), 4),
            "max": round(float(probs.max()), 4),
        },
    }
