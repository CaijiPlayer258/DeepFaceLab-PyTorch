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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ._align import read_dfljpg_meta

_APP = Path(__file__).resolve().parent.parent
# 权重平铺：errface_xgb.ubj 直接在权重根下（weights/errface_xgb.ubj）
_CAND_MODEL_DIRS = [
    Path(os.environ.get("DFL_WEIGHTS_DIR") or (_APP.parent / "weights")),
    _APP.parent / "weights",
    _APP / "engine" / "modelhub",                 # 迁移前的位置（老布局）
]
_MODEL_DIR = next((d for d in _CAND_MODEL_DIRS if d.is_dir()), _CAND_MODEL_DIRS[0])
# 统一命名：errface_xgb.ubj（旧名 CL.ubj / CL.model、老布局 weights/errface/ 只作兼容回退）
_MODEL = _MODEL_DIR / "errface_xgb.ubj"
_LEGACY = ([d / "errface" / "errface_xgb.ubj" for d in _CAND_MODEL_DIRS]
           + [_MODEL_DIR / "CL.ubj", _MODEL_DIR / "CL.model"]
           + [d / "errface" / "CL.ubj" for d in _CAND_MODEL_DIRS])
_BOOSTER = None
_BOOSTER_KEY = None    # 加载成功/失败时那个文件的指纹（换了文件要重新加载）
_BOOSTER_ERR = ""      # 上次加载失败的原因（同一个文件不重复付代价）
_XGB_MOD = None
_XGB_WHY = ""
_LM106 = False        # False=未尝试, None=不可用, 否则为模型对象

# 「能力」状态：文件在不在 / 读不读得了 / 运行库在不在 / 模型加载得起来吗。
# 前端靠它置灰按钮，而不是等用户点了再抛 409 —— 缺模型是状态，不是故障。
_MB = 2 ** 20
_ST = {"key": None, "at": 0.0, "val": None}
_ST_TTL = 2.0
_INSTALL_ZH = "把 errface_xgb.ubj 放进 weights/ 根目录（平铺），或在「设置 → 权重包」里下载 core 包"
_INSTALL_EN = ("Drop errface_xgb.ubj into the weights folder root (flat layout), "
               "or download the core pack in Settings → Weight packs")


def model_path() -> Path:
    return _MODEL


def _resolved() -> Path | None:
    if _MODEL.is_file():
        return _MODEL
    for p in _LEGACY:
        if p.is_file():
            return p
    return None


def _file_key(p) -> tuple:
    """文件指纹：路径 + mtime + 大小。换了文件才重新加载（用户拷模型进来不用重启后端）。"""
    if p is None:
        return ("", 0, 0)
    try:
        st = p.stat()
        return (str(p), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return ("", 0, 0)


def _xgb():
    """(xgboost 模块, 失败说明)。import 只试一次 —— 缺依赖时别每次点击都付一遍代价。"""
    global _XGB_MOD, _XGB_WHY
    if _XGB_MOD is not None or _XGB_WHY:
        return _XGB_MOD, _XGB_WHY
    try:
        import xgboost as xgb
        _XGB_MOD = xgb
    except Exception as ex:
        _XGB_WHY = "缺 xgboost 运行库（%s）" % (str(ex) or type(ex).__name__)
    return _XGB_MOD, _XGB_WHY


def _readable(p) -> tuple:
    """存在之外还要「读得动」：0 字节 / 打不开 都算不可用。→ (ok, 中文原因, 英文原因)"""
    try:
        st = p.stat()
    except OSError as ex:
        e = getattr(ex, "strerror", None) or ex
        return False, "打不开（%s）" % e, "cannot be opened (%s)" % e
    if st.st_size <= 0:
        return False, "是 0 字节", "is 0 bytes"
    try:
        with open(p, "rb") as f:
            if not f.read(1):
                return False, "读不到内容", "cannot be read"
    except OSError as ex:
        e = getattr(ex, "strerror", None) or ex
        return False, "读不了（%s）" % e, "cannot be read (%s)" % e
    return True, "", ""


def _manifest_entry() -> dict:
    """清单里 errface_xgb.ubj 那一条（size / group 用来告诉用户还差多少、下载哪一组）。"""
    cands = []
    try:
        from ._paths import weights_dir
        cands.append(Path(weights_dir()) / "manifest.json")
    except Exception:
        pass
    cands.append(_APP.parent / "weights" / "manifest.json")
    seen = set()
    for c in cands:
        k = str(c)
        if k in seen:
            continue
        seen.add(k)
        try:
            if not c.is_file():
                continue
            data = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in (data.get("files") or []):
            if isinstance(e, dict) and str(e.get("path") or "").replace("\\", "/") in ("errface_xgb.ubj", "errface/errface_xgb.ubj"):
                return e
    return {}


def _booster():
    global _BOOSTER, _BOOSTER_KEY, _BOOSTER_ERR
    path = _resolved()
    key = _file_key(path)
    if _BOOSTER is not None and _BOOSTER_KEY == key:
        return _BOOSTER
    if _BOOSTER_KEY == key and _BOOSTER_ERR:
        return None                     # 同一个文件失败过：结论复用，不反复解析
    if path is None:
        _BOOSTER, _BOOSTER_KEY, _BOOSTER_ERR = None, key, "缺 %s" % _MODEL.name
        return None
    xgb, why = _xgb()
    if xgb is None:
        _BOOSTER, _BOOSTER_KEY, _BOOSTER_ERR = None, key, why
        return None
    try:
        b = xgb.Booster()
        b.load_model(str(path))
    except Exception as ex:
        _BOOSTER, _BOOSTER_KEY, _BOOSTER_ERR = None, key, ("加载失败：%s"
                                                           % (str(ex)[:180] or type(ex).__name__))
        return None
    _BOOSTER, _BOOSTER_KEY, _BOOSTER_ERR = b, key, ""
    return b


def status(force: bool = False) -> dict:
    """能力状态：一次把「能不能用 / 为什么不能用 / 怎么装好」说清。

    可用的判定（存在且可读才算）：
      1) errface_xgb.ubj 在（含旧名 CL.ubj / CL.model 的回退）
      2) 读得动（能打开、非 0 字节）
      3) xgboost 装得上
      4) Booster 真的加载得起来
    任何一条不满足 → available=False，并带回给人看的原因（中英各一份，前端直接贴）。
    """
    resolved = _resolved()
    key = (_file_key(resolved), _file_key(_MODEL))
    now = time.time()
    if not force and _ST["val"] is not None and _ST["key"] == key and (now - _ST["at"]) < _ST_TTL:
        return dict(_ST["val"])

    ent = _manifest_entry()
    want = int(ent.get("size") or 0)
    group = str(ent.get("group") or "core")
    mb = round(want / _MB, 1) if want else 0.0
    zh_size = "（约 %.1f MB）" % mb if want else ""
    en_size = " (about %.1f MB)" % mb if want else ""
    out = {
        "capability": "errface",
        "available": False,
        "model": _MODEL.name,
        "path": str(_MODEL),
        "file": str(resolved) if resolved else "",
        "size_mb": mb,
        "group": group,
        "tried": [str(d) for d in _CAND_MODEL_DIRS],
        "deps": {"xgboost": ""},
        "reason": "",
        "reason_en": "",
        "install": "",
        "install_en": "",
        "checked_at": now,
    }

    def done():
        _ST.update(key=key, at=now, val=dict(out))
        return dict(out)

    if resolved is None:
        out["reason"] = "缺 %s%s —— 在「设置 → 权重包」里下载 %s 包" % (_MODEL.name, zh_size, group)
        out["reason_en"] = ("%s is missing%s — download the %s pack in Settings → Weight packs"
                            % (_MODEL.name, en_size, group))
        out["install"], out["install_en"] = _INSTALL_ZH, _INSTALL_EN
        return done()

    ok, why_zh, why_en = _readable(resolved)
    if not ok:
        out["reason"] = "%s 在 %s，但%s —— 重新拷一份进去" % (_MODEL.name, resolved, why_zh)
        out["reason_en"] = "%s exists at %s but %s — copy it in again" % (_MODEL.name, resolved, why_en)
        out["install"], out["install_en"] = _INSTALL_ZH, _INSTALL_EN
        return done()

    xgb, why = _xgb()
    out["deps"]["xgboost"] = str(getattr(xgb, "__version__", "") or "") if xgb is not None else ""
    if xgb is None:
        out["reason"] = "%s 已在，但%s" % (_MODEL.name, why)
        out["reason_en"] = "%s is present, but the xgboost runtime is not installed" % _MODEL.name
        out["install"] = ("在引擎环境里装 xgboost：runtime\\engine\\python.exe -m pip install xgboost")
        out["install_en"] = "Install xgboost into the engine env: runtime\\engine\\python.exe -m pip install xgboost"
        return done()

    if _booster() is None:
        out["reason"] = "%s 加载失败：%s" % (_MODEL.name, _BOOSTER_ERR or "未知原因")
        out["reason_en"] = "%s failed to load: %s" % (_MODEL.name, "unknown error" if _BOOSTER_ERR else "")
        out["install"], out["install_en"] = _INSTALL_ZH, _INSTALL_EN
        return done()

    out.update(available=True,
               reason="", reason_en="", install="", install_en="")
    return done()


def available() -> bool:
    """有模型 + 读得了 + xgboost 在 + 加载得起来，才算「能用」。"""
    return bool(status().get("available"))


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
        # 平铺布局：<weights>/insightface_2d106.onnx；老的分目录布局作兜底
        for base in _CAND_MODEL_DIRS:
            for rel in ("insightface_2d106.onnx",
                        "onnx/InsightFace2d106/InsightFace2D106.onnx"):
                cand = Path(base) / rel
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
        # 缺文件 / 缺 xgboost / 加载失败 是三件事，别把它们混成一句「缺 ubj」
        st = status(force=True)
        return {"ok": False, "unavailable": True, "capability": "errface",
                "reason": st.get("reason") or ("%s 不可用" % _MODEL.name),
                "install": st.get("install") or "",
                "status": st,
                "error": st.get("reason") or ("模型不可用：%s" % _MODEL.name)}

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
