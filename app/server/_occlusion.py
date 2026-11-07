# -*- coding: utf-8 -*-
"""遮挡率（occlusion）排序器 —— 纯 numpy + cv2，不加载模型、不起引擎进程。

口径（用户定死的，别改）
------------------------
人脸区域 = 68 点凸包，但**只用眼睛及以下**：
    jaw 0..16 + 鼻 27..35 + 双眼 36..47 + 嘴 48..67
    **排除眉毛 17..26**（眉毛以上是额头/头发，XSeg 本来就不标，
    留着会把额头当成"遮挡"）
遮罩  = DFLJPG 的 APP15['xseg_mask']，解出来是单通道 0..1 浮点
    （等价于 DFLJPG.get_xseg_mask()：cv2.imdecode → /255）
遮挡率 = 1 - sum(mask[hull>0]) / count(hull>0)
    · 软面积：每个像素按自己的 0..1 遮罩值计入，不做二值化（半透明遮挡算半个）
    · 全被遮住 → 100%；一点没挡 → 0%
    · mask_ratio = 1 - occ（遮罩覆盖率，原式那一条，照原样一起返回给前端核对）
    · hull_ratio = 凸包面积 / 图像面积

判无效（不参与统计，前端单独列出来）
    · 没有 landmarks / 不是 DFLJPG       → reason="no_landmarks"
    · 没有 xseg_mask                     → reason="no_mask"
    · 凸包面积 < 0.5% 图像面积           → reason="hull_too_small"
    · 文件读不出来 / 解不开              → reason="unreadable"

为什么不用 _oneshot.occlusion_of
    那个是「体检」里的评分：砍眉线以下 + 内缩 21px + 二值遮罩，刻度完全
    不一样（实测 P90=0.017）。这里要的是**排序器**：不内缩、软面积、保留
    边界那一圈，严格按用户给的算法；两套口径互不影响。

性能（本机实测，8 线程）：
    · 673 张：热 0.47~0.53 s（0.70 ms/张），冷盘首次 0.91 s
    · 3345 张：热 1.85~1.87 s（0.55 ms/张），冷盘首次 2.04 s
  主路径**不解码原图**（只解 256² 的遮罩 + 从 JPEG SOF 读宽高），
  所以比 blur/重复图那条（全量解码缩略图）要快一个量级。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:                                        # pragma: no cover
    cv2 = None

# 凸包用的 68 点下标：去掉眉毛 17..26
HULL_IDX = np.array(list(range(0, 17)) + list(range(27, 36)) +
                    list(range(36, 48)) + list(range(48, 68)), dtype=np.int32)

MIN_HULL_RATIO = 0.005      # 凸包面积 < 图像面积 0.5% → 判无效
_MIN_LM = int(HULL_IDX.max()) + 1
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------- 元数据
def _safe_loads(raw):
    """APP15 段是 pickle：借 app.py 里那份白名单反序列化，别在这另写一套。"""
    from .app import _safe_loads as _f      # 惰性：避免 app.py 还在初始化时循环导入
    return _f(raw)


def dfl_dict(data: bytes):
    """从 jpg 字节里解出 DFLJPG 的 APP15 字典（头部那份，引擎只认它）。

    DFLJPG.load_raw 走到 SOS 就跳 EOI —— 所以 APP15 在 SOS 之后的一律读不到，
    这里同样以 SOS 为界（返回 None）。
    """
    i, n = 0, len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:      # SOI/EOI/RST：无长度
            i += 2
            continue
        if i + 4 > n:
            return None
        seg = (data[i + 2] << 8) | data[i + 3]
        if seg < 2:
            return None
        if m == 0xDA:                                    # SOS：压缩数据开始
            return None
        if m == 0xEF:                                    # APP15
            try:
                d = _safe_loads(data[i + 4:i + 2 + seg])
            except Exception:
                return None
            return d if isinstance(d, dict) else None
        i += 2 + seg
    return None


def jpeg_size(data: bytes):
    """从 SOF 段读 (w, h)：只为算 hull_ratio，不值得把 400KB 的图解码一遍。"""
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return (w, h) if w > 0 and h > 0 else None
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        seg = (data[i + 2] << 8) | data[i + 3]
        if seg < 2:
            return None
        i += 2 + seg
    return None


def _mask_float(raw):
    """APP15['xseg_mask'] → 单通道 float32 (0..1)。三种存法都认。"""
    if raw is None or cv2 is None:
        return None
    m = None
    try:
        if isinstance(raw, np.ndarray):
            blob = np.ascontiguousarray(raw).tobytes()
            m = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
            if m is None:
                m = raw
        else:
            m = cv2.imdecode(np.frombuffer(bytes(raw), np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    if m.ndim != 2 or m.size == 0:
        return None
    if m.dtype == np.uint8:
        f = m.astype(np.float32) / 255.0
    else:
        f = m.astype(np.float32)
        mx = float(f.max()) if f.size else 0.0
        if mx > 1.0001:                                  # 0..255 的整型/浮点遮罩
            f = f / 255.0
    if f.min() < 0.0:
        f = np.clip(f, 0.0, 1.0)
    return f


def _landmarks_px(lm, img_w: int, img_h: int):
    """68 点 → 像素坐标（原图尺度）。归一化坐标（4.1 提取器写的 0..1）自动放大。"""
    try:
        pts = np.asarray(lm, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < _MIN_LM:
            return None
    except Exception:
        return None
    pts = np.ascontiguousarray(pts[:, :2])
    if float(np.abs(pts).max()) <= 1.5:                 # 归一化点集
        pts = pts * np.array([float(img_w), float(img_h)], np.float32)
    return pts


def hull_polygon(pts_px, img_w: int, img_h: int, mask_w: int, mask_h: int):
    """像素点集 → 遮罩尺度的凸包多边形（图与遮罩尺寸不同就等比缩）。"""
    q = pts_px[HULL_IDX].astype(np.float32).copy()
    q[:, 0] *= float(mask_w) / float(img_w)
    q[:, 1] *= float(mask_h) / float(img_h)
    return cv2.convexHull(q.reshape(-1, 1, 2).astype(np.float32)).reshape(-1, 2)


def occlusion_one(data: bytes) -> dict:
    """一张图算完：{occ, hull_ratio, mask_ratio, valid, reason}（不读盘）。"""
    meta = dfl_dict(data)
    if not meta:
        return {"occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "no_landmarks"}
    mask = _mask_float(meta.get("xseg_mask"))
    if mask is None:
        # 有 landmarks 没遮罩 → no_mask；两者都缺 → no_landmarks（先说更根本的那个）
        lm0 = meta.get("landmarks")
        return {"occ": None, "hull_ratio": 0.0, "mask_ratio": None, "valid": False,
                "reason": "no_landmarks" if lm0 is None else "no_mask"}
    mh, mw = int(mask.shape[0]), int(mask.shape[1])
    iw, ih = jpeg_size(data) or (0, 0)
    if iw <= 0 or ih <= 0:
        iw, ih = mw, mh
    pts = _landmarks_px(meta.get("landmarks"), iw, ih)
    if pts is None:
        return {"occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "no_landmarks"}
    try:
        hull = hull_polygon(pts, iw, ih, mw, mh)
    except Exception:
        return {"occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "no_landmarks"}
    if hull.shape[0] < 3:
        return {"occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "hull_too_small"}

    area_ratio = float(abs(cv2.contourArea(hull))) / float(mw * mh)
    if area_ratio < MIN_HULL_RATIO:
        return {"occ": None, "hull_ratio": round(area_ratio, 4), "mask_ratio": None,
                "valid": False, "reason": "hull_too_small"}

    canvas = np.zeros((mh, mw), np.uint8)
    cv2.fillConvexPoly(canvas, np.round(hull).astype(np.int32), 255)
    sel = canvas > 0
    npx = int(sel.sum())
    if npx <= 0:
        return {"occ": None, "hull_ratio": round(area_ratio, 4), "mask_ratio": None,
                "valid": False, "reason": "hull_too_small"}

    cover = float(mask[sel].sum()) / float(npx)
    cover = 0.0 if cover < 0.0 else (1.0 if cover > 1.0 else cover)
    return {"occ": round(1.0 - cover, 4), "hull_ratio": round(area_ratio, 4),
            "mask_ratio": round(cover, 4), "valid": True, "reason": ""}


def _one_file(path: Path) -> dict:
    name = path.name
    try:
        data = path.read_bytes()
    except OSError:
        return {"name": name, "occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "unreadable"}
    try:
        r = occlusion_one(data)
    except Exception:
        return {"name": name, "occ": None, "hull_ratio": 0.0, "mask_ratio": None,
                "valid": False, "reason": "unreadable"}
    r["name"] = name
    return r


def _stats(items):
    """统计量只算 valid：无遮挡=0%，别把 "没数据" 混进分布里。"""
    vals = np.array([it["occ"] for it in items if it["valid"]], dtype=np.float32)
    unranked = len(items) - int(vals.size)
    if vals.size == 0:
        st = {"median": None, "p90": None, "max": None, "min": None,
              "mean": None, "n": 0, "unranked": unranked}
    else:
        st = {"median": round(float(np.median(vals)), 4),
              "p90": round(float(np.percentile(vals, 90)), 4),
              "max": round(float(vals.max()), 4),
              "min": round(float(vals.min()), 4),
              "mean": round(float(vals.mean()), 4),
              "n": int(vals.size), "unranked": unranked}
    reasons = {}
    for it in items:
        if not it["valid"]:
            reasons[it["reason"]] = reasons.get(it["reason"], 0) + 1
    st["reasons"] = reasons
    return st


def scan(root: Path, workers: int = 8) -> dict:
    """扫一层（不递归、跳过 _/点 开头的文件，与 _dataset_fast.list_images 同口径）。"""
    t0 = time.time()
    try:
        files = sorted(p for p in Path(root).iterdir()
                       if p.is_file() and p.suffix.lower() in _IMG_EXTS
                       and not p.name.startswith((".", "_")))
    except OSError:
        files = []
    workers = max(1, min(int(workers or 1), 16))
    if len(files) > 32 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            items = list(ex.map(_one_file, files))
    else:
        items = [_one_file(p) for p in files]
    # 排序：有效的按 occ 降序（同值按名字稳定），无效的排在后面
    items.sort(key=lambda it: (0, -it["occ"], it["name"]) if it["valid"]
               else (1, 0.0, it["name"]))
    return {"items": items, "stats": _stats(items), "total": len(files),
            "ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- 移走 / 撤销
def _within(base: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def move_to_trash(root: Path, names, subdir: str = "_trash") -> dict:
    """只搬不删：同盘 rename，跨盘 shutil.move，**永不 unlink 源文件**。

    与 /api/dataset/apply 的唯一实现（那边现在也调这里）：重名自动加 _1/_2，
    目标必须在 root 里面，非法的名字直接跳过。
    """
    root = Path(root)
    sub = str(subdir or "_trash")
    if not sub or "/" in sub or "\\" in sub or ":" in sub or sub in (".", ".."):
        sub = "_trash"
    trash = root / sub
    trash.mkdir(parents=True, exist_ok=True)
    items, failed = [], []
    import os as _os
    import shutil as _sh
    for n in names or []:
        n = str(n or "")
        if not n or "/" in n or "\\" in n or ":" in n or n in (".", ".."):
            continue
        src = root / n
        if not src.exists() or not _within(root, src):
            continue
        dst = trash / n
        if not _within(trash, dst):
            continue
        k = 1
        while dst.exists():
            dst = trash / (src.stem + "_%d" % k + src.suffix)
            k += 1
        ok = False
        try:
            _os.replace(str(src), str(dst))
            ok = True
        except OSError:
            try:
                _sh.move(str(src), str(dst))
                ok = True
            except Exception:
                ok = False
        if not ok:
            failed.append(n)
            continue
        items.append({"name": n, "trash": dst.name})
    return {"moved": len(items), "items": items, "failed": failed,
            "trash": str(trash), "subdir": sub}


def write_manifest(trash: str | Path, payload: dict) -> str:
    """撤销清单落到 _trash 里：人肉排查 / 事后 restore 都能照着它来。

    注意 list_images 会跳过 _ 开头的文件，这份 json 不会污染数据集计数。
    """
    d = Path(trash)
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("_occlusion_move_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    k = 1
    while p.exists():
        p = d / ("_occlusion_move_%s_%d.json" % (time.strftime("%Y%m%d-%H%M%S"), k))
        k += 1
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(p)
