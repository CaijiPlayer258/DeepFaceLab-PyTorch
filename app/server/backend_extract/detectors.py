# -*- coding: utf-8 -*-
"""检测器 / 106 点 / XSeg：名字归一、权重路径、按需加载与缓存（从 app/server/app.py 切出）。

4.1 只留两个检测器：tinymog（快，默认）/ retinaface_10g（重）。名字写错一律 400，
不再「就近找个能用的顶上」—— 这段口径原样搬过来，一个字没改。

只有包内相对 import 的深度变了（from ._detect 换成 from .._detect 之类）。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from .._common.paths import ROOT, WORKSPACE


_det_cache = {}


# 界面下拉里写的是模型名（RetinaFace_10g / TinyMog …），接口里用的是文件名。
# 两张表都认，免得前端还得记住磁盘上的命名。
#
# 4.1 只保留两个检测器，**规范名 = 缓存键 = worker 里认的名字**：
#   tinymog        —— 快（SCRFD+MobileNetV1，2.7 MB），默认
#   retinaface_10g —— 重（召回更好）
# retinaface_500m 这个模型**不存在**（weights/ 里那份是 4.0 的遗留文件，从来没被
# 接进 4.1 的下拉）。以前它靠 alias 落成 det_500m → 有人传它也能"成功"，属于
# 静默回退；现在从别名表里删掉，并且显式 400（见 _removed_detector()）。
_DET_ALIAS = {
    "tinymog": "tinymog",
    "retinaface_10g": "retinaface_10g", "retinaface10g": "retinaface_10g",
    "det_10g": "retinaface_10g", "retinaface": "retinaface_10g",
}
# 规范名 → weights/ 根下的平铺文件名
_DET_ONNX = {"tinymog": "tinymog.onnx", "retinaface_10g": "retinaface_10g.onnx"}
# 已移除 / 从不存在——点到名要明确报错，不能"就近找个能用的顶上"
_REMOVED_DETECTORS = {"retinaface_500m", "retinaface500m", "det_500m",
                      "retinaface_500m.onnx", "det_500m.onnx"}
DEFAULT_DETECTOR = "tinymog"


def detector_key(name) -> str:
    """检测器入参 → 规范名（tinymog / retinaface_10g）。

    未知名字抛 400（不是 404、也不是静默回退）：前端一旦把名字写错，要能立刻看见。
    retinaface_500m / det_500m 单独给一句人话——它是"以前能用、现在没有了"的那一个。
    """
    key = str(name or "").strip()
    low = key.lower().replace("-", "_")
    if low.endswith(".onnx"):
        low = low[:-len(".onnx")]
    if not low:
        return DEFAULT_DETECTOR
    if low in _REMOVED_DETECTORS:
        raise HTTPException(
            400, "检测器 %s 不存在（4.1 已移除 RetinaFace_500m），可选：%s"
            % (key, " / ".join(sorted(_DET_ONNX))))
    canon = _DET_ALIAS.get(low)
    if canon is None:
        raise HTTPException(
            400, "未知检测器 %r，只认：%s" % (key, " / ".join(sorted(_DET_ONNX))))
    return canon


def _get_detector(name=None):
    """按需加载检测器（权重平铺在 weights/ 根下，只读引用）。

    默认 TinyMog。认界面名（TinyMog / RetinaFace_10g）、文件名（det_10g）和规范名；
    retinaface_500m / det_500m 以及任何没见过的名字都 400，不再"退一个能用的"。
    """
    canon = detector_key(name)
    if canon in _det_cache:
        return _det_cache[canon]
    flat = _DET_ONNX[canon]
    p = _detector_path(flat)
    if p is None:
        raise HTTPException(404, "找不到检测模型 %s（weights/%s）" % (canon, flat))
    if canon == "tinymog":
        from .._detect import TinyMogDet
        det = TinyMogDet(str(p), workspace=WORKSPACE, engine_name="tinymog")
    else:
        from .._detect import RetinaFaceDet
        det = RetinaFaceDet(p, 640)
    _det_cache[canon] = det
    return det

def _five_from_box(box):
    """没有关键点时的兜底 5 点：按框比例摆出双眼/鼻尖/嘴角。

    RetinaFace 正常情况下会给出 5 点，这条只在模型没有 kps 输出时兜底，
    对齐质量差一些，但产出的仍然是合法 DFLJPG。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return [
        [x1 + 0.32 * w, y1 + 0.38 * h],   # 左眼
        [x1 + 0.68 * w, y1 + 0.38 * h],   # 右眼
        [x1 + 0.50 * w, y1 + 0.62 * h],   # 鼻尖
        [x1 + 0.35 * w, y1 + 0.80 * h],   # 左嘴角
        [x1 + 0.65 * w, y1 + 0.80 * h],   # 右嘴角
    ]

_lm106_cache = {"m": None, "tried": False}


def _get_landmarks106():
    """加载 insightface 2D106（模型在引擎仓的 modelhub 里）。找不到就返回 None。

    **强制启用**：写入 DFL 图片的 68 点必须是 2D106 的真实检测点。
    5 点只用于求对齐矩阵；写进图片的 68 点必须来自 2D106 的检测结果。
    并让 4.0 的 sort_by_yaw / 错脸检测都能在 4.1 数据上工作。
    """
    import os as _os
    if _lm106_cache["tried"]:
        return _lm106_cache["m"]
    _lm106_cache["tried"] = True
    try:
        from .._landmarks106 import InsightFace106
        cand = _detector_path("insightface_2d106.onnx")
        if cand is not None:
            _lm106_cache["m"] = InsightFace106(str(cand), workspace=WORKSPACE,
                                               engine_name="insightface_2d106")
            return _lm106_cache["m"]
    except Exception:
        pass
    _lm106_cache["m"] = None
    return None

_xseg_cache = {"m": None, "tried": False}


def _get_xseg(model: str | None = None):
    """加载 XSeg 遮罩模型。model: 'lite' / 'xseg' / None(自动，Lite 优先)。

    按 model 分别缓存——切换遮罩类型不会重复加载同一个模型。
    """
    key = (model or "auto").lower()
    cache = _xseg_cache.setdefault("by_model", {})
    if key in cache:
        return cache[key]
    try:
        from .._xseg import XSegMasker
        # 模型可能在两个地方：本仓库 workspace，或者 4.0 引擎仓库的 workspace
        # （4.1 只是壳，模型历来放在引擎那边）
        roots = [ROOT / "weights" / "xseg", WORKSPACE]
        try:
            from .._paths import engine_root
            _er = engine_root()
            if _er and Path(_er) / "workspace" not in roots:
                roots.append(Path(_er) / "workspace")
        except Exception:
            pass
        paths = {}
        for r in roots:
            paths.setdefault("lite", r / "model" / "XSegLite" / "xseglite.onnx")
            paths.setdefault("xseg", r / "model" / "XSeg" / "XSeg.onnx")
        # 按 roots 顺序挑第一个真实存在的
        for k in ("lite", "xseg"):
            for r in roots:
                cand = r / "model" / ("XSegLite/xseglite.onnx" if k == "lite" else "XSeg/XSeg.onnx")
                if cand.is_file():
                    paths[k] = cand
                    break
        order = (["lite", "xseg"] if key == "auto" else [key]) +                 [k for k in ("lite", "xseg") if k != key and key != "auto"]
        for k in order:
            c = paths.get(k)
            if c and c.is_file():
                cache[key] = XSegMasker(str(c), workspace=WORKSPACE,
                                        engine_name=("xseglite" if k == "lite" else "xseg"))
                return cache[key]
    except Exception:
        pass
    _xseg_cache["m"] = None
    return None

_detector_cache = {"m": None}
_fast_detector_cache = {"m": None, "tried": False}
_heavy_detector_cache = {"m": None, "tried": False}

# 平铺名 → 老布局（weights/onnx/<模型>/<原文件名>）。只是兜底，不是默认路径。
_LEGACY_ONNX = {
    "tinymog.onnx": "onnx/TinyMog/TinyMog.onnx",
    "retinaface_10g.onnx": "onnx/RetinaFace/det_10g.onnx",
    # retinaface_500m.onnx 已移除：4.1 不提供这个检测器（见 _REMOVED_DETECTORS）
    "insightface_2d106.onnx": "onnx/InsightFace2d106/InsightFace2D106.onnx",
}


def _detector_path(flat):
    """按**平铺名**找 onnx：weights/ 根优先（$DFL_WEIGHTS_DIR 可改），老布局兜底。"""
    from .._paths import weights_dir
    bases = [weights_dir(), ROOT / "weights", WORKSPACE / "modelhub"]
    for base in bases:
        for rel in (flat, _LEGACY_ONNX.get(flat, "")):
            if not rel:
                continue
            p = base / rel
            if p.is_file():
                return p
    return None


def _get_fast_detector():
    """快检测器：TinyMog（2.7 MB，SCRFD+MobileNetV1，batch 维是动态的）。"""
    if _fast_detector_cache.get("error") is not None:
        # 上次构造就失败了：每次抽取都如实报同一个异常，别退化成"模型没找到"
        raise _fast_detector_cache["error"]
    if _fast_detector_cache["tried"]:
        return _fast_detector_cache["m"]
    _fast_detector_cache["tried"] = True
    try:
        from .._detect import TinyMogDet
        p = _detector_path("tinymog.onnx")
        if p is not None:
            _fast_detector_cache["m"] = TinyMogDet(str(p), workspace=WORKSPACE, engine_name="tinymog",
                                                graph_max_batch=8)   # 抽取走 batch，给它建图（启动预热期捕获，运行期只 replay）
    except Exception as e:                      # noqa: BLE001
        # 以前这里是 pass：签名不匹配（TypeError）被吞成 None，界面拿到「完成 0 帧」。
        _fast_detector_cache["m"] = None
        _fast_detector_cache["error"] = e
        print("[extract] TinyMogDet 构造失败：%s: %s" % (type(e).__name__, e), flush=True)
        raise
    return _fast_detector_cache["m"]


def _get_heavy_detector():
    """重检测器：只认 retinaface_10g（召回更好）。det_500m 已移除，不再回退。"""
    if _heavy_detector_cache.get("error") is not None:
        raise _heavy_detector_cache["error"]
    if _heavy_detector_cache["tried"]:
        return _heavy_detector_cache["m"]
    _heavy_detector_cache["tried"] = True
    try:
        from .._detect import RetinaFaceDet
        for nm in ("retinaface_10g.onnx",):
            p = _detector_path(nm)
            if p is not None:
                _heavy_detector_cache["m"] = RetinaFaceDet(
                    str(p), workspace=WORKSPACE,
                    engine_name=nm[:-len(".onnx")])
                break
    except Exception as e:                      # noqa: BLE001
        _heavy_detector_cache["m"] = None
        _heavy_detector_cache["error"] = e
        print("[extract] RetinaFaceDet 构造失败：%s: %s" % (type(e).__name__, e), flush=True)
        raise
    return _heavy_detector_cache["m"]
