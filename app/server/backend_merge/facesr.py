# -*- coding: utf-8 -*-
"""人脸超分（GPEN-BFR-512）—— 合成器的可选后处理。

放在**帧空间**做，不在芯片上做
--------------------------------
芯片是 224~416，GPEN 要 512：先放大到 512、再缩回去，等于「先糊一遍再加锐」。
所以这里拿的是**合成之后那帧里的脸区域**（原生分辨率），只重采样一次：
裁 → 512 → 推理 → 缩回 → 按遮罩 alpha 混回。

混回强度是**默认 0.6**，不是 1.0
------------------------------
GPEN 是盲修复网络，它会把脸往「训练分布里的清晰人脸」拉。换脸的脸本身是干净的，
只有"缺高频"，100% 替换会把目标人的特征洗掉 —— 实测同素材 100% 时 ArcFace 相似度掉得明显。
所以给档位：off / low(0.35) / mid(0.6) / high(1.0)，默认 off（不改任何现有产出）。

后端优先级：TRT 引擎 → onnxruntime(CUDA) → 都不行就**明确告警并跳过**（不静默）
----------------------------------------------------------------------------
  DFL_SR=off|low|mid|high        档位（默认 off）
  DFL_SR_BACKEND=auto|trt|onnx   后端（auto = 有 engine 用 engine）
  DFL_SR_MODEL=<onnx 路径>        默认 weights/gpen_bfr_512.onnx
  engine 放 weights/_engine/gpen_bfr_512.<sm>.<prec>.engine（用 app/tools/models/build_sr_engine.py 编）

为什么不做 torch 后端：GPEN 官方权重是 ONNX 更省事（torch 版要 vendor 它的网络定义 + StyleGAN2 的
自定义 op），而 bf16/CUDA Graph 那套优化 TRT 引擎天然就有（--bf16）。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_PREF = {"off": 0.0, "low": 0.35, "mid": 0.6, "high": 1.0}
_LOCK = threading.Lock()
_SESS = {"obj": None, "why": "", "backend": "", "name": ""}
_SIZE_DEFAULT = 256          # 256 版单次约 1/4 成本；512 版质量更好但本机吃不下


def _log(msg: str) -> None:
    print("[sr] " + msg, flush=True)


def mode() -> str:
    v = str(os.environ.get("DFL_SR") or "off").strip().lower()
    return v if v in _PREF else "off"


def strength() -> float:
    return float(_PREF.get(mode(), 0.0))


def enabled() -> bool:
    return strength() > 0.0


def _weights_dir() -> Path:
    from .._paths import weights_dir
    return Path(weights_dir())


def size() -> int:
    """工作分辨率：256（默认，快）或 512（质量档）。DFL_SR_SIZE 覆盖。"""
    try:
        v = int(str(os.environ.get("DFL_SR_SIZE") or _SIZE_DEFAULT).strip())
    except (TypeError, ValueError):
        v = _SIZE_DEFAULT
    return 512 if v >= 384 else 256


def _model_path() -> Path:
    p = str(os.environ.get("DFL_SR_MODEL") or "").strip()
    if p:
        return Path(p)
    return _weights_dir() / ("gpen_bfr_%d.onnx" % size())


def _engine_path() -> Path | None:
    """weights/_engine/ 下与本机 SM 匹配的 GPEN 引擎（没有就 None）。"""
    try:
        from ..backend_models.engines import sm_of
    except Exception:                                       # noqa: BLE001
        sm_of = None
    roots = [_weights_dir() / "_engine", _weights_dir() / "engine"]
    try:
        sm = int(sm_of()) if sm_of else 0
    except Exception:                                       # noqa: BLE001
        sm = 0
    for root in roots:
        if not root.is_dir():
            continue
        for suf in (".bf16.engine", ".fp32.engine", ".engine"):
            cand = root / ("gpen_bfr_%d.%d%s" % (size(), sm, suf))
            if cand.is_file():
                return cand
        hits = sorted(root.glob("gpen_bfr_%d*.engine" % size()))
        if hits:
            return hits[0]
    return None


def _build_session():
    """惰性建会话；失败把原因记在 _SESS["why"] 里（调用方只读不猜）。"""
    want = str(os.environ.get("DFL_SR_BACKEND") or "auto").strip().lower()
    eng = _engine_path()
    if want in ("auto", "trt") and eng is not None:
        try:
            from ..backend_models.trt_session import TrtSession
            # use_graph=False：CUDA Graph 在多线程下每条线程各捕一次（实测一条 40~80ms，
            # 8 条合成线程会反复捕 + 抢同一张图，跑流水线时直接卡住）。SR 的开销在算力上
            # （57ms），图重放省不到 2ms，不值得冒这个风险。own_stream=True：别和 torch 抢默认流。
            s = TrtSession(str(eng), use_graph=False, own_stream=True)
            return s, "trt", eng.name
        except Exception as e:                              # noqa: BLE001
            _SESS["why"] = "TRT 引擎加载失败（%s）：%s" % (eng.name, str(e)[:160])
            if want == "trt":
                return None, "trt", eng.name
    onnx = _model_path()
    if not onnx.is_file():
        _SESS["why"] = "找不到模型 %s（DFL_SR_MODEL 可指定）" % onnx
        return None, "", ""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 4
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
        s = ort.InferenceSession(str(onnx), so, providers=provs)
        return s, ("onnx-cuda" if provs and provs[0].startswith("CUDA") else "onnx-cpu"), onnx.name
    except Exception as e:                                  # noqa: BLE001
        _SESS["why"] = "onnxruntime 加载失败：%s" % str(e)[:200]
        return None, "", ""


def session():
    """返回 (会话, 后端名)；不可用时返回 (None, 原因)。"""
    with _LOCK:
        if _SESS["obj"] is None and not _SESS["why"]:
            obj, backend, name = _build_session()
            _SESS.update({"obj": obj, "backend": backend, "name": name})
            if obj is None:
                _log("不可用：" + (_SESS["why"] or "未知原因"))
            else:
                _log("后端 %s · %s" % (backend, name))
        return _SESS["obj"], (_SESS["why"] or "")


def info() -> dict:
    """给 [eff] / status 用的回显。"""
    return {"mode": mode(), "strength": round(strength(), 2), "size": size(),
            "backend": _SESS.get("backend") or "", "model": _SESS.get("name") or "",
            "why": _SESS.get("why") or ""}


_RUN_LOCK = threading.Lock()


def _run(nchw):
    """串行推理：合成车道是 8 条线程，同一个会话**不能**并发 run（TRT/ORT 都会崩）。
    GPU 本来也是串行的，锁住不损失吞吐。"""
    obj, why = session()
    if obj is None:
        return None
    with _RUN_LOCK:
        _infer(obj, nchw)


def _infer(obj, nchw):
    out = obj.run(None, {"input": nchw})
    # TrtSession 与 ort.InferenceSession 都是 run(outputs, feed) 签名
    return out[0] if isinstance(out, (list, tuple)) else out


def apply_roi(roi_u8, mask_f, alpha: float, timings: dict | None = None):
    """在**帧空间**的脸区域上跑一次超分并按 mask*alpha 混回。

    roi_u8 : 合成之后这块 ROI 的 uint8 BGR
    mask_f : 同一块 ROI 的遮罩（0..1，float32，HxW）
    返回新的 roi_u8（失败时返回原图，不抛 —— 合成不该因为超分挂掉，但要打日志）
    """
    import cv2
    import numpy as np

    if alpha <= 0 or roi_u8 is None or roi_u8.size == 0:
        return roi_u8
    timings = timings if isinstance(timings, dict) else {}
    m = np.asarray(mask_f, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != roi_u8.shape[:2]:
        m = cv2.resize(m, (roi_u8.shape[1], roi_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(m > 0.02)
    if xs.size == 0:
        return roi_u8
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = roi_u8[y0:y1, x0:x1]

    t0 = time.time()
    sz = size()
    big = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
    x = big.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    nchw = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    out = _run(nchw)
    if out is None:
        timings["sr_ms"] = timings.get("sr_ms", 0.0) + (time.time() - t0) * 1000.0
        return roi_u8
    y = np.asarray(out, np.float32)[0]
    y = np.transpose(y, (1, 2, 0))
    y = (y * 0.5 + 0.5) * 255.0
    sr = np.clip(y, 0, 255).astype(np.uint8)
    sr = cv2.resize(sr, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)

    sub_m = m[y0:y1, x0:x1][..., None]
    w = np.clip(sub_m * float(alpha), 0.0, 1.0)
    mixed = (sr.astype(np.float32) * w + crop.astype(np.float32) * (1.0 - w))
    out_roi = roi_u8.copy()
    out_roi[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
    timings["sr_ms"] = timings.get("sr_ms", 0.0) + (time.time() - t0) * 1000.0
    return out_roi
