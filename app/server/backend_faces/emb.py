# -*- coding: utf-8 -*-
"""ArcFace 人脸特征（512 维 embedding）—— 人脸库「组 = dfm」判定的唯一实现。

**为什么单独一个模块**：同一张脸要能拿两个来源的向量直接比余弦 —— 库里存下来的
PNG（老库补算）和当前合成帧里裁出来的那一块（自动分配）。两边的预处理只要差一个
像素级细节（BGR/RGB、归一化区间、插值算法），相似度就会整体偏低几档，
0.35 这种阈值立刻失效。所以「怎么裁 → 怎么归一化 → 怎么变成 512 维」只在这里写一次。

模型：权重根（$DFL_WEIGHTS_DIR 或 <仓库>/weights）下的 arcface_mbf.onnx（13 MB，
512 维）。老的分目录布局 onnx/ArcFace/w600k_mbf.onnx 仍然认。

运行设备**固定 CPU**：mbf 一张 112×112 只要 3~6 ms，人脸库里一次最多几十张；
而合成用的 .dfm 动辄 0.5~1.6 GB，占显存才是瓶颈。把 embedding 挤到 CPU 上，
预览时不会因为显存不足把 dfm 顶出显存。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

MODEL_NAME = "arcface_mbf.onnx"
EMB_DIM = 512
INPUT_SIZE = 112

# 老的分目录布局（兼容老权重树，不是默认路径）
_LEGACY_RELS = ("onnx/ArcFace/w600k_mbf.onnx",
                "onnx/ArcFace/arcface_mbf.onnx")

_lock = threading.Lock()
_state = {"sess": None, "inp": "", "path": None, "tried": False, "error": ""}


def candidates() -> list:
    """按优先级列出可能的模型路径（只列，不判断）。"""
    roots = []
    env = str(os.environ.get("DFL_WEIGHTS_DIR") or "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path(__file__).resolve().parents[3] / "weights")
    out = []
    for root in roots:
        out.append(root / MODEL_NAME)
        for rel in _LEGACY_RELS:
            out.append(root / rel)
    return out


def model_path():
    """找到的模型文件；一个都没有返回 None（调用方自己决定是报错还是跳过）。"""
    for p in candidates():
        if p.is_file():
            return p
    return None


def available() -> bool:
    return model_path() is not None


def _session():
    """(session, 输入名)；模型缺失/加载失败都返回 (None, "")，绝不抛给上层。"""
    with _lock:
        if _state["tried"]:
            return _state["sess"], _state["inp"]
        _state["tried"] = True
        path = model_path()
        if path is None:
            _state["error"] = "找不到 %s（找过：%s）" % (
                MODEL_NAME, " / ".join(str(p) for p in candidates()))
            return None, ""
        try:
            import onnxruntime
            so = onnxruntime.SessionOptions()
            so.log_severity_level = 3
            sess = onnxruntime.InferenceSession(
                str(path), so, providers=["CPUExecutionProvider"])
            _state["sess"] = sess
            _state["inp"] = sess.get_inputs()[0].name
            _state["path"] = str(path)
            return _state["sess"], _state["inp"]
        except Exception as e:                             # noqa: BLE001
            _state["error"] = "加载 %s 失败：%s" % (path.name, str(e)[:200])
            return None, ""


def last_error() -> str:
    return str(_state.get("error") or "")


def _prep(img):
    """BGR uint8（任意尺寸）→ (3,112,112) float32，[-1,1]。"""
    import cv2
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] != INPUT_SIZE or arr.shape[1] != INPUT_SIZE:
        arr = cv2.resize(arr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def _normalize(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return v.astype(np.float32, copy=False)


def embed(img):
    """一张 BGR 图 → 512 维 L2 归一化向量；失败返回 None。"""
    blob = _prep(img)
    if blob is None:
        return None
    sess, inp = _session()
    if sess is None:
        return None
    try:
        out = sess.run(None, {inp: blob[None, :, :, :]})[0]
    except Exception:                                       # noqa: BLE001
        return None
    v = np.asarray(out, dtype=np.float32).reshape(-1)
    if v.size != EMB_DIM:
        return None
    return _normalize(v)


def embed_many(imgs, batch: int = 32):
    """批量版：返回与输入等长的列表（失败的位置是 None）。

    离线补算一个老库（几百张）时按 32 张一批跑，比逐张快一个量级。
    """
    out = [None] * len(imgs)
    sess, inp = _session()
    if sess is None:
        return out
    step = max(1, int(batch))
    for st in range(0, len(imgs), step):
        chunk = imgs[st:st + step]
        blobs, keep = [], []
        for j, img in enumerate(chunk):
            b = _prep(img)
            if b is None:
                continue
            blobs.append(b)
            keep.append(st + j)
        if not blobs:
            continue
        try:
            res = sess.run(None, {inp: np.stack(blobs)})[0]
        except Exception:                                   # noqa: BLE001
            continue
        arr = np.asarray(res, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        for j, pos in enumerate(keep):
            v = arr[j].reshape(-1)
            if v.size == EMB_DIM:
                out[pos] = _normalize(v)
    return out


def cosine(a, b) -> float:
    """两个归一化向量的余弦相似度（未归一化也行，这里再归一一次）。"""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size != vb.size or va.size == 0:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(va / na, vb / nb))
