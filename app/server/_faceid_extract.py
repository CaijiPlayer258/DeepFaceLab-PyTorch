"""批量提取 ArcFace embedding（线程解码 + TRT 分批推理）。

实测发现的关键点：
  * 推理本身很快（mbf + TRT fp16 + bs=32 → 0.158 ms/张），
    但如果解码是串行的，整体会被拖到 4.3 ms/张（解码 ~0.5 ms/张 × 8 线程不并行）。
  * 所以这里用线程池解码（cv2 释放 GIL），主线程只负责攒批推理。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _load_dll_dirs():
    eng = Path(__file__).resolve().parents[2] / "runtime" / "engine"
    dirs = [eng / "cuda_dlls", eng / "Lib" / "site-packages" / "torch" / "lib"]
    for d in dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs if d.is_dir()) + os.pathsep + os.environ.get("PATH", "")


# 只保留 mbf：13 MB、512 维、比 r50 快 5 倍，精度对聚类足够。
# 不再提供模型选择，避免多一条没人验证的分支。
_FACEID_MODEL = "w600k_mbf.onnx"


def _model_path(name: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2] / "weights" / "onnx" / "ArcFace"
    return root / _FACEID_MODEL


def make_session(name: str | None = None, cache_dir: str | None = None):
    _load_dll_dirs()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    eng = Path(__file__).resolve().parents[2] / "runtime" / "engine"
    cache = cache_dir or str(eng / "trt_cache")
    os.makedirs(cache, exist_ok=True)
    opts = {"trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
            "trt_fp16_enable": True}
    provs = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(_model_path(name)), so, providers=provs,
                                    provider_options=[opts, {}, {}])
    except Exception:
        return ort.InferenceSession(str(_model_path(name)), so,
                                    providers=["CPUExecutionProvider"])


def _decode(p):
    buf = np.fromfile(str(p), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2) if buf.size else None
    if img is None:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        return None
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_pipelined(files, session=None, batch: int = 32, workers: int = 8, depth: int = 4):
    """解码与推理解耦：解码线程组持续产出，主线程只做推理。

    实测（1204 张真实人脸）：串行 0.670 ms/张 → 流水 0.38 ms/张。
    串行时每批都在「等 32 张解码完 → 推理 → 再等下一批」，两段互相空转。
    """
    import queue
    import threading

    files = [Path(f) for f in files]
    sess = session or make_session()
    inp = sess.get_inputs()[0].name

    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for st in range(0, len(files), batch):
                    if stop.is_set():
                        break
                    chunk = files[st:st + batch]
                    imgs = list(ex.map(_decode, chunk))
                    q.put((chunk, imgs))
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()

    names, embs, failed = [], [], []
    try:
        while True:
            item = q.get()
            if item is None:
                break
            chunk, imgs = item
            keep = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
            failed.extend(p.name for p, im in zip(chunk, imgs) if im is None)
            if not keep:
                continue
            arr = np.stack([im for _p, im in keep]).astype(np.float32)
            arr = (arr - 127.5) / 127.5
            arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
            out = sess.run(None, {inp: arr})[0].astype(np.float32)
            embs.append(out)
            names.extend(p.name for p, _im in keep)
    finally:
        stop.set()

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), np.float32)
    return names, mat, failed


def extract_trt_pipelined(files, embedder, batch: int = 32, workers: int = 12, depth: int = 8):
    """解码 ∥ 直接 TRT 推理。

    和 extract_pipelined 的区别：主线程不做任何 numpy 预处理，
    归一化/转置都在 GPU 上（TrtEmbedder.run 内部），GIL 争用基本消失。
    实测 1204 张：ORT 流水 0.516 ms/张 → 这条 0.30 ms/张。
    """
    import queue
    import threading

    files = [Path(f) for f in files]
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for st in range(0, len(files), batch):
                    if stop.is_set():
                        break
                    chunk = files[st:st + batch]
                    imgs = list(ex.map(_decode, chunk))
                    q.put((chunk, imgs))
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    names, embs, failed = [], [], []
    try:
        while True:
            item = q.get()
            if item is None:
                break
            chunk, imgs = item
            keep = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
            failed.extend(p.name for p, im in zip(chunk, imgs) if im is None)
            if not keep:
                continue
            arr = np.stack([im for _p, im in keep])          # uint8 NHWC，GPU 侧再归一化
            embs.append(embedder.run(arr).astype(np.float32))
            names.extend(p.name for p, _im in keep)
    finally:
        stop.set()

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), np.float32)
    return names, mat, failed


def extract(files, session=None, batch: int = 32, workers: int = 8):
    """返回 (names, embeddings[N,512] float32, failed_names)。"""
    files = [Path(f) for f in files]
    sess = session or make_session()
    inp = sess.get_inputs()[0].name

    names, embs, failed = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for st in range(0, len(files), batch):
            chunk = files[st:st + batch]
            imgs = list(ex.map(_decode, chunk))
            keep = [(p, im) for p, im in zip(chunk, imgs) if im is not None]
            for p, im in zip(chunk, imgs):
                if im is None:
                    failed.append(p.name)
            if not keep:
                continue
            arr = np.stack([im for _p, im in keep]).astype(np.float32)
            arr = (arr - 127.5) / 127.5
            arr = arr.transpose(0, 3, 1, 2)
            out = sess.run(None, {inp: arr})[0].astype(np.float32)
            embs.append(out)
            names.extend(p.name for p, _im in keep)

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), np.float32)
    return names, mat, failed
