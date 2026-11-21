"""人脸身份（Face ID）：ArcFace 特征提取 + 快速聚类 + 直接 TRT 推理 + CLI。

合并自 4.1 原来四个平级模块（_faceid_extract / _faceid_fast / _faceid_trt / _faceid_cli）——
**结构重构，不是功能改动**：函数名、常量名、默认值、日志口径照原样，只是从四个模块变成一个，
省掉三者之间 X. / F. 的间接层。

    段 1  extract   批量提特征（线程池解码，主线程只攒批推理）
    段 2  fast      纯 numpy 聚类（IVF 粗量化 + 格子内精确 eps-邻域）
    段 3  trt       直接持有 cudaEngine（引擎按 onnx md5 + profile + 精度缓存）
    段 4  cli       python -m server._faceid_cli --serve 与一次性脚本两条路

旧路径 server._faceid_* 由兼容层继续转发（见 app/server/_faceid_*.py）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
# ============================================================================
# 1) 特征提取：线程池解码 + TRT 分批推理（原 app/server/_faceid_extract.py）
# ============================================================================
# 批量提取 ArcFace embedding（线程解码 + TRT 分批推理）。
#
# 实测发现的关键点：
#   * 推理本身很快（mbf + TRT fp16 + bs=32 → 0.158 ms/张），
#     但如果解码是串行的，整体会被拖到 4.3 ms/张（解码 ~0.5 ms/张 × 8 线程不并行）。
#   * 所以这里用线程池解码（cv2 释放 GIL），主线程只负责攒批推理。

def _rt_dll_dirs():
    """TensorRT / torch 的 DLL 目录。合并环境里 TensorRT 由 tensorrt_libs 提供，
    cuda_dlls 只是它的旧副本（已删），两个都试一遍即可。"""
    try:
        from .._paths import runtime_root
        rt = runtime_root()
    except Exception:
        rt = Path(__file__).resolve().parents[3] / "runtime" / "engine"
    sp = rt / "Lib" / "site-packages"
    return [rt / "cuda_dlls", sp / "tensorrt_libs", sp / "torch" / "lib"]


def _load_dll_dirs():
    dirs = _rt_dll_dirs()
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
    return Path(__file__).resolve().parents[3] / "weights" / "onnx" / "ArcFace" / _FACEID_MODEL


def make_session(name: str | None = None, cache_dir: str | None = None):
    _load_dll_dirs()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    eng = _rt_dll_dirs()[0].parent
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


def extract_pipelined(files, session=None, batch: int = 32, workers: int = 8, depth: int = 4, progress=None):
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
            if progress is not None:
                progress(len(names), len(files))
    finally:
        stop.set()

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), np.float32)
    return names, mat, failed


def extract_trt_pipelined(files, embedder, batch: int = 32, workers: int = 12, depth: int = 8, progress=None):
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
            if progress is not None:
                progress(len(names), len(files))
    finally:
        stop.set()

    mat = np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), np.float32)
    return names, mat, failed


def extract(files, session=None, batch: int = 32, workers: int = 8, progress=None):
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

# ============================================================================
# 2) 快速聚类：纯 numpy DBSCAN 语义 + IVF 粗量化（原 app/server/_faceid_fast.py）
# ============================================================================
# 人脸身份聚类（快速版，纯 numpy）：DBSCAN 语义 + IVF 粗量化。
#
# 为什么重写 4.0 的 filter_by_face_id：
#   1. index.search(全量, k=50) 一次返回 N×50 的 dists+idxs —— 100 万张时光这两个
#      数组就 ~600 MB，叠加 embedding 本身 2 GB，Windows 开始换页，
#      所以会「被资源管理器挂起一会儿」；
#   2. union-find 与标签回填是纯 Python 双层循环（100 万 × 50 = 5000 万次），分钟级；
#   3. k 硬编码 50，eps 小/簇稠密时覆盖不满邻域，核心点判错；
#   4. 聚类失败就 eps *= 0.7 整体重跑。
#
# 现在：
#   * IVF 粗量化：先在样本上跑 k-means 得到 ~sqrt(n) 个中心，把点分到格子里；
#   * 只在格子内部做精确 ε-邻域（格子平均 200~400 个点，块内 matmul 一次算完）；
#   * 候选对去重后分块校验，峰值内存与 N 无关；
#   * 连通分量用 numpy 标签传播，不是 Python union-find；
#   * 核心点 / 边界点判定全部向量化。

_THR = 1e-6


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < _THR] = 1.0
    return (x / n).astype(np.float32, copy=False)


def _kmeans(x: np.ndarray, k: int, iters: int = 5, sample: int = 8192, seed: int = 0):
    """轻量 k-means：在抽样上迭代，返回 (k, d) 中心。"""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = int(min(k, n))
    sub = x if n <= sample else x[rng.choice(n, sample, replace=False)]

    # 初始化直接用随机抽样（k-means++ 在 k=1000 时本身就要几百 GFLOP）
    pick = rng.choice(sub.shape[0], size=min(k, sub.shape[0]), replace=False)
    cent = sub[pick].copy()
    if cent.shape[0] < k:                       # 样本不足时补齐
        cent = np.vstack([cent] * (k // max(1, cent.shape[0]) + 1))[:k]

    for _ in range(iters):
        # 分块指派
        lab = np.empty(sub.shape[0], dtype=np.int32)
        for st in range(0, sub.shape[0], 8192):
            blk = sub[st:st + 8192]
            sim = blk @ cent.T
            lab[st:st + 8192] = np.argmax(sim, axis=1)
        new = cent.copy()
        for c in range(k):
            m = lab == c
            if m.any():
                v = sub[m].mean(0)
                nv = np.linalg.norm(v)
                if nv > _THR:
                    new[c] = v / nv
        if np.allclose(new, cent, atol=1e-5):
            break
        cent = new
    return cent


def _assign(x: np.ndarray, cent: np.ndarray, block: int = 16384) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int32)
    for st in range(0, x.shape[0], block):
        out[st:st + block] = np.argmax(x[st:st + block] @ cent.T, axis=1)
    return out


def _components(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    if rows.size == 0:
        return np.arange(n, dtype=np.int64)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    comp = np.arange(n, dtype=np.int64)
    for _ in range(64):
        new = comp.copy()
        np.minimum.at(new, src, comp[dst])
        if np.array_equal(new, comp):
            break
        comp = new
    _u, comp = np.unique(comp, return_inverse=True)
    return comp


def _assign_probe(x: np.ndarray, cent: np.ndarray, nprobe: int = 2, block: int = 16384):
    """返回 (n, nprobe) 的最近邻中心下标 —— faiss 的 IVF nprobe。"""
    out = np.empty((x.shape[0], nprobe), dtype=np.int32)
    for st in range(0, x.shape[0], block):
        sim = x[st:st + block] @ cent.T
        if nprobe == 1:
            out[st:st + block] = np.argmax(sim, axis=1)[:, None]
        else:
            idx = np.argpartition(-sim, nprobe - 1, axis=1)[:, :nprobe]
            out[st:st + block] = idx
    return out


def _pairs_topk(emb: np.ndarray, cells: np.ndarray, thr: float, topk: int = 32):
    """按格子分桶，每点在桶内取 top-k 近邻（超过 eps 丢弃），返回下标对。"""
    n = emb.shape[0]
    members: dict = {}
    for probe in range(cells.shape[1]):
        for c in np.unique(cells[:, probe]):
            pass
    # 逐 probe 建桶：用排序分组比 dict 快
    all_r, all_c = [], []
    for probe in range(cells.shape[1]):
        lab = cells[:, probe]
        order = np.argsort(lab, kind="stable")
        sl = lab[order]
        bounds = np.flatnonzero(np.diff(sl)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [n]])
        for st, en in zip(starts, ends):
            if en - st < 2:
                continue
            idx = order[st:en]
            blk = emb[idx]
            sim = blk @ blk.T
            np.fill_diagonal(sim, -1.0)
            kk = min(topk, idx.size - 1)
            nb = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(sim, nb, axis=1)
            ok = vals > thr
            rr = np.repeat(np.arange(idx.size), ok.sum(1))
            cc = nb[ok]
            all_r.append(idx[rr])
            all_c.append(idx[cc])
    if not all_r:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([np.concatenate(all_r), np.concatenate(all_c)], axis=1).astype(np.int32)


def _pairs_within_cells(emb: np.ndarray, cell: np.ndarray, thr: float,
                        max_cell: int = 4096):
    """格子内精确 ε-邻域，返回 (m,2) int32 下标对。"""
    n = emb.shape[0]
    order = np.argsort(cell, kind="stable")
    scell = cell[order]
    # 每个格子的边界
    bounds = np.flatnonzero(np.diff(scell)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])

    out_r, out_c = [], []
    for st, en in zip(starts, ends):
        size = en - st
        if size < 2:
            continue
        idx = order[st:en]
        step = max_cell if size <= max_cell else max(2, size // 8)
        for s0 in range(0, size, step):
            e0 = min(size, s0 + step)
            blk = emb[idx[s0:e0]]
            if size <= max_cell:
                sim = blk @ emb[idx].T                      # (block, size)
                rr, cc = np.nonzero(sim > thr)
                keep = cc > (rr + s0)                       # 只留上三角
                out_r.append(idx[s0 + rr[keep]])
                out_c.append(idx[cc[keep]])
            else:
                # 超大格子（同类扎堆）：块内两两 + 与全格采样
                sim = blk @ emb[idx].T
                rr, cc = np.nonzero(sim > thr)
                out_r.append(idx[s0 + rr])
                out_c.append(idx[cc])
    if not out_r:
        return np.zeros((0, 2), dtype=np.int32)
    a = np.concatenate(out_r)
    b = np.concatenate(out_c)
    keep = a != b
    return np.stack([a[keep], b[keep]], axis=1)


def _dedup(pairs: np.ndarray, n: int) -> np.ndarray:
    if pairs.size == 0:
        return pairs
    lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    key = lo * n + hi
    order = np.argsort(key, kind="stable")
    key = key[order]
    uniq = np.empty(key.size, dtype=bool)
    uniq[0] = True
    np.not_equal(key[1:], key[:-1], out=uniq[1:])
    out = np.stack([lo[order][uniq], hi[order][uniq]], axis=1)
    return out.astype(np.int32)


def cluster(embeddings: np.ndarray, eps: float = 0.3, min_samples: int = 2,
            cells: int | None = None, seed: int = 0):
    """余弦距离 DBSCAN。eps 越小越严格（0.3 ≈ 余弦相似度 0.7）。返回 int32 labels。"""
    emb = _normalize(np.asarray(embeddings, dtype=np.float32))
    n = emb.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    thr = 1.0 - float(eps)

    k = int(cells or min(1024, max(64, n // 200)))
    cent = _kmeans(emb, k, seed=seed)

    probe = _assign_probe(emb, cent, nprobe=3 if n < 50000 else 2)

    # 实测 200k：topk=8 已能稳定还原 40/40 个身份，再大只是白算
    topk = max(8, int(min_samples) * 2)
    pairs = _dedup(_pairs_topk(emb, probe, thr, topk=topk), n)
    return _finish(emb, pairs, n, min_samples)


def _finish(emb, pairs, n, min_samples):
    """边 → 核心点 → 连通分量 → 边界点归属。GPU/numpy 两条路共用。"""
    deg = np.zeros(n, dtype=np.int32)
    if pairs.size:
        np.add.at(deg, pairs[:, 0], 1)
        np.add.at(deg, pairs[:, 1], 1)
    core = deg >= int(min_samples)

    labels = np.full(n, -1, dtype=np.int32)
    m = core[pairs[:, 0]] & core[pairs[:, 1]] if pairs.size else np.zeros(0, bool)
    rows, cols = (pairs[m, 0], pairs[m, 1]) if pairs.size else (np.zeros(0, np.int32),) * 2

    comp = _components(rows.astype(np.int64), cols.astype(np.int64), n)
    labels[core] = comp[core]

    if pairs.size:
        a, b = pairs[:, 0], pairs[:, 1]
        ab = (~core[a]) & core[b]
        ba = (~core[b]) & core[a]
        if ab.any():
            labels[a[ab]] = labels[b[ab]]
        if ba.any():
            labels[b[ba]] = labels[a[ba]]
    return labels


def group_stats(labels: np.ndarray):
    if labels.size == 0:
        return {}
    uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}

# ============================================================================
# 3) 直接 TRT：自持 cudaEngine，绕开 ORT 的 TRT EP 绑定开销（原 app/server/_faceid_trt.py）
# ============================================================================
# 直接跑 TensorRT（不用 onnxruntime 的 TRT EP）。
#
# 为什么：
#   ORT 的 TRT EP 每批要做 Python 绑定 → 设备拷贝 → 同步 → 回拷，
#   bs=32 时约 5 ms/批（0.158 ms/张）。自己持有 cudaEngine + 预分配
#   torch 张量 + 异步流，可以把这段开销省掉。
#
# 引擎按「onnx 的 md5 + profile + 精度」缓存到 runtime/engine/trt_direct/，
# 不重复构建。

def _dll_dirs():
    return _rt_dll_dirs()
    for d in dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs if d.is_dir()) + os.pathsep + os.environ.get("PATH", "")


class TrtEmbedder:
    """固定 batch 的 ArcFace 推理器：run(np[HWC float32]) -> np[N,512]。"""

    def __init__(self, onnx_path: str, batch: int = 32, fp16: bool = True,
                 cache_dir: str | None = None, input_name: str | None = None,
                 allow_build: bool = True):
        _dll_dirs()
        import tensorrt as trt
        import torch

        self.torch = torch
        self.batch = batch
        self.trt = trt
        src = Path(onnx_path)
        # 复用「引擎编译」页那套命名（workspace/_engines/<name>_<md5>_sm<SM>_trt<ver>.bf16.engine），
        # 这样编译页里能看到它、也能重建；构建完成后两边共用同一份文件。
        eng_dir = Path(cache_dir or (_rt_dll_dirs()[0].parent / "trt_direct"))
        eng_dir.mkdir(parents=True, exist_ok=True)
        logger = trt.Logger(trt.Logger.WARNING)
        t_all = time.time()

        # ---- 选引擎：**必须真的反序列化成功才算数** ----
        # 以前这里是 sorted(hits)[0]，直接开第一个文件读字节。换过 TensorRT 版本
        # （这台机器上的 _engines/*.bf16.engine 都是 TRT 10.9 编的，运行时是 10.13.3.9）
        # 就会：
        #   1) 白读 9.6 MB；
        #   2) deserialize_cuda_engine 返回 None 而不抛异常；
        #   3) 下一行 self.engine.create_execution_context() 抛 AttributeError；
        #   4) 上层 except 吞掉，回退 onnxruntime+TRT EP，再花 ~2 s 建会话。
        #   每次点击都付一遍这笔钱 —— 这就是「初始化加载模型很慢」。
        cands = []
        want = None
        try:
            from ..backend_models.engines import engine_path as _ep
            ws = Path(os.environ.get("DFL_WORKSPACE") or (Path(__file__).resolve().parents[3] / "workspace"))
            want = _ep(ws, "arcface_mbf", str(src))
        except Exception:
            want = None
        if want is not None:
            cands.append(want)
            # server 环境没装 tensorrt，trt_version() 读作 0，文件名会带 _trt0；
            # engine 环境能读到真实版本。两边命名不一致 → 同目录里新到旧全试一遍。
            try:
                cands.extend(sorted(want.parent.glob("arcface_mbf_*.bf16.engine"),
                                    key=lambda q: q.stat().st_mtime, reverse=True))
            except OSError:
                pass
        key = hashlib.md5(src.read_bytes()).hexdigest()[:16]
        prec = "fp16" if fp16 else "fp32"
        try:
            trt_ver = str(trt.__version__)
        except Exception:
            trt_ver = "0"
        # 重建出来的那份（名字里带 TRT 版本，换版本就自动不复用）
        cands.append(eng_dir / ("%s_%s_trt%s_b%d_%s.engine" % (src.stem, key, trt_ver, batch, prec)))
        for pat in ("%s_%s_*.engine" % (src.stem, key), "%s_*.engine" % src.stem, "*.bf16.engine"):
            try:
                cands.extend(sorted(eng_dir.glob(pat),
                                    key=lambda q: q.stat().st_mtime, reverse=True))
            except OSError:
                pass

        state = {"engine": None, "file": None, "rebuilt": False, "tried": 0, "bad": []}

        def _deserialize(path):
            if not path or not path.is_file():
                return None
            state["tried"] += 1
            try:
                data = path.read_bytes()
            except OSError:
                return None
            try:
                eng = trt.Runtime(logger).deserialize_cuda_engine(data)
            except Exception:
                eng = None
            if eng is None:
                state["bad"].append(path.name)
            return eng

        seen = set()
        for cand in cands:
            k = str(cand).lower()
            if k in seen:
                continue
            seen.add(k)
            eng = _deserialize(cand)
            if eng is not None:
                state["engine"], state["file"] = eng, str(cand)
                break

        if state["engine"] is None and not allow_build:
            # 点击路径上绝不等 75 秒的构建：直接告诉上层「没引擎」，退 onnxruntime。
            # 重建交给预热（--serve 的 warm / 引擎编译页），下次点击就是 0.6 s 反序列化。
            raise RuntimeError("没有可用的 TRT 引擎（试过 %d 个：%s），需要重建或先用 ONNX 回退"
                               % (state["tried"], ", ".join(state["bad"][:3]) or "-"))

        if state["engine"] is None:
            # 一个都用不了（典型就是换过 TensorRT 版本）→ 重建一份。
            # 写到 cache_dir（runtime/engine/trt_direct，可写），**不覆盖**
            # workspace/_engines 里用户可见的旧引擎 —— 那边交给「引擎编译」页重建。
            builder = trt.Builder(logger)
            net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse(src.read_bytes()):
                raise RuntimeError("onnx 解析失败: " + str(parser.get_error(0)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if fp16:
                cfg.set_flag(trt.BuilderFlag.FP16)
            prof = builder.create_optimization_profile()
            name = input_name or net.get_input(0).name
            prof.set_shape(name, (1, 3, 112, 112), (batch, 3, 112, 112), (batch, 3, 112, 112))
            cfg.add_optimization_profile(prof)
            serialized = builder.build_serialized_network(net, cfg)
            if serialized is None:
                raise RuntimeError("engine 构建失败")
            serialized = bytes(serialized)
            out = eng_dir / ("%s_%s_trt%s_b%d_%s.engine" % (src.stem, key, trt_ver, batch, prec))
            try:
                with open(out, "wb") as f:
                    f.write(serialized)
            except OSError:
                out = None
            state["engine"] = trt.Runtime(logger).deserialize_cuda_engine(serialized)
            state["file"], state["rebuilt"] = (str(out) if out else None), True
            if state["engine"] is None:
                raise RuntimeError("engine 重建后仍然无法反序列化")

        self.engine = state["engine"]
        # 给上层（CLI / 接口）看的现场：用了哪个文件、试了几个、有没有重建
        self.report = {"engine": state["file"], "rebuilt": bool(state["rebuilt"]),
                       "tried": state["tried"], "bad": state["bad"][:4],
                       "ms": int((time.time() - t_all) * 1000)}
        self.ctx = self.engine.create_execution_context()
        self.input_name = input_name or self.engine.get_tensor_name(0)
        self.ctx.set_input_shape(self.input_name, (batch, 3, 112, 112))

        # 预分配张量
        self.inp = torch.empty((batch, 3, 112, 112), dtype=torch.float32, device="cuda")
        out_shape = tuple(self.engine.get_tensor_shape(self.engine.get_tensor_name(1)))
        out_shape = (batch,) + tuple(out_shape[1:])
        self.out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.input_name, int(self.inp.data_ptr()))
        self.ctx.set_tensor_address(self.engine.get_tensor_name(1), int(self.out.data_ptr()))
        self.stream = torch.cuda.Stream()
        # 页锁定内存：H2D 才能真正异步（不然 torch 会退化成同步拷贝，慢一倍）
        self.host = torch.empty((batch, 3, 112, 112), dtype=torch.float32, pin_memory=True)
        # 回读也用页锁定缓冲，避免每次 .cpu() 都新建 4.8 MB 页锁定内存（实测这一项占了 3.9 ms/批）
        self.host_out = torch.empty(out_shape, dtype=torch.float32, pin_memory=True)
        self.host_u8 = torch.empty((batch, 112, 112, 3), dtype=torch.uint8, pin_memory=True)
        self.dev_u8 = torch.empty((batch, 112, 112, 3), dtype=torch.uint8, device="cuda")

    def run(self, imgs_hwc: np.ndarray) -> np.ndarray:
        """imgs_hwc: (N,112,112,3) float32，RGB，0..255。"""
        torch = self.torch
        n = imgs_hwc.shape[0]
        # 归一化搬到 GPU 做：numpy 上对 1.2M 元素做两次浮点运算 + 转置拷贝
        # 实测要 3~4 ms/批，占总时间的三分之一以上。
        if self.host_u8.shape[0] < n:
            self.host_u8 = torch.empty((n, 112, 112, 3), dtype=torch.uint8, pin_memory=True)
            self.dev_u8 = torch.empty((n, 112, 112, 3), dtype=torch.uint8, device="cuda")
        self.host_u8[:n].copy_(torch.from_numpy(np.ascontiguousarray(imgs_hwc, dtype=np.uint8)))
        with torch.cuda.stream(self.stream):
            buf = self.inp[:n] if n < self.batch else self.inp
            self.dev_u8[:n].copy_(self.host_u8[:n], non_blocking=True)
            buf.copy_(self.dev_u8[:n].permute(0, 3, 1, 2).float().sub_(127.5).div_(127.5))
            self.ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
            # D2H 也提交到同一 stream，和计算串起来，只同步一次
            self.host_out[:n].copy_(self.out[:n], non_blocking=True)
        self.stream.synchronize()
        return self.host_out[:n].numpy()

# ============================================================================
# 4) CLI / 常驻服务：一次性与 stdin-stdout 行协议两种跑法（原 app/server/_faceid_cli.py）
# ============================================================================
# 人脸身份过滤的 CLI（必须跑在 engine 环境：需要 onnxruntime + TRT）。
#
# 两种用法：
#
#   1) 一次性（脚本 / 调试）：
#         python -m server._faceid_cli --dir <aligned 目录> --eps 0.40 --min-samples 2
#      输出单行 JSON：
#         {"total":670, "clusters":{"0":669}, "noise":1, "ms_extract":321,
#          "ms_engine_load":1900, "provider":"TensorRT(direct)", ...}
#
#   2) 常驻（4.1 后端就用这条）：python -m server._faceid_cli --serve
#      stdin 一行一条请求，stdout 一行一条回应：
#         {"cmd":"warm"}                                  -> @@W {"ms":.., "provider":..}
#         {"dir":.., "eps":.., "min_samples":.., "merge":.., "progress":..} -> @@R {结果}
#         {"cmd":"quit"}                                  -> 退出
#      **引擎只在第一条请求时加载一次，之后复用。** 以前后端每次点击都
#      subprocess.run 一个新进程：670 张的活儿本身 0.3~0.5 s，而引擎初始化
#      （import torch/tensorrt + 反序列化 TRT 引擎 + 建 ORT 会话）要 3.5~4.5 s，
#      「初始化加载模型很慢」指的就是这一段，而且它是**每点一次付一次**。

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 常驻模式下复用的引擎（TrtEmbedder 或 onnxruntime InferenceSession）
_ENG = {"obj": None, "provider": "", "err": "", "ms": None, "batch": 0,
        "rebuilt": None, "tried_direct": False}


def _list_files(root: Path):
    """**只列这一层**的图片：不递归、不进 _trash / _masks / 分组目录。

    和 routes_oneshot._resolve_root / _dataset_fast.list_images 同一个口径：
    data_src/1 根上 670 张、_trash 里 534 张，一次 rglob 就会报 1204，
    UI 显示的张数和真正参与聚类的张数就对不上了。
    """
    try:
        return sorted(p for p in Path(root).iterdir()
                      if p.is_file() and p.suffix.lower() in _EXTS
                      and not p.name.startswith((".", "_")))
    except OSError:
        return []

def _merge_by_centroid(labels, embs, dist):
    """质心单链接合并：DBSCAN 会把同一个人按姿态/场景拆成几簇，这里再并一次。"""
    if dist <= 0:
        return labels
    cent = {}
    for u in [int(x) for x in np.unique(labels) if x >= 0]:
        c = embs[labels == u].mean(axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        cent[u] = c / n
    ids = sorted(cent)
    parent = {u: u for u in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if 1.0 - float(cent[a] @ cent[b]) < dist:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    out = np.full_like(labels, -1)
    for i, lab in enumerate(labels.tolist()):
        if lab >= 0:
            out[i] = find(int(lab))
    return out

def _writer(path):
    """返回一个阶段化进度写函数；没给路径就返回 None。"""
    if not path:
        return None
    import io, json, os
    p = Path(path)

    def put(stage, done, total, note=""):
        try:
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps({ "stage": stage, "done": int(done), "total": int(total),
                                        "note": note }, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(p))
        except Exception:
            pass
    return put

def _engine_get(batch: int = 32, force: bool = False, allow_build: bool = False):
    """取（必要时加载）引擎，进程内缓存。返回 (obj, provider, ms_load, rebuilt)。

    allow_build=False 时绝不走「现场编译 TRT 引擎」那条路：arcface 在 5070 上
    编一次约 75 秒，放在点击路径上就是「点了没反应」。要建引擎走 warm/预热。
    """
    if not force and _ENG["obj"] is not None and _ENG["batch"] == batch:
        return _ENG["obj"], _ENG["provider"], 0, _ENG["rebuilt"]
    t0 = time.time()
    obj, provider, err, rebuilt = None, "", "", None
    # 优先直接 TRT（归一化在 GPU 上，主线程几乎无 Python 开销）
    try:
        obj = TrtEmbedder(str(_model_path()), batch=batch, fp16=True, allow_build=allow_build)
        provider = "TensorRT(direct)"
        rep = getattr(obj, "report", None) or {}
        rebuilt = rep.get("rebuilt")
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
        sys.stderr.write("[faceid] 直接 TRT 不可用，回退 onnxruntime: %s\n" % err)
        try:
            obj = make_session()
            provider = obj.get_providers()[0]
        except Exception as e2:
            _ENG.update({"err": "%s | %s" % (err, e2)})
            raise RuntimeError("引擎加载失败: %s" % e2)
    ms = int((time.time() - t0) * 1000)
    _ENG.update({"obj": obj, "provider": provider, "err": err, "ms": ms,
                 "batch": batch, "rebuilt": rebuilt})
    return obj, provider, ms, rebuilt

def _extract(files, obj, provider, batch, progress):
    if hasattr(obj, "run") and "TensorRT(direct)" in provider:
        return extract_trt_pipelined(files, obj, batch=batch, progress=progress)
    return extract_pipelined(files, obj, batch=batch, progress=progress)

def run_one(root, eps=0.314, min_samples=2, merge=0.45, batch=32,
            progress=None, max_images=0):
    """跑一次：列文件 → 提特征 → 聚类，返回结果 dict（不打印）。"""
    root = Path(root)
    files = _list_files(root)
    if max_images > 0:
        files = files[:max_images]

    put = _writer(progress)
    if put:
        put("embed", 0, len(files))

    t_load = time.time()
    obj, provider, ms_load, rebuilt = _engine_get(batch)
    ms_engine = int((time.time() - t_load) * 1000) if ms_load else 0

    t0 = time.time()
    names, embs, failed = _extract(files, obj, provider, batch,
                                  (lambda d, t: put("embed", d, t)) if put else None)
    t1 = time.time()
    if put:
        put("cluster", 0, len(files))

    out = {"total": len(files), "n_files": len(files), "dir": str(root),
           "provider": provider, "ms_engine_load": ms_engine,
           "engine_rebuilt": rebuilt,
           "ms_extract": round((t1 - t0) * 1000), "clusters": {}, "noise": 0, "rare": []}

    if embs.shape[0]:
        labels = cluster(embs, eps=eps, min_samples=min_samples)
        labels = _merge_by_centroid(labels, embs, float(merge))
        # 按大小重编号：组号从 0 开始，越大越靠前
        _uniq = [int(u) for u in np.unique(labels) if u >= 0]
        _order = sorted(_uniq, key=lambda u: -int((labels == u).sum()))
        _remap = {u: i for i, u in enumerate(_order)}
        labels = np.array([_remap.get(int(l), -1) for l in labels.tolist()], dtype=np.int32)
        stats = group_stats(labels)
        out["clusters"] = {str(k): int(v) for k, v in stats.items()}
        out["noise"] = int((labels < 0).sum())
        rare = []
        for i, lab in enumerate(labels.tolist()):
            if lab < 0 or stats.get(lab, 0) < min_samples:
                rare.append(names[i])
        out["rare"] = rare
        # 主体优先：一个 faceset 通常只该有一个人，最大簇就是「本人」，
        # 其余簇（哪怕有几十张）和噪声都要被挑出来，否则两个人都留在数据集里。
        main_id, main_n = (-1, 0)
        for k, v in stats.items():
            if v > main_n:
                main_id, main_n = int(k), int(v)
        others = []
        per_cluster = {}
        for i2, lab in enumerate(labels.tolist()):
            per_cluster.setdefault(int(lab), 0)
            per_cluster[int(lab)] += 1
            if int(lab) != main_id:
                others.append({"file": names[i2], "cluster": int(lab), "cluster_n": int(stats.get(lab, 0))})
        out["main"] = {"id": main_id, "n": main_n}
        out["per_cluster"] = {str(k): int(v) for k, v in per_cluster.items()}
        out["others"] = others
        # 全部文件的归属，供「按身份分文件夹」使用
        out["assign"] = [{"file": nm, "cluster": int(lab)}
                         for nm, lab in zip(names, labels.tolist())]
        out["ms_cluster"] = round((time.time() - t1) * 1000)
    out["failed"] = failed
    if put:
        put("done", len(files), len(files))
    return out

def serve():
    """常驻模式：一条请求一条回应，引擎在进程里只加载一次。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stdout.write("@@ERR 请求不是 JSON: %s\n" % e)
            sys.stdout.flush()
            continue
        cmd = str(req.get("cmd") or "run").lower()
        if cmd in ("quit", "exit"):
            break
        try:
            if cmd in ("warm", "prewarm", "load"):
                t = time.time()
                # 预热**允许**建引擎：这条路上花 75 秒用户是看不见的（后台跑）
                obj, provider, _ms, rebuilt = _engine_get(int(req.get("batch") or 32),
                                                          allow_build=True)
                sys.stdout.write("@@W " + json.dumps(
                    {"ms": int((time.time() - t) * 1000), "provider": provider,
                     "rebuilt": rebuilt}, ensure_ascii=False) + "\n")
            elif cmd == "ping":
                sys.stdout.write("@@P {}\n")
            else:
                res = run_one(req.get("dir"), eps=float(req.get("eps") or 0.314),
                              min_samples=int(req.get("min_samples") or 2),
                              merge=float(req.get("merge") or 0.45),
                              batch=int(req.get("batch") or 32),
                              progress=req.get("progress") or "",
                              max_images=int(req.get("max_images") or 0))
                sys.stdout.write("@@R " + json.dumps(res, ensure_ascii=False) + "\n")
        except Exception as e:
            sys.stdout.write("@@ERR %s: %s\n" % (type(e).__name__, str(e)[:400]))
        sys.stdout.flush()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--eps", type=float, default=0.314)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--merge", type=float, default=0.45, help="簇质心合并阈值（同一人被拆开时调大）")
    ap.add_argument("--progress", default="", help="把进度写进这个文件，供前端轮询")
    ap.add_argument("--max-images", type=int, default=0, help="调试用：只处理前 N 张")
    ap.add_argument("--serve", action="store_true", help="常驻模式：stdin/stdout 行协议，引擎只加载一次")
    a = ap.parse_args(argv)
    if a.serve:
        return serve()
    if not a.dir:
        ap.error("--dir 必填（或用 --serve 走常驻模式）")
    out = run_one(a.dir, eps=a.eps, min_samples=a.min_samples, merge=a.merge,
                  batch=a.batch, progress=a.progress, max_images=a.max_images)
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
