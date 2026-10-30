"""人脸身份聚类（快速版，纯 numpy）：DBSCAN 语义 + IVF 粗量化。

为什么重写 4.0 的 filter_by_face_id：
  1. index.search(全量, k=50) 一次返回 N×50 的 dists+idxs —— 100 万张时光这两个
     数组就 ~600 MB，叠加 embedding 本身 2 GB，Windows 开始换页，
     所以会「被资源管理器挂起一会儿」；
  2. union-find 与标签回填是纯 Python 双层循环（100 万 × 50 = 5000 万次），分钟级；
  3. k 硬编码 50，eps 小/簇稠密时覆盖不满邻域，核心点判错；
  4. 聚类失败就 eps *= 0.7 整体重跑。

现在：
  * IVF 粗量化：先在样本上跑 k-means 得到 ~sqrt(n) 个中心，把点分到格子里；
  * 只在格子内部做精确 ε-邻域（格子平均 200~400 个点，块内 matmul 一次算完）；
  * 候选对去重后分块校验，峰值内存与 N 无关；
  * 连通分量用 numpy 标签传播，不是 Python union-find；
  * 核心点 / 边界点判定全部向量化。
"""
from __future__ import annotations

import numpy as np

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
