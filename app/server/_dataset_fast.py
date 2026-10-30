"""数据集扫描/排序的快速实现（向量化 + 线程池）。

为什么重写：
  旧实现在 /api/dataset/scan 里逐张 PIL.open() 全尺寸解码，
  只为算一个 64 位的 dhash 或一个清晰度分数；重复图还是 O(n^2) 的
  Python 双层循环 + bin(x).count('1')。1204 张就要 2.9~5.6 秒，
  4 万张会到分钟级。

现在：
  * cv2.imdecode 直接读缩小后的灰度图（INTER_AREA），解码成本降一个量级；
  * dhash/清晰度/直方图一次遍历全部算完；
  * 汉明距离向量化（numpy 分块 popcount），O(n^2) 但常数小两个数量级；
  * 线程池并发（cv2 解码释放 GIL）。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_THUMB = 64          # 统一缩略尺寸（灰度）


def list_images(root: Path):
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in _EXTS)


def _features_one(path: Path):
    """返回 (dhash:uint64, sharp:float, hist:np.float32[16]) 或 None（坏图）。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if not buf.size:
            return None
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        small = cv2.resize(gray, (_THUMB, _THUMB), interpolation=cv2.INTER_AREA)

        # ---- dhash：9x8，相邻像素比较 → 64 bit ----
        d9 = cv2.resize(small, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (d9[:, 1:] > d9[:, :-1]).reshape(-1)
        packed = np.packbits(bits).tobytes()
        dh = int.from_bytes(packed, "big")

        # ---- 清晰度：高频/低频能量比（与 4.0 Filter 同方向，越大越清晰）----
        dn = cv2.resize(small, (_THUMB // 2, _THUMB // 2), interpolation=cv2.INTER_AREA)
        up = cv2.resize(dn, (_THUMB, _THUMB), interpolation=cv2.INTER_NEAREST)
        b0 = small.astype(np.float32) - up.astype(np.float32)
        e0 = float(np.mean(b0 * b0))
        e2 = float(np.var(dn.astype(np.float32))) + 1e-6
        sharp = e0 / e2

        # ---- 直方图（16 bin，归一化）----
        h, _ = np.histogram(small, bins=16, range=(0, 256))
        hist = (h / max(1, h.sum())).astype(np.float32)
        return dh, sharp, hist
    except Exception:
        return None


def extract(files, workers: int = 8):
    """并发提取特征，返回 (records, bad_names)。

    records: [{name, dhash, sharp, hist}]，顺序与 files 一致。
    """
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        feats = list(ex.map(_features_one, files))
    recs, bad = [], []
    for p, f in zip(files, feats):
        if f is None:
            bad.append(p.name)
            continue
        recs.append({"name": p.name, "dhash": f[0], "sharp": f[1], "hist": f[2]})
    return recs, bad


# ---------------------------------------------------------------- 汉明距离
def _popcount_table():
    t = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        t[i] = bin(i).count("1")
    return t


_POP = _popcount_table()


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,) uint64, b: (m,) uint64 → (n,m) uint8 汉明距离。"""
    x = a[:, None] ^ b[None, :]
    return _POP[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(axis=2, dtype=np.uint16)


def _block_masks(t: int, bits: int = 16):
    """所有 popcount <= t 的位掩码（用于多索引哈希的多探）。"""
    masks = [0]
    if t >= 1:
        masks += [1 << k for k in range(bits)]
    if t >= 2:
        masks += [(1 << a) | (1 << b) for a in range(bits) for b in range(a + 1, bits)]
    return masks


def find_duplicates(dhashes, similarity: float):
    """相似度 >= similarity 判重复，返回每个重复项指向的「首个代表」下标（-1 表示不重复）。

    算法：多索引哈希（Multi-Index Hashing，MIH）。
    64 位哈希切成 4 段 16 位；若两个哈希的汉明距离 <= r，按鸽巢原理，
    至少有一段的距离 <= floor(r/4)。于是：
      * 对每段建 16 位 -> 下标列表 的字典；
      * 查表时把该段的值做 <= t 位的翻转（t = r // 4）得到候选；
      * 候选再用完整 64 位精确校验。
    这是**精确**算法（无漏检），复杂度从 O(n^2) 降到 O(n · 4 · |masks|)。
    当 r 很大（similarity 很低）时退化为原来的分块矩阵算法。
    """
    n = len(dhashes)
    if n == 0:
        return []
    max_dist = int(round((1.0 - float(similarity)) * 64))
    arr = np.asarray(dhashes, dtype=np.uint64)

    t = max_dist // 4
    if t <= 2 and max_dist <= 11:          # 常见档位（sim >= 0.83）
        masks = _block_masks(t)
        blocks = [((arr >> np.uint64(48 - 16 * i)) & np.uint64(0xFFFF)).astype(np.uint16)
                  for i in range(4)]
        lists = [b.tolist() for b in blocks]

        tables = []
        for b in range(4):
            d = {}
            for i, v in enumerate(lists[b]):
                d.setdefault(v, []).append(i)
            tables.append(d)

        dup = [-1] * n
        for i in range(n):
            if dup[i] != -1:                # 已经是别人的重复项，跳过
                continue
            cand = set()
            for b in range(4):
                base = lists[b][i]
                tb = tables[b]
                for m in masks:
                    hit = tb.get(base ^ m)
                    if hit:
                        cand.update(hit)
            if not cand:
                continue
            cand.discard(i)
            if not cand:
                continue
            idx = np.fromiter(cand, dtype=np.int64, count=len(cand))
            dist = _POP[(arr[idx] ^ arr[i]).view(np.uint8)].reshape(idx.size, 8).sum(axis=1)
            near = idx[dist <= max_dist].tolist()
            earlier = [j for j in near if j < i]
            if earlier:                     # 归到更早的代表
                dup[i] = min(earlier)
                continue
            for j in near:                  # 自己当代表，把后面的都标掉
                if j != i and dup[j] == -1:
                    dup[j] = i
        return dup

    # r 很大：退回分块矩阵（保持正确性，慢一点）
    dup = np.full(n, -1, dtype=np.int64)
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        prev = arr[:end]
        d = _hamming(arr[start:end], prev)
        rows = np.arange(start, end)[:, None]
        cols = np.arange(end)[None, :]
        d = np.where(cols < rows, d, np.iinfo(np.uint16).max)
        hit = d <= max_dist
        first = np.argmax(hit, axis=1)
        has = hit.any(axis=1)
        dup[start:end] = np.where(has, first, -1)
    return dup.tolist()


def percentile_cut(values, ratio: float):
    """按分位淘汰最差的一批（ratio ∈ (0,1)），返回阈值。"""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), max(0.0, min(100.0, ratio * 100))))
