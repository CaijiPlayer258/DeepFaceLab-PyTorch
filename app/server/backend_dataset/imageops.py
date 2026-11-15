# -*- coding: utf-8 -*-
"""图片级工具：目录列举、拉普拉斯方差（清晰度）、dHash（去重）（从 app/server/app.py 切出）。

数据集体检（One Shot）和排序/去重都用这三件；函数体一行没改。
"""
from __future__ import annotations

from pathlib import Path


def _list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _lap_var(img) -> float:
    """拉普拉斯方差：越小越糊。"""
    import numpy as np
    g = img.convert("L").resize((256, 256))
    a = np.asarray(g, dtype=np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = a.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * a[dy:dy + h - 2, dx:dx + w - 2]
    return float(out.var())


def _dhash(img, size=8) -> int:
    g = img.convert("L").resize((size + 1, size))
    import numpy as np
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v
