# -*- coding: utf-8 -*-
"""训练几何异常检测器（errface_xgb）。

任务：判断一组 68 点是否可能来自真实人脸。
  good：真实检测点（大侧脸加权，避免退化成"侧脸检测器"）
  bad ：受控破坏 —— 打乱点序 / 随机替换 / 簇塌缩 / 左右眼互换 /
        非等比拉伸 / 强剪切 / 上下半脸错位
        可选 --neg-dir：真实误检样本（DFL 图，直接当 bad）

关键约束：**不要**用旋转/缩放/平移当坏样本——相似变换后仍是合法人脸。

数据准备（推荐）：
    cd app
    python -m tools.build_errface_dataset --src D:/faces --out ../workspace/errface_ds.npz

训练：
    cd app
    python -m tools.train_errface --npz ../workspace/errface_ds.npz \
        --out ../weights/errface_xgb.ubj
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

CLUSTERS = [slice(0, 17), slice(17, 27), slice(27, 36), slice(36, 42), slice(42, 48), slice(48, 68)]
KIND_NAMES = ["打乱点序", "随机替换", "簇塌缩", "左右眼互换", "非等比拉伸", "强剪切", "半脸错位"]


def break_shape(p: np.ndarray, rng, kind: int | None = None) -> np.ndarray:
    q = p.copy()
    k = rng.integers(0, 7) if kind is None else kind
    c = q.mean(0)
    if k == 0:
        q = q[rng.permutation(len(q))]
    elif k == 1:
        idx = rng.choice(len(q), size=int(rng.integers(8, 40)), replace=False)
        spread = q.max(0) - q.min(0)
        q[idx] = c + rng.uniform(-1.2, 1.2, size=(len(idx), 2)) * spread
    elif k == 2:
        s = CLUSTERS[rng.integers(0, len(CLUSTERS))]
        q[s] = q[s].mean(0) + rng.normal(0, 3, size=(q[s].shape[0], 2))
    elif k == 3:
        q[36:42], q[42:48] = q[42:48].copy(), q[36:42].copy()
    elif k == 4:
        q = (q - c) * np.array([rng.uniform(0.4, 0.7), rng.uniform(1.3, 2.2)]) + c
    elif k == 5:
        sh = rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
        q = (q - c) @ np.array([[1, sh], [0, 1]]).T + c
    else:
        q[:36] += rng.uniform(30, 80, size=2) * rng.choice([-1, 1], size=2)
    return q


def load_npz(path: Path):
    d = np.load(path, allow_pickle=False)
    return d["X"].astype(np.float64).reshape(-1, 68, 2), d["ident"].astype(str)


def load_neg_dir(path: Path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from server._align import read_dfljpg_meta
    from server._errface import _img_h

    out = []
    for f in sorted(glob.glob(str(path / "*.jpg"))):
        m = read_dfljpg_meta(f)
        if isinstance(m, dict) and m.get("landmarks") is not None:
            h = _img_h(Path(f))
            if h > 0:
                out.append(np.asarray(m["landmarks"], np.float64) * (512.0 / h))
    return np.stack(out) if out else np.zeros((0, 68, 2))


def build(rows, seed: int, ratio: int = 3, sample_weights=None):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, p in enumerate(rows):
        rep = int(round(sample_weights[i])) if sample_weights is not None else 1
        rep = max(1, rep)
        for _ in range(rep):
            X.append((p + rng.normal(0, 1.0, p.shape)).reshape(-1)); y.append(0)
        m = p[::-1].copy(); m[:, 0] = -m[:, 0]
        X.append(m.reshape(-1)); y.append(0)
        for _ in range(ratio):
            X.append(break_shape(p, rng).reshape(-1)); y.append(1)
    return np.array(X).astype(np.int64), np.array(y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", action="append", required=True,
                    help="数据集，可重复；用 路径:权重 指定该源权重，如 a.npz:1 b.npz:6")
    ap.add_argument("--out", default="../weights/errface_xgb.ubj")
    ap.add_argument("--neg-dir", default="", help="真实误检样本目录（DFL 对齐图）")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--ratio", type=int, default=3, help="每条正样本生成几个破坏样本")
    args = ap.parse_args()

    # 多源合并：不同来源可以给不同权重（例如本项目侧脸样本权重 6）
    Xs, idents, weights = [], [], []
    for spec in args.npz:
        path, _, w = spec.partition(":")
        Xi, ii = load_npz(Path(path))
        w = float(w) if w else 1.0
        print("  载入 %-40s %6d 条  权重 %.1f" % (Path(path).name, len(Xi), w))
        Xs.append(Xi); idents.append(ii); weights.append(np.full(len(Xi), w))
    X = np.vstack(Xs)
    idents = np.concatenate(idents)
    weights = np.concatenate(weights)
    print("合计 %d 条，身份 %d" % (len(X), len(set(idents.tolist()))))

    # 按身份划分：同一人不会同时出现在训练/验证
    uniq = np.array(sorted(set(idents.tolist())))
    rng = np.random.default_rng(42)
    rng.shuffle(uniq)
    cut = max(1, int(len(uniq) * 0.85))
    tr_ids = set(uniq[:cut].tolist())
    mask = np.array([i in tr_ids for i in idents])
    tr, va = X[mask], X[~mask]
    print("训练身份 %d（%d 张） / 验证身份 %d（%d 张）" % (
        len(tr_ids), len(tr), len(uniq) - len(tr_ids), len(va)))

    Xtr, ytr = build(tr, 1, args.ratio, weights[mask])
    Xva, yva = build(va, 2, args.ratio, None)

    neg = load_neg_dir(Path(args.neg_dir)) if args.neg_dir else np.zeros((0, 68, 2))
    if len(neg):
        print("真实误检负样本 %d 张" % len(neg))
        Xtr = np.vstack([Xtr, neg.reshape(len(neg), -1).astype(np.int64)])
        ytr = np.concatenate([ytr, np.ones(len(neg))])

    import xgboost as xgb
    bst = xgb.train(dict(max_depth=6, eta=0.12, objective="binary:logistic",
                         eval_metric="auc", tree_method="hist", nthread=8),
                    xgb.DMatrix(Xtr, label=ytr), args.rounds)

    p = bst.predict(xgb.DMatrix(Xva))
    order = np.argsort(p); rank = np.empty_like(order, float); rank[order] = np.arange(1, len(p) + 1)
    n1 = int(yva.sum()); n0 = len(yva) - n1
    auc = (rank[yva == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.0
    print("验证 AUC = %.4f（按身份划分）" % auc)

    real = np.stack([q.reshape(-1) for q in va]).astype(np.int64)
    pr = bst.predict(xgb.DMatrix(real))
    print("真实脸误报: >=0.5 %.2f%%   >=0.8 %.2f%%" % (
        100 * (pr >= 0.5).mean(), 100 * (pr >= 0.8).mean()))

    r2 = np.random.default_rng(7)
    sample = va[r2.choice(len(va), min(150, len(va)), replace=False)]
    print("各类破坏的召回：")
    for k, name in enumerate(KIND_NAMES):
        Xk = np.stack([break_shape(q, r2, k).reshape(-1) for q in sample]).astype(np.int64)
        pk = bst.predict(xgb.DMatrix(Xk))
        print("  %-12s %5.1f%%" % (name, 100 * (pk >= 0.5).mean()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(out))
    print("saved →", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
