"""人脸身份过滤的 CLI（必须跑在 engine 环境：需要 onnxruntime + TRT）。

用法：
    python -m server._faceid_cli --dir <aligned 目录> --eps 0.314 --min-samples 2
输出：单行 JSON
    {"total":1204, "clusters":{"0":1136,"1":67}, "noise":1,
     "rare":["a.jpg", ...], "ms_extract":810, "ms_cluster":110, "provider":"..."}
rare = 样本数少于 min_samples 的簇 + 噪声点（判定为「稀少身份」，调用方负责移入 _trash）
"""
from __future__ import annotations
import numpy as np

import argparse
import json
import sys
import time
from pathlib import Path

from . import _faceid_extract as X
from . import _faceid_fast as F


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--eps", type=float, default=0.314)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--merge", type=float, default=0.45, help="簇质心合并阈值（同一人被拆开时调大）")
    ap.add_argument("--progress", default="", help="把进度写进这个文件，供前端轮询")
    ap.add_argument("--max-images", type=int, default=0, help="调试用：只处理前 N 张")
    a = ap.parse_args(argv)

    root = Path(a.dir)
    files = sorted(p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if a.max_images > 0:
        files = files[:a.max_images]

    put = _writer(a.progress)
    if put:
        put("embed", 0, len(files))

    # 优先直接 TRT（归一化在 GPU 上，主线程几乎无 Python 开销）；失败则退回 onnxruntime
    t0 = time.time()
    provider = "unknown"
    try:
        from ._faceid_trt import TrtEmbedder
        _emb = TrtEmbedder(str(X._model_path()), batch=a.batch, fp16=True)
        names, embs, failed = X.extract_trt_pipelined(
            files, _emb, batch=a.batch,
            progress=(lambda d, t: put("embed", d, t)) if put else None)
        provider = "TensorRT(direct)"
    except Exception as _e:
        import sys as _s
        _s.stderr.write("[faceid] 直接 TRT 不可用，回退 onnxruntime: %s\n" % _e)
        _sess = X.make_session()
        names, embs, failed = X.extract_pipelined(
            files, _sess, batch=a.batch,
            progress=(lambda d, t: put("embed", d, t)) if put else None)
        provider = _sess.get_providers()[0]
    t1 = time.time()
    if put:
        put("cluster", 0, len(files))

    out = {"total": len(files), "provider": provider,
           "ms_extract": round((t1 - t0) * 1000), "clusters": {}, "noise": 0, "rare": []}
    if embs.shape[0]:
        labels = F.cluster(embs, eps=a.eps, min_samples=a.min_samples)
        labels = _merge_by_centroid(labels, embs, float(a.merge))
        # 按大小重编号：组号从 0 开始，越大越靠前
        _uniq = [int(u) for u in np.unique(labels) if u >= 0]
        _order = sorted(_uniq, key=lambda u: -int((labels == u).sum()))
        _remap = {u: i for i, u in enumerate(_order)}
        labels = np.array([_remap.get(int(l), -1) for l in labels.tolist()], dtype=np.int32)
        stats = F.group_stats(labels)
        out["clusters"] = {str(k): int(v) for k, v in stats.items()}
        out["noise"] = int((labels < 0).sum())
        rare = []
        for i, lab in enumerate(labels.tolist()):
            if lab < 0 or stats.get(lab, 0) < a.min_samples:
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
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
