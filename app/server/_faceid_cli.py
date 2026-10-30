"""人脸身份过滤的 CLI（必须跑在 engine 环境：需要 onnxruntime + TRT）。

用法：
    python -m server._faceid_cli --dir <aligned 目录> --eps 0.314 --min-samples 2
输出：单行 JSON
    {"total":1204, "clusters":{"0":1136,"1":67}, "noise":1,
     "rare":["a.jpg", ...], "ms_extract":810, "ms_cluster":110, "provider":"..."}
rare = 样本数少于 min_samples 的簇 + 噪声点（判定为「稀少身份」，调用方负责移入 _trash）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import _faceid_extract as X
from . import _faceid_fast as F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--eps", type=float, default=0.314)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-images", type=int, default=0, help="调试用：只处理前 N 张")
    a = ap.parse_args(argv)

    root = Path(a.dir)
    files = sorted(p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if a.max_images > 0:
        files = files[:a.max_images]

    # 优先直接 TRT（归一化在 GPU 上，主线程几乎无 Python 开销）；失败则退回 onnxruntime
    t0 = time.time()
    provider = "unknown"
    try:
        from ._faceid_trt import TrtEmbedder
        _emb = TrtEmbedder(str(X._model_path()), batch=a.batch, fp16=True)
        names, embs, failed = X.extract_trt_pipelined(files, _emb, batch=a.batch)
        provider = "TensorRT(direct)"
    except Exception as _e:
        import sys as _s
        _s.stderr.write("[faceid] 直接 TRT 不可用，回退 onnxruntime: %s\n" % _e)
        _sess = X.make_session()
        names, embs, failed = X.extract_pipelined(files, _sess, batch=a.batch)
        provider = _sess.get_providers()[0]
    t1 = time.time()

    out = {"total": len(files), "provider": provider,
           "ms_extract": round((t1 - t0) * 1000), "clusters": {}, "noise": 0, "rare": []}
    if embs.shape[0]:
        labels = F.cluster(embs, eps=a.eps, min_samples=a.min_samples)
        stats = F.group_stats(labels)
        out["clusters"] = {str(k): int(v) for k, v in stats.items()}
        out["noise"] = int((labels < 0).sum())
        rare = []
        for i, lab in enumerate(labels.tolist()):
            if lab < 0 or stats.get(lab, 0) < a.min_samples:
                rare.append(names[i])
        out["rare"] = rare
        out["ms_cluster"] = round((time.time() - t1) * 1000)
    out["failed"] = failed
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
