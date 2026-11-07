"""人脸身份过滤的 CLI（必须跑在 engine 环境：需要 onnxruntime + TRT）。

两种用法：

  1) 一次性（脚本 / 调试）：
        python -m server._faceid_cli --dir <aligned 目录> --eps 0.40 --min-samples 2
     输出单行 JSON：
        {"total":670, "clusters":{"0":669}, "noise":1, "ms_extract":321,
         "ms_engine_load":1900, "provider":"TensorRT(direct)", ...}

  2) 常驻（4.1 后端就用这条）：python -m server._faceid_cli --serve
     stdin 一行一条请求，stdout 一行一条回应：
        {"cmd":"warm"}                                  -> @@W {"ms":.., "provider":..}
        {"dir":.., "eps":.., "min_samples":.., "merge":.., "progress":..} -> @@R {结果}
        {"cmd":"quit"}                                  -> 退出
     **引擎只在第一条请求时加载一次，之后复用。** 以前后端每次点击都
     subprocess.run 一个新进程：670 张的活儿本身 0.3~0.5 s，而引擎初始化
     （import torch/tensorrt + 反序列化 TRT 引擎 + 建 ORT 会话）要 3.5~4.5 s，
     「初始化加载模型很慢」指的就是这一段，而且它是**每点一次付一次**。
"""
from __future__ import annotations
import numpy as np

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import _faceid_extract as X
from . import _faceid_fast as F

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
        from ._faceid_trt import TrtEmbedder
        obj = TrtEmbedder(str(X._model_path()), batch=batch, fp16=True, allow_build=allow_build)
        provider = "TensorRT(direct)"
        rep = getattr(obj, "report", None) or {}
        rebuilt = rep.get("rebuilt")
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
        sys.stderr.write("[faceid] 直接 TRT 不可用，回退 onnxruntime: %s\n" % err)
        try:
            obj = X.make_session()
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
        return X.extract_trt_pipelined(files, obj, batch=batch, progress=progress)
    return X.extract_pipelined(files, obj, batch=batch, progress=progress)

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
        labels = F.cluster(embs, eps=eps, min_samples=min_samples)
        labels = _merge_by_centroid(labels, embs, float(merge))
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