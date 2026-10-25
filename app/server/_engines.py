"""TensorRT 引擎的编译与查找（BF16）。

命名规则（用户定的）：引擎文件名里带**原 ONNX 的 MD5** 和 **SM 版本**，
所以用户换了模型文件，软件一眼就能看出"引擎和 onnx 对不上"。

    <模型名>_<md5 前 12 位>_sm<SM>.bf16.engine
    例：xseglite_3f2a91c4d0e5_sm1200.bf16.engine

SM 编号按"主版本*100 + 次版本*10"：
    30 系 = 8.6  → sm860
    40 系 = 8.9  → sm890
    50 系 = 12.0 → sm1200

为什么不只靠 MD5：同一个 onnx 在不同架构上编译出来的引擎二进制不通用，
换显卡后必须重编。所以两个都对上才算"已编译"。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# 需要编译的模型（路径相对引擎仓的 modelhub / workspace）
MODEL_SPECS = [
    # name, 相对路径候选, 是否每次校验 MD5
    ("xseglite", ["workspace/model/XSegLite/xseglite.onnx", "workspace/model/XSegLite/XSegLite.onnx"], True),
    ("xseg",     ["workspace/model/XSeg/XSeg.onnx"], True),
    ("tinymog",  ["modelhub/onnx/TinyMog/TinyMog.onnx"], False),
    ("retinaface_10g", ["modelhub/onnx/RetinaFace/det_10g.onnx"], False),
    ("insightface_2d106", ["modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx"], False),
]

# 动态维（batch）的模型必须给优化档位，否则 TRT 拒绝构建：
#   "Network has dynamic or shape inputs, but no optimization profile has been defined"
# 值是 (min, opt, max)，维度顺序与 onnx 输入一致。
# 档位越少编得越快：TensorRT 要为每个档位各搜一遍 tactic。
#   · 遮罩永远单张跑 → 单档位（min=opt=max），编译 ~10 s
#   · TinyMog 会按 batch 4 批量跑 → 需要到 4，编译 ~23 s
PROFILES = {
    "tinymog": [(1, 3, 640, 640), (4, 3, 640, 640), (4, 3, 640, 640)],
    "xseglite": [(1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256)],
    "xseg": [(1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256)],
    # 检测/关键点模型也是动态输入（batch 或 H/W），我们永远只喂固定尺寸 → 单档位
    "retinaface_10g": [(1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640)],
    "insightface_2d106": [(1, 3, 192, 192), (1, 3, 192, 192), (1, 3, 192, 192)],
}

_MD5_CACHE = {}


def md5_of(path, limit=None):
    """文件 MD5（前 12 位）。同一路径只算一次，进程内缓存。"""
    key = (str(path), limit)
    if key in _MD5_CACHE:
        return _MD5_CACHE[key]
    h = hashlib.md5()
    with open(str(path), "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    v = h.hexdigest()[:12]
    _MD5_CACHE[key] = v
    return v


def sm_tag():
    """当前 GPU 的 SM 编号（主*100 + 次*10）；拿不到返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return int(major) * 100 + int(minor) * 10
    except Exception:
        return None


def engine_dir(workspace):
    d = Path(workspace) / "_engines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def trt_version():
    """TensorRT 版本串（如 101303）。引擎二进制不跨大版本通用，所以也进文件名。"""
    try:
        import tensorrt as trt
        return trt.__version__.replace(".", "")[:6]
    except Exception:
        return "0"


def engine_path(workspace, name, onnx_path, sm=None):
    sm = sm or sm_tag()
    if sm is None:
        return None
    return engine_dir(workspace) / ("%s_%s_sm%d_trt%s.bf16.engine"
                                    % (name, md5_of(onnx_path), sm, trt_version()))


def find_models(workspace, roots):
    """在若干根目录里找 MODEL_SPECS 里列的 onnx，返回 [{name, onnx, md5, ...}]。"""
    out = []
    for name, rels, verify in MODEL_SPECS:
        found = None
        for root in roots:
            for rel in rels:
                p = Path(root) / rel
                if p.is_file():
                    found = p
                    break
            if found:
                break
        if found is None:
            out.append({"name": name, "onnx": None, "verify": verify})
            continue
        out.append({"name": name, "onnx": str(found), "verify": verify})
    return out


def status(workspace, roots):
    """给前端的状态表：每个模型的 onnx 是否存在、引擎是否就绪/过期。"""
    sm = sm_tag()
    rows = []
    for item in find_models(workspace, roots):
        row = {"name": item["name"], "onnx": item["onnx"], "verify": item["verify"],
               "sm": sm, "md5": None, "engine": None, "state": "no-model", "size_mb": None}
        if item["onnx"] is None:
            rows.append(row)
            continue
        p = Path(item["onnx"])
        row["md5"] = md5_of(p)
        row["size_mb"] = round(p.stat().st_size / (1024 * 1024), 1)
        if sm is None:
            row["state"] = "no-gpu"
            rows.append(row)
            continue
        eng = engine_path(workspace, item["name"], p, sm)
        row["engine"] = str(eng)
        if eng.is_file():
            row["engine_mb"] = round(eng.stat().st_size / (1024 * 1024), 1)
            row["state"] = "ready"
        else:
            # 同名的旧引擎（别的 md5 / 别的 sm）存在 → 说明是"过期"而不是"从没编过"
            stale = list(engine_dir(workspace).glob("%s_*.engine" % item["name"]))
            row["state"] = "stale" if stale else "missing"
        rows.append(row)
    return {"sm": sm, "dir": str(engine_dir(workspace)), "models": rows}


# 每次使用前都必须核对 MD5 的模型（用户口径：遮罩模型安全第一）
VERIFY_ALWAYS = {"xseglite", "xseg"}
_FRESH = {}          # (name, path) -> (md5, 时间戳)
_FRESH_TTL = 10.0    # 10 秒内重复调用不再重算，避免每张图都读 18 MB


def md5_fresh(name, path, ttl=_FRESH_TTL):
    """按策略取 MD5：遮罩类模型每次校验（10 秒内复用），其它用进程内缓存。"""
    key = (name, str(path))
    now = __import__("time").time()
    if name in VERIFY_ALWAYS:
        hit = _FRESH.get(key)
        if hit and (now - hit[1]) < ttl:
            return hit[0]
        v = md5_of(path, limit=None) if False else _md5_direct(path)
        _FRESH[key] = (v, now)
        return v
    return md5_of(path)


def _md5_direct(path):
    """不走缓存的 MD5（校验策略用）。"""
    h = hashlib.md5()
    with open(str(path), "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def try_engine(workspace, name, onnx_path, sm=None):
    """有匹配的引擎（MD5 + SM 都对）就返回 TrtSession，否则 None。

    这是运行时的唯一入口：拿到 None 就继续走 onnxruntime，行为完全不变。
    """
    try:
        sm = sm or sm_tag()
        if sm is None:
            return None
        md5 = md5_fresh(name, onnx_path)
        p = engine_dir(workspace) / ("%s_%s_sm%d_trt%s.bf16.engine"
                                     % (name, md5, sm, trt_version()))
        if not p.is_file():
            return None
        from ._trt_session import TrtSession
        return TrtSession(str(p))
    except Exception:
        return None
