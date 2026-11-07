# -*- coding: utf-8 -*-
"""遮罩编辑 / 工程（从 app.py 按路径前缀抽出）。"""
from __future__ import annotations

import asyncio
import atexit
import base64
import functools
import inspect
import io
import json
import os
import pickle
import queue
import re
import subprocess as _sp
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .app import (IMAGE_EXT, WORKSPACE, _mask_ai_stub, _mask_state)

router = APIRouter()


# ============================================================================
# 错误码契约（所有 /api/mask/* 的失败响应统一长这样）
#
#   {"ok": false, "code": "<code>", "detail": "...", "missing": [...]?}
#
#   missing_weights  501  缺权重 / 缺依赖      → 前端把工具置灰
#   image_missing    404  图片不存在           → 只提示一次，不降级
#   session_expired  409  会话过期 / 被 LRU 淘汰 → 前端重新 load 后重试一次
#   bad_args         400  参数不对
#   infer_failed     500/507/504  其余（含 OOM 507、超时 504）
#
# 路由里 raise MaskErr(...)，由 _mask_api 统一翻成上面的 JSON；漏网的
# HTTPException 也按状态码补一个 code，保证前端永远能按 code 分支。
# ============================================================================
_MASK_STATUS = {
    "missing_weights": 501,
    "image_missing": 404,
    "session_expired": 409,
    "bad_args": 400,
    "infer_failed": 500,
}

# 已存在的 HTTPException 兜底：状态码 → code
_HTTP_CODE = {400: "bad_args", 404: "image_missing", 409: "session_expired",
              501: "missing_weights", 500: "infer_failed", 503: "infer_failed",
              504: "infer_failed", 507: "infer_failed"}


class MaskErr(Exception):
    """带契约 code 的路由错误。status 给了就用它（如 OOM 507 / 超时 504）。"""

    def __init__(self, code, detail, status=None, missing=None):
        Exception.__init__(self, detail)
        self.code = str(code)
        self.detail = str(detail)
        self.status = int(status or _MASK_STATUS.get(self.code) or 500)
        self.missing = list(missing or [])


def _mask_error_body(code, detail, missing=None):
    body = {"ok": False, "code": str(code), "detail": str(detail)}
    if missing:
        body["missing"] = [str(m) for m in missing]   # 缺哪个文件，前端提示里能直接指路
    return body


def _mask_api(fn):
    """统一出参：MaskErr / HTTPException → {"ok":false,"code":...} + 对应状态码。"""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            # async 路由照旧；同步路由（save / 那几个 stub）仍丢线程池，
            # 别把写 PNG 这种事搬回事件循环上。
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)
        except MaskErr as err:
            return JSONResponse(status_code=err.status,
                                content=_mask_error_body(err.code, err.detail, err.missing))
        except HTTPException as ex:
            code = _HTTP_CODE.get(int(ex.status_code or 500), "infer_failed")
            return JSONResponse(status_code=ex.status_code,
                                content=_mask_error_body(code, ex.detail))

    return wrapper


# ============================================================================
# 遮罩落盘 / 读回：对齐「引擎真正读的那一处」
#
# 引擎里没有任何代码读 <aligned>/_masks/ —— 全仓只有本文件和 docs 提到这个名字。
# 训练读遮罩只有一条路：aligned 的 DFLJPG 里 APP15 pickle 段的 'xseg_mask'
#
#   engine/DFLIMG/DFLJPG.py:336  has_xseg_mask()  → dfl_dict.get('xseg_mask')
#   engine/DFLIMG/DFLJPG.py:346  get_xseg_mask()  → cv2.imdecode(...) / 255 → H×W×1 float
#   engine/DFLIMG/DFLJPG.py:360  set_xseg_mask()  → 单通道 uint8 存 PNG（>50000 字节退 JPEG）
#   engine/samplelib/SampleLoader.py:72             只收 .jpg/.jpeg —— 散落的 PNG 直接被排除
#   engine/samplelib/SampleProcessor.py:62          mask 尺寸不对就 INTER_CUBIC 拉回图尺寸
#   engine/samplelib/SampleGeneratorFaceXSeg.py:105 XSeg / XSegLite 训练数据同样走 DFLJPG
#
# 所以「引擎期望的约定」只有一条：mask 必须是单通道、0-255、尺寸 = aligned 图尺寸，
# 存在 <aligned>/<同名>.jpg 的 APP15['xseg_mask'] 里。合成阶段不用遮罩。
#
# 本模块因此双写：
#   1) <aligned>/_masks/<同名>.png —— 灰度 L / 0-255 / 尺寸同原图。人可看的 staging，
#      也是读回时的兜底（引擎不读它，所以不能只写这个）。
#   2) 同一张 mask 经 _align.encode_xseg_mask（= DFLJPG.set_xseg_mask 的同款编码）
#      写回 <aligned>/<同名>.jpg **头部** APP15['xseg_mask'] —— 训练真正吃到的就是它。
#      头部这几个字要紧：数据集的每张图都有两份 APP15（头部一份没遮罩的、EOI 前一份
#      带遮罩的，后者是 _align.rewrite_dfljpg_meta 写的），DFLJPG.load_raw 走到 SOS
#      就跳到 EOI，所以 EOI 前那份引擎永远读不到 —— 那 1203 张带遮罩的图其实一直是
#      空遮罩在训练。这里保存时把 legacy 那份的键并进头部、遮罩写头部、死重删掉。
# ============================================================================
_MASK_SUBDIR = "_masks"


def _mask_out_dir(base) -> Path:
    return Path(base) / _MASK_SUBDIR


def _mask_png_path(base, name) -> Path:
    """<aligned>/_masks/<同名>.png —— 跟 aligned 图一一对应，别换名字。"""
    return _mask_out_dir(base) / (Path(str(name)).stem + ".png")


def _image_wh(p: Path):
    """原图尺寸（读图头，不整解码）。遮罩必须按这个尺寸存，不能写死 1024。"""
    try:
        import cv2
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return int(im.shape[1]), int(im.shape[0])
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(str(p)) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    return 0, 0


def _decode_mask(raw: bytes):
    """PNG/JPEG 字节 → uint8 H×W 单通道 0-255。

    前端 canvas 出来的是 RGBA（A 恒 255），这里按灰度收掉 alpha：
    带透明度的遮罩按「透明 = 不在遮罩里」处理，别把 RGBA 原样落盘 ——
    DFLJPG.get_xseg_mask 解到 4 通道不会自己变单通道。
    """
    import numpy as np
    import cv2
    if not raw:
        return None
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) * alpha
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def _fit_mask(mask, w: int, h: int):
    """尺寸对不上就按原图尺寸拉回去（NEAREST：遮罩是 0/255，别插值出灰边）。"""
    import cv2
    if mask is None or w <= 0 or h <= 0:
        return mask
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)


def _write_mask_png(path: Path, mask) -> int:
    """单通道 uint8 → PNG（灰度，PIL 读出来是 mode L，不是 RGBA）。"""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise MaskErr("infer_failed", "遮罩 PNG 编码失败（%dx%d）" % (mask.shape[1], mask.shape[0]))
    path.write_bytes(buf.tobytes())
    return int(buf.size)


# APP15 里是 pickle（DFL 的元数据段）。只放行数据类对象 —— 数据集混进外部图片时
# 直接 loads 等于让一张 jpg 执行任意代码。比 _align._ALLOWED 多放行一条 _codecs.encode：
# pickle protocol≤2 把 bytes（xseg_mask 就是 bytes）存成 _codecs.encode(str,'latin1')，
# _align 那份白名单没这条，所以带遮罩的文件用 _align.read_dfljpg_meta 一律解不开。
_APP15_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("_codecs", "encode"),
}


class _App15Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _APP15_ALLOWED:
            return pickle.Unpickler.find_class(self, module, name)
        raise pickle.UnpicklingError("blocked global: %s.%s" % (module, name))


def _unpickle_dict(payload: bytes):
    """APP15 的 payload → dict；不是 pickle / 解不开就 None。"""
    if payload[:1] != b"\x80":                 # pickle protocol 2 的开头
        return None
    try:
        obj = _App15Unpickler(io.BytesIO(payload)).load()
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _find_app15_header(data: bytes):
    """头部（SOS 之前）那份 APP15 —— 引擎 DFLJPG.load 看到的只有这一份。"""
    import struct
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:      # 无长度字段的标记
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if m == 0xEF:
            return _unpickle_dict(data[i + 4:i + 2 + seg_len])
        if m == 0xDA:                                   # SOS：APP15 只会在此之前
            return None
        i += 2 + seg_len
    return None


def _find_app15_legacy(data: bytes):
    """挂在 EOI 之前的那份 APP15 —— 旧写法（_align.rewrite_dfljpg_meta）留下的。

    实测整个 data_src/aligned 都是这种双份布局：头部一份没有遮罩的、EOI 前一份
    带 xseg_mask 的。DFLJPG.load_raw 遇到 SOS 直接跳到 EOI，所以带遮罩的那份
    引擎永远看不到 —— 训练一直用的是「没有遮罩」的头部那份。这里把它读出来，
    既用于读回显示，也在下次保存时顺手把遮罩迁到头部（并删掉这份死重）。
    """
    import struct
    eoi = data.rfind(b"\xff\xd9")
    if eoi < 0:
        return None
    pos = data.rfind(b"\xff\xef", 0, eoi)
    while pos > 0:
        if pos + 4 <= eoi:
            seg_len = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            if pos + 2 + seg_len == eoi:            # 正好顶到 EOI 才算数
                obj = _unpickle_dict(data[pos + 4:pos + 2 + seg_len])
                if obj is not None:
                    return obj
        pos = data.rfind(b"\xff\xef", 0, pos)
    return None


def _read_app15_metas(img_path: Path):
    """→ (头部 dict | None, EOI 前那份 legacy dict | None)。"""
    try:
        data = Path(img_path).read_bytes()
    except OSError:
        return None, None
    if not data.startswith(b"\xff\xd8"):
        return None, None
    return _find_app15_header(data), _find_app15_legacy(data)


def _write_app15(img_path: Path, meta: dict) -> int:
    """把 meta 写成头部 APP15（替换掉旧的），返回新文件字节数。

    实现已收敛到 _align.write_app15_head —— 这里原来另有一份，而 _align 那份把
    APP15 追加在 EOI 之前，DFLJPG.load_raw 遇到 SOS 就直接跳到 EOI 划压缩数据，
    压根到不了那儿，引擎侧 has_xseg_mask 永远是 False。现在只有一份实现：
    写在头部、保留原有 meta 键、幂等（连写 3 次字节数不变），并顺手摘掉 EOI 前
    那份 legacy APP15。这里只做「读 → 重写 → 落盘」这一层。
    """
    from ._align import write_app15_head

    data = Path(img_path).read_bytes()
    out = write_app15_head(data, meta)
    Path(img_path).write_bytes(out)
    return len(out)


def _embed_mask_into_dfljpg(img_path: Path, mask):
    """把 mask 写进 aligned DFLJPG 头部 APP15 的 'xseg_mask'，返回 (ok, note)。

    note 给前端看：不是 DFLJPG（普通 jpg / png）时照旧只有 _masks 下的 PNG，
    训练不会读到 —— 那就要先把这套图对齐成 DFLJPG。
    """
    if Path(img_path).suffix.lower() not in (".jpg", ".jpeg"):
        return False, "不是 JPEG：只在 _masks 落盘，引擎读不到"
    try:
        import numpy as np
        from ._align import encode_xseg_mask
    except Exception as ex:                     # 缺 numpy/cv2：PNG 侧写仍然成立
        return False, "载入 _align 失败：%s" % (ex,)
    try:
        head, legacy = _read_app15_metas(img_path)
        if head is None and legacy is None:
            return False, "不是 DFLJPG（没有 APP15）：只在 _masks 落盘，训练读不到"
        # 头部是引擎读的那份，以它为准；旧的那份只补头部缺的键，
        # 别把旧遮罩（xseg_mask）盖回来 —— 下面按本次画的重新写。
        meta = dict(head or {})
        for k, v in (legacy or {}).items():
            meta.setdefault(k, v)
        enc = encode_xseg_mask(mask.astype(np.float32) / 255.0)
        if not enc:
            return False, "xseg_mask 编码失败（>50000 字节且 JPEG 也压不下来）"
        meta["xseg_mask"] = enc
        _write_app15(img_path, meta)
        return True, "%d 字节 mask 写入头部 APP15['xseg_mask']" % len(enc)
    except Exception as ex:
        return False, "写 DFLJPG 失败：%s: %s" % (type(ex).__name__, ex)


def _dfljpg_has_mask(img_path: Path) -> bool:
    if Path(img_path).suffix.lower() not in (".jpg", ".jpeg"):
        return False
    head, legacy = _read_app15_metas(img_path)
    return any((m or {}).get("xseg_mask") is not None for m in (head, legacy))


def _has_mask(img_path: Path) -> bool:
    return _mask_png_path(img_path.parent, img_path.name).is_file() or _dfljpg_has_mask(img_path)


def _load_mask_b64(img_path: Path):
    """已有遮罩 → (base64 PNG 或 None, w, h, source)。

    优先级：头部 APP15（引擎读的那份）> EOI 前那份 legacy APP15（老 apply 步骤
    写的，引擎看不到但用户画过的遮罩在里面）> _masks/<同名>.png（本编辑器落的盘）。
    统一重编码成 PNG 再回前端 —— APP15 里的字节可能是 JPEG（>50KB 的降级），
    而前端 loadExistingMask 拼的是 data:image/png。
    """
    import cv2
    mask, src = None, ""
    head, legacy = _read_app15_metas(img_path)
    for meta, tag in ((head, "dfljpg"), (legacy, "dfljpg-legacy")):
        if mask is None and (meta or {}).get("xseg_mask") is not None:
            mask = _decode_mask(bytes(meta.get("xseg_mask") or b""))
            if mask is not None:
                src = tag
    if mask is None:
        png = _mask_png_path(img_path.parent, img_path.name)
        if png.is_file():
            try:
                mask = _decode_mask(png.read_bytes())
                src = "_masks"
            except OSError:
                mask = None
    if mask is None:
        return None, 0, 0, ""
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        return None, 0, 0, ""
    return base64.b64encode(buf.tobytes()).decode("ascii"), int(mask.shape[1]), int(mask.shape[0]), src


@router.post("/api/project/open")
def mask_project_open(body: dict):
    """打开一个已对齐的人脸集目录，列出图片供遮罩编辑器使用。"""
    raw = (body or {}).get("path") or ""
    rel = raw
    low = rel.lower().replace("/", "\\")
    for pre in ("workspace\\", "workspace"):
        if low.startswith(pre):
            rel = rel[len(pre):].lstrip("\\/")
            break
    d = Path(raw) if (raw and Path(raw).is_absolute()) else (WORKSPACE / rel if rel else WORKSPACE / "data_src" / "aligned")
    if not d.is_dir():
        raise HTTPException(404, "目录不存在: " + str(d))
    files = sorted([p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT])
    resume = 0
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            memo = json.loads(fp.read_text(encoding="utf-8"))
            if memo.get("dir") and Path(memo["dir"]) == d:
                resume = int(memo.get("index") or 0)
        except Exception:
            pass
    _mask_state.update({"dir": str(d), "files": [p.name for p in files], "index": resume})
    return {"path": str(d), "count": len(files), "resume": resume,
            # has_mask 只看 _masks 下有没有同名 PNG：500 张逐个解 APP15 太贵，
            # 而手绘存一次必然留一张 PNG，够前端打勾了。
            "files": [{"name": p.name, "size": p.stat().st_size,
                       "has_mask": _mask_png_path(d, p.name).is_file()} for p in files[:500]]}


@router.get("/api/image/{index}")
def mask_image(index: int):
    """返回当前项目的第 index 张图（base64），以及是否已有遮罩。"""
    import base64
    files = _mask_state.get("files") or []
    if not files:
        raise HTTPException(409, "还没有打开项目")
    index = max(0, min(int(index), len(files) - 1))
    d = Path(_mask_state["dir"])
    p = d / files[index]
    if not p.is_file():
        raise HTTPException(404, "文件不存在: " + files[index])
    raw = p.read_bytes()
    ext = p.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    _mask_state["index"] = index
    return {"index": index, "name": files[index], "total": len(files),
            "image": base64.b64encode(raw).decode(),   # 纯 base64，前端自己拼 data URL
            "mime": mime, "has_mask": _has_mask(p)}    # 以前恒 True，等于没信息


@router.post("/api/mask/predict")
@_mask_api
def mask_predict(body: dict):
    _mask_ai_stub()


@router.post("/api/mask/text")
@_mask_api
def mask_text(body: dict):
    _mask_ai_stub()


@router.post("/api/mask/bisenet")
@_mask_api
def mask_bisenet(body: dict):
    _mask_ai_stub()


@router.post("/api/model/preload")
def mask_preload(body: dict | None = None):
    return {"ok": True, "note": "模型预加载未接"}


@router.get("/api/image/{index}/mask")
def mask_existing(index: int):
    """读这张图已有的遮罩（base64 PNG + 尺寸 + 来源），没有就 {"mask": null}。

    读的顺序 = 引擎的读法：先 DFLJPG 的 APP15['xseg_mask']（DFLJPG.get_xseg_mask
    吃的那份），没有再退 _masks/<同名>.png（手绘落盘的 staging）。
    没有项目时跟以前一样回 null，不抛 409 —— 前端 loadExistingMask 的 catch 只负责收尾。
    """
    files = _mask_state.get("files") or []
    if not files:
        return {"mask": None}
    index = max(0, min(int(index), len(files) - 1))
    p = Path(_mask_state["dir"]) / files[index]
    if not p.is_file():
        raise HTTPException(404, "文件不存在: " + files[index])
    b64, w, h, src = _load_mask_b64(p)
    if not b64:
        return {"mask": None}
    return {"mask": b64, "w": w, "h": h, "source": src}


@router.post("/api/progress/save")
def mask_progress_save(body: dict | None = None):
    """记住看到第几张，下次进来接着干。落盘，重启也还在。"""
    body = body or {}
    idx = int(body.get("index") or 0)
    _mask_state["index"] = idx
    try:
        (WORKSPACE / "_mask_progress.json").write_text(
            json.dumps({"index": idx, "dir": _mask_state.get("dir") or ""}, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass
    return {"ok": True, "index": idx}


@router.get("/api/progress/load")
def mask_progress_load():
    """读上次的进度；文件里还有上次打开的目录，前端可以据此恢复。"""
    idx = _mask_state.get("index") or 0
    d = ""
    fp = WORKSPACE / "_mask_progress.json"
    if fp.is_file():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            idx = int(data.get("index") or idx)
            d = data.get("dir") or ""
        except Exception:
            pass
    return {"index": idx, "dir": d}


@router.post("/api/mask/save")
@_mask_api
def mask_save(body: dict):
    """把画好的遮罩按引擎的约定落盘（双写，见上面「遮罩落盘 / 读回」那一节）：

      1) <aligned>/_masks/<同名>.png  —— 灰度 L / 0-255 / 尺寸 = 原图。staging + 读回兜底。
      2) <aligned>/<同名>.jpg 的 APP15['xseg_mask'] —— DFLJPG.get_xseg_mask 吃的那份，
         也就是 SAEHD 训练 / XSeg 训练真正读到的遮罩。

    以前是 raw = base64 直接 write_bytes：前端 _maskToB64 交上来的是 maskResolution
    方图（默认 1024）的 RGBA，于是 432 的 aligned 图会存出 1024×1024 RGBA 的 PNG ——
    尺寸不对、还有 alpha，引擎那边一个都对不上。现在一律解成单通道、按原图尺寸
    NEAREST 拉正，再编码。

    body 要带 image_index（前端 api.js: post('/mask/save',{image_index,mask})）。
    以前这里读 body["index"]，前端发的是 image_index —— 读不到就退回全局
    _mask_state["index"]，多标签页开着的时候会把 A 的遮罩写进 B 的文件。
    现在缺了就 400，绝不猜。"""
    import base64
    body = body or {}
    files = _mask_state.get("files") or []
    if not files:
        raise MaskErr("session_expired", "还没有打开项目 —— 先 POST /api/project/open 再存")
    raw_idx = body.get("image_index", body.get("index"))
    if raw_idx is None or (isinstance(raw_idx, str) and not raw_idx.strip()):
        raise MaskErr("bad_args", "缺 image_index —— 要写第几张必须由前端指定，"
                                  "不退回全局进度（多标签下会写错文件）")
    try:
        idx = int(raw_idx)
    except (TypeError, ValueError):
        raise MaskErr("bad_args", "image_index 要是整数，收到 %r" % (raw_idx,))
    if idx < 0 or idx >= len(files):
        raise MaskErr("bad_args", "image_index 越界：%d（总共 %d 张）" % (idx, len(files)))
    data = body.get("mask") or ""
    if not isinstance(data, str) or not data:
        raise MaskErr("bad_args", "mask 不能为空（base64 或 data URL）")
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception:
        raise MaskErr("bad_args", "遮罩数据不是合法 base64")
    base = Path(_mask_state["dir"])
    img_path = base / files[idx]
    if not img_path.is_file():
        raise MaskErr("image_missing", "第 %d 张的原图不见了：%s" % (idx, files[idx]))

    mask = _decode_mask(raw)
    if mask is None:
        raise MaskErr("bad_args", "遮罩不是能解码的 PNG/JPEG")
    w, h = _image_wh(img_path)
    if w <= 0 or h <= 0:
        raise MaskErr("image_missing", "读不出原图尺寸：%s" % files[idx])
    mask = _fit_mask(mask, w, h)

    out = _mask_png_path(base, files[idx])
    nbytes = _write_mask_png(out, mask)

    embedded, note = _embed_mask_into_dfljpg(img_path, mask)
    return {"ok": True, "path": str(out), "image_index": idx,
            "w": w, "h": h, "mode": "L", "bytes": nbytes,
            "embedded": bool(embedded), "xseg_mask": "DFLJPG.APP15" if embedded else None,
            "note": note}


@router.get("/api/mask/models")
def mask_models(kind: str = "XSegLite"):
    """扫描 workspace/model/<kind>/ 下的 .onnx / .engine，供遮罩绘制页的模型下拉用。
    4.0 的 _refresh_xseg_file_list() 就是这个逻辑：优先 .engine（TRT 最快）。"""
    base = WORKSPACE / "model" / (kind if kind in ("XSeg", "XSegLite") else "XSegLite")
    files = []
    if base.is_dir():
        for f in sorted(base.iterdir()):
            if f.is_file() and f.suffix.lower() in (".onnx", ".engine"):
                files.append({"name": f.name, "path": str(f),
                              "mb": round(f.stat().st_size / 1048576, 1),
                              "trt": f.suffix.lower() == ".engine"})
    files.sort(key=lambda x: (not x["trt"], x["name"]))
    return {"kind": kind, "dir": str(base), "files": files}


# ============================================================================
# 遮罩绘制 · SAM2 点击分割 / GroundedSAM2 文本检测
#
#   POST /api/mask/sam/load     {"image":..., "variant":"base_plus"}       → 会话 + embedding
#   POST /api/mask/sam/predict  {"session":..., "points":[{x,y,label}], "box":[x1,y1,x2,y2]}
#   POST /api/mask/gdino        {"image":..., "text":"face."}              → 框
#
# 这里的形状是三件事逼出来的：
#
#   1) 模型在**另一个进程**里。SAM2 建模要走 hydra（只装在 runtime/engine），
#      GroundingDINO 还要 torchvision / transformers；后端这份 runtime/server
#      只有 torch。所以推理统一交给 runtime/engine/python.exe，走一行一个 JSON 的
#      stdin/stdout 协议（源码见 _WORKER_SRC，落盘到 %TEMP%\dfl_mask_worker\）。
#   2) 模型只加载一次：那个进程里全局一份 SAM2（换 variant 才重建，同时释放旧的显存）
#      + 一份 GDINO。embedding 按 (image, variant) 存，最多 3 份 LRU，
#      空闲 10 分钟释放（连模型一起放，训练/合成要抢显存时它得让位）。
#   3) 不阻塞事件循环、不并发抢显存：路由是 async，推理丢进 asyncio.to_thread；
#      进程那一侧再用一把 threading.Lock 把所有推理串成一条线。单次 predict 30 s
#      没回来就 kill 推理进程回 504 —— 推理没有中间态，「算完」或「扔掉」，
#      下一次调用会自动重启并重新加载模型。显存不足回 507，进程不崩。
#
# 坐标一律按**原图像素**收发；SAM2 的 predictor 内部归一化到模型输入尺寸、出图时
# 再映射回 _orig_hw，所以返回的 mask 尺寸等于原图（这里仍兜底检查一遍，不对就
# NEAREST 拉回原尺寸）。embedding 命中时把 predictor 的 _features 换回去即可，
# 下面 _restore 就是干这个的（和重新 set_image 出来的 mask 逐像素一致）。
# ============================================================================

_SAM_VARIANTS = ("base_plus", "large", "small", "tiny")   # 变体少而固定，写死
_SAM_DEFAULT = "base_plus"                                 # 1.5 G 上下，画遮罩够用
_SAM_PREDICT_TIMEOUT = 30.0                                # 单次推理上限（超了 kill）
_SAM_LOAD_TIMEOUT = 180.0                                  # 首次 load 含 torch 冷启动
_GDINO_TIMEOUT = 180.0                                     # 首次要 import transformers
_CACHE_MAX = 3                                             # embedding 最多留 3 张图
_IDLE_SEC = 600                                            # 空闲 10 分钟释放

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0   # GUI 应用：子进程不弹黑框


# ---------------------------------------------------------------- 路径
def _root_dir() -> Path:
    """仓库根（workspace 的上一层）。"""
    return Path(__file__).resolve().parents[2]


def _engine_dir() -> Path:
    """app/engine（xlib / vendor 都在这里）。"""
    return Path(__file__).resolve().parents[1] / "engine"


def _weights_root() -> Path:
    try:
        from ._paths import weights_dir
        return Path(weights_dir())
    except Exception:
        return _root_dir() / "weights"


def _engine_python() -> Path:
    """推理进程用的解释器：runtime/engine/python.exe（有 hydra / torchvision / timm）。

    后端这份 runtime/server 只装了 torch，建模 import 不动。
    """
    try:
        from ._paths import engine_python
        p = Path(engine_python())
        if p.is_file():
            return p
    except Exception:
        pass
    import sys as _sys
    return Path(_sys.executable)


def _norm_variant(name) -> str:
    """'sam2.1_hiera_base_plus' / 'base-plus' / '' → base_plus。"""
    v = str(name or "").strip().lower().replace("-", "_")
    for pre in ("sam2.1_hiera_", "sam2_hiera_", "sam2_", "sam_"):
        if v.startswith(pre):
            v = v[len(pre):]
            break
    if v in ("", "default", "auto"):
        return _SAM_DEFAULT
    if v not in _SAM_VARIANTS:
        raise MaskErr("bad_args", "variant 只认 %s，收到 %r"
                      % (" / ".join(_SAM_VARIANTS), name))
    return v


def _sam_ckpt(variant: str) -> Path:
    # 平铺：weights/sam2.1_hiera_<variant>.pt（没有 sam2/checkpoints/ 这一层）
    return _weights_root() / ("sam2.1_hiera_%s.pt" % variant)


def _resolve_image(raw) -> Path:
    """image 参数：绝对路径，或相对 workspace / 仓库根（/ 或 \\ 都行）。"""
    s = str(raw or "").strip().strip('"').strip("'")
    if not s:
        raise MaskErr("bad_args", "image 不能为空（绝对路径，或相对 workspace 的路径）")
    p = Path(s).expanduser()
    if p.is_absolute():
        cands = [p]
    else:
        cands = [WORKSPACE / s, _root_dir() / s, Path(s)]
    for c in cands:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    raise MaskErr("image_missing", "图片不存在：%s" % s)


# ---------------------------------------------------------------- 推理进程源码
_WORKER_SRC = r'''# -*- coding: utf-8 -*-
"""SAM2 / GroundingDINO 推理进程（由 app/server/routes_mask.py 拉起，别直接跑）。

协议：stdin/stdout 一行一个 JSON。
  请求  {"id":n,"op":"load|predict|gdino|drop|stats|ping",...}
  响应  {"id":n,"ok":true,...} 或 {"id":n,"ok":false,"kind":"...","message":"..."}
kind: oom 显存不足 / missing 权重或包缺 / session 会话过期 / image 图读不出来 /
      bad_request 参数不对 / error 其它。stderr 只打日志，不参与协议。

为什么要单独一个进程：hydra、torchvision、transformers 只装在 runtime/engine；
另外超时能直接 kill，OOM 也炸不到 web 进程。
"""
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

_lock = threading.Lock()


def _log(msg):
    try:
        sys.stderr.write("[mask-worker] %s" % (msg,))
        sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception:
        pass


ARGS = {}
for _i in range(1, len(sys.argv) - 1, 2):
    ARGS[sys.argv[_i].lstrip("-").replace("-", "_")] = sys.argv[_i + 1]

ENGINE = Path(ARGS.get("engine", ".")).resolve()
REPO = Path(ARGS.get("repo", ".")).resolve()
WEIGHTS = Path(ARGS.get("weights", str(REPO / "weights"))).resolve()
DEVICE_PREF = str(ARGS.get("device", "auto"))
CACHE_MAX = int(float(ARGS.get("cache", "3")))
IDLE_SEC = float(ARGS.get("idle", "600"))

# 离线兜底：后端 _spawn 里已经设了，这里再设一遍是为了「有人直接跑这个脚本」时
# 也不会去联网等超时。HF_HUB_OFFLINE=1 下 from_pretrained 只读本地，不碰网络。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DFL_WEIGHTS_DIR", str(WEIGHTS))

if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

VARIANTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "sam2.1/sam2.1_hiera_l.yaml"),
}


class WErr(Exception):
    """带 kind 的错误：调用方按 kind 映射 HTTP 状态码。"""

    def __init__(self, kind, message, missing=None):
        Exception.__init__(self, message)
        self.kind = kind
        self.message = message
        self.missing = list(missing or [])


_CACHE = {}


def _torch():
    if "torch" not in _CACHE:
        import torch
        _CACHE["torch"] = torch
    return _CACHE["torch"]


def _empty_cache():
    try:
        t = _CACHE.get("torch")
        if t is not None and t.cuda.is_available():
            t.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(ex):
    try:
        t = _torch()
        if isinstance(ex, t.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    s = str(ex).lower()
    return ("out of memory" in s) or ("cuda error: out of memory" in s)


def _alias_pkg(name, path):
    """把 vendor 里的一个包挂成 name（内部 import 写的是 xlib.models.sam2.*）。"""
    import importlib.util
    p = Path(path)
    init = p / "__init__.py"
    if not init.is_file():
        raise WErr("missing", "建模代码缺失：%s" % init, missing=[str(init)])
    spec = importlib.util.spec_from_file_location(name, str(init), submodule_search_locations=[str(p)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_sam2():
    if "sam2" not in _CACHE:
        import xlib          # noqa: F401  app/engine 已在 sys.path，xlib 是命名空间包
        import xlib.models   # noqa: F401
        if "xlib.models.sam2" not in sys.modules:
            _alias_pkg("xlib.models.sam2", ENGINE / "vendor" / "sam2")
        from xlib.models.sam2.build_sam import build_sam2
        from xlib.models.sam2.sam2_image_predictor import SAM2ImagePredictor
        _CACHE["sam2"] = (build_sam2, SAM2ImagePredictor)
    return _CACHE["sam2"]


def _import_gdino():
    if "gdino" not in _CACHE:
        import types
        import xlib          # noqa: F401
        import xlib.models   # noqa: F401
        if "xlib.models.grounding_dino" not in sys.modules:
            parent = types.ModuleType("xlib.models.grounding_dino")
            parent.__path__ = [str(ENGINE / "vendor")]
            sys.modules["xlib.models.grounding_dino"] = parent
        import xlib.models.grounding_dino.groundingdino.datasets.transforms as T
        from xlib.models.grounding_dino.groundingdino.util.inference import load_model, predict
        _CACHE["gdino"] = (T, load_model, predict)
    return _CACHE["gdino"]


def _png_data_url(mask_u8):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(mask_u8, mode="L").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _read_rgb(path):
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        # 图片没了 == image（HTTP 404 / image_missing），不是 missing（501）。
        # 以前这里报 missing，/api/mask/gdino 的 missing 是 501 —— 前端一看到
        # 501 就把整个 GroundedSAM2 永久置灰，删过的图能把工具锁死。
        raise WErr("image", "图片不存在：%s" % p)
    try:
        with Image.open(str(p)) as im:
            rgb = np.asarray(im.convert("RGB"))
    except Exception as ex:
        raise WErr("image", "图片解码失败 %s：%s" % (p, ex))
    return np.ascontiguousarray(rgb)


def _resize_mask(mask, w, h):
    """兜底：mask 尺寸必须等于原图。"""
    from PIL import Image
    m = np.asarray(mask)
    if m.shape[0] == h and m.shape[1] == w:
        return (m > 0.5).astype(np.uint8) * 255
    u8 = Image.fromarray(((m > 0.5).astype(np.uint8) * 255), mode="L")
    return np.asarray(u8.resize((int(w), int(h)), Image.NEAREST), dtype=np.uint8)


class SamState(object):
    def __init__(self):
        self.model = None
        self.predictor = None
        self.variant = None
        self.device = None
        self.build_ms = 0.0
        self.cache = OrderedDict()      # (path, variant) -> {snapshot,w,h,last,...}
        self.sessions = {}              # session id  -> (path, variant)
        self.idle_since = time.time()

    # -- 设备 --------------------------------------------------------------
    def devices(self):
        t = _torch()
        try:
            cuda = bool(t.cuda.is_available())
        except Exception:
            cuda = False
        if DEVICE_PREF == "cpu":
            return ["cpu"]
        if cuda:
            return ["cuda", "cpu"]      # 先 cuda，建不起来再退 cpu
        return ["cpu"]

    def free_model(self):
        """连模型带 embedding 一起放。

        只清 model/predictor 是不够的：cache 里那几份 snapshot（_features 是
        张量，直接挂在 predictor 上）还活得好好的，显存收不回；sessions 里的
        旧 sid 也还在，下一次 predict 会走 ensure_model(key[1]) 把**旧 variant**
        悄悄重建回来 —— 刚为了省显存释放的模型又回来了，而且还是错的那个。
        """
        self.model = None
        self.predictor = None
        self.variant = None
        self.device = None
        self.cache.clear()
        self.sessions.clear()
        _empty_cache()

    def ensure_model(self, variant):
        if variant not in VARIANTS:
            raise WErr("missing", "未知的 SAM2 变体：%s（可选 %s）"
                       % (variant, " / ".join(sorted(VARIANTS))))
        if self.model is not None and self.variant == variant:
            return
        ckpt_name, cfg_rel = VARIANTS[variant]
        ckpt = WEIGHTS / ckpt_name          # 平铺：weights/sam2.1_hiera_<variant>.pt
        cfg = ENGINE / "vendor" / "sam2" / "configs" / cfg_rel
        if not ckpt.is_file():
            raise WErr("missing",
                       "SAM2 权重缺失：%s —— 先在「设置 → 权重包」下载 sam 包（--group sam）" % ckpt,
                       missing=[str(ckpt)])
        if not cfg.is_file():
            raise WErr("missing", "SAM2 建模配置缺失：%s" % cfg, missing=[str(cfg)])
        build_sam2, Predictor = _import_sam2()
        self.free_model()               # 一次只留一个变体，显存不叠加
        last = None
        for dev in self.devices():
            try:
                t0 = time.time()
                model = build_sam2(str(cfg), str(ckpt), device=dev)
                self.model = model
                self.predictor = Predictor(model)
                self.variant = variant
                self.device = dev
                self.build_ms = (time.time() - t0) * 1000.0
                _log("模型就绪 variant=%s device=%s %.0f ms" % (variant, dev, self.build_ms))
                return
            except Exception as ex:
                last = ex
                _log("device=%s 建模型失败：%r" % (dev, ex))
                self.model = None
                self.predictor = None
                self.variant = None
                self.device = None
                _empty_cache()
        raise WErr("error", "SAM2(%s) 加载失败：%s: %s" % (variant, type(last).__name__, last))

    # -- embedding 快照（省掉重复 set_image 的几秒）-------------------------
    def _snapshot(self):
        p = self.predictor
        return {"features": p._features,
                "orig_hw": p._orig_hw,
                "is_batch": getattr(p, "_is_batch", False),
                "bb": getattr(p, "_bb_feat_sizes", None)}

    def _restore(self, snap):
        p = self.predictor
        p._features = snap["features"]
        p._orig_hw = snap["orig_hw"]
        p._is_batch = snap["is_batch"]
        if snap.get("bb") is not None:
            p._bb_feat_sizes = snap["bb"]
        p._is_image_set = True

    def _forget(self, key):
        for sid, k in list(self.sessions.items()):
            if k == key:
                self.sessions.pop(sid, None)

    def _evict(self):
        dropped = 0
        while len(self.cache) > max(1, CACHE_MAX):
            key = next(iter(self.cache))
            self.cache.pop(key, None)
            self._forget(key)
            dropped += 1
            _log("LRU 释放 embedding：%s" % (key,))
        if dropped:
            _empty_cache()

    # -- 三个动作 ----------------------------------------------------------
    def load(self, sid, path, variant):
        t_all = time.time()
        self.ensure_model(variant)
        key = (str(path), variant)
        ent = self.cache.get(key)
        cached = ent is not None
        if cached:
            self.cache.move_to_end(key)
            self._restore(ent["snapshot"])
            ent["last"] = time.time()
            embed_ms = 0.0
        else:
            img = _read_rgb(path)
            t0 = time.time()
            self.predictor.set_image(img)
            embed_ms = (time.time() - t0) * 1000.0
            ent = {"snapshot": self._snapshot(),
                   "w": int(img.shape[1]), "h": int(img.shape[0]),
                   "last": time.time(), "variant": variant, "path": str(path),
                   "embed_ms": embed_ms}
            self.cache[key] = ent
            self._evict()
        self.sessions[sid] = key
        return {"session": sid, "w": ent["w"], "h": ent["h"], "cached": bool(cached),
                "variant": variant, "device": self.device,
                "build_ms": round(self.build_ms, 1),
                "embed_ms": round(embed_ms, 1),
                "cached_count": len(self.cache),
                "ms": round((time.time() - t_all) * 1000.0, 1)}

    def predict(self, sid, points, box, multimask):
        t_all = time.time()
        key = self.sessions.get(sid)
        if key is None:
            raise WErr("session", "会话不存在或已过期（session=%s），重新 POST /api/mask/sam/load 一次" % sid)
        if self.model is None or self.variant != key[1]:
            # 模型被 free_model() / 空闲回收 / 换 variant 收走了。这时**不能**
            # ensure_model(key[1]) 把旧 variant 重建回来（正是要避免的那件事），
            # 也不算「用当前模型 restore 旧快照」（快照是旧 variant 的，算出来是垃圾）。
            # 直接判会话过期，让前端重新 load 一次。
            raise WErr("session",
                       "会话的模型（variant=%s）已被释放或换成了 %s，重新 load 一次"
                       % (key[1], self.variant or "无"))
        ent = self.cache.get(key)
        if ent is None:
            raise WErr("session", "这张图的 embedding 已被释放（LRU / 空闲清理），重新 load 一次")
        self.cache.move_to_end(key)
        self._restore(ent["snapshot"])
        ent["last"] = time.time()

        coords = labels = bx = None
        if points:
            coords = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
            labels = np.array([int(p.get("label", 1)) for p in points], dtype=np.int32)
        if box:
            bx = np.array([[float(v) for v in list(box)[:4]]], dtype=np.float32)   # Bx4，predict 要这个形状
        if coords is None and bx is None:
            raise WErr("bad_request", "points 和 box 不能都空着")

        t0 = time.time()
        masks, scores, _low = self.predictor.predict(
            point_coords=coords, point_labels=labels, box=bx,
            multimask_output=bool(multimask))
        infer_ms = (time.time() - t0) * 1000.0

        w, h = int(ent["w"]), int(ent["h"])
        out_masks, out_scores = [], []
        for i in range(int(masks.shape[0])):
            out_masks.append(_png_data_url(_resize_mask(masks[i], w, h)))
            out_scores.append(float(scores[i]) if i < len(scores) else 0.0)
        return {"masks": out_masks, "scores": out_scores, "w": w, "h": h,
                "count": len(out_masks), "multimask": bool(multimask),
                "infer_ms": round(infer_ms, 1),
                "ms": round((time.time() - t_all) * 1000.0, 1)}

    def drop(self, sid):
        key = self.sessions.pop(sid, None)
        if key is None:
            return {"dropped": False}
        alive = any(k == key for k in self.sessions.values())
        if not alive:
            self.cache.pop(key, None)
            _empty_cache()
        return {"dropped": True, "cached_count": len(self.cache)}

    def vram(self):
        """占了多大显存（torch 没进来就别为此把它 import 进来）。"""
        if "torch" not in _CACHE or self.model is None:
            return None
        try:
            t = _CACHE["torch"]
            if not t.cuda.is_available():
                return None
            return {"allocated_mb": round(t.cuda.memory_allocated() / 2.0 ** 20, 1),
                    "reserved_mb": round(t.cuda.memory_reserved() / 2.0 ** 20, 1)}
        except Exception:
            return None

    def stats(self):
        return {"variant": self.variant, "device": self.device,
                "sessions": len(self.sessions), "cached": len(self.cache),
                "cache_keys": [list(k) for k in self.cache.keys()],
                "build_ms": round(self.build_ms, 1),
                "vram": self.vram(),
                "gdino": _GDINO.get("device")}


# ---------------------------------------------------------------- GDINO
_GDINO = {"model": None, "device": None, "cfg": None, "idle": 0.0}
_TEXT_ENCODER = "bert-base-uncased"                   # GDINO config 里写死的那个
_TEXT_ENCODER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _gdino_text_encoder():
    """文本编码器目录。优先级与 vendor/groundingdino/util/get_tokenlizer.py 一致：
    环境变量 > <weights>/<name>（平铺的 bert 目录）> <weights>/grounding_dino/<name>（老布局）。
    返回第一个有 vocab.txt 的，都没有就返回首选路径（让报错信息能指出到底该往哪放）。

    注意这里跑在 worker 里，权重根是 --weights 传进来的 WEIGHTS；后端那份
    _weights_root() 在这个进程里不存在（名字不一样，别混）。"""
    cands = []
    env = os.environ.get("DFL_GDINO_TEXT_ENCODER_DIR", "").strip()
    if env:
        cands.append(Path(env))
    cands.append(WEIGHTS / _TEXT_ENCODER)                      # 平铺根下的 bert 目录
    cands.append(WEIGHTS / "grounding_dino" / _TEXT_ENCODER)   # 老布局兜底
    for c in cands:
        try:
            if (c / "vocab.txt").is_file():
                return c
        except OSError:
            continue
    return cands[-1]


def _gdino_missing():
    """缺什么就说什么。**只看文件，不 import 任何东西。**

    以前这里 import transformers 再查 HF 缓存（_hf_has_model）：一是把本地
    weights/bert-base-uncased 那份当不存在，早就放好的词表也照样 501
    （第一次调用光 import transformers 就要 6~7 秒，报错还写着「推理侧 6722 ms」）；
    二是后端这份 runtime/server 根本没装 transformers，问它「有没有 transformers」
    本身就是问错了进程——能加载的事实只有 engine 那个 worker 知道。

    现在改成就地判文件，每次调用都判（5 次 stat，比缓存过期后要重启强），
    补齐文件后不用重启后端。"""
    cfg = ENGINE / "vendor" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt = WEIGHTS / "groundingdino_swint_ogc.pth"          # 平铺：没有 grounding_dino/ 目录
    if not (ENGINE / "vendor" / "groundingdino").is_dir():
        return ("GroundingDINO 建模代码缺失（app/engine/vendor/groundingdino）",
                [str(ENGINE / "vendor" / "groundingdino")])
    if not cfg.is_file():
        return "GroundingDINO 配置缺失：%s" % cfg, [str(cfg)]
    if not ckpt.is_file():
        return ("GroundingDINO 权重缺失：%s —— 下载 sam 包（--group sam）" % ckpt,
                [str(ckpt)])
    te = _gdino_text_encoder()
    if not (te / "vocab.txt").is_file():
        return ("GroundingDINO 的文本编码器词表缺：%s\\vocab.txt —— 放上 vocab.txt"
                "（再加上 config.json / pytorch_model.bin 就能完全离线跑）" % te,
                [str(te / "vocab.txt")])
    if not (te / "config.json").is_file():
        return ("文本编码器缺结构文件 %s\\config.json —— bert-base-uncased 的 config.json"
                "（570 B），下法：curl -L -o \"%%s\" \"https://hf-mirror.com/bert-base-uncased/"
                "resolve/main/config.json\"" % te, [str(te / "config.json")])
    if not any((te / f).is_file() for f in _TEXT_ENCODER_WEIGHTS):
        return ("文本编码器缺权重 %s\\pytorch_model.bin（约 420 MB；GDINO 的 checkpoint 里"
                "不含 bert.* 权重，省不掉）—— 下法：curl -L -o \"%%s\" "
                "\"https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin\""
                % te, [str(te / "pytorch_model.bin")])
    return None


def _gdino_model():
    if _GDINO["model"] is not None:
        _GDINO["idle"] = time.time()
        return _GDINO
    cfg = ENGINE / "vendor" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt = WEIGHTS / "groundingdino_swint_ogc.pth"          # 平铺：没有 grounding_dino/ 目录
    _T, load_model, _predict = _import_gdino()
    last = None
    for dev in Sam.devices():
        try:
            t0 = time.time()
            m = load_model(str(cfg), str(ckpt), device=dev)
            _GDINO.update({"model": m, "device": dev, "cfg": str(cfg), "idle": time.time()})
            _log("GDINO 就绪 device=%s %.0f ms" % (dev, (time.time() - t0) * 1000.0))
            return _GDINO
        except Exception as ex:
            last = ex
            _log("GDINO device=%s 失败：%r" % (dev, ex))
            _empty_cache()
    raise WErr("error", "GroundingDINO 加载失败：%s: %s" % (type(last).__name__, last))


def _op_gdino(req):
    miss = _gdino_missing()
    if miss:
        raise WErr("missing", miss[0], missing=miss[1])
    path = req.get("path") or ""
    text = str(req.get("text") or "face.").strip() or "face."
    box_t = float(req.get("box_threshold") or 0.35)
    text_t = float(req.get("text_threshold") or 0.25)
    t_all = time.time()
    try:
        g = _gdino_model()
    except ImportError as ex:
        # transformers / torchvision / timm 只装在 engine 那份 python 里；
        # 缺包按「装不上」报 501，别混进 500 的推理异常里。
        raise WErr("missing", "推理环境缺包（GroundingDINO 要 transformers + torchvision）：%s" % ex,
                   missing=[str(ex)])
    T, _load, predict = _import_gdino()
    from PIL import Image
    rgb = _read_rgb(path)          # _read_rgb 已经 convert("RGB")，恒为 HxWx3
    tf = T.Compose([T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_t, _ = tf(Image.fromarray(rgb), None)
    t0 = time.time()
    boxes, logits, phrases = predict(model=g["model"], image=img_t, caption=text,
                                     box_threshold=box_t, text_threshold=text_t,
                                     device=g["device"])
    infer_ms = (time.time() - t0) * 1000.0
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    out_boxes, out_scores = [], []
    if boxes is not None and len(boxes):
        arr = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        for row in arr:
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            out_boxes.append([round(min(max(x1, 0.0), w), 1), round(min(max(y1, 0.0), h), 1),
                              round(min(max(x2, 0.0), w), 1), round(min(max(y2, 0.0), h), 1)])
        sc = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        out_scores = [round(float(v), 4) for v in sc.reshape(-1)[:len(out_boxes)]]
    return {"boxes": out_boxes, "scores": out_scores, "phrases": list(phrases or []),
            "text": text, "w": w, "h": h, "device": g["device"],
            "infer_ms": round(infer_ms, 1),
            "ms": round((time.time() - t_all) * 1000.0, 1)}


# ---------------------------------------------------------------- 空闲回收
def _reaper():
    while True:
        time.sleep(15)
        try:
            with _lock:
                now = time.time()
                stale = [k for k, e in Sam.cache.items() if now - float(e.get("last") or 0) > IDLE_SEC]
                for k in stale:
                    Sam.cache.pop(k, None)
                    Sam._forget(k)
                if stale:
                    _empty_cache()
                    _log("空闲释放 embedding x%d" % len(stale))
                if (not Sam.cache) and Sam.model is not None and now - Sam.idle_since > IDLE_SEC:
                    Sam.free_model()
                    _log("空闲释放 SAM 模型")
                if _GDINO["model"] is not None and now - float(_GDINO.get("idle") or 0) > IDLE_SEC:
                    _GDINO["model"] = None
                    _GDINO["device"] = None
                    _empty_cache()
                    _log("空闲释放 GDINO")
        except Exception as ex:
            _log("reaper 出错：%r" % (ex,))


Sam = SamState()


# ---------------------------------------------------------------- 收包
def _op_load(req):
    sid = str(req.get("session") or "")
    if not sid:
        raise WErr("bad_request", "load 缺 session")
    return Sam.load(sid, req.get("path") or "", str(req.get("variant") or "base_plus"))


def _op_predict(req):
    sid = str(req.get("session") or "")
    if not sid:
        raise WErr("bad_request", "predict 缺 session")
    return Sam.predict(sid, req.get("points") or [], req.get("box"),
                       req.get("multimask", True))


def _op_drop(req):
    return Sam.drop(str(req.get("session") or ""))


def _op_stats(req):
    return Sam.stats()


def _op_ping(req):
    info = {"torch": None, "cuda": False, "device": DEVICE_PREF, "pid": os.getpid()}
    try:
        t = _torch()
        info["torch"] = t.__version__
        info["cuda"] = bool(t.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = t.cuda.get_device_name(0)
            info["vram_mb"] = round(t.cuda.get_device_properties(0).total_memory / 2.0 ** 20, 1)
    except Exception as ex:
        info["error"] = repr(ex)
    return info


OPS = {"load": _op_load, "predict": _op_predict, "gdino": _op_gdino,
       "drop": _op_drop, "stats": _op_stats, "ping": _op_ping}


def handle(req):
    rid = req.get("id") or 0
    op = str(req.get("op") or "")
    t0 = time.time()
    try:
        with _lock:
            Sam.idle_since = time.time()
            fn = OPS.get(op)
            if fn is None:
                raise WErr("bad_request", "未知操作：%r" % (op,))
            res = dict(fn(req) or {})
        res["ok"] = True
        res["id"] = rid
        res["ms"] = round((time.time() - t0) * 1000.0, 1)
        return res
    except WErr as err:
        _empty_cache()
        return {"id": rid, "ok": False, "kind": err.kind, "message": err.message,
                "missing": err.missing, "ms": round((time.time() - t0) * 1000.0, 1)}
    except Exception as ex:
        kind = "oom" if _is_oom(ex) else "error"
        _empty_cache()
        _log("op=%s 异常：%s" % (op, traceback.format_exc()[-800:]))
        return {"id": rid, "ok": False, "kind": kind,
                "message": "%s: %s" % (type(ex).__name__, ex),
                "trace": traceback.format_exc()[-1200:],
                "ms": round((time.time() - t0) * 1000.0, 1)}


def main():
    threading.Thread(target=_reaper, daemon=True).start()
    _log("启动 pid=%d engine=%s weights=%s" % (os.getpid(), ENGINE, WEIGHTS))
    out = sys.stdout
    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        try:
            resp = handle(req)
        except Exception as ex:                       # handle 里兜住了，这里是最后一道
            resp = {"id": req.get("id") or 0, "ok": False, "kind": "error",
                    "message": "worker 内部错误：%r" % (ex,)}
        try:
            out.write(json.dumps(resp, ensure_ascii=True) + "\n")
            out.flush()
        except Exception:
            break
    _log("stdin 关了，退出")


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------- 后端侧进程桥
class _InferTimeout(Exception):
    """推理超时（已经 kill 掉进程）。"""


class _InferDown(Exception):
    """推理进程起不来 / 半路死了。"""


class _MaskInfer(object):
    """engine python 推理进程，全后端就这一份。

    一把锁把所有推理串成一条线：SAM 显存 1.5 G 起步，两个请求同时进去只会 OOM。
    请求带自增 id，回包按 id 对上；超过 timeout 直接 kill 进程（推理没有中间态），
    下一次调用自动重启并重新加载模型。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._out = None
        self._seq = 0
        self._src = None
        self._last_err = ""

    # -- 进程 --------------------------------------------------------------
    def _worker_file(self) -> Path:
        """把 _WORKER_SRC 落到 %TEMP%（内容变了才重写），返回路径。"""
        import hashlib
        if self._src is not None and self._src.is_file():
            return self._src
        blob = _WORKER_SRC.encode("utf-8")
        tag = hashlib.sha1(blob).hexdigest()[:10]
        d = Path(tempfile.gettempdir()) / "dfl_mask_worker"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            d = Path(tempfile.mkdtemp(prefix="dfl_mask_"))
        f = d / ("mask_worker_%s.py" % tag)
        if (not f.is_file()) or f.stat().st_size != len(blob):
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_bytes(blob)
            os.replace(str(tmp), str(f))
        self._src = f
        return f

    def alive(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    def _spawn(self):
        py = _engine_python()
        if not Path(py).is_file():
            raise _InferDown("找不到推理用的 python：%s（runtime/engine 那份）" % py)
        src = self._worker_file()
        args = [str(py), str(src),
                "--engine", str(_engine_dir()), "--repo", str(_root_dir()),
                "--weights", str(_weights_root()), "--device", "auto",
                "--cache", str(_CACHE_MAX), "--idle", str(_IDLE_SEC)]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONWARNINGS", "ignore")     # 警告走 stderr，别拖慢
        # 推理进程一律离线：用户机器可能没网，transformers / huggingface_hub 任何一次
        # 「先试试能不能下」都会变成 30 秒超时，最后报一个跟真实原因无关的错。
        # 这三个用硬赋值（不是 setdefault）——外面环境里残留的值不许把它重新打开。
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        # vendor/get_tokenlizer.py 靠这个找 weights/bert-base-uncased（平铺的 bert 目录）；
        # _weights_root() 可能被 $DFL_WEIGHTS_DIR 改过，别让它去猜默认路径。
        env["DFL_WEIGHTS_DIR"] = str(_weights_root())
        try:
            proc = _sp.Popen(args, stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
                             cwd=str(_root_dir()), env=env,
                             creationflags=_CREATE_NO_WINDOW)
        except Exception as ex:
            raise _InferDown("推理进程起不来：%s" % ex)
        self._proc = proc
        self._out = queue.Queue()
        self._last_err = ""
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()
        return proc

    def _read_stdout(self, proc):
        q = self._out
        try:
            for raw in iter(proc.stdout.readline, b""):
                if not raw:
                    break
                q.put(raw.decode("utf-8", "replace").strip())
        except Exception:
            pass
        finally:
            try:
                q.put(None)        # EOF 哨兵：让等待方立刻知道进程没了
            except Exception:
                pass

    def _read_stderr(self, proc):
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    self._last_err = line
        except Exception:
            pass

    def _kill(self, why=""):
        proc, self._proc = self._proc, None
        if why:                       # 以前写进一个没人读的 self._dead，日志里才有用
            _log("推理进程 terminate：%s" % why)
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            for f in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    def stop(self):
        try:
            self._kill("后端退出")
        except Exception:
            pass

    # -- 调用 --------------------------------------------------------------
    def call(self, op, payload=None, timeout=30.0):
        payload = dict(payload or {})
        with self._lock:
            if not self.alive():
                self._spawn()
            self._seq += 1
            rid = self._seq
            req = {"id": rid, "op": op}
            req.update(payload)
            blob = (json.dumps(req, ensure_ascii=True) + "\n").encode("utf-8")
            try:
                self._proc.stdin.write(blob)
                self._proc.stdin.flush()
            except Exception as ex:
                self._kill("写不进推理进程：%s" % ex)
                raise _InferDown("推理进程不可用（%s）；stderr 末尾：%s" % (ex, self._last_err or "无"))
            deadline = time.time() + max(1.0, float(timeout))
            while True:
                remain = deadline - time.time()
                if remain <= 0:
                    self._kill("超时")
                    raise _InferTimeout("推理超过 %.0f 秒" % timeout)
                try:
                    line = self._out.get(timeout=remain)
                except queue.Empty:
                    self._kill("超时")
                    raise _InferTimeout("推理超过 %.0f 秒" % timeout)
                if line is None:
                    tail = self._last_err or "无 stderr 输出"
                    self._kill("进程退出")
                    raise _InferDown("推理进程退出了；stderr 末尾：%s" % tail)
                try:
                    msg = json.loads(line)
                except Exception:
                    continue                    # 不是协议行，丢掉
                if msg.get("id") == rid:
                    return msg
                # 别的 id：上次超时留下的迟到响应，忽略继续等


_infer = _MaskInfer()
atexit.register(_infer.stop)


def _raise_infer(res):
    """把推理进程的 ok=False 翻成契约错误：worker 的 kind → code + 状态码。

      missing (权重/依赖缺)      → missing_weights 501
      image   (图没了/解不开)    → image_missing   404
      session (会话/快照没了)    → session_expired 409
      bad_request                → bad_args        400
      oom                        → infer_failed    507
      其余 error                 → infer_failed    500
    """
    if not res.get("ok"):
        kind = str(res.get("kind") or "")
        msg = str(res.get("message") or "推理失败")
        ms = res.get("ms")
        if ms is not None:
            msg = msg + "（推理侧 %.0f ms）" % float(ms or 0)
        missing = list(res.get("missing") or [])
        if kind == "missing":
            raise MaskErr("missing_weights", msg, 501, missing)
        if kind == "image":
            raise MaskErr("image_missing", msg, 404)
        if kind == "session":
            raise MaskErr("session_expired", msg, 409)
        if kind == "bad_request":
            raise MaskErr("bad_args", msg, 400)
        if kind == "oom":
            raise MaskErr("infer_failed", "显存不足：" + msg, 507, missing)
        raise MaskErr("infer_failed", msg, 500, missing)
    return res


def _call_or_504(op, payload, timeout, what):
    try:
        return _infer.call(op, payload, timeout)
    except _InferTimeout:
        raise MaskErr("infer_failed",
                      "%s 超过 %.0f 秒没算完，已终止推理进程（下次调用会重新加载模型）"
                      % (what, timeout), 504)
    except _InferDown as ex:
        raise MaskErr("infer_failed", "推理进程不可用：%s" % ex, 500)


# ---------------------------------------------------------------- 路由
@router.post("/api/mask/sam/load")
@_mask_api
async def mask_sam_load(body: dict):
    """读图 + set_image 算 embedding（贵，几秒）。同一 (image, variant) 重复调直接命中缓存。"""
    body = body or {}
    variant = _norm_variant(body.get("variant"))
    ckpt = _sam_ckpt(variant)
    if not ckpt.is_file():
        # 缺权重是 501 不是 404：前端据此把 SAM 工具置灰，而不是当成「这张图没了」
        raise MaskErr("missing_weights",
                      "SAM2 权重缺失：%s —— 先在「设置 → 权重包」里下载 sam 包"
                      "（python -m modelhub.fetch --group sam）" % ckpt,
                      501, [str(ckpt)])
    path = _resolve_image(body.get("image"))
    sid = "sam_" + uuid.uuid4().hex[:16]
    res = await asyncio.to_thread(_call_or_504, "load",
                                  {"session": sid, "path": str(path), "variant": variant},
                                  _SAM_LOAD_TIMEOUT, "SAM load")
    _raise_infer(res)
    return {"ok": True, "session": res.get("session") or sid,
            "w": int(res.get("w") or 0), "h": int(res.get("h") or 0),
            "cached": bool(res.get("cached")),
            "variant": variant, "device": res.get("device"),
            "image": str(path), "embed_ms": res.get("embed_ms"),
            "cached_count": res.get("cached_count"),
            "ms": res.get("ms")}


@router.post("/api/mask/sam/predict")
@_mask_api
async def mask_sam_predict(body: dict):
    """点集 / 框推理。label=1 是要的部分，0 是不要的；box 可选。返回若干候选掩码 + 分数。"""
    body = body or {}
    sid = str(body.get("session") or "").strip()
    if not sid:
        raise MaskErr("bad_args", "缺 session —— 先 POST /api/mask/sam/load 换一个")
    raw_points = body.get("points") or []
    if not isinstance(raw_points, list):
        raise MaskErr("bad_args", "points 要是数组 [{x,y,label}]")
    points = []
    for p in raw_points:
        if not isinstance(p, dict):
            raise MaskErr("bad_args", "points 每项要是 {x,y,label}，收到 %r" % (p,))
        try:
            x = float(p.get("x"))
            y = float(p.get("y"))
        except (TypeError, ValueError):
            raise MaskErr("bad_args", "points 每项的 x / y 要是数字，收到 %r" % (p,))
        try:
            lab = int(p.get("label", 1))
        except (TypeError, ValueError):
            lab = 1
        points.append({"x": x, "y": y, "label": 1 if lab == 1 else 0})
    box = body.get("box")
    if box is not None:
        try:
            box = [float(v) for v in list(box)[:4]]
        except (TypeError, ValueError):
            raise MaskErr("bad_args", "box 要是 [x1,y1,x2,y2] 四个数字")
        if len(box) != 4:
            raise MaskErr("bad_args", "box 要是 [x1,y1,x2,y2] 四个数字")
    if not points and box is None:
        raise MaskErr("bad_args", "points 和 box 至少给一个")
    mm = body.get("multimask", True)
    if isinstance(mm, str):
        mm = mm.strip().lower() in ("1", "true", "yes", "on", "y")
    multimask = bool(mm)

    res = await asyncio.to_thread(_call_or_504, "predict",
                                  {"session": sid, "points": points, "box": box,
                                   "multimask": multimask},
                                  _SAM_PREDICT_TIMEOUT, "SAM predict")
    _raise_infer(res)
    return {"ok": True, "masks": res.get("masks") or [], "scores": res.get("scores") or [],
            "w": res.get("w"), "h": res.get("h"), "multimask": multimask,
            "infer_ms": res.get("infer_ms"), "ms": res.get("ms")}


@router.post("/api/mask/gdino")
@_mask_api
async def mask_gdino(body: dict):
    """文本 → 框（GroundedSAM2 的第一步）。框按原图像素给，xyxy。"""
    body = body or {}
    path = _resolve_image(body.get("image"))
    text = str(body.get("text") or "face.").strip() or "face."
    res = await asyncio.to_thread(_call_or_504, "gdino",
                                  {"path": str(path), "text": text,
                                   "box_threshold": body.get("box_threshold", 0.35),
                                   "text_threshold": body.get("text_threshold", 0.25)},
                                  _GDINO_TIMEOUT, "GroundingDINO")
    _raise_infer(res)               # 缺包 / 缺词表 → missing_weights 501
    return {"ok": True, "boxes": res.get("boxes") or [], "scores": res.get("scores") or [],
            "phrases": res.get("phrases") or [], "text": text,
            "w": res.get("w"), "h": res.get("h"), "device": res.get("device"),
            "infer_ms": res.get("infer_ms"), "ms": res.get("ms")}


@router.get("/api/mask/sam/status")
def mask_sam_status():
    """看一眼推理进程里现在有什么（调不动就当没起来）。"""
    try:
        res = _infer.call("stats", {}, 5.0)
    except (_InferTimeout, _InferDown) as ex:
        return {"ok": False, "code": "infer_failed", "detail": str(ex),
                "alive": _infer.alive(), "error": str(ex)}
    if not res.get("ok"):
        # 状态探针本身不失败退出（200），但错误形状和契约一致，前端好统一处理
        return {"ok": False, "code": "infer_failed", "alive": _infer.alive(), **res}
    return {"ok": True, "alive": _infer.alive(), **res}
