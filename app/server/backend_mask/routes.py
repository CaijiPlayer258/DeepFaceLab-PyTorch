# -*- coding: utf-8 -*-
"""遮罩编辑器 / 工程 · HTTP 层（薄）：路由表 + 入参校验 + 契约错误映射。

真身在这个包的另外两个文件里，本文件只把请求翻译成一次调用：

  · backend_mask/painter.py   遮罩落盘/读回（<aligned>/_masks PNG + DFLJPG APP15['xseg_mask']）、
                              SAM2 / GroundingDINO 推理进程桥（另一进程 + JSON lines）
  · backend_mask/occlusion.py 遮挡率排序器（数据集侧按需调，不经过本路由）

所有 /api/mask/* 的失败响应形状与拆分前一致（MaskErr / _mask_api 定义在 painter，
本文件再导出，shim 那边也摸得到）：
    {"ok": false, "code": "<code>", "detail": "...", "missing": [...]?}
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..app import IMAGE_EXT, WORKSPACE, _mask_ai_stub, _mask_state

from .painter import (
    MaskErr, _GDINO_TIMEOUT, _InferDown, _InferTimeout, _SAM_LOAD_TIMEOUT,
    _SAM_PREDICT_TIMEOUT, _call_or_504, _decode_mask, _embed_mask_into_dfljpg, _fit_mask,
    _has_mask, _image_wh, _infer, _load_mask_b64, _mask_api, _mask_png_path, _norm_variant,
    _raise_infer, _resolve_image, _sam_ckpt, _write_mask_png,
)

router = APIRouter()


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
