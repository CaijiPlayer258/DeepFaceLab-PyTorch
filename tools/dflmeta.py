# -*- coding: utf-8 -*-
"""dflmeta —— DFL 脸图元数据（dFLd / APP15）保真工具

2026-10-17 教训：切脸工具落位时用 `cv2.imdecode → cv2.imencode` 重编码，
PNG/JPEG 附加块全丢（dFLd 元数据没了）。Filter 不依赖 landmarks，但
训练 / DFM / 合成都要。本模块提供无损搬运：

- `load_dict(path)`            读 DFLIMG 元数据 dict（无则 None）
- `jpg_with_meta_to_png(src, dst)`  深变 Extractor 的 jpg（APP15 内嵌 pickle）
                                      → png（IDAT 前插 dFLd 块），像素不变、元数据全保留
- `write_meta_into_png(png, d)` 给已有 PNG 换/补 dFLd（原 IDAT 复用，不重编码）
- `save_png_with_meta(img, d, dst)` 新图 + dict 直接写 png（通用）

块格式与 DFLPNG.py 一致：len(4B BE) + b'dFLd' + pickle(payload, protocol=2) + crc32(type+data)。
"""
import os
import pickle
import struct
import sys
import zlib

import cv2
import numpy as np

# 让 `from DFLIMG import ...` 在任意 cwd 下可用（脚本常从 tools\ 启动，sys.path 只有 tools）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PNG_SIG = b"\x89PNG\r\n\x1a\n"
DFL_TYPE = b"dFLd"


def _read(path):
    return np.fromfile(path, dtype=np.uint8).tobytes()


def _write(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.frombuffer(data, dtype=np.uint8).tofile(path)


def _chunk(ctype, data):
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def _normalize(d):
    """与 DFLPNG.dump 相同的 numpy 归一化（ndarray→bytes/list）。"""
    xseg_val = d.get("xseg_mask")

    def _unumpy(obj):
        if isinstance(obj, np.ndarray):
            if obj is xseg_val:
                return obj
            if obj.dtype.kind in ("u", "b"):
                return bytes(obj)
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _unumpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_unumpy(i) for i in obj]
        return obj

    return _unumpy(dict(d))


def _pickle_payload(d):
    data = pickle.dumps(_normalize(d), protocol=2)
    return data.replace(b"numpy._core", b"numpy.core")


def load_dict(path):
    """读 DFLIMG 元数据；无元数据/非 DFL 图返回 None。"""
    try:
        from DFLIMG import DFLIMG

        inst = DFLIMG.load(path)
        if inst is not None and inst.has_data():
            return inst.get_dict()
    except Exception:
        pass
    return None


def _png_chunks(png_bytes):
    if png_bytes[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    chunks, pos = [], 8
    while pos + 12 <= len(png_bytes):
        ln, = struct.unpack(">I", png_bytes[pos:pos + 4])
        ctype = png_bytes[pos + 4:pos + 8]
        cdata = png_bytes[pos + 8:pos + 8 + ln]
        if len(cdata) < ln:
            raise ValueError("truncated chunk")
        chunks.append((ctype, cdata))
        pos += 12 + ln
        if ctype == b"IEND":
            break
    return chunks


def _png_with_meta(chunks, d):
    """在第一个 IDAT 前插入（或替换）dFLd 块。"""
    payload = _pickle_payload(d) if d else None
    out, inserted = [PNG_SIG], False
    for ctype, cdata in chunks:
        if ctype == DFL_TYPE:
            continue
        if payload is not None and not inserted and ctype == b"IDAT":
            out.append(_chunk(DFL_TYPE, payload))
            inserted = True
        out.append(_chunk(ctype, cdata))
    if payload is not None and not inserted:      # 异常文件：塞 IEND 前
        body = b"".join(out[1:])
        idx = body.rfind(struct.pack(">I", 0) + b"IEND")
        if idx >= 0:
            body = body[:idx] + _chunk(DFL_TYPE, payload) + body[idx:]
        return PNG_SIG + body
    return b"".join(out)


def jpg_with_meta_to_png(src, dst):
    """Extractor 的 jpg（含 APP15 元数据）→ png（含 dFLd），像素不变。

    返回 True/False（False = 源图无元数据，依然转换但只保像素）。
    """
    d = load_dict(src)
    img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"decode failed: {src}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"png encode failed: {src}")
    chunks = _png_chunks(buf.tobytes())
    _write(dst, _png_with_meta(chunks, d))
    return d is not None


def write_meta_into_png(png_path, d):
    """给已有 PNG 写入/替换 dFLd（复用原 IDAT，不重编码像素）。"""
    chunks = _png_chunks(_read(png_path))
    _write(png_path, _png_with_meta(chunks, d))


def save_png_with_meta(img, d, dst):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"png encode failed: {dst}")
    _write(dst, _png_with_meta(_png_chunks(buf.tobytes()), d))
