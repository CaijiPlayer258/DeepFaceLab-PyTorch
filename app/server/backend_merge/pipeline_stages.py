# -*- coding: utf-8 -*-
"""分段流水线：抽帧 / 切脸 / 预测 / 遮罩 / 合成 / 编码 六段（_stage_* 全家）+ DFM/XSeg 会话。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import numpy as np   # 原 HEAD 缺这行：_stage_pred_assign 用裸 np，被 except 吃掉后整条分支永远空

import collections
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from .._common.env import (
    _env_int,
)
from .._common.http import (
    _BadInput, _as_dict, _bool_val, _float_val, _int_val,
)
from .._common.proc import (
    _kill_proc_tree,
)
from ._shared import (
    _norm_path, _same_path, _settings_val,
)


# ============================================================
# 分段式视频导出：<视频名>_cache/ + 五个能单独跑的阶段
# ------------------------------------------------------------
# 用户定的口径（推翻原来「一个循环 decode→detect→swap→encode」）：
#
#   D:/x/9月29日 (5).mp4
#   D:/x/9月29日 (5)_cache/           <- 同目录，名字 + _cache
#     frames/   000001.png …          <- 切帧；合成阶段**同名就地覆盖**成合成结果
#     aligned/  000001_00.jpg …       <- 切脸：DFLJPG（original_name + 68 点 + face_mat）
#                                       遮罩阶段把 256 的 XSeg 遮罩写回它的 APP15
#     meta.json                       <- 源信息 / 阶段标记 / settings 快照
#
# 五个阶段各自一个 API（都能单独跑，也从任意已完成的阶段续跑）：
#   POST /api/merge/stage/frames   视频 -> frames/
#   POST /api/merge/stage/faces    frames/ -> aligned/
#   POST /api/merge/stage/mask     aligned/ -> 写回 APP15['xseg_mask']（只有遮罩来源是
#                                  XSeg / XSegLite 时才需要；来源=DFM 时本阶段自动跳过）
#   POST /api/merge/stage/merge    frames/ + aligned/ + dfm -> 就地覆盖 frames/
#   POST /api/merge/stage/encode   frames/ + 源音轨 -> <视频名>_merged.mp4
#   状态 GET /api/merge/stage/status · 取消 POST /api/merge/stage/cancel
#
# 三条硬口径：
#   1) cache 只是缓存：**导出成功就把整个 <名字>_cache/ 删掉**；失败/取消保留，续跑用。
#   2) 合成结果直接覆盖 frames/ 里的同名帧，不另开 merged/（不凭空多一倍空间）。
#      覆盖过的 frames 由 meta.frame_swapped=true 标记 —— 这时再跑切帧会被拒
#      （否则会把原帧盖回来、却留下已经合成过的 meta）。
#   3) 帧图一律最高质量：默认 PNG 无损（IMWRITE_PNG_COMPRESSION=3）；要 jpg 就是
#      quality=100 + 4:4:4，绝不用默认的 95/85/75。
# ============================================================
_STAGE_ORDER = ("frames", "faces", "pred", "mask", "merge", "encode")
_STAGE_LABEL = {"frames": "切帧", "faces": "切脸", "pred": "换脸", "mask": "遮罩",
                "merge": "合成", "encode": "导出", "fused": "快速导出"}
# 老 _merge_job["stage"] 的 7 段进度条（0..6）里没有「遮罩」，给它 2 就是和切脸同一格：
# 老前端照样能看出「还在前面几步」，不会因为多插一步跳到 3 把 31% 显示成合成中。
_STAGE_NUM = {"frames": 1, "faces": 2, "pred": 2, "mask": 2, "merge": 3, "encode": 6}
_STAGE_PCT = {"frames": (0, 10), "faces": (10, 20), "pred": (20, 35), "mask": (35, 48),
              "merge": (48, 92), "encode": (92, 100)}
_STAGE_ALIAS = {
    "frames": "frames", "frame": "frames", "extract-frames": "frames",
    "extract_frames": "frames", "extractframes": "frames", "decode": "frames",
    "切帧": "frames",
    "faces": "faces", "face": "faces", "extract-faces": "faces",
    "extract_faces": "faces", "extractfaces": "faces", "align": "faces",
    "aligned": "faces", "切脸": "faces",
    "pred": "pred", "preds": "pred", "dfm": "pred", "infer": "pred", "face": "faces",
    "swap": "pred", "swapped": "pred", "换脸": "pred", "推理": "pred", "dfm推理": "pred",
    "mask": "mask", "masks": "mask", "xseg": "mask", "xseg-mask": "mask",
    "xseg_mask": "mask", "xsegs": "mask", "seg": "mask", "apply-mask": "mask",
    "segmask": "mask", "遮罩": "mask", "遮罩推理": "mask",
    "merge": "merge", "compose": "merge", "merged": "merge",
    "swap-merge": "merge", "merge-frames": "merge", "合成": "merge",
    "encode": "encode", "mux": "encode", "export": "encode", "video": "encode",
    "导出": "encode",
}
_STAGE_META_NAME = "meta.json"
_STAGE_IMG_EXTS = (".png", ".jpg", ".jpeg")
_STAGE_LOG_MAX = 3000
_STAGE_LOCK = threading.RLock()
_STAGE_JOB = {
    "running": False, "ok": None, "cancelled": False, "error": "", "error_detail": "",
    "phase": "idle", "stage": "", "stage_index": -1, "stage_pct": 0, "pct": 0,
    "msg": "", "video": "", "cache_dir": "", "out_video": "", "frames_total": 0,
    "frames_done": 0, "frame_fps": 0.0, "eta_s": None, "counts": {}, "stages_done": [],
    "log": [], "meta": {}, "seconds": {}, "resumed": {}, "swapped": 0, "copied": 0,
    "no_face": 0, "cache_removed": False, "started_at": None, "finished_at": None,
    "chain": False, "keep_cache": False, "force": False,
    # 生效配置 + 一致性警告（[eff] 行同款；进度页顶部横幅读它）
    "eff": None, "eff_warnings": [], "warnings": [],
}


def _stage_name(raw):
    """阶段名 -> 规范名（frames/faces/mask/merge/encode）。认不出来返回 ""。"""
    return _STAGE_ALIAS.get(str(raw or "").strip().lower(), "")


# ---------------- 遮罩来源（决定 mask 阶段要不要跑） ----------------
def _mask_src(settings, mask_mode=None) -> dict:
    """settings（或 meta.settings）-> {"source": xseg|dfm, "model": lite|xseg, "label", "model_from"}。

    两代口径都认，**显式 mask_source 优先**：
      · 界面 seg 给的原文：DFM / XSeg / XSegLite；
      · mask_mode 整数：6/7/8/9 = XSeg 系列（xseg-prd / xseg-dst / prd×dst / 全叠），
        0/1/2/3/4/5 = 模型自带遮罩（DFM）。

    **为什么不返回 "auto" 了（用户报的 bug 的根因之一）**：
      老前端（app.js 的 mgMergeSettings）把「遮罩来源」和「遮罩模式」折成一个 mask_mode
      整数发出来，XSeg 和 XSegLite 都变成 6/7/8 —— 模型身份在传输里就丢了。这里以前
      回 "auto"（谁在磁盘上就用谁，Lite 优先）并把 label 写死成 "XSeg"，于是：
        · 界面显示 XSegLite、status 显示 XSeg、实际跑的是哪个看磁盘 —— 三者随时能不一致；
        · _mask_signature 对 XSeg / XSegLite 都算出 "xseg:auto"，切模型**不会**让 mask/
          失效，换模型后静默复用另一个模型算出来的遮罩。
      现在：给了 mask_source 就只认它（且**不**退回另一个模型）；只给 mask_mode 时按
      界面默认值 XSegLite 解析，并在 model_from="mask_mode" 里说明「这是推断的」，
      label 与实际加载的模型永远一致。
    """
    settings = settings if isinstance(settings, dict) else {}
    raw = ""
    for k in ("mask_source", "mask_seg", "seg_source", "seg", "mask_model", "xseg_model"):
        v = settings.get(k)
        if v is not None and str(v).strip():
            raw = str(v).strip()
            break
    key = (raw.lower().replace("_", "").replace("-", "").replace(" ", "")
           .replace("×", ""))
    mm = mask_mode if mask_mode is not None else settings.get("mask_mode")
    try:
        mm = int(mm) if str(mm if mm is not None else "").strip() != "" else None
    except (TypeError, ValueError):
        mm = None

    # ① 显式选择：只有它说了算（认不出就往下按 mask_mode 推，并标注 model_from）
    if key in ("dfm", "model", "learned", "learnedmask", "none", "off", "0"):
        return {"source": "dfm", "model": "", "label": "DFM", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if key in ("xseg", "xsegprddst", "xsegdst", "xsegprd", "1"):
        return {"source": "xseg", "model": "xseg", "label": "XSeg", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if key in ("xseglite", "lite", "xseglite2", "2"):
        return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
                "model_from": "settings", "model_note": ""}
    if raw:
        # 认不出的写法：不猜成 dfm，按 mask_mode（没有就按界面默认的 XSegLite）推断
        _note = ("遮罩来源写了 %r，认不出来 —— 按 mask_mode 推断" % raw)
    else:
        _note = "请求里没有 mask_source（只有 mask_mode），模型按界面默认推断"

    # ② 只有 mask_mode：XSeg 系列一律按界面默认的 XSegLite 解析，并标注「这是推断的」
    if mm is not None and mm >= 6:
        return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
                "model_from": "mask_mode", "model_note": _note}
    if mm is not None:
        return {"source": "dfm", "model": "", "label": "DFM", "raw": raw,
                "model_from": "mask_mode", "model_note": ""}
    # ③ settings 里什么都没有：和界面默认一致（遮罩来源默认 XSegLite）
    return {"source": "xseg", "model": "lite", "label": "XSegLite", "raw": raw,
            "model_from": "default", "model_note": _note}


def _stage_files(dirp, exts=_STAGE_IMG_EXTS) -> list:
    """目录里的图，按文件名排序（000001.png … 定长补零，字典序 = 帧序）。"""
    d = Path(dirp)
    if not d.is_dir():
        return []
    out = []
    try:
        for p in d.iterdir():
            try:
                if p.is_file() and p.suffix.lower() in exts:
                    out.append(p)
            except OSError:
                continue
    except OSError:
        return []
    out.sort(key=lambda p: p.name)
    return out


def _stage_frames_dir(cache) -> Path:
    return Path(cache) / "frames"


def _stage_aligned_dir(cache) -> Path:
    return Path(cache) / "aligned"


def _stage_frame_name(i: int) -> str:
    """第 i 帧（0 起）-> 文件名：000001.png（**1 起**，和界面/用户口径一致）。"""
    return "%06d.png" % (int(i) + 1)


def _stage_face_name(i: int, k: int) -> str:
    return "%06d_%02d.jpg" % (int(i) + 1, int(k))


def _stage_meta_path(cache) -> Path:
    return Path(cache) / _STAGE_META_NAME


def _stage_meta_load(cache) -> dict:
    p = _stage_meta_path(cache)
    try:
        if p.is_file():
            with open(str(p), "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict):
                return d
    except (OSError, ValueError):
        pass
    return {}


def _stage_meta_save(cache, meta: dict):
    """meta 落盘：先写 .tmp 再 os.replace —— 断电也不会留下半个 JSON。"""
    if not isinstance(meta, dict):
        return
    p = _stage_meta_path(cache)
    meta["updated_at"] = time.time()
    tmp = Path(str(p) + ".tmp")
    try:
        with open(str(tmp), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        os.replace(str(tmp), str(p))
    except (OSError, TypeError, ValueError):
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _stage_meta_blank(video: Path, probe: dict, settings: dict) -> dict:
    return {
        "version": 1,
        "source": {"video": str(video), "name": video.name,
                   "fps": float(probe.get("fps") or 0.0),
                   "frames": int(probe.get("frames") or 0),
                   "width": int(probe.get("w") or 0), "height": int(probe.get("h") or 0)},
        "settings": dict(settings or {}),
        "stages": {k: {} for k in _STAGE_ORDER},
        "frame_swapped": False,
        "created_at": time.time(),
    }


def _stage_meta_ready(meta: dict, video: Path, probe: dict) -> dict:
    """老 meta 补齐字段（换了源视频 / 老版本 meta 都能直接接着用）。"""
    meta.setdefault("version", 1)
    src = meta.get("source")
    if not isinstance(src, dict):
        src = {}
    src["video"] = str(video)
    src["name"] = video.name
    src["fps"] = float(probe.get("fps") or src.get("fps") or 0.0)
    src["frames"] = int(probe.get("frames") or src.get("frames") or 0)
    src["width"] = int(probe.get("w") or src.get("width") or 0)
    src["height"] = int(probe.get("h") or src.get("height") or 0)
    meta["source"] = src
    st = meta.get("stages")
    if not isinstance(st, dict):
        st = {}
    for k in _STAGE_ORDER:
        if not isinstance(st.get(k), dict):
            st[k] = {}
    meta["stages"] = st
    meta["frame_swapped"] = bool(meta.get("frame_swapped"))
    return meta


# ---------------- 帧图读写（最高质量）----------------
def _stage_imencode(img, ext=".png", quality=None) -> bytes:
    """帧图编码：png 走无损；jpg 走 **quality=100 + 4:4:4**（禁止默认 95/85）。

    采样因子不再在这里手写：cv2 5.0 只认 ImwriteJPEGSamplingFactorParams 枚举，
    裸数字会被拒并**静默**落成 4:2:0（见 _align.py 的 jpeg444_probe 注释）。
    这里跟对齐脸共用同一份探测结果，一次探测、全进程一个策略。

    quality：帧图 JPEG 质量，默认 100（= _STAGE_JPEG_QUALITY；界面已经没这个旋钮，
    只有老客户端/验收脚本会传）。**PNG 分支不看它**：无损，只认 png_compression。
    """
    import cv2
    ext = str(ext or ".png").lower()
    if ext == ".png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        from .._align import jpeg_encode_params
        q = _STAGE_JPEG_QUALITY if quality is None else max(1, min(100, int(quality)))
        ok, buf = cv2.imencode(".jpg", img, jpeg_encode_params(q, 0))
    if not ok:
        raise _BadInput("这一帧编不出图（%s）" % ext, 500)
    return buf.tobytes()


def _stage_imencode_png(img, level: int = 1) -> bytes:
    """PNG 编码，压缩级别可指定。合成写回默认 level=1（体积换速度）。

    压缩策略用 **Z_RLE**：PNG 是无损格式，策略只影响压缩后的字节流，解码出来的
    像素逐位不变（实测 np.array_equal ✔）。1080×1920 实测（单线程）：
        level 1 + DEFAULT：编码 68.95ms · 解码 21.21ms · 1.48MB
        level 1 + RLE    ：编码 60.52ms · 解码 22.56ms · 1.35MB
    —— 每帧省 7ms、文件还小 9%（写盘和后面 ffmpeg 读盘都跟着省）。level 0 虽然编得
    更快（35ms），但体积翻 4 倍（6.2MB），实测整体反而更慢（12.32 vs 13.56 帧/秒）。
    """
    import cv2
    params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(level)]
    strat = getattr(cv2, "IMWRITE_PNG_STRATEGY", None)
    rle = getattr(cv2, "IMWRITE_PNG_STRATEGY_RLE", None)
    if strat is not None and rle is not None:
        params += [int(strat), int(rle)]
    ok, buf = cv2.imencode(".png", img, params)
    if not ok:
        raise _BadInput("这一帧编不出 PNG", 500)
    return buf.tobytes()


def _stage_write_atomic(path, data: bytes):
    """先写 .part 再 os.replace：半张图永远不会被当成「这一帧已经好了」。"""
    p = Path(path)
    tmp = Path(str(p) + ".part")
    with open(str(tmp), "wb") as fh:
        fh.write(data)
    os.replace(str(tmp), str(p))


def _stage_read_image(path):
    return _app_mod._imread_any(path)


def _stage_read_frame(path):
    """整帧 PNG 的快速读法：读字节 + imdecode（实测 20ms，_imread_any 要 40ms）。

    _imread_any 里有一堆兜底分支（非 UTF-8 路径、EXIF、多解码器），合成阶段每帧都
    走一遍是净亏。open() 用的是 Python 的 Unicode API，中文路径没问题。
    """
    import cv2
    import numpy as np
    try:
        with open(str(path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _stage_dst_face_mat(meta: dict, size: int = 256):
    """aligned/*.jpg → 合成画布(size) 的仿射矩阵（2×3）；元数据不全返回 None。

    **为什么不是「直接 resize 到 256」**：这条链的「合成画布」是 facelib 那一套
    （compose_frame 用 get_transform_mat 把 pred 反贴回原帧），所以 dst 遮罩必须落在
    同一张画布上 —— 否则 mask_mode 8/7 的遮罩会整体错开，贴回原帧就是边缘多一条
    挖进脸里的缝。

    **2026-12 更正**：_align.align_mat_from_68 现在只拟合 33 点、与 facelib 逐位等价
    （见 app/docs/24-ALIGN-WHOLEFACE.md），重切之后的 aligned 芯片本来就是 facelib
    画布，所以这张补偿矩阵对它们约等于恒等。历史数据（重切前切的）仍然需要它：那时
    两张画布确实不是同一张（1203 张实测中位缩放差 0.36%、旋转 2.5°，尾部到 13%；
    早期在 256 画布上量到的是「残差 6.1~7.2 px、缩放差 5.5%」）。

    这里把 aligned 先按 image_to_face_mat 的逆变换放回原帧坐标，再用法线（facelib）
    变换压到 size —— 一张矩阵、一次重采样，像素仍然来自 aligned 本身，不需要 1080p 原帧。
    """
    import numpy as np

    if not isinstance(meta, dict):
        return None
    lm = meta.get("source_landmarks") or meta.get("landmarks")
    mat_a = meta.get("image_to_face_mat")
    if not lm or len(lm) != 68 or not mat_a:
        return None
    try:
        from .compose import ensure_engine_path
        ensure_engine_path()                    # facelib 在 4.0 引擎根里（_paths.engine_root）
        from facelib import FaceType
        from facelib.LandmarksProcessor import get_transform_mat

        a = np.asarray(get_transform_mat(np.asarray(lm, np.float32), int(size),
                                         FaceType.WHOLE_FACE, scale=1.0), np.float64)
        a = np.vstack([a, [0.0, 0.0, 1.0]]) if a.shape == (2, 3) else a
        b = np.asarray(mat_a, np.float64)
        b = np.vstack([b, [0.0, 0.0, 1.0]]) if b.shape == (2, 3) else b
        m = a @ np.linalg.inv(b)                # aligned -> 原帧 -> 画布
        m = np.asarray(m, np.float64)[:2]
        if m.shape != (2, 3) or not np.all(np.isfinite(m)):
            return None
        return m
    except Exception:                                       # noqa: BLE001
        return None


def _stage_read_dst_face(path, size: int = 256):
    """aligned/*.jpg（DFLJPG）→ (合成画布上的 (size,size) BGR uint8, 失败原因)。

    成功时返回 (img, "")；失败时 (None, 一句话原因) —— 调用方按「跳过并计数」处理，
    **绝不**退回读 1080p 原帧（那正是这一版删掉的东西）。

    读图是裸读字节 + imdecode（DFLJPG 的 APP15 元数据段不影响 JPEG 解码，和
    _stage_read_frame 同一条快路径，绕开 _imread_any 的兜底分支）。
    """
    import cv2
    import numpy as np
    from .._align import read_dfljpg_meta

    try:
        with open(str(path), "rb") as fh:
            data = fh.read()
    except OSError:
        return None, "文件读不出来"
    if not data:
        return None, "文件是空的"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or getattr(img, "ndim", 0) != 3 or int(img.shape[2]) != 3:
        return None, "JPEG 解不出来/坏图"
    if int(img.shape[0]) < 32 or int(img.shape[1]) < 32:
        return None, "图太小（%dx%d）" % (img.shape[1], img.shape[0])
    m = _stage_dst_face_mat(read_dfljpg_meta(path) or {}, size)
    if m is None:
        return None, "DFLJPG 元数据缺 68 点 / image_to_face_mat"
    out = cv2.warpAffine(img, m, (int(size), int(size)), flags=cv2.INTER_CUBIC)
    if out is None or int(out.shape[0]) != size or int(out.shape[1]) != size:
        return None, "映射到画布失败"
    return out, ""


def _stage_clear_dir(dirp, exts=_STAGE_IMG_EXTS, prefix="") -> int:
    """清掉目录里的阶段产物（只删指定后缀，别误伤 meta.json 之类）。"""
    n = 0
    for p in _stage_files(dirp, exts):
        if prefix and not p.name.startswith(prefix):
            continue
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


# ---------------- 进度 / 日志 ----------------
def _stage_log(msg):
    line = str(msg)[:300]
    with _STAGE_LOCK:
        log = _STAGE_JOB.get("log")
        if not isinstance(log, list):
            log = []
            _STAGE_JOB["log"] = log
        log.append(line)
        if len(log) > _STAGE_LOG_MAX:
            del log[:-_STAGE_LOG_MAX]
        chain = bool(_STAGE_JOB.get("chain"))
    if chain:
        _vid_log(line)
    try:
        print("[stage] " + line, flush=True)
    except UnicodeEncodeError:
        pass


def _stage_set(**kw):
    with _STAGE_LOCK:
        _STAGE_JOB.update(kw)


def _stage_progress(stage, pct, msg=None, **kw):
    pct = max(0, min(100, int(round(float(pct or 0)))))
    lo, hi = _STAGE_PCT.get(stage, (0, 100))
    with _STAGE_LOCK:
        _STAGE_JOB["stage"] = stage
        try:
            _STAGE_JOB["stage_index"] = _STAGE_ORDER.index(stage)
        except ValueError:
            _STAGE_JOB["stage_index"] = -1
        _STAGE_JOB["stage_pct"] = pct
        _STAGE_JOB["pct"] = int(round(lo + (hi - lo) * pct / 100.0))
        if msg is not None:
            _STAGE_JOB["msg"] = str(msg)[:300]
        for k, v in kw.items():
            _STAGE_JOB[k] = v
    _stage_mirror()


def _stage_mirror():
    """chain 模式（/api/merge/export 视频分支）把进度同步到老的 _merge_job。"""
    with _STAGE_LOCK:
        if not _STAGE_JOB.get("chain"):
            return
        snap = dict(_STAGE_JOB)
    stage = str(snap.get("stage") or "")
    with _merge_job_lock:
        _merge_job["pct"] = int(snap.get("pct") or 0)
        _merge_job["stage"] = int(_STAGE_NUM.get(stage, 0))
        _merge_job["msg"] = str(snap.get("msg") or "")
        _merge_job["frames_done"] = int(snap.get("frames_done") or 0)
        _merge_job["total"] = int(snap.get("frames_total") or 0)
        st = dict(_merge_job.get("stats") or {})
        st.update({"done": int(snap.get("frames_done") or 0),
                   "swapped": int(snap.get("swapped") or 0),
                   "copied": int(snap.get("copied") or 0),
                   "no_face": int(snap.get("no_face") or 0),
                   "frame_fps": float(snap.get("frame_fps") or 0.0)})
        _merge_job["stats"] = st


def _stage_cancelled() -> bool:
    with _STAGE_LOCK:
        if _STAGE_JOB.get("cancelled"):
            return True
        chain = bool(_STAGE_JOB.get("chain"))
    if chain:
        return _vid_cancelled()
    return False


class _StageCancel(Exception):
    """用户点了取消。和失败分开记 —— 取消不是错误。"""


def _stage_check_cancel():
    if _stage_cancelled():
        raise _StageCancel()


# ---------------- 阶段实现 ----------------
def _stage_quality_detector(quality) -> str:
    """切脸用的检测档：high = RetinaFace(det_10g)，fast = TinyMog（和抽取器同一套档位）。"""
    q = str(quality or "high").strip().lower()
    return "tinymog" if q in ("fast", "tinymog") else "retinaface_10g"


def _align_detector(body: dict):
    """切脸/对齐用哪个检测器 → (规范名, 来源说明)。**界面选的 detector 优先。**

    以前这里只有 quality 一个入口：cfg["det_align"] = _stage_quality_detector(quality)，
    而界面根本不发 quality（app.js 里是 payload.align_quality || "high"），于是
    「检测器」下拉无论选什么都落成 retinaface_10g —— 界面上写着 TinyMog、导出日志里
    却是 retinaface_10g，detect 段 49ms/帧（retina 前向本来就 ~12ms，还要跟 dfm 抢 GPU）。
    这跟 mask_source 那次是同一个形状：**界面选了，但没传**。

    现在：请求体里有 detector（界面「检测器」下拉发的就是它）就以它为准；
    没有才按 quality 档推 —— 老脚本 / 验收工具（quality=fast）一个字都不用改。
    """
    raw = str((body or {}).get("detector") or "").strip()
    if raw:
        return _detector_val(raw), "界面/请求体 detector=%s" % raw
    q = str((body or {}).get("quality") or (body or {}).get("align_quality") or "high")
    return (_stage_quality_detector(q),
            "请求体没给 detector → 按 quality=%s 推" % q.strip().lower())


def _stage_brief(res: dict) -> str:
    bits = []
    for k in ("count", "frames", "swapped", "copied", "no_face", "out", "seconds"):
        v = res.get(k)
        if v not in (None, "", 0):
            bits.append("%s=%s" % (k, v))
    return " · ".join(bits) or "ok"


def _stage_extract_frames_ffmpeg(cfg: dict, frames_dir):
    """快路径：ffmpeg 把视频直接写成 000001.png…（逐帧进度、可取消）。

    为什么不让 python 逐帧 imencode：1080x1920 的 PNG 编码 ~0.3s/帧，一段 3 分钟的
    片子光切帧就要 20 分钟。ffmpeg 的 png 编码器快一个数量级，输出同样是无损 PNG
    （-compression_level 1：快、略大；这是缓存，导出成功后整个目录都会删）。
    失败（退出码非 0）返回 None，调用方退回进程内解码那条路。
    """
    video = Path(cfg["video"])
    ffmpeg = str(_app_mod._ffmpeg_exe())
    pattern = str(Path(frames_dir) / "%06d.png")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-i", str(video), "-map", "0:v:0", "-an",
           # -vsync 0：原样搬每一帧（不按时间戳丢/补帧），帧数 = 解码出来的帧数
           "-vsync", "0", "-start_number", "1", "-compression_level", "1",
           "-progress", "pipe:1", pattern]
    t0 = time.time()
    made = 0
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=".ffmpeg.log")
    err_path = fh.name
    fh.close()
    code = -1
    try:
        with open(str(err_path), "wb") as errfp:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfp,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=getattr(_app_mod, "_NO_WINDOW", 0))
            try:
                for line in proc.stdout:
                    if _stage_cancelled():
                        _kill_proc_tree(proc)
                        raise _StageCancel()
                    line = line.strip()
                    if not line.startswith("frame="):
                        continue
                    try:
                        made = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    el = max(0.001, time.time() - t0)
                    total = max(int(cfg.get("total") or 0), made)
                    _stage_progress("frames", 100.0 * made / total if total else 0,
                                    "切帧 %d/%s 帧 · %.1f 帧/秒"
                                    % (made, cfg.get("total") or "?", made / el),
                                    frames_done=made, frames_total=total,
                                    frame_fps=round(made / el, 2))
            finally:
                code = proc.wait()
    finally:
        tail = ""
        try:
            with open(str(err_path), "rb") as rf:
                tail = rf.read()[-2000:].decode("utf-8", "ignore").strip()
        except OSError:
            pass
        try:
            os.unlink(str(err_path))
        except OSError:
            pass
    if code != 0:
        _stage_log("ffmpeg 抽帧失败（退出码 %s）：%s —— 退回进程内解码"
                   % (code, (tail or "")[-300:]))
        return None
    cfg["_frames_via"] = "ffmpeg"
    return len(_stage_files(frames_dir))


def _stage_do_frames(cfg: dict) -> dict:
    """阶段 1 · 切帧：视频 -> frames/000001.png…（保留原 fps，PNG 无损）。"""
    # 流水线实况：分段链里这一段只有一路 ffmpeg 解码（一条 lane，名字就叫「解码」）
    _STAGE_METER.set_lanes({"frames": ["解码"]})
    cache = Path(cfg["cache"])
    video = Path(cfg["video"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("frames", {})
    frames_dir = _stage_frames_dir(cache)
    aligned_dir = _stage_aligned_dir(cache)
    total_hint = int(cfg.get("total") or 0)
    have = len(_stage_files(frames_dir))

    if (not cfg["force"]) and st.get("done") and have and (not total_hint or have == total_hint):
        _stage_log("切帧：复用已有 %d 帧（不重跑；要重切传 force=true）" % have)
        cfg["total"] = have
        return {"skipped": True, "count": have, "reason": "already-done"}

    if meta.get("frame_swapped") and not cfg["force"]:
        raise _BadInput("frames/ 已经被合成结果覆盖（meta.frame_swapped=true）—— 重切会把原帧"
                        "盖回来，和 meta 里「已合成」的标记打架。要重来请传 force=true："
                        "它会清掉 frames/ + aligned/ 和后面三个阶段的标记。", 409)

    _stage_log("切帧：清空 frames/ 与 aligned/（帧变了，按帧配对的对齐脸和它们的遮罩一起作废）")
    n_old = _stage_clear_dir(frames_dir)
    n_old += _stage_clear_dir(aligned_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    # 帧变了 -> 按帧配对的脸、换脸结果、遮罩一起作废
    for k in ("faces", "pred", "mask", "merge", "encode"):
        meta["stages"][k] = {}
    meta["frame_swapped"] = False
    meta["stages"]["frames"] = {"done": False, "msg": "切帧中"}
    _stage_meta_save(cache, meta)
    if n_old:
        _stage_log("清掉旧产物 %d 个文件" % n_old)

    t0 = time.time()
    idx = 0
    w = int(meta["source"].get("width") or 0)
    h = int(meta["source"].get("height") or 0)

    # 快路径：ffmpeg 直接写 PNG 序列；失败才退回进程内解码 + cv2.imencode
    got = _stage_extract_frames_ffmpeg(cfg, frames_dir)
    if got is not None:
        idx = int(got)
    else:
        idx = _stage_frames_inproc(cfg, frames_dir, meta, w, h)

    if idx == 0:
        raise _BadInput("一帧都没解出来：%s" % video.name, 422)
    sec = round(time.time() - t0, 2)
    meta["source"]["frames"] = idx
    meta["source"]["width"] = w
    meta["source"]["height"] = h
    rng_f = _stage_range(cfg)
    meta["stages"]["frames"] = {
        "done": True, "count": idx, "seconds": sec, "finished_at": time.time(),
        # 切帧**永远整段**（方案 B：帧索引必须完整，成本也极低）；这里记的 range 是
        # 「frames 覆盖了 1..idx」，后五个阶段各自记自己真正处理的那一段。
        "range": {"start": 1, "end": int(idx)},
        "range_key": "1-%d" % int(idx),
        "range_requested": {"start": int(rng_f["start"]), "end": int(rng_f["end"]),
                            "total": int(rng_f["total"])},
        "msg": "切帧完成：%d 帧" % idx,
        # 帧图的真实编码参数（验收要能一眼看到，不再靠"默认值"猜）
        # png 是无损：ffmpeg 走 -compression_level 1（快、略大）；兜底的进程内那条
        # 走 cv2.imencode PNG level 3。两条都是无损，差别只在文件大小/耗时。
        "format": "png", "extractor": str(cfg.get("_frames_via") or "opencv"),
        "png_compression": (1 if cfg.get("_frames_via") == "ffmpeg" else 3),
        "jpeg_quality": 100,
        # jpg 帧走 4:4:4（cv2 5.0 起必须用枚举，见 _stage_imencode）；写不出 4:4:4 的
        # 机器由 _align.jpeg444_probe 启动时 WARN 一次，落成库默认的 4:2:0。
        "jpeg_subsampling": "4:4:4（仅当 image_format=jpg 时才是 jpg）",
        "width": w, "height": h,
    }
    meta["frame_swapped"] = False
    _stage_meta_save(cache, meta)
    cfg["total"] = idx
    _stage_log("切帧完成：%d 帧 · %dx%d · %.2f s" % (idx, w, h, sec))
    return {"count": idx, "seconds": sec, "frames": idx}


def _stage_frames_inproc(cfg: dict, frames_dir, meta, w, h) -> int:
    """兜底：进程内解码 + 逐帧 PNG（ffmpeg 抽帧不可用时走这条）。"""
    import numpy as np

    video = Path(cfg["video"])
    total_hint = int(cfg.get("total") or 0)
    t0 = time.time()
    idx = 0
    try:
        gen, mi = _app_mod._open_video(str(video), [], first=False)
    except Exception as e:                                  # noqa: BLE001
        raise _BadInput("这个视频解不开：%s（%s）" % (video.name, str(e)[:160]), 422)
    try:
        size = (mi or {}).get("size") or (w, h)
        w, h = int(size[0]), int(size[1])
        while True:
            _stage_check_cancel()
            try:
                raw = next(gen)
            except StopIteration:
                break
            except Exception as e:                          # noqa: BLE001
                raise _BadInput("解码到第 %d 帧失败：%s" % (idx + 1, str(e)[:160]), 500)
            frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3)).copy()
            _stage_tick("frames")        # 交接点：解码 → frames/*.png（下一段从这里读）
            _stage_write_atomic(frames_dir / _stage_frame_name(idx),
                                _stage_imencode(frame, ".png"))
            idx += 1
            el = max(0.001, time.time() - t0)
            total = max(total_hint, idx)
            _stage_progress("frames", 100.0 * idx / total if total else 0,
                            "切帧 %d/%s 帧 · %.1f 帧/秒" % (idx, total_hint or "?", idx / el),
                            frames_done=idx, frames_total=total,
                            frame_fps=round(idx / el, 2))
    finally:
        try:
            gen.close()
        except Exception:                                   # noqa: BLE001
            pass
    return idx



# ---------------- 切脸流水线：读帧 ‖ 检测 ‖ 后处理 ----------------
# 为什么要有这一段：切脸原来是「读一帧 → 检测 → 2D106 → 裁剪 → 编码写盘」一条直线。
# 1080x1920 的 PNG 实测（40 帧平均，retinaface_10g）：
#     读帧 19.0ms（47%）· 检测 17.8ms（44%）· 2D106 1.2ms · 裁剪 0.3ms · 编码写盘 0.6ms
# 串起来 40ms/帧 = 25 帧/秒；而这三段用的是完全不同的资源（磁盘+解码 / GPU / CPU），
# 串行等于每一段都在等另外两段。拆成三段重叠之后，单帧成本 ≈ 三段里最慢的那一段。
_FACES_END = object()                       # 流水线哨兵：这一条队列到头了
_FACES_READ_ENV = "DFL_FACES_READ_THREADS"  # 读帧线程数（1..12）
_FACES_QUEUE_ENV = "DFL_FACES_QUEUE"        # 读帧队列深度（在飞的帧数）
_FACES_BATCH_ENV = "DFL_FACES_BATCH"        # 检测批次（0 = 自动）


def _stage_faces_int(env: str, default: int, lo: int, hi: int) -> int:
    try:
        raw = os.environ.get(env)
        v = int(str(raw).strip()) if (raw is not None and str(raw).strip()) else int(default)
    except (TypeError, ValueError):
        v = int(default)
    return max(int(lo), min(int(hi), v))


def _stage_faces_read_threads() -> int:
    """读帧线程数。PNG 解码在 cv2 里会放开 GIL，所以多线程是真的并行。

    实测（1080x1920 PNG，24 帧平均）：单线程 20.2ms / 2 线程 10.8ms / 6 线程 4.96ms /
    8 线程 4.34ms。默认取 6：再往上每加一个线程只省零点几毫秒，却要跟 2D106 抢 CPU。
    """
    cpu = int(os.cpu_count() or 4)
    return _stage_faces_int(_FACES_READ_ENV, min(6, max(2, cpu // 4)), 1, 12)


def _stage_faces_queue(read_threads: int) -> int:
    """读帧队列深度（同时最多几张帧在飞）。一张 1080x1920 BGR 是 6.2MB。"""
    return _stage_faces_int(_FACES_QUEUE_ENV, max(4, read_threads * 2), 2, 32)


def _stage_faces_batchable(det) -> bool:
    """这个检测器能不能真批量。只看它自己声明的能力，不看配置。"""
    return bool(hasattr(det, "detect_batch") and getattr(det, "needs_canvas", False))


def _stage_faces_batch(det) -> int:
    """一次送几张进检测器（1 = 逐帧）。

    能不能批量由**模型的 onnx 输入签名**决定，这里不猜：
      · TinyMog        input [batch_size, 3, 640, 640]        → 动态 batch，可以；
      · RetinaFace_10g input [1, 3, ?, ?]                     → batch 维写死 1
        （只有 H/W 是动态的），硬塞 batch 只会在 ORT 里报 shape 错，所以恒定逐帧。
    引擎侧还有一层上界（TrtSession.max_batch）：超过它 batch 反而不走 CUDA Graph。
    """
    if not _stage_faces_batchable(det):
        return 1
    # 默认逐帧：**批量在这台机器上是负收益**，实测（整条片子 1439 帧，tinymog）——
    #   逐帧   detect 5.06 ms/帧 → 75.9 帧/秒（单帧走 CUDA Graph replay）
    #   批次 4 detect 9.04 ms/帧 → 69.7 帧/秒（batch>1 走 numpy 路径，图重放用不上）
    # 所以批量做成开关（DFL_FACES_BATCH=4 之类），不默认开。
    want = _stage_faces_int(_FACES_BATCH_ENV, 1, 1, 16)
    if want <= 1:
        return 1
    cap = None
    for sess in (getattr(det, "sess_batch", None), getattr(det, "sess", None)):
        if sess is None:
            continue
        try:
            cap = int(sess.max_batch())
        except Exception:                                      # noqa: BLE001
            cap = None
        if cap:
            break
    if cap is not None:
        if cap <= 1:
            return 1
        want = min(want, cap)
    return max(1, want)


def _stage_faces_canvas(frame, det):
    """按检测器自己的 letterbox 口径把整帧缩进 S×S 画布（批量检测用）。

    这与 TinyMogDet.detect() 里那三行逐字一致（128 填充 + INTER_LINEAR），
    所以批量与逐帧的输入像素相同。
    """
    import cv2
    import numpy as np

    S = int(getattr(det, "input_size", 640) or 640)
    H, W = frame.shape[:2]
    scale = min(S / float(W), S / float(H))
    nw = max(1, int(round(W * scale)))
    nh = max(1, int(round(H * scale)))
    canvas = np.full((S, S, 3), 128, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def _stage_faces_detect(det, imgs, thresh):
    """一批帧 → 一批框（与输入等长、同序）。

    batch = 1 或检测器不支持动态 batch 时，逐帧走 det.detect() —— 和优化前**同一条
    调用路径**，结果逐位一致；只有真支持批量时才走 detect_batch，跑挂了就当场退回逐帧。
    """
    if len(imgs) <= 1 or not _stage_faces_batchable(det):
        return [det.detect(im) for im in imgs]
    items = []
    for im in imgs:
        canvas, scale = _stage_faces_canvas(im, det)
        items.append((im, canvas, scale))
    try:
        return det.detect_batch(items)
    except Exception as e:                                     # noqa: BLE001
        _stage_log("切脸：批量检测失败（%s）→ 这一批退回逐帧" % (str(e)[:120],))
        return [det.detect(im) for im in imgs]


def _stage_do_faces(cfg: dict) -> dict:
    """阶段 2 · 切脸：frames/ -> aligned/000001_00.jpg（DFLJPG：original_name + 68 点 + face_mat）。"""
    import cv2
    import numpy as np
    from .._align import (COARSE_SIZE, align_face_68, coarse_align_from_5,
                         landmarks68_in_source, pose_from_kps, write_dfljpg)

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    # 流水线实况：切脸 = 1 个检测 session（读线程不算 lane，它们不产 bit）
    _STAGE_METER.set_lanes({"det": ["det#0"]})
    st = meta["stages"].setdefault("faces", {})
    frames = _stage_files(_stage_frames_dir(cache))
    if not frames:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    aligned_dir = _stage_aligned_dir(cache)
    # 帧范围：只处理 [start, end] 里的帧（帧索引仍是全局的 —— aligned/ 的命名不许挪位）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rows = [i for i, p in enumerate(frames) if _stage_in_range(rng, p.stem)]
    total = len(rows)                     # 范围内的帧数（游标从这里数到 0）
    have = len(_stage_files(aligned_dir, (".jpg", ".jpeg")))

    if ((not cfg["force"]) and not cfg.get("range_changed") and st.get("done") and have
            and int(st.get("frames") or 0) == len(frames)):
        _stage_log("切脸：复用已有 %d 张对齐脸（第 %d..%d 帧，不重跑）"
                   % (have, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": have, "reason": "already-done"}

    if cfg["force"]:
        n = _stage_clear_dir(aligned_dir, (".jpg", ".jpeg"))
        _stage_log("切脸：force=true，清掉旧的 %d 张对齐脸" % n)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    det_name = str(cfg.get("det_align") or "retinaface_10g")
    thresh = float(cfg.get("align_thresh") or 0.5)
    lm = _landmarks_model()          # 和预览/合成同款：纯 CPU onnxruntime 的 2D106
    if lm is None:
        raise _BadInput("2D106 关键点模型不可用（weights/insightface_2d106.onnx）—— "
                        "没有 68 点就不写对齐脸", 500)

    # 游标 **相对范围**：0..total（total = end-start+1），换范围时旧游标（相对旧范围）作废
    cursor = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    faces_total = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("count") or 0)
    with_faces = 0 if (cfg["force"] or cfg.get("range_changed")) else int(st.get("frames_with_faces") or 0)
    skipped = int(st.get("skipped") or 0)
    if cursor:
        _stage_log("切脸：从第 %d 帧续跑（前 %d 帧的对齐脸已经在）"
                   % (int(rng["start"]) + cursor, cursor))
    _stage_log("切脸：检测器 %s（%s）· thresh=%.2f · 68 点仿射 · 帧范围 %d..%d（%d 帧）"
               % (det_name, str(cfg.get("det_align_src") or "cfg.det_align"), thresh,
                  int(rng["start"]), int(rng["end"]), total))
    # 对齐脸是这条链上唯一的 JPEG 产物：质量回显出来（验收/排障要能一眼看到，
    # 别再靠"默认值"猜）。默认 100 + 4:4:4；PNG 帧完全不受这个数影响。
    _stage_log("切脸：对齐脸 JPEG quality=%d · 4:4:4（界面无旋钮，固定质量）"
               % int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY))
    # 检测器先建好（加载要几百毫秒）：流水线里只有一个线程用它，不再每帧走 _detector()
    det = _detector(det_name)

    t0 = time.time()

    def _save_progress(up_to: int, done: bool = False):
        meta["stages"]["faces"] = {
            "done": bool(done), "done_frames": int(up_to), "frames": len(frames),
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": faces_total, "frames_with_faces": with_faces, "skipped": skipped,
            "detector": det_name, "align": "68", "thumb": "2x box",
            "msg": ("切脸完成：%d 张脸 / %d 帧（第 %d..%d 帧）"
                    % (faces_total, total, int(rng["start"]), int(rng["end"]))) if done
                   else ("切脸 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        _stage_meta_save(cache, meta)

    # ---- 流水线：读帧线程池 → 检测线程 → 本线程做 2D106 / 裁剪 / 编码写盘 ----
    # 三段用的资源互不相干（磁盘+解码 / GPU / CPU），串行做等于每段都在等另外两段。
    from concurrent.futures import ThreadPoolExecutor

    nread = _stage_faces_read_threads()
    depth = _stage_faces_queue(nread)
    batch_n = _stage_faces_batch(det)
    ticks = _StageTicks()
    count0 = faces_total
    _stage_log("切脸流水线：读帧 %d 线程 · 队列 %d · 检测批次 %d%s"
               % (nread, depth, batch_n,
                  "" if batch_n > 1 else "（%s 的 onnx 输入 batch 维固定=1 → 逐帧）" % det_name))

    read_q = queue.Queue(maxsize=depth)     # (pos, frame, err)；满了读帧线程就等，内存可控
    det_q = queue.Queue()                   # (pos, frame, boxes) / ("err", None, 异常)
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=nread, thread_name_prefix="faces-read")

    def _read_job(path):
        t = time.perf_counter()
        try:
            return _stage_read_image(path), None
        except Exception as e:                                 # noqa: BLE001
            return None, e
        finally:
            ticks.add("read", time.perf_counter() - t)

    def _q_put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _feeder():
        """按帧序读图：最多 depth 张在飞，读出来的帧立即进 read_q。

        注意「读完了」和「在飞的都取完了」是两件事：帧迭代器耗尽后 pending 里还压着
        最多 depth 张，必须继续取完再收尾（曾经在这里直接 break，末尾 12 帧被静默丢掉，
        阶段照样报 40/40 完成 —— 所以主循环末尾还有一道 consumed 对账）。
        """
        pending = collections.deque()
        it = iter(range(cursor, total))
        exhausted = False
        try:
            while True:
                while (not exhausted) and (not stop.is_set()) and len(pending) < depth:
                    try:
                        pos = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        pending.append((pos, pool.submit(_read_job, frames[rows[pos]])))
                    except RuntimeError:            # 取消时主线程已经 shutdown 了线程池
                        exhausted = True
                        break
                if not pending:
                    break
                pos, fut = pending.popleft()
                try:
                    img, err = fut.result()
                except Exception as e:                         # noqa: BLE001
                    img, err = None, e
                if not _q_put(read_q, (pos, img, err)):
                    return
        except Exception:                       # 收尾阶段的竞态不该把线程栈打到控制台
            pass
        finally:
            _q_put(read_q, _FACES_END)

    def _detect_worker():
        """从 read_q 取帧、按 batch_n 攒一批送检测器，结果按帧序进 det_q。"""
        ended = False
        try:
            while True:
                item = read_q.get()
                if item is _FACES_END:
                    ended = True
                    item = None
                group = []
                if item is not None:
                    if item[2] is not None:
                        det_q.put(item)
                        continue
                    group.append(item)
                while (not ended) and len(group) < batch_n:
                    try:
                        nxt = read_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _FACES_END:
                        ended = True
                        break
                    if nxt[2] is not None:
                        det_q.put(nxt)
                        continue
                    group.append(nxt)
                if group:
                    _stage_check_cancel()
                    imgs = [g[1] for g in group]
                    t = time.perf_counter()
                    try:
                        with _det_run_lock:                     # thresh 是共享检测器上的字段
                            det.thresh = float(thresh)
                            got = _stage_faces_detect(det, imgs, thresh)
                    except _StageCancel:
                        raise
                    except Exception as e:                      # noqa: BLE001
                        raise _BadInput("检测失败：%s" % (str(e)[:200],), 500)
                    finally:
                        ticks.add("detect", time.perf_counter() - t)
                    for g, boxes in zip(group, got):
                        det_q.put((g[0], g[1], boxes))
                if ended:
                    return
        except BaseException as e:                              # noqa: BLE001
            det_q.put(("err", None, e))
        finally:
            det_q.put(_FACES_END)

    feeder_th = threading.Thread(target=_feeder, name="faces-feed", daemon=True)
    det_th = threading.Thread(target=_detect_worker, name="faces-detect", daemon=True)
    feeder_th.start()
    det_th.start()

    consumed = 0
    try:
        while True:
            item = det_q.get()
            if item is _FACES_END:
                break
            pos, frame, extra = item
            if pos == "err":
                raise extra
            if frame is None:
                raise _BadInput("这一帧读不出来：%s" % frames[rows[pos]], 500)
            if isinstance(extra, BaseException):
                raise extra
            boxes = sorted(list(extra or []),
                           key=lambda b: float((b.get("box") or [0, 0, 0, 0])[0]))
            n = 0
            for k, b in enumerate(boxes):
                _stage_check_cancel()
                box = b.get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(v) for v in box]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                side = int(round(max(bw, bh) * 2.0))
                side -= side % 2
                kps = b.get("kps") or []
                if len(kps) < 5 or side < 32:
                    skipped += 1
                    continue
                try:
                    cmat = coarse_align_from_5(kps)
                    coarse = cv2.warpAffine(frame, cmat, (COARSE_SIZE, COARSE_SIZE),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                    t_lm = time.perf_counter()
                    lm68a = lm.landmarks68(coarse)
                    ticks.add("lm68", time.perf_counter() - t_lm)
                    if lm68a is None:
                        raise ValueError("2D106 没有输出")
                    pts = landmarks68_in_source(lm68a, cmat)
                    t_cr = time.perf_counter()
                    crop, meta_f = align_face_68(frame, np.asarray(pts, np.float64),
                                                 box=[int(round(x1)), int(round(y1)),
                                                      int(round(x2)), int(round(y2))],
                                                 size=side)
                    ticks.add("crop", time.perf_counter() - t_cr)
                except Exception as e:                          # noqa: BLE001
                    skipped += 1
                    _stage_log("切脸跳过 %s 第 %d 张：%s" % (frames[rows[pos]].name, k,
                                                          str(e)[:120]))
                    continue
                if crop is None or not meta_f:
                    skipped += 1
                    _why = (meta_f or {}).get("reject_reason") if isinstance(meta_f, dict) else ""
                    if _why:
                        _stage_log("切脸拒收 %s 第 %d 张：%s"
                                   % (frames[rows[pos]].name, k, _why))
                    continue
                # 配对键 / 68 点 / face_mat 三样都写进 DFLJPG 的 APP15
                meta_f.update({
                    "original_name": frames[rows[pos]].name,
                    "source_filename": frames[rows[pos]].stem,
                    "face_mat": meta_f.get("image_to_face_mat"),
                    "landmarks_68": meta_f.get("landmarks"),
                    "source_landmarks_68": meta_f.get("source_landmarks"),
                    "frame_index": int(rows[pos]),
                    "frame_name": frames[rows[pos]].name, "face_index": int(k),
                    "detector": det_name, "side": int(side),
                })
                try:
                    yaw, pitch, roll = pose_from_kps(kps, box)
                    if yaw is not None:
                        meta_f["yaw"], meta_f["pitch"], meta_f["roll"] = yaw, pitch, roll
                except Exception:                               # noqa: BLE001
                    pass
                _stage_tick("det")       # 交接点：切脸 → aligned/*.jpg（下一段从这里读）
                t_en = time.perf_counter()
                write_dfljpg(aligned_dir / _stage_face_name(rows[pos], k), crop, meta_f,
                             quality=int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY),
                             subsampling=0)
                ticks.add("encode", time.perf_counter() - t_en)
                n += 1
            faces_total += n
            if n:
                with_faces += 1
            else:
                # 这一帧一张脸都没送出（检出 0 张 / 全被拒收）→ 进度页旁路线上的那一帧
                _STAGE_METER.note_no_face()
            el = max(0.001, time.time() - t0)
            _stage_progress("faces", 100.0 * (pos + 1) / total,
                            "切脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.2f 帧/秒"
                            % (pos + 1, total, int(rng["start"]), int(rng["end"]),
                               faces_total, (pos + 1 - cursor) / el),
                            frames_done=pos + 1, frames_total=total,
                            frame_fps=round((pos + 1 - cursor) / el, 2))
            consumed += 1
            if (pos + 1) % 20 == 0:
                _save_progress(pos + 1)
    finally:
        stop.set()
        pool.shutdown(wait=False)
        det_th.join(timeout=1.5)
        feeder_th.join(timeout=1.5)

    sec = round(time.time() - t0, 2)
    done_n = max(0, total - cursor)
    # 对账：流水线少喂了帧就等于少切了脸，绝不能把「跑到 30/40」报成「40/40 完成」
    if consumed != done_n:
        raise _BadInput("切脸流水线提前结束：只有 %d/%d 帧走完了检测（产物不完整，"
                        "重跑 force=true）" % (consumed, done_n), 500)
    fps = float(done_n) / max(1e-6, sec)
    # 三段是**并行**的，所以「每帧在各段上花的时间」加起来会大于每帧墙钟：
    #   read     = 读图总耗时 / 读帧线程数（摊到单帧、按并行度折算）
    #   others   = 每帧真的花在那一段上的时间
    #   stage_util = 这一段占用了整段墙钟的百分之多少（>100% = 多线程并行）
    post = ticks.total("lm68") + ticks.total("crop") + ticks.total("encode")
    prof = {"frames": int(done_n), "faces": int(faces_total - count0), "seconds": sec,
            "fps": round(fps, 2),
            "per_frame_ms": {"read": round(1000.0 * ticks.total("read") / max(1, nread)
                                           / max(1, done_n), 2),
                             "detect": ticks.ms("detect", done_n),
                             "lm68": ticks.ms("lm68", done_n),
                             "crop": ticks.ms("crop", done_n),
                             "encode": ticks.ms("encode", done_n)},
            "stage_util": {"read": round(100.0 * ticks.total("read") / max(1e-6, sec), 1),
                           "detect": round(100.0 * ticks.total("detect") / max(1e-6, sec), 1),
                           "post": round(100.0 * post / max(1e-6, sec), 1)},
            "lm68_ms_per_face": ticks.ms("lm68", max(1, faces_total - count0)),
            "pipeline": {"read_threads": int(nread), "queue": int(depth),
                         "detect_batch": int(batch_n),
                         "batchable": bool(_stage_faces_batchable(det)),
                         "overlap": "read || detect || post"}}
    # 谁在拖后腿：按「占用墙钟的比例」比，读帧那一路要除以线程数
    # （read 的 stage_util 是 6 个线程之和，直接比会把多线程误判成瓶颈）
    eff = {"read": prof["stage_util"]["read"] / max(1, int(nread)),
           "detect": prof["stage_util"]["detect"],
           "post": prof["stage_util"]["post"]}
    prof["stage_share"] = {k: round(v, 1) for k, v in eff.items()}
    prof["bottleneck"] = max(eff.items(), key=lambda kv: kv[1])[0] if eff else ""
    prof["per_frame_total_ms"] = round(1000.0 * sec / max(1, done_n), 2)
    prof["per_frame_ms"]["other"] = round(max(
        0.0, prof["per_frame_total_ms"] - sum(prof["per_frame_ms"].values())), 2)
    _save_progress(total, done=True)
    meta["stages"]["faces"]["seconds"] = sec
    meta["stages"]["faces"]["fps"] = round(fps, 2)
    meta["stages"]["faces"]["profile"] = prof
    _stage_meta_save(cache, meta)
    _stage_log("切脸完成：第 %d..%d 帧（%d 帧）· %d 张脸（%d 帧有脸）· 跳过 %d · %.2f s · "
               "%.2f 帧/秒（读 %s / 检测 %s / 2D106 %s / 裁剪 %s / 写盘 %s ms，"
               "读帧 %d 线程 · 批次 %d）"
               % (int(rng["start"]), int(rng["end"]), total, faces_total,
                  with_faces, skipped, sec, fps,
                  prof["per_frame_ms"]["read"], prof["per_frame_ms"]["detect"],
                  prof["per_frame_ms"]["lm68"], prof["per_frame_ms"]["crop"],
                  prof["per_frame_ms"]["encode"], nread, batch_n))
    return {"count": faces_total, "frames": total, "skipped": skipped, "seconds": sec,
            "fps": round(fps, 2), "profile": prof,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}


# ---------------- ③ 换脸（dfm 推理）----------------
# 产物（和 4.0 的 work 目录一一对应，export_pipeline.py:769/846/856 那三个名字）：
#   pred/000001_00.png        换脸后的正脸（无损 PNG，S×S，S = dfm 输入边长）
#   pred/000001_00_prdm.png   dfm 自带的 prd 遮罩（灰色 PNG，0..255）
#   pred/000001_00_dstdm.png  dfm 自带的 dst 遮罩
# 这一阶段**只做推理**：不碰遮罩组合、不碰合成、不写 frames/。
_PRED_LOCK = threading.RLock()
_DFM_SESS = {}          # dfm -> (sess, input_name, size, nchw, providers)
_INFER_CALLS = collections.Counter()   # 各阶段真正跑了几次推理（验收要断言合成阶段 = 0）


def _stage_pred_dir(cache) -> Path:
    return Path(cache) / "pred"


def _stage_mask_dir(cache) -> Path:
    return Path(cache) / "mask"


def _dfm_engine_name(dfm) -> str:
    r"""dfm → TRT 引擎名（稳定、纯 ASCII）。

    为什么必须 ASCII：TensorRT 的 C++ 侧用 stat() 打开 onnx，非 ASCII 路径直接
    INVALID_VALUE（实测 "阿証証証証-416.dfm" 就过不去），所以引擎名里不能留中文。

    编引擎的办法（没有引擎就自动回落 ort-CUDA，功能不受影响，只是慢 16%）：
        :: 中文模型名先把 onnx 拷成 ASCII 再喂给 builder
        copy "workspace\model\<模型>.dfm" "%TEMP%\dfm.onnx"
        runtime\server\Scripts\python.exe app\server\_trt_build.py ^
            --onnx "%TEMP%\dfm.onnx" --out "workspace\_engines\<本函数的名字>_<md5>_sm<SM>_trt<TRT>.bf16.engine" ^
            --name <本函数的名字> --bf16 --profile "1,416,416,3;4,416,416,3;8,416,416,3"
    引擎文件名里的 md5/SM/TRT 三段必须是这台机器的真实值（[TRT] 那行日志会直接
    打印它在找哪个文件名，照抄即可）。
    """
    stem = Path(str(dfm)).stem
    key = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._-") or "dfm"
    return "dfm_" + key[:40]


# ---------------------------------------------------------------
# dfm 的推理后端开关：**默认只用 onnxruntime（CUDA EP / fp32）**
# ---------------------------------------------------------------
# 用户硬规则：**dfm 只走 onnxruntime，不编 TRT 引擎、不降精度。**
# 所以 .engine 对 dfm 是「默认关」的一档加速：磁盘上就算躺着 dfm 的引擎，
# 也要按开关忽略，并且**明确喊一行日志**（禁止静默换后端）。
# 想回到旧行为（原生 TRT 优先）做对照实验，就显式设 DFL_DFM_TRT=1。
# 注意范围：这个开关**只管 dfm**。XSeg / 检测 / 2D106 的 TRT 不受影响
# （它们的入口是 _stage_xseg / _xseg.py，压根不读这两个变量）。
_DFM_TRT_ENV = "DFL_DFM_TRT"          # 0（默认）= 禁；1 = 允许原生 TRT 引擎
_DFM_ENGINE_ENV = "DFL_DFM_ENGINE"    # 老变量，语义同上（只为兼容显式写 1 的人）
_DFM_TRT_ON = ("1", "true", "yes", "on", "trt", "engine")


def _dfm_trt_enabled() -> bool:
    """dfm 能不能用 .engine —— **默认不能**（用户要求 ONNX only）。"""
    for k in (_DFM_TRT_ENV, _DFM_ENGINE_ENV):
        v = str(os.environ.get(k) or "").strip().lower()
        if v in _DFM_TRT_ON:
            return True                          # 显式要求，尊重
        if v:
            return False                         # 显式关（"0"）压过后面那个变量
    return False                                 # 谁都没设 → 关（ONNX only）


def _dfm_trt_why() -> str:
    """一句话说明「现在是按哪个变量关的」，给日志用。"""
    for k in (_DFM_TRT_ENV, _DFM_ENGINE_ENV):
        v = str(os.environ.get(k) or "").strip()
        if v:
            return "%s=%s" % (k, v)
    return "%s=0（默认）" % _DFM_TRT_ENV


def _dfm_mb(n) -> str:
    try:
        return "%.1fMB" % (float(n) / (1024.0 * 1024.0))
    except Exception:                                   # noqa: BLE001
        return "?MB"


def _dfm_engines_on_disk(ws, name: str) -> list:
    """磁盘上属于这个 dfm 的 .engine（不校验 md5/SM，只为「发现了但忽略」那行日志）。

    按体积从大到小：bf16 引擎几百 MB，int8 的几十 MB，用户看到的最大那个才是重点。
    """
    out = []
    try:
        d = Path(ws) / "_engines"
        if not d.is_dir():
            return out
        out = list(d.glob("%s_*.engine" % name))
        out.sort(key=lambda p: p.stat().st_size, reverse=True)
    except Exception:                                   # noqa: BLE001
        return []
    return out


def _dfm_session(dfm: str):
    """常驻的 dfm session。**默认 torch-CUDA + bf16**（2026-11 用户拍板）。

    判定顺序：

        DFL_DFM_BACKEND=torch（**默认**）      → torch eager（精度 DFL_DFM_DTYPE，默认 bf16）
            起不来 → 喊一行 → 回落到下面那条（绝不静默换后端）
        DFL_DFM_BACKEND=onnx（回退）           → TRT 引擎（要显式 DFL_DFM_TRT=1）/ ort-CUDA
        显存装不下 + DFL_ALLOW_CPU_FALLBACK=1  → CPU EP（慢 ~17 倍，日志写死；
                                                torch 在这条路上没有意义，仍走 ort-CPU）

    为什么把默认从 onnx 换成 torch：实测（app/tools/bench/DFM_TORCH_BF16_REPORT.md）
    torch+bf16 在 batch 8 上是 **20.0 ms/张（416）/ 22.5（384）**，onnx fp32 是 39.5 / 32.6；
    成片逐像素对拍 mean 0.009 LSB、max 5 LSB、>4 LSB 占 0.00%。

    历史背景（留给想看数的人）：416 dfm 上 ort-CUDA 35.9 ms/张（27.9 张/秒）、
    原生 TRT(bf16) 30.9 ms/张（32.3 张/秒）—— 那 ~16% 就是换掉 ONNX 的代价，
    按用户口径**不换**：这条链上 dfm 永远是 onnxruntime，日志里 provider 列表为准。

    唯一的硬要求是**不能静默退到 CPU**：CPU 单张 527 ms（1.9 张/秒），而 ORT 在
    CUDA provider 加载失败时只会打一行 warning 就继续跑，所以这里自己把
    providers 拿出来核对、打醒目日志。

    session 按路径常驻，切模型才重建。
    """
    key = str(dfm)
    if _vram_oom_inject_on():
        # 验收专用（DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1）：**在真的加载之前**
        # 抛一次「CUDA out of memory」，把「强制尝试撞 OOM → 回滚 → 还能再试」这条路
        # 变成可复现的。为什么需要它：WDDM 会把别的进程的显存换页出去，本机实测
        # 「空闲 600MB + force=1」竟然还能跑完 12 帧（慢，但不 OOM）——真 OOM 只能
        # 靠把卡压到连提交都失败，那在 32GB 内存的机器上做不到。
        # 生产路径上这两个开关都没人设，一个字节都不走。
        raise RuntimeError("CUDA out of memory. Tried to allocate 2079 MiB"
                           " (DFL_VRAM_OOM_INJECT=1 注入的 OOM，验收用)")
    cpu = _vram_cpu_on()            # 显存不够 + DFL_ALLOW_CPU_FALLBACK=1 → 整条链走 CPU EP
    # ---- 后端选择：默认 torch（精度 bf16w）；DFL_DFM_BACKEND=onnx 是回退档 ----
    _bkind, _bdtype, _bwhy = _dfm_backend_sel()
    use_torch = (_bkind == "torch" and not cpu)
    if _bkind == "torch" and cpu:
        _stage_log("换脸：DFL_DFM_BACKEND=torch 但这条链已强制 CPU（DFL_ALLOW_CPU_FALLBACK=1）"
                   " → 仍走 onnxruntime CPU EP（torch 的 CPU 前向比 ORT CPU 更慢，没有意义）")
    ckey = key + ("|torch:" + _bdtype if use_torch else ("|cpu" if cpu else ""))
    with _PRED_LOCK:
        hit = _DFM_SESS.get(ckey)
        if hit is not None:
            return hit
        # torch 后端根本不 import onnxruntime（那 0.3~1s 和一堆 DLL 都不需要；
        # 这条路径要能在没装 onnxruntime 的机器上跑）。
        ort = None
        if not use_torch:
            import onnxruntime as ort
            try:
                _ensure_ort_dll_dirs()
            except Exception:                               # noqa: BLE001
                pass
        # 量这一份会话到底吃掉多少显存（权重 + 这个进程的 CUDA context + ORT arena）。
        # 为什么要量：arena 会长（384 模型：建会话 +2.3GB，之后每批 8 张的前向再涨 ~1GB），
        # 估出来的数偏小 —— 而「显存去向」那一栏必须是真数，不能是拍的。
        _own_f0 = _vram_measure_begin()
        ws = Path(WORKSPACE)
        info = {}
        sess = None
        backend = "ort-cuda"
        if use_torch:
            # torch 这一档不走 TRT / 不看 .engine：图已经编成 nn.Module 了
            info = {"reason": _bwhy}
        elif cpu:
            info = {"reason": "DFL_ALLOW_CPU_FALLBACK=1（显存预算装不下，主动回落 CPU）"}
        elif not _dfm_trt_enabled():
            # 用户硬规则：**dfm 只用 onnxruntime**。引擎在磁盘上也不加载 —— 但必须喊出来，
            # 否则「A/B 报告里 416 其实是 TRT 的数字」这种事下次还会悄悄发生。
            engs = _dfm_engines_on_disk(ws, _dfm_engine_name(key))
            if engs:
                more = ("（另有 %d 份同类引擎一起忽略）" % (len(engs) - 1)) if len(engs) > 1 else ""
                _stage_log("[dfm] 发现 TRT 引擎 %s（%s）但按 %s忽略，使用 onnxruntime CUDA%s"
                           % (engs[0].name, _dfm_mb(engs[0].stat().st_size),
                              _dfm_trt_why(), more))
            else:
                _stage_log("[dfm] 按 %s走 ONNX only（磁盘上没有 dfm 的 .engine）"
                           % _dfm_trt_why())
            info = {"reason": "%s —— 按用户要求 dfm 只走 onnxruntime，"
                              "磁盘上有 .engine 也不加载" % _dfm_trt_why()}
            sess = None
        else:
            try:
                from .. import _engines as _E
                sess = _E.try_engine(ws, _dfm_engine_name(key), key,
                                     precision=("bf16",), use_graph=False, info=info)
            except Exception as e:                      # noqa: BLE001
                info = {"reason": "try_engine 抛异常：%r" % (e,)}
                sess = None
        if use_torch:
            # ---- torch 后端（见 backend_models/dfm_torch.py）----
            try:
                from ..backend_models import dfm_torch as _DFMT
                sess = _DFMT.load(key, dtype=_bdtype)
                backend = "torch-" + sess.dtype
                shape = list(sess.get_inputs()[0].shape)
                name = sess.input_name
                provs = list(sess.get_providers())
                _v = sess.vram_mb()
                _stage_log("换脸：dfm 走 torch(%s) · 精度=%s（allow_tf32=%s）· %d×%d · %s ·"
                           " 权重 %d 张量/%s · 预热单张 %.2f ms · 卡上 %s MB · %s · 请求=%s"
                           % (sess.device, sess.dtype,
                              "开" if _bdtype != "fp32" else "关",
                              int(sess.size), int(sess.size),
                              "NHWC" if not sess.nchw else "NCHW",
                              len(sess.store._buffers),
                              "%dMB" % int(sess.graph.weights_mb),
                              float(getattr(sess, "one_ms", 0.0)),
                              _v.get("alloc_mb", "?"), _bwhy, _bdtype))
            except Exception as _e:                     # noqa: BLE001
                # 起不来就明确回落到 onnx —— 但**必须喊出来**，不许静默换后端
                _stage_log("换脸：torch 后端起不来（%s: %s）→ 回落到 onnxruntime CUDA"
                           % (type(_e).__name__, str(_e)[:200]))
                use_torch = False
                sess = None
                backend = "ort-cuda"
                ckey = key + ("|cpu" if cpu else "")     # 别把 onnx 会话挂在 torch 的 key 上
                if ort is None:
                    import onnxruntime as ort
                    try:
                        _ensure_ort_dll_dirs()
                    except Exception:                   # noqa: BLE001
                        pass
        if use_torch:
            pass                  # 上面那一支已经把 sess / shape / name / provs 都备齐了
        elif cpu:
            backend = "ort-cpu-forced"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(key, sess_options=so,
                                        providers=["CPUExecutionProvider"])
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = list(sess.get_providers())
            _stage_log("换脸：★ 当前在 CPU 上跑 ★ dfm=%s 走 CPU EP —— 显存不够且开了 "
                       "DFL_ALLOW_CPU_FALLBACK=1（416 实测 527 ms/张，GPU 约 31 ms/张，"
                       "慢 %.0f 倍）。关掉这个开关就改成明确拒绝。"
                       % (Path(key).name, _VRAM_CPU_SLOW))
        elif sess is not None:
            backend = "trt"
            sess.precision = info.get("precision") or ""
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = ["TensorrtExecutionProvider"]
            _stage_log("换脸：dfm 走原生 TRT(%s) 引擎 %s"
                       % (info.get("precision") or "?", Path(str(info.get("engine") or "")).name))
        else:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(key, sess_options=so,
                                        providers=["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
            shape = list(sess.get_inputs()[0].shape)
            name = sess.get_inputs()[0].name
            provs = list(sess.get_providers())
            if provs[:1] == ["CPUExecutionProvider"]:
                backend = "ort-cpu"
                _stage_log("换脸：★ 警告 ★ dfm 掉到 CPU 了（%s）—— 这张卡上会慢 ~15 倍"
                           "（416 实测 527 ms/张）。查 cublasLt64_12.dll / cudnn64_9.dll "
                           "是否在 PATH 上（torch/lib）；dfm 的 TRT 只能靠"
                           " DFL_DFM_TRT=1 显式打开（默认 ONNX only）。"
                           % (info.get("reason") or "CUDA provider 没加载上"))
            else:
                # 验收要的证据就在这一行：provider 列表 = 真后端，不是"听说用了 GPU"
                _stage_log("换脸：dfm 走 onnxruntime(%s) · providers=%s · 后端=ort-cuda（%s）"
                           % (provs[0], list(provs),
                              str(info.get("reason") or "未查询")[:140]))
        nchw = len(shape) >= 4 and shape[1] in (1, 3)
        try:
            size = int(shape[2] if nchw else shape[1])
        except Exception:                                   # noqa: BLE001
            size = 256
        cap = _sess_batch_cap(sess)
        ent = (sess, name, size, nchw, provs, cap, backend)
        _DFM_SESS[ckey] = ent
        _vram_measure_end("dfm:" + Path(key).name, _own_f0)
        _stage_log("换脸：dfm 已加载 %s · %d×%d · %s · 单次最多 %d 张"
                   % (Path(key).name, size, size, backend, cap))
        return ent


def _mask2d(t, size):
    """dfm 的遮罩输出 -> (size, size) float32 0..1（和 preview worker 里同名函数一致）。"""
    import cv2
    import numpy as np
    m = np.squeeze(t)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = np.asarray(m, np.float32)
    if m.shape[:2] != (int(size), int(size)):
        m = cv2.resize(m, (int(size), int(size)), interpolation=cv2.INTER_CUBIC)
    return m


def _dfm_run_batch(ent, batch_in):
    """一次前向跑 N 张（N = batch_in.shape[0]），返回 [(face, prd_mask, dst_mask)]。

    batch_in: (N,S,S,3) float32 0..1（NHWC；NCHW 的 dfm 在调用方转置）。
    dfm 的 batch 维是动态的（['batch',416,416,3]），所以批处理不用重新导出模型；
    CUDA graph 那种「按 batch 捕获」的坑在 ORT 这边不存在（我们不开 cuda_graph），
    真要开 TRT 引擎就得按 batch 分桶 —— 见 _engines.try_engine 的 graph_max_batch。
    """
    import cv2
    import numpy as np
    sess, name, size, nchw = ent[0], ent[1], ent[2], ent[3]
    cap = int(ent[5] or 1) if len(ent) > 5 else 1
    n_in = int(batch_in.shape[0])
    if cap > 1 and n_in > cap:
        # 一次塞不下（原生 TRT 的优化档位可能只有 1）—— 按上限切开，结果顺序不变
        out = []
        for _i in range(0, n_in, cap):
            out.extend(_dfm_run_batch(ent, batch_in[_i:_i + cap]))
        return out
    x = batch_in
    if nchw:
        x = np.transpose(x, (0, 3, 1, 2))
    _INFER_CALLS["dfm"] += int(x.shape[0])
    outs = sess.run(None, {name: np.ascontiguousarray(x, np.float32)})
    n = int(x.shape[0])
    faces, prds, dsts = [], [], []
    for i in range(n):
        face = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim >= 4 and a.shape[-1] == 3:
                face = a[i] if a.ndim == 4 else a[i][0]
                break
        if face is None:
            a = np.asarray(outs[0])
            face = a[i]
            if face.ndim == 3 and face.shape[-1] == 1:
                face = np.repeat(face, 3, axis=-1)
        face = np.asarray(face, np.float32)
        if face.ndim > 3:
            face = face[0]
        if nchw and face.shape[0] == 3:
            face = np.transpose(face, (1, 2, 0))
        if face.shape[:2] != (size, size):
            face = cv2.resize(face, (size, size), interpolation=cv2.INTER_LANCZOS4)
        prd = dst = None
        if len(outs) >= 3:
            prd = _mask2d(np.asarray(outs[0])[i], size)
            dst = _mask2d(np.asarray(outs[2])[i], size)
        faces.append(np.clip(face, 0, 1))
        prds.append(prd)
        dsts.append(dst)
    return list(zip(faces, prds, dsts))


def _stage_face_rows(aligned) -> list:
    """aligned/ 里的对齐脸按帧归组：[("000001", [路径…]), …]（帧序 = 文件名字典序）。"""
    out, order = {}, []
    for p in aligned:
        key = p.stem.split("_")[0]
        if key not in out:
            out[key] = []
            order.append(key)
        out[key].append(p)
    return [(k, out[k]) for k in order]


def _stage_frame_path(frames_dir, key: str):
    d = Path(frames_dir)
    for ext in _STAGE_IMG_EXTS:
        cand = d / (key + ext)
        if cand.is_file():
            return cand
    return None


def _stage_pred_assign(cfg: dict, frame, metas: list) -> list:
    """这一帧每张脸用哪个 dfm（多模型 = 人脸库 embedding 归组；没有库就全用同一个）。

    和 13-MERGE-SEMANTICS 的口径一致：
      · 没有人脸库 -> 左栏那个 dfm 换全部脸；
      · 有人脸库   -> 按 embedding 归组，**不匹配的脸不给 pred**（⑤ 保持原像素，
        绝不兜给任何模型）。
    返回 [dfm 路径 or ""]（下标 = aligned 里的脸序号）。
    """
    single = str(cfg.get("dfm") or "")
    if not cfg.get("use_library") or not cfg.get("library"):
        return [single] * len(metas)
    from .. import _face_group as _fg
    from .. import _faces as _store
    from .._align import align_face_68

    cache = cfg.setdefault("_pred_plan_cache", {})
    if "profiles" not in cache:
        library = _auto_library(cfg.get("plan_payload") or {})
        index = None
        profiles = []
        if library is not None:
            try:
                index = _store.load_index(library)
                if any(str(g.get("dfm") or "").strip() for g in index["groups"]):
                    profiles = _fg.group_profiles(library, index,
                                                  compute_limit=_AUTO_EMB_LIMIT)
            except Exception:                               # noqa: BLE001
                profiles = []
        cache["profiles"] = profiles or []
        if not cache["profiles"]:
            cache["off"] = True                             # 库不可用 -> 退回单模型
    if cache.get("off"):
        return [single] * len(metas)
    threshold = cfg.get("face_threshold")
    threshold = float(_fg.DEFAULT_THRESHOLD if threshold is None else threshold)
    out = []
    for m in metas:
        dfm = ""
        try:
            lm = np.asarray(m.get("lm"), np.float32)
            box = m.get("box")
            crop, _mf = align_face_68(frame, lm, box, 512)
            vec = _fg.embed_crop(crop) if crop is not None else None
            best = _fg.match_vector(vec, cache["profiles"], threshold) if vec is not None else None
            if best and best.get("matched") and best.get("dfm"):
                dfm = _dfm_by_name(best["dfm"]) or ""
        except Exception:                                   # noqa: BLE001
            dfm = ""
        out.append(dfm or "")
    return out


def _stage_do_pred(cfg: dict) -> dict:
    """阶段 3 · 换脸：frames/ + aligned/ -> pred/（换脸正脸 + dfm 自带两张遮罩）。

    语义照 4.0 export_pipeline.py:769（fo = warpAffine(frame, get_transform_mat(lm,
    swap_sz, WHOLE_FACE)) → dfm 前向）。

    **吞吐**（本机实测，1080×1920、每帧 1 张脸、app/tools/bench/stage_perf_profile.py）：
        read 28.5ms/张（4 线程摊薄后，整帧 PNG 解码 17.7ms 是底）· prep 9ms/张 ·
        infer 37.6ms(ort-CUDA) / 30.4ms(原生 TRT bf16) ← 硬底 · write 26ms/张（2 线程）
    墙钟 ≈ max(三段)：25.6 张/秒(ort-CUDA) / 31.8 张/秒(原生 TRT)。
    所以这一段**已经贴着推理跑了**，再优化 CPU 侧不会让吞吐变高，只会让它更抗噪。
    老实现是一条串行链（读帧 → warp → 攒 batch → 前向 → 写 3 张 PNG），
    一张脸 = 22(读) + 8(float 转换) + 36(前向) + 21(三张 PNG 编码) ≈ 100ms，实测 9.6 张/秒，
    而且 batch 从 1 加到 16 吞吐几乎不动 —— 说明瓶颈根本不在推理。
    现在拆成三段，用队列连起来（谁慢谁是瓶颈，其它两段被完全藏住）：
        读线程 ×N    PNG 解码 + 整帧 → float32 + 按 68 点 warp（1080p ≈ 30ms/帧，
                      4 线程实测摊到 7.3ms/帧）
        推理线程     攒够 batch 一次前向（416 dfm ≈ 36ms/张，GPU 99% 占用，是硬底）
        写线程 ×M    PNG 编码 + 原子落盘（416×416 ×3 ≈ 21ms/张）
    墙钟 ≈ max(三段)，而不是三段之和。

    已经踩过的两个坑（别再踩）：
      · 读线程以前先把 1080p 整帧 astype(float32)/255 再 warp（实测多花 25ms/帧），
        而 4.0 是「uint8 整帧 warp → 再 /255」，两边只差舍入 —— 现在和 4.0 同序；
      · 进度里的「帧/秒」以前拿续跑游标 k 算，而 k 为了不留空洞会滞后 readers 倍：
        25.4 张/秒被显示成 6.3 帧/秒，用户报的「5.8~6.2 帧/秒」就是它，不是真变慢。

    其它三处通用优化（都不是针对某台机器）：
      · batch 按可用显存实测自动选（_pred_pick_batch），OOM / 明显变慢自动退档 8→4→2→1；
      · 输入缓冲按 batch 预分配，每批只做一次 memcpy，不再是 np.stack(...).astype(float32)
        的「分配 + 转换 + 再拷一遍」；
      · 人脸库归组（embedding）仍然在主线程做 —— 它要用整帧，且非默认路径。

    多模型：有人脸库时按 embedding 归组，每组用各自的 dfm；不匹配的脸**不产出 pred**
    （⑤ 保持原像素）。游标 = meta.stages.pred.done_frames（帧数），取消可续。
    """
    import queue
    import sys as _sys
    import cv2
    import numpy as np
    from .compose import ensure_engine_path
    ensure_engine_path()
    from facelib import FaceType
    from facelib.LandmarksProcessor import get_transform_mat
    from .._align import read_dfljpg_meta

    # 流水线实况：换脸 = 1 个 dfm session（一条 lane；写盘线程不产 bit，不算 lane）
    _STAGE_METER.set_lanes({"dfm": ["dfm#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("pred", {})
    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再换脸", 409)
    frames_dir = _stage_frames_dir(cache)
    # 帧范围：只对 [start, end] 里的帧做推理（groups 的下标/游标都按过滤后的列表数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    groups_all = _stage_face_rows(aligned)
    groups = [(k, ps) for k, ps in groups_all if _stage_in_range(rng, k)]
    total = len(groups)
    faces_in_range = sum(len(ps) for _k, ps in groups)
    pred_dir = _stage_pred_dir(cache)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dfm = str(cfg.get("dfm") or "")
    if not dfm or not Path(dfm).is_file():
        raise _BadInput("没有 .dfm 模型（换脸阶段要它）：%s" % (dfm or "空"), 409)

    ent0 = _dfm_session(dfm)                       # 先加载：batch 要按它的显存开销算
    batch, bmeta = _pred_pick_batch(cfg, ent0)
    size0 = int(ent0[2])
    # 并行度：读线程按 CPU 核数给（PNG 解码在 OpenCV 里是放 GIL 的），写线程固定 2 个
    readers = max(1, int(cfg.get("pred_readers")
                         or min(4, max(1, (os.cpu_count() or 4) // 3))))
    writers = max(1, int(cfg.get("pred_writers") or 2))
    use_lib = bool(cfg.get("use_library") and cfg.get("library"))

    # 游标 **相对范围**：0..total（total = 范围内有脸的帧数）
    rchg = bool(cfg.get("range_changed"))
    cursor = 0 if (cfg["force"] or rchg) else max(0, min(int(st.get("done_frames") or 0), total))
    if (not cfg["force"]) and not rchg and st.get("done") and cursor >= total and total:
        _stage_log("换脸：复用已有 %d 帧的 pred/（第 %d..%d 帧，不重跑）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}
    if cursor:
        _stage_log("换脸：从范围内第 %d 帧续跑（前 %d 帧的 pred/ 已经在）" % (cursor + 1, cursor))
    _stage_log("换脸：dfm=%s · batch=%d(%s) · 帧范围 %d..%d（%d 帧 / %d 张脸）· "
               "读 %d 线程 / 写 %d 线程%s"
               % (Path(dfm).name, batch, bmeta.get("source"),
                  int(rng["start"]), int(rng["end"]), total, faces_in_range,
                  readers, writers, " · 人脸库归组" if use_lib else ""))
    if bmeta.get("source") == "auto":
        _stage_log("换脸：batch 自动选择 %d（可用显存 %s / 共 %s MB · 单张约 %s MB · "
                   "留 %d%% 余量 · 最快 %s ms/张）"
                   % (batch, bmeta.get("free_mb"), bmeta.get("total_mb"),
                      bmeta.get("per_item_mb"), int(_PRED_VRAM_SAFETY * 100),
                      bmeta.get("best_per_face_ms")))

    t0 = time.time()
    # force 重跑时从 0 数起：以前会把上一轮的 count 累加（90 张脸跑完显示 180），
    # 于是 faces_per_sec 一直偏大 —— 吞吐对比全是错的数
    done_faces = 0 if (cfg.get("force") or rchg) else int(st.get("count") or 0)
    if cfg.get("force") or rchg:
        st["count"] = 0
    unmatched = int(st.get("unmatched") or 0)
    failed = int(st.get("failed") or 0)
    sizes = {}
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "frames": 0}
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "total": int(total), "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": done_faces, "unmatched": unmatched, "failed": failed,
            "faces_total": int(faces_in_range), "dfm": str(dfm), "model": Path(dfm).name,
            "batch": int(batch), "batch_source": str(bmeta.get("source") or ""),
            "batch_meta": dict(bmeta), "sizes": dict(sizes), "pred_dir": str(pred_dir),
            "readers": int(readers), "writers": int(writers),
            "resumed_from": int(cursor),
            "msg": ("换脸完成：%d 张脸 / %d 帧（第 %d..%d 帧 · 未匹配 %d）"
                    % (done_faces, total, int(rng["start"]), int(rng["end"]), unmatched))
                   if done else ("换脸 %d/%d 帧（第 %d..%d 帧）"
                                 % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["pred"] = node
        _stage_meta_save(cache, meta)

    # ---- 队列与共享状态 ----
    q_frames = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, batch * 2))
    stop = threading.Event()
    boss = {"exc": None}
    pre = np.empty((batch, size0, size0, 3), np.float32)      # 预分配输入缓冲（NHWC）
    pend, completed, plock2 = {}, set(), threading.Lock()
    commit = {"n": cursor}

    def _commit_upto():
        with plock2:
            n = commit["n"]
            while n in completed:
                completed.discard(n)
                n += 1
            commit["n"] = n
            return n

    def _pend_add(gi: int):
        with plock2:
            pend[gi] = int(pend.get(gi) or 0) + 1

    def _pend_done(gi: int):
        nonlocal failed
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    # ---- 写线程：PNG 编码 + 原子落盘 ----
    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, face, prd, dst = item
            _stage_tick("dfm")       # 交接点：dfm 换脸 → pred/*.png
            tw = time.perf_counter()
            try:
                _stage_write_atomic(
                    pred_dir / (key + ".png"),
                    _stage_imencode_png(np.clip(face * 255.0, 0, 255).astype(np.uint8), 1))
                if prd is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_prdm.png"),
                        _stage_imencode_png(np.clip(prd * 255.0, 0, 255).astype(np.uint8), 1))
                if dst is not None:
                    _stage_write_atomic(
                        pred_dir / (key + "_dstdm.png"),
                        _stage_imencode_png(np.clip(dst * 255.0, 0, 255).astype(np.uint8), 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("换脸写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _put_frame(item):
        """带超时的 put：主线程没了（取消/异常）也别把读线程吊死在队列上。"""
        while not stop.is_set():
            try:
                q_frames.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ---- 读线程：解码 + warp（不碰推理，也不碰写盘）----
    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                key, paths = groups[gi]
                fp = _stage_frame_path(frames_dir, key)
                if fp is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                tr = time.perf_counter()
                frame = _stage_read_frame(fp)
                _bump("read_ms", time.perf_counter() - tr)
                if frame is None:
                    _put_frame((gi, key, False, None, None, None))
                    continue
                crops = None
                if not use_lib:
                    tp = time.perf_counter()
                    crops = []
                    for ap in paths:
                        m = read_dfljpg_meta(ap) or {}
                        lm = m.get("source_landmarks") or m.get("landmarks")
                        if not lm or len(lm) != 68:
                            crops.append((ap.stem, None, None))
                            continue
                        try:
                            sz = size0
                            fmat = get_transform_mat(np.asarray(lm, np.float32), sz,
                                                     FaceType.WHOLE_FACE, scale=1.0)
                            # 顺序和 4.0 export_pipeline:720-721 完全一致：**uint8 整帧
                            # warp → 再 /255**。以前是先把 1080p 整帧转 float32 再 warp，
                            # 实测这多花的 25ms/帧（astype 11.8 + 浮点 warp 13.4 vs
                            # uint8 warp 0.18）只是把同样的双线性插值算在 float 上，
                            # 结果只差舍入 —— 却把 4 个读线程全拖成瓶颈。
                            crop = cv2.warpAffine(frame, fmat, (sz, sz), flags=cv2.INTER_CUBIC)
                            crops.append((ap.stem,
                                          np.clip(crop.astype(np.float32) / 255.0, 0, 1), None))
                        except Exception as e:              # noqa: BLE001
                            crops.append((ap.stem, None, str(e)[:100]))
                    _bump("prep_ms", time.perf_counter() - tp)
                # ok=True 才是「这一帧能处理」；frame 只在人脸库归组时才需要带回主线程
                _put_frame((gi, key, True, frame if use_lib else None, None, crops))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_frames.put((None, None, False, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    # ---- 一批：拷进预分配缓冲 → 一次前向 → 交给写线程 ----
    jobs = {}          # dfm -> ([gi], [keys], [crops])
    # best_ms 用阶梯探测的最好成绩起手：否则第一批（可能已经偏大）会被当成基准，
    # 后面的退档守卫就永远不会触发
    stat = {"best_ms": bmeta.get("best_per_face_ms") or None, "shrunk": 0}

    def _flush(use_dfm: str, force: bool = False):
        nonlocal batch, done_faces, failed
        bucket = jobs.get(use_dfm)
        if not bucket:
            return
        gis, keys, crops = bucket
        while keys and (len(keys) >= batch or (force and keys)):
            n = min(len(keys), batch)
            ent = _dfm_session(use_dfm)
            size = int(ent[2])
            sizes[Path(use_dfm).name] = size
            if size0 == size and pre.shape[0] >= n:
                arr = pre[:n]
            else:
                arr = np.empty((n, size, size, 3), np.float32)
            for i in range(n):
                arr[i] = crops[i]
            ti = time.perf_counter()
            try:
                outs = _dfm_run_batch(ent, arr)
            except Exception as e:                          # noqa: BLE001
                if not (_is_oom(e) and n > 1):
                    raise
                step = max(1, n // 2)
                _stage_log("换脸：batch=%d 显存不够（%s）→ 退到 %d 再试"
                           % (n, str(e)[:80], step))
                batch = max(1, min(batch, step))
                stat["shrunk"] += 1
                outs = []
                for i in range(0, n, step):
                    outs.extend(_dfm_run_batch(ent, arr[i:i + step]))
            dt = time.perf_counter() - ti
            per_face = dt * 1000.0 / max(1, n)
            _bump("infer_ms", dt, n)
            # 变慢守卫：比历史最好成绩慢 60% 说明 batch 太大（显存抖动 / 换出），主动退档
            if stat["best_ms"] is None or per_face < stat["best_ms"]:
                stat["best_ms"] = per_face
            elif n > 1 and per_face > stat["best_ms"] * 1.6 and stat["shrunk"] < 4:
                half = max(1, n // 2)
                _stage_log("换脸：batch=%d 时每张 %.1f ms（最好 %.1f ms）→ 退到 %d"
                           % (n, per_face, stat["best_ms"], min(batch, half)))
                batch = max(1, min(batch, half))
                stat["shrunk"] += 1
            for i in range(n):
                q_write.put((gis[i], keys[i], outs[i][0], outs[i][1], outs[i][2]))
                done_faces += 1
            del gis[:n], keys[:n], crops[:n]

    def _submit(gi: int, stem: str, use_dfm: str, crop):
        _pend_add(gi)
        bucket = jobs.setdefault(use_dfm, ([], [], []))
        bucket[0].append(gi)
        bucket[1].append(stem)
        bucket[2].append(crop)
        _flush(use_dfm)

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="pred-w%d" % i)
        t.start()
        threads.append(t)
    hi_all = total
    for r in range(readers):
        lo = cursor + (hi_all - cursor) * r // readers
        hi = cursor + (hi_all - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="pred-r%d" % r)
        t.start()
        threads.append(t)

    k = cursor
    cancelled = {"flag": False}
    got = 0            # 已经处理完的帧数（显示用）。k 是**续跑游标**，它会为了
                       # 「不留空洞」而滞后（4 个读线程各跑一段，游标只认连续前缀），
                       # 以前拿 k 算帧/秒 → 正好少报 readers 倍（实测 6.3 帧/秒 vs
                       # 真实 25.4 张/秒，差 4 倍，就是被当成「慢」的那个数）。
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_frames.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, ok, frame, fc, crops = item
            if not ok:
                _stage_log("换脸跳过 %s：frames/ 里没有这一帧（或读不出来）" % key)
                with plock2:
                    completed.add(gi2)
            elif use_lib:
                # 人脸库归组：要整帧 + embedding，只能在主线程做
                metas = []
                for ap in groups[gi2][1]:
                    m = read_dfljpg_meta(ap) or {}
                    metas.append({"ap": ap,
                                  "lm": m.get("source_landmarks") or m.get("landmarks"),
                                  "box": m.get("source_rect") or None})
                plans = _stage_pred_assign(cfg, frame, metas)
                for meta_f, use_dfm in zip(metas, plans):
                    lm = meta_f["lm"]
                    if not lm or len(lm) != 68:
                        failed += 1
                        _stage_log("换脸跳过 %s：没有 68 点" % meta_f["ap"].name)
                        continue
                    if not use_dfm:
                        unmatched += 1
                        continue
                    size = int(_dfm_session(use_dfm)[2])
                    fmat = get_transform_mat(np.asarray(lm, np.float32), size,
                                             FaceType.WHOLE_FACE, scale=1.0)
                    crop = cv2.warpAffine(frame, fmat, (size, size), flags=cv2.INTER_CUBIC)
                    _submit(gi2, meta_f["ap"].stem, use_dfm,
                            np.clip(crop.astype(np.float32) / 255.0, 0, 1))
            else:
                for stem, crop, err in (crops or []):
                    if crop is None:
                        failed += 1
                        _stage_log("换脸跳过 %s%s" % (stem, ("：" + err) if err else "：没有 68 点"))
                        continue
                    _submit(gi2, stem, dfm, crop)
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("pred", 100.0 * got / total,
                            "换脸 %d/%d 帧（第 %d..%d 帧）· %d 张脸 · %.1f 帧/秒 · "
                            "%.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]), done_faces,
                               got / el, done_faces / el, batch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
            if (gi - cursor) % 20 == 0:
                _save(k)
        else:
            k = total
    except _StageCancel:
        # 取消也要把游标落盘 —— 但**必须等排空之后**再落：以前在 except 里立刻
        # _save(commit)，那一刻队里还有没写完的脸，游标会停在 0（实测 60 张脸
        # 取消时 done_frames=0，续跑等于从头再来）。现在置标志位，排空在 finally 里做。
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            for use_dfm in list(jobs):
                _flush(use_dfm, force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
            # 等写盘排空：游标必须对得上磁盘上真的有的文件（否则续跑会跳过没写的帧）
            t_wait = time.time()
            while time.time() - t_wait < 600:
                with plock2:
                    left = sum(int(v) for v in pend.values())
                if left <= 0:
                    break
                time.sleep(0.05)
        k = _commit_upto()
        for t in threads:
            t.join(timeout=5.0)
    if cancelled["flag"]:
        _save(k)
        _stage_log("换脸取消：停在 %d/%d 帧（游标已存；已跑完 %d 张脸）"
                   % (k, total, done_faces))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    sec = round(time.time() - t0, 2)
    prof_out = {"read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
                "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
                "readers": int(readers), "writers": int(writers),
                "per_face_ms": {
                    "read": round(prof["read_ms"] / max(1, total - cursor), 2),
                    "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
                    "infer": round(prof["infer_ms"] / max(1, prof["faces"]), 2),
                    "write": round(prof["write_ms"] / max(1, prof["faces"]), 2),
                }}
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "faces_per_sec": round(done_faces / sec, 2) if sec else None,
        "frame_fps": round((total - cursor) / sec, 2) if sec else None,
        "profile": prof_out, "batch_final": int(batch)})
    _stage_log("换脸完成：第 %d..%d 帧（%d 帧）· %d 张脸 · 未匹配 %d · 失败 %d · %.2f s · "
               "%.1f 张/秒（batch %d · 读 %d/写 %d 线程 · 每张 读 %.1f + 预处理 %.1f + "
               "推理 %.1f + 写 %.1f ms）"
               % (int(rng["start"]), int(rng["end"]), total, done_faces, unmatched,
                  failed, sec, done_faces / sec if sec > 0 else 0.0, batch, readers,
                  writers, prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_face_ms"]["write"]))
    return {"count": done_faces, "frames": total, "unmatched": unmatched, "failed": failed,
            "seconds": sec, "faces_per_sec": round(done_faces / sec, 2) if sec else None,
            "batch": int(batch), "lane": "pipeline",
            "range": {"start": int(rng["start"]), "end": int(rng["end"])}}


# ---------------- ④ 遮罩推理 ----------------
_XSEG_SESS = {}          # (model_key, onnx 路径) -> (_xseg.XSegMasker)
_XSEG_SESS_LOCK = threading.RLock()


def _engine_workspace():
    """4.0 引擎仓的 workspace（模型历来也放那边）。拿不到就 None。"""
    try:
        from .._paths import engine_root
        r = engine_root()
        return r if r else None
    except Exception:                                       # noqa: BLE001
        return None


def _xseg_resolve(model):
    """按界面选择解析 XSeg / XSegLite 的 onnx（不加载，只给路径 + md5）。"""
    from .. import _xseg as _xseglib
    return _xseglib.resolve(model, WORKSPACE, _engine_workspace())


def _mask_signature(src: dict, cfg: dict = None) -> str:
    """遮罩阶段的「来源指纹」：来源/模型/模型文件变了，mask/ 就失效（不许静默复用）。

    以前是 "xseg:auto" —— 对 XSeg / XSegLite 是同一个值，于是「界面从 XSegLite 切到
    XSeg 再点合成」会拿到上一个模型算的遮罩，而且没有任何提示。现在把**实际会用到的
    那个 onnx 的 md5 前 8 位**放进指纹：换模型、换文件、换目录都会让它失效。
    """
    if src.get("source") == "dfm":
        return "dfm"
    key = str(src.get("model") or "")
    try:
        res = _xseg_resolve(key)
    except Exception:                                       # noqa: BLE001
        return "xseg:%s" % (key or "auto")
    if not res.get("key"):
        return "xseg:%s" % (key or "auto")
    return "xseg:%s:%s" % (res["key"], (res.get("md5") or "")[:8] or "nomd5")


def _xseg_sess_alive(m) -> tuple:
    """缓存的 masker 还能直接用吗：TRT 的引擎文件还在不在？onnx 还是原来那份？

    为什么要查：用户重编引擎、删引擎、或者换了一个同名 onnx 之后，**不重启软件**再跑
    遮罩阶段，必须如实重新加载并如实回显新的后端 —— 不能因为「对象还挂在缓存里」就
    继续假装自己在用 TRT（这正是「引擎明明没了还在报 trt」的那类谎话）。
    返回 (alive, 原因)。
    """
    info = getattr(m, "info", {}) or {}
    if str(info.get("backend") or "") != "trt":
        # 反向也要查一遍：上次是回落到 onnxruntime，后来用户把引擎编出来了 ——
        # 下一次遮罩阶段就该用上 TRT，不能因为对象还挂在缓存里就一直跑 onnxruntime
        try:
            from .. import _engines as _E
            sm, tv = _E.sm_tag(), _E.trt_version()
            md5 = info.get("md5") or _E.md5_fresh(str(info.get("name") or ""),
                                                  str(info.get("onnx") or ""))
            d = _E.engine_dir(WORKSPACE)
            for prec in ("int8", "bf16"):
                p = d / ("%s_%s_sm%s_trt%s.%s.engine"
                         % (info.get("name"), md5, sm, tv, prec))
                if p.is_file():
                    return False, "TRT 引擎已经编出来了（%s）" % p.name
        except Exception:                                   # noqa: BLE001
            pass
        return True, ""
    ep = str(info.get("engine_path") or "")
    if ep and not Path(ep).is_file():
        return False, "引擎文件不在了（%s）" % Path(ep).name
    try:
        from .._engines import md5_fresh
        cur = md5_fresh(str(info.get("name") or ""), str(info.get("onnx") or ""))
        if cur and info.get("md5") and cur != str(info["md5"]):
            return False, "onnx 变了（md5 %s → %s）" % (info["md5"], cur)
    except Exception:                                       # noqa: BLE001
        pass
    return True, ""


def _stage_xseg(model):
    """④ 用的 XSeg 遮罩器：**按界面选择**加载，后端走 _engines.try_engine。

    返回 (masker, res)；masker.info 里有 backend / engine / precision / reason。
    这里不走 app._get_xseg 是有意的：那份缓存按「请求的 key」存，但内部会静默退到
    另一个模型（Lite 丢了就跑 XSeg），遮罩阶段必须知道「选的和跑的到底是不是同一个」。
    缓存按 (模型, onnx 路径) 存 —— 切模型不会重复加载同一个文件；每次复用前还要过一遍
    _xseg_sess_alive（引擎被删/onnx 被换 → 重新加载，后端如实变化）。
    """
    from .. import _xseg as _xseglib
    res = _xseglib.resolve(model, WORKSPACE, _engine_workspace())
    if not res.get("path"):
        raise _BadInput("找不到 XSeg 模型（workspace/model/XSegLite/xseglite.onnx 或 "
                        "workspace/model/XSeg/XSeg.onnx）—— 遮罩来源选了 %s 就必须有它；"
                        "找过：%s" % (res.get("label") or model or "XSeg",
                                     "、".join(res.get("candidates") or [])[:300]), 500)
    # 显存不够 + DFL_ALLOW_CPU_FALLBACK=1 时，遮罩也必须在 CPU 上：
    # 不然 dfm 让出了显存、XSeg 又去抢 TRT 引擎 —— 半吊子状态比全 CPU 更难查。
    cpu = _vram_cpu_on()
    key = (res["key"], str(res["path"]), bool(cpu))
    with _XSEG_SESS_LOCK:
        hit = _XSEG_SESS.get(key)
    if hit is not None:
        alive, why = _xseg_sess_alive(hit)
        if alive:
            return hit, res
        _stage_log("遮罩：缓存的遮罩器不能再用（%s）→ 重新加载并重新判定后端" % why)
        with _XSEG_SESS_LOCK:
            _XSEG_SESS.pop(key, None)
    _own_f0 = _vram_measure_begin()
    m = _xseglib.XSegMasker(str(res["path"]),
                            workspace=(None if cpu else WORKSPACE),
                            engine_name=res["engine"],
                            providers=(["CPUExecutionProvider"] if cpu else None))
    _vram_measure_end("xseg:" + Path(str(res["path"])).name, _own_f0)
    if cpu:
        _stage_log("遮罩：★ 当前在 CPU 上跑 ★ %s 走 CPU EP（DFL_ALLOW_CPU_FALLBACK=1）"
                   % Path(str(res["path"])).name)
    with _XSEG_SESS_LOCK:
        _XSEG_SESS[key] = m
    return m, res


# dst 遮罩的像素来源（写进 meta.stages.mask.dst_input）：aligned = aligned/*.jpg 里的
# 对齐脸（再经一张补偿矩阵映射到 facelib 的合成画布）。老版本没有这个字段
# （那时每次都要解码 1080p 原帧再对齐）。
_MASK_DST_INPUT = "aligned"


def _stage_do_mask(cfg: dict) -> dict:
    """阶段 4 · 遮罩推理：aligned/ + pred/ -> mask/（每个对齐脸两张灰图）。

    语义照 4.0 export_pipeline.py:758-799（_phase_mask_worker）：
      · XSeg / XSegLite 来源：mask/<key>_xseg_prd.png = XSeg(pred)，
        mask/<key>_xseg_dst.png = XSeg(dst 对齐脸)
      · DFM 来源：遮罩就是 dfm 自带的那两条分支（③ 已经落盘 prdm/dstdm），这一步
        只把它们按 mask/ 的命名**落盘留档**（便于检查），不跑任何推理。
    两种来源都会把 dfm 的两张遮罩一起放进 mask/（mask_mode=9 全乘要用）。
    游标 = meta.stages.mask.done_frames（对齐脸张数）。

    **dst 输入 = aligned/<frame>_<fidx>.jpg：不读 frames/，只补一张几何补偿矩阵**
      以前为了拿「dst 对齐脸」先解码 1080p 原帧（≈24 ms/张脸，同一帧多张脸还各解码一次），
      再用 facelib 的 WHOLE_FACE 变换压到 256 —— 解码原帧这一步是纯浪费：同一张脸就在
      aligned/*.jpg 里（切脸阶段 68 点仿射 _align.align_mat_from_68 的直接产物）。
      现在读 aligned，再用一张补偿矩阵（image_to_face_mat⁻¹ × facelib WHOLE_FACE，见
      _stage_dst_face_mat）映射到合成画布。**为什么要补偿**：这条链的合成画布是 facelib 的
      （③ pred、⑤ compose_frame 都用它），而 aligned 芯片在 2026-12 之前是另一张画布
      （68 点拟合，中位 0.36% 缩放 + 2.5° 旋转，尾部 13%）—— 直接 resize 会让 dst 遮罩
      整体错开，mask_mode 8/7 贴回原帧就是边缘一条挖进脸里的缝。重切之后两边已经是同一张
      矩阵，这张补偿对它们约等于恒等，留着是为了兼容没重切过的老项目。
      dst 输入整条读下来实测 5.6 ms/张（解码 3.4 + 元数据 0.6 + 矩阵 0.5 + 映射 1.1），
      省掉的是 1080p 解码（≈24 ms）和每张脸重算一遍的 68 点变换。
      读不出来 / 损坏的那一张**跳过并计数**（failed + 日志点名 + meta 的 dst_bad_aligned），
      既不拿全帧兜底、也不算进成功数；一张都没成就报错。

    **吞吐**：这条链以前是纯串行（读 pred → 元数据 → 读原帧 → XSeg(prd) → 写盘 →
    XSeg(dst) → 写盘 → 复制 dfm 遮罩），一台机器上 2059 张脸 ≈ 26 张/秒，而原生 TRT
    的 XSegLite 纯推理只要 1.44 ms/图（实测），也就是 92% 的时间不在推理上。现在拆成
    读 N 线程 / 推理 1 线程 / 写 M 线程三段（和 ③ 换脸同一套结构），墙钟从「各段相加」
    变成「取最长的那段」；dst 输入换成 aligned 之后读线程不再是瓶颈（口径见
    app/tools/bench/stage_perf_profile.py 与 %TEMP% 里的 before/after 基准）。
    """
    import sys as _sys
    import numpy as np

    # 流水线实况：遮罩 = 1 个 XSeg session（DFM 来源时这块不推理，仍然是 1 条 lane）
    _STAGE_METER.set_lanes({"xseg": ["xse#0"]})
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("mask", {})
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    cfg["mask_src"] = src
    sig = _mask_signature(src, cfg)
    aligned_all = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    if not aligned_all:
        raise _BadInput("aligned/ 里没有对齐脸 —— 先跑切脸（POST /api/merge/stage/faces）", 409)
    if not (meta["stages"].get("faces") or {}).get("done"):
        raise _BadInput("切脸（faces）还没完成 —— 先跑切脸，再补遮罩", 409)
    if not (meta["stages"].get("pred") or {}).get("done"):
        raise _BadInput("换脸（pred）还没完成 —— 遮罩阶段要用换脸后的正脸，先跑换脸", 409)
    # 帧范围：只对 [start, end] 里的对齐脸推理（total/游标都按这一段的张数数）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    aligned = _stage_filter_range(rng, aligned_all, lambda p: p.stem.split("_")[0])
    total = len(aligned)
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if (st.get("signature") and st.get("signature") != sig and not cfg["force"]
            and int(st.get("done_frames") or 0) > 0):
        raise _BadInput("遮罩来源变了（%s → %s）：mask/ 里的东西已经失效，"
                        "不能静默复用。传 force=true 重跑遮罩阶段"
                        % (st.get("signature"), sig), 409)
    # 老版本（dst 输入 = 原帧重新对齐）算出来的 mask/ 不能静默复用：来源没变，但几何变了。
    # 这里不报 409（挡的是「换了模型还复用旧遮罩」），直接当 force 整段重算并说清原因。
    stale_dst = (bool(st.get("done")) and bool(st.get("inference"))
                 and str(st.get("dst_input") or "") != _MASK_DST_INPUT)
    if stale_dst:
        _stage_log("遮罩：现有 mask/ 是「dst 输入 = 原帧重新对齐」那一版算的（dst_input=%s）"
                   "→ 口径已改成 aligned/，几何变了，整段重算"
                   % (st.get("dst_input") or "-",))
    redo = bool(cfg["force"] or rchg or stale_dst)
    if (not redo) and st.get("done") and st.get("signature") == sig \
            and int(st.get("done_frames") or 0) >= total:
        _stage_log("遮罩：复用已有 %d 张遮罩（第 %d..%d 帧，不重跑；要重做传 force=true）"
                   % (total, int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": int(st.get("count") or 0), "reason": "already-done"}

    # ---- 按界面选择加载模型；后端 = TRT 有就用、没有就 onnxruntime（结论全部回显）----
    xseg = None
    backend = "none"
    engine_file = ""
    engine_prec = ""
    fallback = ""
    model_onnx = ""
    model_label = str(src.get("label") or "")
    if src["source"] == "xseg":
        xseg, res = _stage_xseg(src.get("model"))
        info = dict(getattr(xseg, "info", {}) or {})
        backend = str(info.get("backend") or "ort")
        engine_file = str(info.get("engine") or "")
        engine_prec = str(info.get("precision") or "")
        fallback = str(info.get("reason") or "")
        model_onnx = str(info.get("onnx") or res.get("path") or "")
        model_label = res.get("label") or model_label
        if backend == "trt":
            _stage_log("遮罩：模型=%s（%s）· 后端=TRT(%s) · 引擎=%s"
                       % (model_label, Path(model_onnx).name, engine_prec or "?", engine_file))
        else:
            _stage_log("遮罩：模型=%s（%s）· 后端=onnxruntime · 没有可用的 TRT 引擎 → 回落"
                       "（原因：%s）" % (model_label, Path(model_onnx).name,
                                        fallback or "未知"))
    engine = backend          # 老字段（导出/验收还在用它）—— 语义就是 trt|ort|none
    cursor = 0 if redo else max(0, min(int(st.get("done_frames") or 0), total))
    if cursor:
        _stage_log("遮罩：从范围内第 %d 张续跑（前 %d 张已经在）" % (cursor + 1, cursor))
    _stage_log("遮罩：来源=%s · 模型=%s · 引擎=%s%s · 第 %d..%d 帧（%d 张对齐脸）%s"
               % (src.get("label") or src.get("source"), model_label or "-", engine,
                  ("(%s)" % engine_file) if engine_file else "",
                  int(rng["start"]), int(rng["end"]), total,
                  ("" if not src.get("model_note") else
                   " · 注意：" + str(src.get("model_note")))))

    t0 = time.time()
    # force / 换范围 / dst 输入口径变了 都从 0 数（游标和计数都相对范围）
    wrote = 0 if redo else int(st.get("count") or 0)
    if redo:
        st["count"] = 0
    failed = int(st.get("failed") or 0)
    means = []
    t_infer = 0.0          # 纯推理累计（两张 256 的 XSeg 前向，不含读图/写盘）
    # 逐段打点（和 ③ 换脸同一套口径：读图 / 预处理 / 推理 / 写盘，各自累计 ms）。
    # dst 输入改成 aligned 之后，几何映射并在读图段里（一次 558→256 的重采样）→ 预处理段
    # prep_ms 恒为 0（保留这个打点名，是为了和上一版的 profile 直接对比）；
    # 读图 = pred PNG 解码 + aligned JPEG 解码 + 补偿矩阵 + 映射到 256。
    prof = {"read_ms": 0.0, "prep_ms": 0.0, "infer_ms": 0.0, "write_ms": 0.0,
            "faces": 0, "images": 0, "bad_aligned": 0}
    bad_aligned = []       # 读不出来的 aligned 文件名（最多留 20 个，进 meta / 日志）
    plock = threading.Lock()

    def _bump(ms_name: str, sec: float, n: int = 0, images: int = 0):
        with plock:
            prof[ms_name] += sec * 1000.0
            if n:
                prof["faces"] += n
            if images:
                prof["images"] += images

    def _bad_aligned(name: str):
        """记一张读不出来的 aligned 文件（计数 + 最多留 20 个名字）。"""
        with plock:
            prof["bad_aligned"] = int(prof.get("bad_aligned") or 0) + 1
            if len(bad_aligned) < 20:
                bad_aligned.append(str(name))

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "total": total,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "count": wrote, "failed": failed, "signature": sig,
            "mask_source": src.get("label") or src.get("source"),
            "mask_model": str(src.get("model") or ""),
            "mask_model_from": str(src.get("model_from") or ""),
            "mask_onnx": model_onnx, "mask_label": model_label,
            "engine": engine,                 # 老字段，语义 = backend
            "backend": backend,               # trt | ort | none
            "engine_file": engine_file,       # 实际加载的引擎文件名（trt 才有）
            "engine_precision": engine_prec,  # int8 / bf16
            "fallback_reason": fallback,      # 没上 TRT 的一句话原因（有引擎时为空）
            "size": 256, "inference": (src["source"] == "xseg"),
            # dst 遮罩的像素来源：aligned（aligned/*.jpg 本身）/ dfm（dfm 自带，不推理）。
            # 老版本没有这个字段 —— _stage_do_mask 靠它认出「从原帧重建」那一版并重算。
            "dst_input": (_MASK_DST_INPUT if src["source"] == "xseg" else "dfm"),
            "dst_bad_aligned": int(prof.get("bad_aligned") or 0),
            "dir": str(mask_dir), "resumed_from": int(cursor),
            "msg": ("遮罩完成：%d 张脸（第 %d..%d 帧 · 来源 %s）"
                    % (wrote, int(rng["start"]), int(rng["end"]),
                       src.get("label") or src.get("source"))) if done
                   else ("遮罩 %d/%d 张（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["mask"] = node
        _stage_meta_save(cache, meta)

    def _copy_dfm_masks(key: str):
        """把 ③ 的 dfm 遮罩按 mask/ 的命名落盘（来源=DFM 时就是这一步的产物）。"""
        n = 0
        for src_name, dst_name in (("_prdm", "_dfm_prd"), ("_dstdm", "_dfm_dst")):
            a = pred_dir / (key + src_name + ".png")
            if not a.is_file():
                continue
            try:
                _stage_write_atomic(mask_dir / (key + dst_name + ".png"), a.read_bytes())
                n += 1
            except OSError:
                pass
        return n

    # ---- 串行链 → 三段流水线 ----
    # 以前这里是一条串行链：读 pred PNG → 读 DFLJPG 元数据 → 读 1080p 原帧 → 对齐到
    # 256 → XSeg(prd) → 编码写盘 → XSeg(dst) → 编码写盘 → 复制 dfm 的两张遮罩。
    # 纯推理（原生 TRT int8，见 stage_perf_profile.py 的实测）只有 1.44 ms/张，
    # 两张也就 2.9 ms；2059 张脸却只有 26 张/秒（≈38 ms/张）—— 也就是 92% 的时间
    # 不在推理上，而且串行意味着「各段相加」。现在拆成三段用队列连起来：
    #     读线程 ×N   pred PNG 解码 + aligned JPEG 解码并映射到 256（_stage_read_dst_face）
    #     推理线程    攒够 mask_batch 张脸，一次前向（ORT 动态 batch 或 TRT 逐张）
    #     写线程 ×M   PNG 编码 + 原子落盘 + dfm 两张遮罩的文件复制
    # 墙钟 ≈ max(三段) 而不是三段之和。
    # dst 输入换成 aligned 之后，读线程从「每张脸解码一张 1080p PNG（≈24 ms/张）」降到
    # 「解码一张 558 的 JPEG 再映射到 256（≈4 ms/张）」，同一帧的多张脸也不再共用帧缓存
    # （每张脸读自己那张 aligned）；写线程仍是「每张脸 4 个 PNG + 2 次文件复制」（~24 ms/张）。
    readers = max(1, int(cfg.get("mask_readers")
                         or min(6, max(2, (os.cpu_count() or 4) // 4))))
    writers = max(1, int(cfg.get("mask_writers")
                         or min(6, max(2, (os.cpu_count() or 4) // 8))))
    # 默认**不批量**（1）：本机实测 256² 逐张 4.79 ms/图 比 batch4 的 5.24 ms/图 更快
    # （ORT 在 batch>1 时换 kernel，流水线里还被队列拉长），而批量会换规约顺序、遮罩
    # 逐像素差 1~2 LSB。要试就传 mask_batch=N（引擎真吃批时的收益见 _xseg.mask_batch）。
    mbatch = max(1, int(cfg.get("mask_batch") or 1))
    if src["source"] == "xseg" and xseg is not None:
        try:
            # TRT 的 xseglite 引擎是静态 batch=1（_engines.PROFILES 单档位）→ 老实逐张；
            # ort-CUDA 的输入是动态 batch → 一次多张（256² 实测 4.02 → 2.68 ms/张）
            mbatch = max(1, min(mbatch, int(xseg.batch_limit())))
        except Exception:                                   # noqa: BLE001
            mbatch = 1
    _stage_log("遮罩：流水线 读 %d 线程 / 推理 batch %d / 写 %d 线程 · 后端 %s"
               % (readers, mbatch, writers, backend))

    if src["source"] != "xseg":
        # 来源=DFM：遮罩就是 ③ 落盘的那两张，这里只按 mask/ 的命名留档，不推理。
        k = cursor
        try:
            for k in range(cursor, total):
                _stage_check_cancel()
                key = aligned[k].stem
                try:
                    _copy_dfm_masks(key)
                    wrote += 1
                except Exception as e:                      # noqa: BLE001
                    failed += 1
                    _stage_log("遮罩跳过 %s：%s" % (key, str(e)[:140]))
                el = max(0.001, time.time() - t0)
                _stage_progress("mask", 100.0 * (k + 1) / total,
                                "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒"
                                % (k + 1, total, int(rng["start"]), int(rng["end"]),
                                   (k + 1 - cursor) / el),
                                frames_done=k + 1, frames_total=total,
                                frame_fps=round((k + 1 - cursor) / el, 2))
        except _StageCancel:
            _save(k)                # 取消也要把游标落盘：下次从这里接着跑
            _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
            raise
        sec = round(time.time() - t0, 2)
        done_n = max(1, total - cursor)
        _save(total, done=True, extra={
            "seconds": sec, "finished_at": time.time(),
            "fps": round((total - cursor) / sec, 2) if sec else None,
            "mask_mean": None, "infer_seconds": 0.0, "infer_ms_per_face": 0.0,
        })
        _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d · %.2f s · %.1f 张/秒 · "
                   "来源 %s（不推理，只落 dfm 自带遮罩）"
                   % (int(rng["start"]), int(rng["end"]), total, failed, sec,
                      (total - cursor) / sec if sec > 0 else 0.0,
                      src.get("label") or src.get("source")))
        return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
                "mask_source": src.get("label") or src.get("source"),
                "signature": sig, "engine": engine, "backend": backend,
                "engine_file": engine_file,
                "range": {"start": int(rng["start"]), "end": int(rng["end"])},
                "infer_ms_per_face": 0.0}

    # ---------------- XSeg：读 / 推理 / 写 三段流水线 ----------------
    q_prep = queue.Queue(maxsize=max(2, readers * 2))
    q_write = queue.Queue(maxsize=max(4, writers * mbatch * 2))
    stop = threading.Event()
    boss = {"exc": None}
    pend, completed, plock2 = {}, set(), threading.Lock()
    commit = {"n": cursor}

    def _commit_upto():
        with plock2:
            n = commit["n"]
            while n in completed:
                completed.discard(n)
                n += 1
            commit["n"] = n
            return n

    def _pend_add(gi: int):
        with plock2:
            pend[gi] = int(pend.get(gi) or 0) + 1

    def _pend_done(gi: int):
        with plock2:
            left = int(pend.get(gi) or 0) - 1
            if left <= 0:
                pend.pop(gi, None)
                completed.add(gi)
            else:
                pend[gi] = left

    def _put_prep(item):
        while not stop.is_set():
            try:
                q_prep.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _u8(m):
        return None if m is None else np.clip(m * 255.0, 0, 255).astype(np.uint8)

    def _writer():
        nonlocal failed
        while True:
            item = q_write.get()
            if item is None:
                return
            gi, key, prd_u8, dst_u8 = item
            _stage_tick("xseg")      # 交接点：XSeg 遮罩 → mask/*.png
            tw = time.perf_counter()
            try:
                _copy_dfm_masks(key)
                if prd_u8 is not None:
                    # PNG level=1（无损，只是少压一层）：掩码是灰度，level 3 每张只快一点点
                    _stage_write_atomic(mask_dir / (key + "_xseg_prd.png"),
                                        _stage_imencode_png(prd_u8, 1))
                if dst_u8 is not None:
                    _stage_write_atomic(mask_dir / (key + "_xseg_dst.png"),
                                        _stage_imencode_png(dst_u8, 1))
            except Exception as e:                          # noqa: BLE001
                failed += 1
                _stage_log("遮罩写盘失败 %s：%s" % (key, str(e)[:140]))
            finally:
                _bump("write_ms", time.perf_counter() - tw)
                _pend_done(gi)

    def _reader(lo: int, hi: int):
        try:
            for gi in range(lo, hi):
                if stop.is_set():
                    break
                ap = aligned[gi]
                key = ap.stem
                tr = time.perf_counter()
                pm = pred_dir / (key + ".png")
                pred = _stage_read_image(pm) if pm.is_file() else None
                # dst 输入 = aligned/*.jpg（切脸阶段 68 点仿射的直接产物），只做一次「映射到
                # 合成画布」的重采样：不读 1080p 原帧（见 _stage_read_dst_face）。
                dst, why_dst = _stage_read_dst_face(ap)
                _bump("read_ms", time.perf_counter() - tr)
                why = ""
                if pred is None:
                    why = "pred/ 里没有换脸结果（先跑换脸）"
                elif dst is None:
                    why = "aligned 用不了 %s：%s" % (ap.name, why_dst)
                    _bad_aligned(ap.name)
                if why:
                    _put_prep((gi, key, None, None, why))
                    continue
                _put_prep((gi, key, pred, dst, ""))
        except Exception as e:                              # noqa: BLE001
            boss["exc"] = boss["exc"] or e
            stop.set()
            try:
                q_prep.put((None, None, None, None, None))
            except Exception:                               # noqa: BLE001
                pass

    jobs = []

    def _flush(force: bool = False):
        nonlocal t_infer, wrote
        while jobs and (len(jobs) >= mbatch or (force and jobs)):
            n = min(len(jobs), mbatch)
            chunk = jobs[:n]
            imgs, slots = [], []
            for i, (_gi, _k, pred, dst, _why) in enumerate(chunk):
                imgs.append(pred)
                slots.append((i, 0))
                if dst is not None:
                    imgs.append(dst)
                    slots.append((i, 1))
            ti = time.perf_counter()
            outs = xseg.mask_batch(imgs, out_size=256)
            dt = time.perf_counter() - ti
            t_infer += dt
            _bump("infer_ms", dt, n, len(imgs))
            _INFER_CALLS["xseg"] += len(imgs)
            prd_m = [None] * n
            dst_m = [None] * n
            for (i, slot), mk in zip(slots, outs):
                if slot:
                    dst_m[i] = mk
                else:
                    prd_m[i] = mk
                    means.append(float(np.asarray(mk).mean()))
            for i, (gi, key, _p, _d, _w) in enumerate(chunk):
                q_write.put((gi, key, _u8(prd_m[i]), _u8(dst_m[i])))
                wrote += 1
            del jobs[:n]

    _save(cursor)
    threads = []
    for i in range(writers):
        t = threading.Thread(target=_writer, daemon=True, name="mask-w%d" % i)
        t.start()
        threads.append(t)
    for r in range(readers):
        lo = cursor + (total - cursor) * r // readers
        hi = cursor + (total - cursor) * (r + 1) // readers
        if lo >= hi:
            continue
        t = threading.Thread(target=_reader, args=(lo, hi), daemon=True, name="mask-r%d" % r)
        t.start()
        threads.append(t)

    got = 0                     # 已经处理完的张数（显示用；游标 commit 会为了不留空洞而滞后）
    k = cursor
    cancelled = {"flag": False}
    try:
        for gi in range(cursor, total):
            _stage_check_cancel()
            item = q_prep.get()
            if item[0] is None:
                break
            got += 1
            gi2, key, pred, dst, why = item
            if pred is None:
                failed += 1
                _stage_log("遮罩跳过 %s：%s" % (key, why or "pred/ 里没有换脸结果（先跑换脸）"))
                with plock2:
                    completed.add(gi2)
            else:
                _pend_add(gi2)
                jobs.append((gi2, key, pred, dst, ""))
                _flush()
            _commit_upto()
            k = commit["n"]
            el = max(0.001, time.time() - t0)
            _stage_progress("mask", 100.0 * got / total,
                            "遮罩 %d/%d 张（第 %d..%d 帧）· %.1f 张/秒 · batch %d"
                            % (got, total, int(rng["start"]), int(rng["end"]),
                               got / el, mbatch),
                            frames_done=got, frames_total=total,
                            frame_fps=round(got / el, 2))
    except _StageCancel:
        cancelled["flag"] = True
    finally:
        stop.set()
        exc_t = _sys.exc_info()[0]
        broken = boss.get("exc") is not None or exc_t not in (None, _StageCancel)
        if not broken:
            _flush(force=True)
        for _i in range(writers):
            q_write.put(None)
        if not broken:
            t_wait = time.time()
            while time.time() - t_wait < 600:
                with plock2:
                    left = sum(int(v) for v in pend.values())
                if left <= 0:
                    break
                time.sleep(0.05)
        k = _commit_upto()
        for t in threads:
            t.join(timeout=5.0)
    if cancelled["flag"]:
        # 取消也要把游标落盘（排空之后再落，才和磁盘上真的有的文件对得上）
        _save(k)
        _stage_log("遮罩取消：停在 %d/%d 张（游标已存）" % (k, total))
        raise _StageCancel()
    if boss.get("exc") is not None:
        raise boss["exc"]
    if total > 0 and int(prof["faces"]) == 0:
        # 一张都没推理成功 → 别把空 mask/ 当「已完成」（下游合成会拿不到遮罩却以为跑过了）
        raise _BadInput("遮罩：%d 张对齐脸一张都没推理成功（aligned 读不出来 %d 张 / pred 缺失？）"
                        "—— 空 mask/ 不算成功。先看 aligned/ 和 pred/：%s"
                        % (total, int(prof.get("bad_aligned") or 0),
                           "、".join(bad_aligned[:5]) or "-"), 500)
    sec = round(time.time() - t0, 2)
    done_n = max(1, total - cursor)
    n_faces = max(1, prof["faces"])
    prof_out = {
        "read_ms": round(prof["read_ms"], 1), "prep_ms": round(prof["prep_ms"], 1),
        "infer_ms": round(prof["infer_ms"], 1), "write_ms": round(prof["write_ms"], 1),
        "readers": int(readers), "writers": int(writers), "mask_batch": int(mbatch),
        "backend": backend, "engine_file": engine_file,
        "images": int(prof["images"]),
        "dst_input": _MASK_DST_INPUT, "bad_aligned": int(prof.get("bad_aligned") or 0),
        "dst_mat": "image_to_face_mat⁻¹ · facelib WHOLE_FACE",
        # 每张脸各段各花多少 ms（读图 / 预处理（这一版恒 0）/ 2 次推理 / 编码+落盘）
        "per_face_ms": {
            "read": round(prof["read_ms"] / max(1, total - cursor), 2),
            "prep": round(prof["prep_ms"] / max(1, total - cursor), 2),
            "infer": round(prof["infer_ms"] / n_faces, 2),
            "write": round(prof["write_ms"] / n_faces, 2),
        },
        # 单张纯推理 = 总推理时间 / 推理张数（一张脸两张图）
        "per_image_ms": round(prof["infer_ms"] / max(1, prof["images"]), 3),
    }
    _save(total, done=True, extra={
        "seconds": sec, "finished_at": time.time(),
        "fps": round((total - cursor) / sec, 2) if sec else None,
        "mask_mean": (round(sum(means) / len(means), 4) if means else None),
        # TRT vs ORT 的对比就靠这两个数：同一个视频、同一份模型、同一个预处理
        "infer_seconds": round(t_infer, 3),
        "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                              if src["source"] == "xseg" else 0.0),
        "profile": prof_out,
    })
    _stage_log("遮罩完成：第 %d..%d 帧 · %d 张 · 失败 %d%s · %.2f s · %.1f 张/秒 · "
               "后端 %s%s · 来源 %s · dst 输入 %s · 每张 读 %.1f + 预处理 %.1f + 推理 %.1f"
               "（%.2f ms/图 × %d 图）+ 写 %.1f ms（读 %d / 写 %d 线程 · batch %d）"
               % (int(rng["start"]), int(rng["end"]), total, failed,
                  ("（其中 aligned 读不出来 %d）" % int(prof.get("bad_aligned") or 0))
                  if int(prof.get("bad_aligned") or 0) else "", sec,
                  (total - cursor) / sec if sec > 0 else 0.0,
                  backend, ("(%s)" % engine_file) if engine_file else "",
                  src.get("label") or src.get("source"), _MASK_DST_INPUT,
                  prof_out["per_face_ms"]["read"], prof_out["per_face_ms"]["prep"],
                  prof_out["per_face_ms"]["infer"], prof_out["per_image_ms"],
                  prof["images"], prof_out["per_face_ms"]["write"],
                  readers, writers, mbatch))
    return {"count": wrote, "failed": failed, "total": total, "seconds": sec,
            "mask_source": src.get("label") or src.get("source"),
            "signature": sig, "engine": engine, "backend": backend,
            "engine_file": engine_file, "dst_input": _MASK_DST_INPUT,
            "bad_aligned": int(prof.get("bad_aligned") or 0),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "mask_batch": int(mbatch), "profile": prof_out,
            "infer_ms_per_face": (round(t_infer * 1000.0 / done_n, 3)
                                  if src["source"] == "xseg" else 0.0)}





def _stage_profile_blank() -> dict:
    """合成阶段的逐帧打点容器（五段 + 帧计数）。"""
    return {"frames": 0, "faces": 0, "no_face": 0, "swapped": 0,
            "read_ms": 0.0, "prep_ms": 0.0, "mask_ms": 0.0, "ct_ms": 0.0, "paste_ms": 0.0,
            "blend_ms": 0.0, "encode_ms": 0.0, "write_ms": 0.0, "frame_ms": 0.0}


def _stage_merge_face_inputs(cache, ap, cfg) -> dict:
    """一张对齐脸的合成输入：pred PNG + 三组遮罩（全从磁盘读，不跑推理）。"""
    import numpy as np
    from .._align import read_dfljpg_meta
    from .compose import _read_gray

    key = ap.stem
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    m = read_dfljpg_meta(ap) or {}
    lm = m.get("source_landmarks") or m.get("landmarks")
    if not lm or len(lm) != 68:
        return None
    pred = _stage_read_image(pred_dir / (key + ".png"))
    if pred is None:
        return None
    return {
        "key": key,
        "landmarks": np.asarray(lm, np.float32),
        "pred": pred.astype(np.float32) / 255.0,
        "prd_dfm": _read_gray(pred_dir / (key + "_prdm.png")),
        "dst_dfm": _read_gray(pred_dir / (key + "_dstdm.png")),
        "xseg_prd": _read_gray(mask_dir / (key + "_xseg_prd.png")),
        "xseg_dst": _read_gray(mask_dir / (key + "_xseg_dst.png")),
    }


def _merge_worker_count(cfg: dict) -> int:
    """这次合成开几个 worker（默认 8；请求体 / 环境变量可覆盖，1..32）。"""
    for key in ("merge_workers", "merge_threads", "threads"):
        raw = cfg.get(key)
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            return max(1, min(32, int(raw)))
        except (TypeError, ValueError):
            continue
    return max(1, min(32, int(_MERGE_WORKERS_DEFAULT or 8)))


def _merge_cv2_threads(cfg: dict, workers: int) -> int:
    """每个 worker 里该给 OpenCV 留几条内部线程（0 = 这一版不去动 cv2 的设置）。

    OpenCV 的 parallel_for 线程池是**进程级**的：不显式设，每个 worker 里的
    warpAffine / seamlessClone / GaussianBlur 都会各自摊到全部 32 条线程上
    （cv2.getNumThreads() 默认 = cpu 数），8 个 worker 一起跑就是 256 条线程抢 16 核，
    每帧的 cv2 调用从 6ms 涨到 20-40ms —— 这正是「8 线程只有 31% CPU」的主因之一。
    核少、活多的时候（workers <= 1）反过来：让 cv2 自己把**一帧**切到多核才划算
    （实测 1 worker：cv2=32 时 4.27 帧/秒，cv2=1 时 2.86 帧/秒）。
    """
    raw = cfg.get("merge_cv2_threads")
    if raw is None:
        raw = _MERGE_CV2_THREADS_DEFAULT
    try:
        want = int(raw or 0)
    except (TypeError, ValueError):
        want = 0
    if want > 0:
        return min(64, want)
    cpu = os.cpu_count() or 4
    if int(workers) <= 1:
        return max(1, min(cpu, 8))
    # worker 一多就**别**再让 OpenCV 自己摊线程：1080×1920 · seamless · 8 worker 实测
    #   cv2=1 → 13.56 帧/秒 · 7.4 核      cv2=8 → 12.89 帧/秒 · 8.7 核
    # 同样的吞吐，cv2=1 少烧 1.3 个核（用户要留给训练/预览）。只有「核相对 worker
    # 还很富余」时才给 cv2 留内部线程。
    return max(1, min(4, cpu // (4 * max(1, int(workers)))))


def _stage_do_merge(cfg: dict) -> dict:
    """阶段 5 · 合成：**纯 CPU**。frames/ + aligned/ + pred/ + mask/ -> 就地覆盖 frames/。

    语义基准 = 4.0 MergeStudio/core/export_pipeline.py 的 _phase_merge_worker()
    （见 app/server/_merge_compose.py 顶部的逐段对照表）：遮罩准备（mask_mode 组合 +
    erode/blur）→ 色彩迁移（ct_mode）→ 融合（overlay / hist-match / seamless /
    seamless-hist-match）→ 反贴回原帧（face_mat 逆变换）→ 写盘。

    这一阶段**不许碰推理**：dfm / XSeg 都已经在 ③④ 落盘，这里只读文件。
    验收会断言「推理调用计数 = 0」且显存不增长（profile 里也有 gpu_mem 前后值）。
    续跑游标 = meta.stages.merge.done_frames（就地覆盖只往前推）。
    """
    import cv2
    import numpy as np
    from .compose import compose_frame

    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("merge", {})
    frames_all = _stage_files(_stage_frames_dir(cache))
    if not frames_all:
        raise _BadInput("frames/ 里没有帧图 —— 先跑切帧", 409)
    # 帧范围：只覆盖 [start, end] 里的帧（就地覆盖 frames/，别的帧一个都不碰）
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    frames = _stage_filter_range(rng, frames_all)
    total = len(frames)
    if ((not cfg["force"]) and not rchg and st.get("done")
            and int(st.get("done_frames") or 0) >= total and total):
        _stage_log("合成：复用已完成的结果（第 %d..%d 帧，不重跑）"
                   % (int(rng["start"]), int(rng["end"])))
        return {"skipped": True, "count": total, "reason": "already-done"}
    for need, label in (("faces", "切脸"), ("pred", "换脸"), ("mask", "遮罩")):
        if not (meta["stages"].get(need) or {}).get("done"):
            raise _BadInput("合成要用 %s 的产物，但「%s」阶段还没完成 —— 先跑%s"
                            "（POST /api/merge/stage/%s）" % (label, label, label, need), 409)
    src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
    sig = _mask_signature(src, cfg)
    mst = meta["stages"].get("mask") or {}
    if mst.get("signature") and mst.get("signature") != sig:
        raise _BadInput("遮罩来源变了（mask/ 是 %s 产出的，现在是 %s）—— 先重跑遮罩阶段"
                        "（force=true），别拿旧遮罩合成" % (mst.get("signature"), sig), 409)

    aligned = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    by_frame = {}
    for ap in aligned:
        by_frame.setdefault(ap.stem.split("_")[0], []).append(ap)

    # 游标 **相对范围**：0..total
    cursor = 0 if (rchg or (cfg["force"] and not meta.get("frame_swapped"))) \
        else int(st.get("done_frames") or 0)
    cursor = max(0, min(cursor, total))
    if meta.get("frame_swapped") and cursor == 0 and int(st.get("done_frames") or 0) == 0:
        raise _BadInput("frames/ 已经是合成结果（frame_swapped=true），但 meta 里没有续跑位置 —— "
                        "只能重跑切帧 + 切脸 + 换脸 + 遮罩", 409)
    if meta.get("frame_swapped") and rchg:
        _stage_log("合成：frames/ 已经就地合成过，这次换到第 %d..%d 帧 —— 这一段会在已合成的帧上"
                   "再合成一次；要绝对干净请先 force 重切帧"
                   % (int(rng["start"]), int(rng["end"])))
    if cursor:
        _stage_log("合成：从范围内第 %d 帧续跑（前 %d 帧已经是合成结果）" % (cursor + 1, cursor))

    meta["frame_swapped"] = True
    comp_cfg = dict(cfg.get("config") or {})
    comp_cfg["mask_mode"] = int((cfg.get("settings") or {}).get("mask_mode", 6) or 0)
    threads = _merge_worker_count(cfg)
    cv2_threads = _merge_cv2_threads(cfg, threads)
    png_level = int(cfg.get("png_level") if cfg.get("png_level") is not None else 1)
    _stage_log("合成（纯 CPU）：第 %d..%d 帧（%d 帧）· 遮罩 %s/%s · ct=%s · blend=%s · "
               "%d 线程（每个 cv2 %d 线程）· PNG level %d"
               % (int(rng["start"]), int(rng["end"]), total,
                  src.get("label") or src.get("source"),
                  comp_cfg.get("mask_mode"), comp_cfg.get("color_transfer_mode"),
                  comp_cfg.get("mode"), threads, cv2_threads, png_level))

    prof = dict(_stage_profile_blank())
    for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms", "encode_ms",
              "write_ms", "frame_ms"):
        prof[k] = float(st.get("profile", {}).get(k) or 0.0)
    prof["frames"] = cursor
    swapped = int(st.get("swapped") or 0)
    copied = int(st.get("copied") or 0)
    no_face = int(st.get("no_face") or 0)
    failed = int(st.get("failed") or 0)
    t0 = time.time()
    start_cursor = cursor
    infer0 = int(sum(_INFER_CALLS.values()))
    sess0 = len(_DFM_SESS)
    gpu0 = _gpu_mem_used_mb()
    stop = threading.Event()
    lock = threading.Lock()

    def _save(up_to: int, done: bool = False, extra: dict = None):
        node = {
            "done": bool(done), "done_frames": int(up_to), "frames": total,
            "frames_all": len(frames_all),
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "range_key": rkey, "range_total": int(total),
            "swapped": swapped, "copied": copied, "no_face": no_face, "failed": failed,
            "resumed_from": int(start_cursor), "in_place": True,
            "aligned_faces": len(aligned), "pure_cpu": True,
            "threads": threads, "png_level": png_level,
            "mask_mode": comp_cfg.get("mask_mode"),
            "ct_mode": comp_cfg.get("color_transfer_mode"),
            "blend": comp_cfg.get("mode"),
            "profile": {k: round(float(v), 2) for k, v in prof.items()},
            "msg": ("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d）"
                    % (int(rng["start"]), int(rng["end"]), total, swapped, copied)) if done
                   else ("合成 %d/%d 帧（第 %d..%d 帧）"
                         % (up_to, total, int(rng["start"]), int(rng["end"]))),
        }
        if extra:
            node.update(extra)
        meta["stages"]["merge"] = node
        _stage_meta_save(cache, meta)

    def _one(fp):
        """一帧：读图 → 逐脸合成 → PN G编码 → 写盘。返回 (rec, 各段耗时)。"""
        if stop.is_set():
            # 取消后不再开新帧（排队里的那些也直接放掉，别让 shutdown 干等）。
            return {"key": fp.stem, "ok": True, "no_face": True, "skipped": True}, {}
        t = time.time()
        tim = {}
        frame = _stage_read_frame(fp)
        if frame is None:
            return {"key": fp.stem, "ok": False, "why": "读不出来"}, tim
        tim["read_ms"] = (time.time() - t) * 1000.0
        aps = by_frame.get(fp.stem) or []
        faces = []
        for ap in aps:
            f = _stage_merge_face_inputs(cache, ap, cfg)
            if f is not None:
                faces.append(f)
        if not faces:
            return {"key": fp.stem, "ok": True, "no_face": True, "faces": len(aps)}, tim
        t_c = time.time()
        out = compose_frame(frame, faces, comp_cfg, tim)
        tim["frame_ms"] = (time.time() - t_c) * 1000.0
        t_e = time.time()
        data = _stage_imencode_png(out, png_level)
        tim["encode_ms"] = (time.time() - t_e) * 1000.0
        _stage_tick("compose")   # 交接点：合成 → merged/*.png
        t_w = time.time()
        _stage_write_atomic(fp, data)
        tim["write_ms"] = (time.time() - t_w) * 1000.0
        return {"key": fp.stem, "ok": True, "swapped": 1, "faces": len(faces)}, tim

    from concurrent.futures import ThreadPoolExecutor
    chunk = max(threads * 2, 8)
    # 流水线实况：合成 lane 数 = 这个池**实际**的线程数（不分段链写死 8）
    _STAGE_METER.set_lanes({"compose": ["t%d" % _i for _i in range(int(threads))]})
    prog = {"done": cursor}

    def _report():
        """进度 + meta（每 chunk 一次；和以前同一个口径）。"""
        el = max(0.001, time.time() - t0)
        rate = (prog["done"] - start_cursor) / el
        eta = int((total - prog["done"]) / rate) if rate > 0 and prog["done"] < total else None
        n = max(1, prof["frames"] - cursor)
        _stage_progress(
            "merge", 100.0 * prog["done"] / total,
            "合成 %d/%d 帧（第 %d..%d 帧）· %.1f 帧/秒（换脸 %d · 原样 %d）"
            % (prog["done"], total, int(rng["start"]), int(rng["end"]),
               rate, swapped, copied),
            frames_done=prog["done"], frames_total=total,
            frame_fps=round(rate, 2), eta_s=eta,
            swapped=swapped, copied=copied, no_face=no_face,
            profile={"per_frame_ms":
                     {k: round(float(prof.get(k) or 0.0) / n, 2)
                      for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms",
                                "blend_ms", "encode_ms", "write_ms", "frame_ms")},
                     "fps": round(rate, 2)})
        _save(prog["done"])

    # cv2 内部线程数：跟 worker 数一起调（见 _merge_cv2_threads 的注释）。
    cv2_prev = 0
    cv2_set = 0
    if cv2_threads > 0:
        try:
            cv2_prev = int(cv2.getNumThreads())
            if cv2_prev != int(cv2_threads):
                cv2.setNumThreads(int(cv2_threads))
                cv2_set = int(cv2_threads)
        except Exception:                                         # noqa: BLE001
            cv2_prev, cv2_set = 0, 0
    pool = None
    try:
        # **整段一个池**：以前是每个 chunk 重建一次 ThreadPoolExecutor，
        # 每个 chunk 末尾都有一道隐式栅栏（shutdown(wait=True)）—— 最后一批 worker
        # 空等，下一批才开工。8 worker / chunk=16 时白丢 6~12% 吞吐。
        pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="merge")
        done_in_chunk = 0
        # 一次性把所有帧提交进池：worker 干完一帧立刻拿下一帧，没有 chunk 边界。
        for rec, tim in pool.map(_one, frames[cursor:]):
            if _stage_cancelled():
                stop.set()
                raise _StageCancel()
            if rec.get("skipped"):
                continue                  # 取消尾巴：不算帧、不动游标
            if not rec.get("ok"):
                failed += 1
                _stage_log("合成跳过 %s：%s" % (rec.get("key"), rec.get("why")))
            elif rec.get("no_face"):
                copied += 1
                no_face += 1
            else:
                swapped += 1
            prof["frames"] += 1
            prof["faces"] += int(rec.get("faces") or 0)
            for kk, vv in tim.items():
                prof[kk] = float(prof.get(kk) or 0.0) + float(vv or 0.0)
            prog["done"] += 1
            done_in_chunk += 1
            if done_in_chunk >= chunk:
                done_in_chunk = 0
                _report()
        _report()                     # 收尾也报一次（进度到 100%），和以前每 chunk 末尾一致
    except _StageCancel:
        _save(prog["done"])
        _stage_log("合成取消：停在 %d/%d 帧（游标已存）" % (prog["done"], total))
        raise
    finally:
        if pool is not None:
            # 取消时别等已经排队的帧：置 stop + cancel_futures，worker 下一帧开头就退出。
            pool.shutdown(wait=not stop.is_set(), cancel_futures=True)
        if cv2_set:
            try:
                cv2.setNumThreads(cv2_prev)
            except Exception:                                     # noqa: BLE001
                pass

    sec = round(time.time() - t0, 2)
    n = max(1, prof["frames"] - cursor)
    per = {k: round(float(prof.get(k) or 0.0) / n, 2)
           for k in ("read_ms", "prep_ms", "mask_ms", "ct_ms", "paste_ms", "blend_ms",
                     "encode_ms", "write_ms", "frame_ms")}
    infer_n = int(sum(_INFER_CALLS.values())) - infer0
    gpu1 = _gpu_mem_used_mb()
    sess_after = len(_DFM_SESS)
    fps = round((total - start_cursor) / sec, 2) if sec > 0 else 0.0
    prof_out = {
        "per_frame_ms": per,
        "frames": int(prof["frames"]), "faces": int(prof["faces"]),
        "threads": threads, "png_level": png_level,
        "fps": fps, "pure_cpu": True,
        "inference_calls": infer_n,
        "dfm_sessions_before": sess0, "dfm_sessions_after": sess_after,
        "gpu_mem_mb_before": gpu0, "gpu_mem_mb_after": gpu1,
        "wall_s": sec,
    }
    _save(total, done=True, extra={"seconds": sec, "profile": prof_out,
                                   "fps": fps, "inference_calls": infer_n})
    _stage_log("合成完成：第 %d..%d 帧（%d 帧 · 换脸 %d · 原样 %d，无人脸 %d · 失败 %d）· "
               "%.2f s · %.2f 帧/秒"
               % (int(rng["start"]), int(rng["end"]), total, swapped, copied,
                  no_face, failed, sec, fps))
    _stage_log("合成 profile（每帧 ms）：%s"
               % " · ".join("%s=%.2f" % (k, v) for k, v in per.items()))
    _stage_log("合成阶段推理调用：%d（必须为 0）· 显存 %s → %s MB"
               % (infer_n, gpu0, gpu1))
    if infer_n:
        _stage_log("警告：合成阶段发生了 %d 次推理 —— 违反「纯 CPU」约定" % infer_n)
    return {"count": total, "swapped": swapped, "copied": copied, "no_face": no_face,
            "failed": failed, "seconds": sec, "fps": fps,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "inference_calls": infer_n, "profile": per}


def _stage_do_encode(cfg: dict) -> dict:
    """阶段 4 · 导出：frames/（已是合成结果）+ 源音轨 -> <视频名>_merged.mp4。

    帧是 PNG（无损），编码这一层不需要任何 qmin/qmax 兜底：crf16 / yuv420p /
    out_range=tv / 音轨 copy，和原来的成片参数一模一样。写 .part.mp4 再改名，
    中途挂掉不会留下"看起来像成片"的半截文件。
    """
    cache = Path(cfg["cache"])
    video = Path(cfg["video"])
    meta = cfg["meta"]
    st = meta["stages"].setdefault("encode", {})
    frames_all = _stage_files(_stage_frames_dir(cache))
    if not frames_all:
        raise _BadInput("frames/ 里没有帧图 —— 先跑合成", 409)
    # 帧范围：只编码 [start, end] 里的帧 -> 成片时长 = N / fps；音轨从源视频同一位置切
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    rchg = bool(cfg.get("range_changed"))
    frames = _stage_filter_range(rng, frames_all)
    total = len(frames)
    out = Path(cfg["out"])
    if _same_path(out, video):
        raise _BadInput("输出不能是源视频本身：%s" % out)
    if ((not cfg["force"]) and not rchg and st.get("done")
            and out.is_file() and out.stat().st_size > 0):
        _stage_log("导出：复用已有成片 %s（不重跑）" % out)
        return {"skipped": True, "count": total, "out": str(out), "reason": "already-done"}
    if not (meta["stages"].get("merge") or {}).get("done"):
        raise _BadInput("合成（merge）还没完成——现在导出会得到一版没换脸的成片。先跑合成", 409)
    if not frames:
        raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                        % (int(rng["start"]), int(rng["end"])), 409)

    fps = float(cfg.get("fps") or (meta.get("source") or {}).get("fps") or 30.0)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("输出目录建不出来：%s（%s）" % (out.parent, e))
    part = out.with_name(out.stem + ".part" + out.suffix)

    pattern = str(_stage_frames_dir(cache) / "%06d.png")
    # 范围内的第一帧帧号（frames/ 是 000001.png 起的定长序号）
    start_no = (_stage_frame_index(frames[0].stem) or int(rng["start"])) if frames else int(rng["start"])
    dur = (total / fps) if fps > 0 else 0.0
    t0_audio = ((start_no - 1) / fps) if fps > 0 else 0.0
    # 全片：命令和以前**一模一样**（回归靠它）；有范围才加 -start_number / -frames:v / -ss / -t。
    # 命令本身收敛在 _vid_cmd()：快速导出（fused）调的是同一个函数，参数不许各写一套。
    cmd = _vid_cmd(pattern, part, video, fps,
                   img_start=(1 if rng["full"] else int(start_no)),
                   audio_ss=(None if rng["full"] else t0_audio),
                   audio_t=(None if rng["full"] else dur),
                   limit=(None if rng["full"] else int(total)))

    _stage_log("导出：第 %d..%d 帧（%d 帧 @ %.6g fps · 时长 %.4f s）-> %s"
               % (int(start_no), int(start_no) + total - 1, total, fps, dur, out.name))
    if not rng["full"]:
        _stage_log("导出：音轨从源视频 %.4f s 起 copy、长 %.4f s（帧边界误差 ≤ 1 帧）"
                   % (t0_audio, dur))

    _STAGE_METER.set_lanes({"encode": ["e0"]})   # ffmpeg 一个进程 = 一条 lane
    _enc_seen = {"n": 0}

    def _on_prog(done):
        pct = (100.0 * done / total) if total else 0
        _stage_progress("encode", pct, "编码 %d/%d 帧" % (done, total),
                        frames_done=min(done, total), frames_total=total)
        # 交接点：编码 → 成片。一次回调可能跨好几帧（ffmpeg 的 -progress 有节流），
        # 所以按增量补 tick —— 图上的光点才和「真的编了几帧」对得上。
        _n = int(done) - int(_enc_seen["n"])
        if _n > 0:
            _enc_seen["n"] = int(done)
            _stage_tick("encode", _n)

    enc = _vid_encode(cmd, part, total, on_progress=_on_prog)
    code, tail, sec = enc["code"], enc["tail"], enc["seconds"]

    if code != 0:
        try:
            if part.is_file():
                part.unlink()
        except OSError:
            pass
        raise _BadInput("编码失败（ffmpeg 退出码 %s）：%s" % (code, (tail or "")[-600:]), 500)
    if not part.is_file() or part.stat().st_size <= 0:
        raise _BadInput("编码跑完了却没有产物：%s" % part, 500)
    os.replace(str(part), str(out))
    # sec 来自 _vid_encode（它自己量的墙钟）—— 这里不再重算，免得两处口径打架
    mb = round(out.stat().st_size / 1048576.0, 2)
    meta["stages"]["encode"] = {
        "done": True, "count": total, "seconds": sec, "out": str(out), "mb": mb,
        "fps": fps, "crf": 16, "pix_fmt": "yuv420p",
        # 老字段 range = **色彩范围**（tv），别和帧范围混：帧范围在 frame_range/range_key
        "range": "tv", "color_range": "tv",
        "frame_range": {"start": int(start_no), "end": int(start_no) + total - 1},
        "range_key": rkey, "range_total": int(total),
        "duration": round(dur, 6), "audio_ss": round(t0_audio, 6), "audio": "copy",
        "finished_at": time.time(),
        "msg": "导出完成：%s（%.2f MB · 第 %d..%d 帧 · %.3f s）" % (
            out.name, mb, int(start_no), int(start_no) + total - 1, dur),
    }
    _stage_meta_save(cache, meta)
    _stage_log("导出完成：%s（%.2f MB · %.2f s · %d 帧 · 时长 %.4f s）"
               % (out, mb, sec, total, dur))
    _stage_log("编码命令：%s" % " ".join('"%s"' % a if " " in str(a) else str(a) for a in cmd))

    # ---- cache 的一生：成功导出 = 这份缓存没用了，删干净 ----
    removed = _stage_drop_cache(cfg)
    _stage_set(cache_removed=removed)
    return {"count": total, "out": str(out), "seconds": sec, "mb": mb,
            "duration": round(dur, 6), "start_frame": int(start_no),
            "end_frame": int(start_no) + total - 1,
            "range": {"start": int(rng["start"]), "end": int(rng["end"])},
            "cache_removed": removed}


def _stage_drop_cache(cfg: dict) -> bool:
    """导出成功后删掉整个 <名字>_cache/；失败/取消**不删**（留着续跑）。"""
    cache = Path(cfg["cache"])
    if cfg.get("keep_cache"):
        _stage_log("DFL_STAGE_KEEP=1：这次不删 cache（测试用）：%s" % cache)
        return False
    shutil.rmtree(str(cache), ignore_errors=True)
    gone = not cache.exists()
    _stage_log("cache 清理：%s（%s）" % ("已删除" if gone else "没删干净", cache))
    return gone


_STAGE_FN = {
    "frames": _stage_do_frames,
    "faces": _stage_do_faces,
    "pred": _stage_do_pred,
    "mask": _stage_do_mask,
    "merge": _stage_do_merge,
    "encode": _stage_do_encode,
}


# ---------------- 请求 -> cfg ----------------
_STAGE_KEEP = str(os.environ.get("DFL_STAGE_KEEP") or "").strip().lower() in ("1", "true", "yes", "on")
# ③ 换脸一批几张 / ⑤ 合成几个线程 / ⑤ 写 PNG 的压缩级别（1 = 体积换速度）
# ③ 换脸一批几张：**0 = 按可用显存自动选**（_pred_pick_batch）；env 或请求体给了就照用。
# 以前默认写死 8 —— 8GB 卡上 batch 16 会抖到 5 张/秒，12GB 空闲时 8 又浪费一半。
_PRED_BATCH_ENV = str(os.environ.get("DFL_PRED_BATCH") or "").strip()
_PRED_BATCH_DEFAULT = max(0, int(_PRED_BATCH_ENV)) if _PRED_BATCH_ENV.isdigit() else 0


# ⑤ 合成的并发度：**默认 8**（用户口径：8 个 worker，其余核留给训练/预览）。
#
# 以前这里写的是 max(4, min(8, cpu//4)) —— 在 16 核 32 线程上恰好也是 8，但在 8 核机
# 上是 4、在 64 线程服务器上还是 8：和核数无关的常数。现在**默认明确钉在 8**，只是想
# 换就能换（环境变量 / 请求体），别再去改代码。
#
# 线程数只是故事的一半：OpenCV 每个进程默认开 32 条内部线程，8 个 worker × 32 =
# 256 条可运行线程抢 16 个核。实测（1080×1920 · seamless · 120 帧，见 ⑤ 阶段报告）
# process_cpu 8.5 核、吞吐 11 帧/秒，而同样这堆活用 8 个独立进程只花 ~4.5 核就跑到
# 11.4 帧/秒 —— 一半的 CPU 在 OpenCV 的 spin-wait 里空转。所以两个旋钮必须一起给：
#   DFL_MERGE_WORKERS / body.merge_workers        worker 数（默认 8）
#   DFL_MERGE_CV2_THREADS / body.merge_cv2_threads
#                                                每个 worker 里 OpenCV 的线程数
#                                                （不设 = 自动：cpu//workers，上限 4）
_MERGE_WORKERS_DEFAULT = max(1, _env_int("DFL_MERGE_WORKERS",
                                         _env_int("DFL_MERGE_THREADS", 8)))
_MERGE_WORKERS_DEFAULT = min(32, _MERGE_WORKERS_DEFAULT)
# 兼容老名字（别的地方/老脚本还在用 _MERGE_THREADS_DEFAULT）
_MERGE_THREADS_DEFAULT = _MERGE_WORKERS_DEFAULT
_MERGE_CV2_THREADS_DEFAULT = max(0, _env_int("DFL_MERGE_CV2_THREADS", 0))
_MERGE_PNG_LEVEL = max(0, min(9, int(str(os.environ.get("DFL_MERGE_PNG_LEVEL") or "1").strip() or 1)))
# 帧图 JPEG 质量：**固定 100**（4:4:4），界面不再有这个旋钮。
# 理由（用户定的）：默认就是无损优先，低画质没有使用场景，留个能调的旋钮只会误导
# —— 上一版设置页那颗「质量 95」既不可编辑、后端也从来没读过它。
# 老客户端/验收脚本仍然可以传 jpeg_quality 覆盖（1..100）；不传就是 100。
# 注意：**PNG 路径完全不受影响**（无损，只认 png_compression）。
_STAGE_JPEG_QUALITY = 100


# ---------------- 帧范围（1 起、闭区间） ----------------
# 方案 B：**切帧仍然整段**（索引完整、成本极低），后面五个阶段只处理 [start, end]。
# 口径：都不传 = 全片；只给 start = start..结尾；只给 end = 1..end。
# 为什么 frames 也记 range：界面/验收/续跑都要能一眼看出「这一段是哪一段」，
# 而且下游阶段的指纹要带上范围 —— 改了范围，已经跑完的产物必须重跑。
_RANGE_START_KEYS = ("start_frame", "frame_start")
_RANGE_END_KEYS = ("end_frame", "frame_end")


def _range_int(name: str, raw):
    """帧范围里的一个数：None/空 -> None；不是整数 -> 422（不静默兜底）。"""
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    text = str(raw).strip()
    try:
        return int(text, 10)
    except (TypeError, ValueError):
        pass
    try:
        f = float(text)
    except (TypeError, ValueError):
        raise _BadInput("帧范围的 %s 要是整数，收到 %r" % (name, raw), 422)
    if not f.is_integer():
        raise _BadInput("帧范围的 %s 要是整数，收到 %r" % (name, raw), 422)
    return int(f)


def _stage_range_raw(body: dict) -> tuple:
    """请求体 -> (start_raw, end_raw)。认 start_frame/end_frame、frame_start/frame_end、range{}。"""
    body = _as_dict(body)
    s = e = None
    nested = body.get("range")
    if isinstance(nested, dict):
        s = nested.get("start", nested.get("start_frame"))
        e = nested.get("end", nested.get("end_frame"))
    if s is None:
        for k in _RANGE_START_KEYS:
            if body.get(k) is not None:
                s = body.get(k)
                break
    if e is None:
        for k in _RANGE_END_KEYS:
            if body.get(k) is not None:
                e = body.get(k)
                break
    return s, e


def _stage_range_norm(body: dict, total: int) -> dict:
    """请求体 + 源视频总帧数 -> {"start","end","total","all","full","key"}（1 起、闭区间）。

    非法输入一律 422，并说清楚是哪一条不对（调用方只负责把它原样回给界面）：
      start < 1 / end < 1 / start > end / end > 源总帧数。
    """
    total = int(total or 0)
    raw_s, raw_e = _stage_range_raw(body)
    s = _range_int("start_frame", raw_s)
    e = _range_int("end_frame", raw_e)
    if s is None and e is None:
        s, e = 1, (total if total > 0 else 0)
    else:
        if total <= 0:
            raise _BadInput("拿不到源视频的总帧数，没法按帧范围校验（先确认这个视频能解码）", 422)
        s = 1 if s is None else int(s)
        e = total if e is None else int(e)
        if s < 1:
            raise _BadInput("start_frame 从 1 数起，收到 %d" % s, 422)
        if e < 1:
            raise _BadInput("end_frame 从 1 数起，收到 %d" % e, 422)
        if s > e:
            raise _BadInput("帧范围反了：start_frame=%d > end_frame=%d" % (s, e), 422)
        if e > total:
            raise _BadInput("end_frame=%d 超过源视频总帧数 %d" % (e, total), 422)
    n = (int(e) - int(s) + 1) if (int(s) >= 1 and int(e) >= int(s)) else total
    full = bool(total > 0 and int(s) <= 1 and int(e) >= total) or \
        bool(total <= 0 and int(s) <= 1 and int(e) <= 0)
    return {"start": int(s), "end": int(e), "total": int(n), "all": total,
            "full": bool(full),
            "key": ("%d-%d" % (int(s), int(e))) if int(e) >= int(s) else "all"}


def _stage_range(cfg: dict) -> dict:
    """cfg -> 这次的帧范围（老 cfg / 没配就是全片）。"""
    r = cfg.get("range")
    if isinstance(r, dict) and r.get("start") is not None:
        return r
    total = int(cfg.get("total") or 0)
    return {"start": 1, "end": total, "total": total, "all": total,
            "full": True, "key": "all"}


def _stage_sync_range(cfg: dict, stage: str) -> bool:
    """把「这个阶段上一次跑的范围 vs 这次要的范围」算清楚，写回 cfg["range_changed"]。

    改了范围 = 阶段指纹变了：旧游标（相对旧范围数的）作废，这一段要重跑。
    幂等：precheck（单阶段 API）和 worker（链式导出）都调用它，两边口径永远一致。
    切帧永远整段，不存在「范围变了」。
    """
    rng = _stage_range(cfg)
    rkey = str(cfg.get("range_key") or rng.get("key") or "")
    st = ((cfg.get("meta") or {}).get("stages") or {}).get(stage) or {}
    old = str(st.get("range_key") or "")
    chg = bool(old) and old != rkey and stage != "frames"
    cfg["range_changed"] = bool(chg)
    return chg


def _stage_frame_index(key) -> int:
    """帧键（000123 / 000123_00 / Path）-> 1 起的帧号；认不出来返回 0。"""
    name = key.stem if isinstance(key, Path) else str(key or "")
    head = name.split("_")[0].strip()
    try:
        return int(head, 10)
    except (TypeError, ValueError):
        return 0


def _stage_in_range(rng: dict, key) -> bool:
    """这一帧在范围内吗。帧号认不出来的一律**保留**（宁可多跑一次，别把产物漏掉）。"""
    i = _stage_frame_index(key)
    if i <= 0:
        return True
    return int(rng["start"]) <= i <= int(rng["end"])


def _stage_filter_range(rng: dict, paths, keyfn=None) -> list:
    """按帧范围过滤一列文件（默认用 stem，对齐脸传 keyfn 取 000123_00 的前半段）。"""
    out = []
    for p in paths:
        k = keyfn(p) if keyfn else p
        if _stage_in_range(rng, k):
            out.append(p)
    return out


def _stage_dfm_pair(body: dict, need: bool):
    """dfm/模型名：合成阶段必须有；切帧/切脸/导出只是顺带记进 meta（没有也能跑）。"""
    explicit = bool(str(body.get("dfm") or "").strip()) or bool(str(body.get("model") or "").strip())
    if not explicit:
        explicit = bool(list((Path(WORKSPACE) / "model").glob("*.dfm")))
    if need or explicit:
        return _dfm_of(body)
    return "", ""


def _stage_out_path(video: Path, rng: dict, body: dict) -> Path:
    """输出文件名：带范围后缀（xxx_merged_0100-0200.mp4），免得把全片结果盖掉；全片不带。

    分段链（_stage_cfg）和快速导出（_fused_cfg）**同一条规则**：用户点的是同一个
    「输出路径」，两条路不该给出不一样的文件名。
    """
    tag = "" if rng["full"] else "_%04d-%04d" % (int(rng["start"]), int(rng["end"]))
    out_raw = str(body.get("output_dir") or body.get("output") or "").strip()
    if out_raw:
        op = _norm_path(out_raw)
        if op is None:
            raise _BadInput("output_dir 不合法：%s" % out_raw)
        if op.suffix.lower() == ".mp4":
            out = op if not tag else op.with_name(op.stem + tag + op.suffix)
        else:
            if op.exists() and op.is_file():
                raise _BadInput("输出路径是一个已存在的文件：%s（要目录或 .mp4）" % op)
            out = op / (video.stem + "_merged" + tag + ".mp4")
    else:
        out = video.parent / (video.stem + "_merged" + tag + ".mp4")
    if out.is_dir():
        raise _BadInput("输出路径是个目录：%s（要 .mp4 文件）" % out)
    return out


def _stage_cfg(body: dict, stage: str, chain: bool = False) -> dict:
    """请求体 -> 这一次阶段运行的 cfg（也会把 meta.json 读出来、缺字段补齐）。"""
    body = _as_dict(body)
    video = _norm_path(body.get("video") or body.get("source_video"))
    cache = _norm_path(body.get("cache_dir"))
    if video is None and cache is not None:
        base = cache.name
        if base.lower().endswith("_cache"):
            base = base[:-6]
        for ext in sorted(_VIDEO_EXT):
            cand = cache.parent / (base + ext)
            if cand.is_file():
                video = cand
                break
        if video is None:
            raise _BadInput("从 cache_dir 反推不到源视频：%s（同目录没有 %s*）" % (cache, base), 404)
    if video is None:
        raise _BadInput("缺少 video：要一个视频文件路径（或 cache_dir）")
    if not video.is_file():
        raise _BadInput("视频不存在：%s" % video, 404)
    if video.suffix.lower() not in _VIDEO_EXT:
        raise _BadInput("不是视频文件：%s（只认 %s）"
                        % (video.suffix or "无扩展名", " ".join(sorted(_VIDEO_EXT))))
    if cache is None:
        cache = Path(str(video.parent / (video.stem + "_cache")))
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _BadInput("cache 目录建不出来：%s（%s）" % (cache, e))

    probe = _vid_probe(video)
    # 帧范围：1 起、闭区间；都不传 = 全片。非法（start>end / start<1 / end>总帧数）一律 422。
    rng = _stage_range_norm(body, int(probe.get("frames") or 0))
    rkey = str(rng["key"])
    settings = _settings_val(body.get("settings"))
    meta = _stage_meta_load(cache)
    if not meta:
        meta = _stage_meta_blank(video, probe, settings)
    meta = _stage_meta_ready(meta, video, probe)
    if settings:
        meta["settings"] = dict(settings)
    # 范围进 meta（界面/验收/续跑都读它）；只有真的变了才写盘，别每次点都改 meta。
    prev_rng = meta.get("range") if isinstance(meta.get("range"), dict) else {}
    meta["range"] = {"start": int(rng["start"]), "end": int(rng["end"]),
                     "total": int(rng["total"]), "frames": int(rng["all"]),
                     "full": bool(rng["full"]), "key": rkey}
    if (int(prev_rng.get("start") or 0) != int(rng["start"])
            or int(prev_rng.get("end") or 0) != int(rng["end"])
            or int(prev_rng.get("frames") or 0) != int(rng["all"])):
        _stage_meta_save(cache, meta)

    fps = _float_val("fps", body.get("fps"), 1.0, 240.0, None)
    dfm, model_name = _stage_dfm_pair(body, need=(stage == "merge"))
    xseg = _xseg_of(body)

    out = _stage_out_path(video, rng, body)

    snap = dict(_cache_snap())
    snap.update({"mode": "video", "path": str(video), "dir": "", "names": [], "srcdir": ""})
    library_raw = str(body.get("faces_project") or body.get("face_library")
                      or body.get("library") or "").strip()
    plan_payload = {"project": library_raw, "faces_project": library_raw}
    face_thr = _float_val("face_threshold", body.get("face_threshold"), 0.0, 1.0, None)
    if face_thr is not None:
        plan_payload["face_threshold"] = float(face_thr)

    models = []
    if body.get("models"):
        models = _models_of(body) or []
    if not models and dfm:
        models = [{"name": str(model_name), "dfm": str(dfm), "faces": [], "index": 0}]
    if not models:
        raise _BadInput("没有可用的 .dfm 模型（合成阶段要它）：先在训练页把模型导出成 .dfm（%s）"
                        % (Path(WORKSPACE) / "model"), 409)
    if not model_name:
        model_name = str(models[0].get("name") or "")

    cfg = {
        "video": str(video), "cache": str(cache), "out": str(out),
        "total": int(probe.get("frames") or 0),
        # 帧范围（1 起闭区间）：range_total = 后五个阶段真正要处理的帧数
        "range": rng, "range_key": rkey, "range_total": int(rng["total"]),
        "range_full": bool(rng["full"]), "range_changed": False,
        "fps": float(fps if fps else (probe.get("fps") or 0.0)),
        "probe": probe, "settings": dict(settings), "meta": meta,
        "force": bool(_bool_val(body.get("force"), False)),
        "keep_cache": bool(_STAGE_KEEP or _bool_val(body.get("keep_cache"), False)),
        "detector": _detector_val(body.get("detector")),
        "det_align": _align_detector(body)[0],
        "det_align_src": _align_detector(body)[1],
        "thresh": _float_val("thresh", body.get("thresh"), 0.0, 1.0, 0.5),
        "align_thresh": _float_val("align_thresh", body.get("align_thresh"), 0.0, 1.0, 0.5),
        "dfm": str(dfm), "model": str(model_name), "xseg": str(xseg),
        "mask_mode": int((settings or {}).get("mask_mode", 6) or 0),
        "mask_src": _mask_src(settings, (settings or {}).get("mask_mode")),
        "config": dict(_merge_settings_to_config(settings)),
        "snap": snap, "plan_payload": plan_payload, "single_models": models,
        "use_library": True, "chain": bool(chain), "stages": [stage],
        "library": library_raw, "face_threshold": face_thr,
        # ③ 的批大小 / ⑤ 的线程数与 PNG 压缩级别（都能从请求体或环境变量调）
        "pred_batch": _int_val("batch", body.get("batch") or body.get("pred_batch"), 0, 64,
                               _PRED_BATCH_DEFAULT),
        # ⑤ 的并发度：默认 8（用户口径），请求体 merge_workers / 老名字 threads、
        # merge_threads 都认；环境变量 DFL_MERGE_WORKERS / DFL_MERGE_THREADS 兜底。
        "merge_workers": _int_val(
            "merge_workers",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _MERGE_WORKERS_DEFAULT),
        # 每个 worker 里 OpenCV 的线程数（0 = 按核数/worker 数自动算）
        "merge_cv2_threads": _int_val("merge_cv2_threads", body.get("merge_cv2_threads"),
                                      0, 64, _MERGE_CV2_THREADS_DEFAULT),
        # 兼容：老代码/老日志还读 merge_threads
        "merge_threads": _int_val(
            "merge_threads",
            body.get("merge_workers") if body.get("merge_workers") is not None
            else (body.get("threads") if body.get("threads") is not None
                  else body.get("merge_threads")),
            1, 32, _MERGE_WORKERS_DEFAULT),
        "png_level": _int_val("png_compression", body.get("png_compression"), 0, 9,
                              _MERGE_PNG_LEVEL),
        # 帧图 JPEG 质量：固定 100（界面没有旋钮）；老客户端传了就认，但**默认 100**
        "jpeg_quality": _int_val("jpeg_quality", body.get("jpeg_quality"), 1, 100,
                                 _STAGE_JPEG_QUALITY),
        # ③④ 两条流水线的并行度（0 = 自动；验收/测速脚本可以点名覆盖）
        "pred_readers": _int_val("pred_readers", body.get("pred_readers"), 0, 16, 0),
        "pred_writers": _int_val("pred_writers", body.get("pred_writers"), 0, 16, 0),
        "mask_readers": _int_val("mask_readers", body.get("mask_readers"), 0, 16, 0),
        "mask_writers": _int_val("mask_writers", body.get("mask_writers"), 0, 16, 0),
        "mask_batch": _int_val("mask_batch", body.get("mask_batch"), 0, 32, 0),
    }
    # 两个用户可调的性能旋钮（设置页「性能」区）：请求体 -> 进程 env + cfg["knobs"]。
    # 分段链的合成只有 CPU 实现，所以这里 refine 会把「选了 gpu」改回 cpu 并带一条
    # [eff-warn] 文案（[eff] 那行照实写 cpu，不让界面和日志各说一套）。
    _kn = _knobs.refine_compose(_knobs.resolve(body),
                                str((cfg.get("config") or {}).get("mode") or ""),
                                fused=False)
    for _t in _knobs.apply(_kn):
        _knobs.warn_line(_t)
    cfg["knobs"] = _kn
    return cfg


def _stage_precheck(cfg: dict, stage: str) -> dict:
    """阶段前的闸门：缺前置产物就 409；已完成就 skip（幂等）；frames 被覆盖过就拒。

    六步链：frames → faces → pred → mask → merge → encode，每一步只读它前面那几步的
    产物。少跑一步就 409 点名前一步（不再"偷偷在合成里现算"）。
    """
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    frames = _stage_files(_stage_frames_dir(cache))
    aligned_all = _stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))
    # 帧范围：后五个阶段只认 [start, end] 里的帧；下面的 skip / 覆盖判定全按范围内的数算
    rng = _stage_range(cfg)
    frames_in = _stage_filter_range(rng, frames)
    aligned = _stage_filter_range(rng, aligned_all, lambda p: p.stem.split("_")[0])
    st = meta["stages"].get(stage) or {}
    force = bool(cfg.get("force"))
    # 上一次这个阶段跑的是另一段范围 -> 旧游标（相对旧范围数）作废：自动重跑，不是 409。
    # 产物是按帧号落盘的，换了范围就把这段重算一遍 —— 日志里会说清楚（见 merge_stage）。
    chg = _stage_sync_range(cfg, stage)

    def _need(stage_name: str, why: str = ""):
        if not (meta["stages"].get(stage_name) or {}).get("done"):
            raise _BadInput("%s还没完成 —— 先跑%s（POST /api/merge/stage/%s）%s"
                            % (_STAGE_LABEL.get(stage_name, stage_name),
                               _STAGE_LABEL.get(stage_name, stage_name), stage_name,
                               ("：" + why) if why else ""), 409)

    def _gap(up: str, label: str):
        """上游阶段只跑过某一段：盖不住这次要的范围就 409，并点名要重跑哪一个。"""
        ust = meta["stages"].get(up) or {}
        u = ust.get("range") if isinstance(ust.get("range"), dict) else {}
        if not u or up == "frames":
            return                       # frames 永远整段切；老 meta 没写范围 = 整段
        us = int(u.get("start") or 1)
        ue = int(u.get("end") or 0)
        if us <= int(rng["start"]) and (ue <= 0 or ue >= int(rng["end"])):
            return
        raise _BadInput("%s只处理了第 %d..%d 帧，这次要第 %d..%d 帧 —— 先用同样的范围重跑%s"
                        "（force=true），别拿别的范围的结果凑"
                        % (label, us, ue, int(rng["start"]), int(rng["end"]), label), 409)

    if stage == "frames":
        # 「已经切完了」先判：续跑时 frames 往往已经被合成结果覆盖过，
        # 这一步本来就无事可做（跳过），不该拿 frame_swapped 去拦它。
        if st.get("done") and frames and not force:
            hint = int((meta.get("source") or {}).get("frames") or 0)
            if not hint or len(frames) == hint:
                return {"skip": True, "reason": "already-done", "count": len(frames)}
        if meta.get("frame_swapped") and not force:
            raise _BadInput("frames/ 已经被合成结果覆盖（meta.frame_swapped=true）—— 重切会把原帧"
                            "盖回来、却留下「已合成」的标记。要重来请传 force=true："
                            "会清掉 frames/ + aligned/ + pred/ + mask/ 和后面几个阶段的标记。", 409)
        return {"skip": False}

    if stage == "faces":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧（POST /api/merge/stage/frames）", 409)
        if (st.get("done") and not force and not chg
                and int(st.get("frames") or 0) == len(frames) and aligned_all):
            return {"skip": True, "reason": "already-done", "count": len(aligned)}
        return {"skip": False}

    if stage == "pred":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧（POST /api/merge/stage/frames）", 409)
        _need("faces")
        _gap("faces", "切脸")
        if not aligned:
            raise _BadInput("aligned/ 里没有第 %d..%d 帧的对齐脸 —— 先用同样的范围跑切脸"
                            % (int(rng["start"]), int(rng["end"])), 409)
        groups = len(_stage_face_rows(aligned))
        if (st.get("done") and not force and not chg and int(st.get("done_frames") or 0) >= groups):
            return {"skip": True, "reason": "already-done", "count": int(st.get("count") or 0)}
        return {"skip": False}

    if stage == "mask":
        _need("faces")
        _need("pred", "遮罩阶段要拿换脸结果去跑 XSeg")
        _gap("faces", "切脸")
        _gap("pred", "换脸")
        if not aligned:
            raise _BadInput("aligned/ 里没有第 %d..%d 帧的对齐脸 —— 先用同样的范围跑切脸"
                            % (int(rng["start"]), int(rng["end"])), 409)
        src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
        sig = _mask_signature(src, cfg)
        cfg["mask_src"] = src
        if (st.get("signature") and st.get("signature") != sig and not force
                and int(st.get("done_frames") or 0) > 0):
            raise _BadInput("遮罩来源变了（%s → %s）：mask/ 里的东西已经失效，不能静默复用。"
                            "传 force=true 重跑遮罩阶段" % (st.get("signature"), sig), 409)
        if (st.get("done") and not force and not chg and st.get("signature") == sig
                and int(st.get("done_frames") or 0) >= len(aligned)):
            return {"skip": True, "reason": "already-done", "count": int(st.get("count") or 0)}
        return {"skip": False}

    if stage == "merge":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧", 409)
        if not frames_in:
            raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                            % (int(rng["start"]), int(rng["end"])), 409)
        _need("faces")
        _need("pred")
        _need("mask")
        _gap("faces", "切脸")
        _gap("pred", "换脸")
        _gap("mask", "遮罩")
        src = _mask_src(cfg.get("settings"), cfg.get("mask_mode"))
        sig = _mask_signature(src, cfg)
        mst = meta["stages"].get("mask") or {}
        if mst.get("signature") and mst.get("signature") != sig:
            raise _BadInput("遮罩来源变了（mask/ 是 %s 产出的，现在是 %s）—— 先重跑遮罩阶段"
                            "（force=true），别拿旧遮罩合成" % (mst.get("signature"), sig), 409)
        if st.get("done") and not force and not chg:
            return {"skip": True, "reason": "already-done", "count": len(frames_in)}
        if meta.get("frame_swapped") and int(st.get("done_frames") or 0) == 0:
            raise _BadInput("frames/ 已经是合成结果（frame_swapped=true）但没有续跑位置 —— "
                            "只能重跑切帧 + 切脸 + 换脸 + 遮罩", 409)
        return {"skip": False}

    if stage == "encode":
        if not frames:
            raise _BadInput("frames/ 是空的 —— 先跑切帧 + 合成", 409)
        if not frames_in:
            raise _BadInput("frames/ 里没有第 %d..%d 帧 —— 这个范围超出了已切好的帧"
                            % (int(rng["start"]), int(rng["end"])), 409)
        _need("merge")
        _gap("merge", "合成")
        out = Path(cfg["out"])
        if st.get("done") and out.is_file() and out.stat().st_size > 0 and not force and not chg:
            return {"skip": True, "reason": "already-done", "count": len(frames_in)}
        return {"skip": False}

    return {"skip": False}


def _stage_counts(cache) -> dict:
    """各阶段产物数量（status 里给界面/验收看）。"""
    pred_dir = _stage_pred_dir(cache)
    mask_dir = _stage_mask_dir(cache)
    return {"frames": len(_stage_files(_stage_frames_dir(cache))),
            "faces": len(_stage_files(_stage_aligned_dir(cache), (".jpg", ".jpeg"))),
            "pred": len([x for x in _stage_files(pred_dir) if "_prdm" not in x.name
                         and "_dstdm" not in x.name]),
            "masks": len(_stage_files(mask_dir))}


def _stage_stages_meta(meta: dict) -> dict:
    st = meta.get("stages") if isinstance(meta, dict) else {}
    if not isinstance(st, dict):
        st = {}
    return {k: (dict(st.get(k)) if isinstance(st.get(k), dict) else {}) for k in _STAGE_ORDER}


def _stage_done_list(meta: dict) -> list:
    st = _stage_stages_meta(meta)
    return [k for k in _STAGE_ORDER if st.get(k, {}).get("done")]


def _stage_next(meta: dict) -> str:
    st = _stage_stages_meta(meta)
    for k in _STAGE_ORDER:
        if not st.get(k, {}).get("done"):
            return k
    return ""


def _stage_brief_meta(cfg: dict, extra: dict = None) -> dict:
    cache = Path(cfg["cache"])
    meta = cfg["meta"]
    out = {
        "stage": str(cfg["stages"][0]) if cfg.get("stages") else "",
        "video": str(cfg.get("video") or ""),
        "cache_dir": str(cache),
        "out_video": str(cfg.get("out") or ""),
        "counts": _stage_counts(cache),
        "stages": _stage_stages_meta(meta),
        # 流水线实况（进度页那张 SVG 图的数据源）。这里不能叫 stages —— 那个键从 4.0
        # 起就是「每段的元数据」（done/msg/…），两个形状混一个键名是最容易出幻觉的坑。
        "stages_live": _STAGE_METER.snapshot(),
        "stages_done": _stage_done_list(meta),
        "next_stage": _stage_next(meta),
        "frame_swapped": bool(meta.get("frame_swapped")),
        "meta": meta,
        "dfm": str(cfg.get("dfm") or ""),
        "model": str(cfg.get("model") or ""),
        "detector": str(cfg.get("detector") or ""),
        "align_detector": str(cfg.get("det_align") or ""),
        "fps": float(cfg.get("fps") or 0.0),
        "total": int(cfg.get("total") or 0),
        # 帧范围：range 是这次要处理的一段，range_total = 这一段多少帧（相对进度的分母）
        "range": dict(cfg.get("range") or {}),
        "range_total": int(cfg.get("range_total") or 0),
        "range_changed": bool(cfg.get("range_changed")),
        "settings": dict(cfg.get("settings") or {}),
        # 帧图编码口径（JSON 回显）：PNG 无损，JPEG 固定 100 —— 界面没有这个旋钮，
        # 但验收/排障要能从响应里读到真实值，不用去翻 meta.json。
        "jpeg_quality": int(cfg.get("jpeg_quality") or _STAGE_JPEG_QUALITY),
        "png_level": int(cfg.get("png_level") if cfg.get("png_level") is not None else 1),
        "image_format": str((cfg.get("settings") or {}).get("image_format") or "jpg"),
    }
    if extra:
        out.update(extra)
    return out


# ---------------- 线程入口 ----------------
def _stage_worker(cfg: dict):
    """跑一个或一串阶段；异常一律落成终态，cache/缓存按口径处理。"""
    stages = list(cfg.get("stages") or ["frames"])
    ok, err, cancelled = True, "", False
    try:
        _STAGE_METER.begin()          # 流水线实况：这一单开始记（串行六段累同一份）
        for stage in stages:
            _stage_check_cancel()
            # 链式导出不走 precheck：范围指纹/游标失效必须在这里也算一遍（和单阶段口径一致）
            _stage_sync_range(cfg, stage)
            _stage_log("===== 阶段 %s（%s）开始 =====" % (stage, _STAGE_LABEL.get(stage, stage)))
            t0 = time.time()
            res = _STAGE_FN[stage](cfg) or {}
            sec = round(time.time() - t0, 2)
            if res.get("skipped"):
                _stage_log("阶段 %s：已完成，跳过（%s）" % (stage, res.get("reason") or ""))
            else:
                _stage_log("阶段 %s 完成：%s · %.2f s" % (stage, _stage_brief(res), sec))
            if cfg.get("chain") and stage == "encode" and res.get("out"):
                with _merge_job_lock:            # 老 /export/status 的 encode 段
                    _merge_job["encode"] = {"seconds": sec, "code": 0,
                                            "out": str(res.get("out")),
                                            "mb": res.get("mb")}
            with _STAGE_LOCK:
                secs = dict(_STAGE_JOB.get("seconds") or {})
                if not res.get("skipped"):
                    secs[stage] = sec
                _STAGE_JOB["seconds"] = secs
                done = list(_STAGE_JOB.get("stages_done") or [])
                if stage not in done:
                    done.append(stage)
                _STAGE_JOB["stages_done"] = done
                _STAGE_JOB["skipped"] = (_STAGE_JOB.get("skipped") or []) + \
                    ([stage] if res.get("skipped") else [])
                if res.get("out"):
                    _STAGE_JOB["out_video"] = str(res["out"])
                if res.get("cache_removed"):
                    _STAGE_JOB["cache_removed"] = True
    except _StageCancel:
        ok, cancelled = False, True
        err = "已取消（已完成的阶段和产物都保留，下次从这里接着跑）"
    except _BadInput as e:
        ok, err = False, str(e.msg)[:1200]
    except Exception as e:                                  # noqa: BLE001
        traceback.print_exc()
        ok, err = False, "%s: %s" % (e.__class__.__name__, str(e)[:600])
    finally:
        # 强制尝试失败：先回滚现场（放掉本进程会话 + 重算当下的账 + 改写 err），
        # 再往 meta/status 落字 —— 顺序反了用户就会看到「导出失败」而不是「确实不够」。
        if (not ok) and (not cancelled) and cfg.get("vram_force"):
            _rb = _vram_forced_rollback(
                str(_STAGE_LABEL.get(str(_STAGE_JOB.get("stage") or cfg.get("stage") or ""),
                                     "阶段")),
                _vram_models_mb([str(cfg.get("dfm") or "")]),
                {"mode": "none", "procs": 1}, err)
            if _rb:
                cfg["vram_oom"] = _rb
                err = str(_rb.get("msg") or err)
        cache = Path(cfg["cache"])
        meta = cfg.get("meta") or {}
        # 失败/取消：把状态和错误写进 meta（断电续跑时一眼能看出停在哪），cache 保留
        if not ok:
            try:
                if not isinstance(meta, dict):
                    meta = {}
                meta["last_error"] = err
                meta["last_stage"] = str(_STAGE_JOB.get("stage") or "")
                _stage_meta_save(cache, meta)
            except Exception:                               # noqa: BLE001
                pass
        else:
            try:
                _stage_meta_save(cache, meta)
            except Exception:                               # noqa: BLE001
                pass
        # ---- 缓存语义：整条链结束（成功/失败/取消）都清空组均值 + auto_plan ----
        cache_state = clear_face_group_cache("分段导出完成" if ok
                                             else ("分段导出取消" if cancelled else "分段导出失败"))
        _stage_log("缓存已清空：组均值 %s 条 · 其它 %s · 磁盘缓存文件 %s 个"
                   % ((cache_state or {}).get("profiles"),
                      (cache_state or {}).get("extra") or "无",
                      len((cache_state or {}).get("files") or [])))
        chain = bool(cfg.get("chain"))
        _oom = dict(cfg.get("vram_oom") or {})
        with _STAGE_LOCK:
            pct = 100 if ok else int(_STAGE_JOB.get("pct") or 0)
            if _oom:
                vp = dict(_oom.get("plan") or _STAGE_JOB.get("vram_plan") or {})
                vp["forced_failed"] = True
                _STAGE_JOB["vram_plan"] = vp
            _STAGE_JOB["vram_oom"] = (_oom or None)
            _STAGE_JOB.update({
                "running": False, "ok": bool(ok), "cancelled": bool(cancelled),
                "phase": ("done" if ok else ("cancelled" if cancelled else "failed")),
                "finished_at": time.time(), "error": "" if ok else err,
                "error_detail": "" if ok else "\n".join(list(_STAGE_JOB.get("log") or [])[-30:]),
                "pct": pct, "stage_pct": (100 if ok else int(_STAGE_JOB.get("stage_pct") or 0)),
                "cache_removed": (not cache.exists()),
                "cache": cache_state,
                "msg": (("%s 完成" % _STAGE_LABEL.get(str(_STAGE_JOB.get("stage") or ""), "导出"))
                        if ok else ("已取消" if cancelled else "失败：" + err[:80])),
                "meta": meta,
            })
        if chain:
            with _merge_job_lock:
                _merge_job.update({
                    "running": False, "ok": bool(ok) and not cancelled,
                    "cancelled": bool(cancelled), "finished_at": time.time(),
                    "pct": (100 if ok else 0),
                    "pct_at_failure": (None if ok else int(_merge_job.get("pct") or 0)),
                    "stage": (6 if ok else -1),
                    "msg": ("合成完成" if ok else ("已取消" if cancelled else "合成失败")),
                    "error": ("" if ok else err),
                    "error_detail": ("" if ok else "\n".join(_log_tail(
                        list(_merge_job.get("log") or []), 30))[:4000]),
                    "cache": cache_state,
                    # active=false = 这一单真的收尾完了（分段/快速导出共用一个位，
                    # 前端靠它决定「能不能再点一次导出」）
                    "active": False,
                })
            # 同一条理由：分段链收尾也把本进程会话放掉（下一次预览/导出懒加载）
            try:
                _vram_release_inproc("分段导出结束")
            except Exception:                               # noqa: BLE001
                pass
            _vid_close_log()


def _stage_gpu_paths(cfg: dict, stage: str) -> list:
    """这个阶段**真要加载**的 GPU 模型（没有就不算显存账）。

    按阶段点名，不按"导出"一把梭：
      · pred / merge —— 每个选中模型一份 dfm session（_dfm_session 常驻）
      · mask         —— XSeg 遮罩 onnx（来源是 DFM 时本阶段本来就跳过）
      · faces        —— 检测器 + 2D106 关键点（onnx，走 CUDA EP）
      · frames/encode —— 一个 GPU 模型都不加载（ffmpeg 干活），不列
    """
    if stage in ("pred", "merge"):
        return [str(m.get("dfm") or "") for m in (cfg.get("single_models") or [])
                if str(m.get("dfm") or "")]
    if stage == "mask":
        src = cfg.get("mask_src") or {}
        if str(src.get("source") or "").lower() == "xseg" and str(cfg.get("xseg") or ""):
            return [str(cfg["xseg"])]
        return []
    if stage == "faces":
        # 2D106 是纯 CPU（见 _landmarks_model 的注释），不算显存账；只算检测器。
        try:
            flat = _app_mod._DET_ONNX[_detector_val(cfg.get("det_align"))]
            p = _app_mod._detector_path(flat)
        except Exception:                                   # noqa: BLE001
            p = None
        return [str(p)] if p else []
    return []


def _stage_model_resident(stage: str, paths: list, cfg: dict) -> bool:
    """这个阶段要的那份模型是不是**已经在当前进程里**了（缓存命中 = 不再新占显存）。

    阶段链是同一个后端进程里连着跑的：pred 建的 dfm session、faces 加载的检测器、
    mask 建的 XSeg 都常驻在缓存里。闸门如果还把「一份新会话」算进需求，等于把已经
    吃掉的那 2GB 再算一遍 —— 于是第 3 个阶段开始必被拒（而它其实一个字节都不用再分配）。
    """
    paths = [str(p) for p in (paths or []) if str(p or "")]
    if not paths:
        return False
    if stage in ("pred", "merge"):
        live = {str(k).split("|")[0] for k in list(_DFM_SESS)}
        return all(p in live for p in paths)
    if stage == "faces":
        return _det_cached(cfg.get("det_align"))
    if stage == "mask":
        try:
            with _XSEG_SESS_LOCK:
                live = {str(k[1]) for k in list(_XSEG_SESS) if len(k) > 1}
            return all(p in live for p in paths)
        except Exception:                                   # noqa: BLE001
            return False
    return False


def _stage_vram_gate(cfg: dict, stage: str, force: bool = False,
                     dry: bool = False) -> dict:
    """阶段起手前的显存闸门 —— **在起 job / 加载模型 / 开线程之前**跑。

    用户口径（截图 2）：显存不够时点导出，不许先起一个 job 再把六个阶段全打失败。
    所以这里只算账（_vram_plan，不加载任何模型），装不下就抛 400 + 结构化字段。

    frames / encode 不加载 GPU 模型，但它们同样要先过一遍账：
    连「一份 CUDA context + 一帧的解码缓冲」都放不下时，后面五个阶段一个也跑不成，
    与其让用户在第二步才吃到 400，不如第一步就说清楚。
    """
    label = _STAGE_LABEL.get(stage, stage)
    paths = _stage_gpu_paths(cfg, stage)
    opts = {"mode": "none", "procs": 1, "label": label,
            "aux": bool(stage not in ("frames", "encode"))}
    # 这一份会话是不是**已经**在本进程里了（pred → mask → merge 是同一条链，上一个
    # 阶段加载的 dfm 还在 _DFM_SESS 里）。在的话就不再算一份新会话 —— 那 2GB 已经在
    # 卡上、也已经在「空闲」里扣过了，再算一遍就是重复计数（阶段 3 之后必被拒）。
    try:
        if paths and _stage_model_resident(stage, paths, cfg):
            opts["resident"] = True
    except Exception:                                       # noqa: BLE001
        pass
    # 真装不下时允许放掉本进程里**别人的**会话（没有任务在跑才敢，见 _vram_plan）
    opts["reclaim_inproc"] = True
    # 真起手（非 dry_run）：先卸常驻预览 worker 的模型 → 再探 → 再算账（顺序见 _vram_plan）。
    # 分段链每个阶段都过这道闸，但 worker 里第一次之后就空了，第二次起是零成本的空转。
    opts["reclaim_first"] = (not dry)
    # force：「仍然尝试（可能 OOM）」—— 只放行最小档，真 OOM 由 _stage_worker 回滚
    opts["force"] = bool(force)
    return _vram_guard(paths, opts, label=label, force=bool(force))


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 stages 与 _app_mod、app、clear_face_group_cache、compose、media_encoder、media_frames、meter、pipeline_video、preview、vram
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import app as _app_mod

from ..app import (
    WORKSPACE, _merge_job, _merge_job_lock, _merge_settings_to_config,
)

from .._face_group import clear_face_group_cache

from .compose import (
    _AUTO_EMB_LIMIT, _auto_library, _cache_snap, _det_run_lock, _detector, _detector_val,
    _dfm_by_name, _landmarks_model,
)

from .._media.encoder import (
    _vid_cmd, _vid_probe,
)

from .._media.frames import (
    _VIDEO_EXT,
)

from . import knobs as _knobs

from .meter import (
    _STAGE_METER, _StageTicks, _stage_tick,
)

from .pipeline_video import (
    _log_tail, _vid_cancelled, _vid_close_log, _vid_encode, _vid_log,
)

from .preview import (
    _det_cached, _dfm_of, _models_of, _xseg_of,
)

from .vram import (
    _PRED_VRAM_SAFETY, _VRAM_CPU_SLOW, _dfm_backend_sel, _ensure_ort_dll_dirs,
    _gpu_mem_used_mb, _is_oom,
    _pred_pick_batch, _sess_batch_cap, _vram_cpu_on, _vram_forced_rollback, _vram_guard,
    _vram_measure_begin, _vram_measure_end, _vram_models_mb, _vram_oom_inject_on,
    _vram_release_inproc,
)
