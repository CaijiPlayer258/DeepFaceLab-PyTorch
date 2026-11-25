# -*- coding: utf-8 -*-
"""效能档位回显：把 cfg 里「实际生效的值」读出来给界面。（文件上半是 _eff.py 原身）

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from ._shared import (
    DEFAULT_BLEND, DEFAULT_CT_MODE, _DEFAULT_DETECTOR,
)


# ---------------------------------------------------------------- 字段表
# 顺序就是 [eff] 行里的顺序：先「选了什么模型/检测器/遮罩」，再「怎么合成」，
# 最后是「这次跑多大」。
FIELDS = (
    "detector", "mask_source", "ct_mode", "blend", "erode", "blur",
    "models", "faces_project", "library_enabled", "fast_export",
    "frame_range", "workers", "batch", "jpeg_quality",
    # 用户可调的两个性能旋钮（设置页「性能」区那两个下拉）。它们的值不是 cfg 字段，
    # 而是由 backend_merge/knobs.py 从请求体翻译成进程 env —— 来源也多一档 "auto"
    # （界面选了「自动」：档位交给后端定，不写死成具体值）。
    "dfm_dtype", "compose_backend",
)


# 界面默认值（index.html 里控件的初始值）。**只用来对账**：请求体缺字段时，
# 拿后端的默认值和它比，不一致就说明「界面显示的和实际跑的」可能不是一回事。
UI_DEFAULTS = {
    "detector": "tinymog",          # <select id="mgDetector"> 第一项 TinyMog
    "mask_source": "XSegLite",      # .seg[data-k="seg"] 默认 on
    "ct_mode": "mt",                # <select id="mgColor"> 第一项 mt
    "blend": "seamless",            # <select id="mgMode"> selected
    "erode": 0,                     # #mgErode value=0
    "blur": 4,                      # #mgBlur value=4
    "models": [],                   # 左栏选中的模型（空 = 没选）
    "faces_project": "",            # 人脸库
    "library_enabled": True,
    "fast_export": True,            # 「快速导出」默认开
    "frame_range": "全部",           # #mgFrameFrom/#mgFrameTo 都是 0
    "workers": None,                # #mgWorkers value=0 = 自动
    "batch": None,                  # 界面没有这个旋钮
    "jpeg_quality": 100,            # 界面没有这个旋钮，固定 100
    # 设置页「性能」区两个下拉的初始值。dfm_dtype 的「自动」= 不写死档位（后端默认档，
    # 落到哪一档看 [eff] 的 dfm 列：torch · bf16w）；compose_backend 的「自动」就是
    # 现在这条 CPU 路径（GPU 合成要显式选，docs/26 实测不是处处更快）。
    "dfm_dtype": "auto",
    "compose_backend": "cpu",
}


# 界面上的中文名（警告里说「界面可能显示 XXX」时用它，而不是内部规范名）
UI_LABELS = {
    "detector": "TinyMog",
    "mask_source": "XSegLite",
    "ct_mode": "mt",
    "blend": "seamless",
    "erode": "0",
    "blur": "4",
    "models": "未选模型",
    "faces_project": "未选人脸库",
    "library_enabled": "开",
    "fast_export": "开",
    "frame_range": "全部",
    "workers": "自动",
    "batch": "自动",
    "jpeg_quality": "100",
    "dfm_dtype": "自动",
    "compose_backend": "自动（CPU）",
}


# 请求体里每个字段可能出现的键（settings 里的键按 settings.xxx 写）。
# 只用于判断「请求体有没有明确给这个字段」，以及 [req-unused] 的白名单。
BODY_KEYS = {
    "detector": ("detector",),
    "mask_source": ("settings.mask_source", "settings.mask_seg", "settings.seg_source",
                    "settings.seg", "settings.mask_model", "settings.xseg_model"),
    "ct_mode": ("settings.ct_mode", "settings.color_transfer_mode",
                "settings.color_transfer", "ct_mode"),
    "blend": ("settings.blend", "settings.mode", "blend"),
    "erode": ("settings.mask_erode", "settings.erode_mask_modifier", "settings.erode_mask"),
    "blur": ("settings.mask_blur", "settings.blur_mask_modifier", "settings.blur_mask",
             "settings.mask_feather"),
    "models": ("models", "model", "dfm"),
    "faces_project": ("faces_project", "face_library", "library"),
    "library_enabled": ("faces_project", "face_library", "library"),
    "fast_export": ("fused", "fast_export"),
    "frame_range": ("start_frame", "end_frame", "frame_from", "frame_to", "range"),
    "workers": ("workers", "merge_workers", "merge_threads", "threads", "num_workers"),
    "batch": ("batch", "pred_batch"),
    "jpeg_quality": ("jpeg_quality",),
    # 两个性能旋钮：compose_backend 在请求体里叫 compose_gpu（1/0），别写错名字 ——
    # 写错了 [_has] 就永远 False，「界面选的」会被当成「默认」。
    "dfm_dtype": ("dfm_dtype",),
    "compose_backend": ("compose_gpu", "compose_backend"),
}


# 入口认识、但不会进 [eff] 的键（预检 / 显存覆盖 / 排障旋钮 / 兼容别名…）。
# 不在这个集合里的键会被当成「没人消费」打到 [req-unused]。
KNOWN_KEYS = {
    # 通用
    "video", "source_video", "path", "fps", "force", "dry_run", "thresh",
    "align_thresh", "face_threshold", "keep_cache", "quality", "align_quality",
    "output", "output_dir", "out", "out_video", "cache_dir", "input_dir",
    "project", "project_dir", "frames_dir", "aligned_dir", "merged_dir",
    "mask_onnx", "xseg", "xseg_onnx", "dfm", "dfms", "model", "models",
    "settings", "detector", "faces_project", "face_library", "library",
    "fused", "fast_export", "workers", "batch", "num_workers", "threads",
    "merge_workers", "merge_threads", "merge_cv2_threads", "pred_batch",
    "pred_readers", "pred_writers", "mask_readers", "mask_writers", "mask_batch",
    "frame_budget", "fused_batch", "fused_batch_wait_ms", "sess_gpu", "sess_cpu",
    "sess_strategy", "sess_level", "png_compression", "jpeg_quality",
    "start_frame", "end_frame", "frame_from", "frame_to", "range",
    "encoder", "hwaccel", "image_format", "fmt", "panel_size", "face", "idx",
    "frame", "n", "step", "t", "threshold", "raw", "debug", "compose", "dir",
    "image", "items", "skip_faces", "no_swap_faces", "auto", "auto_faces",
    "timings", "frame_file", "w", "encoding", "png_level", "use_library",
    "face_type", "blend", "ct_mode", "mask_source", "mask_mode", "mask_erode",
    "mask_blur", "erode_mask", "blur_mask", "color_transfer", "mode",
    "keep_aligned", "video_name", "source", "fuse", "fused_only",
    # 两个用户可调的性能旋钮（backend_merge/knobs.py 消费，不进 KNOWN 会被误报 [req-unused]）
    "dfm_dtype", "compose_gpu", "compose_backend",
}


_PATH_KEYS = {
    "video", "source_video", "path", "output", "output_dir", "out", "out_video",
    "cache_dir", "input_dir", "project", "project_dir", "frames_dir",
    "aligned_dir", "merged_dir", "mask_onnx", "xseg", "xseg_onnx", "dfm",
    "frame_file", "dir", "face_library", "faces_project", "library",
    "model_path", "python", "output_path",
}


_B64_RUN = re.compile(r"[A-Za-z0-9+/=_-]{160,}")


_B64_URL = re.compile(r"^data:[^,]{0,80};base64,", re.I)


_LOCK = threading.Lock()


_LAST_REQ = {}          # route -> 上一次 [req] 的指纹（预览这种高频入口去重）


_LAST_EFF = {}          # route -> 上一次 [eff] 的指纹


# ---------------------------------------------------------------- 小工具
def _safe_print(text: str) -> None:
    """stdout 不是 utf-8（桌面壳直接起 serve.py 时）也不许把日志吞掉。"""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        try:
            buf = getattr(os.sys.stdout, "buffer", None)
            if buf is not None:
                buf.write((text + "\n").encode("utf-8", "replace"))
                buf.flush()
        except Exception:                                   # noqa: BLE001
            pass
    except Exception:                                       # noqa: BLE001
        pass


def _json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=False, default=str)
    except Exception:                                       # noqa: BLE001
        return json.dumps(str(obj), ensure_ascii=False)


def _settings(body) -> dict:
    raw = (body or {}).get("settings")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except Exception:                                   # noqa: BLE001
            return {}
    return raw if isinstance(raw, dict) else {}


def _has(body, name) -> bool:
    """请求体里有没有**明确给出**这个字段（空串 / 空数组 = 没给）。"""
    body = body or {}
    settings = _settings(body)
    for key in BODY_KEYS.get(name, ()):
        if key.startswith("settings."):
            v = settings.get(key.split(".", 1)[1])
        else:
            v = body.get(key)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, tuple, dict)) and len(v) == 0:
            continue
        return True
    return False


def _basename(value) -> str:
    text = str(value or "")
    if not text:
        return text
    try:
        return os.path.basename(text.rstrip("\\/")) or text
    except Exception:                                       # noqa: BLE001
        return text


def _scrub_str(text: str) -> str:
    if len(text) > 80 and _B64_URL.match(text):
        return "<base64 %d chars>" % len(text)
    if _B64_RUN.search(text or ""):
        return "<base64 %d chars>" % len(text)
    # 绝对路径 / 含分隔符的长串：只留文件名（脱敏的第一条要求）
    if len(text) > 3 and (re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\")
                          or ("/" in text and "://" not in text and " " not in text)):
        return _basename(text)
    return text if len(text) <= 160 else text[:157] + "..."


def sanitize(obj, key: str = "") -> object:
    """请求体脱敏：路径只留文件名、base64 换成长度、超长字符串截断。"""
    if isinstance(obj, dict):
        return {k: sanitize(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, key) for v in obj]
    if isinstance(obj, str):
        low = key.lower()
        if low in _PATH_KEYS or low.endswith("_dir") or low.endswith("_path"):
            return _basename(obj)
        # models[] 里的 dfm 之类：值本身就是路径
        if obj and (re.match(r"^[A-Za-z]:[\\/]", obj) or obj.startswith("\\\\")):
            return _basename(obj)
        return _scrub_str(obj)
    return obj


def unused_keys(body: dict) -> list:
    """请求体里没有任何入口消费的键（"界面发了、后端不认识"）。"""
    body = body if isinstance(body, dict) else {}
    return sorted(str(k) for k in body.keys() if str(k) not in KNOWN_KEYS)


def log_request(route: str, body: dict, dedupe: bool = False,
                ignore: tuple = ()) -> list:
    """打一行 [req]（脱敏）+ 每个没人消费的键一行 [req-unused]。

    返回未使用的键列表（调用方可以顺手塞进 status）。dedupe=True 时只在内容变了才打
    （预览每帧一次，不去重会淹掉别的日志）；ignore 里的键不计入「内容变没变」
    （预览的 t 每帧都变，但配置没变）。
    """
    body = body if isinstance(body, dict) else {}
    clean = sanitize(body)
    blob = _json(clean)
    unused = unused_keys(body)
    fp_body = {k: v for k, v in clean.items() if k not in set(ignore or ())}
    fingerprint = _json(fp_body) + "|" + ",".join(unused)
    with _LOCK:
        changed = _LAST_REQ.get(route) != fingerprint
        if changed or not dedupe:
            _LAST_REQ[route] = fingerprint
    if changed or not dedupe:
        _safe_print("[req] " + blob)
        for k in unused:
            _safe_print("[req-unused] " + str(k))
    return unused


# ---------------------------------------------------------------- 生效配置
def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [str(x) for x in a] == [str(x) for x in b]
    return str(a).strip().lower() == str(b).strip().lower()


def _range_text(rng) -> str:
    if not isinstance(rng, dict) or not rng:
        return "全部"
    if rng.get("full"):
        return "全部"
    start = int(rng.get("start") or 0)
    end = int(rng.get("end") or 0)
    total = int(rng.get("total") or 0)
    if start <= 0 and end <= 0:
        return "全部"
    return "%d..%d（%d 帧）" % (start, end, total)


def _shown(value) -> str:
    if isinstance(value, (list, tuple)):
        return "、".join(str(x) for x in value) if value else "[]"
    if isinstance(value, dict):
        return _json(value)
    return str(value)


def _warn_text(kind: str, field: str, value, src_note: str, requested) -> str:
    label = UI_LABELS.get(field, field)
    shown = _shown(value)
    if kind == "inferred":
        return ("%s 由 %s 推断为 %s，请求体未指定 %s"
                % (field, src_note or "别的字段", shown, field))
    if kind == "missing":
        return ("%s 未在请求体中出现，按 default=%s 执行（界面可能显示 %s）"
                % (field, shown, label))
    return ("%s 由 %s 改为 %s（%s），请求体写的是 %s"
            % (field, _shown(requested), shown, src_note or "后端改写", _shown(requested)))


def build(route: str, kind: str, body: dict, values: dict,
          inferred: dict = None, quiet: set = None, requested: dict = None,
          warnings_out: list = None, src_override: dict = None) -> dict:
    """拼出生效配置 + 一致性自检。

    values     规范字段 -> 真正生效的值（调用方从 cfg 里取，别在这里重算一遍）
    inferred   缺字段但值是推出来的：{field: "quality=high"}（会写进 src_note）
    quiet      推出来也不用警告的字段（契约上就该推的，例如 mask_source ← mask_mode）
    requested  请求体写的原值：{field: 2}，和 values 不一致 = 后端改写过
    src_override  直接指定某些字段的来源（{field: "auto"}）—— 只有「自动」这一档需要：
                  请求体给了 dfm_dtype=auto/compose_gpu=auto 时它既不是 ui（没钉死值）
                  也不是 default（不是没给），来源就是「自动」本身。
    """
    body = body if isinstance(body, dict) else {}
    inferred = dict(inferred or {})
    quiet = set(quiet or ())
    requested = dict(requested or {})
    src_override = dict(src_override or {})
    warned = set()
    warnings = []

    out = {"route": str(route), "kind": str(kind), "at": round(time.time(), 3)}
    notes = {}
    for name in FIELDS:
        value = values.get(name)
        if name == "frame_range":
            value = _range_text(value)
        if name == "models":
            value = [str(x) for x in (value or [])]
            if not value:
                value = []
        src = src_override.get(name) or (
            "ui" if _has(body, name) else ("inferred" if name in inferred else "default"))

        # ---- 三类警告（quiet 里的字段不报警：它在这次入口里本来就不适用）----
        default = UI_DEFAULTS.get(name)
        if name in quiet:
            pass
        elif name == "models" and src == "default" and value and not _has(body, name):
            # 请求体没点名模型、后端自己挑了一个 —— 和「界面显示 TinyMog、跑 retinaface」
            # 是同一个形状：谁都没说，但确实换了东西。
            text = _warn_text("missing", name, value, "", None)
            warnings.append({"code": "eff-missing", "field": name,
                             "level": "warn", "text": text})
            warned.add(name)
        elif src == "inferred" and not _same(value, default):
            text = _warn_text("inferred", name, value, inferred.get(name), None)
            warnings.append({"code": "eff-inferred", "field": name,
                             "level": "warn", "text": text})
            warned.add(name)
        elif src == "default" and default not in (None, "", []) and not _same(value, default):
            text = _warn_text("missing", name, value, "", None)
            warnings.append({"code": "eff-missing", "field": name,
                             "level": "warn", "text": text})
            warned.add(name)
        if name in requested and not _same(value, requested[name]) and name not in warned \
                and name not in quiet:
            text = _warn_text("rewritten", name, value,
                              "显存预算降档" if name in ("workers", "batch") else "后端改写",
                              requested[name])
            warnings.append({"code": "eff-rewritten", "field": name,
                             "level": "warn", "text": text})
            warned.add(name)

        out[name] = value
        out[name + "_src"] = src
        if name in inferred:
            notes[name] = inferred[name]
    if notes:
        out["src_note"] = notes
    out["warnings"] = warnings
    if warnings_out is not None:
        warnings_out.extend(warnings)
    return out


def line_of(eff: dict) -> str:
    """[eff] 那一行的 JSON（不含 warnings；产物日志里也贴同一份）。"""
    return _json({k: v for k, v in (eff or {}).items() if k != "warnings"})


def log_eff(eff: dict, dedupe: bool = False) -> bool:
    """打一行 [eff] + 每条 [eff-warn]。返回这次是否真的打了（去重后可能没打）。"""
    route = str((eff or {}).get("route") or "")
    # 指纹里去掉时间戳和 warnings：去重看的是「配置变没变」，不是「这次几点跑的」
    blob = _json({k: v for k, v in eff.items() if k not in ("warnings", "at")})
    with _LOCK:
        changed = _LAST_EFF.get(route) != blob
        if changed or not dedupe:
            _LAST_EFF[route] = blob
    if not (changed or not dedupe):
        return False
    _safe_print("[eff] " + line_of(eff))
    for w in (eff or {}).get("warnings") or []:
        _safe_print("[eff-warn] " + str(w.get("text") or w))
    return True


def warning_texts(warnings) -> list:
    out = []
    for w in warnings or []:
        text = str(w.get("text") if isinstance(w, dict) else w or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def reset() -> None:
    with _LOCK:
        _LAST_REQ.clear()
        _LAST_EFF.clear()


# 上一次预览真正生效的配置：预览是高频入口（播放/拖动时每帧一次），[eff] 只在配置
# 变化时打一行；/api/merge/preview/status 读这里，界面也能对照。
_PREVIEW_EFF = {}
_PREVIEW_EFF_LOCK = threading.Lock()


# ============================================================
# 生效配置（[eff] / [req] / 一致性自检）
# ------------------------------------------------------------
# 用户的原话：「我们的控制台、日志显示，做得是不是和没有一样」。事实是检测器那个
# bug 里日志一直写着 retinaface_10g，可没人能从流水账里看出「这跟我设的不一样」。
# 这里把真正生效的配置摊成一行 JSON（每个字段带来源 ui/default/inferred），把
# 「请求体 vs 界面默认」对不上账的地方打成 [eff-warn] 并塞进 status.warnings[]。
# ============================================================
def _eff_model_names(cfg) -> list:
    names = []
    for m in (cfg.get("single_models") or cfg.get("models") or []):
        try:
            n = str(m.get("name") or "")
        except Exception:                                   # noqa: BLE001
            n = str(m or "")
        if n and n not in names:
            names.append(n)
    if not names and cfg.get("model"):
        names = [str(cfg.get("model"))]
    return names


def _eff_basename(value) -> str:
    """只留名字：日志和界面都不需要整条用户路径。"""
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.basename(text.rstrip("\\/")) or text


def _eff_values_of(cfg: dict, kind: str) -> dict:
    """cfg -> [eff] 的字段值。**只从 cfg 取**，不重新算一遍（否则就不是「生效」了）。"""
    cfg = cfg or {}
    conf = cfg.get("config") or {}
    settings = cfg.get("settings") or {}
    mask = cfg.get("mask_src")
    if not isinstance(mask, dict):
        mask = _mask_src(settings, settings.get("mask_mode"))
    if kind == "project":
        workers, batch, fast = cfg.get("workers"), cfg.get("batch"), False
    elif kind == "fused":
        workers = cfg.get("merge_workers")
        batch = cfg.get("fused_batch") or cfg.get("pred_batch")
        fast = True
    else:
        workers = cfg.get("merge_workers")
        batch = cfg.get("pred_batch")
        fast = bool(cfg.get("chain"))
    q = cfg.get("jpeg_quality")
    if q is None:
        # 别再退回 cfg['quality'] —— 那是**对齐质量档位**（"high"/"fast"），
        # 和 JPEG 压缩质量是两件事。以前这里 int("high") 抛异常才落到 100，
        # 而预览那条路 cfg['quality'] 有时是个数字（92），于是日志里冒出
        # 「jpeg_quality ... 按 default=92 执行（界面可能显示 100）」—— 界面和实际对不上。
        # 帧图编码的口径是**固定 100 + 4:4:4**（见 _stage_imencode / _vid_log），
        # 这里就按那个口径兜底。
        q = _STAGE_JPEG_QUALITY
    kn = _knobs_of(cfg)
    return {
        "detector": str(cfg.get("det_align") or cfg.get("detector") or _DEFAULT_DETECTOR),
        "mask_source": str(mask.get("label") or ""),
        "ct_mode": str(conf.get("color_transfer_mode") or settings.get("ct_mode") or DEFAULT_CT_MODE),
        "blend": str(conf.get("mode") or settings.get("blend") or DEFAULT_BLEND),
        "erode": int(conf.get("erode_mask_modifier") or 0),
        "blur": int(conf.get("blur_mask_modifier") or 0),
        "models": _eff_model_names(cfg),
        # 人脸库只留名字：日志/界面都不需要整条用户路径
        "faces_project": _eff_basename(cfg.get("library") or ""),
        "library_enabled": bool(cfg.get("use_library", True)),
        "fast_export": bool(fast),
        "frame_range": dict(cfg.get("range") or {}),
        "workers": int(workers or 0),
        "batch": int(batch or 0),
        "jpeg_quality": int(q or 0),
        # 两个性能旋钮：值来自 cfg["knobs"]（请求体 → env 的那一份），不是这里重算的
        "dfm_dtype": str(kn.get("dfm_dtype") or "auto"),
        "compose_backend": str(kn.get("compose_backend") or "cpu"),
    }


def _knobs_of(cfg) -> dict:
    """cfg -> 这一单的两个性能旋钮（没有就走「谁都没说」那一档，按进程 env 报）。

    预览 / 项目导出这些入口不经过 _fused_cfg / _stage_cfg，也就没有 cfg["knobs"]；
    这时候按 env 现算一份 —— [eff] 的这两列永远有值，不会出现「这一栏是空的」。
    """
    raw = (cfg or {}).get("knobs")
    if isinstance(raw, dict) and raw.get("dfm_dtype"):
        return raw
    try:
        from . import knobs as _knobs
        return _knobs.resolve({})
    except Exception:                                       # noqa: BLE001
        return {}


def _eff_inferred_of(cfg: dict, body: dict) -> dict:
    """哪些字段是后端从别的字段推出来的（写进 src_note，必要时打 [eff-warn]）。"""
    inf = {}
    body = body if isinstance(body, dict) else {}
    if not str(body.get("detector") or "").strip():
        q = str(body.get("quality") or body.get("align_quality") or "").strip().lower()
        if q:
            inf["detector"] = "quality=%s" % q
    mask = cfg.get("mask_src") if isinstance(cfg.get("mask_src"), dict) else {}
    if str(mask.get("model_from") or "") == "mask_mode":
        inf["mask_source"] = "mask_mode=%s" % (cfg.get("mask_mode"),)
    return inf


def _eff_requested_of(body: dict, kind: str) -> dict:
    """请求体里写的原值（只取和 [eff] 同一个旋钮的那几个键，用来发现「被后端改写」）。"""
    body = body if isinstance(body, dict) else {}
    out = {}

    def _num(*keys):
        for k in keys:
            v = body.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            try:
                return int(float(v))
            except (TypeError, ValueError):
                continue
        return None

    w = _num("workers") if kind == "project" else _num("merge_workers", "threads", "merge_threads")
    if w is not None:
        out["workers"] = w
    if kind == "fused":
        # eff 的 batch 就是 fused_batch；body 的 pred_batch/batch 是另一个旋钮，别拿来对账
        b = _num("fused_batch")
    elif kind == "project":
        b = _num("batch")
    else:
        b = _num("batch", "pred_batch")
    if b is not None:
        out["batch"] = b
    q = _num("jpeg_quality")
    if q is not None:
        out["jpeg_quality"] = q
    return out


def _eff_publish(route: str, kind: str, body: dict, cfg: dict,
                 dedupe: bool = False) -> dict:
    """打 [eff] + [eff-warn]，返回生效配置（调用方把它塞进 status）。

    请求体那一行 [req] 在路由入口打（每次 HTTP 请求一行，dry_run 预检也算），
    这里只负责「真正生效的那份配置」。
    """
    body = body if isinstance(body, dict) else {}
    quiet = {"mask_source"}          # mask_source ← mask_mode 是契约，不算对不上
    if str(kind).startswith("stage:"):
        # 单个阶段不做「快速导出」这个选择（fused 是 /api/merge/export 的旋钮）
        quiet.add("fast_export")
        # 六阶段里只有 faces 真正用检测器（其余阶段拿的是上游产物）
        if kind != "stage:faces":
            quiet.add("detector")
    if kind == "project":
        # DFL 项目导出：没有「快速导出」开关；jpeg 质量是它自己的帧编码旋钮（界面没这控件）
        quiet.update({"fast_export", "jpeg_quality"})
    if kind == "project-modes":
        # 4.0 MergeStudio 老项目模式：不选快速导出、不读人脸库、帧图质量是子进程的事
        quiet.update({"fast_export", "jpeg_quality", "library_enabled"})
    if kind == "preview":
        # 预览不是导出：没有「快速导出」这个选择
        quiet.add("fast_export")
    kn = _knobs_of(cfg)
    # 兜底：只有快速导出（fused）才有 GPU 合成。别让 [eff] 在分段链 / 项目导出 / 预览上
    # 说 gpu（那几条路的合成只有 compose_frame 的 CPU 实现）—— 改成 cpu 并带一句原因，
    # 文案和流水线里那一条同一个口径（选了 X、实际跑 Y、原因 Z）。
    if str(kind) != "fused" and str(kn.get("compose_backend") or "") == "gpu":
        kn = _knobs.refine_compose(kn, "", fused=False)
        if isinstance(cfg, dict):
            cfg["knobs"] = kn
    eff = _eff.build(route, kind, body, _eff_values_of(cfg, kind),
                     inferred=_eff_inferred_of(cfg, body),
                     quiet=quiet,
                     requested=_eff_requested_of(body, kind),
                     # 两个性能旋钮的来源是 knobs 算出来的（ui / auto / default），
                     # 不能靠「请求体里有没有这个键」推 —— 「自动」也是请求体给了的。
                     src_override={"dfm_dtype": kn.get("dfm_dtype_src"),
                                   "compose_backend": kn.get("compose_backend_src")})
    eff["req_unused"] = _eff.unused_keys(body)
    # 被 refine_compose 判回 CPU 的那一档（典型：融合方式 seamless 不进 GPU，
    # 或这一单根本不是快速导出）要把**原因**一起回显：前端抽屉靠 eff.compose_why
    # 挂「已降级」小标、把来源 chip 从「界面」换成「降级」（app.js mgpEffPaint）。
    # 只留在 knobs 里的话，用户点开抽屉看到的还是「cpu · 界面」—— 那正是这一轮要消灭的观感：
    # 值对、来源对，但看起来像「我选的就是 cpu」，只有顶部横幅能解释（而横幅可以关掉）。
    if str(kn.get("compose_why") or ""):
        eff["compose_why"] = str(kn["compose_why"])
    # dfm 的推理后端 / 精度：**不是 cfg 字段**（它由进程级环境变量定），所以不走 build()
    # 那套「请求体缺字段」的自检，直接挂到 [eff] 那一行的末尾。用户要能一眼看到
    # 「我这单的 dfm 到底跑在 onnx 还是 torch、fp32 还是 bf16」——这一栏就是那个答案。
    eff["dfm"] = _dfm_eff_text(cfg)
    # 人脸超分（GPEN）也是**进程级 env**，不是 cfg 字段 —— 同样直接挂到 [eff] 上。
    # 开着就必须留痕：它按帧吃 GPU，不看这一栏会以为"只是稍微慢一点"。
    try:
        from . import facesr as _sr
        _lv = int(kn.get("sr") or 0)
        if _lv > 0 or _sr.enabled():
            _info = _sr.info()
            eff["sr"] = {
                "level": _lv if _lv > 0 else _info.get("level", 0),
                "size": 256 if (_lv if _lv > 0 else 0) <= 50 else 512,
                "strength": round((_lv if _lv > 0 else 0) / 50.0, 2) if 0 < (_lv if _lv > 0 else 0) <= 50
                            else round(((_lv - 50) / 50.0), 2) if (_lv if _lv > 0 else 0) > 50 else 0.0,
                "backend": _info.get("backend") or "",
                "model": _info.get("model") or "",
                "why": _info.get("why") or "",
            }
            if _sr.enabled() and _sr.session()[0] is None:
                eff.setdefault("warnings", []).append({
                    "code": "eff-sr-missing", "field": "sr", "level": "warn",
                    "text": "人脸超分选了 %s，但后端没起来（%s）——这一单按原样输出"
                            % (_sr.mode(), (_sr.info().get("why") or "未知原因")),
                })
    except Exception as _e:                                  # noqa: BLE001
        eff["sr"] = {"error": "%s: %s" % (_e.__class__.__name__, str(_e)[:120])}
    # knobs 自带的告警（env 覆盖了请求体 / 选了 gpu 但这单上不了 GPU）也要进横幅。
    # 这些在「值已经确定」的时刻就算得出来，不用等流水线开跑。
    for _code, _field, _text in (kn.get("warnings") or []):
        if not _text:
            continue
        _w = {"code": str(_code or "eff-knob"), "field": str(_field or ""),
              "level": "warn", "text": str(_text)}
        eff.setdefault("warnings", []).append(_w)
        if cfg is not None and isinstance(cfg, dict):
            cfg.setdefault("_eff_warnings", []).append(_w)
    _eff.log_eff(eff, dedupe=dedupe)
    return eff


def _dfm_eff_text(cfg=None) -> str:
    """'torch · bf16' / 'onnx · fp32' —— [eff] 与状态栏共用一份口径。

    精度优先取**这一单**的旋钮（cfg["knobs"]）：env 是在 cfg 建好之后才写的，
    这里若只读 env，[eff] 会回显上一单的精度（「界面上选 fp32、日志里 bf16w」这种
    对不上账，正是这次要治的东西）。没有 knobs 的入口（预览/项目）才读 env。
    """
    knob = str((_knobs_of(cfg).get("dfm_dtype") or "")).strip().lower()
    if knob == "auto":
        knob = ""                                       # 「自动」= 交给后端挑，别冒充具体档
    try:
        from .vram import _dfm_backend_sel
        kind, dt, _why = _dfm_backend_sel()
    except Exception:                                   # noqa: BLE001
        return ("torch · %s" % knob) if knob else "?"
    if kind == "torch":
        return "torch · %s" % (knob or dt or "tf32")
    return "onnx · fp32"


def _eff_status_patch(eff: dict) -> dict:
    """生效配置 -> 塞进 status 的几个键（warnings[] 里也放一份纯文本）。"""
    warns = list((eff or {}).get("warnings") or [])
    return {"eff": eff, "eff_warnings": warns,
            "warnings": [str(w.get("text") or w) for w in warns]}


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 eff 与 _eff、stages
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .. import _eff as _eff

from . import knobs as _knobs

from .pipeline_stages import (
    _STAGE_JPEG_QUALITY, _mask_src,
)
