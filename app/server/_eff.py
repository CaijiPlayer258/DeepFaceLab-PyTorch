# -*- coding: utf-8 -*-
"""生效配置（effective config）：一行 [eff] + 请求体脱敏回显 + 一致性自检。

为什么有它
----------
「检测器」那个 bug 里，界面显示 TinyMog、后端跑 retinaface_10g 跑了几十次。日志里
其实每一单都写着「检测器 retinaface_10g thresh=0.50」—— 信息在，但没人能从一串
流水账里看出「这跟我设的不一样」。所以这里把**真正生效的那份配置**摊成一行 JSON：

    [eff] {"route":"/api/merge/export","kind":"fused","detector":"tinymog",
           "detector_src":"ui","mask_source":"XSegLite","mask_src":"inferred", ...}

每个字段都带来源：
    ui        请求体里明确给了（界面发的）
    default   请求体没有，后端按默认值执行
    inferred  后端从别的字段推出来的（例如 quality=high → retinaface_10g）

三类警告（都进 status.warnings[]，同时打一行 [eff-warn]）：
    1) missing    请求体没这个字段，后端默认值又和界面默认值不一致
                  → "detector 未在请求体中出现，按 default=retinaface_10g 执行（界面可能显示 TinyMog）"
    2) inferred   值是从别的字段推的，而且推出来和界面默认值不一致
                  → "detector 由 quality=high 推断为 retinaface_10g，请求体未指定 detector"
    3) rewritten  请求体给了值，后端改写了（夹范围 / 按显存预算降档）
                  → "workers 由 2 改为 1（显存预算降档），请求体写的是 2"

请求体回显（[req]）只留文件名、去掉 base64 —— 一条请求里可能有整段视频路径和
data URL，原样打进日志既没法看也没法贴出来。没被任何入口消费的键单独打
[req-unused] <k>（"界面发了、后端不认识" 是同一类对不上账的 bug）。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

# ---------------------------------------------------------------- 字段表
# 顺序就是 [eff] 行里的顺序：先「选了什么模型/检测器/遮罩」，再「怎么合成」，
# 最后是「这次跑多大」。
FIELDS = (
    "detector", "mask_source", "ct_mode", "blend", "erode", "blur",
    "models", "faces_project", "library_enabled", "fast_export",
    "frame_range", "workers", "batch", "jpeg_quality",
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
          warnings_out: list = None) -> dict:
    """拼出生效配置 + 一致性自检。

    values     规范字段 -> 真正生效的值（调用方从 cfg 里取，别在这里重算一遍）
    inferred   缺字段但值是推出来的：{field: "quality=high"}（会写进 src_note）
    quiet      推出来也不用警告的字段（契约上就该推的，例如 mask_source ← mask_mode）
    requested  请求体写的原值：{field: 2}，和 values 不一致 = 后端改写过
    """
    body = body if isinstance(body, dict) else {}
    inferred = dict(inferred or {})
    quiet = set(quiet or ())
    requested = dict(requested or {})
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
        src = "ui" if _has(body, name) else ("inferred" if name in inferred else "default")

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
