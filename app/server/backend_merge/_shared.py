# -*- coding: utf-8 -*-
"""合成域共享件：常量表、请求参数校验（mask/settings/检测器）、路径与排版小工具。backend_merge 的最底层。

从 app/server/routes_merge.py 按关注点原样切出来的（16741 行的单文件 → 按域分家），
函数体 / 注释 / 日志文案逐字节照抄，只补 import；同包之间的 import 放在文件末尾
（见文末「环上的依赖」），另有一处 __file__ 相对路径下标跟着目录层级改了。
旧行号 → 现文件 的对照表：app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from .._common.http import (
    _BadInput, _float_val, _int_val,
)
from .._common.proc import (
    _proc_mem_mb,
)


try:                                    # app.py 里那张表是唯一真源
    from ..app import _MASK_MODE_LABEL as _MASK_LABELS
except Exception:                       # pragma: no cover - 表被改名时的退路
    _MASK_LABELS = {"full": 0, "dst": 1, "prd": 2, "prd×dst": 4,
                    "xseg-prd": 6, "xseg-dst": 7, "xseg-prd×dst": 8}


# ---- 路径 ----
def _norm_path(raw):
    """把界面传进来的路径收敛成一个可用的 Path（做存在性判断前必须过这一步）。

    * 空串 / None / 只有空白 -> None
    * 去掉首尾空白和包裹的引号（从资源管理器拖进来常带）
    * 反斜杠、正斜杠都当分隔符（Windows 下 Path 两种都认）
    * 相对路径按仓库根解析（和 app.py 顶部"不带盘符的路径相对 4.1 根"的约定一致）
    * normpath 收掉 . / .. 和结尾多余的分隔符
    """
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'").strip()
    if not s:
        return None
    s = os.path.expandvars(os.path.expanduser(s))
    p = Path(s)
    if not p.anchor:                    # 没盘符也没根 = 相对路径
        p = _REPO_ROOT / p
    return Path(os.path.normpath(str(p)))


def _same_path(a, b) -> bool:
    """Windows 上同一个路径大小写可以不一样，比较前统一 normcase。"""
    if a is None or b is None:
        return False
    try:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))
    except Exception:
        return False
# MergeStudio.core.config.mode_dict —— 但**只收融合方式**。
# original / raw-rgb / raw-predict / debug 是「视图」（看原图 / 看原始预测 / 看调试
# 信息），不是融合方式：以前混在这张表里，于是下拉里既有 overlay 又有 debug，
# 传 mode=debug 后端还默默按 overlay 出片（_merge_runner 的 _MODE_EFFECTIVE 根本
# 不认它）—— 用户选什么都不会变，等于假选项。
# 现在：mode/blend 只认这 4 个，其余一律 400；视图走独立参数
#   raw=1（原图，不合成） / debug=1（合成 + 遮罩/边界调试叠加）。
_MERGE_MODES = ("overlay", "hist-match", "seamless", "seamless-hist-match")
# 老前端/老脚本可能还在往 mode 里塞这些「视图名」：明确 400 并说清楚该用哪个参数，
# 不许静默折成 overlay（那正是「选了没反应」的老 bug 的温床）。
_MERGE_VIEW_MODES = ("original", "raw-rgb", "raw-predict", "debug")
# MergeStudio.core.config.ctm_dict（+ "sot" 这个直白写法：worker 的 ct_map 把它
# 折成 ctm_dict 里那一档，见 preview_worker.py 的 ct_map["sot"]）
_COLOR_TRANSFER = ("none", "rct", "lct", "mkl", "mkl-m", "idt", "idt-m",
                   "sot", "sot-m", "mix-m", "mt", "lut")
# 接口契约里的两条主参数（★ 默认值就是「不传 settings」时的结果，改这里 = 改默认）
_CT_MODES = ("mt", "rct", "lct", "mkl", "sot")
# 以前这里还有一份只写两个值的 _BLEND_MODES（seamless / overlay），谁都没用 ——
# 留着只会让人以为 hist-match / seamless-hist-match 不合法。真白名单就一个：_MERGE_MODES。
DEFAULT_CT_MODE = "mt"
DEFAULT_BLEND = "seamless"
_IMG_FORMATS = {"jpg": "jpg", "jpeg": "jpg", "png": "png"}

# ---- 检测器白名单 ----
# 4.1 只有这两个：tinymog（默认，快）/ retinaface_10g（重，召回好）。
# retinaface_500m 这个模型不存在，已经从所有列表里删掉；再点到它要明确 400，
# 不许"就近退一个能用的顶上"（静默回退曾让用户以为在跑 500m）。
_DETECTORS = ("tinymog", "retinaface_10g")
_DET_ALIAS_API = {
    "tinymog": "tinymog",
    "retinaface_10g": "retinaface_10g", "retinaface10g": "retinaface_10g",
    "det_10g": "retinaface_10g", "retinaface": "retinaface_10g",
}
_REMOVED_DETECTORS = ("retinaface_500m", "retinaface500m", "det_500m",
                      "retinaface_500m.onnx", "det_500m.onnx")
_DEFAULT_DETECTOR = "tinymog"      # ★ 默认检测器（所有 entrypoint 共用这一个常量）


def _mask_mode_val(raw, default=6) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if isinstance(raw, bool):
        raise _BadInput("「mask_mode」要是 0~9 的整数，收到 %r" % (raw,))
    text = str(raw).strip()
    if text.isdigit():
        return _int_val("mask_mode", text, 0, 9, default)
    key = text.lower().replace("*", "×")
    if key in _MASK_LABELS:
        return int(_MASK_LABELS[key])
    raise _BadInput("「mask_mode」只认 0~9 或 %s，收到 %r"
                    % (" / ".join(sorted(_MASK_LABELS)), raw))


def _settings_val(raw) -> dict:
    """校验并规范化合成参数。

    以前 settings 原样丢给 _merge_settings_to_config()，非法值要等子进程跑到一半
    才炸（erode_mask 传 "abc" 会在 _merge_runner 里 int() 崩）。这里先翻译成
    MergeStudio 的键名并逐个校验，_merge_settings_to_config 直接取到就用 ——
    参数才真正生效。
    """
    if raw is None:
        raw = {}
    if isinstance(raw, str):                       # 有些前端会把 settings 二次 encode
        try:
            raw = json.loads(raw or "{}")
        except Exception:
            raise _BadInput("settings 不是合法 JSON：%s" % str(raw)[:80])
    if not isinstance(raw, dict):
        raise _BadInput("settings 要是一个 JSON 对象，收到 %s" % type(raw).__name__)

    out = dict(raw)                                # 保留用户原值，方便回显

    def _first(*names):
        """按顺序取第一个非空值（两代键名：新的在前）。"""
        for n in names:
            v = raw.get(n)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            return v
        return None

    # ---- blend（= MergeStudio 的 mode）----
    # 契约字段叫 blend，老写法叫 mode；认一个就够，两个都给时以 blend 为准。
    # 缺省 = seamless。**只认 4 个融合方式**（白名单见 _MERGE_MODES）。
    mode = str(_first("blend", "mode") or DEFAULT_BLEND).strip().lower() or DEFAULT_BLEND
    if mode in _MERGE_VIEW_MODES:
        raise _BadInput(
            "settings.blend 只认 4 个融合方式：%s；收到 %r —— 这是「视图」，不是融合方式，"
            "视图请用 raw=1（原图）/ debug=1（调试）"
            % (" / ".join(_MERGE_MODES), mode))
    if mode not in _MERGE_MODES:
        raise _BadInput("settings.blend 只认 %s，收到 %r"
                        % (" / ".join(_MERGE_MODES), raw.get("blend", raw.get("mode"))))
    out["blend"] = mode
    out["mode"] = mode                             # 老键名照样回显，前端不用改
    out["mask_mode"] = _mask_mode_val(raw.get("mask_mode"), 6)

    # ---- ct_mode（= MergeStudio 的 color_transfer_mode）----
    ct = str(_first("ct_mode", "color_transfer_mode", "color_transfer")
             or DEFAULT_CT_MODE).strip().lower() or DEFAULT_CT_MODE
    if ct not in _COLOR_TRANSFER:
        raise _BadInput("settings.ct_mode 只认 %s（也认 %s），收到 %r"
                        % (" / ".join(_CT_MODES), " / ".join(_COLOR_TRANSFER),
                           raw.get("ct_mode", raw.get("color_transfer"))))
    out["ct_mode"] = ct
    out["color_transfer_mode"] = ct
    out["color_transfer"] = ct

    # ---- 遮罩腐蚀 / 羽化（mask_erode / mask_blur 是契约字段）----
    erode = _first("mask_erode", "erode_mask_modifier", "erode_mask")
    blur = _first("mask_blur", "blur_mask_modifier", "blur_mask")
    out["mask_erode"] = _int_val("settings.mask_erode", erode, -100, 100, 0)
    out["mask_blur"] = _int_val("settings.mask_blur", blur, -100, 100, 0)
    out["erode_mask_modifier"] = out["mask_erode"]
    out["blur_mask_modifier"] = out["mask_blur"]

    fmt = str(raw.get("image_format") or "jpg").strip().lower()
    if fmt not in _IMG_FORMATS:
        raise _BadInput("settings.image_format 只认 jpg / png，收到 %r"
                        % (raw.get("image_format"),))
    out["image_format"] = _IMG_FORMATS[fmt]

    if raw.get("face_margin") is not None:
        out["face_margin"] = _float_val("settings.face_margin", raw.get("face_margin"),
                                        -4.0, 4.0, None)
    return out


# ---------------- 进程内存（工作集 / 提交）：导出前后差，用户看的就是这两个数 ----
def _fmt_mb(v):
    return "?" if v is None else "%.1f" % float(v)


def _trim_ws(why: str = "") -> dict:
    """把本进程的**工作集**还给系统（EmptyWorkingSet），返回前后读数。

    解决的是「谁都没在跑，任务管理器还显示 1.x GB」：Python/NT 堆把刚释放的页留在
    进程里（已提交 ≠ 驻留），系统压力不大就不主动回收。**它只降工作集，不降提交**
    —— 提交要等对象真的被释放才降，所以日志里两个数都打，不许拿工作集下降冒充
    「泄漏修好了」。

    代价：下次访问这些页要重新缺页（预览第一帧慢几十毫秒量级）。默认开，
    DFL_FUSED_TRIM_WS=0 可关。
    """
    m0 = _proc_mem_mb()
    if str(os.environ.get("DFL_FUSED_TRIM_WS") or "1").strip().lower() in (
            "0", "false", "no", "off"):
        return {"did": False, "before": m0, "after": m0, "why": "DFL_FUSED_TRIM_WS=0"}
    try:
        import ctypes
        if not sys.platform.startswith("win"):
            return {"did": False, "before": m0, "after": m0, "why": "非 Windows"}
        k32 = ctypes.WinDLL("kernel32.dll")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_size_t]
        ok = k32.SetProcessWorkingSetSize(ctypes.c_void_p(k32.GetCurrentProcess()),
                                          ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        return {"did": bool(ok), "before": m0, "after": _proc_mem_mb(), "why": why}
    except Exception as e:                                  # noqa: BLE001
        return {"did": False, "before": m0, "after": m0, "why": "%r" % (e,)}
_PRED_BATCH_CAP = 32          # 上限（再大对单卡没意义，batch 16 起反而开始抖）


# ============================================================
# 环上的依赖：同包 import 放在文件末尾
# ------------------------------------------------------------
# 拆分前这些都是同一个模块里的全局名，可以随手互相引用；拆开后 shared 与 _REPO_ROOT
# 互相要用对方的东西（合成 / 流水线 / 显存账本本来就咬在一起）。放在文件末尾
# import 是唯一静态成立的做法：任何模块被 import 时自己的 def 和赋值都已跑完，
# 对方再来取名字一定拿得到，先 import 谁都一样。
# ============================================================
from .._paths import ROOT as _REPO_ROOT
