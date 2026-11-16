# -*- coding: utf-8 -*-
"""检测器名字的**唯一**映射点：4.1 的规范名 ↔ 4.0 MergeStudio 引擎认的名字。

为什么必须单独一份
----------------
项目模式导出（<帧目录> → 4.0 MergeStudio 子进程一口气出 mp4）里，检测器是
**4.0 那边**的 MergeStudio/core/detector/factory.py 按名字建实例的
（DetectorFactory.DETECTORS 的键：TinyMog / RetinaFace_10g / YOLOv8 …），
而 4.1 的规范名是小写 tinymog / retinaface_10g。

两边名字不同，一旦散着写就会各自漂。2026-11 那次事故就是这么来的：
_merge_worker 组命令行时**根本没带检测器**，于是 4.0 用引擎默认 YOLOv8
（最重那个），而「查看生效配置」里硬写 tinymog —— 界面/日志说一套、实跑另一套，
用户体感是「选了 tinymog 还是慢」。

所以规矩只有两条：
  1. **转名只在这张表里做**（engine() / canon()），别处不许再写第二份；
  2. **未知名字当场报错**（ValueError，带可选值清单）—— 绝不静默回默认。
     这条比「能跑起来」重要：静默回落正是上面那次事故的形状。
"""
from __future__ import annotations

# 4.1 没有给 detector 时的默认档（和 backend_merge._shared._DEFAULT_DETECTOR 同口径）
DEFAULT = "tinymog"

# 4.1 规范名 → 4.0 引擎名（= DetectorFactory.DETECTORS 的键）
ENGINE = {
    "tinymog": "TinyMog",
    "retinaface_10g": "RetinaFace_10g",
    # 4.0 引擎的历史默认。允许**显式**点名（老脚本/对拍要用），
    # 但它永远不会成为 4.1 的默认值 —— 默认是谁，看上面 DEFAULT。
    "yolov8": "YOLOv8",
}

# 别名 → 4.1 规范名（两边写法的并集）
_ALIAS = {
    "tinymog": "tinymog",
    "retinaface_10g": "retinaface_10g", "retinaface10g": "retinaface_10g",
    "retinaface": "retinaface_10g", "det_10g": "retinaface_10g",
    "yolov8": "yolov8", "yolov8face": "yolov8",
}

# 4.1 规范名 → 给人看的标签（日志/回显统一用这个，别各写各的）
LABEL = {"tinymog": "TinyMog", "retinaface_10g": "RetinaFace_10g", "yolov8": "YOLOv8"}


def _key(raw) -> str:
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "")


def canon(raw, default: str = "") -> str:
    """任意写法 → 4.1 规范名。空值给 default（default 也为空则报错）。"""
    k = _key(raw)
    if not k:
        if default:
            return default
        raise ValueError("缺少检测器名：可选 %s" % " / ".join(sorted(set(_ALIAS.values()))))
    hit = _ALIAS.get(k)
    if hit is None:
        raise ValueError("未知检测器 %r：只认 %s（4.1 规范名）—— 不会替你挑一个默认"
                         % (str(raw), " / ".join(sorted(set(_ALIAS.values())))))
    return hit


def engine(raw, default: str = "") -> str:
    """任意写法 → 4.0 MergeStudio 引擎名（DetectorFactory 里那个键）。"""
    return ENGINE[canon(raw, default)]


def label(raw, default: str = "") -> str:
    """任意写法 → 给人看的标签（= 4.0 引擎名，两者本来就一致）。"""
    return LABEL[canon(raw, default)]


def table() -> list:
    """对照表（验收/文档用）：[(4.1 规范名, 4.0 引擎名, [别名...])]。"""
    return [(k, ENGINE[k], sorted(a for a, v in _ALIAS.items() if v == k))
            for k in sorted(ENGINE)]
