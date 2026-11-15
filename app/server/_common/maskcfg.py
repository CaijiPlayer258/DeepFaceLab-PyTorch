# -*- coding: utf-8 -*-
"""遮罩模式表：界面的「遮罩来源 × 遮罩模式」→ MergeStudio 的 mask_mode 整数。

原来挨着 app.py 的 _merge_job，但用主有两处（合成参数翻译、routes_mask 的标签表），
所以放公共层：谁也别再从 app 里 import 这张表。
"""
from __future__ import annotations


# 界面上的「遮罩来源 × 遮罩模式」→ MergeStudio 的 mask_mode 整数
#   0 full · 1 dst · 2 learned-prd · 3 learned-dst · 4 learned-prd×learned-dst
#   6 XSeg-prd · 7 XSeg-dst · 8 XSeg-prd×XSeg-dst
_MASK_MODE_LABEL = {
    "prd×dst": 4, "prd": 2, "dst": 1, "full": 0,
    "xseg-prd×dst": 8, "xseg-prd": 6, "xseg-dst": 7,
}


def _mask_mode_int(v, default=6):
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        pass
    key = str(v or "").strip().lower().replace("*", "×").replace("x", "x")
    return _MASK_MODE_LABEL.get(key, default)
