# -*- coding: utf-8 -*-
"""自检：**不读版本号**，直接探测「那些修复是不是真的在这份代码里」。

为什么不用版本号：用户可能只拷了几个文件、或者改了源码没重启 —— `version.txt`
写着 b45，跑着 b23 的代码。版本号会说谎，`inspect.getsource` 不会。

每一项是一个「只有修好了才会出现」的特征串。任何一项 False，就说明这一份安装是
旧的/被覆盖过，用户报的症状按新代码去查是查不出来的。
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List


# (键, 模块, 特征串, 这一项代表什么)
_PROBES = [
    ("pass0_frame_step", "backend_merge.pass0", "_cur = i0 - 1",
     "扫帧每轮走 2 帧（两个 P0 的根）"),
    ("fused_face_type", "backend_merge.pipeline_fused", "_ft_of(comp_cfg)",
     "快速导出取脸按 face_type（b23 忘 import 导致换脸恒 0）"),
    ("merge_final_dstdm", "backend_merge.merge_final", "_dstdm",
     "导出侧带上 DFM 的 dst 遮罩"),
    ("merge_final_逐帧", "backend_merge.merge_final", "for frame in range(_start, _last + 1)",
     "逐帧走完（不再只写有脸的帧）"),
    ("passes_part_rename", "backend_merge.pipeline_passes", "os.replace(str(part), str(out))",
     "分趟导出的成片会落到承诺路径"),
    ("passes_audio_trim", "backend_merge.pipeline_passes", "\"-t\", \"%.6f\"",
     "区间导出音频跟着截断"),
    ("preview_mask_source", "backend_merge.preview", "_src in (\"xseg\"",
     "预览尊重「遮罩来源」选择"),
    ("groups_per_group_budget", "backend_faces.groups", "_had = read_vector",
     "组均值预算按「这次真算了」计"),
    ("groups_orphan_skip", "backend_faces.groups", "orphan_skipped",
     "orphan 组不参与匹配"),
    ("library_no_fallback", "backend_merge.pipeline_stages", "return [\"\"] * len(metas)",
     "库不可用不再兜给左栏模型"),
    ("detect_unify_sess", "\u005fvision.detect", "DFL_DET_UNIFY",
     "单张/批量检测用同一份权重"),
    ("extract_dir_input", "backend_extract.align_stream", "_dir_input = True",
     "目录输入不再产出 0 张脸"),
    ("extract_truncation", "backend_extract.align_stream", "_truncated",
     "断流不再当正常结束"),
    ("train_nan_detector", "@app/engine/models/ModelBase.py", "not all(_math.isfinite",
     "crash detector 认 NaN"),
    ("save_atomic", "@app/engine/core/leras/layers/Saveable.py", "os.replace(str(p_tmp), str(p))",
     "存档落盘是一步原子（断电不蒸发）"),
]


def _src_of(target: str):
    """@ 开头 = 相对仓库根的文件路径（engine/ 不在 server 包里）；否则是模块名。"""
    if target.startswith("@"):
        try:
            from ._paths import ROOT
            return (ROOT / target[1:]).read_text(encoding="utf-8", errors="replace")
        except Exception:                                      # noqa: BLE001
            return None
    import importlib
    # __package__ 是 "server"，所以拼成 server.backend_merge.pass0（用 .. 会跑到顶层去）
    for _full in ((__package__ + "." + target) if __package__ else target, target):
        try:
            m = importlib.import_module(_full)
            break
        except Exception:                                      # noqa: BLE001
            m = None
    if m is None:
        return None
    try:
        return inspect.getsource(m)
    except Exception:                                          # noqa: BLE001
        return None


def selfcheck() -> Dict[str, Any]:
    """返回 {ok, version, missing: [...], items: [...]}。

    ok=False 时 missing 里就是「这一份安装缺哪几项修复」—— 直接告诉用户该更新。
    """
    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for key, modname, needle, what in _PROBES:
        src = _src_of(modname)
        if src is None:
            items.append({"key": key, "ok": None, "what": what,
                          "why": "读不到 %s 的源码（打包/编译过？）" % modname})
            continue
        hit = needle in src
        items.append({"key": key, "ok": bool(hit), "what": what})
        if not hit:
            missing.append("%s：%s" % (what, key))
    ok = not missing
    return {"ok": bool(ok), "missing": missing, "items": items}
