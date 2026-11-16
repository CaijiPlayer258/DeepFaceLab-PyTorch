# -*- coding: utf-8 -*-
"""续跑指纹：同一项目换了源素材要能发现（原 frames.py:28-101）。

完成判据是按**帧号**做的（done_idx，见 align_stream）：换一段素材时帧号从 0 重排，
旧帧就会被当成「已完成」整段跳过 —— 静默少抽，且一个字节的日志都不留。
判定逻辑不动（改它风险大：它同时管着真正的「断点续跑」），只把事实说出来。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# ---------------- 续跑指纹：同一项目换了源素材 ----------------
# 完成判据是按**帧号**做的（done_idx，见 _align_stream）：换一段素材时帧号从 0 重排，
# 旧帧就会被当成「已完成」整段跳过 —— 静默少抽，且一个字节的日志都不留。
# 判定逻辑不动（改它风险大：它同时管着真正的「断点续跑」），只把事实说出来。
SRC_FP_NAME = "_extract_src.json"


def _src_fingerprint(src, w, h, dur):
    """源素材指纹：文件名 / 分辨率 / 时长（+ size、mtime 留作排查线索）。

    只取这三项做判定是有意的：它们与「换了一段素材」强相关，又不会被重编码、
    重新下载造成的字节差异误报。size/mtime 只记录、不参与比较。
    """
    p = Path(str(src))
    try:
        st = p.stat()
        size, mtime = int(st.st_size), round(float(st.st_mtime), 1)
    except OSError:
        size, mtime = 0, 0.0
    return {"video": str(p), "name": p.name, "w": int(w or 0), "h": int(h or 0),
            "duration": round(float(dur or 0), 3), "size": size, "mtime": mtime}


def _fp_diff(old, cur):
    """两份指纹的人话差异；完全一致返回 ""。"""
    if not isinstance(old, dict) or not isinstance(cur, dict):
        return ""
    bad = []
    if (old.get("name") or "") != (cur.get("name") or ""):
        bad.append("文件名 %s → %s" % (old.get("name") or "?", cur.get("name") or "?"))
    if (old.get("w"), old.get("h")) != (cur.get("w"), cur.get("h")):
        bad.append("分辨率 %sx%s → %sx%s"
                   % (old.get("w"), old.get("h"), cur.get("w"), cur.get("h")))
    od, cd = float(old.get("duration") or 0), float(cur.get("duration") or 0)
    if od > 0 and cd > 0 and abs(od - cd) > max(0.5, 0.02 * od):
        bad.append("时长 %.2fs → %.2fs" % (od, cd))
    return "；".join(bad)


def _read_src_fp(face_dir):
    """读上次的源指纹；没有/读坏都返回 None（读坏不阻断抽取）。"""
    try:
        p = Path(face_dir) / SRC_FP_NAME
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None


def _note_resume_src(face_dir, src, w, h, dur, done_n):
    """续跑命中时核对源素材；换过素材就往 JOB 告警 + 日志留一行。返回差异串。"""
    cur = _src_fingerprint(src, w, h, dur)
    old = _read_src_fp(face_dir)
    diff = _fp_diff(old, cur)
    if done_n > 0:
        if old is None:
            JOB.warn("extract-resume-src-unknown",
                     "续跑命中 %d 帧（aligned/ 里已有同号文件），但本项目没有源素材指纹"
                     "（4.0/旧版本留下的目录，或第一次由旧版写入）—— 无法确认这批帧与本次"
                     "素材是同一段；若刚换了素材，建议换项目或清空 aligned/。" % done_n)
        elif diff:
            JOB.warn("extract-resume-src-changed",
                     "续跑命中 %d 帧（aligned/ 里已有同号文件），但源视频与上次不同（%s）："
                     "这些帧可能来自另一段素材，本次会跳过所有「同号」帧 —— "
                     "换素材请换项目或清空 aligned/。" % (done_n, diff))
    try:
        (Path(face_dir) / SRC_FP_NAME).write_text(
            json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:                                   # noqa: BLE001 —— 写不了指纹不能弄挂抽取
        pass
    return diff


