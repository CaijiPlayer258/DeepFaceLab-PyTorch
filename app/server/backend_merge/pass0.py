# -*- coding: utf-8 -*-
"""Pass 0：扫描整片，产出**人脸分配表**（多模型分趟导出的第一步）。

产物 <视频>_cache/pool/index.json：
    {video, frames, stride, detected, swap_faces,
     faces: [{frame, idx, box:[x,y,w,h], kps68:[[x,y]x68], model, group, similarity, matched}]}

为什么单独一趟：
  · 检测 + 68 点 + embedding 只需做一次，后面每一趟（模型 A/B/C…）都复用这张表；
  · 表里的 swap_faces 就是前端「人脸暂存池」的**分母**（只数会被换的脸）；
  · 分母在扫描过程中是动态增长的，扫完才冻结 —— 界面在 Pass 0 不显示百分比。

i18n：本模块只产出机器字段，不带任何界面文案；节点名由前端 i18n 表给。
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _pool_dir(video: Path) -> Path:
    d = video.parent / (video.stem + "_cache") / "pool"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scan(video, detector="tinymog", thresh=0.5, plan_payload=None, models=None,
         stride=1, start=0, end=None, out_dir=None, log=print,
         with_kps=False, on_frame=None) -> dict:
    """扫一遍视频，把「哪一帧的第几张脸归哪个模型」写成 index.json。

    plan_payload 交给 _auto_face_plan 判归属（有人脸库就按 embedding 判，没有就用 models）。
    models 是显式分配（带 faces 的那份）；给了就**优先**，不去打扰人脸库。
    """
    import cv2
    import numpy as np
    from .compose import _detect_boxes, _frame_landmarks

    video = Path(str(video))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("打不开视频：%s" % video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(stride or 1))
    i0 = max(0, int(start or 0))
    i1 = int(end) if end is not None else total
    if i1 <= 0 or i1 > total:
        i1 = total

    explicit = [m for m in (models or []) if (m.get("faces") or [])]
    named = set()
    for m in explicit:
        named.update(int(x) for x in (m.get("faces") or []))
    any_all = any(not (m.get("faces") or []) for m in (models or []))

    faces_out = []
    detected = 0
    t0 = time.time()
    idx = i0
    # 只在开始时 seek 一次，之后**顺序读**。每帧都 cap.set(POS_FRAMES) 会让解码器
    # 反复回退到关键帧（实测 40 帧要 18.5s），顺序读只要 35ms/帧。
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    # ★ 这里以前是 `_cur = i0` + `cap.read()`，语义上等于「每轮走 2 帧」：
    #   read() = grab() + retrieve()，它自己就把解码器往前推了一帧，而下面的
    #   while 又按 _cur 补 grab。实测（逐帧编码唯一颜色再读回来验）：
    #     index.json 记 frame=k，实际检测的是第 2k 帧（0→0, 1→2, 2→4 …）；
    #     80 帧的片子，标签到 40 就 EOF —— **只扫了前一半**。
    #   后果正是用户报的那两条：只有前一半帧有人脸（「两小时只换了 50 分钟」，
    #   理论正好一半），而记在 k 的框其实属于 2k 帧那张脸，pass_swap 拿它去贴
    #   第 k 帧就是「人脸位置对不上」（时间轴错位，不是几何错位）。
    #
    #   现在和 pass_swap / merge_final 用同一个 cur 约定：_cur 指向「已经读过的
    #   最后一帧」，-1 表示一帧都没读；取帧用 retrieve()（不前进）。
    _cur = i0 - 1
    while idx < i1:
        while _cur < idx:                     # stride>1 时把中间的帧读掉
            if not cap.grab():
                break
            _cur += 1
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        boxes = list(_detect_boxes(frame, detector, thresh) or [])
        detected += len(boxes)
        if on_frame:
            try:
                on_frame(int(len(boxes)))
            except Exception:                              # noqa: BLE001
                pass
        owner = {}       # 脸序号 -> (model, group, sim, matched)
        if explicit:
            for bi in range(len(boxes)):
                if any_all or bi in named:
                    for m in (models or []):
                        fs_ = m.get("faces") or []
                        if (not fs_) or (bi in [int(x) for x in fs_]):
                            owner[bi] = (str(m.get("name") or ""), "", None, True)
                            break
            # 显式只点了一部分脸（或者前端把某个模型丢了）：剩下的脸交给库去判。
            # 不补这一步，没被点名的脸永远不换 —— 用户看到「两个模型只换了一个人」。
            if plan_payload is not None and len(owner) < len(boxes):
                try:
                    from .compose import _auto_face_plan
                    _plan2 = _auto_face_plan(plan_payload, frame, plan_payload.get("snap") or {},
                                            "video", "", int(idx), detector, float(thresh),
                                            boxes=boxes)
                except Exception as _e2:                       # noqa: BLE001
                    from .loud import note as _ln
                    _ln("pass0 库兜底", _e2, "没被点名的脸不会换（只换显式点名的那几张）", log)
                    _plan2 = None
                for f in ((_plan2 or {}).get("faces") or []):
                    _bi = int(f.get("idx") or 0)
                    if _bi in owner or not f.get("matched"):
                        continue
                    owner[_bi] = (str(f.get("dfm") or ""), str(f.get("group") or ""),
                                  f.get("similarity"), True)
        elif plan_payload is not None:
            try:
                from .compose import _auto_face_plan
                plan = _auto_face_plan(plan_payload, frame, plan_payload.get("snap") or {},
                                       "video", "", int(idx), detector, float(thresh),
                                       boxes=boxes)
            except Exception as e:                            # noqa: BLE001
                log("[pass0] 第 %d 帧归属判定失败（%s）" % (idx, str(e)[:100]))
                plan = None
            if plan:
                for f in (plan.get("faces") or []):
                    if not f.get("matched"):
                        continue
                    owner[int(f.get("idx") or 0)] = (str(f.get("dfm") or ""),
                                                     str(f.get("group") or ""),
                                                     f.get("similarity"), True)
        else:
            for bi in range(len(boxes)):
                owner[bi] = (str((models or [{}])[0].get("name") or ""), "", None, True)
        for bi, meta in owner.items():
            if bi >= len(boxes):
                continue
            b = boxes[bi]
            box = list(b.get("box") or [])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            wh = [int(round(x1)), int(round(y1)), int(round(max(0.0, x2 - x1))),
                  int(round(max(0.0, y2 - y1)))]
            lm = None   # 68 点默认不算（每张脸 ~350ms，Pass 0 只需要框和归属）
            if with_kps:
                lm = _frame_landmarks(frame, b, [x1, y1, x2, y2])
            faces_out.append({"frame": int(idx), "idx": int(bi), "box": wh,
                              "kps5": ([[round(float(a), 2), round(float(b2), 2)]
                                       for a, b2 in (b.get("kps") or [])] or None),
                              "kps68": (np.asarray(lm).round(2).tolist() if lm is not None else None),
                              "model": meta[0], "group": meta[1],
                              "similarity": meta[2], "matched": bool(meta[3])})
        if idx % 100 == 0:
            log("[pass0] frame %d/%d · 检测 %d · 待换脸 %d · %.0fs"
                % (idx, i1, detected, len(faces_out), time.time() - t0))
        idx += stride
    cap.release()

    out = {"video": str(video), "frames": int(i1), "stride": int(stride),
           "detected": int(detected), "swap_faces": int(len(faces_out)),
           "models": [str(m.get("name") or "") for m in (models or [])],
           "elapsed_s": round(time.time() - t0, 1), "faces": faces_out}
    d = Path(str(out_dir)) if out_dir else _pool_dir(video)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "index.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log("[pass0] 完成：检测 %d 张 · 待换脸 %d 张 · %.1fs · %s"
        % (out["detected"], out["swap_faces"], out["elapsed_s"], p))
    return out
