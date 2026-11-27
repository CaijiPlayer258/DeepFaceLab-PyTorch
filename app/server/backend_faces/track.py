# -*- coding: utf-8 -*-
"""跨帧人脸轨迹：同一张脸在连续帧里保持同一个归属。

**为什么需要**：现在的判定是**每帧独立**的 —— 检测 → embedding → 和组均值比余弦
→ 过阈值才换。所以一帧被头发/手/别人的身体挡住，embedding 掉下去，`matched=False`，
成片里那张脸**突然变回原样**，下一帧又换回来 —— 用户看到的就是闪烁。
而它明明还是同一个人：在同一个位置、上一帧和下一帧都是这个组。

**做法**（不动 embedding 本身，只在判定外面加一层）：
  1. 用相邻帧的框做关联（IoU ≥ iou_min）→ 得到「轨迹」；
  2. 某张脸这一帧没匹配上，但和它关联的那条轨迹**历史上确定过**归属（hits ≥ min_hits）
     且那条轨迹没断太久（≤ max_gap 帧）→ **沿用轨迹的归属**；
  3. 但这一帧的相似度也不能太低：低于 `threshold × relax` 就不继承 ——
     否则真的换了一个人也会被硬拉回上一张脸。

**默认不开在「陌生人」上**：轨迹必须先靠自己的帧匹配成功过（min_hits），
所以它不会把「从没匹配上的脸」凭空变成某个组。

关掉：DFL_FACE_TRACK=0。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return str(os.environ.get("DFL_FACE_TRACK") or "1").strip().lower() not in (
        "0", "false", "no", "off")


def _iou(a, b) -> float:
    """[x1,y1,x2,y2] 的交并比。"""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return 0.0
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return float(inter / den) if den > 0 else 0.0


class FaceTracker:
    """一条视频一条轨迹集合。**非线程安全** —— 每条链路用自己的实例。

    max_gap   轨迹断了几帧还算同一条（30fps 下 3 帧 ≈ 0.1s）
    iou_min   相邻帧的框要重叠多少才算同一张脸
    relax     这一帧相似度 >= threshold*relax 才允许继承（防「真换人了」）
    min_hits  轨迹至少靠自己匹配成功过几次，才允许被继承
    """

    def __init__(self, max_gap: Optional[int] = None, iou_min: Optional[float] = None,
                 relax: Optional[float] = None, min_hits: Optional[int] = None):
        self.max_gap = int(max_gap if max_gap is not None else _envi("DFL_TRACK_GAP", 3))
        self.iou_min = float(iou_min if iou_min is not None
                             else _envf("DFL_TRACK_IOU", 0.42))
        self.relax = float(relax if relax is not None
                           else _envf("DFL_TRACK_RELAX", 0.70))
        self.min_hits = int(min_hits if min_hits is not None
                            else _envi("DFL_TRACK_HITS", 2))
        self.tracks: List[Dict[str, Any]] = []
        # name -> 磁盘路径。faces 里存的是**名字**，models 里要的是**路径** ——
        # 而某一帧可能整帧都没匹配上（plan.models 为空），这时候重建就得靠这张表。
        self._dfm_paths: Dict[str, str] = {}
        self.stats = {"frames": 0, "carried": 0, "rejected_low_sim": 0,
                      "new_tracks": 0, "expired": 0}

    # ---- 内部 ----
    def _active(self, frame_idx: int) -> List[Dict[str, Any]]:
        keep, dead = [], 0
        for t in self.tracks:
            if int(frame_idx) - int(t["last"]) <= self.max_gap:
                keep.append(t)
            else:
                dead += 1
        if dead:
            self.stats["expired"] = int(self.stats.get("expired") or 0) + dead
        self.tracks = keep
        return keep

    def _best_track(self, box, active) -> Optional[Dict[str, Any]]:
        best, best_iou = None, 0.0
        for t in active:
            v = _iou(box, t["box"])
            if v > best_iou:
                best, best_iou = t, v
        return best if best_iou >= self.iou_min else None

    # ---- 对外 ----
    def smooth(self, frame_idx: int, boxes, plan: Optional[dict],
               threshold: Optional[float] = None) -> Optional[dict]:
        """用轨迹修正一帧的 plan（就地改 faces / unmatched / models）。

        plan 为 None（没库 / 库空）时什么也不做，但**仍然更新轨迹**，这样
        后面切到「有库」时轨迹是热的。
        """
        if not enabled():
            return plan
        self.stats["frames"] = int(self.stats.get("frames") or 0) + 1
        boxes = list(boxes or [])
        active = self._active(int(frame_idx))
        if not boxes:
            return plan

        try:
            thr = float(threshold if threshold is not None
                        else (plan or {}).get("threshold") or 0.0)
        except (TypeError, ValueError):
            thr = 0.0
        floor = thr * self.relax if thr > 0 else 0.0

        for m in ((plan or {}).get("models") or []):
            try:
                nm = str(m.get("name") or "")
                pth = str(m.get("dfm") or "")
                if nm and pth:
                    self._dfm_paths[nm] = pth
            except Exception:                              # noqa: BLE001
                pass
        faces = list((plan or {}).get("faces") or [])
        by_face = {int(f.get("face", f.get("idx", -1))): f for f in faces}

        for i, b in enumerate(boxes):
            if i not in by_face:
                continue
            f = by_face[int(i)]
            t = self._best_track(b.get("box") or b, active)
            sim = f.get("similarity")
            if f.get("matched") and f.get("dfm"):
                # 这一帧自己匹配成功：以观测为准，刷新轨迹
                if t is None:
                    t = {"box": list(b.get("box") or b), "group": "", "dfm": "",
                         "name": "", "sim": 0.0, "hits": 0, "last": int(frame_idx)}
                    active.append(t)
                    self.tracks.append(t)
                    self.stats["new_tracks"] = int(self.stats.get("new_tracks") or 0) + 1
                t.update({"box": list(b.get("box") or b),
                          "group": str(f.get("group") or ""),
                          "dfm": str(f.get("dfm") or ""),
                          "name": str(f.get("group_name") or ""),
                          "sim": float(sim or 0.0),
                          "hits": int(t.get("hits") or 0) + 1,
                          "last": int(frame_idx)})
                continue
            # 这一帧没匹配上 —— 看看能不能沿用轨迹
            if t is None or not str(t.get("dfm") or ""):
                continue
            if int(t.get("hits") or 0) < self.min_hits:
                continue
            if sim is not None and float(sim) < floor:
                # 相似度掉得太狠：可能是真的换成别人了，不继承
                self.stats["rejected_low_sim"] = int(
                    self.stats.get("rejected_low_sim") or 0) + 1
                continue
            f["group"] = str(t.get("group") or "")
            f["group_name"] = str(t.get("name") or "")
            f["dfm"] = str(t.get("dfm") or "")
            f["model"] = str(t.get("dfm") or "")
            f["matched"] = True
            f["from_track"] = True          # 让调用方/日志能区分「这一帧自己匹配的」
            f["dfm_path"] = self._dfm_paths.get(str(t.get("dfm") or ""), "")
            t.update({"box": list(b.get("box") or b), "last": int(frame_idx)})
            self.stats["carried"] = int(self.stats.get("carried") or 0) + 1

        if plan is None:
            return plan
        # 重建 models / unmatched（faces 改了，这两份必须跟着）
        by_dfm: Dict[str, Dict[str, Any]] = {}
        unmatched: List[int] = []
        for f in faces:
            i = int(f.get("face", f.get("idx", -1)))
            if f.get("matched") and f.get("dfm"):
                ent = by_dfm.setdefault(str(f["dfm"]), {"faces": []})
                ent["faces"].append(i)
                if f.get("dfm_path"):
                    ent["path"] = f["dfm_path"]
            else:
                unmatched.append(i)
        models = []
        for pos, (nm, info) in enumerate(by_dfm.items()):
            # 路径优先用这一帧 plan 给的，其次用跨帧记下来的（整帧没命中时 models 是空的）
            _p = str(info.get("path") or self._dfm_paths.get(nm) or "")
            if _p:
                self._dfm_paths[nm] = _p
            models.append({"name": nm, "dfm": _p,
                           "faces": sorted(set(info["faces"])),
                           "index": int(pos)})
        plan["models"] = [m for m in models if m.get("dfm")]
        plan["unmatched"] = sorted(set(unmatched))
        plan["matched"] = sum(1 for f in faces if f.get("matched"))
        plan["track_carried"] = sum(1 for f in faces if f.get("from_track"))
        return plan

    def summary(self) -> str:
        """一行汇总（收尾打日志用）。"""
        s = self.stats
        if not int(s.get("frames") or 0):
            return ""
        return ("人脸轨迹：%d 帧 · 沿用轨迹 %d 次 · 相似度太低拒沿用 %d 次 · "
                "新建 %d 条 · 过期 %d 条（gap=%d iou=%.2f relax=%.2f hits=%d）"
                % (int(s.get("frames") or 0), int(s.get("carried") or 0),
                   int(s.get("rejected_low_sim") or 0),
                   int(s.get("new_tracks") or 0), int(s.get("expired") or 0),
                   self.max_gap, self.iou_min, self.relax, self.min_hits))
