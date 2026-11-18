# -*- coding: utf-8 -*-
"""Pass k：只装一个模型，跑它名下的脸，把「脸原料」按 chunk 落盘（docs/37 第 2 步）。

产物（默认 <视频>_cache/pool/）：
    pass_<model>.json     脸清单：{frame, idx, chunk, key, size, box, model}
    chunks/<model>_0000.npz  相邻 chunk_frames 帧的原料合批，键 = "f<frame>_<idx>"
        每键两个数组：pred (S,S,3 u8) · prd_dfm (S,S u8)

为什么一趟只装一个模型：两个 1.5GB 的 dfm 会话在 8GB 卡上会互相把对方挤进换页，
实测 warm 阶段直接卡住（线程栈见 workspace/*.stacks.txt）。一趟一模型，显存就只有一份。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _pool_dir(video: Path) -> Path:
    d = video.parent / (video.stem + "_cache") / "pool"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(video, index_path, model_name, dfm_path, chunk_frames=16,
        limit=None, log=print, out_dir=None) -> dict:
    """跑一趟：只处理 index.json 里属于 model_name 的脸。

    返回 {model, faces, chunks, size, elapsed_s, path}。
    """
    import cv2
    from .pipeline_stages import _dfm_session, _dfm_run_batch
    from facelib.LandmarksProcessor import get_transform_mat
    from facelib.FaceType import FaceType

    video = Path(str(video))
    idx = json.loads(Path(str(index_path)).read_text(encoding="utf-8"))
    mine = [f for f in (idx.get("faces") or []) if str(f.get("model") or "") == str(model_name)]
    if limit:
        mine = mine[:int(limit)]
    if not mine:
        log("[pass:%s] 这张表里没有属于它的脸" % model_name)
        return {"model": model_name, "faces": 0, "chunks": 0, "size": 0, "elapsed_s": 0.0,
                "path": ""}

    ent = _dfm_session(str(dfm_path))
    size = int(ent[2])
    t0 = time.time()
    d = Path(str(out_dir)) if out_dir else _pool_dir(video)
    (d / "chunks").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("打不开视频：%s" % video)
    cur = -1
    buf, buf_from, buf_key = {}, None, 0
    chunks, done = [], 0
    by_frame = {}
    for f in mine:
        by_frame.setdefault(int(f["frame"]), []).append(f)

    def _flush():
        nonlocal buf, chunks
        if not buf:
            return
        name = "chunks/%s_%04d.npz" % (model_name, len(chunks))
        np.savez_compressed(str(d / name), **buf)
        chunks.append(name)
        buf = {}

    for frame in sorted(by_frame):
        while cur < frame:                                   # 顺序读，不逐帧 seek
            ok = cap.grab()
            if not ok:
                break
            cur += 1
        ok, img = cap.retrieve()
        if not ok or img is None:
            continue
        faces = by_frame[frame]
        ins, mats = [], []
        for f in faces:
            try:
                x, y, w, h = [float(v) for v in (f.get("box") or ())]
            except Exception:                                # noqa: BLE001
                continue
            from .compose import _frame_landmarks
            # _frame_landmarks 要的是 {box:[x1,y1,x2,y2]}；index 里存的是 [x,y,w,h]
            det = {"box": [x, y, x + w, y + h], "kps": f.get("kps5") or []}
            lm = _frame_landmarks(img, det, [x, y, x + w, y + h])
            if lm is None:
                continue
            mat = get_transform_mat(np.asarray(lm, np.float32), size, FaceType.WHOLE_FACE, scale=1.0)
            warped = cv2.warpAffine(img, mat, (size, size), flags=cv2.INTER_CUBIC)
            ins.append(np.clip(warped.astype(np.float32) / 255.0, 0, 1))
            mats.append((f, mat))
        if not ins:
            continue
        outs = _dfm_run_batch(ent, np.stack(ins, axis=0), want_u8=True)
        for (f, _mat), got in zip(mats, outs):
            try:
                pred, prd, _dst = got[0], got[1], (got[2] if len(got) > 2 else None)
            except Exception:                                # noqa: BLE001
                continue
            p = np.asarray(pred)
            if p.dtype != np.uint8:
                p = np.clip(p * 255.0, 0, 255).astype(np.uint8)
            m = np.asarray(prd)
            if m.dtype != np.uint8:
                m = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            if m.ndim == 3:
                m = m[:, :, 0]
            key = "f%06d_%d" % (int(f["frame"]), int(f["idx"]))
            if buf_from is None:
                buf_from = int(f["frame"])
            buf[key + "_pred"] = p
            buf[key + "_mask"] = m
            done += 1
        if buf_from is not None and (frame - buf_from) >= int(chunk_frames):
            _flush()
            buf_from = None
    _flush()
    cap.release()

    meta = {"model": str(model_name), "dfm": str(dfm_path), "size": int(size),
            "faces": int(done), "chunks": list(chunks),
            "elapsed_s": round(time.time() - t0, 1)}
    p = d / ("pass_%s.json" % model_name)
    p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log("[pass:%s] %d 张脸 · %d 个 chunk · %d px · %.1fs · %s"
        % (model_name, done, len(chunks), size, meta["elapsed_s"], p))
    meta["path"] = str(p)
    return meta
