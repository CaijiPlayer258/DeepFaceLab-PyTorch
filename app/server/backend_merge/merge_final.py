# -*- coding: utf-8 -*-
"""合流（docs/37 第 3 步）：闸门打开后，按帧号顺序抽「人脸暂存池」的水，一次合成、一次编码。

设计要点（写在前面，改的时候别破坏）：
  · **每帧只混合一次**：把所有模型的脸一起交给 compose_frame —— 分趟只影响"谁算 pred"，
    不影响"怎么贴"。这样结果与今天单趟导出等价（seamless 遇重叠脸另计）。
  · 顺序抽水：帧号升序，编码器输入顺序必须和原片一致。
  · loader 预取：读 chunk 的工作交给线程池，合成线程不等磁盘。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _pool_dir(video: Path) -> Path:
    return video.parent / (video.stem + "_cache") / "pool"


def _load_chunks(pool: Path, names) -> dict:
    """把若干 chunk 读进内存：{key: array}。chunk 是"相邻 16 帧"的合批，一次读一个。"""
    out = {}
    for nm in names:
        with np.load(pool / nm) as z:
            for k in z.files:
                out[k] = z[k]
    return out


def merge(video, passes, out_path=None, cfg=None, frames=None, log=print,
          limit=None, prefetch=2, png_dir=None, on_frame=None) -> dict:
    """passes = [pass_<model>.json 的路径, ...]（顺序无关，按帧号合并）。"""
    import cv2
    from .compose import compose_frame

    video = Path(str(video))
    pool = _pool_dir(video)
    metas = [json.loads(Path(str(p)).read_text(encoding="utf-8")) for p in passes]
    names = []
    for m in metas:
        for c in (m.get("chunks") or []):
            if c not in names:
                names.append(c)

    t0 = time.time()
    cache = {}
    with ThreadPoolExecutor(max_workers=max(1, int(prefetch))) as pool_ex:
        futs = {nm: pool_ex.submit(_load_chunks, pool, [nm]) for nm in names}
        for nm in names:
            cache.update(futs[nm].result())      # 消费顺序 = 提交顺序 = 帧序（顺序读）
            futs.pop(nm, None)

    keys = sorted({k.rsplit("_", 1)[0] for k in cache if k.endswith("_pred")})
    by_frame = {}
    for k in keys:
        fr, idx = k[1:].split("_")
        by_frame.setdefault(int(fr), []).append((int(idx), k))

    cfg = dict(cfg or {})
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("打不开视频：%s" % video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = None
    if png_dir:
        # 生产路径：帧写 PNG，交给 _merge_runner._build_encode_cmd 那套 ffmpeg
        Path(str(png_dir)).mkdir(parents=True, exist_ok=True)
        out_path = str(png_dir)
    else:
        if out_path is None:
            out_path = str(video.with_name(video.stem + "_passes.mp4"))
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cur = -1
    cur_frame = 0
    written = 0
    swapped = 0
    for frame in sorted(by_frame):
        while cur < frame:
            if not cap.grab():
                break
            cur += 1
        ok, img = cap.retrieve()
        if not ok or img is None:
            continue
        faces = []
        for _idx, k in sorted(by_frame[frame]):
            pred = cache.get(k + "_pred")
            mask = cache.get(k + "_mask")
            lm = cache.get(k + "_lm")
            if pred is None or mask is None or lm is None:
                continue
            faces.append({"landmarks": np.asarray(lm, np.float32),
                          "pred": np.asarray(pred, np.float32) / 255.0,
                          "prd_dfm": np.asarray(mask, np.float32) / 255.0,
                          "xseg_prd": None, "xseg_dst": None,
                          "dst_dfm": None})
        out = compose_frame(img, faces, cfg, {}) if faces else img
        # 注：GPU seamless（compose_gpu.seamless_merge）需要把 compose 的图层准备
        # （warp / 遮罩 / 色彩迁移）整段搬到 GPU 上下文里，属于单独一单的活。
        # 这里**不假接**：宁可慢，也不能让「看起来走了 GPU」而结果不同。
        if faces:
            swapped += 1
        if vw is not None:
            vw.write(out)
        else:
            cv2.imwrite(str(Path(str(png_dir)) / ("%08d.png" % (written + 1))), out)
        written += 1
        if on_frame:
            try:
                on_frame(written)
            except Exception:                              # noqa: BLE001
                pass
        if limit and written >= int(limit):
            break
    if vw is not None:
        vw.release()
    cap.release()
    res = {"out": str(out_path), "frames": int(written), "swapped": int(swapped),
           "faces_keys": int(len(keys)), "source_frames": int(total),
           "elapsed_s": round(time.time() - t0, 1), "size": [w, h]}
    log("[merge] %d 帧（其中 %d 帧贴了脸）· %dx%d · %.1fs · %s"
        % (res["frames"], res["swapped"], w, h, res["elapsed_s"], out_path))
    return res
