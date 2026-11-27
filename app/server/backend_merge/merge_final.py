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
          limit=None, prefetch=2, png_dir=None, on_frame=None,
          start=0, end=None) -> dict:
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
    # ---- 懒加载：按帧序**一块一块**读，不把整池灌进内存 ----
    # 以前是把所有 chunk 一起 submit、全部读进来再合成。
    # 4K 一单实测服务端 RSS 峰值 5.0 GB，其中几百 MB 就是池里的脸原料
    # （416² 的 pred/mask/landmarks/xprd/xdst，每张脸 ~1.5 MB）。
    # 池本来就是「按帧序抽水」，所以窗口里留 2 块就够；读完即丢。
    import collections
    key_chunk = {}                     # "f<frame>_<idx>_pred" -> chunk 名
    for nm in names or []:
        try:
            with np.load(pool / nm) as z:
                for k in z.files:
                    key_chunk.setdefault(k, nm)
        except Exception:                                     # noqa: BLE001
            continue
    cache = collections.OrderedDict()
    # 缓存上限**不是常量**：它至少得能装下「同一帧要用的全部 chunk」。
    # 以前写死 2，于是 3 个模型在同一帧都有脸时：先读 A、B，再读 C 就把 A 挤出去，
    # 紧接着取 A 的数组拿到 None → continue → 那张脸**直接不贴，而且一行日志都没有**。
    # 现在由 _chunk_arrays 按当次需求动态抬高下限（只会变大，不会缩小）。
    _CACHE = {"max": 2}

    def _chunk_arrays(nm, need=1):
        if need > _CACHE["max"]:
            _CACHE["max"] = int(need)
        hit = cache.get(nm)
        if hit is not None:
            cache.move_to_end(nm)
            return hit
        arrs = _load_chunks(pool, [nm])
        cache[nm] = arrs
        while len(cache) > _CACHE["max"]:
            cache.popitem(last=False)
        return arrs

    keys = sorted({k.rsplit("_", 1)[0] for k in key_chunk if k.endswith("_pred")})
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

    # ★ 逐帧走完整条片 —— **没有脸的帧也必须写出去**。
    #
    # 以前是 `for frame in sorted(by_frame)`：只遍历「池里有脸」的帧，于是成片帧数
    # = 有脸的帧数而不是源片帧数。ffmpeg 用 -framerate fps -start_number 1 编码、
    # -c:a copy 直接拷音轨 → 视频被压成「有脸那几段」的总长、音轨还是原长，
    # 两者从第一个没脸的帧开始就错开。
    #
    # 用户说的「人脸位置根本对不上」和「两小时的片子只换了 50 分钟」是同一个 bug
    # 的两个说法：不是几何错位，是**时间轴错位**。
    # 区间：start/end 都是 **0 起**（调用方负责把界面的 1 起换过来）。
    # 以前这两个参数根本不存在，于是「选 100~200 帧导出」静默变成整片 ——
    # 音画不错开，只是范围选择完全失效，用户只会觉得「怎么还是全长」。
    _start = max(0, int(start or 0))
    _last = int(total) - 1 if int(total) > 0 else 0
    if by_frame:
        _last = max(_last, max(by_frame))
    if end is not None:
        _last = min(_last, int(end) - 1)
    if _last < _start:
        _last = _start
    _short_at = -1
    expected_faces = len(keys)
    swapped_faces = 0
    cur = _start - 1
    if _start > 0:
        # ★ seek 必须验：set 失败时捕获器还停在 0，而 cur 已经当成 _start-1，
        #   于是第 k 帧的画面会被当成 _start+k 帧去贴脸 —— 区间导出整段错位。
        _ok_seek = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, _start))
        if not _ok_seek:
            from .loud import note as _note_seek
            _note_seek("merge 定位到起始帧", RuntimeError("cap.set 返回 False"),
                       "区间起点可能不准，改用逐个 grab 走位", log)
            _cur_probe = 0
            while _cur_probe < _start and cap.grab():
                _cur_probe += 1
            cur = _cur_probe - 1
    cur_frame = 0
    written = 0
    swapped = 0
    for frame in range(_start, _last + 1):
        while cur < frame:
            if not cap.grab():
                break
            cur += 1
        ok, img = cap.retrieve()
        if not ok or img is None:
            # EOF（正常）和「中途解不出来」（坏帧）在这里长得一样，但后果差很远：
            # 后者会让成片从某一帧开始整段缺失，而调用方照样报「已完成」。
            # 记一笔，收尾时和「应该写到第几帧」对账。
            _short_at = int(frame)
            break                     # 读到 EOF：源片比 total 短，别再空转
        faces = []
        _rows = by_frame.get(frame) or []
        _need = {key_chunk.get(k + "_pred") for _i, k in _rows}
        for _nm in _need:
            if _nm:
                _chunk_arrays(_nm, need=len(_need))
        for _idx, k in sorted(_rows):
            _arr = cache.get(key_chunk.get(k + "_pred")) or {}
            pred = _arr.get(k + "_pred")
            mask = _arr.get(k + "_mask")
            lm = _arr.get(k + "_lm")
            if pred is None or mask is None or lm is None:
                continue
            _xp = _arr.get(k + "_xprd")
            _xd = _arr.get(k + "_xdst")
            _dstm = _arr.get(k + "_dstdm")   # DFM 自带的 dst 遮罩（mask_mode 3/4/5/9 要用）
            faces.append({"landmarks": np.asarray(lm, np.float32),
                          "pred": np.asarray(pred, np.float32) / 255.0,
                          "prd_dfm": np.asarray(mask, np.float32) / 255.0,
                          # XSeg 遮罩必须一起给：composition 的遮罩口径要和导出一致
                          # （没有它时实测差 34 LSB，差异全在脸框内）
                          "xseg_prd": (np.asarray(_xp, np.float32) / 255.0) if _xp is not None else None,
                          "xseg_dst": (np.asarray(_xd, np.float32) / 255.0) if _xd is not None else None,
                              'dst_dfm': (np.asarray(_dstm, np.float32) / 255.0) if _dstm is not None else None})
        swapped_faces += len(faces)
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
    if _short_at >= 0 and _short_at < _last:
        from .loud import note as _note_short
        _note_short("merge 逐帧读源片",
                   RuntimeError("第 %d 帧起解不出来（要到第 %d 帧）" % (_short_at, _last)),
                   "成片会缺第 %d 帧之后的内容" % _short_at, log)
    # ★ 池里的脸数和真正贴上去的对账。chunk 读不开（:63）、数组缺一个（:154）都会
    #   静默 continue —— 用户看到的是「某些帧的脸没换」，日志里一个字都没有。
    _missing = int(expected_faces) - int(swapped_faces)
    if _missing > 0:
        from .loud import note as _note_miss
        _note_miss("merge 取池里的人脸",
                   RuntimeError("池里有 %d 张脸，实际只贴上 %d 张"
                                % (int(expected_faces), int(swapped_faces))),
                   "有 %d 张脸没换（chunk 损坏或缺失）" % _missing, log)
    res = {"out": str(out_path), "frames": int(written), "swapped": int(swapped),
           "faces_keys": int(len(keys)), "source_frames": int(total),
           # fps 必须回传：pipeline_passes 要用它给 ffmpeg 定 -framerate。
           # 以前这里没有这个键，那边 `float(res.get("fps") or 25.0)` 就永远拿 25 ——
           # 30fps 的源片编出来会被拉长 30/25 = 1.2 倍，而音轨是 -c:a copy 原速，
           # 于是从第一秒起音画就漂，越到后面越离谱。（fused / 分段两条路的 fps 都
           # 来自 _vid_probe，只有 passes 这条掉了。）
           "fps": float(fps),
           "elapsed_s": round(time.time() - t0, 1), "size": [w, h]}
    log("[merge] %d 帧（其中 %d 帧贴了脸）· %dx%d · %.1fs · %s"
        % (res["frames"], res["swapped"], w, h, res["elapsed_s"], out_path))
    return res
