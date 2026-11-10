# -*- coding: utf-8 -*-
r"""切脸阶段（② faces）的逐帧 profile：读图 / 检测 / 2D106 / 裁剪 / JPEG 编码写盘。

为什么要有这个工具：用户报「切脸 ~22 帧/秒、1439 帧约 65 秒」。要优化就得先知道
这 45ms 花在哪。这个脚本**不改任何生产代码**，只在 import 之后给几个函数套计时：
    rm._stage_read_image        读帧图（frames/000001.png）
    rm._detect_boxes            人脸检测（含检测器 forward）
    rm._landmarks_model().landmarks68   2D106 关键点
    _align.align_face_68        68 点仿射裁剪
    _align.write_dfljpg         对齐脸落盘（JPEG q100 + 4:4:4 编码 + 写文件）
剩下的时间归「其它」（框排序、pose、meta 落盘、进度上报）。

它跑的是**真的 _stage_do_faces**（同一份代码、同一个检测器、同样的 thresh），
只是把帧目录换成了 %TEMP% 里的一份拷贝。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\faces_profile.py
    runtime\server\Scripts\python.exe app\tools\faces_profile.py --frames 60 --det retina
    runtime\server\Scripts\python.exe app\tools\faces_profile.py --read-bench

退出码 0 = 跑完（不是断言）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMP = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
BOX = TEMP / "dfl_faces_profile"
SRC_CACHE = REPO / "workspace" / "9月29日 (5)_cache"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.server import app as A                      # noqa: E402  （先 import 才有 router）
from app.server import _align                        # noqa: E402
from app.server import routes_merge as rm            # noqa: E402


# ---------------------------------------------------------------- 计时
class Clock:
    def __init__(self):
        self.d = {}

    def add(self, k, dt):
        self.d[k] = self.d.get(k, 0.0) + dt

    def bump(self, k, n=1):
        self.d[k + ".n"] = self.d.get(k + ".n", 0) + n

    def ms(self, k, n):
        return 1000.0 * self.d.get(k, 0.0) / max(1, n)


C = Clock()
LOCK = threading.Lock()


def _wrap(mod, name, key, path=None):
    """把 mod.name 换成计时版；path 可选（'a.b' 这种点号属性）。"""
    obj = mod
    for part in (path.split(".") if path else []):
        obj = getattr(obj, part)
    orig = getattr(obj, name)

    def inner(*a, **kw):
        t = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            dt = time.perf_counter() - t
            with LOCK:
                C.add(key, dt)
    setattr(obj, name, inner)
    return orig


def install_hooks():
    _wrap(rm, "_stage_read_image", "read")
    _wrap(rm, "_detect_boxes", "detect")
    _wrap(_align, "align_face_68", "crop")
    _wrap(_align, "write_dfljpg", "encode")
    orig_lm = rm._landmarks_model

    class LMWrap:
        def __init__(self, m):
            self._m = m

        def landmarks68(self, img):
            t = time.perf_counter()
            try:
                return self._m.landmarks68(img)
            finally:
                dt = time.perf_counter() - t
                with LOCK:
                    C.add("lm68", dt)

        def __getattr__(self, k):
            return getattr(self._m, k)

    def lm_factory():
        m = orig_lm()
        if m is None:
            return None
        if not getattr(lm_factory, "wrapped", False):
            lm_factory.wrapped = True
            lm_factory.m = LMWrap(m)
        return lm_factory.m

    rm._landmarks_model = lm_factory


# ---------------------------------------------------------------- 准备
def build_box(n_frames: int, src: Path, link: bool = False) -> Path:
    """做一个干净的临时 cache，frames/ 里只放前 n 帧。

    link=True 时不拷贝：用**目录联接**把真 frames/ 挂进来（只读），2.7GB 的帧零成本，
    用来跑整条片子（1439 帧）；产物照样写进临时 aligned/，不碰用户的缓存。
    """
    frames = rm._stage_files(src / "frames")
    if not frames:
        raise SystemExit("源缓存里没有 frames/：%s" % src)
    cache = BOX / "cache"
    if cache.exists():
        # 联接目录不能用 rmtree 粗暴删（会顺着链接删到真帧），先摘掉链接
        link_dir = cache / "frames"
        if link_dir.exists() and link_dir.is_dir():
            subprocess.run(["cmd", "/c", "rmdir", str(link_dir)], capture_output=True)
        shutil.rmtree(cache, ignore_errors=True)
    (cache / "aligned").mkdir(parents=True)
    if link:
        cache.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(cache / "frames"),
                            str(src / "frames")], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("建 frames/ 目录联接失败：%s" % (r.stdout or r.stderr))
    else:
        (cache / "frames").mkdir(parents=True)
        for p in frames[:n_frames]:
            shutil.copy2(str(p), str(cache / "frames" / p.name))
    return cache


def make_cfg(cache: Path, n: int, det: str, thresh: float) -> dict:
    video = Path(str(cache).replace("_cache", ""))
    meta = {
        "version": 1,
        "source": {"video": str(SRC_CACHE.parent / "9月29日 (5).mp4"),
                   "name": "probe.mp4", "fps": 30.0, "frames": n,
                   "width": 1080, "height": 1920},
        "settings": {}, "stages": {k: {} for k in rm._STAGE_ORDER},
        "frame_swapped": False, "created_at": time.time(),
        "range": {"start": 1, "end": n, "total": n, "all": n, "full": True,
                  "key": "1-%d" % n},
    }
    cfg = {
        "cache": str(cache), "video": str(video), "total": n, "meta": meta,
        "range": dict(meta["range"]), "range_key": "1-%d" % n, "range_changed": False,
        "force": True,
        "detector": det, "det_align": det, "thresh": thresh, "align_thresh": thresh,
        "dfm": "", "model": "", "xseg": "", "keep_cache": True, "probe": {},
        "settings": {}, "snap": {}, "stages": ["faces"], "chain": False,
        "pred_batch": 0, "merge_threads": 1, "png_level": 0,
    }
    return cfg


# ---------------------------------------------------------------- 读图基准
def read_bench(n: int):
    """三种读法的单帧耗时：_imread_any / 裸 open+imdecode / 线程池预取。"""
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    frames = rm._stage_files(SRC_CACHE / "frames")[:n]
    if not frames:
        return {}

    def t(fn, label):
        t0 = time.perf_counter()
        for p in frames:
            fn(p)
        return (time.perf_counter() - t0) * 1000.0 / len(frames)

    def imread_any(p):
        return A._imread_any(p)

    def raw_decode(p):
        with open(str(p), "rb") as fh:
            data = fh.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    out = {"_imread_any": round(t(imread_any, "a"), 2),
           "open+imdecode": round(t(raw_decode, "b"), 2)}
    for nw in (2, 4, 6, 8):
        with ThreadPoolExecutor(max_workers=nw) as ex:
            t0 = time.perf_counter()
            list(ex.map(raw_decode, frames))
            out["pool%d" % nw] = round((time.perf_counter() - t0) * 1000.0 / len(frames), 2)
    return out


# ---------------------------------------------------------------- 「跳过无人脸帧」的误判率
def skip_eval(cache: Path, thresholds=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0)) -> dict:
    """评估低成本判据「上一帧没脸 + 画面几乎没变 → 这一帧直接跳过检测」。

    判据（每帧 ~0.06ms，比检测便宜两个数量级）：
        thumb = 32x32 灰度
        skip  = (上一帧检测到 0 张脸) and (mean|thumb - prev_thumb| < T)
    真值 = aligned/ 里这一帧有没有脸（就是刚跑完的这一遍的产物）。

    误判的两种：
        FN 假阴性（危险）：真值有脸，却被跳过 → 这一帧的脸永久丢失
        FP 假阳性（无害）：真值没脸，没跳过 → 白跑一次检测
    输出每个阈值下的 FN / FP / 能省掉多少帧。
    """
    import cv2
    import numpy as np

    frames = rm._stage_files(cache / "frames")
    aligned = rm._stage_files(cache / "aligned", (".jpg", ".jpeg"))
    truth = set()
    for p in aligned:
        try:
            truth.add(int(p.stem.split("_")[0]))
        except (ValueError, IndexError):
            continue
    if not frames:
        return {}
    th = {t: {"skip": 0, "fn": 0, "fp": 0} for t in thresholds}
    prev_thumb = None
    prev_has = True              # 第一帧一定检测
    t_thumb = 0.0
    for p in frames:
        img = None
        with open(str(p), "rb") as fh:
            data = fh.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev_has = True
            continue
        t0 = time.perf_counter()
        thumb = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        diff = 0.0 if prev_thumb is None else float(
            np.abs(thumb.astype(np.int16) - prev_thumb.astype(np.int16)).mean())
        t_thumb += time.perf_counter() - t0
        idx = 0
        try:
            idx = int(p.stem)
        except ValueError:
            idx = 0
        has = idx in truth
        for t in thresholds:
            d = th[t]
            if (not prev_has) and diff < t:
                d["skip"] += 1
                if has:
                    d["fn"] += 1
            else:
                d["fp"] += 0
        prev_thumb = thumb
        prev_has = has
    n = len(frames)
    noface = n - len(truth)
    return {
        "frames": n, "frames_with_face": len(truth), "frames_without_face": noface,
        "thumb_ms": round(1000.0 * t_thumb / max(1, n), 3),
        "thresholds": {str(t): {
            "skipped": v["skip"],
            "skipped_pct": round(100.0 * v["skip"] / max(1, n), 1),
            "fn": v["fn"],
            "fn_rate_pct": round(100.0 * v["fn"] / max(1, len(truth)), 2),
        } for t, v in th.items()},
    }


# ---------------------------------------------------------------- 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--det", default="retina", choices=("retina", "tiny"))
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--read-bench", action="store_true")
    ap.add_argument("--link", action="store_true",
                    help="整条片子：frames/ 用目录联接挂进来（不拷贝）")
    ap.add_argument("--warm", type=int, default=8, help="计时前的预热帧数（GPU 升频）")
    ap.add_argument("--skip-eval", action="store_true",
                    help="跑完再评估「跳过无人脸帧」判据的误判率")
    ap.add_argument("--json", default="")
    ap.add_argument("--src", default=str(SRC_CACHE))
    args = ap.parse_args()

    det = {"retina": "retinaface_10g", "tiny": "tinymog"}[args.det]
    src = Path(args.src)
    n = int(args.frames) if args.frames > 0 else len(rm._stage_files(src / "frames"))
    if args.link:
        n = len(rm._stage_files(src / "frames"))

    rb = read_bench(min(24, n)) if args.read_bench else {}
    if rb:
        print("[read] 单帧毫秒：" + json.dumps(rb, ensure_ascii=False), flush=True)

    cache = build_box(n, src, link=bool(args.link))
    cfg = make_cfg(cache, n, det, args.thresh)

    install_hooks()
    # 预热：检测器 / 2D106 / cuda 都还没加载，别把首帧的加载时间算进 fps。
    # 检测本身也要预热：5070 在 P8 低功耗档时单帧能慢到 50ms，连跑十几帧 GPU 才升频，
    # 只跑 1 帧的话 fps 会被首帧的降频状态带偏（这一条踩过，量出来的数忽大忽小）。
    t0 = time.perf_counter()
    warm = rm._detector(det)
    lm = rm._landmarks_model()
    if lm is None:
        raise SystemExit("2D106 模型不可用")
    warm_paths = rm._stage_files(cache / "frames")[:max(0, int(args.warm))]
    for p in warm_paths:
        img = rm._stage_read_image(p)
        if img is not None:
            rm._stage_faces_detect(warm, [img], args.thresh)
    warm_up = time.perf_counter() - t0

    C.d.clear()
    t0 = time.perf_counter()
    res = rm._stage_do_faces(cfg)
    wall = time.perf_counter() - t0

    frames_n = int(res.get("frames") or n)
    faces_n = int(res.get("count") or 0)
    fps = frames_n / max(1e-6, wall)

    # 阶段自己也会出一份 profile（meta.stages.faces.profile）—— 以它为准：
    # 那是生产代码内部量的，跟界面上看到的、验收断言的是同一份数。
    prof = res.get("profile") or {}
    pf = prof.get("per_frame_ms") or {}
    order = [("读帧 read", "read"), ("检测 detect", "detect"), ("2D106 lm68", "lm68"),
             ("裁剪 crop", "crop"), ("编码写盘 encode", "encode"), ("其它 other", "other")]
    total_ms = float(prof.get("per_frame_total_ms") or (1000.0 * wall / max(1, frames_n)))
    table = {
        "detector": det, "frames": frames_n, "faces": faces_n,
        "wall_ms": round(total_ms, 2), "fps": round(float(prof.get("fps") or fps), 2),
        "warmup_s": round(warm_up, 2),
        "per_frame_ms": {lbl: pf.get(k) for lbl, k in order if k in pf},
        "lm68_ms_per_face": prof.get("lm68_ms_per_face"),
        "pipeline": prof.get("pipeline") or {},
        "read_bench": rb,
    }
    print("=" * 70)
    print("检测器 %-14s 帧 %d · 脸 %d" % (det, frames_n, faces_n))
    print("流水线 %s" % json.dumps(table["pipeline"], ensure_ascii=False))
    print("总 %6.2f ms/帧  →  %5.2f 帧/秒" % (total_ms, table["fps"]))
    print("-" * 70)
    for lbl, k in order:
        if k not in pf:
            continue
        v = float(pf.get(k) or 0.0)
        print("  %-18s %7.2f ms  %5.1f%%" % (lbl, v, 100.0 * v / max(total_ms, 1e-6)))
    print("  (2D106 按脸算：%s ms/张，%d 张)" % (table["lm68_ms_per_face"], faces_n))
    print("=" * 70)
    if args.skip_eval:
        se = skip_eval(cache)
        table["skip_eval"] = se
        print("跳过无人脸帧的判据（上一帧无脸 + 32x32 灰度差 < T）：")
        print("  真值 %d 帧有脸 / %d 帧无脸 · 缩略图开销 %.3f ms/帧"
              % (se.get("frames_with_face", 0), se.get("frames_without_face", 0),
                 se.get("thumb_ms", 0)))
        for t, v in (se.get("thresholds") or {}).items():
            print("  T=%-4s 跳过 %4d 帧（%4.1f%%）· 假阴性 %3d 帧（%5.2f%% 的有脸帧会丢）"
                  % (t, v["skipped"], v["skipped_pct"], v["fn"], v["fn_rate_pct"]))
    if args.json:
        Path(args.json).write_text(json.dumps(table, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print("写 %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
