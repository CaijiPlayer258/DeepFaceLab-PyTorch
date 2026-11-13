# -*- coding: utf-8 -*-
r"""检测输入对拍：detect(img)  vs  prep(img)+detect_ctx(ctx)  —— 必须**逐位一致**。

为什么要有这个脚本：整条导出链上「检测器 letterbox」被拆成了两步（decode 线程做
resize/填边，检测线程只做前向 + NMS）。拆的前提是**两步拼起来和一步一模一样** ——
同一个 _prep_ctx、同一个 INTER_LINEAR、同一张 canvas，框和 5 个关键点必须 0 误差。
差一个像素就会改合成结果（贴回的位置换了），所以这里按「逐框 0.000px」验收，比
「≤0.5px」更严：这条路上本来就不该有任何近似。

顺带把「让 ffmpeg 直接解 640」那条路的代价也量出来（--scale-route）：swscale 的
bilinear 和 cv2 的 INTER_LINEAR 不是同一个核，实测差 1.5 LSB 的像素 → 框偏 3~7px、
还会丢脸。**这就是为什么最终没走 -vf scale 那条路。**

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\det_prep_parity.py
    runtime\server\Scripts\python.exe app\tools\det_prep_parity.py --frames 24 --scale-route
    runtime\server\Scripts\python.exe app\tools\det_prep_parity.py --video X.mp4

退出码：0 = 全部一致；2 = 有误差（按 max|Δbox| / max|Δkps| 报）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("DFL_STAGE_KEEP", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                              # noqa: E402

from app.server import app as A                                 # noqa: E402
from app.server import routes_merge as rm                       # noqa: E402

DETECTORS = ("tinymog", "retinaface_10g")


def grab(video: Path, n: int, start: int = 1):
    probe = rm._vid_probe(video)
    w, h = int(probe["w"]), int(probe["h"])
    proc, nxt = rm._fused_open_frames(video, start, size=(w, h), mode="raw", bits=8)
    out = []
    try:
        for _ in range(n):
            f = nxt()
            if f is None:
                break
            out.append(f)
    finally:
        rm._kill_proc_tree(proc)
    return out, w, h


def boxes_of(det, frames, use_ctx: bool):
    res = []
    for f in frames:
        if use_ctx:
            ctx = det.prep(f)
            res.append(det.detect_ctx(ctx))
        else:
            res.append(det.detect(f))
    return res


def delta(ref, got):
    wb = wk = ws = 0.0
    n = 0
    mismatch = 0
    for a, b in zip(ref, got):
        if len(a) != len(b):
            mismatch += 1
            continue
        for fa, fb in zip(a, b):
            n += 1
            wb = max(wb, max(abs(x - y) for x, y in zip(fa["box"], fb["box"])))
            for pa, pb in zip(fa["kps"], fb["kps"]):
                wk = max(wk, max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])))
            ws = max(ws, abs(fa["score"] - fb["score"]))
    return {"faces": n, "count_mismatch_frames": mismatch,
            "max_box_px": round(wb, 4), "max_kps_px": round(wk, 4),
            "max_score": round(ws, 6)}


def scale_route(video: Path, frames, sel, det, sw, sh):
    """ffmpeg -vf scale 直接解小图 → 检测 → 坐标乘回去（B 的「省掉 resize」那条路）。"""
    ff = str(A._ffmpeg_exe())
    n = max(sel) + 1
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin",
           "-i", str(video), "-map", "0:v:0", "-an",
           "-vf", "scale=%d:%d:flags=bilinear" % (sw, sh), "-vsync", "0",
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=getattr(A, "_NO_WINDOW", 0))
    small = []
    nb = sw * sh * 3
    try:
        while True:
            buf = proc.stdout.read(nb)
            if not buf or len(buf) < nb:
                break
            small.append(np.frombuffer(buf, np.uint8).reshape(sh, sw, 3).copy())
    finally:
        rm._kill_proc_tree(proc)
    H = frames[0].shape[0]
    W = frames[0].shape[1]
    sx, sy = W / float(sw), H / float(sh)
    got = []
    pix = []
    for f, i in zip(frames, sel):
        if i >= len(small):
            continue
        bs = det.detect(small[i])
        got.append([{"box": [b["box"][0] * sx, b["box"][1] * sy,
                             b["box"][2] * sx, b["box"][3] * sy],
                     "kps": [[p[0] * sx, p[1] * sy] for p in b["kps"]],
                     "score": b["score"]} for b in bs])
        ctx = det.prep(f)
        nh, nw = small[i].shape[:2]
        ref = ctx["canvas"][:nh, :nw]
        d = np.abs(ref.astype(np.int16) - small[i].astype(np.int16))
        pix.append((float(d.mean()), int(d.max())))
    return got, {"mean_lsb": round(statistics.mean(p[0] for p in pix), 3),
                 "max_lsb": max(p[1] for p in pix)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO / "workspace" / "9月29日 (5).mp4"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--scan", type=int, default=200, help="最多扫多少帧找有脸的帧")
    ap.add_argument("--scale-route", action="store_true", help="重量一遍 -vf scale 那条路（慢）")
    a = ap.parse_args()
    video = Path(a.video)
    print("视频：%s" % video)
    frames, W, H = grab(video, max(a.frames, a.scan))
    tiny = rm._detector("tinymog")
    tiny.thresh = 0.5
    sel = [i for i, f in enumerate(frames) if tiny.detect(f)][:a.frames]
    if not sel:
        print("[FAIL] 前 %d 帧一张脸都没有，换个 --video / 调 --scan" % len(frames))
        return 2
    sub = [frames[i] for i in sel]
    print("有脸帧 %d 张：%s（%dx%d）" % (len(sub), sel, W, H))

    bad = 0
    report = {}
    for name in DETECTORS:
        det = rm._detector(name)
        det.thresh = 0.5
        ref = boxes_of(det, sub, False)
        got = boxes_of(det, sub, True)
        d = delta(ref, got)
        report[name] = d
        ok = (d["count_mismatch_frames"] == 0 and d["max_box_px"] == 0.0
              and d["max_kps_px"] == 0.0 and d["max_score"] == 0.0)
        print("%-15s detect() vs prep()+detect_ctx()：%s  %s"
              % (name, "一致" if ok else "不一致", json.dumps(d, ensure_ascii=False)))
        bad += 0 if ok else 1
        # canvas 只有一份实现：再算一遍必须逐位相同
        worst = 0
        for f in sub:
            c1 = det.prep(f)["canvas"]
            c2 = det.prep(f)["canvas"]
            worst = max(worst, int(np.abs(c1.astype(np.int16) - c2.astype(np.int16)).max()))
        print("%-15s canvas 复算 max|diff| = %d" % (name, worst))
        bad += 0 if worst == 0 else 1

        if a.scale_route:
            sw = int(round(W * 640.0 / max(W, H)))
            sh = int(round(H * 640.0 / max(W, H)))
            got2, pix = scale_route(video, sub, sel, det, sw, sh)
            d2 = delta(ref, got2)
            print("%-15s -vf scale=%d:%d：%s 像素 %s"
                  % (name, sw, sh, json.dumps(d2, ensure_ascii=False),
                     json.dumps(pix, ensure_ascii=False)))
            report[name + "|scale"] = dict(d2, **pix)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
