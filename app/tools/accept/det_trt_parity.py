# -*- coding: utf-8 -*-
r"""检测器 EP 对拍：同一个 retinaface_10g，ORT-CUDA fp32  vs  workspace 里的 TRT bf16。

为什么要有这个脚本：detectors.py 曾经**不传 workspace/engine_name**，于是 23MB 的 bf16
引擎躺在 workspace/_engines 里没人用，生产一直跑 ORT fp32（8.86ms vs 3.81ms）。这种
「配置看起来生效、实际没接上」的事故只能靠断言兜住，不能靠日志肉眼。

它查三件事：
  1) 接线：DFL_DET_TRT 不为 0 时，_get_detector("retinaface_10g") 必须拿到 TrtSession；
     为 0 时必须拿到 onnxruntime。拿不到 → 退出码 2。
  2) 精度：同帧同权重，boxes / 5 点最大位移（px）分布；超过 --max-px（默认 0.3）→ 退出码 2。
  3) 数量：NMS 后的框数是否一致。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\det_trt_parity.py --frames 100
"""
from __future__ import annotations

import argparse, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--max-px", type=float, default=0.3)
    a = ap.parse_args()

    import cv2
    from app.server._common.paths import WORKSPACE
    from app.server import _detect as dk
    from app.server.backend_extract import detectors as det_mod

    W = REPO / "weights"
    p = W / "retinaface_10g.onnx"

    # ---- 1) 接线 ----
    bad = 0
    for flag, want in (("1", "TrtSession"), ("0", "InferenceSession")):
        import os
        os.environ["DFL_DET_TRT"] = flag
        det_mod._det_cache.clear()
        got = type(det_mod._get_detector("retinaface_10g").sess).__name__
        ok = (got == want)
        print("[接线] DFL_DET_TRT=%s → %-16s（要 %s）%s" % (flag, got, want, "OK" if ok else "**不一致**"))
        bad += 0 if ok else 1
    det_mod._det_cache.clear()
    if bad:
        print("退出码 2：检测器 EP 接线不对")
        return 2

    # ---- 2/3) 同一帧对拍 ----
    ort = dk.RetinaFaceDet(p, 640)
    trt = dk.RetinaFaceDet(p, 640, workspace=str(WORKSPACE), engine_name="retinaface_10g")
    print("[对拍] ORT=%s / TRT=%s(%s)"
          % (type(ort.sess).__name__, type(trt.sess).__name__, getattr(trt.sess, "precision", "-")))

    import shutil
    exe = shutil.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    vf = "select='between(n,%d,%d)'" % (a.start, a.start + a.frames - 1)
    import subprocess
    proc = subprocess.Popen([exe, "-v", "error", "-i", str(WORKSPACE / "9月29日 (5).mp4"),
                             "-vf", vf, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    FSZ = 1080 * 1920 * 3
    d5, db, dcount, same = [], [], 0, 0
    n = 0
    while True:
        b = proc.stdout.read(FSZ)
        if len(b) < FSZ:
            break
        f = np.frombuffer(b, np.uint8).reshape(1920, 1080, 3)
        ort.thresh = trt.thresh = 0.5
        ra, rb = ort.detect(f), trt.detect(f)
        dcount += 0 if len(ra) == len(rb) else 1
        n += 1
        if ra and rb:
            ka = np.asarray(ra[0]["kps"], np.float64)
            kb = np.asarray(rb[0]["kps"], np.float64)
            if np.array_equal(ka, kb) and ra[0]["box"] == rb[0]["box"]:
                same += 1
            d5.append(float(np.linalg.norm(ka - kb, axis=1).max()))
            db.append(float(max(abs(x - y) for x, y in zip(ra[0]["box"], rb[0]["box"]))))
    proc.stdout.close()
    proc.kill()

    def q(v, p_):
        s = sorted(v)
        return s[min(len(s) - 1, max(0, int(round(p_ * (len(s) - 1)))))] if s else float("nan")

    print("[对拍] 帧 %d · 两臂都检出 %d · 逐位相同 %d · 框数不同 %d 帧"
          % (n, len(d5), same, dcount))
    if d5:
        print("[对拍] 5 点最大位移：中位 %.3f · p95 %.3f · max %.3f px"
              % (statistics.median(d5), q(d5, .95), max(d5)))
        print("[对拍] box 坐标最大差：中位 %.3f · max %.3f px"
              % (statistics.median(db), max(db)))
    fail = (max(d5) > a.max_px) if d5 else True
    if fail:
        print("退出码 2：5 点位移超过 %.2f px —— 换 EP 会改检测结果，要不要接受由人定" % a.max_px)
        return 2
    if dcount:
        print("退出码 2：有 %d 帧 NMS 后框数不一致" % dcount)
        return 2
    print("OK：接线对、框数一致、5 点位移 ≤ %.2f px" % a.max_px)
    return 0


if __name__ == "__main__":
    sys.exit(main())
