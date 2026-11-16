# -*- coding: utf-8 -*-
r"""检测 EP 对拍：同一个检测器，GPU 那份 vs CPU 那份，**逐框**比。

为什么单独一个工具：DFL_DET_EP 把检测从 GPU 挪到 CPU 之后，「像素不变」这件事只能靠
「检测结果逐框一致」来保证 —— 检测出的是 5 点关键点，它决定对齐矩阵，矩阵决定 dfm 的
输入，输入决定合成像素。框错了 0.5px，成片就换了一张脸。所以这里分开量三层：

  ① 原始输出（cls/bbox/kps 三组 × 三个 stride，**不做阈值过滤**，按 anchor 索引对齐）
     → 这是模型本身的数值差，bf16/TRT 与 fp32/CPU 的差别只在这里出现；
  ② 解码后的框与关键点（canvas 坐标 → 原图坐标，单位 px）
     → 真正喂给对齐的东西；这里的 Δ 才是「换 EP 会不会换脸」的判据（阈值 0.5px）；
  ③ 最终人脸列表（detect() 的输出，box/kps 已 round(0.1)、score round(4)）
     → 逐框一致 = 这个列表逐元素相同（数量、顺序、每个数）。

跑法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\accept\det_ep_parity.py
    ... --det tinymog --frames 30 --clip <某段 mp4>
退出码：0 = 框一致（≤0.5px 且人脸数相同）；2 = 不一致/量不出来。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
os.environ.setdefault("DFL_WORKSPACE", str(REPO / "workspace"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np                                             # noqa: E402

TOL_BOX_PX = 0.5          # 用户口径：超过 0.5px 就说明不能换


def _clip(path=None, frames=30):
    """取一段真片子：默认从 workspace 里第一支 mp4 的第 3 秒切 frames 帧。"""
    import cv2
    if path:
        return Path(path)
    src = sorted((REPO / "workspace").glob("*.mp4"))
    if not src:
        raise SystemExit("workspace 里没有 mp4，给 --clip")
    out = Path(os.environ.get("TEMP") or ".") / "dfl_det_ep_parity" / "clip.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file():
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-ss", "3",
                            "-t", "%.2f" % (frames / 30.0 + 0.2), "-i", str(src[0]),
                            "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("ffmpeg 切样片失败：%s" % (r.stderr or "")[-300:])
    return out


def _frames(path, n):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while len(out) < n:
            ok, img = cap.read()
            if not ok or img is None:
                break
            out.append(img)
    finally:
        cap.release()
    return out


def _blob(canvas):
    """和 TinyMogDet._forward / forward_batch 完全同一套预处理（BGR→RGB、CHW、-127.5/128）。"""
    S = canvas.shape[0]
    blob = np.empty((1, 3, S, S), np.float32)
    blob[0] = canvas[:, :, ::-1].transpose(2, 0, 1)
    blob -= 127.5
    blob *= (1.0 / 128.0)
    return blob


def _decode_raw(outs, S, steps=(8, 16, 32), anchors=2):
    """原始输出 → (scores, boxes_px, kps_px)（canvas 坐标，**不过阈值**）。

    和 detect 里的解码逐位同一套公式（中心 ± 距离、kps + 中心、乘 stride），
    只是不做 scores>=thresh 的过滤 —— 这样两边数组形状一致，可以逐元素比。
    """
    sc, bx, kp = [], [], []
    for i, stride in enumerate(steps):
        h = w = S // stride
        c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        c = (c * stride).reshape(-1, 2)
        if anchors > 1:
            c = np.stack([c] * anchors, axis=1).reshape(-1, 2)
        scores = np.asarray(outs[i * 3]).reshape(-1)
        bb = np.asarray(outs[i * 3 + 1]).reshape(-1, 4) * stride
        kk = np.asarray(outs[i * 3 + 2]).reshape(-1, 10) * stride
        n = min(len(scores), len(c), len(bb), len(kk))   # int8/静态引擎可能多给几行
        c, scores, bb, kk = c[:n], scores[:n], bb[:n], kk[:n]
        boxes = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1],
                          c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], axis=-1)
        kps = kk.reshape(-1, 5, 2)
        kps[:, :, 0] += c[:, 0][:, None]
        kps[:, :, 1] += c[:, 1][:, None]
        sc.append(scores)
        bx.append(boxes)
        kp.append(kps)
    return np.concatenate(sc), np.concatenate(bx, 0), np.concatenate(kp, 0)


def _flat_faces(faces):
    """[[box x4, kps x10, score]] —— 用来做逐元素比较。"""
    out = []
    for f in faces or []:
        row = [float(v) for v in (f.get("box") or [])]
        for p in (f.get("kps") or []):
            row.extend([float(p[0]), float(p[1])])
        row.append(float(f.get("score") or 0.0))
        out.append(row)
    return np.asarray(out, np.float64) if out else np.zeros((0, 17))


def main(argv=None):
    ap = argparse.ArgumentParser(description="检测 EP 对拍（GPU vs CPU）")
    ap.add_argument("--det", default="tinymog", help="检测器（tinymog / retinaface_10g）")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--clip", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--gpu-impl", default="auto", choices=("auto", "trt", "ort"),
                    help="GPU 那份用什么：auto=和 fused 一样（有 TRT 引擎就用引擎）；"
                         "ort=强制 onnxruntime CUDA fp32（用来分开「EP 带来的差」与"
                         "「bf16 引擎带来的差」）")
    args = ap.parse_args(argv)

    from server import app as _A                                 # noqa: F401,E402  先 import 后端（循环依赖）
    from server import routes_merge as rm                        # noqa: E402
    from server import _detect as dk                             # noqa: E402

    clip = _clip(args.clip or None, max(args.frames, 30))
    frames = _frames(clip, args.frames)
    print("样片 %s · 取到 %d 帧 · 检测器 %s" % (clip, len(frames), args.det), flush=True)

    if args.gpu_impl == "ort":
        # 同一个 ONNX、同一个 EP、只差精度：用来回答「0.2px 是 EP 的锅还是 bf16 引擎的锅」
        from server._detect import TinyMogDet, RetinaFaceDet
        from server.app import _DET_ONNX, _detector_path
        from server._ortcuda import providers as _prov
        p = _detector_path(_DET_ONNX[rm._detector_val(args.det)])
        if rm._detector_val(args.det) == "tinymog":
            gpu = TinyMogDet(str(p), workspace=None, providers=list(_prov()))
        else:
            gpu = RetinaFaceDet(p, 640, providers=list(_prov()))
    else:
        gpu = rm._detect_for(args.det, "gpu")
    cpu = rm._detect_for(args.det, "cpu")
    if gpu is None or cpu is None:
        raise SystemExit("有 EP 起不来：gpu=%r cpu=%r" % (gpu, cpu))
    dk_ep = {gpu: "gpu", cpu: "cpu"}
    print("GPU 会话：%s · CPU 会话：%s（intra_op=%d）"
          % (type(gpu.sess).__name__, type(cpu.sess).__name__, dk.threads()), flush=True)

    res = {"det": args.det, "frames": len(frames), "gpu_sess": type(gpu.sess).__name__,
           "cpu_sess": type(cpu.sess).__name__, "cpu_threads": dk.threads(),
           "raw": [], "faces": [], "per_frame": []}
    worst_raw = 0.0
    worst_box_px = 0.0
    worst_kps_px = 0.0
    worst_face_max = 0.0
    n_diff_faces = 0
    n_cnt_diff = 0
    n_frames = 0
    for i, img in enumerate(frames):
        ctx = gpu.prep(img)                      # prep 与 EP 无关（同一个函数、同一个尺寸）
        ctxc = cpu.prep(img)
        if not np.array_equal(ctx["canvas"], ctxc["canvas"]):
            raise SystemExit("两个检测器的 canvas 不一致 —— prep 都不一致，后面的对比没有意义")
        S = int(ctx["canvas"].shape[0])
        blob = _blob(ctx["canvas"])
        og = gpu.sess.run(None, {gpu.input_name: blob})
        oc = cpu.sess.run(None, {cpu.input_name: blob})
        raw = []
        for k in range(min(len(og), len(oc))):
            a = np.asarray(og[k], np.float64).reshape(-1)
            b = np.asarray(oc[k], np.float64).reshape(-1)
            if a.shape != b.shape:
                raw.append({"i": k, "shape": [list(a.shape), list(b.shape)]})
                continue
            raw.append({"i": k, "max": float(np.max(np.abs(a - b))) if a.size else 0.0})
        sg, bg, kg = _decode_raw(og, S)
        sc, bc, kc = _decode_raw(oc, S)
        if not (sg.shape == sc.shape and bg.shape == bc.shape and kg.shape == kc.shape):
            raise SystemExit("第 %d 帧解码形状不一致：%s vs %s"
                             % (i + 1, [sg.shape, bg.shape, kg.shape],
                                [sc.shape, bc.shape, kc.shape]))
        dbox = float(np.max(np.abs(bg - bc))) if bg.size else 0.0
        dkps = float(np.max(np.abs(kg - kc))) if kg.size else 0.0
        raw_max = max([float(r.get("max") or 0.0) for r in raw] or [0.0])
        # detect() 那份（round(0.1) 之后的人脸列表）
        fg = _flat_faces(gpu.detect(img))
        fc = _flat_faces(cpu.detect(img))
        same = (fg.shape == fc.shape) and bool(np.array_equal(fg, fc))
        cnt_same = (fg.shape[0] == fc.shape[0])
        fmax = 0.0
        if not same:
            n_diff_faces += 1
            if fg.shape == fc.shape and fg.size:
                fmax = float(np.max(np.abs(fg - fc)))
        if not cnt_same:
            n_cnt_diff += 1
        res["per_frame"].append({"i": i + 1, "raw_max": raw_max,
                                 "box_px": round(dbox, 4), "kps_px": round(dkps, 4),
                                 "faces_gpu": int(fg.shape[0]), "faces_cpu": int(fc.shape[0]),
                                 "same": bool(same), "face_max": round(fmax, 4),
                                 "count_same": bool(cnt_same)})
        res["raw"].append(raw)
        worst_raw = max(worst_raw, raw_max)
        worst_box_px = max(worst_box_px, dbox)
        worst_kps_px = max(worst_kps_px, dkps)
        worst_face_max = max(worst_face_max, fmax)
        n_frames += 1
        if (i + 1) % 10 == 0:
            print("  第 %d 帧：raw max=%.5f · box Δ=%.4f px · kps Δ=%.4f px · 人脸 %d/%d %s"
                  % (i + 1, raw_max, dbox, dkps, fg.shape[0], fc.shape[0],
                     "一致" if same else "不同"), flush=True)

    res.update({"raw_max": worst_raw, "box_px_max": worst_box_px, "kps_px_max": worst_kps_px,
                "frames_diff_faces": n_diff_faces, "frames_diff_count": n_cnt_diff,
                "tol_box_px": TOL_BOX_PX})
    print("")
    print("=" * 78)
    print("检测 EP 对拍：%s · %d 帧" % (args.det, n_frames))
    print("  ① 原始输出 max|Δ|        ：%.6f" % worst_raw)
    print("  ② 解码框 max|Δ|          ：%.4f px（判据 ≤ %.1f）" % (worst_box_px, TOL_BOX_PX))
    print("     解码 kps max|Δ|       ：%.4f px" % worst_kps_px)
    print("  ③ detect() 人脸数         ：%s（%d/%d 帧数量不同）"
          % ("每帧都对得上" if n_cnt_diff == 0 else "有帧丢/多了脸", n_cnt_diff, n_frames))
    print("     逐元素（round 0.1 之后）：%d/%d 帧不同 · 最大 %.4f —— 框本身就差 %.2f px，"
          "round 到 0.1 必然翻出不同字面值；判定按 ② 的 %.1f px 口径走"
          % (n_diff_faces, n_frames, worst_face_max, worst_box_px, TOL_BOX_PX))
    ok = (worst_box_px <= TOL_BOX_PX) and (n_cnt_diff == 0)
    print("  结论：%s" % ("框差 ≤ 0.5px 且脸数一致 → 换 EP 不会换脸（但做不到逐位相同）"
                        if ok else "框差 > 0.5px 或有帧丢脸 → 这个检测器不能换 EP"))
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        print("  JSON：%s" % args.json)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
