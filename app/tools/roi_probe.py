# -*- coding: utf-8 -*-
r"""Full-frame detection vs ROI detection -- data only, nothing is wired into any flow.

Usage (from repo root):
  runtime\unified\python.exe app/tools/roi_probe.py --video path/to/video.mp4
      [--frames 60] [--start 0] [--scale 2.5] [--pad 0] [--json out.json]

For each frame it runs two detections:
  A) whole frame letterboxed to 640 (what happens today)
  B) a square ROI centred on the previous frame face, side = face_long_side * scale,
     out-of-frame area padded (default black), scaled to 640
Then it compares: number of detections, best score, IoU of the box after mapping back,
and how a small face (< 40 px) fares on each side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np                                    # noqa: E402


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ua + ub - inter
    return float(inter / den) if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import os
    if args.pad:
        os.environ["DFL_ROI_PAD"] = str(args.pad)

    import cv2
    import server.app                                    # noqa: F401
    from server._vision.detect import TinyMogDet
    from server._vision.roi import crop_square_padded, map_box_back, square_side

    det = TinyMogDet(str(ROOT / "weights" / "tinymog.onnx"), workspace=ROOT / "workspace")
    cap = cv2.VideoCapture(str(ROOT / args.video))
    if not cap.isOpened():
        print("cannot open: %s" % args.video)
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start))

    rows = []
    prev = None
    for k in range(int(args.frames)):
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        a = det.detect(frame, min_face=8)
        row = {"frame": int(args.start) + k, "A_n": len(a),
               "A_best": round(float(max((b["score"] for b in a), default=0.0)), 4)}
        if a:
            sz = [max(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) for b in a]
            row["A_face_px"] = round(float(max(sz)), 1)
        if prev is not None:
            bx = prev["box"]
            cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
            side = square_side(bx[2] - bx[0], bx[3] - bx[1], args.scale, frame_size=(W, H))
            canvas, meta = crop_square_padded(frame, cx, cy, side, 640, pad=args.pad)
            bb = det.detect(canvas, min_face=8)
            row["B_side"] = int(side)
            row["B_n"] = len(bb)
            row["B_best"] = round(float(max((x["score"] for x in bb), default=0.0)), 4)
            if bb:
                back = map_box_back(bb[0]["box"], meta)
                row["B_iou_vs_A"] = round(_iou(back, bx), 3)
                row["B_face_px"] = round(float(max(
                    bb[0]["box"][2] - bb[0]["box"][0],
                    bb[0]["box"][3] - bb[0]["box"][1])), 1)
        if a:
            prev = a[0]
        rows.append(row)
    cap.release()

    hdr = "%-8s %-5s %-8s %-8s %-6s %-8s %-10s %s"
    print(hdr % ("frame", "A_n", "A_best", "A_facepx", "B_n", "B_best",
                 "B_iou_vs_A", "B_facepx"))
    for r in rows:
        print("%-8d %-5d %-8.3f %-8s %-6s %-8s %-10s %s" % (
            r["frame"], r["A_n"], r["A_best"], r.get("A_face_px", "-"),
            r.get("B_n", "-"), r.get("B_best", "-"),
            r.get("B_iou_vs_A", "-"), r.get("B_face_px", "-")))

    det_a = sum(1 for r in rows if r["A_n"])
    det_b = sum(1 for r in rows if r.get("B_n"))
    ious = [r["B_iou_vs_A"] for r in rows if "B_iou_vs_A" in r]
    print("")
    print("frames sampled: %d" % len(rows))
    print("  A (whole frame) detected %d frames / B (ROI) detected %d frames" % (det_a, det_b))
    if ious:
        print("  B mapped-back IoU vs A: median %.3f / min %.3f"
              % (float(np.median(ious)), float(min(ious))))
    sa = [r["A_face_px"] for r in rows if r.get("A_face_px")]
    sb = [r["B_face_px"] for r in rows if r.get("B_face_px")]
    if sa:
        print("  face long side in source: median %.0f px" % float(np.median(sa)))
    if sb:
        print("  face long side inside ROI canvas: median %.0f px" % float(np.median(sb)))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("  detail written to %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())