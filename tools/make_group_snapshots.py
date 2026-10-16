# -*- coding: utf-8 -*-
import argparse, os, math
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--cell", type=int, default=160)
ap.add_argument("--cols", type=int, default=10)
ap.add_argument("--out-dir", default=None)
a = ap.parse_args()

work = a.work
out_dir = a.out_dir or os.path.join(work, "groups")
os.makedirs(out_dir, exist_ok=True)

groups = sorted(d for d in os.listdir(work) if d.startswith("faceid_group_") or d == "faceid_noise")
for g in groups:
    gd = os.path.join(work, g)
    files = sorted(f for f in os.listdir(gd) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        continue
    cols = a.cols
    rows = math.ceil(len(files) / cols)
    sheet = np.full((rows * a.cell, cols * a.cell, 3), 18, np.uint8)
    for i, f in enumerate(files):
        buf = np.fromfile(os.path.join(gd, f), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        t = cv2.resize(img, (a.cell, a.cell))
        r, c = divmod(i, cols)
        sheet[r*a.cell:r*a.cell+a.cell, c*a.cell:(c+1)*a.cell] = t
    out = os.path.join(out_dir, g + ".png")
    ok, buf = cv2.imencode(".png", sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if ok:
        buf.tofile(out)
    print(out, "  (", len(files), "faces )")

print("[done]")
