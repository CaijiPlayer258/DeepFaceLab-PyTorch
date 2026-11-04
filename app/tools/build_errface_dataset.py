# -*- coding: utf-8 -*-
"""把一堆人脸图片整理成 errface 训练用的 68 点数据集（.npz）。

两种输入都吃：
  1) DFL 对齐图（带 APP15 元数据）→ 直接读 landmarks，按 512/图高 缩放
  2) 普通人脸图片（任意来源、任意构图）→ TinyMog 检测 + 对齐 + 2D106 → 68 点

身份键（用于按人划分训练/测试，避免同一人两边都有）：
  优先用子目录名；没有子目录就用文件名第一个 '_' 之前的部分。

用法：
    cd app
    python -m tools.build_errface_dataset --src D:/faces --out ../workspace/errface_ds.npz
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def _identity(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="图片根目录（可含子目录，一个子目录=一个人）")
    ap.add_argument("--out", default="../workspace/errface_ds.npz")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张（0=全部）")
    ap.add_argument("--batch", type=int, default=2000, help="每多少张落一次盘")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from server._align import read_dfljpg_meta
    from server._errface import _img_h

    root = Path(args.src)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(str(root / "**" / ext), recursive=True))
    files = sorted(files)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("没找到图片:", root)
        return 2
    print("图片 %d 张，开始读关键点…" % len(files))

    det = lm106 = None
    X, idents, paths = [], [], []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        if not X:
            return
        np.savez_compressed(out, X=np.asarray(X, np.float32),
                            ident=np.asarray(idents), path=np.asarray(paths))
        print("  已写盘 %d 条 → %s" % (len(X), out))

    for i, f in enumerate(files):
        p = Path(f)
        pts = None
        try:
            m = read_dfljpg_meta(p)
            if isinstance(m, dict) and m.get("landmarks") is not None:
                h = _img_h(p)
                if h > 0:
                    pts = np.asarray(m["landmarks"], np.float64) * (512.0 / h)
        except Exception:
            pts = None

        if pts is None:
            if det is None:
                from server._detect import TinyMogDet
                from server._landmarks106 import InsightFace106
                from server._ortcuda import _candidate_roots
                det = TinyMogDet("modelhub/onnx/TinyMog/TinyMog.onnx")
                for r in _candidate_roots():
                    cand = Path(r) / "modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx"
                    if cand.is_file():
                        lm106 = InsightFace106(str(cand))
                        break
            import cv2
            img = cv2.imread(str(p))
            if img is not None:
                faces = det.detect(img)
                if faces:
                    fc = max(faces, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
                    box = fc["box"]
                    side = int(round(max(box[2] - box[0], box[3] - box[1]) * 2.0))
                    side -= side % 2
                    from server._align import align_face
                    crop, _meta = align_face(img, fc["kps"], box=box, size=side)
                    if crop is not None and lm106 is not None:
                        l68 = lm106.landmarks68(crop)
                        if l68 is not None:
                            pts = np.asarray(l68, np.float64) * (512.0 / crop.shape[0])

        if pts is None or pts.shape != (68, 2):
            continue
        X.append(pts.reshape(-1))
        idents.append(_identity(p, root))
        paths.append(str(p))
        if len(X) % args.batch == 0:
            flush()
            print("  进度 %d/%d" % (i + 1, len(files)))
    flush()

    idents = np.asarray(idents)
    print("完成：%d 条，%d 个身份，特征 136 维" % (len(X), len(set(idents.tolist()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
