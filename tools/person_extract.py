# -*- coding: utf-8 -*-
"""person_extract —— 人物夹初始化切脸（深变引擎版，px 前缀命名）

2026-10-17 用户指定：
  ① 人物夹自适应切脸不再用 SCRFD 分桶（adaptive-run 在人物夹场景退役），检测+切脸
     统一走深变 Extractor（默认 RetinaFace_10g + insightface-2d106det，动态尺寸直出），
     算法用 -d/-l/-t 随时换；ig_harvest（IG 源收集）不受影响。
  ② 脸图文件名以 px<短边4位>__ 开头（如 px0512__中文夹__照片b__a1b2c3d4e5_0.png），
     资源管理器按名称排序 = 按分辨率排序，方便按前缀筛。
     （老 adaptive-run / 帧图指定人脸 wrapper 已同步改为同款前缀；launcher organize
     反查已兼容剥前缀。旧 _px512 后缀命名的存量产物不受影响。）

用法：
  python tools\\person_extract.py --root "G:\\Idol\\Giselle" --out "C:\\SRC\\Giselle" --recursive
      [-d RetinaFace_10g] [-l insightface-2d106det] [-t whole_face]
      [--input-size N] [--allow-nonempty]

流程：枚举原图（跳过 cutted/filtered/video/隐藏夹）→ 扁平名（与 launcher
safe_flatten_name 完全一致，organize 母图反查依赖）→ %TEMP% ASCII 中转
（webp→png；中转名 p00001 纯 ASCII 防中文路径坑）→ Extractor 一把切 →
脸图按 px 前缀 + 扁平名落位 → 打印账目。
归位仍用老 launcher organize（母图反查）：
  F:\\dfl\\_internal\\Py310-CUDA\\python.exe F:\\dfl\\z11-AgentTools\\dfl_extract_launcher.py ^
      organize --input <root> --output <out> --recursive
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "python", "python.exe")
EXTRACTOR = os.path.join(ROOT, "Extractor", "Extractor.py")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")   # 与 launcher IMAGE_EXTENSIONS 对齐
SKIP_DIRS = {"cutted", "filtered", "filterd", "video", "__macosx"}


def flat_name(root, path):
    """复刻 dfl_extract_launcher.safe_flatten_name（organize 母图反查契约）。"""
    rel = os.path.relpath(path, root)
    rp = Path(rel)
    stem = "__".join(rp.with_suffix("").parts)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._")
    orig_stem = Path(path).stem
    if not stem:
        stem = orig_stem
    if stem == orig_stem:
        return os.path.basename(path)
    digest = hashlib.sha1(rel.encode("utf-8", errors="surrogatepass")).hexdigest()[:10]
    return f"{stem}__{digest}{os.path.splitext(path)[1].lower()}"


def iter_images(root, recursive):
    files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            )
            for fn in filenames:
                if fn.lower().endswith(IMG_EXTS) and not fn.lower().startswith("._"):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and fn.lower().endswith(IMG_EXTS) and not fn.lower().startswith("._"):
                files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="人物夹初始化切脸（深变引擎，px 前缀命名）")
    ap.add_argument("--root", required=True, help="人物素材夹（图片留在原位，之后用 launcher organize 归位）")
    ap.add_argument("--out", required=True, help="脸图输出目录（如 C:\\SRC\\<人物>）")
    ap.add_argument("--recursive", action="store_true", help="递归子夹（与 organize --recursive 成对）")
    ap.add_argument("-d", "--detector", default="RetinaFace_10g")
    ap.add_argument("-l", "--landmarker", default="insightface-2d106det")
    ap.add_argument("-t", "--face-type", default="whole_face")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--allow-nonempty", action="store_true", help="输出目录非空时允许继续")
    args = ap.parse_args()

    root, out = os.path.abspath(args.root), os.path.abspath(args.out)
    if not os.path.isdir(root):
        sys.exit(f"[person_extract] 目录不存在: {root}")
    try:
        if os.path.commonpath([root, out]).lower() == root.lower():
            sys.exit("[person_extract] --out 不能在 --root 里面（organize 会把脸图当原图）")
    except ValueError:
        pass  # 不同盘符
    os.makedirs(out, exist_ok=True)
    if os.listdir(out) and not args.allow_nonempty:
        sys.exit("[person_extract] 输出目录非空（防重复切）；确认要追加请加 --allow-nonempty")

    images = iter_images(root, args.recursive)
    print(f"[person_extract] 原图 {len(images)} 张 (root={root})")
    if not images:
        return 0

    import cv2
    import numpy as np

    stage_in = os.path.join(os.environ["TEMP"], "person_extract_in")
    stage_out = os.path.join(os.environ["TEMP"], "person_extract_out")
    for d in (stage_in, stage_out):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    mapping = {}   # p00001 -> (源路径, 扁平stem)
    skipped = []
    for i, p in enumerate(images):
        ext = os.path.splitext(p)[1].lower()
        staged = f"{i:05d}"
        if ext == ".webp":
            buf = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                skipped.append(os.path.basename(p))
                continue
            cv2.imencode(".png", img)[1].tofile(os.path.join(stage_in, staged + ".png"))
        else:
            shutil.copy2(p, os.path.join(stage_in, staged + ext))
        mapping[staged] = (p, os.path.splitext(flat_name(root, p))[0])

    cmd = [PY, EXTRACTOR, "-i", stage_in, "-o", stage_out,
           "-d", args.detector, "-l", args.landmarker,
           "-t", args.face_type, "-m", "image"]
    if args.input_size:
        cmd += ["--input-size", str(args.input_size)]
    print(f"[person_extract] Extractor: {' '.join(cmd[2:])}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[person_extract] Extractor 失败 exit={r.returncode}（中转目录保留: {stage_in}）")

    n_out = 0
    faces_by_src = {}
    for f in sorted(os.listdir(stage_out)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        suffix = ""
        if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
            stem, suffix = stem.rsplit("_", 1)[0], "_" + stem.rsplit("_", 1)[-1]
        if stem not in mapping:
            continue
        src, flat_stem = mapping[stem]
        img = cv2.imdecode(np.fromfile(os.path.join(stage_out, f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        px = min(img.shape[0], img.shape[1])
        dst = os.path.join(out, f"px{px:04d}__{flat_stem}{suffix}.png")
        cv2.imencode(".png", img)[1].tofile(dst)
        n_out += 1
        faces_by_src[src] = faces_by_src.get(src, 0) + 1

    no_face = [src for src, _ in mapping.values() if src not in faces_by_src]
    print(f"[person_extract] 送切 {len(mapping)} 张 | 解码失败 {len(skipped)} 张 | 产出脸图 {n_out} 张")
    print(f"[person_extract] 有脸母图 {len(faces_by_src)} | 无脸母图 {len(no_face)}（organize 会归 filtered）")
    for p in no_face[:10]:
        print(f"    无脸: {os.path.relpath(p, root)}")
    if len(no_face) > 10:
        print(f"    ... 其余 {len(no_face) - 10} 张略")
    print("[person_extract] 脸图命名 px<短边4位>__ 开头，资源管理器按名称排序即按分辨率分组")
    return 0


if __name__ == "__main__":
    main()
