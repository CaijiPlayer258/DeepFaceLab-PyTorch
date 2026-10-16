# -*- coding: utf-8 -*-
"""ig_harvest —— 深变 IG 切脸 + csv 账本（锚定 IG 源文件）

用法：
  python tools/ig_harvest.py --src "G:\\ig\\liz.yeyo" --out "C:\\SRC\\from_ig\\liz.yeyo" [选项]

账本：<源目录>/.dfl/harvest.csv，每行 = 一张已切的源文件名（含扩展名）。
  - 以 IG 源文件为准：脸图移动/改名不影响账本（源不动）
  - 新增图：不在 csv 里即待切
  - 人工剔除脸图后：源图仍是"已切"状态（不重复切）；要重切单张，从 csv 删该行再跑
  - 源图被修改：不感知（无指纹）——需重切同上手动删行

流程：csv 查账 → 未切的图 %TEMP% ASCII 中转 → Extractor（默认 RetinaFace_10g +
2d106det，杂景通吃）→ 产物按源图原名落位（_N 多脸后缀）→ csv 追加 + jsonl 审计日志。
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "python", "python.exe")
EXTRACTOR = os.path.join(ROOT, "Extractor", "Extractor.py")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def ledger_path(src):
    return os.path.join(src, ".dfl", "harvest.csv")


def load_ledger(src):
    p = ledger_path(src)
    done = set()
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8-sig", newline="") as fp:
            for row in csv.reader(fp):
                if row and row[0].strip():
                    done.add(row[0].strip())
    return done


def append_ledger(src, names):
    p = ledger_path(src)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, "a", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["source_file"])          # 表头：源文件名（含扩展名）
        for n in names:
            w.writerow([n])


def main():
    ap = argparse.ArgumentParser(description="IG 切脸 + csv 账本（锚定源文件）")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-d", "--detector", default="RetinaFace_10g")
    ap.add_argument("-l", "--landmarker", default="insightface-2d106det")
    ap.add_argument("-t", "--face-type", default="whole_face")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="无视账本全量重切（账本照常追加去重）")
    args = ap.parse_args()

    src, out = args.src, args.out
    if not os.path.isdir(src):
        sys.exit(f"[ig_harvest] 源目录不存在: {src}")
    os.makedirs(out, exist_ok=True)

    all_imgs = [f for f in sorted(os.listdir(src)) if f.lower().endswith(IMG_EXTS)]
    done = load_ledger(src)
    todo = [f for f in all_imgs if args.force or f not in done]
    print(f"[ig_harvest] 源 {len(all_imgs)} 张 | 账本已切 {len(done)} | 本次待切 {len(todo)} -> {out}")
    if not todo:
        print("[ig_harvest] 无新图，完成")
        return 0

    # ASCII 中转（webp 转 png；中转名 h<i> 与源名映射）
    stage_in = os.path.join(os.environ["TEMP"], "ig_harvest_in")
    stage_out = os.path.join(os.environ["TEMP"], "ig_harvest_out")
    for d in (stage_in, stage_out):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    import cv2
    import numpy as np
    mapping = {}
    staged_list = []
    for i, f in enumerate(todo):
        p = os.path.join(src, f)
        stem = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1].lower()
        if ext == ".webp":
            buf = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [skip] 解码失败: {f}")
                continue
            tmp = os.path.join(stage_in, f"{i:05d}.png")
            cv2.imencode(".png", img)[1].tofile(tmp)
        else:
            tmp = os.path.join(stage_in, f"{i:05d}{ext}")
            shutil.copy2(p, tmp)
        mapping[f"{i:05d}"] = f
        staged_list.append(f)

    cmd = [PY, EXTRACTOR, "-i", stage_in, "-o", stage_out,
           "-d", args.detector, "-l", args.landmarker,
           "-t", args.face_type, "-m", "image"]
    if args.input_size:
        cmd += ["--input-size", str(args.input_size)]
    print(f"[ig_harvest] Extractor: {' '.join(cmd[2:])}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[ig_harvest] Extractor 失败 exit={r.returncode}")

    # 落位：Extractor 输出 <n>_N.jpg（或 png）——映射回源名，保多脸后缀
    n_out = 0
    no_face = []
    produced_src = set()
    for f in sorted(os.listdir(stage_out)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(f)[0]
        suffix = ""
        if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
            stem, suffix = stem.rsplit("_", 1)[0], "_" + stem.rsplit("_", 1)[-1]
        orig = mapping.get(stem)
        if orig is None:
            continue
        orig_stem = os.path.splitext(orig)[0]
        dst = os.path.join(out, orig_stem + suffix + ".png")
        srcp = os.path.join(stage_out, f)
        buf = np.fromfile(srcp, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imencode(".png", img)[1].tofile(dst)   # 统一转 png 落位（元数据已丢失，脸集后续走 Filter 不依赖）
        n_out += 1
        produced_src.add(orig)
    # 未检出脸的源图也记账（避免反复重切无脸图）
    no_face = [f for f in staged_list if f not in produced_src]
    if no_face:
        print(f"[ig_harvest] 未检出脸 {len(no_face)} 张（同样记账防重切）")

    shutil.rmtree(stage_in, ignore_errors=True)
    shutil.rmtree(stage_out, ignore_errors=True)

    append_ledger(src, staged_list)

    log = os.path.join(out, ".harvest_log.jsonl")
    with open(log, "a", encoding="utf-8") as fp:
        fp.write(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M"), "src": src,
            "staged": len(staged_list), "produced_faces": n_out,
            "no_face": len(no_face),
            "combo": f"{args.detector}+{args.landmarker}",
            "input_size": args.input_size,
        }, ensure_ascii=False) + "\n")

    print(f"[ig_harvest] 完成: {len(staged_list)} 源图 → {n_out} 脸 | 无脸 {len(no_face)} | 账本 {ledger_path(src)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
