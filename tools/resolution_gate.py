# -*- coding: utf-8 -*-
"""resolution_gate —— 分辨率预筛门（blur 之前跑）

用法：
  python tools/resolution_gate.py --input <目录> [--min 256] [--move] [--recursive]

行为：
  扫描目录内全部图片的短边分辨率：
  - >= min  → 保留（不动）
  - <  min  → 默认列出统计；--move 时移到 <input>/filtered_by_res/ 子夹
产出统计报告（各档位数量），供 blur 分档前决策。

背景：深变 Extractor 动态尺寸模式无最小脸门槛（老 track 有 256 门槛），
IG 图库会产出远景/合影小脸。blur --train-res 归一会把这些小脸拉到同
口径（间接惩罚），但在 blur 之前先物理隔离更干净——避免污染 blur 的
分位基线（大量小脸会拉低数据集中位数，让阈值偏松）。
"""
import argparse
import os
import shutil
import cv2
import numpy as np

IMG = (".jpg", ".jpeg", ".png", ".webp")


def main():
    ap = argparse.ArgumentParser(description="分辨率预筛门（blur 前置）")
    ap.add_argument("--input", required=True, help="待筛目录")
    ap.add_argument("--min", type=int, default=256, help="短边最小阈值（默认 256）")
    ap.add_argument("--move", action="store_true", help="把 <min 的移到 filtered_by_res/ 子夹（默认只统计）")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    args = ap.parse_args()

    root = args.input
    buckets = {"<128": 0, "128-255": 0, "256-399": 0, "400-511": 0, "512+": 0}
    below = []
    files = []
    for root_dir, dirs, fs in os.walk(root):
        rel = os.path.relpath(root_dir, root)
        if not args.recursive and rel != ".":
            dirs.clear()
            continue
        for f in fs:
            if f.lower().endswith(IMG):
                files.append(os.path.join(root_dir, f))
        if not args.recursive:
            break

    print(f"[res_gate] 扫描 {len(files)} 张图（min={args.min}）")
    for p in files:
        buf = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            buckets["<128"] += 1  # 解码失败的按最小档统计
            below.append(p)
            continue
        short = min(img.shape[:2])
        if short < 128:
            k = "<128"
        elif short < 256:
            k = "128-255"
        elif short < 400:
            k = "256-399"
        elif short < 512:
            k = "400-511"
        else:
            k = "512+"
        buckets[k] += 1
        if short < args.min:
            below.append(p)

    for k, v in buckets.items():
        print(f"  {k}: {v}")

    print(f"\n[res_gate] <{args.min}px 的共 {len(below)} 张")
    if args.move and below:
        move_dir = os.path.join(root, "filtered_by_res")
        os.makedirs(move_dir, exist_ok=True)
        n = 0
        for p in below:
            dst = os.path.join(move_dir, os.path.basename(p))
            if os.path.exists(dst):
                stem, ext = os.path.splitext(os.path.basename(p))
                dst = os.path.join(move_dir, f"{stem}_{n}{ext}")
            shutil.move(p, dst)
            n += 1
        print(f"[res_gate] 已移 {n} 张 -> {move_dir}")
    elif below:
        print(f"[res_gate] 加 --move 执行移动")


if __name__ == "__main__":
    main()
