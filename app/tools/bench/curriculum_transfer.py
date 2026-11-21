# -*- coding: utf-8 -*-
"""分辨率课程的档间迁移：权重 + 优化器状态。

encoder / decoder 是卷积，形状与分辨率无关（DeepFakeArchi 的 downscale 层数是常数 4/5），
所以跨档时可以逐字节复制；只有 Inter 的三个张量随 lowest_dense_res 变化。

优化器状态同理：
  · 形状一致的数组  → 原样复制（占 99%）
  · 形状变化的数组  → 用与权重相同的算子重采样（--moment resample）或置零（--moment zero）

用法：
  python curriculum_transfer.py --src model_128 --dst model_192 \
      --src-res 128 --dst-res 192 --archi-opts udt --moment resample
两个模型目录都必须已经存在（dst 由训练器新建过一次，哪怕只跑了 1 次迭代）。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("D:/MyProject/DeepFaceLab-PyTorch-4.1")
sys.path.insert(0, str(REPO / "app" / "engine"))
sys.path.insert(0, str(REPO / "app" / "engine" / "models" / "Model_SAEHD"))

import resize_inter as ri   # noqa: E402


def copy_verbatim(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def transfer_opt(src: Path, dst: Path, old_hw: int, new_hw: int, mode: str, method: str) -> tuple[int, int]:
    """按索引对齐地迁移优化器状态。返回 (复制的数组数, 特殊处理数)。"""
    a = ri.load_saveable(src)
    b = ri.load_saveable(dst)
    same = 0
    fixed = 0
    for key, target in b.items():
        arr = a.get(key)
        if arr is None or arr.shape == target.shape:
            same += 1
            continue
        if mode == "zero":
            b[key] = np.zeros_like(target)
        else:
            # 三张 Inter 张量的矩：沿空间维重采样，其余维度保持不变
            out = np.array(target, copy=True)
            if arr.ndim == 2 and arr.shape[0] == target.shape[1] and arr.shape[1] == target.shape[0]:
                out = ri.interpolate_dense_weight(arr.T, old_hw, new_hw, method=method).T.astype(target.dtype)
            elif arr.ndim == 2:
                out = ri.interpolate_dense_weight(arr, old_hw, new_hw, method=method).astype(target.dtype)
                if out.shape != target.shape:
                    out = ri.interpolate_dense_weight(arr.T, old_hw, new_hw, method=method).T.astype(target.dtype)
            elif arr.ndim == 1:
                out = ri.interpolate_bias(arr, old_hw, new_hw, method=method).astype(target.dtype)
            if out.shape != target.shape:
                out = np.zeros_like(target)
            b[key] = out
        fixed += 1
    ri.save_saveable(dst, b)
    return same, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源模型目录")
    ap.add_argument("--dst", required=True, help="目标模型目录（已由训练器创建）")
    ap.add_argument("--src-res", type=int, required=True)
    ap.add_argument("--dst-res", type=int, required=True)
    ap.add_argument("--archi-opts", default="udt")
    ap.add_argument("--method", default="blend")
    ap.add_argument("--moment", default="resample", choices=("resample", "zero"))
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    old_low = ri._inter_grid(args.src_res, args.archi_opts)
    new_low = ri._inter_grid(args.dst_res, args.archi_opts)
    print(f"grid {old_low} -> {new_low}")

    copied, resized = 0, 0
    for p in sorted(src.glob("*")):
        name = p.name
        if p.suffix == ".pth" and ("decoder" in name or "encoder" in name):
            copy_verbatim(p, dst / name)
            copied += 1
        elif p.suffix == ".pth" and "inter" in name:
            tgt = dst / name
            subprocess.run([str(REPO / "runtime" / "engine" / "python.exe"),
                            str(REPO / "app" / "engine" / "models" / "Model_SAEHD" / "resize_inter.py"),
                            "--model-dir", str(dst),
                            "--new-res", str(args.dst_res),
                            "--method", args.method], check=True)
            resized += 1
    print(f"权重：复制 {copied}，重采样 {resized}")

    for p in sorted(src.glob("*_opt.pth")):
        same, fixed = transfer_opt(p, dst / p.name, old_low, new_low, args.moment, args.method)
        print(f"优化器 {p.name}: 原样 {same}，处理 {fixed}")


if __name__ == "__main__":
    main()
