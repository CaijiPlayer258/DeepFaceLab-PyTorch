# -*- coding: utf-8 -*-
"""分辨率课程训练驱动（实验用）。

流程：
  1) 把对齐人脸按各档分辨率重采样成 data_src_R / data_dst_R
  2) 第 i 档：新建模型（resolution=R_i）→ 用 resize_inter 把上一档的 Inter 权重搬过来
  3) 训练 N_i 次迭代（调用引擎，按日志里的 iter 计数到点停止）
  4) 结束后用同一份评测（掩码内 PSNR/SSIM）比较"课程"与"直接从目标分辨率开始"

本脚本只做 1) 与 2) 的机械部分，训练由 run_stage() 调外部进程完成。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

REPO = Path("D:/MyProject/DeepFaceLab-PyTorch-4.1")
PY = REPO / "runtime" / "engine" / "python.exe"


def resample_dataset(src_dir: Path, dst_dir: Path, res: int) -> int:
    """把 aligned 目录里的图统一缩放到 res×res。返回张数。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if img.shape[0] != res or img.shape[1] != res:
            img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA if img.shape[0] > res else cv2.INTER_CUBIC)
        cv2.imwrite(str(dst_dir / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    return n


def run_stage(model_dir: Path, model_name: str, src: Path, dst: Path, iters: int, res: int, log: Path) -> None:
    cmd = [str(PY), "app/engine/main.py", "train",
           "--model-dir", str(model_dir), "--model", "SAEHD",
           "--training-data-src-dir", str(src), "--training-data-dst-dir", str(dst),
           "--no-preview", "--silent-start", "--force-model-name", model_name]
    print(" ".join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        import re
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            m = re.search(r"iter:\s*(\d+)", line)
            if m and int(m.group(1)) >= iters:
                proc.terminate()
                break
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="128:800,256:500,416:300", help="阶段:迭代数")
    ap.add_argument("--src", default="workspace/data_src/aligned")
    ap.add_argument("--dst", default="workspace/data_src/aligned")
    ap.add_argument("--work", default="workspace/_curriculum")
    args = ap.parse_args()

    work = REPO / args.work
    work.mkdir(parents=True, exist_ok=True)
    stages = [tuple((int(x) if i == 0 else int(x)) for i, x in enumerate(s.split(":"))) for s in args.plan.split(",")]
    print("plan:", stages)

    for res, iters in stages:
        d_src = work / f"data_src_{res}"
        d_dst = work / f"data_dst_{res}"
        n = resample_dataset(REPO / args.src, d_src, res)
        m = resample_dataset(REPO / args.dst, d_dst, res)
        print(f"res={res} 数据集 {n}/{m} 张", flush=True)
        run_stage(work / f"model_{res}", f"cur{res}_SAEHD", d_src, d_dst, iters, res, work / f"log_{res}.txt")


if __name__ == "__main__":
    main()
