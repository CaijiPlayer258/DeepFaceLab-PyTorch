# -*- coding: utf-8 -*-
r"""跑一趟模型：只装这一个 dfm，把它名下的脸跑成原料 chunk（docs/37 第 2 步）。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\pass_swap.py ^
        --video workspace\x.mp4 --model 张元英 --dfm workspace\model\张元英.dfm --limit 40

index.json 由 pass0_scan.py 产出（--start/--end 需与它一致）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "app"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dfm", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--chunk-frames", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from server.backend_merge.pass_swap import run

    v = Path(a.video)
    if not v.is_file():
        v = REPO / a.video
    dfm = Path(a.dfm)
    if not dfm.is_file():
        dfm = REPO / a.dfm
    index = Path(a.index) if a.index else (v.parent / (v.stem + "_cache") / "pool" / "index.json")
    if not index.is_file():
        print("缺 index.json（先跑 pass0_scan.py）：%s" % index)
        return 2
    meta = run(v, index, a.model, dfm, chunk_frames=a.chunk_frames,
               limit=(a.limit or None), log=print)
    print("产物：", meta.get("path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
