# -*- coding: utf-8 -*-
r"""跑一次 Pass 0（人脸分配表），产物 <视频>_cache/pool/index.json。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\pass0_scan.py --video workspace\x.mp4 --model 张元英 \
        --model2 阿証証証証-416 --start 0 --end 300

--model / --model2 / --model3 … 依次分配给第 0/1/2… 张脸；不给就是「第一个模型管全部」。
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
    ap.add_argument("--detector", default="tinymog")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--model", action="append", default=[], help="模型名，按出现顺序给第 0/1/2… 张脸")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    from server.backend_merge.pass0 import scan

    v = Path(a.video)
    if not v.is_file():
        v = REPO / a.video
    models = []
    for i, nm in enumerate(a.model):
        dfm = REPO / "workspace" / "model" / (nm if nm.endswith(".dfm") else nm + ".dfm")
        if not dfm.is_file():
            print("找不到模型：%s" % dfm)
            return 2
        models.append({"name": nm.replace(".dfm", ""), "dfm": str(dfm), "faces": [i]})
    if len(models) == 1:
        models[0]["faces"] = []            # 单模型 = 我管全部脸
    out = scan(v, detector=a.detector, thresh=a.thresh, models=models,
               stride=a.stride, start=a.start, end=a.end,
               out_dir=(a.out_dir or None), log=print)
    print("index.json:", out["swap_faces"], "/", out["detected"], "faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
