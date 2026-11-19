# -*- coding: utf-8 -*-
r"""一条命令跑完「分趟导出」：Pass0 → 每个模型一趟 → 合流 → 成片（docs/37）。

用法（仓库根）：
    runtime\server\Scripts\python.exe app\tools\bench\passes_export.py ^
        --video workspace\x.mp4 --model 张元英 --model 阿証証証証-416 ^
        --start 0 --end 300 --out workspace\x_passes.mp4

注意：这里用的是 OpenCV 的 mp4v 编码（只是为了端到端能出片）。
**接进 /api/merge/export 时会换成导出用的 ffmpeg 那一套**，届时再拿它对拍单趟导出。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "app"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", action="append", required=True, help="模型名，按第 0/1/2… 张脸分配")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--mask-mode", type=int, default=6)
    a = ap.parse_args()

    from server.backend_merge.pass0 import scan
    from server.backend_merge.pass_swap import run as pass_run
    from server.backend_merge.merge_final import merge

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
        models[0]["faces"] = []

    t0 = time.time()
    idx = scan(v, models=models, start=a.start, end=a.end, log=print)
    metas = []
    for m in models:
        r = pass_run(v, _pool(v) / "index.json", m["name"], m["dfm"], log=print, want_xseg=True)
        if r.get("faces"):
            metas.append(r["path"])
    if not metas:
        print("没有任何模型产出脸原料（这一单没有可换的脸）")
        return 1
    out = a.out or str(v.with_name(v.stem + "_passes.mp4"))
    res = merge(v, metas, out_path=out,
                cfg={"mask_mode": int(a.mask_mode), "color_transfer_mode": "mt",
                     "mode": "seamless"}, log=print)
    print("总计 %.1fs · 成片 %s" % (time.time() - t0, res["out"]))
    return 0


def _pool(v: Path) -> Path:
    from server.backend_merge.pass0 import _pool_dir
    return _pool_dir(v)


if __name__ == "__main__":
    raise SystemExit(main())
