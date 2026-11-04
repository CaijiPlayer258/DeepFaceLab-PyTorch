# -*- coding: utf-8 -*-
"""显存探测（**未完成**，见下方"已知阻塞点"）。

已知阻塞点
==========
独立进程里跑 `nn.initialize()` 会抛：
    Exception: nn devices are not initialized. Run initialize_main_env in main process
即使先调 `nn.initialize_main_env()` 也一样。原因是设备枚举依赖
`core/leras/device.py:Devices.initialize_main_env()` 里设置的环境变量/子进程状态，
而在 main.py 之外手动触发的顺序和它不一致。要继续做，需要：
  1) 读 device.py:153-198，搞清 `NN_DEVICES_INITIALIZED` 在哪条分支被置 1；
  2) 或者绕开 `nn.initialize()`，直接用 torch.cuda + 手动构造 DeviceConfig。
在此之前这个脚本**不能**接进界面。

原始说明：把模型建出来、喂一个假 batch 走一遍前向+反向，读真实峰值。

为什么要"探测"而不是"公式"：权重/梯度/动量/激活都能按层算，但
cuDNN workspace（几百 MB~2 GB，取决于运行时选中的算法）和分配器碎片算不出来。
真跑一遍拿 torch.cuda.max_memory_reserved() 是最准的。

用法（引擎 python，在 app/ 目录下）：
    python -m server._vram_probe --model-dir ../workspace/model --name _smoketrain \
        --resolution 128 --batch 8
输出一行 JSON：{"ok":true,"peak_gb":5.71,"params":78.6, ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 直接按文件路径运行时 sys.path[0] 是 tools/ 而不是引擎目录，core/models 都找不到
_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True, help="模型名（不含 _SAEHD_data.dat）")
    ap.add_argument("--resolution", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--archi", default="")
    a = ap.parse_args()

    def _log(msg):
        print("[probe] " + msg, file=sys.stderr, flush=True)

    model_dir = Path(a.model_dir).resolve()
    _log("模型目录 %s" % model_dir)

    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "error": "没有可用的 CUDA 设备"}))
        return 0

    from core.leras import nn
    nn.initialize()

    import models

    # 复用已存在的模型文件：ModelBase 会从 data.dat 里读出全部参数
    kwargs = dict(is_training=True, saved_models_path=model_dir,
                  training_data_src_path=Path("."), training_data_dst_path=Path("."),
                  pretraining_data_path=None, pretrained_model_path=None,
                  no_preview=True, force_model_name=a.name, force_gpu_idxs=None,
                  cpu_only=False, silent_start=True, debug=False)

    # 临时改分辨率/批次：直接写进 data.dat 的 options 更有把握，
    # 但为了不动用户的模型文件，这里用环境变量覆盖（Model 读 options 时生效）
    if a.resolution:
        os.environ["DFL_PROBE_RESOLUTION"] = str(a.resolution)
    if a.batch:
        os.environ["DFL_PROBE_BATCH"] = str(a.batch)
    if a.archi:
        os.environ["DFL_PROBE_ARCHI"] = a.archi

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    _log("开始建图…")
    model = models.import_model("SAEHD")(**kwargs)
    _log("建图完成")

    params = 0
    for p in model.get_model_weights():
        params += p.numel()

    # 一个假 batch：形状取自模型自己的分辨率
    res = int(model.options.get("resolution", 128))
    bs = int(model.options.get("batch_size", 1))
    dev = torch.device("cuda")
    dt = torch.bfloat16 if model.options.get("use_bf16") else torch.float32

    def t(*shape):
        return torch.zeros(*shape, device=dev, dtype=dt)

    imgs = [t(bs, 3, res, res) for _ in range(4)]
    masks = [t(bs, 1, res, res) for _ in range(4)]

    print("[probe] 构建完成，params=%.1fM  res=%d  batch=%d" % (params / 1e6, res, bs), file=sys.stderr)

    warped_src, target_src, target_srcm, target_srcm_em = imgs[0], imgs[1], masks[0], masks[1]
    warped_dst, target_dst, target_dstm, target_dstm_em = imgs[2], imgs[3], masks[2], masks[3]

    _log("开始前向+反向…")
    model.train_one_step(warped_src, target_src, target_srcm, target_srcm_em,
                         warped_dst, target_dst, target_dstm, target_dstm_em)
    torch.cuda.synchronize()
    _log("前反向完成")

    peak = torch.cuda.max_memory_reserved() / (1 << 30)
    alloc = torch.cuda.max_memory_allocated() / (1 << 30)
    print(json.dumps({"ok": True, "peak_gb": round(peak, 2), "alloc_gb": round(alloc, 2),
                      "params_m": round(params / 1e6, 1), "resolution": res, "batch": bs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
