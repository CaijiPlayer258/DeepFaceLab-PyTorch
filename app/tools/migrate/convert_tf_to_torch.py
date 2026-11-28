#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TF(.npy) → torch(.pth) —— 独立的转换器。

为什么单独一个脚本：4.0 把「转换」塞在合成器的「转换并导出 DFM」里，一个按钮干两件事，
脚本内部还得先判断有没有 .pth、没有才去转，然后继续往下导 onnx —— 逻辑缠在一起。
这里只做一件事：把 iperov 原版 TF 2.0 的 .npy 读进 torch，写成 .pth。

真正的加载/保存复用 export_dfm.py 的 convert_npy_to_pth()：
它走 models.import_model(is_exporting=True) + model.save()，这是 4.0 验证过能用的那条路。

用法:
    python app/tools/migrate/convert_tf_to_torch.py \
        --model SAEHD --model-dir workspace/model --name DF-UDT416_SAEHD
"""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[2]           # app/
sys.path.insert(0, str(_APP / "engine"))             # core / models
sys.path.insert(0, str(_APP / "tools" / "migrate"))  # 同目录的 export_dfm
sys.path.insert(0, str(_APP))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="TF(.npy) → torch(.pth)，只做转换，不导 dfm")
    p.add_argument("--model", required=True, dest="model_name",
                   help="模型类型：SAEHD / DeepFakeLarge / LIAELarge / XSeg / XSegLite")
    p.add_argument("--model-dir", required=True, dest="model_dir", help="模型目录")
    p.add_argument("--name", required=True, dest="name",
                   help="模型名；带不带 _SAEHD 后缀都行")
    a = p.parse_args(argv)

    from export_dfm import convert_npy_to_pth

    md = Path(a.model_dir)
    stem = a.name
    suf = "_" + a.model_name
    if stem.endswith(suf):
        stem = stem[: -len(suf)]

    # 已经转过的判据：**逐个模块比对**，不是「目录里有一个 .pth 就算完」。
    # 以前写的是 glob('*.pth') 非空就 [skip]，于是 encoder.pth 转出来、
    # 后面解码器全炸掉的情况下，再点一次「转换」会直接跳过并退出 0 ——
    # 界面显示「完成」，目录里还是缺文件，用户一脸问号（这次 decoder_dst 就是这样）。
    _pre = stem + "_" + a.model_name
    _MODS = ("encoder", "inter", "inter_AB", "inter_B", "decoder",
             "decoder_src", "decoder_dst", "decoder_mask")
    _expect = [m for m in _MODS if (md / (_pre + "_" + m + ".npy")).exists()]
    _missing = [m for m in _expect if not (md / (_pre + "_" + m + ".pth")).exists()]
    if _expect and not _missing:
        print("[skip] %s 已经是 torch 权重（%d 个模块齐全：%s）"
              % (stem, len(_expect), ", ".join(_expect)))
        return 0
    if _missing:
        print("[convert] %s 还缺 %d 个模块：%s"
              % (stem, len(_missing), ", ".join(_missing)))
    if not _expect:
        # .npy 已经被上一次转换归档进 old/：这时判据换成「.pth 齐不齐」，
        # 否则每次点「转换」都会把已经转好的权重再写一遍（模型从 pth 加载、再 save）。
        _have = [m for m in _MODS if (md / (_pre + "_" + m + ".pth")).exists()]
        if len(_have) >= 3:
            print("[skip] %s 的 .npy 已归档，现有 %d 个 .pth：%s"
                  % (stem, len(_have), ", ".join(_have)))
            return 0

    # 失败原因落到文件：stdout 在模型加载层被吞过，界面只能看到「退出码 1」
    _errlog = md / "_convert.error.log"
    try:
        if _errlog.exists():
            _errlog.unlink()
    except Exception:
        pass

    print("[convert] %s (%s): .npy -> .pth" % (stem, a.model_name))
    try:
        convert_npy_to_pth(a.model_name, str(md), stem)
    except Exception:
        import traceback as _tb
        try:
            _errlog.write_text(_tb.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise
    print("[OK] %s 转换完成" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())