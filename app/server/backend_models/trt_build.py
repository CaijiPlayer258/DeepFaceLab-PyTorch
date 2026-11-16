"""编译 TensorRT 引擎（BF16）—— 独立进程跑，主进程不阻塞。

为什么单独一个进程：
  · 编译 det_10g 那种模型要几分钟，绝不能卡住 GUI
  · TensorRT 的 builder 会吃掉大量显存，跑完就释放，不和推理进程抢

用法：
  python _trt_build.py --onnx X.onnx --out Y.engine --name tinymog --bf16

协议（stdout，与 _scan_runner 一致）：
  @@P <pct> <stage>        进度
  @@OK <engine_path>       成功
  @@ERR <message>          失败
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _say(s):
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    # min,opt,max 三组形状，用分号隔开，逗号分隔维度；batch 维动态的模型必须给
    ap.add_argument("--profile", default="")
    # 优化级别 0..5（默认 3）。实测 1 与 3 的**推理速度完全一样**，但构建快 3.5 倍：
    #   TinyMog  lvl3 38.0 s / 2.20 ms   ·   lvl1 11.0 s / 2.18 ms
    ap.add_argument("--opt-level", type=int, default=1)
    args = ap.parse_args()

    try:
        import tensorrt as trt
    except Exception as e:
        _say("@@ERR 没有安装 tensorrt（%s）" % e)
        return 2

    _say("@@P 5 TensorRT %s · 解析 %s" % (trt.__version__, os.path.basename(args.onnx)))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 必须用 parse_from_file：TinyMog 这类模型带**外部权重**（.onnx.data），
    # 用 parse(bytes) 只会解析图结构，权重找不到，报 INVALID_GRAPH: Failed to import initializer
    if not parser.parse_from_file(str(os.path.abspath(args.onnx))):
        msgs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        _say("@@ERR ONNX 解析失败: %s" % ("; ".join(msgs[:3]) or "unknown"))
        return 3
    _say("@@P 20 网络已解析，开始构建")

    cfg = builder.create_builder_config()
    try:
        cfg.builder_optimization_level = max(0, min(5, int(args.opt_level)))
    except Exception:
        pass

    # 动态维模型必须先建优化档位，否则 build 直接报错
    dyn = [i for i in range(network.num_inputs)
           if any(d < 0 for d in tuple(network.get_input(i).shape))]
    if dyn and not args.profile:
        _say("@@ERR 输入含动态维，但没有给 --profile")
        return 5
    if args.profile:
        parts = [tuple(int(v) for v in p.split(",")) for p in args.profile.split(";")]
        if len(parts) != 3:
            _say("@@ERR --profile 需要 min;opt;max 三段")
            return 5
        prof = builder.create_optimization_profile()
        for i in dyn or [0]:
            name = network.get_input(i).name
            prof.set_shape(name, parts[0], parts[1], parts[2])
        cfg.add_optimization_profile(prof)
        _say("@@P 20 优化档位 %s" % (args.profile,))
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)
    if args.bf16:
        # 只开 BF16。之前额外开 FP16 会让构建变慢，而推理并没有更快。
        try:
            cfg.set_flag(trt.BuilderFlag.BF16)
        except Exception as e:
            _say("@@P 20 设置 BF16 失败（退回 FP32）: %s" % e)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        _say("@@ERR 构建失败（见 TensorRT 日志）")
        return 4

    out = os.path.abspath(args.out)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialized)
    os.replace(tmp, out)          # 原子替换：中途失败不会留下半个引擎
    _say("@@P 100 完成 %.1f s，%.1f MB" % (time.time() - t0, os.path.getsize(out) / 1024 / 1024))
    _say("@@OK " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
