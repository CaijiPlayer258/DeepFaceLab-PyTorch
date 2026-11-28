"""编译 TensorRT 引擎（BF16 / INT8）—— 独立进程跑，主进程不阻塞。

为什么单独一个进程：
  · 编译 det_10g 那种模型要几分钟，绝不能卡住 GUI
  · TensorRT 的 builder 会吃掉大量显存，跑完就释放，不和推理进程抢

用法：
  python _trt_build.py --onnx X.onnx --out Y.engine --name tinymog --bf16
  python _trt_build.py --onnx xseglite.onnx --out xseglite...int8.engine \
         --name xseglite --int8 --calib-dir workspace/data_dst/aligned

INT8 的取舍（docs/09-PERF.md、docs/41-ENGINE-PRECISION.md）：
  只有**算力密集**的遮罩值得量化（XSegLite 2.04 → 0.751 ms，2.7×）。
  检测是 launch 瓶颈（Graph 更值），特征点是硬坐标（关键点漂移 2.26 px，否决）。
  所以精度按模型定，不是全局开关 —— 见 backend_models/engines.py 的 INT8_MODELS。

协议（stdout，与 _scan_runner 一致）：
  @@P <pct> <stage>        进度
  @@OK <engine_path>       成功
  @@ERR <message>          失败
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _say(s):
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def _calib_batches(dirs, n, size=256, seed=0):
    """从对齐脸目录里按固定种子抽 N 张，按**推理同一条口径**预处理。

    口径必须和 _vision/xseg.py 的 mask() 完全一致：
        cv2.dnn.blobFromImage(img, 1/255, (256,256), swapRB=True, crop=False)
    否则量化出来的 scale 是按另一套输入分布估的，质量对不上。

    而且必须是**真实对齐脸** —— 合成噪声/纯随机图会让激活范围估偏
    （docs/09-PERF.md 第 5 节：校准集用真实对齐脸，本次 200 张）。
    DFLJPG 也行：APP15 段被 imdecode 忽略；中文路径要用 np.fromfile。
    """
    import cv2
    import numpy as np

    cands = []
    for d in (dirs or []):
        p = Path(str(d))
        if not p.is_dir():
            continue
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            cands.extend(str(x) for x in p.glob(pat))
    cands = sorted(set(cands))
    if not cands:
        return []
    rng = np.random.RandomState(int(seed))
    pick = rng.choice(len(cands), size=min(int(n), len(cands)), replace=False)
    out = []
    for i in pick:
        try:
            buf = np.fromfile(cands[int(i)], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None or getattr(img, "ndim", 0) != 3:
                continue
            if img.shape[0] != size or img.shape[1] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            out.append(cv2.dnn.blobFromImage(img, 1.0 / 255.0, (size, size),
                                             swapRB=True, crop=False))
        except Exception:
            continue
    return out


def _make_calibrator(trt):
    """IInt8EntropyCalibrator2 的子类（要在 import tensorrt 之后才能建）。

    device buffer 用 torch 分配：比拉 pycuda 少一个依赖，且显存由 torch 的
    分配器管（和推理进程同一套，不会互相抢）。
    """

    class _Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self, batches, cache_path, input_name, batch_size=1):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._b = list(batches or [])
            self._i = 0
            self._cache = str(cache_path or "")
            self._name = str(input_name or "")
            self._bs = max(1, int(batch_size))
            self._keep = None          # 保住 device 张量，别让它在 build 中途被释放
            self._total = len(self._b)
            self.written = 0

        # --- 协议 ---
        def get_batch_size(self):
            return self._bs

        def get_batch(self, names):
            if self._i >= self._total:
                return None
            import torch
            arr = self._b[self._i]
            self._i += 1
            t = torch.from_numpy(arr).contiguous().cuda()
            self._keep = t
            if self._i == 1 or self._i % 20 == 0 or self._i == self._total:
                _say("@@P %d 校准 %d/%d" % (20 + int(60.0 * self._i / max(1, self._total)),
                                            self._i, self._total))
            return [int(t.data_ptr())]

        def read_calibration_cache(self):
            # 有缓存就不重新校准：200 张跑一遍要几十秒，而且结果只跟数据与网络有关
            if self._cache and os.path.isfile(self._cache):
                try:
                    with open(self._cache, "rb") as f:
                        return f.read()
                except Exception:
                    return None
            return None

        def write_calibration_cache(self, cache):
            if not self._cache:
                return
            try:
                with open(self._cache, "wb") as f:
                    f.write(bytes(cache))
                self.written = len(bytes(cache))
            except Exception as e:
                _say("@@P 20 写校准缓存失败（不影响引擎）: %s" % e)

    return _Calib


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
    # --- INT8 校准通道 ---
    ap.add_argument("--int8", action="store_true", default=False,
                    help="开 INT8（熵校准量化）。必须同时给 --calib-dir")
    ap.add_argument("--calib-dir", action="append", default=[],
                    help="校准集目录（真实对齐脸），可给多次；只取前 N 张")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--calib-cache", default="",
                    help="校准表缓存路径；默认 <out>.calib.cache")
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

    if args.int8:
        if not args.calib_dir:
            _say("@@ERR --int8 需要 --calib-dir（真实对齐脸目录）")
            return 6
        _bs = 1
        if args.profile:
            try:
                _bs = max(1, int(args.profile.split(";")[1].split(",")[0]))
            except Exception:
                _bs = 1
        batches = _calib_batches(args.calib_dir, args.calib_n, seed=args.calib_seed)
        if not batches:
            _say("@@ERR 校准集是空的（%s）—— 目录里没有 jpg/png"
                 % ", ".join(args.calib_dir))
            return 6
        _cache = args.calib_cache or (str(args.out) + ".calib.cache")
        _say("@@P 22 校准集 %d 张 · %s · 缓存 %s"
             % (len(batches), " + ".join(Path(d).name for d in args.calib_dir),
                os.path.basename(_cache)))
        _Calib = _make_calibrator(trt)
        cal = _Calib(batches, _cache, network.get_input(0).name, batch_size=_bs)
        try:
            cfg.set_flag(trt.BuilderFlag.INT8)
            cfg.int8_calibrator = cal
        except Exception as e:
            _say("@@ERR 设置 INT8 失败: %s" % e)
            return 7

    if args.bf16:
        # 只开 BF16。之前额外开 FP16 会让构建变慢，而推理并没有更快。
        # INT8 时这一项仍然要开：没有被量化的层（首尾+回退）走 BF16，比 FP32 快。
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
    mb = os.path.getsize(out) / 1024 / 1024
    _say("@@P 100 完成 %.1f s，%.1f MB" % (time.time() - t0, mb))

    if args.int8:
        # 留一份「这份引擎是用什么校准出来的」：换了校准集要重编，缓存也要一起换
        try:
            meta = {"model": args.name, "onnx": os.path.abspath(args.onnx),
                    "precision": "int8", "calib_n": len(batches),
                    "calib_dirs": [os.path.abspath(d) for d in args.calib_dir],
                    "calib_seed": int(args.calib_seed),
                    "calib_cache": os.path.abspath(str(args.calib_cache or (str(args.out) + ".calib.cache"))),
                    "engine_mb": round(mb, 1),
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(out + ".calib.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    _say("@@OK " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
