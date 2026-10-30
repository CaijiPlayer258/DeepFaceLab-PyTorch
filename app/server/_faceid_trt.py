"""直接跑 TensorRT（不用 onnxruntime 的 TRT EP）。

为什么：
  ORT 的 TRT EP 每批要做 Python 绑定 → 设备拷贝 → 同步 → 回拷，
  bs=32 时约 5 ms/批（0.158 ms/张）。自己持有 cudaEngine + 预分配
  torch 张量 + 异步流，可以把这段开销省掉。

引擎按「onnx 的 md5 + profile + 精度」缓存到 runtime/engine/trt_direct/，
不重复构建。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def _dll_dirs():
    eng = Path(__file__).resolve().parents[2] / "runtime" / "engine"
    dirs = [eng / "cuda_dlls", eng / "Lib" / "site-packages" / "torch" / "lib"]
    for d in dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs if d.is_dir()) + os.pathsep + os.environ.get("PATH", "")


class TrtEmbedder:
    """固定 batch 的 ArcFace 推理器：run(np[HWC float32]) -> np[N,512]。"""

    def __init__(self, onnx_path: str, batch: int = 32, fp16: bool = True,
                 cache_dir: str | None = None, input_name: str | None = None):
        _dll_dirs()
        import tensorrt as trt
        import torch

        self.torch = torch
        self.batch = batch
        self.trt = trt
        src = Path(onnx_path)
        # 复用「引擎编译」页那套命名（workspace/_engines/<name>_<md5>_sm<SM>_trt<ver>.bf16.engine），
        # 这样编译页里能看到它、也能重建；构建完成后两边共用同一份文件。
        eng_dir = Path(cache_dir or (Path(__file__).resolve().parents[2] / "runtime" / "engine" / "trt_direct"))
        eng_dir.mkdir(parents=True, exist_ok=True)
        cache = None
        try:
            from ._engines import engine_path as _ep
            ws = Path(os.environ.get("DFL_WORKSPACE") or (Path(__file__).resolve().parents[2] / "workspace"))
            cache = _ep(ws, "arcface_mbf", str(src))
        except Exception:
            cache = None
        if cache is not None:
            # server 环境没装 tensorrt，trt_version() 读作 0，文件名会带 _trt0；
            # engine 环境能读到真实版本。两边命名不一致 → 直接按前缀找已有引擎。
            hits = sorted(cache.parent.glob("arcface_mbf_*.bf16.engine"))
            if hits:
                cache = hits[0]
        if cache is None:
            key = hashlib.md5(src.read_bytes()).hexdigest()[:16]
            cache = eng_dir / f"{src.stem}_{key}_b{batch}_{'fp16' if fp16 else 'fp32'}.engine"

        logger = trt.Logger(trt.Logger.WARNING)
        if cache.exists():
            with open(cache, "rb") as f:
                serialized = f.read()
        else:
            builder = trt.Builder(logger)
            net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse(src.read_bytes()):
                raise RuntimeError("onnx 解析失败: " + str(parser.get_error(0)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if fp16:
                cfg.set_flag(trt.BuilderFlag.FP16)
            prof = builder.create_optimization_profile()
            name = input_name or net.get_input(0).name
            prof.set_shape(name, (1, 3, 112, 112), (batch, 3, 112, 112), (batch, 3, 112, 112))
            cfg.add_optimization_profile(prof)
            serialized = builder.build_serialized_network(net, cfg)
            if serialized is None:
                raise RuntimeError("engine 构建失败")
            serialized = bytes(serialized)
            with open(cache, "wb") as f:
                f.write(serialized)

        self.engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
        self.ctx = self.engine.create_execution_context()
        self.input_name = input_name or self.engine.get_tensor_name(0)
        self.ctx.set_input_shape(self.input_name, (batch, 3, 112, 112))

        # 预分配张量
        self.inp = torch.empty((batch, 3, 112, 112), dtype=torch.float32, device="cuda")
        out_shape = tuple(self.engine.get_tensor_shape(self.engine.get_tensor_name(1)))
        out_shape = (batch,) + tuple(out_shape[1:])
        self.out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.input_name, int(self.inp.data_ptr()))
        self.ctx.set_tensor_address(self.engine.get_tensor_name(1), int(self.out.data_ptr()))
        self.stream = torch.cuda.Stream()
        # 页锁定内存：H2D 才能真正异步（不然 torch 会退化成同步拷贝，慢一倍）
        self.host = torch.empty((batch, 3, 112, 112), dtype=torch.float32, pin_memory=True)
        # 回读也用页锁定缓冲，避免每次 .cpu() 都新建 4.8 MB 页锁定内存（实测这一项占了 3.9 ms/批）
        self.host_out = torch.empty(out_shape, dtype=torch.float32, pin_memory=True)
        self.host_u8 = torch.empty((batch, 112, 112, 3), dtype=torch.uint8, pin_memory=True)
        self.dev_u8 = torch.empty((batch, 112, 112, 3), dtype=torch.uint8, device="cuda")

    def run(self, imgs_hwc: np.ndarray) -> np.ndarray:
        """imgs_hwc: (N,112,112,3) float32，RGB，0..255。"""
        torch = self.torch
        n = imgs_hwc.shape[0]
        # 归一化搬到 GPU 做：numpy 上对 1.2M 元素做两次浮点运算 + 转置拷贝
        # 实测要 3~4 ms/批，占总时间的三分之一以上。
        if self.host_u8.shape[0] < n:
            self.host_u8 = torch.empty((n, 112, 112, 3), dtype=torch.uint8, pin_memory=True)
            self.dev_u8 = torch.empty((n, 112, 112, 3), dtype=torch.uint8, device="cuda")
        self.host_u8[:n].copy_(torch.from_numpy(np.ascontiguousarray(imgs_hwc, dtype=np.uint8)))
        with torch.cuda.stream(self.stream):
            buf = self.inp[:n] if n < self.batch else self.inp
            self.dev_u8[:n].copy_(self.host_u8[:n], non_blocking=True)
            buf.copy_(self.dev_u8[:n].permute(0, 3, 1, 2).float().sub_(127.5).div_(127.5))
            self.ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
            # D2H 也提交到同一 stream，和计算串起来，只同步一次
            self.host_out[:n].copy_(self.out[:n], non_blocking=True)
        self.stream.synchronize()
        return self.host_out[:n].numpy()
