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
import time
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
                 cache_dir: str | None = None, input_name: str | None = None,
                 allow_build: bool = True):
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
        logger = trt.Logger(trt.Logger.WARNING)
        t_all = time.time()

        # ---- 选引擎：**必须真的反序列化成功才算数** ----
        # 以前这里是 sorted(hits)[0]，直接开第一个文件读字节。换过 TensorRT 版本
        # （这台机器上的 _engines/*.bf16.engine 都是 TRT 10.9 编的，运行时是 10.13.3.9）
        # 就会：
        #   1) 白读 9.6 MB；
        #   2) deserialize_cuda_engine 返回 None 而不抛异常；
        #   3) 下一行 self.engine.create_execution_context() 抛 AttributeError；
        #   4) 上层 except 吞掉，回退 onnxruntime+TRT EP，再花 ~2 s 建会话。
        #   每次点击都付一遍这笔钱 —— 这就是「初始化加载模型很慢」。
        cands = []
        want = None
        try:
            from ._engines import engine_path as _ep
            ws = Path(os.environ.get("DFL_WORKSPACE") or (Path(__file__).resolve().parents[2] / "workspace"))
            want = _ep(ws, "arcface_mbf", str(src))
        except Exception:
            want = None
        if want is not None:
            cands.append(want)
            # server 环境没装 tensorrt，trt_version() 读作 0，文件名会带 _trt0；
            # engine 环境能读到真实版本。两边命名不一致 → 同目录里新到旧全试一遍。
            try:
                cands.extend(sorted(want.parent.glob("arcface_mbf_*.bf16.engine"),
                                    key=lambda q: q.stat().st_mtime, reverse=True))
            except OSError:
                pass
        key = hashlib.md5(src.read_bytes()).hexdigest()[:16]
        prec = "fp16" if fp16 else "fp32"
        try:
            trt_ver = str(trt.__version__)
        except Exception:
            trt_ver = "0"
        # 重建出来的那份（名字里带 TRT 版本，换版本就自动不复用）
        cands.append(eng_dir / ("%s_%s_trt%s_b%d_%s.engine" % (src.stem, key, trt_ver, batch, prec)))
        for pat in ("%s_%s_*.engine" % (src.stem, key), "%s_*.engine" % src.stem, "*.bf16.engine"):
            try:
                cands.extend(sorted(eng_dir.glob(pat),
                                    key=lambda q: q.stat().st_mtime, reverse=True))
            except OSError:
                pass

        state = {"engine": None, "file": None, "rebuilt": False, "tried": 0, "bad": []}

        def _deserialize(path):
            if not path or not path.is_file():
                return None
            state["tried"] += 1
            try:
                data = path.read_bytes()
            except OSError:
                return None
            try:
                eng = trt.Runtime(logger).deserialize_cuda_engine(data)
            except Exception:
                eng = None
            if eng is None:
                state["bad"].append(path.name)
            return eng

        seen = set()
        for cand in cands:
            k = str(cand).lower()
            if k in seen:
                continue
            seen.add(k)
            eng = _deserialize(cand)
            if eng is not None:
                state["engine"], state["file"] = eng, str(cand)
                break

        if state["engine"] is None and not allow_build:
            # 点击路径上绝不等 75 秒的构建：直接告诉上层「没引擎」，退 onnxruntime。
            # 重建交给预热（--serve 的 warm / 引擎编译页），下次点击就是 0.6 s 反序列化。
            raise RuntimeError("没有可用的 TRT 引擎（试过 %d 个：%s），需要重建或先用 ONNX 回退"
                               % (state["tried"], ", ".join(state["bad"][:3]) or "-"))

        if state["engine"] is None:
            # 一个都用不了（典型就是换过 TensorRT 版本）→ 重建一份。
            # 写到 cache_dir（runtime/engine/trt_direct，可写），**不覆盖**
            # workspace/_engines 里用户可见的旧引擎 —— 那边交给「引擎编译」页重建。
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
            out = eng_dir / ("%s_%s_trt%s_b%d_%s.engine" % (src.stem, key, trt_ver, batch, prec))
            try:
                with open(out, "wb") as f:
                    f.write(serialized)
            except OSError:
                out = None
            state["engine"] = trt.Runtime(logger).deserialize_cuda_engine(serialized)
            state["file"], state["rebuilt"] = (str(out) if out else None), True
            if state["engine"] is None:
                raise RuntimeError("engine 重建后仍然无法反序列化")

        self.engine = state["engine"]
        # 给上层（CLI / 接口）看的现场：用了哪个文件、试了几个、有没有重建
        self.report = {"engine": state["file"], "rebuilt": bool(state["rebuilt"]),
                       "tried": state["tried"], "bad": state["bad"][:4],
                       "ms": int((time.time() - t_all) * 1000)}
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
