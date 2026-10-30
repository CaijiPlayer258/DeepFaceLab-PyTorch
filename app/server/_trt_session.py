"""用我们自己的 .engine 文件做推理（TensorRT pybind + torch 张量）。

为什么不用 onnxruntime 的 TensorrtExecutionProvider：
  它只认自己缓存的引擎（按内部指纹命名），没法直接用我们按
  "模型 MD5 + SM" 命名的 .engine 文件 —— 而这个命名正是需求。

接口与 onnxruntime.InferenceSession 对齐：get_inputs / get_outputs / get_providers / run。
上层（_xseg / _detect）只要把 sess 换掉就行，解码逻辑不用动。
"""
from __future__ import annotations

import threading

import numpy as np


class _Info:
    """模仿 ort 的 NodeArg：只要 name / shape / type 三个属性。"""

    def __init__(self, name, shape, type_str):
        self.name = name
        self.shape = list(shape)
        self.type = type_str


class TrtSession:
    def __init__(self, engine_path, device_id: int = 0, use_graph: bool = True):
        import tensorrt as trt
        import torch

        self._torch = torch
        self.device_id = int(device_id)
        # 图重放只在「单线程 + 延迟密集」的模型上划算；多线程/算力密集一律关掉
        self.use_graph = bool(use_graph)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("引擎反序列化失败: %s" % engine_path)
        self.path = str(engine_path)

        to_torch = {trt.DataType.FLOAT: torch.float32,
                    trt.DataType.HALF: torch.float16,
                    trt.DataType.INT32: torch.int32,
                    trt.DataType.INT8: torch.int8,
                    trt.DataType.BOOL: torch.bool}
        self._to_torch = to_torch
        self.input_names, self.output_names = [], []
        self._in, self._out = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            kind = self.engine.get_tensor_dtype(name)
            info = _Info(name, shape, str(kind))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self._in.append(info)
            else:
                self.output_names.append(name)
                self._out.append(info)
        self._local = threading.local()

    # ---- 与 ort 一致的元数据接口 ----
    def get_inputs(self):
        return self._in

    def get_outputs(self):
        return self._out

    def get_providers(self):
        return ["TensorrtExecutionProvider"]

    def _state(self):
        st = getattr(self._local, "st", None)
        if st is None:
            st = {"ctx": self.engine.create_execution_context(),
                  "stream": self._torch.cuda.Stream(device=self.device_id)}
            self._local.st = st
        return st

    def _graph_for(self, shape):
        """按输入形状缓存一张 CUDA Graph：缓冲区固定、只 replay。
        捕获失败返回 None（调用方回退到普通执行）。"""
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = self._graph_cache = {}
        return cache.get(tuple(shape))

    def run_tensor(self, feed: dict):
        """同 run()，但输入是已经躺在 GPU 上的 torch.Tensor（跳过 numpy 转换与 H2D）。"""
        t = self._torch
        st = self._state()
        ctx = st["ctx"]
        name = self.input_names[0]
        x = feed[name]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.contiguous()
        with t.cuda.device(self.device_id):
            ctx.set_input_shape(name, tuple(x.shape))
            ctx.set_tensor_address(name, int(x.data_ptr()))
            outs = []
            for oname in self.output_names:
                shape = tuple(ctx.get_tensor_shape(oname))
                dtype = self._to_torch.get(self.engine.get_tensor_dtype(oname), t.float32)
                buf = t.empty(shape, dtype=dtype, device="cuda")
                ctx.set_tensor_address(oname, int(buf.data_ptr()))
                outs.append(buf)
            cur = t.cuda.current_stream(self.device_id)
            st["stream"].wait_stream(cur)
            ok = ctx.execute_async_v3(stream_handle=st["stream"].cuda_stream)
            cur.wait_stream(st["stream"])
            cur.synchronize()
        if not ok:
            raise RuntimeError("TensorRT 执行失败")
        return [o.cpu().numpy() for o in outs]

    def run(self, _names, feed: dict):
        t = self._torch
        st = self._state()
        ctx = st["ctx"]
        torch = t
        name0 = self.input_names[0]
        key = tuple(np.ascontiguousarray(feed[name0]).shape)
        g = self._graph_for(key)
        if g is not None and self.use_graph and getattr(self, "_graph_ok", True):
            try:
                with torch.cuda.device(self.device_id):
                    g["x"].copy_(torch.from_numpy(np.ascontiguousarray(feed[name0])).view_as(g["x"]), non_blocking=True)
                    g["graph"].replay()
                    torch.cuda.synchronize(self.device_id)
                return [b.cpu().numpy() for b in g["outs"]]
            except Exception:
                self._graph_ok = False
        with torch.cuda.device(self.device_id):
            name = self.input_names[0]
            arr = np.ascontiguousarray(feed[name])
            ctx.set_input_shape(name, tuple(arr.shape))
            x = torch.from_numpy(arr).cuda(non_blocking=True)
            ctx.set_tensor_address(name, int(x.data_ptr()))

            outs = []
            for oname in self.output_names:
                shape = tuple(ctx.get_tensor_shape(oname))
                dtype = self._to_torch.get(self.engine.get_tensor_dtype(oname), torch.float32)
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                ctx.set_tensor_address(oname, int(buf.data_ptr()))
                outs.append(buf)

            cur = torch.cuda.current_stream(self.device_id)
            st["stream"].wait_stream(cur)
            ok = ctx.execute_async_v3(stream_handle=st["stream"].cuda_stream)
            cur.wait_stream(st["stream"])
            cur.synchronize()

            # 首次见到这个形状：把这次执行捕获成图，下次直接 replay
            if g is None and self.use_graph and getattr(self, "_graph_ok", True):
                try:
                    gx = torch.empty_like(x)
                    gx.copy_(x)
                    ctx.set_tensor_address(name, int(gx.data_ptr()))
                    ctx.execute_async_v3(stream_handle=st["stream"].cuda_stream)
                    st["stream"].synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize(self.device_id)
                    ctx.set_tensor_address(name, int(gx.data_ptr()))
                    self._graph_cache[key] = {"graph": graph, "x": gx, "outs": outs}
                except Exception:
                    self._graph_ok = False
        if not ok:
            raise RuntimeError("TensorRT 执行失败")
        return [o.cpu().numpy() for o in outs]
