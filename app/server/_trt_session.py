"""用我们自己的 .engine 文件做推理（TensorRT pybind + torch 张量）。

为什么不用 onnxruntime 的 TensorrtExecutionProvider：
  它只认自己缓存的引擎（按内部指纹命名），没法直接用我们按
  "模型 MD5 + SM" 命名的 .engine 文件 —— 而这个命名正是需求。

接口与 onnxruntime.InferenceSession 对齐：get_inputs / get_outputs / get_providers / run。
上层（_xseg / _detect）只要把 sess 换掉就行，解码逻辑不用动。

CUDA Graph（2026-11 实测，RTX 5070 Laptop / TRT 10.13.3.9 / torch 2.9.1）：
  · 图按 **输入形状** 缓存，一个形状只捕获一次，之后只 replay；
  · 捕获本身 7~20 ms（batch 越大越久：b1 7 ms，b8 20 ms），replay 与普通执行逐位相同
    （batch 3 / batch 8 都实测 max|Δ| = 0.0）；
  · **单张是最频繁的路径**，以前 run_tensor 里根本没有图（只有 run 里有），
    等于"开了 use_graph 但单张一次都没用上"。现在两条路共用同一套图。
  · batch>1 也有收益（纯推理 1.29 → 1.10 ms/帧），但形状会随队列在 1..8 之间变，
    每种形状要多付一次捕获 —— 所以捕获次数有上限（graph_max_captures），
    超了就不再捕获，已捕获的形状照样 replay，并打一行 [TRT] 日志说明。
  · 捕获/回放失败只影响那一个形状，并且**一定**打日志（以前是一旦失败就永久关图、
    一句话都不说）。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


class _Info:
    """模仿 ort 的 NodeArg：只要 name / shape / type 三个属性。"""

    def __init__(self, name, shape, type_str):
        self.name = name
        self.shape = list(shape)
        self.type = type_str


def _say(msg):
    print("[TRT] " + msg, flush=True)


class TrtSession:
    # 捕获次数上限：形状在 1..8 之间变的话最多 8 张图，给点余量
    DEFAULT_MAX_CAPTURES = 8

    def __init__(self, engine_path, device_id: int = 0, use_graph: bool = True,
                 graph_max_batch: int = 1, graph_max_captures: int = None):
        import tensorrt as trt
        import torch

        self._torch = torch
        self.device_id = int(device_id)
        # 图重放只在「单线程 + 延迟密集」的模型上划算；多线程/算力密集一律关掉
        self.use_graph = bool(use_graph)
        self.graph_max_batch = max(1, int(graph_max_batch))
        self.graph_max_captures = int(graph_max_captures or self.DEFAULT_MAX_CAPTURES)
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
        self.precision = None          # 由 _engines.try_engine 填（int8 / bf16）
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

        # ---- CUDA Graph 状态（以前 _graph_ok 要等第一次失败才出现，现在显式初始化）----
        # 图里有固定的输入/输出缓冲，多线程共用同一张图会互相覆盖 —— 所以图按线程隔离，
        # _graph_cache 是"当前线程的图"，self._graph_shapes_seen 只用来做状态展示。
        self._graph_ok = True
        self._graph_shapes_seen = set()
        self._graph_capture_errors = 0
        self._graph_skip_logged = set()
        self._graph_disabled_reason = ""
        self.graph_stats = {"replay": 0, "capture": 0, "capture_ms": 0.0,
                            "capture_fail": 0, "bypass": 0}

        _say("载入引擎 %s（输入 maxbatch=%s, use_graph=%s）"
             % (Path(self.path).name, self.max_batch(), self.use_graph))

    @property
    def _graph_cache(self):
        """当前线程的 {shape: 图}。多线程各存各的（缓冲不能共享）。"""
        g = getattr(self._local, "graphs", None)
        if g is None:
            g = {}
            self._local.graphs = g
        return g

    # ---- 与 ort 一致的元数据接口 ----
    def get_inputs(self):
        return self._in

    def get_outputs(self):
        return self._out

    def get_providers(self):
        return ["TensorrtExecutionProvider"]

    # ---- 引擎能力 ----
    def max_batch(self):
        """引擎允许的最大 batch（静态引擎返回它的固定值；拿不到返回 None）。"""
        name = self.input_names[0]
        try:
            n = int(self.engine.get_tensor_profile_shape(name, 0)[2][0])
            if n > 0:
                return n
        except Exception:
            pass
        try:
            n = int(self.engine.get_tensor_shape(name)[0])
            return n if n > 0 else None
        except Exception:
            return None

    def describe(self):
        """一行状态（给日志/自检用）。"""
        return ("%s use_graph=%s(max_batch=%d) graph_ok=%s shapes=%d replay=%d capture=%d%s"
                % (Path(self.path).name, self.use_graph, self.graph_max_batch, self._graph_ok,
                   len(self._graph_shapes_seen), self.graph_stats["replay"],
                   self.graph_stats["capture"],
                   ("" if not self._graph_disabled_reason
                    else " (图已关: %s)" % self._graph_disabled_reason)))

    # ---- 上下文 / 张量 ----
    def _state(self):
        st = getattr(self._local, "st", None)
        if st is None:
            st = {"ctx": self.engine.create_execution_context(),
                  "stream": self._torch.cuda.Stream(device=self.device_id)}
            self._local.st = st
        return st

    def _out_specs(self, ctx):
        """设置完输入形状后才能取到输出形状。"""
        specs = []
        for oname in self.output_names:
            shape = tuple(ctx.get_tensor_shape(oname))
            dtype = self._to_torch.get(self.engine.get_tensor_dtype(oname), self._torch.float32)
            specs.append((oname, shape, dtype))
        return specs

    def _alloc(self, specs):
        return [self._torch.empty(s, dtype=d, device="cuda") for (_, s, d) in specs]

    def _set_shape(self, ctx, name, shape):
        """设置输入形状，并且**核对真的设上了**。

        TensorRT 对静态引擎（例如老 int8 的 batch=1）只会在 stderr 打一行
        "Static dimension mismatch"，然后继续按 batch=1 跑 —— 上层拿到的是
        少一半的输出，或者直接 IndexError。这里当场把它变成异常。
        """
        ctx.set_input_shape(name, shape)
        got = tuple(ctx.get_tensor_shape(name))
        if got != tuple(shape):
            raise RuntimeError("引擎不接受输入形状 %s（实际生效 %s）：多半是静态 batch 引擎被喂了批量"
                               % (tuple(shape), got))

    def _bind(self, ctx, x, outs):
        ctx.set_tensor_address(self.input_names[0], int(x.data_ptr()))
        for oname, buf in zip(self.output_names, outs):
            ctx.set_tensor_address(oname, int(buf.data_ptr()))

    def _launch(self, ctx, x, outs):
        """普通执行（不经过图）。"""
        t = self._torch
        st = self._state()
        self._bind(ctx, x, outs)
        cur = t.cuda.current_stream(self.device_id)
        st["stream"].wait_stream(cur)
        ok = ctx.execute_async_v3(stream_handle=st["stream"].cuda_stream)
        cur.wait_stream(st["stream"])
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 返回 False（shape=%s）" % (tuple(x.shape),))

    # ---- CUDA Graph ----
    def _graph_lookup(self, shape):
        if not (self.use_graph and self._graph_ok):
            return None
        return self._graph_cache.get(shape)

    def _replay(self, g, x, shape):
        """把输入拷进静态缓冲再 replay；失败就丢掉这张图并返回 None（调用方走普通执行）。"""
        t = self._torch
        try:
            with t.cuda.device(self.device_id):
                g["x"].copy_(x.reshape(g["x"].shape), non_blocking=True)
                g["graph"].replay()
                t.cuda.synchronize(self.device_id)
        except Exception as e:                                  # noqa: BLE001
            self._graph_cache.pop(shape, None)
            _say("CUDA Graph replay 失败（shape=%s）：%r —— 丢掉这张图，本次改走普通执行"
                 % (shape, e))
            return None
        self.graph_stats["replay"] += 1
        return [b.cpu().numpy() for b in g["outs"]]

    def _maybe_capture(self, ctx, shape, x, outs):
        """首次见到这个形状：把这次执行捕获成图，下次直接 replay。"""
        if not self.use_graph or not self._graph_ok or shape in self._graph_cache:
            return
        batch = int(shape[0]) if shape else 1
        if batch > self.graph_max_batch:
            if "_batch" not in self._graph_skip_logged:
                self._graph_skip_logged.add("_batch")
                _say("batch=%d 超过 CUDA Graph 上限 %d → 这些形状不走图，照常推理"
                     "（batch>1 的纯推理只快约 5%，却要在抽取中段多付 3~10 ms 捕获）；"
                     "要开就把 graph_max_batch 调大" % (batch, self.graph_max_batch))
            return
        if len(self._graph_cache) >= self.graph_max_captures:
            if "_cap" not in self._graph_skip_logged:
                self._graph_skip_logged.add("_cap")
                _say("CUDA Graph 已捕获 %d 种形状（上限 %d），不再捕获新形状：%s —— "
                     "批次形状一直在变时，捕获开销（每次 7~20 ms）比 replay 省下的还多"
                     % (len(self._graph_cache), self.graph_max_captures, shape))
            return
        t = self._torch
        name = self.input_names[0]
        t0 = time.perf_counter()
        try:
            with t.cuda.device(self.device_id):
                gx = t.empty_like(x)
                gx.copy_(x)
                ctx.set_input_shape(name, shape)
                self._bind(ctx, gx, outs)
                graph = t.cuda.CUDAGraph()
                with t.cuda.graph(graph):
                    ok = ctx.execute_async_v3(stream_handle=t.cuda.current_stream().cuda_stream)
                t.cuda.synchronize(self.device_id)
                if not ok:
                    raise RuntimeError("捕获期间 execute_async_v3 返回 False")
                self._bind(ctx, gx, outs)          # 捕获会临时改地址，恢复成静态缓冲
                self._graph_cache[shape] = {"graph": graph, "x": gx, "outs": outs}
                self._graph_shapes_seen.add(shape)
        except Exception as e:                                  # noqa: BLE001
            self._graph_capture_errors += 1
            self.graph_stats["capture_fail"] += 1
            _say("CUDA Graph 捕获失败（shape=%s）：%r —— 该形状回退普通执行（第 %d 次失败）"
                 % (shape, e, self._graph_capture_errors))
            if self._graph_capture_errors >= 3:
                self._graph_ok = False
                self._graph_disabled_reason = "连续 %d 次捕获失败" % self._graph_capture_errors
                _say("CUDA Graph 连续失败 %d 次，全局关闭图重放；推理本身不受影响"
                     % self._graph_capture_errors)
            return
        ms = (time.perf_counter() - t0) * 1000.0
        self.graph_stats["capture"] += 1
        self.graph_stats["capture_ms"] += ms
        _say("CUDA Graph 捕获 shape=%s 用时 %.1f ms（之后该形状只 replay）" % (shape, ms))

    # ---- 推理 ----
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
        shape = tuple(x.shape)
        with t.cuda.device(self.device_id):
            self._set_shape(ctx, name, shape)
            g = self._graph_lookup(shape)
            if g is not None:
                r = self._replay(g, x, shape)
                if r is not None:
                    return r
                self.graph_stats["bypass"] += 1
            outs = self._alloc(self._out_specs(ctx))
            self._launch(ctx, x, outs)
            self._maybe_capture(ctx, shape, x, outs)
        return [o.cpu().numpy() for o in outs]

    def run(self, _names, feed: dict):
        t = self._torch
        st = self._state()
        ctx = st["ctx"]
        name = self.input_names[0]
        arr = np.ascontiguousarray(feed[name])
        shape = tuple(arr.shape)
        with t.cuda.device(self.device_id):
            self._set_shape(ctx, name, shape)
            g = self._graph_lookup(shape)
            if g is not None:
                # 直接 H2D 进静态缓冲，省掉一次 39 MB 的临时张量分配（batch8 实测 ~0.1 ms/批）
                try:
                    gx = g["x"]
                    gx.copy_(t.from_numpy(arr).reshape(gx.shape), non_blocking=True)
                    g["graph"].replay()
                    t.cuda.synchronize(self.device_id)
                    self.graph_stats["replay"] += 1
                    return [b.cpu().numpy() for b in g["outs"]]
                except Exception as e:                          # noqa: BLE001
                    self._graph_cache.pop(shape, None)
                    self.graph_stats["bypass"] += 1
                    _say("CUDA Graph replay 失败（shape=%s）：%r —— 丢掉这张图，本次改走普通执行"
                         % (shape, e))
            x = t.from_numpy(arr).cuda(non_blocking=True)
            outs = self._alloc(self._out_specs(ctx))
            self._launch(ctx, x, outs)
            self._maybe_capture(ctx, shape, x, outs)
        return [o.cpu().numpy() for o in outs]
