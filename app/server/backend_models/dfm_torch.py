# -*- coding: utf-8 -*-
"""dfm 的 torch 推理后端（CUDA eager）—— 替代 onnxruntime。

为什么要它：另一套实现用 torch eager + bf16 跑 dfm。实测（app/tools/bench/DFM_TORCH_REPORT.md）
torch eager + TF32 比 ONNX 快 3~13%，而 bf16 是**这台 5070 Laptop 唯一能真正省钱的一档**
（显存减半、tensor core 翻倍）。这个模块把那条实验路线做成可切换的生产后端。

接口刻意和 onnxruntime 对齐（`_dfm_run_batch` 一行都不用改）：

    s = load("x.dfm")                        # 按 DFL_DFM_DTYPE 定精度
    s.run(None, {"in_face:0": batch_np})     # ORT 风格：-> [np.ndarray, ...]（ONNX 输出序）
    s.run(batch_np)                          # 直连：uint8(0..255) 或 float32(0..1)，NHWC

后端选择（pipeline_stages._dfm_session 读）：
    DFL_DFM_BACKEND=onnx | torch            默认 onnx（不动现状）
    DFL_DFM_DTYPE=fp32 | tf32 | bf16 | bf16w | auto   只对 torch 生效；**默认 bf16w**
        用户 2026-11 拍板：**默认走 bf16**（精度可接受，速度翻倍），fp32 留作逃生门。
        bf16w（权重也压 bf16）经同进程交替 A/B 实测**没有一档更慢**（416 b1 −12.9%、
        416 b8 −0.6%、384 b1 −41.6%、384 b8 −21.8%），还少占 250MB~1.1GB 显存，
        所以它是默认档；bf16（autocast、权重 fp32）留作对照。auto = 按权重体量选（旧口径）。
        fp32  严格 fp32（关 TF32）——**比 ORT 慢 2.4 倍**，只在要对拍时用；
        tf32  cudnn/cublas 的 TF32（ORACLE：ORT 的 CUDA EP 默认也是 TF32，所以这才是
              与 onnxruntime 同一个数学模式的一档）；
        bf16  **默认**：权重保持 fp32、前向走 torch.autocast(bfloat16)
              （两档 bf16 里精度更好的那个：tail 的 Div/Sqrt/MatMul 留在 fp32）；
        bf16w 权重也压成 bf16（更快/更省显存，误差大一点；见 bench 表）
    DFL_DFM_CL=1                            权重转 channels_last（384 那份 B8 快 13%）
    DFL_DFM_MB=8                            预编译的 batch 上限（超过就分块跑）
    DFL_DFM_PIN=0                           关掉 pinned 出参缓冲（默认开）
    DFL_DFM_GRAPH=1                         **CUDA Graph 重放**（默认 0 = 现状 eager）
        一次 dfm 前向是 156 个 aten 算子提交（416 那张图）+ 1 次流同步。静默态那 156 次
        提交只花 ~5ms，但在生产的 15 条 CPU 车道（cv2/numpy + Python 胶水）旁边抢 GIL，
        它变成 20.3~22.6ms —— 帧被压在 ≥20.3ms。图重放把「156 次提交」换成「1 次
        cudaGraphLaunch」，提交环的 Python 开销直接消失（数值逐位不变：同一份权重、
        同一串算子、同一 device 输入）。
        形状必须固定：按 batch 1/2/4/8 各捕一份（DFL_DFM_GRAPH_B 可改），在 dfm 自己的
        流（self.gstream）上捕获。捕获失败/形状不匹配 → 回退 eager 并打日志（不静默）。
    DFL_DFM_GRAPH_B=1,2,4,8                要捕获的 batch 档（其余档回退 eager）
    DFL_DFM_GRAPH_WARM=3                    捕获前在捕获流上热几次（cuDNN 选算法会变）

数值关系（本机实测，见 DFM_TORCH_BF16_REPORT.md）：torch fp32 与 ONNX fp32 的
max|diff| 是 1e-3 量级 —— 与 ORT 自己换 provider（CUDA vs CPU）的差同量级，
不是「算错了」，是卷积核选择不同；bf16 的那一行报告里单独给。
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from . import dfm_graph as _G

_DTYPES = ("fp32", "tf32", "bf16", "bf16w", "auto")

# 「auto」的门槛：**权重体量**（MB）。这不是拍脑袋的数 —— 它是「这张网是带宽受限还是算力受限」
# 的分界，而且和 _vram_batch_table 里那张表的切分点（900MB）是同一条线。实测：
#   · 1567MB（384 张元英）：batch 8 上 bf16w 21.9 vs bf16 22.5 ms/张，还省 1.1GB 显存 → 选 bf16w
#   ·  514MB（416 阿証）  ：bf16 已经算力受限，压权重只多省 250MB，不值得为它换档 → 选 bf16
_DFM_AUTO_BF16W_MB = 900.0


def _env(name, default=""):
    import os
    v = os.environ.get(name)
    return default if v is None or v == "" else str(v).strip()


def env_dtype(default="bf16w") -> str:
    v = _env("DFL_DFM_DTYPE", default).lower()
    if v in ("fp16", "half"):          # 只做兼容：dfm 在 8GB 卡上 fp16 定不了位，按 bf16 处理
        v = "bf16"
    return v if v in _DTYPES else default


def env_backend(default="torch") -> str:
    v = _env("DFL_DFM_BACKEND", default).lower()
    return "torch" if v in ("torch", "pt", "pytorch", "eager") else "onnx"


def _flag(name, default="0"):
    return str(_env(name, default)).lower() in ("1", "true", "yes", "on")


def _env_batches(name, default="1,2,4,8"):
    """"1,2,4,8" -> {1,2,4,8}；空/写错都退回默认（图只按这些 batch 捕）。"""
    out = set()
    for tok in str(_env(name, default)).replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) > 0:
            out.add(int(tok))
    return out or {1}


class _GraphRun:
    """一个 batch 档的捕获结果：静态入参 + 静态出参 + 图本身。

    三个字段就是「图重放」的全部约定：
      · stat_in  固定的 device 入参（形状 = (b,S,S,3) fp32 NHWC）。每帧只有一次 memcpy 进来。
      · graph    捕获好的图。replay() = **1 次** cudaGraphLaunch，替代 156 次 aten 提交。
      · sout     捕获那一拍分到的出参（来自图的私有显存池）。**每次 replay 都会被覆写**，
                 所以调用方必须在这一拍内取走（_outs 会立刻 D2H 到 pinned）。
      · pin      主机侧 pinned 舞台（np.copyto 进它，再异步 H2D）—— 少一次 pageable 拷贝。
    """

    __slots__ = ("batch", "graph", "stat_in", "sout", "pin", "pin_np",
                 "cap_ms", "vram_mb")

    def __init__(self, batch, graph, stat_in, sout, pin, cap_ms, vram_mb):
        self.batch = int(batch)
        self.graph = graph
        self.stat_in = stat_in
        self.sout = tuple(sout)
        self.pin = pin
        self.pin_np = pin.numpy()
        self.cap_ms = float(cap_ms)
        self.vram_mb = float(vram_mb)


class _Inp:
    """ORT get_inputs() 的替身：_sess_batch_cap 会读 .shape[0]。"""
    __slots__ = ("name", "shape", "type")

    def __init__(self, name, shape):
        self.name = name
        self.shape = list(shape)
        self.type = "tensor(float)"


class DfmTorch:
    """一份 .dfm 的 torch 会话：一份权重 + 每个 batch 一张折叠后的图。"""

    def __init__(self, path, dtype="bf16", max_batch=8, device="cuda",
                 channels_last=False, pin=True, verbose=True, warmup=True):
        self.path = str(path)
        self.name = self.path
        self.dtype = dtype if dtype in _DTYPES else "tf32"
        self.device = torch.device(device)
        self.mb = max(1, int(max_batch))       # 预编译/单次前向的 batch 上限
        self.channels_last = bool(channels_last)
        self.pin = bool(pin)
        # 出参 pinned buffer 的复用开关（默认关：一次前向一条，谁拿到谁拥有，
        # 不存在"下一次前向把上一次的 numpy 覆盖掉"这种坑）。实测复用只省 0.2ms/次，
        # 不值得为它背一个生命周期约定 —— 见 app/tools/bench/dfm_out_bench.py 的表。
        self.pin_reuse = _env("DFL_DFM_PIN_REUSE", "0").lower() in ("1", "true", "yes", "on")
        # 出参搬法：merged（默认，一条 pinned + 一次同步）| legacy（每张量各自分配+同步）。
        # 留 legacy 是为了随时能 A/B 和回退 —— 它不是"死代码"，是逃生门。
        self.out_mode = _env("DFL_DFM_OUT", "merged").lower()
        self._pins = {}
        self.lock = threading.RLock()
        self.nets = {}
        # ---- CUDA Graph（DFL_DFM_GRAPH；默认关）----
        # graph_on 可以在运行时用 set_graph() 翻 —— A/B 台架就是靠它在**同一份会话**上交替
        # （不重载模型 = 不改 cuDNN 状态 = 两个臂比的是同一个计算）。
        self.graph_on = _flag("DFL_DFM_GRAPH", "0")
        self.g_batches = _env_batches("DFL_DFM_GRAPH_B", "1,2,4,8")
        try:
            self.g_warm = max(1, int(_env("DFL_DFM_GRAPH_WARM", "3") or 3))
        except (TypeError, ValueError):
            self.g_warm = 3
        self.gstream = None            # dfm 自己的流（懒建：CPU 会话不该碰 CUDA）
        self._gs = {}                  # batch -> _GraphRun | False（False = 捕过了、失败）
        self.g_fail = {}               # batch -> 失败原因（给日志/诊断）
        self._g_warned = set()         # 只喊一次的「这个 batch 不走图」
        # t_in / t_fwd / t_out 是墙钟三段；t_gpu 是 CUDA event 量的**前向真忙**；
        # t_pin / t_copy / t_dsync 把出参段再拆成「分配 + 拷贝入队 + 等同步」。
        self.stats = {"calls": 0, "faces": 0, "ms": 0.0, "last_ms": 0.0,
                      "compile_ms": 0.0, "t_in": 0.0, "t_fwd": 0.0, "t_out": 0.0,
                      "t_gpu": 0.0, "t_pin": 0.0, "t_copy": 0.0, "t_dsync": 0.0,
                      "t_np": 0.0, "g_calls": 0.0, "g_cap_ms": 0.0}
        self.log = []

        t0 = time.perf_counter()
        self.graph = _G.read_onnx(self.path)
        info = {"size": 0, "nchw": False}
        shp = list(self.graph.input_shape)
        self.nchw = bool(len(shp) >= 4 and shp[1] == 3)
        self.size = int(shp[2] if self.nchw else shp[1]) if len(shp) >= 4 else 256
        self.input_name = self.graph.input_name
        self.input_shape = [(-1 if (i == 0 and not isinstance(shp[0], int)) else int(d))
                            for i, d in enumerate(shp)]
        self.input_shape[0] = -1
        self.output_names = list(self.graph.output_names)
        self._log("读取 .dfm %.0fms · %d 节点 · 权重 %.0fMB · 输入 %dx%d %s"
                  % ((time.perf_counter() - t0) * 1000.0, len(self.graph.nodes),
                     self.graph.weights_mb, self.size, self.size,
                     "NCHW" if self.nchw else "NHWC"))

        if self.dtype == "auto":
            # 按**权重体量**选：大的（≥900MB）带宽受限，压 bf16 既快又省显存；小的算力受限，
            # 压权重不划算（实测见常量注释）。质量两者差 <1 LSB，选谁都不改像素。
            self.dtype = "bf16w" if self.graph.weights_mb >= _DFM_AUTO_BF16W_MB else "bf16"
            self._log("精度 auto → %s（权重 %.0fMB %s %.0fMB）"
                      % (self.dtype, self.graph.weights_mb,
                         "≥" if self.dtype == "bf16w" else "<", _DFM_AUTO_BF16W_MB))
        self._apply_precision_flags()
        # 先把「最大那一档」编出来并把权重搬上卡：其余 batch 的图共用这一份张量对象，
        # 所以显存成本恒等于一份权重（见 dfm_graph.ParamStore 的注释）。
        first = self.mb
        net, st = self._compile(first)
        self.store = _G.ParamStore(net.params)
        self.store.move(self.device, dtype=self._torch_dtype, channels_last=self.channels_last)
        self._shared = {id(v): (v, self.store._buffers[k])
                        for k, v in net.params.items()}
        net.ps = self.store
        self.nets[first] = net
        self.meta = dict(st)
        self._log("编译 batch=%d %.0fms · %d 个真算子（折叠掉 %d）· 权重搬卡 %.0fMB"
                  % (first, self.stats["compile_ms"], st["runtime_nodes"], st["folded"],
                     self.graph.weights_mb * (0.5 if self.dtype == "bf16w" else 1.0)))
        for b in sorted(self._bucket_order()):
            if b not in self.nets:
                self._net(b)
        if warmup:
            self.warm()

    # ------------------------------------------------------------ 精度 / 编译
    def _apply_precision_flags(self):
        """TF32 是**进程级**开关（torch 没有 per-op 的 TF32）。只在加载时设一次并写进日志。

        为什么不每次前向临时开/关：那是全局状态，多线程下 = 互相踩；
        而这个进程里 dfm 就是 torch 唯一的常驻用户（训练是子进程）。
        """
        tf32 = self.dtype in ("tf32", "bf16", "bf16w")
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        self._log("精度 %s · TF32=%s · 权重=%s · channels_last=%s"
                  % (self.dtype, "开" if tf32 else "关",
                     {("fp32"): "fp32", ("tf32"): "fp32", ("bf16"): "fp32",
                      ("bf16w"): "bf16"}[self.dtype],
                     "开" if self.channels_last else "关"))

    @property
    def _torch_dtype(self):
        return torch.bfloat16 if self.dtype == "bf16w" else None

    def _bucket_order(self):
        """默认预编译 1..max_batch（流水线里 dfm 一张到八张都出现过）。"""
        return [b for b in range(1, self.mb + 1)]

    def _compile(self, batch):
        t0 = time.perf_counter()
        net, st = _G.compile_graph(self.graph, int(batch))
        dt = (time.perf_counter() - t0) * 1000.0
        self.stats["compile_ms"] = dt
        return net, st

    def _net(self, batch):
        """取（必要时编译）某个 batch 的图。调用方持锁。"""
        net = self.nets.get(int(batch))
        if net is not None:
            return net
        net, st = self._compile(int(batch))
        # 与第一张图**逐张量比对**过的共享表：同一份 initializer 直接复用卡上那一份，
        # 折叠出来的差异常量（实测只有一个 2 元素的小张量随 batch 变）才单独搬。
        store = _G.ParamStore()
        reused = new = 0
        for k, v in net.params.items():
            hit = self._shared.get(id(v))
            if hit is not None:
                store._buffers[k] = hit[1]
                reused += 1
            else:
                store.register_buffer(
                    k, (v.to(self._torch_dtype) if (self._torch_dtype and v.is_floating_point())
                        else v).to(self.device), persistent=False)
                new += 1
        net.ps = store
        self.nets[int(batch)] = net
        if new:
            self._log("batch=%d 复用 %d 个权重张量、另搬 %d 个"
                      % (int(batch), reused, new))
        return net

    def warm(self):
        """把 1..max_batch 每一档都真跑一次（零张量，不上传）。

        为什么必须预热：每个**新** batch 形状第一次进 cuDNN 都要选算法，实测 batch=3 的
        第一次前向 947ms（稳态 40ms）—— 那个数会污染 _pred_pick_batch 的阶梯和
        流水线第一帧。零张量不影响算法选择（cuDNN 只看形状/布局，不看数值）。
        """
        if self.device.type != "cuda":
            return
        t0 = time.perf_counter()
        per = {}
        with torch.no_grad():
            for b in sorted(self.nets):
                x = torch.zeros((b, self.size, self.size, 3), dtype=torch.float32,
                                device=self.device)
                if self.nchw:
                    x = x.permute(0, 3, 1, 2).contiguous()
                s0 = time.perf_counter()
                self._forward_one(x)
                per[b] = round((time.perf_counter() - s0) * 1000.0, 1)
                # 每档预热完就把 cuDNN 的 workspace 还给驱动。为什么必须有这一句：
                # 关掉 TF32（DFL_DFM_DTYPE=fp32）时 cuDNN 会给 416 的 batch 8 挑一个大
                # workspace 的算法，8 档各留一块 -> torch 的 reserved 涨到 8292MB（比 8GB
                # 卡还大，只能换页），实测 b1 从 85ms 掉到 147ms、b4 从 346ms 掉到 549ms。
                # 那些 workspace 一旦用完就没有 live 引用，还给驱动就不会叠。
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.warm_ms = per
        self.warm_s = (time.perf_counter() - t0)
        # 上面 per[1] 是**这个进程的第一次前向**（含 cuDNN 选算法 / 分配器首次落地），
        # 实测能到 249ms —— 那不是「一张脸要多久」。稳态值单独量一次再报。
        one = per.get(1, 0.0)
        if 1 in self.nets:
            x = torch.zeros((1, self.size, self.size, 3), dtype=torch.float32,
                            device=self.device)
            if self.nchw:
                x = x.permute(0, 3, 1, 2).contiguous()
            s0 = time.perf_counter()
            self._forward_one(x)
            torch.cuda.synchronize()
            one = round((time.perf_counter() - s0) * 1000.0, 2)
        self.one_ms = one
        self._log("预热 batch %s · 稳态单张 %.2f ms（首帧 %.1f）· 合计 %.0f ms"
                  % (sorted(per), one, per.get(1, 0.0), self.warm_s * 1000.0))
        self.stats.update({"calls": 0, "faces": 0, "ms": 0.0, "last_ms": 0.0,
                           "t_in": 0.0, "t_fwd": 0.0, "t_out": 0.0, "t_gpu": 0.0,
                           "t_pin": 0.0, "t_copy": 0.0, "t_dsync": 0.0, "t_np": 0.0,
                           "g_calls": 0.0})
        if self.graph_on and self.device.type == "cuda":
            # 图必须在**生产第一帧之前**捕好：cuDNN 选算法/私有显存池落地都是毫秒级，
            # 落在流水线里就是第一帧的时间。这里顺手把每一档都捕一遍（失败会打日志回退）。
            for _b in sorted(self.g_batches):
                if _b in self.nets:
                    self._graph_for(_b)

    # ------------------------------------------------------------ 前向
    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            t = x
            if t.dtype == torch.uint8:
                t = t.to(self.device).float().mul_(1.0 / 255.0)
            else:
                t = t.to(self.device, dtype=torch.float32)
        else:
            a = np.asarray(x)
            if a.dtype == np.uint8:
                t = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
                t = t.float().mul_(1.0 / 255.0)
            else:
                if not a.flags["C_CONTIGUOUS"]:
                    a = np.ascontiguousarray(a)
                t = torch.from_numpy(a.astype(np.float32, copy=False)).to(self.device)
        if self.nchw and t.dim() == 4 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2)
        return t

    # ------------------------------------------------------------ CUDA Graph
    def _run_net(self, net, xd):
        """eager / 捕获 / 对拍**共用**的一串算子。

        为什么必须共用：图里录下来的算子序列要和 eager 那条**一模一样**（同一份权重、
        同一 device 输入、同一个 cast 位置），对拍才可能是逐位 0 差。任何「图里多一步」
        「图里少一步」都会让对拍变成两个不同的计算。
        """
        if self.dtype == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return net(xd)
        if self.dtype == "bf16w":
            return net(xd.to(torch.bfloat16))
        return net(xd)

    def set_graph(self, on=True, batches=None):
        """运行时开关（A/B 台架靠它在同一份会话上交替，不重载模型 = 不改 cuDNN 状态）。

        注意：**捕获要求此刻没有别的 CUDA 工作在飞**（cudaStreamCapture 的硬约束）。
        生产里在加载期捕（单线程），A/B 台架在两轮之间翻，都不碰这条线；
        别在流水线跑着的时候调 set_graph(True)。
        """
        self.graph_on = bool(on)
        if batches is not None:
            self.g_batches = set(int(b) for b in batches)
        if not self.graph_on:
            return False
        for _b in sorted(self.g_batches):
            if _b in self.nets:
                self._graph_for(_b)          # 预捕（失败会打日志 + 回退 eager）
        return True

    def graph_status(self) -> dict:
        return {"on": bool(self.graph_on),
                "buckets": sorted(self.g_batches),
                "captured": {b: round(g.cap_ms, 1) for b, g in sorted(self._gs.items())
                             if isinstance(g, _GraphRun)},
                "failed": dict(self.g_fail),
                "calls": int(self.stats.get("g_calls") or 0),
                "cap_ms": round(float(self.stats.get("g_cap_ms") or 0.0), 1)}

    def _graph_for(self, b):
        """这一档能不能走图；不能就返回 None（调用方走 eager）。

        「不能」的三种情况都会**喊一次**：非 CUDA / NCHW（布局不同就不是同一个计算）、
        batch 不在捕获档里、捕获失败（以后不再重试，避免每帧一次异常）。
        """
        if not self.graph_on:
            return None
        b = int(b)
        if self.device.type != "cuda" or self.nchw:
            key = "nchw" if self.nchw else "cpu"
            if key not in self._g_warned:
                self._g_warned.add(key)
                self._log("CUDA Graph 不适用于这个会话（%s）—— 全部走 eager"
                          % ("NCHW 布局" if self.nchw else "非 CUDA device"))
            return None
        g = self._gs.get(b)
        if isinstance(g, _GraphRun):
            return g
        if g is False:                       # 捕过了、失败过
            return None
        if b not in self.g_batches:
            if b not in self._g_warned:
                self._g_warned.add(b)
                self._log("CUDA Graph：batch=%d 不在 DFL_DFM_GRAPH_B=%s 里 → 这一档走 eager"
                          % (b, ",".join(str(x) for x in sorted(self.g_batches))))
            return None
        try:
            g = self._capture(b)
        except Exception as e:                             # noqa: BLE001
            self._gs[b] = False
            self.g_fail[b] = "%s: %s" % (type(e).__name__, str(e)[:300])
            self._log("★ CUDA Graph 捕获失败 batch=%d（%s）→ 回退 eager（这一档不再试）"
                      % (b, self.g_fail[b]))
            return None
        self._gs[b] = g
        return g

    def _capture(self, b):
        """在 dfm 自己的流上捕一份 b 张的图。返回 _GraphRun（失败抛异常，由 _graph_for 兜）。"""
        if self.gstream is None:
            self.gstream = torch.cuda.Stream(device=self.device)
        net = self._net(int(b))
        s = self.gstream
        cur = torch.cuda.current_stream()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        alloc0 = torch.cuda.memory_allocated()
        stat_in = torch.zeros((int(b), self.size, self.size, 3),
                              dtype=torch.float32, device=self.device)
        pin = torch.empty((int(b), self.size, self.size, 3), dtype=torch.float32,
                          pin_memory=True)
        # 捕获前热几次：**同一形状第一次进 cuDNN 会挑算法**（实测 batch=3 首帧 947ms），
        # 在捕获流上热，才让图里录的是稳态那套 kernel。
        s.wait_stream(cur)
        with torch.cuda.stream(s), torch.no_grad():
            ys = None
            for _ in range(self.g_warm):
                ys = self._run_net(net, stat_in)
            del ys
        torch.cuda.synchronize()
        cap = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cap, stream=s):
            with torch.no_grad():
                _ys = self._run_net(net, stat_in)
                # 出参的 bf16→fp32 也录进图里：_outs 每帧原本要再提交 3 个 cast kernel
                # （+3 次分配）。放进图 = 每次 replay 少 3 次提交，且逐位不变
                # （同一个 aten cast、同一份 bf16 输入）。
                sout = tuple(y if y.dtype == torch.float32 else y.float() for y in _ys)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        vram = (torch.cuda.memory_allocated() - alloc0) / 1048576.0
        gr = _GraphRun(int(b), cap, stat_in, sout, pin, dt, vram)
        self.stats["g_cap_ms"] += dt
        self._log("CUDA Graph 捕获 batch=%d %.0fms（热 %d 次）· 静态入参 %s · 出参 %s ·"
                  " 设备显存 +%.0fMB"
                  % (int(b), dt, self.g_warm, tuple(stat_in.shape),
                     [tuple(t.shape) for t in gr.sout], vram))
        return gr

    def _forward_graph(self, x, gr):
        """图重放一次：主机舞台 memcpy → 1 次 H2D → **1 次** cudaGraphLaunch → D2H。

        提交环上只剩 5~6 次 CUDA API（原来是 156 次 aten 提交 + 1 次流同步）—— 这才是
        「生产态 20.3ms 的墙」的来源（静默态那 156 次只要 5ms，抢的是 GIL 不是驱动）。
        """
        b = gr.batch
        t0 = time.perf_counter()
        want = tuple(gr.stat_in.shape)
        if isinstance(x, torch.Tensor):
            if tuple(x.shape) != want:
                raise ValueError("图入参形状不匹配: %s vs %s" % (tuple(x.shape), want))
            if x.is_cuda:
                src = x
            else:
                gr.pin.copy_(x)                  # CPU 张量：dtype 转换落进 pinned 舞台
                if x.dtype == torch.uint8:       # 和 _to_device 同一口径：0..255 → 0..1
                    gr.pin.mul_(1.0 / 255.0)
                src = gr.pin
        else:
            a = x
            if tuple(a.shape) != want:
                raise ValueError("图入参形状不匹配: %s vs %s" % (tuple(a.shape), want))
            if a.dtype == np.uint8:
                # 乘 1/255（不是除 255）：_to_device 那条路就是 mul_(1/255)，逐位同口径
                np.multiply(a, np.float32(1.0 / 255.0), out=gr.pin_np, casting="unsafe")
            else:
                np.copyto(gr.pin_np, a, casting="unsafe")
            src = gr.pin
        t1 = time.perf_counter()
        ev0 = ev1 = None
        if self.device.type == "cuda":
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        if isinstance(src, torch.Tensor) and src.is_cuda:
            # 入参若是别的流上算出来的，图流要等它（否则读到没写完的 buffer）
            self.gstream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.gstream):
            gr.stat_in.copy_(src, non_blocking=True)
            if ev0 is not None:
                ev0.record()                     # 事件夹住**只有图**那段（不含 H2D）
            gr.graph.replay()
            t2 = time.perf_counter()
            if ev1 is not None:
                ev1.record()
            outs = self._outs(gr.sout)
        t3 = time.perf_counter()
        if ev0 is not None:
            try:
                self.stats["t_gpu"] += float(ev0.elapsed_time(ev1))
            except Exception:                              # noqa: BLE001
                pass
        self.stats["calls"] += 1
        self.stats["g_calls"] += 1
        self.stats["faces"] += b
        self.stats["last_ms"] = (t3 - t0) * 1000.0
        self.stats["ms"] += self.stats["last_ms"]
        self.stats["t_in"] += (t1 - t0) * 1000.0
        self.stats["t_fwd"] += (t2 - t1) * 1000.0
        self.stats["t_out"] += (t3 - t2) * 1000.0
        return outs

    def _out(self, t):
        """单个张量 → numpy（旧口径：自己的 pinned + 自己的同步）。_outs 的兼容入口。"""
        return self._outs((t,))[0]

    def _outs(self, ys):
        """一整批出参 → numpy：**一次** pinned 分配 + **一次**流同步。

        为什么不是一个张量一张量地搬（旧写法见下），实测（app/tools/bench/dfm_out_bench.py
        与 fused 150 帧的 t_pin/t_copy/t_dsync）：

          · 旧写法每个张量各自 cudaHostAlloc + copy_ + current_stream().synchronize()。
            416 一次前向有 3 个出参 → 3 次分配、3 次**全流**同步；
          · 第 1 次同步确实要等前向算完（那是必须的，省不掉）；第 2、3 次同步
            **一分钱数据都不搬**，纯粹是 WDDM 的提交/回环，白等；
          · 拼成一条 pinned buffer 之后只剩那一次必须的同步，且 numpy 视图是
            zero-copy（.numpy() 不复制 26MB）。

        精度：这里**不改任何 dtype/布局** —— 只是把「3 条到 3 块 host 内存」改成
        「3 条到 1 块的 3 段」，逐位不变。gpu 侧仍然先 float() 到 fp32（输出本来
        就是 fp32，实测 .float() 是 no-op）。

        t_pin  = detach + dtype 转换 + 分配（CPU 侧，几微秒）
        t_copy = 所有 copy_ 的入队（异步，正常应当远小于一次拷贝的传输时间）
        t_dsync = 那**一次**同步等的时间（含前向尾巴 —— 它就是 t_out 的大头）
        """
        t0 = time.perf_counter()
        if self.pin and self.out_mode == "legacy":
            out = []
            for y in ys:
                t = y.detach()
                if t.dtype != torch.float32:
                    t = t.float()
                buf = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
                t1 = time.perf_counter()
                buf.copy_(t, non_blocking=True)
                t2 = time.perf_counter()
                torch.cuda.current_stream().synchronize()
                t3 = time.perf_counter()
                self.stats["t_pin"] += (t1 - t0) * 1000.0
                self.stats["t_copy"] += (t2 - t1) * 1000.0
                self.stats["t_dsync"] += (t3 - t2) * 1000.0
                t0 = t3
                out.append(buf.numpy())
            return out
        tens = []
        for y in ys:
            t = y.detach()
            if t.dtype != torch.float32:
                t = t.float()
            tens.append(t)
        if not (self.pin and tens and tens[0].is_cuda):
            # 不做 pinned 的老路（pin=False）：保持原语义，出参 dtype 不强制 fp32
            out = []
            for y in ys:
                out.append(y.detach().cpu().numpy())
            self.stats["t_dsync"] += (time.perf_counter() - t0) * 1000.0
            return out
        tot = 0
        for t in tens:
            tot += int(t.numel())
        key = tuple(tuple(t.shape) for t in tens)
        buf = self._pins.get(key) if self.pin_reuse else None
        if buf is None:
            buf = torch.empty(tot, dtype=torch.float32, pin_memory=True)
            if self.pin_reuse:
                self._pins[key] = buf
        t1 = time.perf_counter()
        views = []
        off = 0
        for t in tens:
            n = int(t.numel())
            v = buf[off:off + n].view(t.shape)
            v.copy_(t, non_blocking=True)
            off += n
            views.append(v)
        t2 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        t3 = time.perf_counter()
        arrs = [v.numpy() for v in views]
        t4 = time.perf_counter()
        self.stats["t_pin"] += (t1 - t0) * 1000.0
        self.stats["t_copy"] += (t2 - t1) * 1000.0
        self.stats["t_dsync"] += (t3 - t2) * 1000.0
        self.stats["t_np"] += (t4 - t3) * 1000.0
        return arrs

    def run(self, output_names=None, input_feed=None):
        """两种调用方式：

            run(None, {"in_face:0": x})   # 和 onnxruntime 一模一样（_dfm_run_batch 用的）
            run(x)                        # 直连；x = (N,S,S,3)，uint8(0..255) 或 float32(0..1)
        """
        if input_feed is None:
            x = output_names
        else:
            names = self.output_names if not output_names else list(output_names)
            feeds = input_feed
            x = feeds.get(self.input_name)
            if x is None:
                x = next(iter(feeds.values()))
        return self.forward(x)

    def forward(self, x):
        a = x
        n = int(a.shape[0]) if isinstance(a, torch.Tensor) else int(np.shape(a)[0])
        with self.lock:
            outs = []
            i = 0
            while i < n:
                b = min(n - i, self.mb)
                part = a[i:i + b] if i else a[:b]
                outs.append(self._forward_one(part))
                i += b
        if len(outs) == 1:
            return outs[0]
        return [np.concatenate([o[k] for o in outs], 0) for k in range(len(outs[0]))]

    def _forward_one(self, x):
        # 图这一档在**搬上卡之前**分流：省的正是「156 次 aten 提交」那段主机时间。
        # 形状不匹配/没捕/捕失败 → _graph_for 返回 None，下面原样走 eager（绝不静默换算法）。
        _b0 = int(x.shape[0]) if isinstance(x, torch.Tensor) else int(np.shape(x)[0])
        if self.graph_on and _b0 and _b0 in self.g_batches:
            _gr = self._graph_for(_b0)
            if _gr is not None and _gr.batch == _b0:
                return self._forward_graph(x, _gr)
        t0 = time.perf_counter()
        xd = self._to_device(x)
        b = int(xd.shape[0])
        net = self._net(b)
        # CUDA event 夹住**只有前向**那一段 —— 这是「GPU 真忙」的唯一直接证据：
        # 不看它，t_fwd 只是入队时间、t_out 里混着前向尾巴，两个数都会说谎。
        ev0 = ev1 = None
        if self.device.type == "cuda":
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
        t1 = time.perf_counter()
        with torch.no_grad():
            ys = self._run_net(net, xd)
        t2 = time.perf_counter()
        if ev1 is not None:
            ev1.record()
        outs = self._outs(ys)
        t3 = time.perf_counter()
        if ev0 is not None:
            try:
                self.stats["t_gpu"] += float(ev0.elapsed_time(ev1))
            except Exception:                              # noqa: BLE001
                pass
        self.stats["calls"] += 1
        self.stats["faces"] += b
        self.stats["last_ms"] = (t3 - t0) * 1000.0
        self.stats["ms"] += self.stats["last_ms"]
        self.stats["t_in"] += (t1 - t0) * 1000.0
        self.stats["t_fwd"] += (t2 - t1) * 1000.0
        self.stats["t_out"] += (t3 - t2) * 1000.0
        return outs

    # ------------------------------------------------------------ ORT 兼容面
    def get_inputs(self):
        return [_Inp(self.input_name, self.input_shape)]

    def get_outputs(self):
        return [_Inp(nm, [-1]) for nm in self.output_names]

    def get_providers(self):
        return ["TorchExecutionProvider"]

    def max_batch(self):                       # _sess_batch_cap 认这个名字（callable → 走这条）
        return int(self.mb)

    # ------------------------------------------------------------ 诊断
    def _log(self, msg):
        line = "[dfm-torch] " + msg
        self.log.append(line)
        print(line, flush=True)

    def vram_mb(self) -> dict:
        """这个会话在卡上真正占了多少（torch 自己的账本，不是 nvidia-smi）。"""
        try:
            return {"alloc_mb": round(torch.cuda.max_memory_allocated() / 1048576.0, 1),
                    "reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576.0, 1),
                    "weights_mb": round(sum(v.numel() * v.element_size()
                                            for v in self.store._buffers.values()) / 1048576.0, 1)}
        except Exception:                                  # noqa: BLE001
            return {}

    def per_face_ms(self) -> float:
        return (self.stats["ms"] / self.stats["faces"]) if self.stats["faces"] else 0.0

    def describe(self) -> str:
        return "torch-%s/%s/b%d%s" % (self.device.type, self.dtype, self.mb,
                                      "+graph" if (self.graph_on and self._gs) else "")

    def close(self):
        with self.lock:
            self._gs.clear()
            self.gstream = None
            self.nets.clear()
            try:
                self.store._buffers.clear()
            except Exception:                              # noqa: BLE001
                pass
            self._shared.clear()
            self._pins.clear()
            self.graph.close()
            try:
                torch.cuda.empty_cache()
            except Exception:                              # noqa: BLE001
                pass


_CACHE = {}
_CACHE_LOCK = threading.RLock()


def load(dfm_path, dtype=None, max_batch=None, channels_last=None, pin=None,
         device="cuda", cache=True):
    """加载一份 .dfm 成 torch 会话（同路径 + 同精度只建一次）。"""
    dtype = env_dtype() if dtype is None else dtype
    if max_batch is None:
        max_batch = int(_env("DFL_DFM_MB", "8") or 8)
    if channels_last is None:
        channels_last = _env("DFL_DFM_CL", "0").lower() in ("1", "true", "yes", "on")
    if pin is None:
        pin = _env("DFL_DFM_PIN", "1").lower() not in ("0", "false", "no", "off")
    key = (str(dfm_path), dtype, int(max_batch), bool(channels_last), str(device))
    if cache:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                return hit
    s = DfmTorch(dfm_path, dtype=dtype, max_batch=max_batch, device=device,
                 channels_last=channels_last, pin=pin)
    if cache:
        with _CACHE_LOCK:
            _CACHE[key] = s
    return s


def unload(path=None):
    """卸掉（默认全部）会话，把显存还回去。"""
    with _CACHE_LOCK:
        keys = [k for k in _CACHE if path is None or k[0] == str(path)]
        for k in keys:
            try:
                _CACHE.pop(k).close()
            except Exception:                              # noqa: BLE001
                pass
        return len(keys)
