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
        self.lock = threading.RLock()
        self.nets = {}
        self.stats = {"calls": 0, "faces": 0, "ms": 0.0, "last_ms": 0.0,
                      "compile_ms": 0.0, "t_in": 0.0, "t_fwd": 0.0, "t_out": 0.0}
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
                           "t_in": 0.0, "t_fwd": 0.0, "t_out": 0.0})

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

    def _out(self, t):
        """一个输出张量 → numpy（pinned 缓冲 + 异步拷贝：pageable 下 27MB 要 9ms）。"""
        t = t.detach()
        if t.dtype != torch.float32:
            t = t.float()
        if self.pin and t.is_cuda:
            buf = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
            buf.copy_(t, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return buf.numpy()
        return t.cpu().numpy()

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
        t0 = time.perf_counter()
        xd = self._to_device(x)
        b = int(xd.shape[0])
        net = self._net(b)
        t1 = time.perf_counter()
        with torch.no_grad():
            if self.dtype == "bf16":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ys = net(xd)
            elif self.dtype == "bf16w":
                ys = net(xd.to(torch.bfloat16))
            else:
                ys = net(xd)
        t2 = time.perf_counter()
        outs = [self._out(y) for y in ys]
        t3 = time.perf_counter()
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
        return "torch-%s/%s/b%d" % (self.device.type, self.dtype, self.mb)

    def close(self):
        with self.lock:
            self.nets.clear()
            try:
                self.store._buffers.clear()
            except Exception:                              # noqa: BLE001
                pass
            self._shared.clear()
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
