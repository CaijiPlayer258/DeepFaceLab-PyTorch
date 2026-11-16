# -*- coding: utf-8 -*-
"""`.dfm`(ONNX ModelProto) -> torch nn.Module：静态折叠 + 代码生成。

这个文件是 `app/tools/bench/dfm_torch_graph.py` 的产品化版本，两处关键差别：

  1) **不依赖 onnx 包**。4.1 的 server venv 里没有 onnx（torch 有、onnxruntime 有），
     而 dfm 就是一份 ONNX protobuf —— 所以这里自己按 protobuf 线格式扫（mmap + 只跳不读
     raw_data），和 `backend_merge/vram.py:_dfm_info` 同一套路。实测读 1.6GB 的
     `张元英.dfm` 只要 0.2~0.4s，比 `onnx.load`（2.3s）还快，而且不额外占一份内存。
  2) **权重只有一份**。同一份权重要在 batch 1..8 各编一张图（折叠要把 batch 定死），
     如果每张图各自 `register_buffer` 一份权重，8 张图 = 8 份 514MB/1567MB → 直接顶爆
     8GB 卡。这里把常量放进 `ParamStore`，各 batch 的 forward 都引用**同一个** store，
     所以「编译 N 张图」的显存成本恒等于「一份权重」。

为什么是「折叠后代码生成」而不是逐节点解释器：
  · 解释器的 Python 开销（416 那张图 1065 个节点）会把 torch 的收益全吃掉；
  · 折叠（Shape/Reshape/Mod/Slice/Pad/Range/Gather/ConstantOfShape…）在 batch 定死之后
    能整段算成常量 —— 416：1065 节点 → 156 个真算子；384：183 → 183（tf2onnx 那份本来就干净）；
  · 生成的 forward 是纯 aten 序列，将来上 torch.compile 也是一张干净的图。

约定（和 dfm 的 ONNX 原样一致，别改）：
  · 输入/输出保持 ONNX 的布局语义（两份样本都是 NHWC `[B,H,W,3]`）；
  · 遇到没实现的算子**直接报错并打印节点**，不静默回退 —— 静默回退等于给出一份
    不知道在算什么的时间。
"""
from __future__ import annotations

import mmap
import os
import struct
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OnnxGraph", "read_onnx", "compile_graph", "ParamStore", "input_info"]


# ============================================================ protobuf 线格式
# 只认 ONNX 里 dfm 用得到的那几个 message：
#   ModelProto(7=graph) · GraphProto(1=node, 5=initializer, 11=input, 12=output)
#   NodeProto(1=input, 2=output, 3=name, 4=op_type, 5=attribute)
#   AttributeProto(1=name, 2=f, 3=i, 4=s, 5=t, 7=floats, 8=ints, 9=strings, 20=type)
#   TensorProto(1=dims, 2=data_type, 4=float_data, 5=int32_data, 7=int64_data,
#               8=name, 9=raw_data, 10=double_data, 13=external_data, 14=data_location)
#   ValueInfoProto(1=name, 2=type) · TypeProto(1=tensor_type)
#   TypeProto.Tensor(1=elem_type, 2=shape) · TensorShapeProto.Dimension(1=dim_value, 2=dim_param)

def _varint(buf, i, end):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def _fields(buf, i, end):
    """遍历一个 message 的字段：(fno, wire, value)。

    wire=0 → int；wire=2 → (start, n)（**不拷贝**，raw_data 只跳不读）；
    wire=5 → 4 字节 float 的位模式；wire=1 → 8 字节（double）的位模式。
    """
    while i < end:
        key, i = _varint(buf, i, end)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i, end)
            yield fno, wt, v
        elif wt == 2:
            n, i = _varint(buf, i, end)
            yield fno, wt, (i, n)
            i += n
        elif wt == 5:
            yield fno, wt, struct.unpack_from("<f", buf, i)[0]
            i += 4
        elif wt == 1:
            yield fno, wt, struct.unpack_from("<d", buf, i)[0]
            i += 8
        else:
            raise ValueError("protobuf wire type %d 不支持（offset %d）" % (wt, i))


def _s64(v):
    """ONNX 的 int64 都是**有符号**的，而 protobuf 的 varint 读出来是无符号 64 位。

    少了这一步，axes = [-1] 会变成 18446744073709551615 —— 后果不是报错而是
    「torch.mean(dim=(1.8e19,)) 直接抛」或更糟：Slice(starts=-1) 静默取到别的元素。
    """
    v = int(v)
    return v - (1 << 64) if v >= (1 << 63) else v


def _ints(buf, wt, v):
    """repeated int64 字段：可能是单值（wire 0）也可能是 packed（wire 2）。"""
    if wt == 0:
        return [_s64(v)]
    out = []
    i, n = v[0], v[1]
    end = i + n
    while i < end:
        x, i = _varint(buf, i, end)
        out.append(_s64(x))
    return out


def _name(buf, v):
    i, n = v
    return bytes(buf[i:i + n]).decode("utf-8", "replace")


# ------------------------------------------------------------ TensorProto
_TORCH_DT = {
    1: torch.float32, 2: torch.uint8, 3: torch.int8, 4: torch.uint16,
    5: torch.int16, 6: torch.int32, 7: torch.int64, 9: torch.bool,
    10: torch.float16, 11: torch.float64, 12: torch.uint32, 13: torch.uint64,
    16: torch.bfloat16,
}
_NP_DT = {   # onnx data_type -> numpy dtype 名（只在没有 raw_data 时用来拼数组）
    1: "f4", 2: "u1", 3: "i1", 4: "u2", 5: "i2", 6: "i4", 7: "i8",
    9: "?", 10: "f2", 11: "f8", 12: "u4", 13: "u8",
}


def _tensor_from_proto(buf, s, e):
    """TensorProto → torch.Tensor。

    raw_data 用 `torch.frombuffer` 直接**映射**到 mmap（零拷贝），再由调用方 `.to(device)`
    搬一次；没有 raw_data（老式 float_data/int64_data）才拼。
    """
    import numpy as np
    dims, dt, raw = [], 1, None
    name = ""
    lists = {4: [], 5: [], 7: [], 10: []}
    location = 0
    for fno, wt, v in _fields(buf, s, e):
        if fno == 1:
            dims.extend(_ints(buf, wt, v))
        elif fno == 2 and wt == 0:
            dt = int(v)
        elif fno == 8 and wt == 2:
            name = _name(buf, v)
        elif fno == 9 and wt == 2:
            raw = v
        elif fno in lists:
            if wt == 2:
                i, n = v
                code = {4: "f4", 5: "i4", 7: "i8", 10: "f8"}[fno]
                lists[fno].extend(np.frombuffer(bytes(buf[i:i + n]),
                                                dtype="<" + code).tolist())
            else:
                lists[fno].append(v)
        elif fno == 14 and wt == 0:
            location = int(v)
    if location == 1:
        raise RuntimeError("initializer %r 用了 external data（.dfm 不该有），读不了" % (name,))
    shape = [int(d) for d in dims]
    tdt = _TORCH_DT.get(int(dt))
    if tdt is None:
        raise RuntimeError("initializer %r 的 data_type=%d 不支持" % (name, dt))
    if raw is not None:
        i, n = raw
        if n == 0:
            return torch.zeros(shape, dtype=tdt)
        count = 1
        for d in shape:
            count *= int(d)
        try:
            with warnings.catch_warnings():     # 「buffer 不可写」是只读 mmap 的必然，不是问题
                warnings.simplefilter("ignore")
                t = torch.frombuffer(buf, dtype=tdt, count=count, offset=int(i))
        except TypeError:                       # 老 torch 不支持 offset：退一步拷一份
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = torch.frombuffer(memoryview(buf)[i:i + n], dtype=tdt, count=count)
        return t.reshape(shape)
    code = {1: 4, 6: 5, 7: 7, 11: 10}.get(int(dt))
    vals = lists.get(code) or []
    if not vals:
        return torch.zeros(shape, dtype=tdt)
    arr = np.asarray(vals, dtype="<" + _NP_DT[int(dt)]).reshape(shape)
    return torch.from_numpy(arr.copy())


# ------------------------------------------------------------ AttributeProto
def _attrs(buf, s, e):
    """AttributeProto → (name, value)；value 是 int/float/str/list/Tensor。"""
    name = ""
    vals = {}
    for fno, wt, v in _fields(buf, s, e):
        if fno == 1 and wt == 2:
            name = _name(buf, v)
        elif fno == 2 and wt == 5:
            vals["f"] = float(v)
        elif fno == 3 and wt == 0:
            vals["i"] = _s64(v)
        elif fno == 4 and wt == 2:
            i, n = v
            vals["s"] = bytes(buf[i:i + n]).decode("utf-8", "replace")
        elif fno == 5 and wt == 2:
            i, n = v
            vals["t"] = _tensor_from_proto(buf, i, i + n)
        elif fno == 7:                       # floats（packed 或重复）
            if wt == 5:
                vals.setdefault("floats", []).append(float(v))
        elif fno == 8:                       # ints
            vals.setdefault("ints", []).extend(_ints(buf, wt, v))
    if "t" in vals:
        return name, vals["t"]
    if "i" in vals:
        return name, vals["i"]
    if "f" in vals:
        return name, vals["f"]
    if "s" in vals:
        return name, vals["s"]
    if "ints" in vals:
        return name, list(vals["ints"])
    if "floats" in vals:
        return name, list(vals["floats"])
    return name, None


# ------------------------------------------------------------ ValueInfoProto
def _value_info(buf, s, e):
    """ValueInfoProto → (name, [dim int 或 None])；拿不到 shape 时 dims 是 None。"""
    name, dims = "", None
    for fno, wt, v in _fields(buf, s, e):
        if fno == 1 and wt == 2:
            name = _name(buf, v)
        elif fno == 2 and wt == 2:                   # TypeProto
            i0, n0 = v
            for f2, w2, v2 in _fields(buf, i0, i0 + n0):
                if f2 != 1 or w2 != 2:               # tensor_type
                    continue
                i1, n1 = v2
                for f3, w3, v3 in _fields(buf, i1, i1 + n1):
                    if f3 != 2 or w3 != 2:           # shape
                        continue
                    i2, n2 = v3
                    dims = []
                    for f4, w4, v4 in _fields(buf, i2, i2 + n2):
                        if f4 != 1 or w4 != 2:       # dim
                            continue
                        i3, n3 = v4
                        dv = None
                        for f5, w5, v5 in _fields(buf, i3, i3 + n3):
                            if f5 == 1 and w5 == 0:
                                dv = int(v5)
                        dims.append(dv)
    return name, dims


# ============================================================ 图
class OnnxGraph:
    """一份 .dfm 的静态描述（节点 + 常量），不含设备相关的东西。"""

    __slots__ = ("path", "nodes", "const", "input_name", "input_shape",
                 "output_names", "producer", "mm", "n_init", "weights_mb")

    def __init__(self):
        self.path = ""
        self.nodes = []          # [{"op","name","ins","outs","at"}]
        self.const = {}          # name -> torch.Tensor（CPU；raw_data 直接 mmap 映射）
        self.input_name = ""
        self.input_shape = []
        self.output_names = []
        self.producer = ""
        self.mm = None
        self.n_init = 0
        self.weights_mb = 0.0

    def detach_buf(self):
        """把常量从 mmap 上摘下来（拷成独立内存），然后关掉 mmap。

        只在「就要常量、不再省这一次拷贝」的路径上调（bench / 反查）。
        """
        self.const = {k: v.clone() for k, v in self.const.items()}
        self.close()

    def close(self):
        mm, self.mm = self.mm, None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                # 还有常量视图挂在 mmap 上（正常：session 活多久它就在多久）——
                # 这时不关，交给 GC。别为了关它把 1.5GB 常量再拷一遍。
                self.mm = mm

    def summary(self) -> dict:
        return {"path": self.path, "producer": self.producer,
                "nodes": len(self.nodes), "init": self.n_init,
                "weights_mb": round(self.weights_mb, 1),
                "input": self.input_name, "shape": list(self.input_shape),
                "outputs": list(self.output_names)}


def _tensor_name(buf, s, e):
    for fno, wt, v in _fields(buf, s, e):
        if fno == 8 and wt == 2:
            return _name(buf, v)
    return ""


def _node(buf, s, e):
    ins, outs, name, op, at = [], [], "", "", {}
    for fno, wt, v in _fields(buf, s, e):
        if fno == 1 and wt == 2:
            ins.append(_name(buf, v))
        elif fno == 2 and wt == 2:
            outs.append(_name(buf, v))
        elif fno == 3 and wt == 2:
            name = _name(buf, v)
        elif fno == 4 and wt == 2:
            op = _name(buf, v)
        elif fno == 5 and wt == 2:
            k, val = _attrs(buf, v[0], v[0] + v[1])
            if k:
                at[k] = val
    return {"op": op, "name": name, "ins": ins, "outs": outs, "at": at}


def read_onnx(path) -> OnnxGraph:
    """扫一遍 .dfm（mmap，raw_data 零拷贝映射）→ OnnxGraph。"""
    g = OnnxGraph()
    g.path = str(path)
    with open(g.path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    g.mm = mm
    graph = None
    for fno, wt, v in _fields(mm, 0, len(mm)):
        if fno == 2 and wt == 2 and not g.producer:
            g.producer = _name(mm, v)
        elif fno == 7 and wt == 2:
            graph = v
            break
    if graph is None:
        g.close()
        raise RuntimeError("不是 ONNX（找不到 ModelProto.graph）：%s" % g.path)
    gs, gn = graph[0], graph[0] + graph[1]
    nbytes = 0
    for fno, wt, v in _fields(mm, gs, gn):
        if fno == 1 and wt == 2:                     # node
            g.nodes.append(_node(mm, v[0], v[0] + v[1]))
        elif fno == 5 and wt == 2:                   # initializer
            i, n = v
            g.const[_tensor_name(mm, i, i + n)] = _tensor_from_proto(mm, i, i + n)
            nbytes += int(n)
            g.n_init += 1
        elif fno == 11 and wt == 2 and not g.input_name:      # input
            nm, dims = _value_info(mm, v[0], v[0] + v[1])
            g.input_name, g.input_shape = nm, (dims or [])
        elif fno == 12 and wt == 2:                  # output
            nm, _dims = _value_info(mm, v[0], v[0] + v[1])
            g.output_names.append(nm)
    g.weights_mb = nbytes / 1048576.0
    return g


def input_info(path):
    """只读图头：{"size","nchw","name","shape","outputs"} —— 建 session 之前就要知道边长。"""
    g = read_onnx(path)
    try:
        shp = list(g.input_shape)
        nchw = len(shp) >= 4 and shp[1] == 3
        size = int(shp[2] if nchw else shp[1]) if len(shp) >= 4 else 256
        return {"size": size, "nchw": bool(nchw), "name": g.input_name,
                "shape": shp, "outputs": list(g.output_names)}
    finally:
        g.close()


# ============================================================ 单算子（折叠 + 形状推断 + 对拍共用）
# 同一份语义三用：常量折叠（CPU 真张量）、形状推断（meta 张量）、bench 对拍。
_CAST_DT = {
    1: torch.float32, 2: torch.uint8, 3: torch.int8, 4: torch.uint16, 5: torch.int16,
    6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64,
    12: torch.uint32, 13: torch.uint64, 16: torch.bfloat16,
}

_RUNTIME_OPS = {
    "Conv", "ConvTranspose", "MatMul", "Gemm", "Add", "Sub", "Mul", "Div",
    "LeakyRelu", "Relu", "Sigmoid", "Tanh", "PReLU", "Clip", "Softmax",
    "Concat", "Transpose", "Reshape", "Pad", "Slice", "DepthToSpace",
    "ReduceMean", "ReduceSum", "Sqrt", "Reciprocal", "Pow", "Cast",
    "BatchNormalization", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Resize", "Upsample", "Unsqueeze", "Squeeze", "Flatten", "Identity",
    "Shape", "Gather", "ConstantOfShape", "Range", "Mod", "Constant",
}
_FOLD_ONLY = {"Shape", "ConstantOfShape", "Range", "Mod"}


def _slice_spec(starts, ends, axes, steps, rank):
    axes = list(range(len(starts))) if axes is None else list(axes)
    steps = [1] * len(starts) if steps is None else list(steps)
    idx = [slice(None)] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = int(a) % rank
        e = int(e)
        idx[a] = slice(int(s), None if e >= 2 ** 62 else e, int(st))
    return tuple(idx)


def run_op(op, ins, at):
    """执行一个节点。ins 是已就绪的张量（可能是 meta 张量）。"""
    if op == "Constant":
        for k in ("value", "value_float", "value_int", "value_floats", "value_ints"):
            v = at.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                return v
            if isinstance(v, list):
                return torch.tensor(v)
            return torch.tensor(v)
        raise RuntimeError("Constant 节点缺少 value")
    if op == "Conv":
        x, w = ins[0], ins[1]
        b = ins[2] if len(ins) > 2 else None
        pads = at.get("pads", [0, 0, 0, 0])
        st = at.get("strides", [1, 1])
        dl = at.get("dilations", [1, 1])
        grp = at.get("group", 1)
        if pads[0] == pads[2] and pads[1] == pads[3]:
            pad = (pads[0], pads[1])
        else:
            x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
            pad = (0, 0)
        return F.conv2d(x, w, b, stride=tuple(st), padding=pad,
                        dilation=tuple(dl), groups=grp)
    if op == "ConvTranspose":
        x, w = ins[0], ins[1]
        b = ins[2] if len(ins) > 2 else None
        return F.conv_transpose2d(x, w, b, stride=tuple(at.get("strides", [1, 1])),
                                  padding=tuple(at.get("pads", [0, 0, 0, 0])[:2]),
                                  output_padding=tuple(at.get("output_padding", [0, 0])),
                                  groups=at.get("group", 1),
                                  dilation=tuple(at.get("dilations", [1, 1])))
    if op == "MatMul":
        return torch.matmul(ins[0], ins[1])
    if op == "Gemm":
        a, b = ins[0], ins[1]
        if at.get("transA"):
            a = a.t()
        if at.get("transB"):
            b = b.t()
        y = at.get("alpha", 1.0) * (a @ b)
        if len(ins) > 2:
            y = y + at.get("beta", 1.0) * ins[2]
        return y
    if op == "Add":
        return ins[0] + ins[1]
    if op == "Sub":
        return ins[0] - ins[1]
    if op == "Mul":
        return ins[0] * ins[1]
    if op == "Div":
        return ins[0] / ins[1]
    if op == "LeakyRelu":
        return F.leaky_relu(ins[0], at.get("alpha", 0.01))
    if op == "Relu":
        return F.relu(ins[0])
    if op == "Sigmoid":
        return torch.sigmoid(ins[0])
    if op == "Tanh":
        return torch.tanh(ins[0])
    if op == "Clip":
        lo, hi = at.get("min"), at.get("max")
        if lo is None and len(ins) > 1:
            lo = ins[1]
        if hi is None and len(ins) > 2:
            hi = ins[2]
        return torch.clamp(ins[0],
                           None if lo is None else float(lo),
                           None if hi is None else float(hi))
    if op == "Softmax":
        return torch.softmax(ins[0], dim=int(at.get("axis", -1)))
    if op == "Concat":
        return torch.cat(list(ins), dim=int(at["axis"]))
    if op == "Transpose":
        return ins[0].permute(*at["perm"])
    if op == "Reshape":
        shape = [int(v) for v in ins[1].tolist()]
        if at.get("allowzero"):
            return ins[0].reshape(shape)
        cur = list(ins[0].shape)
        for i, s in enumerate(shape):
            if s == 0:
                shape[i] = cur[i]
        return ins[0].reshape(shape)
    if op == "Pad":
        pads = ([int(v) for v in at["pads"]] if "pads" in at
                else [int(v) for v in ins[1].tolist()])
        mode = at.get("mode", "constant")
        val = 0.0
        if "value" in at:
            val = float(at["value"])
        elif len(ins) > 2:
            val = float(ins[2].reshape(-1)[0])
        rank = len(ins[0].shape)
        pl = []
        for i in range(rank - 1, -1, -1):
            pl.extend([pads[i], pads[i + rank]])
        if mode == "constant":
            return F.pad(ins[0], pl, mode="constant", value=val)
        return F.pad(ins[0], pl, mode="reflect" if mode == "reflect" else "replicate")
    if op == "Slice":
        x = ins[0]
        starts = at["starts"] if "starts" in at else [int(v) for v in ins[1].tolist()]
        ends = at["ends"] if "ends" in at else [int(v) for v in ins[2].tolist()]
        axes = at.get("axes")
        if axes is None and len(ins) > 3:
            axes = [int(v) for v in ins[3].tolist()]
        steps = at.get("steps")
        if steps is None and len(ins) > 4:
            steps = [int(v) for v in ins[4].tolist()]
        if steps is not None and any(int(v) < 0 for v in steps):
            # ONNX 的反向切片（常见于 Shape 算术里取「最后一维」）：torch 的切片语法
            # 不接受负步长，改成 index_select(arange)。
            axes_l = list(range(len(starts))) if axes is None else [int(a) for a in axes]
            for s0, e0, a0, st0 in zip(starts, ends, axes_l, steps):
                a0 = a0 % x.dim()
                n = x.shape[a0]
                sel = list(range(*slice(int(s0), int(e0), int(st0)).indices(n)))
                x = x.index_select(a0, torch.tensor(sel, device=x.device))
            return x
        return x[_slice_spec(starts, ends, axes, steps, len(x.shape))]
    if op == "DepthToSpace":
        bs = int(at["blocksize"])
        # ONNX 两种排布（写反了像素全错但形状一样）：
        #   DCR: out[n,c,h*bs+i,w*bs+j] = in[n,(c*bs+i)*bs+j,h,w]  == pixel_shuffle
        #   CRD: out[n,c,h*bs+i,w*bs+j] = in[n,c*bs*bs+i*bs+j,h,w] == 通道主序
        if at.get("mode", "DCR") == "CRD":
            b, c, h, w = ins[0].shape
            cc = c // (bs * bs)
            y = ins[0].reshape(b, cc, bs, bs, h, w)
            return y.permute(0, 1, 4, 2, 5, 3).reshape(b, cc, h * bs, w * bs)
        return F.pixel_shuffle(ins[0], bs)
    if op == "ReduceMean":
        axes = at.get("axes")
        return torch.mean(ins[0], dim=None if axes is None else tuple(int(a) for a in axes),
                          keepdim=bool(at.get("keepdims", 1)))
    if op == "ReduceSum":
        axes = at.get("axes")
        return torch.sum(ins[0], dim=None if axes is None else tuple(int(a) for a in axes),
                         keepdim=bool(at.get("keepdims", 1)))
    if op == "Sqrt":
        return torch.sqrt(ins[0])
    if op == "Reciprocal":
        return 1.0 / ins[0]
    if op == "Pow":
        return torch.pow(ins[0], ins[1] if len(ins) > 1 else at["exponent"])
    if op == "Cast":
        return ins[0].to(_CAST_DT[int(at["to"])])
    if op == "BatchNormalization":
        return F.batch_norm(ins[0], ins[1], ins[2], ins[3], ins[4], training=False,
                            eps=at.get("epsilon", 1e-5))
    if op == "MaxPool":
        return F.max_pool2d(ins[0], kernel_size=tuple(at["kernel_shape"]),
                            stride=tuple(at.get("strides", [1, 1])),
                            padding=tuple(at.get("pads", [0, 0, 0, 0])[:2]))
    if op == "AveragePool":
        return F.avg_pool2d(ins[0], kernel_size=tuple(at["kernel_shape"]),
                            stride=tuple(at.get("strides", [1, 1])),
                            padding=tuple(at.get("pads", [0, 0, 0, 0])[:2]))
    if op == "GlobalAveragePool":
        return torch.mean(ins[0], dim=(-2, -1), keepdim=True)
    if op in ("Resize", "Upsample"):
        x = ins[0]
        mode = at.get("mode", "nearest")
        sizes, scales = None, None
        if len(ins) > 3 and ins[3] is not None:
            sizes = [int(v) for v in ins[3].tolist()]
        if scales is None and len(ins) > 2 and ins[2] is not None:
            scales = [float(v) for v in ins[2].tolist()][-2:]
        if "scales" in at:
            scales = [float(v) for v in at["scales"]][-2:]
        ctm = at.get("coordinate_transformation_mode", "half_pixel")
        kw = dict(size=tuple(sizes[-2:])) if sizes is not None else dict(scale_factor=tuple(scales))
        if mode == "nearest":
            return F.interpolate(x, mode="nearest", **kw)
        return F.interpolate(x, mode="bilinear", align_corners=(ctm == "align_corners"), **kw)
    if op == "Unsqueeze":
        y = ins[0]
        for a in sorted(int(v) for v in at["axes"]):
            y = y.unsqueeze(a if a >= 0 else a + len(y.shape) + 1)
        return y
    if op == "Squeeze":
        axes = at.get("axes")
        y = ins[0]
        if axes is None:
            return y.reshape([s for s in y.shape if s != 1])
        for a in sorted((int(v) for v in axes), reverse=True):
            y = y.squeeze(a)
        return y
    if op == "Flatten":
        return torch.flatten(ins[0], int(at.get("axis", 1)))
    if op == "Identity":
        return ins[0]
    if op == "Shape":
        return torch.tensor(list(ins[0].shape), dtype=torch.int64)
    if op == "Gather":
        ax = int(at.get("axis", 0))
        data, idx = ins[0], ins[1].to(torch.int64)
        if idx.dim() == 0:
            return data.index_select(ax, idx.reshape(1)).squeeze(ax)
        y = torch.index_select(data, ax, idx.reshape(-1))
        shape = list(data.shape[:ax]) + list(idx.shape) + list(data.shape[ax + 1:])
        return y.reshape(shape)
    if op == "ConstantOfShape":
        v = at.get("value")
        fill = 0.0 if v is None else float(v.reshape(-1)[0])
        return torch.full([int(i) for i in ins[0].tolist()], fill, dtype=torch.float32)
    if op == "Range":
        return torch.arange(float(ins[0]), float(ins[1]), float(ins[2]))
    if op == "Mod":
        if at.get("fmod", 0):
            return torch.fmod(ins[0], ins[1])
        return torch.remainder(ins[0], ins[1])
    raise NotImplementedError("op not implemented: %s" % op)


# ============================================================ 代码生成
def _emit(op, ins, at, out):
    """一行（或几行）源码：把 out 赋成算子结果。ins 是源码级名字。"""
    a = ", ".join(ins)
    if op == "Conv":
        pads = at.get("pads", [0, 0, 0, 0])
        st = tuple(at.get("strides", [1, 1]))
        dl = tuple(at.get("dilations", [1, 1]))
        grp = int(at.get("group", 1))
        pre = ""
        if pads[0] == pads[2] and pads[1] == pads[3]:
            pad = (pads[0], pads[1])
        else:
            # ONNX Conv pads = [h_beg, w_beg, h_end, w_end]；F.pad 要 (w_beg,w_end,h_beg,h_end)
            pre = ("    %s = F.pad(%s, (%d, %d, %d, %d))\n"
                   % (ins[0], ins[0], pads[1], pads[3], pads[0], pads[2]))
            pad = (0, 0)
        b = ins[2] if len(ins) > 2 else "None"
        return pre + ("    %s = F.conv2d(%s, %s, %s, stride=%s, padding=%s, dilation=%s, "
                      "groups=%d)\n" % (out, ins[0], ins[1], b, st, pad, dl, grp))
    if op == "ConvTranspose":
        b = ins[2] if len(ins) > 2 else "None"
        return ("    %s = F.conv_transpose2d(%s, %s, %s, stride=%s, padding=%s, "
                "output_padding=%s, groups=%d)\n"
                % (out, ins[0], ins[1], b, tuple(at.get("strides", [1, 1])),
                   tuple(at.get("pads", [0, 0, 0, 0])[:2]),
                   tuple(at.get("output_padding", [0, 0])), int(at.get("group", 1))))
    if op == "MatMul":
        return "    %s = torch.matmul(%s, %s)\n" % (out, ins[0], ins[1])
    if op == "Add":
        return "    %s = %s + %s\n" % (out, ins[0], ins[1])
    if op == "Sub":
        return "    %s = %s - %s\n" % (out, ins[0], ins[1])
    if op == "Mul":
        return "    %s = %s * %s\n" % (out, ins[0], ins[1])
    if op == "Div":
        return "    %s = %s / %s\n" % (out, ins[0], ins[1])
    if op == "LeakyRelu":
        return "    %s = F.leaky_relu(%s, %r)\n" % (out, ins[0], float(at.get("alpha", 0.01)))
    if op == "Relu":
        return "    %s = F.relu(%s)\n" % (out, ins[0])
    if op == "Sigmoid":
        return "    %s = torch.sigmoid(%s)\n" % (out, ins[0])
    if op == "Tanh":
        return "    %s = torch.tanh(%s)\n" % (out, ins[0])
    if op == "Concat":
        return "    %s = torch.cat([%s], dim=%d)\n" % (out, a, int(at["axis"]))
    if op == "Transpose":
        return "    %s = %s.permute(%s)\n" % (out, ins[0],
                                              ", ".join(str(int(p)) for p in at["perm"]))
    if op == "Reshape":
        return "    %s = %s.reshape(%s)\n" % (out, ins[0], list(at["__shape"]))
    if op == "Pad":
        pads = [int(v) for v in at["pads"]]
        rank = int(at["__rank"])
        pl = []
        for i in range(rank - 1, -1, -1):
            pl.extend([pads[i], pads[i + rank]])
        mode = at.get("mode", "constant")
        if mode == "constant":
            return ("    %s = F.pad(%s, %s, mode='constant', value=%r)\n"
                    % (out, ins[0], tuple(pl), float(at.get("value", 0.0))))
        return ("    %s = F.pad(%s, %s, mode='%s')\n"
                % (out, ins[0], tuple(pl), 'reflect' if mode == 'reflect' else 'replicate'))
    if op == "Slice":
        lit = ", ".join("slice(%s, %s, %s)" % (s[0], s[1], s[2]) for s in at["__idx"])
        return "    %s = %s[%s]\n" % (out, ins[0], lit)
    if op == "DepthToSpace":
        bs = int(at["blocksize"])
        if at.get("mode", "DCR") == "CRD":
            return ("    _b, _c, _h, _w = %s.shape\n"
                    "    %s = %s.reshape(_b, _c // %d, %d, %d, _h, _w)"
                    ".permute(0, 1, 4, 2, 5, 3).reshape(_b, _c // %d, _h * %d, _w * %d)\n"
                    % (ins[0], out, ins[0], bs * bs, bs, bs, bs * bs, bs, bs))
        return "    %s = F.pixel_shuffle(%s, %d)\n" % (out, ins[0], bs)
    if op == "ReduceMean":
        axes = at.get("axes")
        dim = "None" if axes is None else "(" + ", ".join(str(int(v)) for v in axes) + ",)"
        return ("    %s = torch.mean(%s, dim=%s, keepdim=%r)\n"
                % (out, ins[0], dim, bool(at.get("keepdims", 1))))
    if op == "Sqrt":
        return "    %s = torch.sqrt(%s)\n" % (out, ins[0])
    if op == "Reciprocal":
        return "    %s = 1.0 / %s\n" % (out, ins[0])
    if op == "Pow":
        return "    %s = torch.pow(%s, %s)\n" % (out, ins[0],
                                                 ins[1] if len(ins) > 1 else at.get("exponent"))
    if op == "Cast":
        return "    %s = %s.to(%r)\n" % (out, ins[0], _CAST_DT[int(at["to"])])
    if op == "Identity":
        return "    %s = %s\n" % (out, ins[0])
    if op == "MaxPool":
        return ("    %s = F.max_pool2d(%s, %s, %s, %s)\n"
                % (out, ins[0], tuple(at["kernel_shape"]),
                   tuple(at.get("strides", [1, 1])), tuple(at.get("pads", [0, 0, 0, 0])[:2])))
    if op == "Gather":
        return ("    %s = torch.index_select(%s, %d, %s.to(torch.int64).reshape(-1))\n"
                % (out, ins[0], int(at.get("axis", 0)), ins[1]))
    if op == "Unsqueeze":
        y = "(%s)" % ins[0]
        for ax in sorted(int(v) for v in at["axes"]):
            y = "torch.unsqueeze(%s, %d)" % (y, ax)
        return "    %s = %s\n" % (out, y)
    if op == "Squeeze":
        axes = at.get("axes")
        if axes is None:
            return "    %s = %s.reshape([s for s in %s.shape if s != 1])\n" % (out, ins[0], ins[0])
        y = "(%s)" % ins[0]
        for ax in sorted((int(v) for v in axes), reverse=True):
            y = "torch.squeeze(%s, %d)" % (y, ax)
        return "    %s = %s\n" % (out, y)
    if op == "Flatten":
        return "    %s = torch.flatten(%s, %d)\n" % (out, ins[0], int(at.get("axis", 1)))
    if op == "Clip":
        lo, hi = at.get("min"), at.get("max")
        return ("    %s = torch.clamp(%s, %s, %s)\n"
                % (out, ins[0], "None" if lo is None else float(lo),
                   "None" if hi is None else float(hi)))
    if op == "BatchNormalization":
        return ("    %s = F.batch_norm(%s, %s, %s, %s, %s, training=False, eps=%r)\n"
                % (out, ins[0], ins[1], ins[2], ins[3], ins[4],
                   float(at.get("epsilon", 1e-5))))
    if op == "Softmax":
        return "    %s = torch.softmax(%s, dim=%d)\n" % (out, ins[0], int(at.get("axis", -1)))
    raise NotImplementedError("emit not implemented: %s" % op)


# ============================================================ 折叠 + 编译
class ParamStore(nn.Module):
    """一份权重（+折叠后留下的常量），所有 batch 的 forward 共享它。

    为什么要单独拎出来：`compile_graph` 是「按 batch 编一张图」，而 batch 1/2/4/8
    各编一张就有 4 个 Module；如果每张图各自持有权重，384 那份 1567MB 会变成 6.2GB。
    共享 store 之后，「编 N 张图」的显存恒等于「一份权重」。
    """

    def __init__(self, params=None):
        super().__init__()
        self.moved = 0
        for k, v in (params or {}).items():
            self.register_buffer(k, v, persistent=False)

    def move(self, device, dtype=None, channels_last=False):
        """权重搬一次：device / dtype / 内存格式。返回搬了几个张量（给日志）。"""
        n = 0
        for k, v in list(self._buffers.items()):
            t = v
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            if channels_last and t.dim() == 4:
                t = t.to(memory_format=torch.channels_last)
            self._buffers[k] = t.to(device)
            n += 1
        self.moved = n
        return n


def compile_graph(g, batch, verbose: bool = False):
    """把 OnnxGraph 折叠 + 代码生成成 nn.Module（batch 必须固定）。

    返回 (net, stats)。net 的 forward 里常量从 `self.ps.<name>` 取，所以调用方**必须**
    把 ParamStore 挂到 net.ps 上（`net.ps = store`）。
    """
    if int(batch) <= 0:
        raise ValueError("batch 必须 > 0（dfm 的图是动态 batch，折叠要定死）")
    const = dict(g.const)
    in_name = g.input_name
    dims = []
    for d in (g.input_shape or [int(batch)]):
        dims.append(int(d) if isinstance(d, int) and int(d) > 0 else int(batch))
    if dims and dims[0] <= 0:
        dims[0] = int(batch)
    valshape = {k: list(v.shape) for k, v in const.items()}
    valshape[in_name] = list(dims)

    def meta_shape(op, ins_names, at):
        shapes = [valshape.get(i) for i in ins_names]
        if any(s is None for s in shapes):
            return None
        # 先试「常量留在 CPU」（Pad/Slice/Reshape/Gather 要读值，meta 张量读不了）；
        # 再试「全部 meta」（纯算术跨 device 会报错）。两条都不行就算形状未知。
        for keep_real in (True, False):
            try:
                meta = [const[i] if (i in const and keep_real)
                        else (const[i].to("meta") if i in const
                              else torch.empty(s, device="meta", dtype=torch.float32))
                        for i, s in zip(ins_names, shapes)]
                with torch.no_grad():
                    return list(run_op(op, meta, at).shape)
            except Exception:                        # noqa: BLE001
                continue
        return None

    nodes_rt = []
    prod = {}
    for node in g.nodes:
        op = node["op"]
        ins = list(node["ins"])
        at = dict(node["at"])
        out = node["outs"][0]
        prod[out] = node
        all_const = all((i == "" or i in const) for i in ins)

        if op == "Constant":
            v = run_op(op, [], at)
            const[out] = v
            valshape[out] = list(v.shape)
            continue
        if op == "Shape":
            sh = valshape.get(ins[0])
            if sh is None:
                raise RuntimeError("Shape 节点的输入形状未知: %s" % ins[0])
            const[out] = torch.tensor(sh, dtype=torch.int64)
            valshape[out] = [len(sh)]
            continue
        if all_const and op in _RUNTIME_OPS:
            args = [const[i] for i in ins if i != ""]
            try:
                with torch.no_grad():
                    v = run_op(op, args, at)
            except Exception as e:                   # noqa: BLE001
                sh = [(aa.dtype, tuple(aa.shape), aa.reshape(-1)[:6].tolist())
                      if aa.numel() <= 8 else (aa.dtype, tuple(aa.shape)) for aa in args]
                raise RuntimeError("折叠 %s (%s) 失败: %s；attrs=%s；inputs=%s"
                                   % (op, node["name"], e, at, sh)) from e
            const[out] = v
            valshape[out] = list(v.shape)
            continue
        if op in _FOLD_ONLY:
            raise RuntimeError("无法折叠的 %s 节点（输出 %s）" % (op, out))
        if op not in _RUNTIME_OPS:
            raise NotImplementedError("未实现算子 %s（节点 %s）" % (op, node["name"]))
        sh = meta_shape(op, ins, at)
        if sh is None:
            chain, cur = [], (ins[1] if len(ins) > 1 else None)
            while cur and cur in prod and len(chain) < 12:
                pn = prod[cur]
                chain.append("%s(%s)[const=%s][shape=%s]"
                             % (pn["op"], cur, cur in const, valshape.get(cur)))
                cur = pn["ins"][0] if pn["ins"] else None
            raise RuntimeError("形状推断在 %s (%s) 断了；inputs=%s shapes=%s\nchain:\n%s"
                               % (op, node["name"], ins, [valshape.get(i) for i in ins],
                                  "\n".join(chain)))
        valshape[out] = sh
        nodes_rt.append({"op": op, "ins": ins, "out": out, "at": at})
        for extra in node["outs"][1:]:
            valshape[extra] = None

    # ---- 代码生成：一条直线的 forward，中间变量名按「最后一次使用」回收 ----
    body = []
    name_of = {in_name: "x"}
    params = {}
    ctr = [0]
    vctr = [0]
    last_use = {}
    for _i, _n in enumerate(nodes_rt):
        for _t in _n["ins"]:
            last_use[_t] = _i
    for _o in g.output_names:
        last_use[_o] = len(nodes_rt) + 1
    pool, live = [], {}

    def ref(t):
        if t == "":
            return "None"
        if t in name_of:
            return name_of[t]
        if t in const:
            ctr[0] += 1
            an = "c%d" % ctr[0]
            params[an] = const[t]
            name_of[t] = "ps." + an
            return name_of[t]
        raise RuntimeError("值未定义: %s" % t)

    for _i, nd in enumerate(nodes_rt):
        op = nd["op"]
        at = dict(nd["at"])
        ins_src = [ref(i) for i in nd["ins"]]
        for _nm, _lu in list(live.items()):
            if _lu < _i:
                pool.append(_nm)
                del live[_nm]
        if pool:
            out_src = pool.pop()
        else:
            vctr[0] += 1
            out_src = "v%d" % vctr[0]
        live[out_src] = last_use.get(nd["out"], len(nodes_rt) + 1)
        name_of[nd["out"]] = out_src
        if op == "Reshape":
            shape = [int(v) for v in const[nd["ins"][1]].tolist()]
            src_shape = valshape.get(nd["ins"][0]) or []
            for _k, _s in enumerate(shape):      # ONNX 的 0 = 复制输入那一维
                if _s == 0 and _k < len(src_shape):
                    shape[_k] = int(src_shape[_k])
            at["__shape"] = shape
            ins_src = ins_src[:1]
        elif op == "Slice":
            starts = [int(v) for v in const[nd["ins"][1]].tolist()]
            ends = [int(v) for v in const[nd["ins"][2]].tolist()]
            axes = ([int(v) for v in const[nd["ins"][3]].tolist()]
                    if len(nd["ins"]) > 3 and nd["ins"][3] else None)
            steps = ([int(v) for v in const[nd["ins"][4]].tolist()]
                     if len(nd["ins"]) > 4 and nd["ins"][4] else None)
            rank = len(valshape[nd["ins"][0]])
            spec = _slice_spec(starts, ends, axes, steps, rank)
            at["__idx"] = [(None, None, None) if s == slice(None)
                           else (s.start, s.stop, s.step) for s in spec]
            ins_src = ins_src[:1]
        elif op == "Pad":
            rank = len(valshape[nd["ins"][0]])
            if len(nd["ins"]) > 1 and nd["ins"][1] in const:
                at["pads"] = [int(v) for v in const[nd["ins"][1]].tolist()]
            at["__rank"] = rank
            if len(nd["ins"]) > 2 and nd["ins"][2] in const:
                at["value"] = float(const[nd["ins"][2]].reshape(-1)[0])
            ins_src = ins_src[:1]
        if op == "Gather" and len(ins_src) > 1 and ins_src[1].startswith("ps."):
            # 索引常量必须**在编译期**变成 int64，不能留到运行时 .to(int64)。
            # 为什么：416 那张图的 Gather 索引是 float 常量（值 0..2559）。权重压成
            # bf16（DFL_DFM_DTYPE=bf16w）时 bf16 的整数精度只有 256，1500 会被舍成
            # 1504 —— 要么 CUDA 断言 "index out of bounds"，要么**静默取错通道**。
            _bn = ins_src[1][3:]
            _ik = _bn + "_i64"
            if _ik not in params:
                params[_ik] = params[_bn].to(torch.int64)
            ins_src[1] = "ps." + _ik
        body.append(_emit(op, ins_src, at, out_src))

    outs = [name_of[o] for o in g.output_names]
    src = ("import torch\nimport torch.nn.functional as F\n\n\n"
           "class _DFMTorchNet(torch.nn.Module):\n"
           "    def forward(self, x):\n"
           "        ps = self.ps\n"
           + "".join(("    " + l) if l.strip() else l
                     for l in "".join(body).splitlines(keepends=True))
           + "        return (%s,)\n" % (", ".join(outs),))
    if verbose or os.environ.get("DFM_DEBUG_SRC"):
        print(src[:1200])

    ns = {}
    exec(compile(src, "<dfm:%s:b%d>" % (os.path.basename(g.path), int(batch)), "exec"), ns)
    net = ns["_DFMTorchNet"]()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    net.source = src
    # 常量：name -> CPU 张量。initializer 是 g.const 里**同一个对象**（所以不同 batch
    # 编出来的图天然共享权重），折叠出来的常量是新对象（小，可为每个 batch 各留一份）。
    net.params = params
    stats = {"batch": int(batch), "runtime_nodes": len(nodes_rt),
             "folded": len(const) - len(g.const), "params": len(params),
             "out_shapes": [valshape.get(o) for o in g.output_names],
             "src_lines": src.count("\n")}
    return net, stats
