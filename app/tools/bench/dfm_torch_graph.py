# -*- coding: utf-8 -*-
"""ONNX 图 → torch nn.Module（dfm 的权重搬进 PyTorch，推理不再经过 onnxruntime）。

为什么要有这个文件：dfm 里**没有** torch state_dict（两个样本的 metadata_props 都是空，
producer 分别是 tf2onnx 1.9.3 与 pytorch 2.12.1），权重只以 ONNX initializer 的形式存在。
所以「PyTorch 能不能跑 dfm」等价于「能不能把这张图 + 这批 initializer 翻成一个 nn.Module」。
这里做的就是这件事，并且是**静态折叠后代码生成**——不是逐节点解释器：
  · 逐节点解释器的 Python 开销（1065 个节点 ≈ 每张几 ms）会把 torch 的收益吃掉，
    torch.compile 也拿不到干净的图；生成的 forward 是纯 aten 序列，compile 才有意义。
  · 折叠（Shape/Reshape/Mod/Slice/Pad/Range/Gather/Concat…）在 batch 固定后能整段算掉，
    416 那份 torch.onnx.export 的图 1065 节点 → 折叠后只剩两百来行真算子。

用法：
    from dfm_torch_graph import build_from_onnx
    m = build_from_onnx("x.dfm", batch=8)      # -> nn.Module, 输入 NHWC float32 [0,1]
    y = m(torch.rand(8,416,416,3,device="cuda"))

约定：
  · 只实现 SAEHD dfm 实际出现的算子；遇到没实现的算子**直接报错并打印节点**，
    不静默回退（静默回退等于给出一份不知道在算什么的时间）。
  · 输入/输出保持 ONNX 的 NHWC 语义（dfm 的 in_face:0 = [B,H,W,3]），
    图内部的 Transpose 保留原样 —— 配合 channels_last 它们是不拷贝的 view。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx import numpy_helper

# ---------------------------------------------------------------- 属性解析

def _attrs(node):
    d = {}
    for a in node.attribute:
        t = a.type
        if t == onnx.AttributeProto.INT:
            d[a.name] = int(a.i)
        elif t == onnx.AttributeProto.FLOAT:
            d[a.name] = float(a.f)
        elif t == onnx.AttributeProto.STRING:
            d[a.name] = a.s.decode("utf-8")
        elif t == onnx.AttributeProto.INTS:
            d[a.name] = [int(v) for v in a.ints]
        elif t == onnx.AttributeProto.FLOATS:
            d[a.name] = [float(v) for v in a.floats]
        elif t == onnx.AttributeProto.STRINGS:
            d[a.name] = [v.decode("utf-8") for v in a.strings]
        elif t == onnx.AttributeProto.TENSOR:
            d[a.name] = torch.from_numpy(numpy_helper.to_array(a.t).copy())
        else:
            d[a.name] = None
    return d


_ONNX2TORCH_DTYPE = {
    1: torch.float32, 2: torch.uint8, 3: torch.int8, 4: torch.int16,
    5: torch.int16, 6: torch.int32, 7: torch.int64, 9: torch.bool,
    10: torch.float16, 11: torch.float64, 12: torch.int32, 13: torch.uint8,
    16: torch.bfloat16,
}


def _dtype(v):
    if isinstance(v, int):
        return _ONNX2TORCH_DTYPE.get(v, torch.float32)
    return _ONNX2TORCH_DTYPE.get(int(v), torch.float32)


def _t(x):
    """constant → torch.Tensor（int64 保持 int64；float 保持 float32）。"""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


def _slice_spec(starts, ends, axes, steps, rank):
    axes = list(range(len(starts))) if axes is None else list(axes)
    steps = [1] * len(starts) if steps is None else list(steps)
    idx = [slice(None)] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = a % rank
        idx[a] = slice(int(s), int(e) if int(e) < 2**62 else None, int(st))
    return tuple(idx)


# ---------------------------------------------------------------- 单算子执行
# 同一份语义两用：常量折叠（CPU 真张量）+ 形状推断（meta 张量）+ 对拍。

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


def torch_run(op, ins, at, ctx):
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
        lo = at.get("min")
        hi = at.get("max")
        if lo is None and len(ins) > 1:
            lo = ins[1]
        if hi is None and len(ins) > 2:
            hi = ins[2]
        return torch.clamp(ins[0],
                           None if lo is None else float(lo),
                           None if hi is None else float(hi))
    if op == "Softmax":
        ax = at.get("axis", -1)
        return torch.softmax(ins[0], dim=int(ax))
    if op == "Concat":
        return torch.cat(list(ins), dim=at["axis"])
    if op == "Transpose":
        return ins[0].permute(*at["perm"])
    if op == "Reshape":
        shape = [int(v) for v in ins[1].tolist()]
        if at.get("allowzero"):
            return ins[0].reshape(shape)
        cur = list(ins[0].shape)
        z = [i for i, s in enumerate(shape) if s == 0]
        for i in z:
            shape[i] = cur[i]
        return ins[0].reshape(shape)
    if op == "Pad":
        pads = [int(v) for v in at["pads"]] if "pads" in at else [int(v) for v in ins[1].tolist()]
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
            # ONNX 的反向切片（常见于 Shape 算术里取「最后一维」）：torch 的
            # 切片语法不接受负步长，改成 index_select(arange)。
            axes_l = list(range(len(starts))) if axes is None else [int(a) for a in axes]
            for s0, e0, a0, st0 in zip(starts, ends, axes_l, steps):
                a0 = a0 % x.dim()
                n = x.shape[a0]
                sel = list(range(*slice(int(s0), int(e0), int(st0)).indices(n)))
                x = x.index_select(a0, torch.tensor(sel, device=x.device))
            return x
        return x[_slice_spec(starts, ends, axes, steps, len(x.shape))]
    if op == "DepthToSpace":
        bs = at["blocksize"]
        # ONNX 两种排布（别写反，写反了像素全错但形状一样）：
        #   DCR: out[n, c, h*bs+i, w*bs+j] = in[n, (c*bs + i)*bs + j, h, w]  == pixel_shuffle
        #   CRD: out[n, c, h*bs+i, w*bs+j] = in[n, c*bs*bs + i*bs + j, h, w]  == 通道主序
        if at.get("mode", "DCR") == "CRD":
            b, c, h, w = ins[0].shape
            cc = c // (bs * bs)
            y = ins[0].reshape(b, cc, bs, bs, h, w)
            y = y.permute(0, 1, 4, 2, 5, 3).reshape(b, cc, h * bs, w * bs)
            return y
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
        return ins[0].to(_dtype(at["to"]))
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
        sizes = None
        scales = None
        if len(ins) > 3 and ins[3] is not None:
            sizes = [int(v) for v in ins[3].tolist()]
        if scales is None and len(ins) > 2 and ins[2] is not None:
            sc = [float(v) for v in ins[2].tolist()]
            scales = sc[-2:]
        if "scales" in at:
            scales = [float(v) for v in at["scales"]][-2:]
        ctm = at.get("coordinate_transformation_mode", "half_pixel")
        if sizes is not None:
            kw = dict(size=tuple(sizes[-2:]))
        else:
            kw = dict(scale_factor=tuple(scales))
        if mode == "nearest":
            return F.interpolate(x, mode="nearest", **kw)
        return F.interpolate(x, mode="bilinear",
                             align_corners=(ctm == "align_corners"),
                             **kw)
    if op == "Unsqueeze":
        axes = at["axes"]
        y = ins[0]
        for a in sorted(int(v) for v in axes):
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
        ax = at.get("axis", 1)
        return torch.flatten(ins[0], ax)
    if op == "Identity":
        return ins[0]
    if op == "Shape":
        return torch.tensor(list(ins[0].shape), dtype=torch.int64)
    if op == "Gather":
        ax = at.get("axis", 0)
        data = ins[0]
        idx = ins[1].to(torch.int64)
        if idx.dim() == 0:
            return data.index_select(ax, idx.reshape(1)).squeeze(ax)
        y = torch.index_select(data, ax, idx.reshape(-1))
        shape = list(data.shape[:ax]) + list(idx.shape) + list(data.shape[ax + 1:])
        return y.reshape(shape)
    if op == "ConstantOfShape":
        v = at.get("value")
        fill = 0.0 if v is None else float(v.reshape(-1)[0])
        shape = [int(i) for i in ins[0].tolist()]
        return torch.full(shape, fill, dtype=torch.float32)
    if op == "Range":
        return torch.arange(float(ins[0]), float(ins[1]), float(ins[2]))
    if op == "Mod":
        if at.get("fmod", 0):
            return torch.fmod(ins[0], ins[1])
        return torch.remainder(ins[0], ins[1])
    raise NotImplementedError(f"op not implemented: {op}")


# ---------------------------------------------------------------- 代码生成
# 只覆盖「折叠之后仍然存在」的算子；其余应当已被折叠成常量。

def _emit(op, ins, at, out, ctx):
    """返回一行源码：把 out 赋成算子结果。ins 是源码级名字。"""
    a = ", ".join(ins)
    if op == "Conv":
        pads = at.get("pads", [0, 0, 0, 0])
        st = tuple(at.get("strides", [1, 1]))
        dl = tuple(at.get("dilations", [1, 1]))
        grp = at.get("group", 1)
        pre = ""
        if pads[0] == pads[2] and pads[1] == pads[3]:
            pad = (pads[0], pads[1])
        else:
            # ONNX Conv pads = [h_beg, w_beg, h_end, w_end]；F.pad 要 (w_beg, w_end, h_beg, h_end)
            pre = (f"    {ins[0]} = F.pad({ins[0]}, "
                   f"({pads[1]}, {pads[3]}, {pads[0]}, {pads[2]}))\n")
            pad = (0, 0)
        b = ins[2] if len(ins) > 2 else "None"
        return pre + (f"    {out} = F.conv2d({ins[0]}, {ins[1]}, {b}, stride={st}, "
                      f"padding={pad}, dilation={dl}, groups={grp})\n")
    if op == "ConvTranspose":
        b = ins[2] if len(ins) > 2 else "None"
        return (f"    {out} = F.conv_transpose2d({ins[0]}, {ins[1]}, {b}, "
                f"stride={tuple(at.get('strides', [1, 1]))}, "
                f"padding={tuple(at.get('pads', [0, 0, 0, 0])[:2])}, "
                f"output_padding={tuple(at.get('output_padding', [0, 0]))}, "
                f"groups={at.get('group', 1)})\n")
    if op == "MatMul":
        return f"    {out} = torch.matmul({ins[0]}, {ins[1]})\n"
    if op == "Add":
        return f"    {out} = {ins[0]} + {ins[1]}\n"
    if op == "Sub":
        return f"    {out} = {ins[0]} - {ins[1]}\n"
    if op == "Mul":
        return f"    {out} = {ins[0]} * {ins[1]}\n"
    if op == "Div":
        return f"    {out} = {ins[0]} / {ins[1]}\n"
    if op == "LeakyRelu":
        return f"    {out} = F.leaky_relu({ins[0]}, {at.get('alpha', 0.01)})\n"
    if op == "Relu":
        return f"    {out} = F.relu({ins[0]})\n"
    if op == "Sigmoid":
        return f"    {out} = torch.sigmoid({ins[0]})\n"
    if op == "Tanh":
        return f"    {out} = torch.tanh({ins[0]})\n"
    if op == "Concat":
        return f"    {out} = torch.cat([{a}], dim={at['axis']})\n"
    if op == "Transpose":
        return f"    {out} = {ins[0]}.permute({', '.join(str(p) for p in at['perm'])})\n"
    if op == "Reshape":
        shape = [int(v) for v in at["__shape"]]
        return f"    {out} = {ins[0]}.reshape({shape})\n"
    if op == "Pad":
        pads = [int(v) for v in at["pads"]]
        rank = at["__rank"]
        pl = []
        for i in range(rank - 1, -1, -1):
            pl.extend([pads[i], pads[i + rank]])
        mode = at.get("mode", "constant")
        val = float(at.get("value", 0.0))
        if mode == "constant":
            return f"    {out} = F.pad({ins[0]}, {tuple(pl)}, mode='constant', value={val})\n"
        return f"    {out} = F.pad({ins[0]}, {tuple(pl)}, mode='{'reflect' if mode == 'reflect' else 'replicate'}')\n"
    if op == "Slice":
        idx = at["__idx"]
        lit = ", ".join("slice(%s, %s, %s)" % (s[0], s[1], s[2]) for s in idx)
        return f"    {out} = {ins[0]}[{lit}]\n"
    if op == "DepthToSpace":
        bs = at["blocksize"]
        if at.get("mode", "DCR") == "CRD":
            return (f"    _b, _c, _h, _w = {ins[0]}.shape\n"
                    f"    {out} = {ins[0]}.reshape(_b, _c // {bs * bs}, {bs}, {bs}, _h, _w)"
                    f".permute(0, 1, 4, 2, 5, 3).reshape(_b, _c // {bs * bs}, _h * {bs}, _w * {bs})\n")
        return f"    {out} = F.pixel_shuffle({ins[0]}, {bs})\n"
    if op == "ReduceMean":
        axes = at.get("axes")
        dim = "None" if axes is None else "(" + ", ".join(str(int(v)) for v in axes) + ",)"
        return f"    {out} = torch.mean({ins[0]}, dim={dim}, keepdim={bool(at.get('keepdims', 1))})\n"
    if op == "Sqrt":
        return f"    {out} = torch.sqrt({ins[0]})\n"
    if op == "Reciprocal":
        return f"    {out} = 1.0 / {ins[0]}\n"
    if op == "Pow":
        return f"    {out} = torch.pow({ins[0]}, {ins[1] if len(ins) > 1 else at.get('exponent')})\n"
    if op == "Cast":
        return f"    {out} = {ins[0]}.to({_dtype(at['to'])})\n"
    if op == "Identity":
        return f"    {out} = {ins[0]}\n"
    if op == "MaxPool":
        return (f"    {out} = F.max_pool2d({ins[0]}, {tuple(at['kernel_shape'])}, "
                f"{tuple(at.get('strides', [1, 1]))}, {tuple(at.get('pads', [0, 0, 0, 0])[:2])})\n")
    if op == "Gather":
        ax = at.get("axis", 0)
        return (f"    {out} = torch.index_select({ins[0]}, {ax}, "
                f"{ins[1]}.to(torch.int64).reshape(-1))\n")
    if op == "Unsqueeze":
        y = f"({ins[0]})"
        for a in sorted(int(v) for v in at["axes"]):
            y = f"torch.unsqueeze({y}, {a})"
        return f"    {out} = {y}\n"
    if op == "Squeeze":
        axes = at.get("axes")
        if axes is None:
            return f"    {out} = {ins[0]}.reshape([s for s in {ins[0]}.shape if s != 1])\n"
        y = f"({ins[0]})"
        for a in sorted((int(v) for v in axes), reverse=True):
            y = f"torch.squeeze({y}, {a})"
        return f"    {out} = {y}\n"
    if op == "Flatten":
        return f"    {out} = torch.flatten({ins[0]}, {at.get('axis', 1)})\n"
    if op == "Clip":
        lo = at.get("min")
        hi = at.get("max")
        return (f"    {out} = torch.clamp({ins[0]}, "
                f"{'None' if lo is None else float(lo)}, "
                f"{'None' if hi is None else float(hi)})\n")
    if op == "BatchNormalization":
        return (f"    {out} = F.batch_norm({ins[0]}, {ins[1]}, {ins[2]}, {ins[3]}, {ins[4]}, "
                f"training=False, eps={at.get('epsilon', 1e-5)})\n")
    if op == "Softmax":
        return f"    {out} = torch.softmax({ins[0]}, dim={int(at.get('axis', -1))})\n"
    raise NotImplementedError(f"emit not implemented: {op}")


# ---------------------------------------------------------------- 图构建

class _Const:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


class _Sym:
    """非张量值：形状已知（或未知）的运行期张量。"""
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = shape


def _shape_of(x):
    return list(x.shape) if x is not None else None


def build_from_onnx(path, batch, verbose=False):
    """把 .dfm(ONNX) 编译成 nn.Module。batch 必须固定（dfm 图是动态 batch，折叠要定死）。"""
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph

    const = {}
    for t in g.initializer:
        const[t.name] = torch.from_numpy(numpy_helper.to_array(t).copy())

    inp = g.input[0]
    in_name = inp.name
    valshape = {k: list(v.shape) for k, v in const.items()}
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(int(d.dim_value) if d.dim_value else int(batch))
    valshape[in_name] = dims

    def meta_shape(op, ins_names, at):
        shapes = [valshape.get(i) for i in ins_names]
        if any(s is None for s in shapes):
            return None
        # 先试「常量留在 CPU」：Pad/Slice/Reshape/Gather 这些要读值，meta 张量读不了；
        # 再试「全部 meta」：纯算术（Add/Mul/MatMul）跨 device 会报错。两条都不行才算形状未知。
        for keep_real in (True, False):
            try:
                meta = [const[i] if (i in const and keep_real)
                        else (const[i].to("meta") if i in const
                              else torch.empty(s, device="meta", dtype=torch.float32))
                        for i, s in zip(ins_names, shapes)]
                with torch.no_grad():
                    y = torch_run(op, meta, at, None)
                return list(y.shape)
            except Exception:
                continue
        return None

    nodes_rt = []
    _reported = []
    prod = {}
    for node in g.node:
        op = node.op_type
        ins = list(node.input)
        at = _attrs(node)
        out = node.output[0]
        prod[out] = node
        all_const = all((i == "" or i in const) for i in ins)

        if op == "Constant":
            v = torch_run(op, [], at, None)
            const[out] = v
            valshape[out] = list(v.shape)
            continue
        if op == "Shape":
            sh = valshape.get(ins[0])
            if sh is None:
                raise RuntimeError("Shape 节点的输入形状未知: " + ins[0])
            const[out] = torch.tensor(sh, dtype=torch.int64)
            valshape[out] = [len(sh)]
            continue
        if all_const and op in _RUNTIME_OPS:
            _args = [const[i] for i in ins if i != ""]
            try:
                with torch.no_grad():
                    v = torch_run(op, _args, at, None)
            except Exception as _e:
                _sh = [(a.dtype, tuple(a.shape), a.reshape(-1)[:6].tolist()) if a.numel() <= 8
                       else (a.dtype, tuple(a.shape)) for a in _args]
                raise RuntimeError("fold %s (%s) failed: %s; attrs=%s; inputs=%s" %
                                   (op, node.name, _e, at, _sh)) from _e
            const[out] = v
            valshape[out] = list(v.shape)
            continue
        if op in _FOLD_ONLY:
            raise RuntimeError("无法折叠的 %s 节点（输出 %s）" % (op, out))
        if op not in _RUNTIME_OPS:
            raise NotImplementedError("未实现算子 %s（节点 %s）" % (op, node.name))
        sh = meta_shape(op, ins, at)
        if sh is None and not _reported:
            _reported.append(1)
            chain = []
            cur = ins[1] if len(ins) > 1 else None
            while cur and cur in prod and len(chain) < 12:
                pn = prod[cur]
                chain.append("%s(%s)[const=%s][shape=%s]" %
                             (pn.op_type, cur, cur in const, valshape.get(cur)))
                cur = pn.input[0] if pn.input else None
            raise RuntimeError("shape inference broke at %s (%s); inputs=%s shapes=%s\nchain:\n%s" %
                               (op, node.name, ins, [valshape.get(i) for i in ins],
                                "\n".join(chain)))
        valshape[out] = sh
        nodes_rt.append({"op": op, "ins": ins, "out": out, "at": at})
        for extra in node.output[1:]:
            valshape[extra] = None

    # ---- 代码生成（不是逐节点解释器：compile 才有干净图）
    body = []
    name_of = {in_name: "x"}
    params = {}
    ctr = [0]
    vctr = [0]
    # 变量名复用：生成的 forward 是一条直线，如果每个中间张量都占一个独立局部名，
    # Python 会把它们**全部同时留在作用域里** —— 416 batch8 实测因此涨到 9.07GB
    # （超过 8GB 显存，驱动回退到系统内存，一张 4.7 秒）。这里按「最后一次被用到的位置」
    # 回收名字，峰值回落到 2~3GB。
    last_use = {}
    for _i, _nd in enumerate(nodes_rt):
        for _t in _nd["ins"]:
            last_use[_t] = _i
    for _o in g.output:
        last_use[_o.name] = len(nodes_rt) + 1
    pool = []
    live = {}

    def ref(t):
        if t == "":
            return "None"
        if t in name_of:
            return name_of[t]
        if t in const:
            ctr[0] += 1
            an = "c%d" % ctr[0]
            params[an] = const[t]
            name_of[t] = "self." + an     # 存全名：同一个常量第二次被引用时也要带 self.
            return name_of[t]
        raise RuntimeError("值未定义: " + t)

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
            at["__shape"] = [int(v) for v in const[nd["ins"][1]].tolist()]
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
            at["__idx"] = [(None, None, None) if s == slice(None) else (s.start, s.stop, s.step)
                           for s in spec]
            ins_src = ins_src[:1]
        elif op == "Pad":
            rank = len(valshape[nd["ins"][0]])
            at["pads"] = [int(v) for v in const[nd["ins"][1]].tolist()]
            at["__rank"] = rank
            if len(nd["ins"]) > 2 and nd["ins"][2] in const:
                at["value"] = float(const[nd["ins"][2]].reshape(-1)[0])
            ins_src = ins_src[:1]
        body.append(_emit(op, ins_src, at, out_src, None))

    outs = [name_of[o.name] for o in g.output]
    head = ("import torch\n"
            "import torch.nn.functional as F\n\n"
            "class _DFMTorchNet(torch.nn.Module):\n"
            "    def forward(self, x):\n")
    body_src = "".join(body)
    body_src = "".join(("    " + l) if l.strip() else l
                      for l in body_src.splitlines(keepends=True))
    src = head + body_src + "        return (%s,)\n" % (", ".join(outs))

    if __import__("os").environ.get("DFM_DEBUG_SRC"):
        print(src[:600])
    ns = {}
    exec(compile(src, "<dfm>", "exec"), ns)
    cls = ns["_DFMTorchNet"]

    net = cls()
    for an, v in params.items():
        net.register_buffer(an, v, persistent=False)
    net.eval()
    net.source = src
    net.n_runtime_nodes = len(nodes_rt)
    net.n_folded = len(const)
    net.out_names = [o.name for o in g.output]
    return net
