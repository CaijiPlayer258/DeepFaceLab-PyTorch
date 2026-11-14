# -*- coding: utf-8 -*-
r"""节点级对拍：用「前缀子图」二分找出 torch 图与 onnxruntime 的**第一个分叉节点**。

为什么要二分而不是把所有中间张量都当输出：torch.onnx.export 出来的 dfm 里
graph.value_info 是空的，onnx.shape_inference 也补不出来（动态 batch），
于是没法给 ORT 声明中间输出的类型。改成「把前 k 个节点切出来当一个小模型」，
输出用动态 float 声明，就能逐点对拍，代价是每次要新建一次 session —— 所以二分。

    runtime\engine\python.exe app\tools\bench\dfm_node_parity.py <model.dfm> [batch]
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfm_torch_graph import _attrs, torch_run, _RUNTIME_OPS   # noqa: E402


def _torch_prefix(m, x_np, k):
    """跑到第 k 个节点（含），返回该节点的输出张量。"""
    g = m.graph
    const = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in g.initializer}
    valshape = {kk: list(v.shape) for kk, v in const.items()}
    inp = g.input[0]
    valshape[inp.name] = list(x_np.shape)
    env = {inp.name: torch.from_numpy(x_np.copy())}
    for i, n in enumerate(g.node):
        ins = list(n.input)
        at = _attrs(n)
        out = n.output[0]
        if n.op_type == "Constant":
            env[out] = torch_run(n.op_type, [], at, None)
        elif n.op_type == "Shape":
            env[out] = torch.tensor(valshape[ins[0]], dtype=torch.int64)
        elif all((j == "" or j in const) for j in ins) and n.op_type in _RUNTIME_OPS:
            env[out] = torch_run(n.op_type, [const[j] for j in ins if j != ""], at, None)
        else:
            args = [env[j] if j in env else const[j] for j in ins if j != ""]
            env[out] = torch_run(n.op_type, args, at, None)
            valshape[out] = list(env[out].shape)
        if i == k:
            return env[out]
    raise IndexError(k)


_DT = {torch.float32: TensorProto.FLOAT, torch.float16: TensorProto.FLOAT16,
       torch.float64: TensorProto.DOUBLE, torch.int64: TensorProto.INT64,
       torch.int32: TensorProto.INT32, torch.bool: TensorProto.BOOL,
       torch.uint8: TensorProto.UINT8, torch.int8: TensorProto.INT8}


def _ort_prefix(m, x_np, k, providers, dtype=torch.float32):
    import onnxruntime as ort
    g2 = copy.deepcopy(m.graph)
    del g2.node[k + 1:]
    out_name = m.graph.node[k].output[0]
    del g2.output[:]
    g2.output.append(helper.make_tensor_value_info(out_name, _DT.get(dtype, TensorProto.FLOAT), None))
    del g2.value_info[:]
    model = helper.make_model(g2, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    blob = model.SerializeToString()
    last = None
    for cand in [dtype, torch.float32, torch.int64, torch.float16, torch.bool]:
        g2.output[-1].type.tensor_type.elem_type = _DT.get(cand, TensorProto.FLOAT)
        try:
            sess = ort.InferenceSession(helper.make_model(
                g2, opset_imports=list(m.opset_import), ir_version=m.ir_version
            ).SerializeToString(), sess_options=so, providers=providers)
            return sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
        except Exception as e:
            last = e
    raise last


def _diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return 9.9, "shape %s != %s" % (a.shape, b.shape)
    if a.dtype.kind not in "fc":
        return float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))), ""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))), ""


def _torch_all(m, x_np):
    """整图跑一遍 torch 侧，返回 (env, order)。"""
    g = m.graph
    const = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in g.initializer}
    valshape = {kk: list(v.shape) for kk, v in const.items()}
    inp = g.input[0]
    valshape[inp.name] = list(x_np.shape)
    env = {inp.name: torch.from_numpy(x_np.copy())}
    order = []
    for n in g.node:
        ins = list(n.input)
        at = _attrs(n)
        out = n.output[0]
        if n.op_type == "Constant":
            env[out] = torch_run(n.op_type, [], at, None)
        elif n.op_type == "Shape":
            env[out] = torch.tensor(valshape[ins[0]], dtype=torch.int64)
        elif all((j == "" or j in const) for j in ins) and n.op_type in _RUNTIME_OPS:
            env[out] = torch_run(n.op_type, [const[j] for j in ins if j != ""], at, None)
        else:
            args = [env[j] if j in env else const[j] for j in ins if j != ""]
            env[out] = torch_run(n.op_type, args, at, None)
            valshape[out] = list(env[out].shape)
            order.append(out)
    return env, order


def all_compare(m, x, providers, tol):
    """一次性把「4 维 float32 的中间张量」全暴露给 ORT，逐节点对拍。"""
    import onnxruntime as ort
    env, order = _torch_all(m, x)
    names = [n for n in order if n in env and env[n].dtype == torch.float32
             and env[n].dim() == 4]
    g2 = copy.deepcopy(m.graph)
    del g2.output[:]
    del g2.value_info[:]
    for nm in names:
        g2.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))
    model = helper.make_model(g2, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=so, providers=providers)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    nad = 0
    first = None
    for nm, v in zip(names, outs):
        got = env[nm].detach().numpy()
        d = float(np.max(np.abs(np.asarray(v, dtype=np.float64) - got.astype(np.float64)))) \
            if np.asarray(v).shape == got.shape else 9.9
        if d > tol:
            nad += 1
            if first is None:
                first = nm
            if nad <= 6:
                print("  DIVERGE %-28s diff=%.4f" % (nm, d))
    print("4-D float tensors compared: %d, divergent: %d, first: %s" % (len(names), nad, first))
    return nad


def main():
    path = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-3
    m = onnx.load(str(path), load_external_data=False)
    res = 416 if "416" in path else 384
    x = np.random.default_rng(0).random((B, res, res, 3), dtype=np.float32)
    providers = ["CPUExecutionProvider"]

    if "--all" in sys.argv:
        all_compare(m, x, providers, tol)
        return 0

    n_nodes = len(m.graph.node)
    print("nodes", n_nodes)
    t_end = _torch_prefix(m, x, n_nodes - 1)
    d_end, msg = _diff(_ort_prefix(m, x, n_nodes - 1, providers, t_end.dtype),
                       t_end.detach().numpy())
    print("last node diff", d_end, msg)
    if d_end <= tol:
        print("NO DIVERGENCE (<= %g)" % tol)
        return 0
    lo, hi = 0, n_nodes - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        tb_ = _torch_prefix(m, x, mid)
        a = _ort_prefix(m, x, mid, providers, tb_.dtype)
        b = tb_.detach().numpy()
        d, msg = _diff(a, b)
        ok = d <= tol
        print("  mid %4d %-14s diff=%.6g %s %s" % (
            mid, m.graph.node[mid].op_type, d, "OK" if ok else "DIFF", msg))
        if ok:
            lo = mid
        else:
            hi = mid
    k = hi
    n = m.graph.node[k]
    print("\nFIRST DIVERGENT NODE @%d: %s (%s)" % (k, n.op_type, n.name))
    print("  inputs:", list(n.input))
    for a_ in n.attribute:
        print("  attr", a_.name, "ints", list(a_.ints), "i", a_.i, "f", a_.f)
    ta_ = _torch_prefix(m, x, k)
    ta = ta_.detach().numpy()
    oa = np.asarray(_ort_prefix(m, x, k, providers, ta_.dtype))
    print("  torch", ta.shape, "mean %.5f min %.5f max %.5f" % (ta.mean(), ta.min(), ta.max()))
    print("  ort  ", oa.shape, "mean %.5f min %.5f max %.5f" % (oa.mean(), oa.min(), oa.max()))
    if k > 0:
        tb_ = _torch_prefix(m, x, k - 1)
        tb = tb_.detach().numpy()
        ob = np.asarray(_ort_prefix(m, x, k - 1, providers, tb_.dtype))
        d2, msg2 = _diff(ob, tb)
        print("  previous node diff:", d2, msg2)
        print("  torch prev", tb.shape, "mean %.5f min %.5f max %.5f" % (tb.mean(), tb.min(), tb.max()))
        print("  ort prev  ", ob.shape, "mean %.5f min %.5f max %.5f" % (ob.mean(), ob.min(), ob.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
