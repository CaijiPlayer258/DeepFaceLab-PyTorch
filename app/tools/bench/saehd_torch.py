# -*- coding: utf-8 -*-
"""SAEHD(df-udt) 前向 —— 纯 torch 复刻，只为 inter 换分辨率实验服务。

为什么不用 app/engine 里的 leras：那套要 nn.initialize()/device 一堆前置，
这里只要 Encoder/Inter/Decoder 三段，照着 core/leras/archis/DeepFakeArchi.py
（opts 含 't','d','u'）逐层写死更省事，也不会碰训练侧的副作用。

权重来源：DFL 的 *.pth 是 **zip 里的 param_N.npy**（无 pickle），
按形状唯一匹配到层，匹配不上就报错（不猜）。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 416
# 可变开关（只给实验用）
PIXEL_NORM = 1
DENSE_ACT = 0
FLATTEN_HWC = 0   # 0 = 通道优先(c,h,w)；1 = (h,w,c)
E_CH = 80
AE_CH = 320
D_CH = 80
D_MASK_CH = 26


def _lrelu(x, s):
    return F.leaky_relu(x, s)


def _same_pad(x, k, stride):
    """TF 'SAME'：pad 总量 = max((out-1)*s + k - in, 0)，多出来的一侧放在**后边**。
    PyTorch 只能对称 pad，所以要自己补。416/k5/s2 时是 (1,2)/(1,2)，和 k//2=2 不一样。"""
    import math
    n, c, h, w = x.shape
    oh, ow = math.ceil(h / stride), math.ceil(w / stride)
    ph = max((oh - 1) * stride + k - h, 0)
    pw = max((ow - 1) * stride + k - w, 0)
    if ph or pw:
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    return x


class Conv(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, stride=stride, padding=0)
        self.k = k
        self.stride = stride

    def forward(self, x):
        return self.c(_same_pad(x, self.k, self.stride))


def depth_to_space_tf(x, r):
    """TF(NHWC) 的 depth_to_space 通道次序 ≠ torch.pixel_shuffle(NCHW)。"""
    b, c, h, w = x.shape
    oc = c // (r * r)
    x = x.view(b, oc, r, r, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, oc, h * r, w * r)


class Down(nn.Module):
    def __init__(self, cin, cout, k=5):
        super().__init__()
        self.conv = Conv(cin, cout, k, 2)

    def forward(self, x):
        return _lrelu(self.conv(x), 0.1)


class Res(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.conv1 = Conv(ch, ch, k)
        self.conv2 = Conv(ch, ch, k)

    def forward(self, x):
        y = _lrelu(self.conv1(x), 0.2)
        y = self.conv2(y)
        return _lrelu(x + y, 0.2)


class Up(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = Conv(cin, cout * 4, k)

    def forward(self, x):
        x = _lrelu(self.conv(x), 0.1)
        return depth_to_space_tf(x, 2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3, E_CH, 5)
        self.res1 = Res(E_CH, 3)
        self.down2 = Down(E_CH, E_CH * 2, 5)
        self.down3 = Down(E_CH * 2, E_CH * 4, 5)
        self.down4 = Down(E_CH * 4, E_CH * 8, 5)
        self.down5 = Down(E_CH * 8, E_CH * 8, 5)
        self.res5 = Res(E_CH * 8, 3)

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res5(x)
        if FLATTEN_HWC:
            x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        else:
            x = x.reshape(x.shape[0], -1)
        # 'u' → pixel_norm（和 iperov 一致，eps=1e-6）
        if PIXEL_NORM:
            x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return x


class Inter(nn.Module):
    def __init__(self, res=RES):
        super().__init__()
        self.res = res
        low = res // 32
        self.dense1 = nn.Linear(E_CH * 8 * low * low, AE_CH)
        self.dense2 = nn.Linear(AE_CH, low * low * AE_CH)
        self.low = low

    def forward(self, x):
        x = self.dense1(x)
        if DENSE_ACT:
            x = _lrelu(x, 0.1)
        x = self.dense2(x)
        if FLATTEN_HWC:
            return x.reshape(-1, self.low, self.low, AE_CH).permute(0, 3, 1, 2)
        return x.reshape(-1, AE_CH, self.low, self.low)


class Decoder(nn.Module):
    def __init__(self, with_mask=True):
        super().__init__()
        self.up0 = Up(AE_CH, D_CH * 8, 3)
        self.up1 = Up(D_CH * 8, D_CH * 8, 3)
        self.up2 = Up(D_CH * 8, D_CH * 4, 3)
        self.up3 = Up(D_CH * 4, D_CH * 2, 3)
        self.res0 = Res(D_CH * 8, 3)
        self.res1 = Res(D_CH * 8, 3)
        self.res2 = Res(D_CH * 4, 3)
        self.res3 = Res(D_CH * 2, 3)
        self.out_conv = Conv(D_CH * 2, 3, 1)
        self.out_conv1 = Conv(D_CH * 2, 3, 3)
        self.out_conv2 = Conv(D_CH * 2, 3, 3)
        self.out_conv3 = Conv(D_CH * 2, 3, 3)
        self.upscalem0 = Up(AE_CH, D_MASK_CH * 8, 3)
        self.upscalem1 = Up(D_MASK_CH * 8, D_MASK_CH * 8, 3)
        self.upscalem2 = Up(D_MASK_CH * 8, D_MASK_CH * 4, 3)
        self.upscalem3 = Up(D_MASK_CH * 4, D_MASK_CH * 2, 3)
        # 遮罩头还要再上一档：208²×52 → 416²×26，然后 1×1 到单通道
        self.upscalem4 = Up(D_MASK_CH * 2, D_MASK_CH, 3)
        self.out_convm = Conv(D_MASK_CH, 1, 1)

    def forward(self, z):
        x = self.up0(z)
        x = self.res0(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = self.up3(x)
        x = self.res3(x)
        x = torch.cat((self.out_conv(x), self.out_conv1(x),
                       self.out_conv2(x), self.out_conv3(x)), 1)
        out = torch.sigmoid(depth_to_space_tf(x, 2))
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = self.upscalem3(m)
        m = self.upscalem4(m)
        mask = torch.sigmoid(self.out_convm(m))
        return out, mask


def read_pth(path):
    out = []
    with zipfile.ZipFile(str(path)) as zf:
        names = sorted(zf.namelist(), key=lambda n: int("".join(ch for ch in n if ch.isdigit()) or 0))
        for nm in names:
            with zf.open(nm) as f:
                out.append(np.lib.format.read_array(f, allow_pickle=False))
    return out


def assign(module, arrays, label):
    """按形状唯一匹配把 numpy 权重塞进 module（不依赖存储顺序）。"""
    used = [False] * len(arrays)
    misses = []
    for name, p in module.state_dict().items():
        want = tuple(p.shape)
        idx = None
        for i, a in enumerate(arrays):
            if used[i]:
                continue
            if tuple(a.shape) == want:
                idx = i
                break
        if idx is None and len(want) == 2:          # Dense 存的是 (in, out)，Linear 要 (out, in)
            for i, a in enumerate(arrays):
                if used[i] or a.ndim != 2:
                    continue
                if tuple(a.shape) == want[::-1]:
                    idx = i
                    break
        if idx is None:
            misses.append((name, want))
            continue
        used[idx] = True
        arr = arrays[idx]
        if tuple(arr.shape) != want:
            arr = arr.T
        with torch.no_grad():
            p.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
    left = [i for i, u in enumerate(used) if not u]
    if misses or left:
        raise SystemExit("%s 对不上：缺 %s / 多 %d 个" % (label, misses[:3], len(left)))
    return len(list(module.state_dict()))


# ---- 按 DFL 的**存储顺序**装配（形状有重复，只能靠顺序，不能靠形状）----
ENC_ORDER = ["down1.conv.c", "res1.conv1.c", "res1.conv2.c",
             "down2.conv.c", "down3.conv.c", "down4.conv.c", "down5.conv.c",
             "res5.conv1.c", "res5.conv2.c"]
DEC_ORDER = ["up0.conv.c", "up1.conv.c", "up2.conv.c", "up3.conv.c",
             "res0.conv1.c", "res0.conv2.c", "res1.conv1.c", "res1.conv2.c",
             "res2.conv1.c", "res2.conv2.c", "res3.conv1.c", "res3.conv2.c",
             "upscalem0.conv.c", "upscalem1.conv.c", "upscalem2.conv.c", "upscalem3.conv.c",
             "out_conv.c", "out_conv1.c", "out_conv2.c", "out_conv3.c",
             "upscalem4.conv.c", "out_convm.c"]


def _get(mod, path):
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def assign_ordered(module, arrays, order, label):
    """按 (权重, 偏置) 成对顺序装配；形状必须对得上（含 Dense 的转置）。"""
    if len(arrays) != len(order) * 2:
        got, want = len(arrays), len(order) * 2
        raise SystemExit("%s 参数个数不符：文件 %d，按顺序要 %d" % (label, got, want))
    for i, path in enumerate(order):
        conv = _get(module, path)
        w, b = arrays[i * 2], arrays[i * 2 + 1]
        ww = w if tuple(w.shape) == tuple(conv.weight.shape) else w.T
        if tuple(ww.shape) != tuple(conv.weight.shape):
            raise SystemExit("%s 第 %d 个（%s）形状不符：%s vs %s"
                             % (label, i, path, w.shape, tuple(conv.weight.shape)))
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(ww)))
            conv.bias.copy_(torch.from_numpy(np.ascontiguousarray(b)))
    return len(order)


def load_models(model_dir="workspace/model", name="New416_SAEHD", res=RES):
    d = Path(model_dir) / "workspace" / "model"
    enc = Encoder()
    inter = Inter(res)
    dec = Decoder()
    n1 = assign_ordered(enc, read_pth(d / (name + "_encoder.pth")), ENC_ORDER, "encoder")
    n3 = assign_ordered(dec, read_pth(d / (name + "_decoder_src.pth")), DEC_ORDER, "decoder_src")
    ia = read_pth(d / (name + "_inter.pth"))
    with torch.no_grad():
        inter.dense1.weight.copy_(torch.from_numpy(ia[0].T.copy()))
        inter.dense1.bias.copy_(torch.from_numpy(ia[1].copy()))
        inter.dense2.weight.copy_(torch.from_numpy(ia[2].T.copy()))
        inter.dense2.bias.copy_(torch.from_numpy(ia[3].copy()))
    n2 = 4
    return enc.eval(), inter.eval(), dec.eval(), (n1, n2, n3)


if __name__ == "__main__":
    e, i, d, counts = load_models("D:/MyProject/DeepFaceLab-PyTorch-4.1")
    print("assign ok:", counts)
    print("params enc/inter/dec =", sum(p.numel() for p in e.parameters()),
          sum(p.numel() for p in i.parameters()), sum(p.numel() for p in d.parameters()))
