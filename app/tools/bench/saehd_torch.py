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
E_CH = 80
AE_CH = 320
D_CH = 80
D_MASK_CH = 26


def _lrelu(x, s):
    return F.leaky_relu(x, s)


class Conv(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2)

    def forward(self, x):
        return self.c(x)


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
        return F.pixel_shuffle(x, 2)


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
        x = x.reshape(x.shape[0], -1)
        # 'u' → pixel_norm（和 iperov 一致，eps=1e-6）
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
        x = self.dense2(x)
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
        out = torch.sigmoid(F.pixel_shuffle(x, 2))
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


def load_models(model_dir="workspace/model", name="New416_SAEHD", res=RES):
    d = Path(model_dir) / "workspace" / "model"
    enc = Encoder()
    inter = Inter(res)
    dec = Decoder()
    n1 = assign(enc, read_pth(d / (name + "_encoder.pth")), "encoder")
    n2 = assign(inter, read_pth(d / (name + "_inter.pth")), "inter")
    n3 = assign(dec, read_pth(d / (name + "_decoder_src.pth")), "decoder_src")
    return enc.eval(), inter.eval(), dec.eval(), (n1, n2, n3)


if __name__ == "__main__":
    e, i, d, counts = load_models("D:/MyProject/DeepFaceLab-PyTorch-4.1")
    print("assign ok:", counts)
    print("params enc/inter/dec =", sum(p.numel() for p in e.parameters()),
          sum(p.numel() for p in i.parameters()), sum(p.numel() for p in d.parameters()))
