"""
XSegLite: Lightweight CNN XSeg. Conv-BN-ReLU throughout, no aux heads.
  python core/xseglite_torch.py          # smoke test
"""

import math
import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    """Conv-BN-ReLU: BN fuses into Conv at inference, zero-cost."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ECA(nn.Module):
    """Efficient Channel Attention – 1d conv on GAP weights."""
    def __init__(self, ch):
        super().__init__()
        k = int(abs(math.log2(ch) / 2.0 + 0.5))
        if k % 2 == 0: k += 1
        k = max(k, 3)
        self.conv1d = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        gap = x.mean(dim=[2, 3], keepdim=False).unsqueeze(1)
        a = self.conv1d(gap).squeeze(1).sigmoid()
        return x * a.unsqueeze(2).unsqueeze(3)


class DSCPool(nn.Module):
    """Depthwise-separable 2× down-sample."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class XSegLiteTorch(nn.Module):
    """
    XSegLite: Conv-BN-ReLU 4-stage U-Net + ECA.
    No aux heads. Fully convolutional, any resolution.
    """
    def __init__(self, in_ch=3, base_ch=32, use_eca=True, n_stages=4):
        super().__init__()
        n = n_stages

        def block(ci, co):
            return nn.Sequential(conv3x3(ci, co))

        def enc_stage(ci, co, n_blk):
            blks = []
            for i in range(n_blk):
                _co = co if i == n_blk - 1 else ci
                blks.append(block(ci, _co))
                ci = _co
            return nn.ModuleList(blks), DSCPool(co, co)

        def dec_stage(ci, co, skip_ch, n_blk):
            parts = nn.ModuleDict()
            parts['up'] = nn.Sequential(
                nn.ConvTranspose2d(ci, ci, 3, 2, 1, output_padding=1, bias=False),
                nn.BatchNorm2d(ci),
                nn.ReLU(inplace=True),
            )
            concat_ch = ci + skip_ch
            blks = []
            _ci = concat_ch
            for i in range(n_blk):
                _co = ci if i == 0 else (co if i == n_blk - 1 else _ci)
                blks.append(block(_ci, _co))
                _ci = _co
            parts['blks'] = nn.ModuleList(blks)
            return parts

        # 4-stage config
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8]
        n_enc = [2, 2, 2, 3]
        bridge_ch = base_ch*8
        d_ci = [bridge_ch, base_ch*4, base_ch*2, base_ch]
        d_co = [base_ch*4, base_ch*2, base_ch,   base_ch]
        d_sk = [bridge_ch, base_ch*4, base_ch*2, base_ch]
        d_n = [3, 2, 2, 2]

        # encoder
        self.enc_blks = nn.ModuleList()
        self.enc_pool = nn.ModuleList()
        ci = in_ch
        for co, nb in zip(chs, n_enc):
            blks, pool = enc_stage(ci, co, nb)
            self.enc_blks.append(blks)
            self.enc_pool.append(pool)
            ci = co

        # bottleneck
        self.bridge = block(bridge_ch, bridge_ch)
        self.bridge_eca = ECA(bridge_ch) if use_eca else nn.Identity()

        # decoder
        self.dec_parts = nn.ModuleList()
        for ci_, co_, sk_, nb in zip(d_ci, d_co, d_sk, d_n):
            self.dec_parts.append(dec_stage(ci_, co_, sk_, nb))

        # output
        self.out_conv = nn.Conv2d(base_ch, 1, 3, 1, 1, bias=True)

    def forward(self, x):
        # encoder
        skips = []
        for blks, pool in zip(self.enc_blks, self.enc_pool):
            for b in blks:
                x = b(x)
            skips.append(x)
            x = pool(x)

        # bottleneck
        x = self.bridge_eca(self.bridge(x))

        # decoder
        skips.reverse()
        for idx, parts in enumerate(self.dec_parts):
            s = skips[idx]
            x = parts['up'](x)
            x = torch.cat([x, s], dim=1)
            for b in parts['blks']:
                x = b(x)

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)


if __name__ == '__main__':
    m = XSegLiteTorch(3, 32)
    x = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        lo, pr = m(x)
    print(f'logits {lo.shape}  pred {pr.shape}')
    print(f'params {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
