import math
import torch
import torch.nn as torch_nn
from core.leras.nn import nn


class ECA(nn.LayerBase):
    """
    Efficient Channel Attention (ECANet, CVPR 2020).
    1D convolution across channel dimension with adaptive kernel size.

    kernel_size k = |log2(C)/γ + b/γ|_odd, γ=2, b=1
    """
    def __init__(self, in_ch, name=None, **kwargs):
        self.in_ch = int(in_ch)
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        k = int(abs(math.log2(self.in_ch) / 2.0 + 0.5))
        if k % 2 == 0:
            k += 1
        k = max(k, 3)
        self.kernel_size = k

        dev = getattr(nn, 'device', None)
        self.conv1d = torch_nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.conv1d.to(device=dev)

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            gap = torch.mean(x, dim=[1, 2], keepdim=False)
        else:
            gap = torch.mean(x, dim=[2, 3], keepdim=False)

        gap = gap.unsqueeze(1)
        attn = self.conv1d(gap)
        attn = attn.squeeze(1)
        attn = torch.sigmoid(attn)

        if nhwc:
            attn = attn.unsqueeze(1).unsqueeze(1)
        else:
            attn = attn.unsqueeze(2).unsqueeze(3)

        return x * attn


nn.ECA = ECA
