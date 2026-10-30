import torch
import torch.nn as torch_nn
from core.leras.nn import nn

class InstanceNorm2D(nn.LayerBase):
    def __init__(self, in_ch, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        # DFL TF version uses glorot_uniform for (C,) scale.
        self.weight = torch_nn.Parameter(torch.empty(self.in_ch, dtype=self.dtype, device=dev))
        torch_nn.init.xavier_uniform_(self.weight.view(1, -1))
        self.weight.data = self.weight.data.view(-1)
        self.bias = torch_nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype, device=dev))

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1, 1, 1, self.in_ch)
        else:
            shape = (1, self.in_ch, 1, 1)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)

        spatial_axes = nn.conv2d_spatial_axes
        x_mean = x.mean(dim=spatial_axes, keepdim=True)
        # TF reduce_std uses population std (unbiased=False), then + 1e-5 (outside sqrt).
        x_std = x.std(dim=spatial_axes, keepdim=True, unbiased=False) + 1e-5

        x = (x - x_mean) / x_std
        x = x * weight + bias
        return x

nn.InstanceNorm2D = InstanceNorm2D