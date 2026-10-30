import torch
import torch.nn as torch_nn
from core.leras.nn import nn

class FRNorm2D(nn.LayerBase):
    """
    滤波器响应归一化层实现
    Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        self.weight = torch_nn.Parameter(torch.ones(self.in_ch, dtype=self.dtype, device=dev))
        self.bias = torch_nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype, device=dev))
        self.eps = torch_nn.Parameter(torch.tensor([1e-6], dtype=self.dtype, device=dev))

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1, 1, 1, self.in_ch)
        else:
            shape = (1, self.in_ch, 1, 1)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)

        nu2 = torch.mean(torch.square(x), dim=nn.conv2d_spatial_axes, keepdim=True)
        x = x * (1.0 / torch.sqrt(nu2 + torch.abs(self.eps)))
        return x * weight + bias
nn.FRNorm2D = FRNorm2D