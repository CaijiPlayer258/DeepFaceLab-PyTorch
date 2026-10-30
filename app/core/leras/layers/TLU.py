import torch
import torch.nn as torch_nn
from core.leras.nn import nn

class TLU(nn.LayerBase):
    """
    阈值线性单元（Thresholded Linear Unit）
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
        self.tau = torch_nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype, device=dev))

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1, 1, 1, self.in_ch)
        else:
            shape = (1, self.in_ch, 1, 1)
        tau = self.tau.view(shape)
        return torch.maximum(x, tau)
nn.TLU = TLU