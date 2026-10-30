import torch
import torch.nn as torch_nn
from core.leras.nn import nn

class ScaleAdd(nn.LayerBase):
    """
    缩放加法层：x0 + x1 * weight
    """
    def __init__(self, ch, dtype=None, name=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype
        self.ch = ch

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        self.weight = torch_nn.Parameter(torch.zeros(self.ch, dtype=self.dtype, device=dev))

    def forward(self, inputs):
        if nn.data_format == "NHWC":
            shape = (1, 1, 1, self.ch)
        else:
            shape = (1, self.ch, 1, 1)
        weight = self.weight.view(shape)

        x0, x1 = inputs
        x = x0 + x1 * weight

        return x
nn.ScaleAdd = ScaleAdd