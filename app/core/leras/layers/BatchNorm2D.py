import torch
import torch.nn as torch_nn
from core.leras.nn import nn

class BatchNorm2D(nn.LayerBase):
    """
    BatchNorm2D for PyTorch
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, name=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        self.weight = torch_nn.Parameter(torch.ones(self.dim, dtype=self.dtype, device=dev))
        self.bias = torch_nn.Parameter(torch.zeros(self.dim, dtype=self.dtype, device=dev))
        self.register_buffer('running_mean', torch.zeros(self.dim, dtype=self.dtype, device=dev))
        # DFL TF version initializes running_var to zeros and treats BN as inference-only.
        self.register_buffer('running_var', torch.zeros(self.dim, dtype=self.dtype, device=dev))

    def get_weights(self):
        # Match DeepFaceLab-master behavior: BN saves weight/bias + running stats.
        # Keep running_* as buffers (not trainable), but include them in Saveable IO.
        return [self.weight, self.bias, self.running_mean, self.running_var]

    def forward(self, x):
        # Match DeepFaceLab-master: no running stats updates ("currently not for training").
        if nn.data_format == "NHWC":
            shape = (1, 1, 1, self.dim)
        else:
            shape = (1, self.dim, 1, 1)

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)
        running_mean = self.running_mean.view(shape)
        running_var = self.running_var.view(shape)

        x = (x - running_mean) / torch.sqrt(running_var + self.eps)
        x = x * weight + bias
        return x

nn.BatchNorm2D = BatchNorm2D