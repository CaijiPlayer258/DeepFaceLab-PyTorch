import torch
import torch.nn as torch_nn
import numpy as np
from core.leras.nn import nn


def _apply_initializer_2d_(param: torch_nn.Parameter, initializer, dtype: torch.dtype):
    if initializer is None:
        return False
    try:
        init_val = initializer(tuple(param.shape), dtype=dtype)
    except TypeError:
        init_val = initializer(tuple(param.shape))

    if isinstance(init_val, torch.Tensor):
        src = init_val.to(device=param.device, dtype=param.dtype).reshape(param.shape)
        param.data.copy_(src)
        return True
    if isinstance(init_val, np.ndarray):
        src = torch.from_numpy(init_val).to(device=param.device, dtype=param.dtype).reshape(param.shape)
        param.data.copy_(src)
        return True
    return False

class AdaIN(nn.LayerBase):
    """
        DeepFaceLab-master semantics:
        inputs: (x, mlp)
            x   : feature map (N,C,H,W) or (N,H,W,C) depending on nn.data_format
            mlp : style vector (N, mlp_ch)
    """
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None, dtype=None, name=None, **kwargs):
        self.in_ch = in_ch
        self.mlp_ch = mlp_ch
        self.kernel_initializer = kernel_initializer

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') else torch.float32
        self.dtype = dtype

        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        self.weight1 = torch_nn.Parameter(torch.empty(self.mlp_ch, self.in_ch, dtype=self.dtype, device=dev))
        self.bias1 = torch_nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype, device=dev))
        self.weight2 = torch_nn.Parameter(torch.empty(self.mlp_ch, self.in_ch, dtype=self.dtype, device=dev))
        self.bias2 = torch_nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype, device=dev))

        kernel_initializer = self.kernel_initializer
        if kernel_initializer is None:
            # TF he_normal
            torch_nn.init.kaiming_normal_(self.weight1, nonlinearity='relu')
            torch_nn.init.kaiming_normal_(self.weight2, nonlinearity='relu')
        else:
            _apply_initializer_2d_(self.weight1, kernel_initializer, self.dtype)
            _apply_initializer_2d_(self.weight2, kernel_initializer, self.dtype)

    def forward(self, inputs):
        x, mlp = inputs

        gamma = torch.matmul(mlp, self.weight1) + self.bias1.view(1, self.in_ch)
        beta = torch.matmul(mlp, self.weight2) + self.bias2.view(1, self.in_ch)

        if nn.data_format == "NHWC":
            shape = (-1, 1, 1, self.in_ch)
        else:
            shape = (-1, self.in_ch, 1, 1)

        x_mean = torch.mean(x, dim=nn.conv2d_spatial_axes, keepdim=True)
        x_std = torch.std(x, dim=nn.conv2d_spatial_axes, keepdim=True, unbiased=False) + 1e-5

        x = (x - x_mean) / x_std
        x = x * gamma.view(shape)
        x = x + beta.view(shape)
        return x

nn.AdaIN = AdaIN