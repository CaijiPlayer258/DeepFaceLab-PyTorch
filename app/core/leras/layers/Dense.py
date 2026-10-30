import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from core.leras.nn import nn


def _apply_initializer_(param: torch_nn.Parameter, initializer, dtype: torch.dtype):
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

class Dense(nn.LayerBase):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
        """
        use_wscale          enables weight scale (equalized learning rate)
        maxout_ch           https://link.springer.com/article/10.1186/s40537-019-0233-0
                            typical 2-4 if you want to enable DenseMaxout behaviour
        """
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') and nn.floatx is not None else torch.float32

        self.dtype = dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        if self.maxout_ch > 1:
            weight_shape = (self.in_ch, self.out_ch*self.maxout_ch)
        else:
            weight_shape = (self.in_ch, self.out_ch)

        self.weight = torch_nn.Parameter(torch.empty(*weight_shape, dtype=self.dtype, device=dev))
        kernel_initializer = self.kernel_initializer

        if self.use_wscale:
            gain = 1.0
            fan_in = weight_shape[0]
            he_std = gain / np.sqrt(fan_in)
            self.wscale = he_std
            if kernel_initializer is None:
                torch_nn.init.normal_(self.weight, mean=0.0, std=1.0)
            else:
                _apply_initializer_(self.weight, kernel_initializer, self.dtype)
        else:
            if kernel_initializer is None:
                torch_nn.init.xavier_uniform_(self.weight)
            else:
                _apply_initializer_(self.weight, kernel_initializer, self.dtype)

        if self.use_bias:
            self.bias = torch_nn.Parameter(torch.empty(self.out_ch, dtype=self.dtype, device=dev))
            bias_initializer = self.bias_initializer
            if bias_initializer is None:
                torch_nn.init.zeros_(self.bias)
            else:
                _apply_initializer_(self.bias, bias_initializer, self.dtype)
        else:
            self.bias = None

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        x = F.linear(x, weight.t())

        if self.maxout_ch > 1:
            x = x.view(-1, self.out_ch, self.maxout_ch)
            x = x.max(dim=-1)[0]

        if self.use_bias:
            x = x + self.bias.view(1, self.out_ch)

        return x


nn.Dense = Dense
