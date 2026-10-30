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

class DepthwiseConv2D(nn.LayerBase):
    """
    深度可分离卷积
    use_wscale  启用权重缩放（均衡学习率）
    """
    def __init__(self, in_ch, kernel_size, strides=1, padding='SAME', depth_multiplier=1, dilations=1, 
                 use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, 
                 trainable=True, dtype=None, name=None, **kwargs):
        if not isinstance(strides, int):
            raise ValueError("步长必须是整数类型")
        if not isinstance(dilations, int):
            raise ValueError("膨胀率必须是整数类型")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') and nn.floatx is not None else torch.float32

        if isinstance(padding, str):
            if padding == "SAME":
                padding = ((kernel_size - 1) * dilations + 1) // 2
            elif padding == "VALID":
                padding = 0
            else:
                raise ValueError("错误的padding类型，应为VALID、SAME或整数")

        self.in_ch = in_ch
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        dev = getattr(nn, 'device', None)
        # PyTorch depthwise conv使用(in_ch * depth_multiplier, 1, H, W)格式
        self.weight = torch_nn.Parameter(
            torch.empty(
                self.in_ch * self.depth_multiplier,
                1,
                self.kernel_size,
                self.kernel_size,
                dtype=self.dtype,
                device=dev,
            )
        )

        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
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
            self.bias = torch_nn.Parameter(torch.empty(self.in_ch * self.depth_multiplier, dtype=self.dtype, device=dev))
            bias_initializer = self.bias_initializer
            if bias_initializer is None:
                torch_nn.init.zeros_(self.bias)
            else:
                _apply_initializer_(self.bias, bias_initializer, self.dtype)
        else:
            self.bias = None

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # PyTorch的depthwise卷积使用groups=in_ch
        x = F.conv2d(x, weight, bias=self.bias, stride=self.strides, 
                    padding=self.padding, dilation=self.dilations, groups=self.in_ch)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x


nn.DepthwiseConv2D = DepthwiseConv2D

