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

class Conv2DTranspose(nn.LayerBase):
    """
    use_wscale      启用权重缩放（均衡学习率）
                    如果kernel_initializer为None，将强制使用random_normal
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs ):
        if not isinstance(strides, int):
            raise ValueError ("strides必须是整数类型")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') and nn.floatx is not None else torch.float32

        # Compute base padding; output_padding is computed dynamically to match DFL deconv_length.
        if padding == 'SAME':
            self.padding = (kernel_size - 1) // 2
        elif padding == 'VALID':
            self.padding = 0
        else:
            self.padding = 0

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_mode = padding
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.dtype = dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # PyTorch ConvTranspose2d使用(in_ch, out_ch, H, W)格式
        dev = getattr(nn, 'device', None)
        self.weight = torch_nn.Parameter(
            torch.empty(self.in_ch, self.out_ch, self.kernel_size, self.kernel_size, dtype=self.dtype, device=dev)
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
            self.bias = torch_nn.Parameter(torch.empty(self.out_ch, dtype=self.dtype, device=dev))
            bias_initializer = self.bias_initializer
            if bias_initializer is None:
                torch_nn.init.zeros_(self.bias)
            else:
                _apply_initializer_(self.bias, bias_initializer, self.dtype)
        else:
            self.bias = None

    def forward(self, x):
        nhwc = (nn.data_format == 'NHWC')
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # Match TF deconv_length output sizing.
        in_h, in_w = int(x.shape[2]), int(x.shape[3])

        target_h = self.deconv_length(in_h, self.strides, self.kernel_size, self.padding_mode)
        target_w = self.deconv_length(in_w, self.strides, self.kernel_size, self.padding_mode)

        # PyTorch formula: out = (in-1)*s - 2*p + k + output_padding
        base_h = (in_h - 1) * self.strides - 2 * self.padding + self.kernel_size
        base_w = (in_w - 1) * self.strides - 2 * self.padding + self.kernel_size
        out_pad_h = int(target_h - base_h)
        out_pad_w = int(target_w - base_w)
        if out_pad_h < 0 or out_pad_h >= self.strides:
            out_pad_h = max(0, min(out_pad_h, self.strides - 1))
        if out_pad_w < 0 or out_pad_w >= self.strides:
            out_pad_w = max(0, min(out_pad_w, self.strides - 1))

        x = F.conv_transpose2d(
            x,
            weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.padding,
            output_padding=(out_pad_h, out_pad_w),
        )

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "

        return r

    def deconv_length(self, dim_size, stride_size, kernel_size, padding):
        assert padding in {'SAME', 'VALID', 'FULL'}
        if dim_size is None:
            return None
        if padding == 'VALID':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'FULL':
            dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
        elif padding == 'SAME':
            dim_size = dim_size * stride_size
        return dim_size


nn.Conv2DTranspose = Conv2DTranspose
