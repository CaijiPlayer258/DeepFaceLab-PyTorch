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

class Conv2D(nn.LayerBase):
    """
    PyTorch Conv2D implementation
    use_wscale  bool enables equalized learning rate
    """
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, name=None, **kwargs):
        if not isinstance(strides, int):
            raise ValueError ("strides must be an int type")
        if not isinstance(dilations, int):
            raise ValueError ("dilations must be an int type")
        kernel_size = int(kernel_size)

        if dtype is None:
            dtype = nn.floatx if hasattr(nn, 'floatx') and nn.floatx is not None else torch.float32

        self.pad_4 = None
        self._same_padding = False  # TF-style SAME padding flag
        if isinstance(padding, str):
            if padding == "SAME":
                if strides > 1 or dilations > 1:
                    # TF SAME padding depends on input size (computed in forward)
                    self._same_padding = True
                    padding = 0
                else:
                    padding = (((kernel_size - 1) * dilations + 1) // 2)
            elif padding == "VALID":
                padding = 0
            else:
                raise ValueError("Wrong padding type. Should be VALID SAME or INT")
        elif isinstance(padding, (list, tuple)):
            if len(padding) != 4:
                raise ValueError("Wrong padding type. Should be VALID SAME or INT or 4x INTs")
            self.pad_4 = tuple(int(x) for x in padding)
            padding = 0
        else:
            padding = int(padding)

        self.in_ch = in_ch
        self.out_ch = out_ch
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
        # PyTorch uses (out_ch, in_ch, H, W) format
        self.weight = torch_nn.Parameter(
            torch.empty(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size, dtype=self.dtype, device=dev)
        )

        kernel_initializer = self.kernel_initializer
        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = he_std
            if kernel_initializer is None:
                # DFL: force random_normal when wscale enabled.
                torch_nn.init.normal_(self.weight, mean=0.0, std=1.0)
            else:
                _apply_initializer_(self.weight, kernel_initializer, self.dtype)
        else:
            if kernel_initializer is None:
                # DFL TF default initializer (tf.get_variable default) ~= glorot_uniform.
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
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale

        # Match TF behavior: explicit pad then VALID conv.
        if self._same_padding:
            # 原版 iperov DeepFakeArchi：SAME = 对称 pad ((k-1)*d+1)//2 + VALID conv
            # （不是 TF 自动 SAME 的不对称 pad！stride>1 时也必须对称）
            pad = ((self.kernel_size - 1) * self.dilations + 1) // 2
            x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)
            padding = 0
        elif self.pad_4 is not None:
            pt, pb, pl, pr = self.pad_4
            x = F.pad(x, (pl, pr, pt, pb), mode='constant', value=0.0)
            padding = 0
        elif self.padding != 0:
            p = int(self.padding)
            x = F.pad(x, (p, p, p, p), mode='constant', value=0.0)
            padding = 0
        else:
            padding = 0

        x = F.conv2d(x, weight, bias=self.bias, stride=self.strides, padding=padding, dilation=self.dilations)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def __str__(self):
        r = f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
        return r


nn.Conv2D = Conv2D
