import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

class BlurPool(nn.LayerBase):
    """
    模糊池化层，用于抗锯齿下采样
    """
    def __init__(self, filt_size=3, stride=2, name=None, **kwargs):
        self.filt_size = filt_size
        self.stride = stride
        
        # 计算padding
        pad = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad0 = pad[0]
        self.pad1 = pad[1]

        # 生成滤波器系数
        if self.filt_size == 1:
            a = np.array([1.,])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:, None] * a[None, :]
        a = a / np.sum(a)
        self.a = a
        
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # 创建不可训练的卷积核
        a = self.a.astype(np.float32)
        # Base kernel in torch layout (1,1,H,W); later repeated to (C,1,H,W)
        self.register_buffer('k', torch.from_numpy(a[None, None, :, :]))

    def forward(self, x):
        nhwc = (nn.data_format == 'NHWC')
        if nhwc:
            x = x.permute(0, 3, 1, 2).contiguous()

        ch = int(x.shape[1])
        k = self.k.to(device=x.device, dtype=x.dtype).repeat(ch, 1, 1, 1)

        # Match TF asymmetric padding.
        x = F.pad(x, (self.pad0, self.pad1, self.pad0, self.pad1), mode='constant', value=0.0)
        x = F.conv2d(x, k, stride=self.stride, groups=ch)

        if nhwc:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x
nn.BlurPool = BlurPool