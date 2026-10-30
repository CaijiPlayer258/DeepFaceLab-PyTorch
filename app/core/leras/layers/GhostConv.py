import math
import torch
import torch.nn.functional as F
from core.leras.nn import nn


class GhostModule(nn.LayerBase):
    """
    Ghost Module from GhostNet (CVPR 2020).
    Generates half the features via cheap depthwise ops to reduce FLOPs.

    Primary: 1x1 conv → intrinsic (out_ch * ratio)
    Cheap:   3x3 DW on intrinsic → ghost (out_ch - intrinsic)
    Concat → 1x1 fusion → FRN + TLU
    """
    def __init__(self, in_ch, out_ch, ghost_ratio=0.5, kernel_size=3, strides=1,
                 name=None, dtype=None, **kwargs):
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.ghost_ratio = float(ghost_ratio)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        intrinsic = max(1, int(self.out_ch * self.ghost_ratio))
        cheap_ch = intrinsic  # DWConv with depth_multiplier=1 keeps channels

        self.primary_conv = nn.Conv2D(self.in_ch, intrinsic, kernel_size=1, padding='SAME')
        self.primary_frn  = nn.FRNorm2D(in_ch=intrinsic)
        self.primary_tlu  = nn.TLU(in_ch=intrinsic)

        self.cheap_conv = nn.DepthwiseConv2D(intrinsic, kernel_size=self.kernel_size, strides=self.strides)
        self.cheap_frn  = nn.FRNorm2D(in_ch=intrinsic)
        self.cheap_tlu  = nn.TLU(in_ch=intrinsic)

        self.fusion_conv = nn.Conv2D(intrinsic + cheap_ch, self.out_ch, kernel_size=1, padding='SAME')
        self.fusion_frn  = nn.FRNorm2D(in_ch=self.out_ch)
        self.fusion_tlu  = nn.TLU(in_ch=self.out_ch)

        if self.in_ch != self.out_ch or self.strides != 1:
            self.shortcut_conv = nn.Conv2D(self.in_ch, self.out_ch, kernel_size=1, padding='SAME',
                                           strides=self.strides)
            self.shortcut_frn  = nn.FRNorm2D(in_ch=self.out_ch)
            self.shortcut_tlu  = nn.TLU(in_ch=self.out_ch)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        identity = x

        intrinsic = self.primary_tlu(self.primary_frn(self.primary_conv(x)))
        ghost = self.cheap_tlu(self.cheap_frn(self.cheap_conv(intrinsic)))

        ch_axis = nn.conv2d_ch_axis
        out = torch.cat([intrinsic, ghost], dim=ch_axis)
        out = self.fusion_tlu(self.fusion_frn(self.fusion_conv(out)))

        if self.shortcut_conv is not None:
            identity = self.shortcut_tlu(self.shortcut_frn(self.shortcut_conv(identity)))

        return out + identity


class DFC(nn.LayerBase):
    """
    Simplified Decoupled Fully Connected attention from GhostNetV2.
    GlobalAvgPool → 1×1 Conv → HardSigmoid → channel-wise re-weight.
    """
    def __init__(self, in_ch, name=None, **kwargs):
        self.in_ch = int(in_ch)
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.conv = nn.Conv2D(self.in_ch, self.in_ch, kernel_size=1, padding='SAME')

    def forward(self, x):
        nhwc = (nn.data_format == "NHWC")
        if nhwc:
            spatial_dims = [1, 2]
        else:
            spatial_dims = [2, 3]
        gap = torch.mean(x, dim=spatial_dims, keepdim=True)
        attn = self.conv(gap)
        attn = F.hardsigmoid(attn)
        return x * attn


class DSCDownsample(nn.LayerBase):
    """
    Depthwise Separable Convolution for 2× downsampling.
    3×3 DWConv stride=2 → FRN+TLU → 1×1 Pointwise → FRN+TLU.
    """
    def __init__(self, in_ch, out_ch=None, kernel_size=3, name=None, **kwargs):
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch) if out_ch is not None else self.in_ch
        self.kernel_size = int(kernel_size)
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.dw_conv = nn.DepthwiseConv2D(self.in_ch, kernel_size=self.kernel_size, strides=2)
        self.dw_frn  = nn.FRNorm2D(in_ch=self.in_ch)
        self.dw_tlu  = nn.TLU(in_ch=self.in_ch)

        self.pw_conv = nn.Conv2D(self.in_ch, self.out_ch, kernel_size=1, padding='SAME')
        self.pw_frn  = nn.FRNorm2D(in_ch=self.out_ch)
        self.pw_tlu  = nn.TLU(in_ch=self.out_ch)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_frn(x)
        x = self.dw_tlu(x)
        x = self.pw_conv(x)
        x = self.pw_frn(x)
        x = self.pw_tlu(x)
        return x


nn.GhostModule = GhostModule
nn.DFC = DFC
nn.DSCDownsample = DSCDownsample
