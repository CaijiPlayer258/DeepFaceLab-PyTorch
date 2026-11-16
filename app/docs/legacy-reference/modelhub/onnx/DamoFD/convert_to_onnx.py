"""
Convert DamoFD (SCRFD + MasterNet backbone) from PyTorch to ONNX.

Uses ONLY plain PyTorch modules (no mmcv dependency) to match the exact
checkpoint structure. This avoids mmcv version compatibility issues.

Key: All weights use mmcv-compatible naming (.conv submodule, etc.)
so checkpoint keys match exactly.

Architecture (from DamoFD_lms.py config):
  backbone: MasterNet with SuperNet struct
  neck: PAFPN(in_channels=[32,64,120,160], out_channels=16, start_level=1, num_outs=3)
  bbox_head: SCRFDHead(in_channels=16, feat_channels=64, stacked_convs=2,
                       num_anchors=2, strides=[8,16,32], use_kps=True, dw_conv=True)
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# mmcv-compatible wrappers
# ---------------------------------------------------------------------------

class Conv2dOnly(nn.Module):
    """Plain Conv2d with NO norm, NO activation.
    Matches mmcv ConvModule(norm_cfg=None, act_cfg=None).
    State dict: .conv.weight, .conv.bias
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        return self.conv(x)


class Conv2dReLU(nn.Module):
    """Conv2d + ReLU (no norm).
    Matches mmcv ConvModule(norm_cfg=None, act_cfg=dict(type='ReLU')).
    State dict: .conv.weight, .conv.bias
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.activate = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.activate(self.conv(x))


class Conv2dGNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU.
    Matches mmcv ConvModule(norm_cfg=dict(type='GN', num_groups=N)).
    State dict: .conv.weight, .gn.weight, .gn.bias
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 groups=1, num_groups=16, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.gn = nn.GroupNorm(num_groups, out_ch)
        self.activate = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.activate(self.gn(self.conv(x)))


class DepthwiseSeparableConvGN(nn.Module):
    """DepthwiseSeparableConvModule with GroupNorm.
    Matches mmcv DepthwiseSeparableConvModule with norm_cfg=dict(type='GN', N=16).
    State dict:
      .depthwise_conv.conv.weight, .depthwise_conv.gn.weight, .depthwise_conv.gn.bias
      .pointwise_conv.conv.weight, .pointwise_conv.gn.weight, .pointwise_conv.gn.bias
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, num_groups=16):
        super().__init__()
        self.depthwise_conv = Conv2dGNReLU(
            in_ch, in_ch, kernel_size, padding=padding,
            groups=in_ch, num_groups=num_groups, bias=False)
        self.pointwise_conv = Conv2dGNReLU(
            in_ch, out_ch, 1, num_groups=num_groups, bias=False)
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# ---------------------------------------------------------------------------
# PAFPN
# ---------------------------------------------------------------------------

class PAFPN(nn.Module):
    """PAFPN neck matching mmdet structure EXACTLY with plain PyTorch.

    Config:
      in_channels=[32, 64, 120, 160]
      out_channels=16
      start_level=1
      num_outs=3
    """
    def __init__(self, in_channels, out_channels=16, start_level=1, num_outs=3):
        super().__init__()
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level

        # Lateral convs: 1x1, no norm, no act → Conv2dOnly
        self.lateral_convs = nn.ModuleList()
        for i in range(start_level, self.num_ins):
            self.lateral_convs.append(
                Conv2dOnly(in_channels[i], out_channels, 1, bias=True))

        # FPN convs: 3x3, no norm, no act → Conv2dOnly
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_outs):
            self.fpn_convs.append(
                Conv2dOnly(out_channels, out_channels, 3, padding=1, bias=True))

        # Downsample convs: 3x3 stride=2 + ReLU → Conv2dReLU
        self.downsample_convs = nn.ModuleList()
        for _ in range(num_outs - 1):
            self.downsample_convs.append(
                Conv2dReLU(out_channels, out_channels, 3, stride=2, padding=1, bias=True))

        # PAFPN convs: 3x3, no norm, no act → Conv2dOnly
        self.pafpn_convs = nn.ModuleList()
        for _ in range(num_outs - 1):
            self.pafpn_convs.append(
                Conv2dOnly(out_channels, out_channels, 3, padding=1, bias=True))

    def forward(self, inputs):
        # Lateral convs
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i + self.start_level]))

        # Top-down pathway (coarsest to finest)
        fpn_outs = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            prev = F.interpolate(fpn_outs[0], size=laterals[i].shape[2:],
                                 mode='nearest')
            fpn_outs.insert(0, laterals[i] + prev)

        # Apply FPN convs
        for i in range(len(fpn_outs)):
            fpn_outs[i] = self.fpn_convs[i](fpn_outs[i])

        # Bottom-up pathway (PAFPN)
        outs = [fpn_outs[0]]
        for i in range(1, len(fpn_outs)):
            outs.append(self.downsample_convs[i - 1](outs[-1]) + fpn_outs[i])

        # Apply PAFPN convs after bottom-up merge
        for i in range(len(outs) - 1):
            outs[i + 1] = self.pafpn_convs[i](outs[i + 1])

        return tuple(outs)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class Scale(nn.Module):
    """Learnable scale parameter (matching mmcv Scale)."""
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
    def forward(self, x):
        return x * self.scale


class Integral(nn.Module):
    """Integral layer for distribution-based regression."""
    def __init__(self, reg_max=8):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            'project', torch.linspace(0, self.reg_max, self.reg_max + 1))
    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class SCRFDHead(nn.Module):
    """SCRFD head matching mmdet structure with plain PyTorch.

    Config:
      in_channels=16, num_classes=1, feat_channels=64, stacked_convs=2
      num_anchors=2, strides=[8,16,32], use_kps=True, dw_conv=True
      norm_cfg=dict(type='GN', num_groups=16), cls_reg_share=True, strides_share=True
    """
    def __init__(self, in_channels=16, num_classes=1, feat_channels=64,
                 stacked_convs=2, num_anchors=2, strides=[8, 16, 32],
                 use_kps=True, num_kps=5, dw_conv=True, reg_max=8):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_anchors = num_anchors
        self.strides = strides
        self.use_kps = use_kps
        self.NK = num_kps
        self.dw_conv = dw_conv

        # Integral (present in checkpoint for weight loading)
        self.integral = Integral(reg_max)

        # Cls convs - strides_share=True, key='0'
        # Using DepthwiseSeparableConvGN matching mmcv DepthwiseSeparableConvModule
        cls_convs = nn.ModuleList()
        for i in range(stacked_convs):
            in_ch = in_channels if i == 0 else feat_channels
            if dw_conv:
                conv = DepthwiseSeparableConvGN(in_ch, feat_channels, 3, 1)
            else:
                conv = Conv2dGNReLU(in_ch, feat_channels, 3, padding=1)
            cls_convs.append(conv)

        self.cls_stride_convs = nn.ModuleDict({'0': cls_convs})

        # Classification head (plain nn.Conv2d, no .conv wrapper)
        self.stride_cls = nn.ModuleDict({
            '0': nn.Conv2d(feat_channels, num_anchors * num_classes, 3,
                           padding=1, bias=True)
        })

        # Regression head (plain nn.Conv2d)
        self.stride_reg = nn.ModuleDict({
            '0': nn.Conv2d(feat_channels, num_anchors * 4, 3,
                           padding=1, bias=True)
        })

        # Keypoint head (plain nn.Conv2d)
        if use_kps:
            self.stride_kps = nn.ModuleDict({
                '0': nn.Conv2d(feat_channels, num_anchors * num_kps * 2, 3,
                               padding=1, bias=True)
            })
        else:
            self.stride_kps = nn.ModuleDict({})

        # Scales
        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

    def forward_single(self, x, scale, stride):
        cls_feat = x
        cls_convs = self.cls_stride_convs['0']
        for conv in cls_convs:
            cls_feat = conv(cls_feat)

        reg_feat = cls_feat  # cls_reg_share=True

        cls_score = self.stride_cls['0'](cls_feat)
        bbox_pred = scale(self.stride_reg['0'](reg_feat))

        if self.use_kps and '0' in self.stride_kps:
            kps_pred = self.stride_kps['0'](reg_feat)
        else:
            kps_pred = bbox_pred.new_zeros(
                (bbox_pred.shape[0], self.NK * 2, bbox_pred.shape[2],
                 bbox_pred.shape[3]))

        # ONNX format: permute + reshape + sigmoid
        B = cls_score.shape[0]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            B, -1, self.num_classes).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
        kps_pred = kps_pred.permute(0, 2, 3, 1).reshape(B, -1, self.NK * 2)

        return cls_score, bbox_pred, kps_pred

    def forward(self, feats):
        results = []
        for x, scale, stride in zip(feats, self.scales, self.strides):
            results.append(self.forward_single(x, scale, stride))

        # Flatten: [cls_0, bbox_0, kps_0, cls_1, bbox_1, kps_1, ...]
        flat = []
        for cls_s, bbox_s, kps_s in results:
            flat.append(cls_s)
            flat.append(bbox_s)
            flat.append(kps_s)
        return tuple(flat)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DamoFDModel(nn.Module):
    """Full DamoFD model: MasterNet backbone + PAFPN + SCRFDHead."""

    def __init__(self):
        super().__init__()
        from modelscope.models.cv.tinynas_classfication import plain_net_utils

        # Backbone (MasterNet structure)
        struct = ('SuperConvK3BNRELU(3,32,2,1)'
                  'SuperResIDWE1K3(32,32,2,8,1)'
                  'SuperResIDWE1K7(32,64,2,40,1)'
                  'SuperResIDWE1K7(64,120,2,40,2)'
                  'SuperResIDWE1K5(120,160,2,120,1)')
        self.backbone = plain_net_utils.PlainNet(
            num_classes=2048, plainnet_struct=struct,
            no_create=False, no_reslink=False, no_BN=False, use_se=False,
        )

        # Neck
        self.neck = PAFPN(
            in_channels=[32, 64, 120, 160],
            out_channels=16, start_level=1, num_outs=3,
        )

        # Head
        self.head = SCRFDHead(
            in_channels=16, num_classes=1, feat_channels=64,
            stacked_convs=2, num_anchors=2, strides=[8, 16, 32],
            use_kps=True, num_kps=5, dw_conv=True, reg_max=8,
        )

    def forward(self, x):
        # Backbone: sequential through blocks, collecting at each downsampling level
        f0 = self.backbone.module_list[0](x)   # stride 2
        f1 = self.backbone.module_list[1](f0)  # stride 4
        f2 = self.backbone.module_list[2](f1)  # stride 8
        f3 = self.backbone.module_list[3](f2)  # stride 16
        f4 = self.backbone.module_list[4](f3)  # stride 32

        neck_feats = self.neck([f1, f2, f3, f4])
        return self.head(neck_feats)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def load_pt_model():
    """Build model and load checkpoint weights."""
    pt_model_path = Path(
        r"C:\Users\nobody\.cache\modelscope\iic\cv_ddsar_face-detection_iclr23-damofd\pytorch_model.pt"
    )

    if not pt_model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_model_path}")

    print(f"Loading checkpoint: {pt_model_path}")
    checkpoint = torch.load(str(pt_model_path), map_location="cpu",
                            weights_only=True)
    raw_sd = checkpoint['state_dict']

    # Build model
    model = DamoFDModel()
    print(f"Model built with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Remap keys: bbox_head. → head.
    new_sd = {}
    for k, v in raw_sd.items():
        k = k.replace('detector.module.', '')
        k = k.replace('bbox_head.', 'head.')
        new_sd[k] = v

    # Load state dict
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    # Filter out num_batches_tracked from missing (non-parameter buffers)
    real_missing = [k for k in missing if not k.endswith('num_batches_tracked')]

    print(f"Missing keys: {len(real_missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if real_missing:
        print(f"  Missing: {real_missing[:10]}")
    if unexpected:
        print(f"  Unexpected: {list(unexpected)[:10]}")

    if real_missing or unexpected:
        raise RuntimeError("Keys mismatch - check architecture")

    print("  All keys matched successfully!")
    model.eval()
    return model


def convert_damofd_to_onnx():
    """Convert PyTorch model to ONNX."""
    onnx_model_path = Path(__file__).parent / "DamoFD.onnx"

    print("=" * 80)
    print("DamoFD Model Converter: PT -> ONNX (plain PyTorch, mmcv-compat naming)")
    print("=" * 80)
    print(f"Output: {onnx_model_path}")

    try:
        model = load_pt_model()

        # Test inference
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            test_out = model(dummy)

        print(f"\nTest inference outputs:")
        for i, t in enumerate(test_out):
            print(f"  [{i}] shape={tuple(t.shape)}, "
                  f"min={t.min().item():.4f}, max={t.max().item():.4f}")

        max_cls = max(test_out[0].max().item(), test_out[3].max().item(),
                      test_out[6].max().item())
        print(f"Max classification score: {max_cls:.4f}")
        if max_cls < 0.5:
            print("WARNING: Max scores low - investigate!")
        else:
            print("OK: Scores look reasonable (>0.5)")

        # Export ONNX
        print(f"\nExporting...")
        torch.onnx.export(
            model, dummy, str(onnx_model_path),
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=["input"],
            output_names=["cls_0", "bbox_0", "kps_0",
                         "cls_1", "bbox_1", "kps_1",
                         "cls_2", "bbox_2", "kps_2"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "cls_0": {0: "batch_size"}, "bbox_0": {0: "batch_size"}, "kps_0": {0: "batch_size"},
                "cls_1": {0: "batch_size"}, "bbox_1": {0: "batch_size"}, "kps_1": {0: "batch_size"},
                "cls_2": {0: "batch_size"}, "bbox_2": {0: "batch_size"}, "kps_2": {0: "batch_size"},
            },
        )

        # Verify
        import onnx
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)

        file_size = os.path.getsize(onnx_model_path)
        print(f"\nVerification passed! File size: {file_size / (1024 * 1024):.2f} MB")
        print("Conversion completed successfully!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx(onnx_path=None):
    """Test the exported ONNX model on a synthetic face image."""
    import cv2
    import onnxruntime as ort

    if onnx_path is None:
        onnx_path = Path(__file__).parent / "DamoFD.onnx"
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        print(f"ONNX model not found: {onnx_path}")
        return

    session = ort.InferenceSession(str(onnx_path))

    # Synthetic face
    img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    cv2.circle(img, (320, 320), 100, (180, 120, 100), -1)
    cv2.circle(img, (300, 300), 15, (80, 80, 80), -1)
    cv2.circle(img, (340, 300), 15, (80, 80, 80), -1)
    cv2.circle(img, (320, 340), 10, (60, 60, 60), -1)

    blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, (640, 640),
                                  (127.5,) * 3, swapRB=True)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})

    max_scores = []
    for idx, stride in enumerate([8, 16, 32]):
        scores = outputs[idx * 3][0]
        mx = scores.max()
        max_scores.append(mx)
        scores_flat = scores.ravel()
        n01 = (scores_flat > 0.01).sum()
        n05 = (scores_flat > 0.5).sum()
        print(f"  stride {stride:2d}: max={mx:.4f}, >0.01={n01:5d}, >0.5={n05:4d}")

    overall = max(max_scores)
    print(f"\nOverall max score: {overall:.4f}  {'PASS' if overall > 0.5 else 'FAIL'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--onnx-path', type=str, default=None)
    args = parser.parse_args()

    if args.test_only:
        test_onnx(args.onnx_path)
    else:
        success = convert_damofd_to_onnx()
        if success:
            print("\n" + "=" * 80)
            print("Testing exported ONNX:")
            print("=" * 80)
            test_onnx()
        sys.exit(0 if success else 1)
