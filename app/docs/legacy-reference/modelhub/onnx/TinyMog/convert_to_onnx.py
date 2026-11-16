"""
Convert TinyMog (SCRFD + MobileNetV1 backbone) from PyTorch to ONNX.

TinyMog uses the SCRFD architecture with a MobileNetV1 backbone.
Uses mmcv-lite for ConvModule/DepthwiseSeparableConvModule.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TinyMogBackbone(nn.Module):
    """MobileNetV1 backbone matching TinyMog checkpoint structure."""

    def __init__(self):
        super().__init__()
        from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

        def make_dw_blocks(inp, oup, stride, num_blocks):
            """Create depthwise separable conv blocks without Sequential."""
            blocks = nn.ModuleList()
            blocks.append(DepthwiseSeparableConvModule(
                inp, oup, 3, stride=stride, padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True)))
            for _ in range(num_blocks - 1):
                blocks.append(DepthwiseSeparableConvModule(
                    oup, oup, 3, stride=1, padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True)))
            return blocks

        # Stem: follows checkpoint structure with TWO blocks
        # stem.0: regular Conv2d(3, 16, 3, stride 2) + BN + ReLU
        # stem.1: DepthwiseSeparableConvModule(16, 16, 3, stride 1) + BN + ReLU
        self.stem = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            DepthwiseSeparableConvModule(
                16, 16, 3, stride=1, padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True))
        )

        # Each layer is a ModuleList of DepthwiseSeparableConvModule
        # This matches the checkpoint's backbone.layer1.0.depthwise_conv... naming
        self.layer1 = nn.ModuleList(make_dw_blocks(16, 40, 2, 2))
        self.layer2 = nn.ModuleList(make_dw_blocks(40, 72, 2, 3))
        self.layer3 = nn.ModuleList(make_dw_blocks(72, 152, 2, 2))
        self.layer4 = nn.ModuleList(make_dw_blocks(152, 288, 2, 6))

    def forward(self, x):
        x = self.stem(x)           # 640->320, C=16
        for block in self.layer1:
            x = block(x)           # 320->160, C=40
        f0 = x
        for block in self.layer2:
            x = block(x)           # 160->80, C=72
        f1 = x
        for block in self.layer3:
            x = block(x)           # 80->40, C=152
        f2 = x
        for block in self.layer4:
            x = block(x)           # 40->20, C=288
        f3 = x
        return [f0, f1, f2, f3]


class PAFPN_TinyMog(nn.Module):
    """PAFPN neck matching TinyMog checkpoint structure.

    CRITICAL FIX: Added missing pafpn_convs (extra convs after bottom-up merge).
    The checkpoint has pafpn_convs keys that were silently dropped by the old converter.
    """

    def __init__(self):
        super().__init__()
        from mmcv.cnn import ConvModule

        # Lateral convs - no norm, no activation
        self.lateral_convs = nn.ModuleList([
            ConvModule(72, 16, 1, norm_cfg=None, act_cfg=None),
            ConvModule(152, 16, 1, norm_cfg=None, act_cfg=None),
            ConvModule(288, 16, 1, norm_cfg=None, act_cfg=None),
        ])
        # FPN convs - no norm, no activation
        self.fpn_convs = nn.ModuleList([
            ConvModule(16, 16, 3, padding=1, norm_cfg=None, act_cfg=None),
            ConvModule(16, 16, 3, padding=1, norm_cfg=None, act_cfg=None),
            ConvModule(16, 16, 3, padding=1, norm_cfg=None, act_cfg=None),
        ])
        # Downsample convs (bottom-up) - with ReLU
        self.downsample_convs = nn.ModuleList([
            ConvModule(16, 16, 3, stride=2, padding=1, norm_cfg=None,
                       act_cfg=dict(type='ReLU', inplace=True)),
            ConvModule(16, 16, 3, stride=2, padding=1, norm_cfg=None,
                       act_cfg=dict(type='ReLU', inplace=True)),
        ])
        # PAFPN convs after bottom-up merge - no norm, no activation
        # *** THIS WAS MISSING in the old converter ***
        self.pafpn_convs = nn.ModuleList([
            ConvModule(16, 16, 3, padding=1, norm_cfg=None, act_cfg=None),
            ConvModule(16, 16, 3, padding=1, norm_cfg=None, act_cfg=None),
        ])

    def forward(self, inputs):
        # inputs: [f0(C=40), f1(C=72), f2(C=152), f3(C=288)]
        # start_level=1: skip f0, use f1, f2, f3
        laterals = []
        for i in range(3):
            laterals.append(self.lateral_convs[i](inputs[i + 1]))

        # Top-down
        fpn_outs = [laterals[2]]
        for i in range(1, -1, -1):
            prev = F.interpolate(fpn_outs[0], size=laterals[i].shape[2:], mode='nearest')
            fpn_outs.insert(0, laterals[i] + prev)

        # FPN convs
        for i in range(3):
            fpn_outs[i] = self.fpn_convs[i](fpn_outs[i])

        # Bottom-up pathway (downsample)
        outs = [fpn_outs[0]]
        for i in range(1, 3):
            outs.append(self.downsample_convs[i - 1](outs[-1]) + fpn_outs[i])

        # Apply PAFPN convs after bottom-up merge (was MISSING in old converter)
        for i in range(len(outs) - 1):
            outs[i + 1] = self.pafpn_convs[i](outs[i + 1])

        return tuple(outs)


class Integral(nn.Module):
    """Integral layer from SCRFDHead for distribution-based regression."""

    def __init__(self, reg_max=8):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            'project', torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class TinyMogHead(nn.Module):
    """SCRFDHead matching TinyMog checkpoint (strides_share=False, BN, no scale, DW conv)."""

    def __init__(self):
        super().__init__()
        from mmcv.cnn import DepthwiseSeparableConvModule

        strides = [8, 16, 32]
        self.strides = strides
        self.NK = 5
        self.use_kps = True

        # Integral for distribution-based regression (present in checkpoint)
        self.integral = Integral(reg_max=8)

        # strides_share=False, so separate convs for each stride
        self.cls_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        self.stride_kps = nn.ModuleDict()

        for stride in strides:
            key = f'({stride}, {stride})'
            convs = nn.ModuleList()
            for i in range(2):  # stacked_convs=2
                in_ch = 16 if i == 0 else 64
                conv = DepthwiseSeparableConvModule(
                    in_ch, 64, 3, padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU', inplace=True))
                convs.append(conv)
            self.cls_stride_convs[key] = convs
            self.stride_cls[key] = nn.Conv2d(64, 2, 3, padding=1)   # num_anchors=2, num_classes=1
            self.stride_reg[key] = nn.Conv2d(64, 8, 3, padding=1)   # 2 anchors * 4
            self.stride_kps[key] = nn.Conv2d(64, 20, 3, padding=1)  # 2 anchors * 10

    def forward_single(self, x, stride):
        key = f'({stride}, {stride})'
        convs = self.cls_stride_convs[key]
        for conv in convs:
            x = conv(x)

        cls_score = self.stride_cls[key](x)
        bbox_pred = self.stride_reg[key](x)
        kps_pred = self.stride_kps[key](x)

        B = cls_score.shape[0]
        # num_classes=1, cls_out_channels=num_anchors * num_classes = 2*1=2
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, -1, 1).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
        kps_pred = kps_pred.permute(0, 2, 3, 1).reshape(B, -1, self.NK * 2)

        return cls_score, bbox_pred, kps_pred

    def forward(self, feats):
        results = []
        for x, stride in zip(feats, self.strides):
            results.append(self.forward_single(x, stride))

        flat = []
        for cls_s, bbox_s, kps_s in results:
            flat.append(cls_s)
            flat.append(bbox_s)
            flat.append(kps_s)
        return tuple(flat)


class TinyMogModel(nn.Module):
    """Full TinyMog model: MobileNetV1 backbone + PAFPN + SCRFDHead."""

    def __init__(self):
        super().__init__()
        self.backbone = TinyMogBackbone()
        self.neck = PAFPN_TinyMog()
        self.head = TinyMogHead()

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        return self.head(neck_feats)


def _remap_key(ckpt_key: str) -> str:
    """
    Remap checkpoint keys to match the model's state dict.

    The checkpoint was created from mmdet/MMOCR with inline Sequential ordering,
    while the model uses mmcv's DepthwiseSeparableConvModule.

    Key transformations:
    - backbone.stem.0.X (standard Conv+BN): no change needed
    - backbone.stem.1.0 -> backbone.stem.1.depthwise_conv.conv
    - backbone.stem.1.1 -> backbone.stem.1.depthwise_conv.bn
    - backbone.stem.1.3 -> backbone.stem.1.pointwise_conv.conv
    - backbone.stem.1.4 -> backbone.stem.1.pointwise_conv.bn
    - backbone.layerN.M.0 -> backbone.layerN.M.depthwise_conv.conv
    - backbone.layerN.M.1 -> backbone.layerN.M.depthwise_conv.bn
    - backbone.layerN.M.3 -> backbone.layerN.M.pointwise_conv.conv
    - backbone.layerN.M.4 -> backbone.layerN.M.pointwise_conv.bn
    - bbox_head.* -> head.*

    Returns None if the key should be skipped (e.g. integral, pafpn_convs).
    """
    # Rename bbox_head -> head
    key = ckpt_key.replace('bbox_head.', 'head.')

    # Handle backbone depthwise separable blocks
    # Pattern: backbone.(stem.1|layer\d+.\d+).(0|1|3|4).*
    m = re.match(r'(backbone\.(?:stem\.1|layer\d+\.\d+))\.(\d+)\.(.+)', key)
    if m:
        prefix = m.group(1)   # e.g. backbone.layer1.0
        idx = m.group(2)      # e.g. 0, 1, 3, 4
        suffix = m.group(3)   # e.g. weight, bias, running_mean, etc.

        remap = {
            '0': 'depthwise_conv.conv',
            '1': 'depthwise_conv.bn',
            '3': 'pointwise_conv.conv',
            '4': 'pointwise_conv.bn',
        }
        if idx in remap:
            return f'{prefix}.{remap[idx]}.{suffix}'

    return key


def convert_tinymog_to_onnx():
    pt_model_path = Path(
        r"C:\Users\nobody\.cache\modelscope\iic\cv_manual_face-detection_tinymog\pytorch_model.pt"
    )
    onnx_model_path = Path(__file__).parent / "TinyMog.onnx"

    print("=" * 80)
    print("TinyMog Model Converter: PT -> ONNX")
    print("=" * 80)
    print(f"Input:  {pt_model_path}")
    print(f"Output: {onnx_model_path}")

    if not pt_model_path.exists():
        print(f"ERROR: Input file not found: {pt_model_path}")
        return False

    try:
        model = TinyMogModel()
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load checkpoint
        checkpoint = torch.load(str(pt_model_path), map_location="cpu", weights_only=True)
        raw_sd = checkpoint['state_dict']

        # Use proper key remapping instead of fragile shape-based matching
        new_sd = {}
        mapped = 0
        skipped = 0
        for ck in raw_sd:
            mk = _remap_key(ck)
            if mk is None:
                skipped += 1
                continue
            if mk in model.state_dict():
                new_sd[mk] = raw_sd[ck]
                mapped += 1
            else:
                print(f"  WARNING: Remapped key '{mk}' not found in model "
                      f"(from checkpoint key '{ck}')")
                skipped += 1

        print(f"  Mapped {mapped} keys, skipped {skipped} keys")

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"Missing keys: {len([k for k in missing if not k.endswith('num_batches_tracked')])}")
        print(f"Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"  Sample unexpected: {list(unexpected)[:5]}")
        if missing:
            print(f"  Sample missing: {[k for k in missing if not k.endswith('num_batches_tracked')][:5]}")

        model.eval()

        # Export
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"Exporting with dummy input shape: {dummy_input.shape}")

        torch.onnx.export(
            model, dummy_input, str(onnx_model_path),
            export_params=True, opset_version=11, do_constant_folding=True,
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

        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_model_path)))

        file_size = os.path.getsize(onnx_model_path)
        print(f"\nVerification passed! File size: {file_size / (1024 * 1024):.2f} MB")
        print("Conversion completed successfully!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = convert_tinymog_to_onnx()
    sys.exit(0 if success else 1)
