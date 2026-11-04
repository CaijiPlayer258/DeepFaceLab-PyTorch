"""
Convert MTCNN (PNet/RNet/ONet) from PyTorch to ONNX.

MTCNN is a cascaded face detector with 3 stages:
- PNet: Fully convolutional proposal network
- RNet: Refinement network (24x24 input)
- ONet: Output network (48x48 input, produces landmarks)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32)),
        ]))
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),
            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128)),
        ]))
        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),
            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))
        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=1)
        return c, b, a


def load_npy_weights(model, npy_path):
    w = np.load(npy_path, allow_pickle=True)[()]
    torch_w = {k: torch.from_numpy(v) for k, v in w.items()}
    model.load_state_dict(torch_w)


def convert_mtcnn_to_onnx():
    weights_dir = Path(
        r"C:\Users\nobody\.cache\modelscope\iic\cv_manual_face-detection_mtcnn\weights"
    )
    output_dir = Path(__file__).parent

    print("=" * 80)
    print("MTCNN Model Converter: NPY -> ONNX")
    print("=" * 80)

    models_to_convert = [
        ("PNet", PNet(), weights_dir / "pnet.npy", "pnet.onnx", (1, 3, 320, 320)),
        ("RNet", RNet(), weights_dir / "rnet.npy", "rnet.onnx", (1, 3, 24, 24)),
        ("ONet", ONet(), weights_dir / "onet.npy", "onet.onnx", (1, 3, 48, 48)),
    ]

    success = True
    for name, model, w_path, out_name, dummy_shape in models_to_convert:
        print(f"\n--- Converting {name} ---")
        if not w_path.exists():
            print(f"ERROR: Weights not found: {w_path}")
            success = False
            continue

        load_npy_weights(model, w_path)
        model.eval()
        print(f"  Weights loaded from {w_path}")

        out_path = output_dir / out_name
        dummy_input = torch.randn(*dummy_shape)
        print(f"  Dummy input shape: {dummy_input.shape}")

        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["bbox_reg", "face_prob"] if name != "ONet" else ["landmarks", "bbox_reg", "face_prob"],
            dynamic_axes=(
                {"input": {0: "batch_size", 2: "height", 3: "width"}}
                if name == "PNet"
                else {"input": {0: "batch_size"}}
            ),
        )

        # Verify
        try:
            import onnx
            onnx_model = onnx.load(str(out_path))
            onnx.checker.check_model(onnx_model)
            file_size = os.path.getsize(out_path)
            print(f"  Verification passed!")
            print(f"  File size: {file_size / 1024:.1f} KB")
        except ImportError:
            print(f"  Saved to {out_path} (onnx checker not available)")

    print("\n" + "=" * 80)
    print("MTCNN conversion completed!" if success else "MTCNN conversion had errors!")
    return success


if __name__ == "__main__":
    success = convert_mtcnn_to_onnx()
    sys.exit(0 if success else 1)
