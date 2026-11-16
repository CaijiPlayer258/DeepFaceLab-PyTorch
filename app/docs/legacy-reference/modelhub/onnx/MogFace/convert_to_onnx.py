"""
Convert MogFace from PyTorch to ONNX.

MogFace: ResNet101 backbone + LFPN + MogPredNet face detector.
Input: BGR image, normalized with specific mean/std
Output: confidence scores and bounding box locations
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def convert_mogface_to_onnx():
    pt_model_path = Path(
        r"C:\Users\nobody\.cache\modelscope\iic\cv_resnet101_face-detection_cvpr22papermogface\pytorch_model.pt"
    )
    onnx_model_path = Path(__file__).parent / "MogFace.onnx"

    print("=" * 80)
    print("MogFace Model Converter: PT -> ONNX")
    print("=" * 80)
    print(f"Input:  {pt_model_path}")
    print(f"Output: {onnx_model_path}")

    if not pt_model_path.exists():
        print(f"ERROR: Input file not found: {pt_model_path}")
        return False

    try:
        from modelscope.models.cv.face_detection.mogface.models.mogface import MogFace

        # Load model
        net = MogFace()
        pretrained_dict = torch.load(str(pt_model_path), map_location="cpu", weights_only=True)
        net.load_state_dict(pretrained_dict, strict=False)
        net.eval()
        print("Model loaded successfully")

        total_params = sum(p.numel() for p in net.parameters())
        print(f"Total parameters: {total_params:,}")

        # Export to ONNX
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"Exporting with dummy input shape: {dummy_input.shape}")

        torch.onnx.export(
            net,
            dummy_input,
            str(onnx_model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["confidences", "locations"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "confidences": {0: "batch_size", 1: "num_anchors"},
                "locations": {0: "batch_size", 1: "num_anchors"},
            },
        )

        # Verify
        import onnx

        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)

        file_size = os.path.getsize(onnx_model_path)
        print(f"\nVerification passed!")
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")
        print("Conversion completed successfully!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = convert_mogface_to_onnx()
    sys.exit(0 if success else 1)
