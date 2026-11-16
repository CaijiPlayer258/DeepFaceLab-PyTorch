"""
Convert ULFD (Ultra-Light-Fast-Generic-Face-Detector) from PyTorch to ONNX.

The ULFD model is a lightweight SSD-based face detector with a MobileNet backbone.
Input: 640x480 BGR image, normalized (mean=127, std=128), BGR->RGB
Output: confidences (softmax) + decoded boxes (corner form, normalized 0-1)
"""
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def convert_ulfd_to_onnx():
    # Model paths
    pt_model_path = Path(r"C:\Users\nobody\.cache\modelscope\iic\cv_manual_face-detection_ulfd\pytorch_model.pt")
    onnx_model_path = Path(__file__).parent / "ULFD.onnx"

    print("=" * 80)
    print("ULFD Model Converter: PT -> ONNX")
    print("=" * 80)
    print(f"Input:  {pt_model_path}")
    print(f"Output: {onnx_model_path}")

    if not pt_model_path.exists():
        print(f"ERROR: Input file not found: {pt_model_path}")
        return False

    try:
        from modelscope.models.cv.face_detection.ulfd_slim.vision.ssd.fd_config import (
            define_img_size, priors, image_size, center_variance, size_variance,
        )
        from modelscope.models.cv.face_detection.ulfd_slim.vision.ssd.mb_tiny_fd import create_mb_tiny_fd
        from modelscope.models.cv.face_detection.ulfd_slim.vision import box_utils

        # Set up for 640x480 input
        define_img_size(640)
        print(f"Image size: {image_size}")
        print(f"Number of priors: {len(priors)}")

        # Create model in test mode
        net = create_mb_tiny_fd(2, is_test=True, device="cpu")
        print("Model created successfully")

        # Load weights
        state_dict = torch.load(str(pt_model_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        print("Weights loaded successfully")

        # Set to eval mode
        net.eval()

        # Export to ONNX
        dummy_input = torch.randn(1, 3, image_size[1], image_size[0])
        print(f"Exporting with dummy input shape: {dummy_input.shape}")

        torch.onnx.export(
            net,
            dummy_input,
            str(onnx_model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["confidences", "boxes"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "confidences": {0: "batch_size"},
                "boxes": {0: "batch_size"},
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
    success = convert_ulfd_to_onnx()
    sys.exit(0 if success else 1)
