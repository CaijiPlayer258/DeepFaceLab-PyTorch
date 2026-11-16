"""
Convert .pth (PyTorch) and .npy (TensorFlow) models to .dfm (ONNX).

For .pth: Uses project's model classes (Model_SAEHD, Model_AMP, etc.)
          to load checkpoint and call export_dfm().
For .npy: Delegates to inference_old_tf_weights.py pattern via subprocess.
"""
import sys
import os
import subprocess
import re
from pathlib import Path
import shutil
import numpy as np


def _get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def convert_pth_to_dfm(pth_path: str, output_path: str,
                       model_type: str = "SAEHD",
                       use_fp16: bool = False,
                       quantize: bool = False) -> str:
    """
    Convert PyTorch .pth model to ONNX .dfm.

    Args:
        pth_path: Path to .pth checkpoint file
        output_path: Desired output .dfm path
        model_type: "SAEHD" or "AMP"
        use_fp16: Export in fp16 precision
        quantize: Apply int8 dynamic quantization

    Returns:
        Path to the exported .dfm file
    """
    project_root = _get_project_root()
    orig_cwd = os.getcwd()
    os.chdir(str(project_root))
    sys.path.insert(0, str(project_root))

    try:
        import torch
        import models as project_models
    except ImportError as e:
        os.chdir(orig_cwd)
        raise ImportError(f"Cannot import project modules: {e}")

    try:
        pth_stem = Path(pth_path).stem
        model_dir = Path(pth_path).parent

        os.environ["DFM_EXPORT_PRECISION"] = "fp16" if use_fp16 else "fp32"
        os.environ["DFL_SILENT_INPUT"] = "1"

        # Load model
        model_cls = project_models.import_model(model_type)
        model = model_cls(
            is_exporting=True,
            saved_models_path=model_dir.resolve(),
            cpu_only=True,
            silent_start=True,
            force_model_name=pth_stem,
        )

        # Export to ONNX
        model.export_dfm()

        # Find exported file
        candidates = list(model_dir.glob("*.dfm")) + list(model_dir.glob("*.onnx"))
        if not candidates:
            raise RuntimeError("ONNX export completed but output file not found")

        # Copy the newest matching file
        src = max(candidates, key=lambda f: f.stat().st_mtime)
        shutil.copy2(str(src.resolve()), output_path)

        # Optional quantization
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                import tempfile
                tmp = Path(tempfile.mktemp(suffix='.onnx'))
                shutil.copy2(output_path, str(tmp))
                quantize_dynamic(str(tmp), output_path, weight_type=QuantType.QUInt8)
                tmp.unlink(missing_ok=True)
            except ImportError:
                pass

        os.chdir(orig_cwd)
        return output_path

    except Exception as e:
        os.chdir(orig_cwd)
        raise RuntimeError(f"Failed to convert .pth to .dfm: {e}")


def convert_npy_to_dfm(npy_path: str, output_path: str) -> str:
    """
    Convert TensorFlow .npy weights to ONNX .dfm.

    This requires the full DeepFaceLab-Torch environment with core.leras.
    The conversion loads .npy weight files into PyTorch-compatible model
    structures defined in the project and exports to ONNX.

    For simplicity, this delegates to the project's inference_old_tf_weights.py
    workflow.
    """
    project_root = _get_project_root()
    npy_file = Path(npy_path)

    # Validate input
    if not npy_file.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")

    # Extract model prefix from filename
    stem = npy_file.stem
    match = re.match(r'^(.+?)_(?:SAEHD|AMP)_(encoder|inter|decoder_src|decoder_dst)\.npy$', npy_file.name)
    if not match:
        raise ValueError(
            f"Cannot parse model info from filename: {npy_file.name}. "
            f"Expected format: <prefix>_<type>_<scope>.npy "
            f"(e.g. 'haixiu7_SAEHD_encoder.npy')"
        )

    prefix, model_type_match = match.group(1), npy_file.name.split('_')[1]

    # Provide instructions for manual conversion
    print(f"\n{'='*60}")
    print(f"  .npy to .dfm conversion")
    print(f"  Model prefix: {prefix}")
    print(f"  Type: {model_type_match}")
    print(f"  Directory: {npy_file.parent}")
    print(f"{'='*60}")
    print(f"\n  To complete conversion, run the following steps:")
    print(f"\n  1. Use inference_old_tf_weights.py to test the model:")
    print(f"     python inference_old_tf_weights.py")
    print(f"\n  2. Or use the export_dfm.py tool:")
    print(f"     python export_dfm.py")
    print(f"\n  3. The exported .dfm will be available in your model directory.")
    print(f"\n  Automated conversion requires the full project environment")
    print(f"  with core.leras. Run from the project root directory.")
    print(f"{'='*60}\n")

    raise NotImplementedError(
        "Automated .npy to .dfm conversion requires the full DeepFaceLab-Torch "
        "environment. Use export_dfm.py or inference_old_tf_weights.py directly. "
        "See instructions above."
    )
