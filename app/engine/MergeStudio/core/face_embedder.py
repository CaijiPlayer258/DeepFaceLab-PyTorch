"""
ArcFace face embedding. Singleton ONNX session.
Supports multiple model locations for compatibility.
"""
import cv2
import numpy as np
import onnxruntime
from pathlib import Path

_arcface_session = None
_arcface_input_name = None


def _find_model():
    """Find ArcFace ONNX model, checking multiple possible locations."""
    # Locations to check in priority order
    candidates = [
        Path(__file__).parent.parent.parent / "modelhub" / "onnx" / "ArcFace" / "w600k_mbf.onnx",
        Path(__file__).parent.parent.parent / "modelhub" / "onnx" / "ArcFace" / "w600k_r50.onnx",
        Path(__file__).parent.parent.parent / "workspace" / "model" / "ArcFace" / "arcface.onnx",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"ArcFace model not found. Checked: {[str(c) for c in candidates]}"
    )


def _get_arcface_session():
    global _arcface_session, _arcface_input_name
    if _arcface_session is not None:
        return _arcface_session, _arcface_input_name
    model_path = _find_model()
    _arcface_session = onnxruntime.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    _arcface_input_name = _arcface_session.get_inputs()[0].name
    return _arcface_session, _arcface_input_name


def compute_embedding(face_img: np.ndarray) -> np.ndarray:
    """
    Compute 512-dim embedding from a face image.
    face_img: uint8 BGR, any size (will be resized to 112x112 internally).
    Returns: float32 array of shape (512,), L2-normalized.
    """
    sess, inp_name = _get_arcface_session()
    h, w = face_img.shape[:2]
    if (h, w) != (112, 112):
        face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_CUBIC)
    # BGR to RGB, normalize to [-1, 1]
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    # NHWC → NCHW
    blob = np.transpose(rgb, (2, 0, 1))[None, :, :, :]
    output = sess.run(None, {inp_name: blob})[0]
    embedding = output.squeeze()  # (512,)
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def compute_embedding_rgb(face_rgb: np.ndarray) -> np.ndarray:
    """
    Compute 512-dim embedding from an RGB face image.
    face_rgb: uint8 RGB, any size.
    """
    img = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    return compute_embedding(img)
