"""
ArcFace embedding extractor using w600k_mbf ONNX model (+TRT BF16).
Reference: FacesetProcessor/Filter.py ArcFaceONNXExtractor
"""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime


class ArcFaceExtractor:
    """Face embedding extractor using small ArcFace model (+TRT BF16)."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(
                Path(__file__).parent.parent.parent / "modelhub" / "onnx" / "ArcFace" / "w600k_mbf.onnx"
            )

        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # ── TRT BF16 加速 ──
        try:
            from xlib.trt import find_trt_engine, TRTInferenceSession
            _trt_path = find_trt_engine(model_path, 'w600k_mbf')
            if _trt_path:
                self.session = TRTInferenceSession(_trt_path)
        except Exception:
            pass

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """Extract 512-dim embedding from aligned face image."""
        resized = cv2.resize(face_img, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, :, :, :]
        embedding = self.session.run(None, {self.input_name: input_tensor})[0]
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm
