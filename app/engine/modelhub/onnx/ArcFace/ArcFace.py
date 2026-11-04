from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class ArcFace:
    """
    ArcFace face recognition model.
    Supports w600k_mbf (MobileFaceNet) and w600k_r50 (ResNet50).

    Input: [N, 3, 112, 112] BGR image, normalized to [-1, 1]
    Output: [N, 512] embedding vector
    """

    MODEL_FILES = {
        'w600k_mbf': 'w600k_mbf.onnx',
        'w600k_r50': 'w600k_r50.onnx',
    }

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo, model_name: str = 'w600k_mbf'):
        if device_info not in ArcFace.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        if model_name not in self.MODEL_FILES:
            raise Exception(f'Unknown model: {model_name}')
        path = weights.path('onnx/ArcFace/' + self.MODEL_FILES[model_name])
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), self.MODEL_FILES[model_name].replace('.onnx', ''))
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def extract(self, img):
        """Extract embedding from aligned face image (112x112 BGR)."""
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((112, 112)).ch(3).to_ufloat32()
        inp = ip.get_image('NCHW')
        pred = self._sess.run(None, {self._input_name: inp})[0]
        return pred
