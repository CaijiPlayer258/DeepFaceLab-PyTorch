from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class FAN:
    """
    Face Alignment Network (2D/3D FAN).

    Input:  [1, 3, 256, 256] BGR image
    Output: [1, 68, 64, 64] heatmaps
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo, landmarks_3D: bool = False):
        if device_info not in FAN.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        name = '3DFAN.onnx' if landmarks_3D else '2DFAN.onnx'
        path = weights.path('onnx/FAN/' + name)
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name
        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), name.replace('.onnx', ''))
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

        self.landmarks_3D = landmarks_3D

    def extract(self, img):
        """Extract heatmaps from face image (256x256 BGR)."""
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((256, 256)).ch(3).to_ufloat32()
        inp = ip.get_image('NCHW')
        heatmaps = self._sess.run(None, {self._input_name: inp})[0]
        return heatmaps
