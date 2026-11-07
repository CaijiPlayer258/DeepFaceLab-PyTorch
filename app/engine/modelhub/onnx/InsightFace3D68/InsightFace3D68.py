from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class InsightFace3D68:
    """
    InsightFace 3D landmark detector (68 pts, 1k3d68 model).

    Input:  [N, 3, 192, 192] BGR image
    Output: [1, 3309] (68×3 3D landmarks + scores)
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in InsightFace3D68.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        path = weights.path('insightface_3d68.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), '1k3d68')
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
        """Detect 3D landmarks (192x192 BGR)."""
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((192, 192)).ch(3).to_ufloat32()
        inp = ip.get_image('NCHW')
        pred = self._sess.run(None, {self._input_name: inp})[0][0]
        return pred
