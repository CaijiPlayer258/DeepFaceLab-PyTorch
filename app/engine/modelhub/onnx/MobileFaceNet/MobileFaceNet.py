"""MobileFaceNet+SE face landmark detector (68 points)."""
from pathlib import Path
from modelhub import weights
import cv2
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class MobileFaceNet:
    @staticmethod
    def get_available_devices():
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        path = weights.path('onnx/MobileFaceNet/landmark_detection_56_se_external.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name
        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'landmark_detection_56_se_external')
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
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (56, 56))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :]

        out = self._sess.run(None, {self._input_name: tensor})[0][0]  # (136,)
        landmarks = out.reshape(-1, 2)
        landmarks[:, 0] = landmarks[:, 0] * W
        landmarks[:, 1] = landmarks[:, 1] * H
        return [landmarks]
