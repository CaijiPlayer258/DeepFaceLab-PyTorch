from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class FaceParser:
    """
    Face parsing / segmentation model.
    Parses face into semantic regions (skin, hair, eyes, etc.).

    Input:  [1, 3, 512, 512] BGR image, normalized to [-1, 1]
    Output: [1, 19, 512, 512] class probabilities per pixel (19 classes)
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in FaceParser.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        path = weights.path('onnx/FaceParser/faceparser.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # ── TRT BF16 加速 ──
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'faceparser')
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
        """
        Parse face image into semantic segmentation map (512x512 BGR).
        Returns: [1, 19, 512, 512] float32 class logits
        """
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((512, 512)).ch(3).to_ufloat32()
        inp = ip.get_image('NCHW')
        pred = self._sess.run(None, {self._input_name: inp})
        return pred[0]
