from pathlib import Path
from modelhub import weights
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class FaceEnhancer:
    """
    Face enhancement / super-resolution network.

    Input:  bgr [1, 3, 256, 256] + param/param1 [1, 1]
    Output: [1, 3, 1024, 1024] enhanced face
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in FaceEnhancer.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        path = weights.path('faceenhancer.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_names = [inp.name for inp in self._sess.get_inputs()]

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'FaceEnhancer')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def enhance(self, img, param=1.0, param1=1.0):
        """
        Enhance face resolution 4x (256 → 1024).
        param/param1 control enhancement strength.
        """
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((256, 256)).ch(3).to_ufloat32()
        bgr = ip.get_image('NCHW')
        p = np.array([[param]], dtype=np.float32)
        feed = {self._input_names[0]: bgr,
                self._input_names[1]: p,
                self._input_names[2]: p}
        return self._sess.run(None, feed)[0]
