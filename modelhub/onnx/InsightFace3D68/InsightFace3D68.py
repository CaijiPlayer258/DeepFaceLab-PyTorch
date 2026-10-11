from pathlib import Path
from typing import List
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class InsightFace3D68:
    """
    InsightFace 3D landmark detector (68 pts, 1k3d68 model).

    Input:  [N, 3, 192, 192] BGR image
    Output: [1, 3309] = 1103x3 展平；标记点在**最后 68 行**（前 68 行为 0）
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in InsightFace3D68.get_available_devices():
            raise Exception(f'device_info {device_info} not available')
        path = Path(__file__).parent / '1k3d68.onnx'
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
        """Detect 3D landmarks (192x192 BGR)。

        ⚠️ 输入必须是 0..255 原始像素（insightface 官方惯例，与 InsightFace2D106 封装一致）。
        早期实现用 to_ufloat32() 把输入除了 255，模型输出会退化：68 点云只有真实人脸的
        ~0.17 倍，写出的 metadata landmarks 尺度错误、对齐取景偏心（表现为"一部分脸正常
        一部分不可用"）。2026-09-30 改为 as_float32()（仅转类型，保持 0..255），修正后
        点云/人脸框比回到 0.87。勿改回 to_ufloat32()。

        返回 (3309,) = reshape(-1,3) 后 1103 行，最后 68 行才是标记点；x/y 为 [-1,1] 惯例
        归一化坐标（(x+1)*W/2 即像素）。
        """
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        ip.resize((192, 192)).ch(3).as_float32()   # 0..255，勿改成 to_ufloat32()
        inp = ip.get_image('NCHW')
        pred = self._sess.run(None, {self._input_name: inp})[0][0]
        return pred
