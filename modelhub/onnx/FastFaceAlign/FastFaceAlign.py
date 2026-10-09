from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class FastFaceAlign:
    """
    FastFaceAlign — ultra-lightweight one-stage face detection with rotation alignment.

    Single model detects a rotated square face box ``(cx, cy, w, h, θ)`` in one shot:
      - conf : objectness logit (sigmoid)
      - reg  : (cx_offset, cy_offset, s, θ) — sigmoid offset, exp side (×stride), tanh angle (×π)
    No NMS, no landmarks — the angle θ directly drives rotation-corrected alignment.

    Input (B,3,192,192) -> Output conf (B,1,12,12), reg (B,4,12,12). Feature stride = 16.

    Weight source: FastFaceAlign project `runs/fastface_wider/model_iter_830000.onnx`
    (highest trained checkpoint, 830k iters).
    """

    IMG_SIZE = 192
    STRIDE = 16

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in FastFaceAlign.get_available_devices():
            raise Exception(f'device_info {device_info} is not in available devices for FastFaceAlign')

        path = Path(__file__).parent / 'FastFaceAlign.onnx'
        self._sess = sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = sess.get_inputs()[0].name

        # ── TRT 加速 ─────────────────────────────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'FastFaceAlign')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    # ------------------------------------------------------------------
    #  Decode helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _decode(self, conf: np.ndarray, reg: np.ndarray, threshold: float):
        """
        Decode one batch item's raw outputs to rotated boxes.

        Args:
            conf: (1, Hf, Wf) raw logits
            reg : (4, Hf, Wf) (cx_offset, cy_offset, s, θ) raw logits
            threshold: confidence threshold

        Returns:
            list of (conf, cx, cy, w, h, θ) in 192×192 input space, conf descending.
        """
        Hf, Wf = conf.shape[1], conf.shape[2]
        conf_s = self._sigmoid(conf[0])                      # (Hf, Wf)
        cx_off = self._sigmoid(reg[0])                       # (Hf, Wf)
        cy_off = self._sigmoid(reg[1])
        side = np.exp(reg[2]) * self.STRIDE                  # square side in px
        theta = np.tanh(reg[3]) * np.pi                      # [-π, π]

        rows, cols = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing='ij')
        cx = (cols + cx_off) * self.STRIDE                   # (Hf, Wf)
        cy = (rows + cy_off) * self.STRIDE

        ys, xs = np.where(conf_s > threshold)
        items = []
        for y, x in zip(ys, xs):
            items.append((float(conf_s[y, x]), float(cx[y, x]), float(cy[y, x]),
                          float(side[y, x]), float(side[y, x]), float(theta[y, x])))
        items.sort(key=lambda t: t[0], reverse=True)
        return items

    # ------------------------------------------------------------------
    #  Standard bbox interface (compatible with DetectorFactory)
    # ------------------------------------------------------------------
    def extract(self, img, threshold: float = 0.5, fixed_window=0, min_face_size=40,
                input_mode='one_stage', resize_mode='letterbox', input_size=None):
        """
        Standard detector interface — returns axis-aligned [l,t,r,b] per batch item.
        (bbox = rotated square's circumscribed axis-aligned rect)

        arguments
         img    np.ndarray      ndim 2,3,4  (BGR)
         threshold(0.5)         confidence threshold
         min_face_size(40)      minimum face side in original image pixels

        returns a list of [l,t,r,b] for every batch dimension of img
        """
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        N, H, W, _ = ip.get_dims()

        img_scale = ip.fit_in(self.IMG_SIZE, self.IMG_SIZE, pad_to_target=True, allow_upscale=False)
        ip.ch(3).swap_ch().to_uint8().as_float32()
        x = ip.get_image('NCHW')                             # (N,3,192,192) float32

        conf, reg = self._sess.run(None, {self._input_name: x})

        faces_per_batch = []
        for b in range(N):
            faces = []
            for c, cx, cy, w, h, th in self._decode(conf[b], reg[b], threshold):
                if img_scale != 1.0:
                    cx, cy, w, h = cx / img_scale, cy / img_scale, w / img_scale, h / img_scale
                l, t, r, btm = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                # clamp 到图像范围（与其他检测器一致）
                l, t = max(0.0, l), max(0.0, t)
                r, btm = min(float(W), r), min(float(H), btm)
                if r <= l or btm <= t:
                    continue
                if min(r - l, btm - t) < min_face_size:
                    continue
                faces.append((l, t, r, btm))
            faces_per_batch.append(faces)
        return faces_per_batch

    # ------------------------------------------------------------------
    #  Rotated-box interface (rotation-corrected alignment)
    # ------------------------------------------------------------------
    def extract_rotated(self, img, threshold: float = 0.5, min_face_size=40,
                        input_mode='one_stage', resize_mode='letterbox', input_size=None):
        """
        Detect rotated face boxes, mapped back to original image space.

        Returns a list per batch item, each a list of
        (conf, cx, cy, w, h, θ) — rotated square box, conf descending.
        θ in radians, positive = counter-clockwise (OpenCV convention).
        """
        from xlib.image import ImageProcessor
        ip = ImageProcessor(img)
        N, H, W, _ = ip.get_dims()

        img_scale = ip.fit_in(self.IMG_SIZE, self.IMG_SIZE, pad_to_target=True, allow_upscale=False)
        ip.ch(3).swap_ch().to_uint8().as_float32()
        x = ip.get_image('NCHW')

        conf, reg = self._sess.run(None, {self._input_name: x})

        faces_per_batch = []
        for b in range(N):
            faces = []
            for c, cx, cy, w, h, th in self._decode(conf[b], reg[b], threshold):
                if img_scale != 1.0:
                    cx, cy, w, h = cx / img_scale, cy / img_scale, w / img_scale, h / img_scale
                if min(w, h) < min_face_size:
                    continue
                faces.append((c, cx, cy, w, h, th))
            faces_per_batch.append(faces)
        return faces_per_batch
