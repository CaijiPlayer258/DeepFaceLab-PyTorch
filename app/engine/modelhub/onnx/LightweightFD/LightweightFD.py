"""Lightweight-Face-Detection (Qualcomm AI Hub) using ONNX."""
from pathlib import Path
from modelhub import weights
import numpy as np
import cv2
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class LightweightFD:
    @staticmethod
    def get_available_devices():
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        path = weights.path('onnx/LightweightFD/face_det_lite.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'face_det_lite')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def extract(self, img, threshold=0.5, fixed_window=0, min_face_size=20,
                input_mode='one_stage', resize_mode='letterbox', input_size=None):
        H, W = img.shape[:2]
        blue = img[:, :, 0].astype(np.float32)
        resized = cv2.resize(blue, (640, 480))
        tensor = resized[np.newaxis, np.newaxis, :, :] / 255.0

        heatmap, bbox, landmark = self._sess.run(None, {self._sess.get_inputs()[0].name: tensor})

        scores = 1.0 / (1.0 + np.exp(-heatmap[0, 0]))
        b = bbox[0]  # (4, 60, 80)

        # bbox values seem to be per-pixel offsets in a different space
        # Use heatmap peak position to estimate bbox
        keep = scores >= threshold
        if not keep.any():
            return [[]]

        ys, xs = np.where(keep)
        face_list = []
        for yi, xi in zip(ys, xs):
            sc = scores[yi, xi]
            # Heatmap position in 640x480 space
            cx = (xi + 0.5) / 80.0 * 640
            cy = (yi + 0.5) / 60.0 * 480
            # Estimate size from bbox values (noisy but gives reasonable scale)
            bw = max(20, abs(b[2, yi, xi]) * 8)
            bh = max(20, abs(b[3, yi, xi]) * 8)

            x1 = (cx - bw / 2) / 640 * W
            y1 = (cy - bh / 2) / 480 * H
            x2 = (cx + bw / 2) / 640 * W
            y2 = (cy + bh / 2) / 480 * H
            face_list.append((x1, y1, x2, y2, sc))

        if not face_list:
            return [[]]

        # NMS
        boxes = np.array([[f[0], f[1], f[2], f[3]] for f in face_list])
        sc_arr = np.array([f[4] for f in face_list])
        keep_idx = self._nms(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], sc_arr, 0.4)

        faces = []
        for i in keep_idx:
            x1, y1, x2, y2, _ = face_list[i]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if min(x2 - x1, y2 - y1) >= min_face_size:
                faces.append((float(x1), float(y1), float(x2), float(y2)))
        return [faces] if faces else [[]]

    def _nms(self, x1, y1, x2, y2, scores, thresh):
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep
