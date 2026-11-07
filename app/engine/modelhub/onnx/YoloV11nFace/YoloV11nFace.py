"""YOLOv11n-Face face detector."""
from pathlib import Path
from modelhub import weights
import numpy as np
import cv2
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)


class YoloV11nFace:
    @staticmethod
    def get_available_devices():
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        path = weights.path('yolov11n_face.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'yolov11n-face')
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

        # YOLO preprocessing: BGR→RGB, resize to 640x640 (stretch)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :]

        out = self._sess.run(None, {'images': blob})[0][0].T  # (8400, 5)

        if out.shape[0] == 0:
            return [[]]

        scores = out[:, 4]
        keep = scores >= threshold
        out = out[keep]
        if len(out) == 0:
            return [[]]

        # Decode: [cx, cy, w, h] format
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x1 = (cx - w / 2) / 640.0 * W
        y1 = (cy - h / 2) / 640.0 * H
        x2 = (cx + w / 2) / 640.0 * W
        y2 = (cy + h / 2) / 640.0 * H

        # NMS
        keep_idx = self._nms(x1, y1, x2, y2, out[:, 4], 0.4)

        faces = []
        for i in keep_idx:
            ix1, iy1 = max(0, int(x1[i])), max(0, int(y1[i]))
            ix2, iy2 = min(W, int(x2[i])), min(H, int(y2[i]))
            if min(ix2 - ix1, iy2 - iy1) >= min_face_size:
                faces.append((float(ix1), float(iy1), float(ix2), float(iy2)))
        return [faces] if faces else [[]]

    def _nms(self, x1, y1, x2, y2, scores, thresh):
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep
