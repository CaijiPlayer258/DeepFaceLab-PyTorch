"""BlazeFace face detector with sliding window."""
from pathlib import Path
from modelhub import weights
import math
import cv2
import numpy as np
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)
from xlib.face import FRect


class BlazeFace:
    @staticmethod
    def get_available_devices():
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        path = weights.path('blazeface.onnx')
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), 'blaze')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def extract(self, img, threshold=0.5, fixed_window=0,
                enable_alignment=True, return_frect=False, return_landmarks=False,
                input_mode='one_stage', resize_mode='letterbox', input_size=None):
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        N, H, W = img.shape[0], img.shape[1], img.shape[2]

        # Pre-scale: BlazeFace expects faces to fill ~60-80% of 128x128.
        # Scale so the image fits in ~160px on the longest side.
        target = 160.0
        scale = min(1.0, target / max(H, W))
        new_w, new_h = int(W * scale), int(H * scale)
        skip_resize = (scale >= 1.0)

        stride = 80
        tile_size = 128
        results_batch = [[] for _ in range(N)]

        for b in range(N):
            frame = img[b]
            if not skip_resize:
                frame = cv2.resize(frame, (new_w, new_h))
            hh, ww = frame.shape[:2]
            all_dets = []

            for y in range(0, hh, stride):
                for x in range(0, ww, stride):
                    x1, y1 = x, y
                    x2, y2 = min(x + tile_size, ww), min(y + tile_size, hh)
                    tile = frame[y1:y2, x1:x2]
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        tile = cv2.copyMakeBorder(tile, 0, tile_size - tile.shape[0],
                                                  0, tile_size - tile.shape[1],
                                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

                    rgb = tile[:, :, ::-1].astype(np.float32) / 255.0
                    inp = np.transpose(rgb, (2, 0, 1))[np.newaxis, :]

                    try:
                        outs = self._sess.run(None, {self._input_name: inp,
                            "conf_threshold": np.array([threshold], dtype=np.float32),
                            "max_detections": np.array([25], dtype=np.int64),
                            "iou_threshold": np.array([0.3], dtype=np.float32)})
                    except Exception:
                        continue

                    boxes = outs[0][0]
                    if boxes.ndim == 1:
                        boxes = boxes.reshape(1, 16)
                    scores_arr = outs[1][0] if len(outs) > 1 else np.ones(len(boxes), dtype=np.float32)

                    for det, sc in zip(boxes, scores_arr):
                        if sc < threshold:
                            continue
                        (top_y, top_x, bot_y, bot_x,
                         ley_x, ley_y, rey_x, rey_y,
                         nose_x, nose_y, mou_x, mou_y,
                         lea_x, lea_y, rea_x, rea_y) = det

                        # Normalized [0,1] in tile -> original image coords
                        nx1 = (top_x * tile_size + x1) / scale
                        ny1 = (top_y * tile_size + y1) / scale
                        nx2 = (bot_x * tile_size + x1) / scale
                        ny2 = (bot_y * tile_size + y1) / scale
                        nx1, ny1 = max(0, int(nx1)), max(0, int(ny1))
                        nx2, ny2 = min(W, int(nx2)), min(H, int(ny2))
                        if nx2 - nx1 < 5 or ny2 - ny1 < 5:
                            continue

                        left_eye = np.array([ley_x * tile_size + x1, ley_y * tile_size + y1], dtype=np.float32) / scale
                        right_eye = np.array([rey_x * tile_size + x1, rey_y * tile_size + y1], dtype=np.float32) / scale
                        nose = np.array([nose_x * tile_size + x1, nose_y * tile_size + y1], dtype=np.float32) / scale
                        left_mouth = np.array([mou_x * tile_size + x1, mou_y * tile_size + y1], dtype=np.float32) / scale
                        right_mouth = np.array([lea_x * tile_size + x1, lea_y * tile_size + y1], dtype=np.float32) / scale

                        all_dets.append((nx1, ny1, nx2, ny2, sc,
                                         left_eye, right_eye, nose, left_mouth, right_mouth))

            if not all_dets:
                results_batch[b] = []
                continue

            # Global NMS
            ba = np.array([[d[0], d[1], d[2], d[3]] for d in all_dets])
            sa = np.array([d[4] for d in all_dets])
            keep = self._nms(ba[:, 0], ba[:, 1], ba[:, 2], ba[:, 3], sa, 0.4)

            faces = []
            for i in keep:
                x1, y1, x2, y2, sc, le, re, no, lm, rm = all_dets[i]
                if enable_alignment:
                    eye_dx, eye_dy = re[0] - le[0], re[1] - le[1]
                    angle_rad = math.atan2(eye_dy, eye_dx)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    half = max(x2 - x1, y2 - y1) / 2.0
                    corners = np.array([[-half, -half], [-half, half],
                                        [half, half], [half, -half]], dtype=np.float32)
                    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
                    rotated = (corners @ rot.T) + np.array([cx, cy])
                    rotated[:, 0] = np.clip(rotated[:, 0], 0, W)
                    rotated[:, 1] = np.clip(rotated[:, 1], 0, H)
                    ax1 = max(0, int(np.min(rotated[:, 0])))
                    ay1 = max(0, int(np.min(rotated[:, 1])))
                    ax2 = min(W, int(np.max(rotated[:, 0])))
                    ay2 = min(H, int(np.max(rotated[:, 1])))
                    if ax2 - ax1 >= 5 and ay2 - ay1 >= 5:
                        faces.append([ax1, ay1, ax2, ay2])
                else:
                    faces.append([x1, y1, x2, y2])

            results_batch[b] = faces

        return results_batch

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
