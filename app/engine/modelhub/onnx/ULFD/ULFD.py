from pathlib import Path
from modelhub import weights
from typing import List, Tuple

import cv2
import numpy as np

from xlib.onnxruntime import (
    InferenceSession_with_device,
    ORTDeviceInfo,
    get_available_devices_info,
)


class ULFD:
    """
    ULFD (Ultra-Light-Fast-Generic-Face-Detector) - lightweight SSD face detector.

    Input: 640x480 BGR image
    Preprocessing: BGR->RGB, resize to (640, 480), subtract mean 127, divide by 128
    Output: face bounding boxes in original image space [x1, y1, x2, y2]
    """

    INPUT_SIZE = (640, 480)  # (width, height)

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in ULFD.get_available_devices():
            raise Exception(f"device_info {device_info} not available for ULFD")

        model_path = weights.path("ulfd.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"ULFD ONNX model not found at {model_path}")

        self._sess = InferenceSession_with_device(str(model_path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # ULFD parameters
        self.img_width, self.img_height = self.INPUT_SIZE
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = 128.0
        self.nms_thresh = 0.3
        self.det_thresh = 0.5
        self.candidate_size = 1500
        self.keep_top_k = 750

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(model_path), 'ULFD')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image: BGR->RGB, resize, normalize, HWC->CHW."""
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to input size (maintaining aspect ratio is NOT needed for ULFD,
        # it distorts to the fixed input size)
        resized = cv2.resize(img_rgb, (self.img_width, self.img_height))

        # Normalize: subtract mean, divide by std
        resized = resized.astype(np.float32)
        resized -= self.mean
        resized /= self.std

        # HWC -> CHW with batch dimension
        blob = resized.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return blob.astype(np.float32)

    def _nms(self, dets: np.ndarray) -> np.ndarray:
        """Non-maximum suppression."""
        x1, y1, x2, y2, scores = (
            dets[:, 0],
            dets[:, 1],
            dets[:, 2],
            dets[:, 3],
            dets[:, 4],
        )
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        return dets[keep]

    def extract(
        self, img: np.ndarray, threshold: float = 0.5, fixed_window: int = 0,
        input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = None
    ) -> List[List[float]]:
        """
        Detect faces in an image.

        Args:
            img: BGR image as numpy array (H, W, 3)
            threshold: detection confidence threshold
            fixed_window: unused (for API compatibility)

        Returns:
            List of [x1, y1, x2, y2] face bounding boxes in original image space.
            Wrapped in an outer list for multi-image API compatibility.
        """
        orig_h, orig_w = img.shape[:2]
        self.det_thresh = threshold

        # Preprocess
        blob = self._preprocess(img)

        # Run inference
        confidences, boxes = self._sess.run(None, {self._input_name: blob})
        confidences = confidences[0]  # (N, 2) - bg, face scores
        boxes = boxes[0]  # (N, 4) - normalized corner form

        # Filter by threshold (class index 1 = face)
        face_scores = confidences[:, 1]
        mask = face_scores > self.det_thresh
        if not np.any(mask):
            return [[]]

        filtered_boxes = boxes[mask]
        filtered_scores = face_scores[mask]

        # Sort by score, keep top-k
        order = filtered_scores.argsort()[::-1][: self.candidate_size]
        filtered_boxes = filtered_boxes[order]
        filtered_scores = filtered_scores[order]

        # Scale boxes back to original image coordinates
        filtered_boxes[:, 0] *= orig_w
        filtered_boxes[:, 1] *= orig_h
        filtered_boxes[:, 2] *= orig_w
        filtered_boxes[:, 3] *= orig_h

        # Clip to image bounds
        filtered_boxes[:, 0] = np.clip(filtered_boxes[:, 0], 0, orig_w)
        filtered_boxes[:, 1] = np.clip(filtered_boxes[:, 1], 0, orig_h)
        filtered_boxes[:, 2] = np.clip(filtered_boxes[:, 2], 0, orig_w)
        filtered_boxes[:, 3] = np.clip(filtered_boxes[:, 3], 0, orig_h)

        # Combine boxes and scores for NMS
        dets = np.column_stack([filtered_boxes, filtered_scores])

        # Apply NMS
        dets = self._nms(dets)

        # Keep top-k after NMS
        dets = dets[: self.keep_top_k]

        if len(dets) == 0:
            return [[]]

        # Return as list of [x1, y1, x2, y2]
        faces = [
            (float(x1), float(y1), float(x2), float(y2))
            for x1, y1, x2, y2, _ in dets
        ]
        return [faces]
