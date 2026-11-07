from pathlib import Path
from modelhub import weights
from typing import List, Tuple
import math

import cv2
import numpy as np

from xlib.onnxruntime import (
    InferenceSession_with_device,
    ORTDeviceInfo,
    get_available_devices_info,
)


class MogFace:
    """
    MogFace: ResNet101-based face detector from CVPR 2022.

    Uses ONNX Runtime for inference.
    Input: BGR image
    Output: face bounding boxes in original image space [x1, y1, x2, y2]
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in MogFace.get_available_devices():
            raise Exception(f"device_info {device_info} not available for MogFace")

        model_path = weights.path("mogface.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"MogFace ONNX model not found at {model_path}")

        self._sess = InferenceSession_with_device(str(model_path), device_info)
        self._input_name = self._sess.get_inputs()[0].name

        # MogFace parameters
        self.confidence_threshold = 0.82
        self.nms_threshold = 0.4
        self.top_k = 5000
        self.keep_top_k = 750

        # Precomputed prior boxes (lazy init on first call)
        self._priors = None

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(model_path), 'MogFace')
        except Exception:
            pass
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def _get_priors(self, height: int, width: int) -> np.ndarray:
        """Generate MogFace prior boxes (inlined from modelscope MogPriorBox)."""
        scale_list = [0.68]
        stride_list = [4, 8, 16, 32, 64, 128]
        anchor_size_list = [16, 32, 64, 128, 256, 512]

        final_anchors = []
        for idx, stride in enumerate(stride_list):
            cur_h, cur_w = height, width
            tmp = stride
            while tmp != 1:
                tmp //= 2
                cur_h = (cur_h + 1) // 2
                cur_w = (cur_w + 1) // 2
            for i in range(cur_h):
                for j in range(cur_w):
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    side = anchor_size_list[idx] * scale_list[0]
                    final_anchors.append([cx, cy, side, side])

        anchors = np.array(final_anchors, dtype=np.float32)
        # normalize: [cx,cy,w,h] -> [x0,y0,x1,y1]
        item1 = anchors[:, :2] - (anchors[:, 2:] - 1) / 2
        item2 = anchors[:, :2] + (anchors[:, 2:] - 1) / 2
        normalized = np.concatenate([item1, item2], axis=1)
        # transform: [x0,y0,x1,y1] -> [cx,cy,w,h]
        transformed = np.concatenate([
            (normalized[:, :2] + normalized[:, 2:]) / 2,
            normalized[:, 2:] - normalized[:, :2] + 1
        ], axis=1)
        return transformed.astype(np.float32)

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocess BGR image for MogFace model.

        Pads image to multiples of 32 to avoid FPN shape mismatches.

        Returns: (blob, scale_factor, pad_top, pad_left)
        where blob has shape (1, 3, H, W) and scale_factor is the resize factor.
        """
        h, w = img.shape[:2]
        scale = 1.0

        # Optional downscale if too large
        if max(h, w) > 1500:
            scale = 1000.0 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            h, w = img.shape[:2]

        # Pad to multiples of 32 to avoid FPN shape mismatches (e.g., 63 vs 64)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        pad_top, pad_left = pad_h, pad_w
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = img.shape[:2]

        # Convert to float
        img = img.astype(np.float32)

        # BGR -> RGB
        img_rgb = img[:, :, ::-1].copy()

        # Normalize with RGB-ordered mean/std
        img_rgb -= np.array([[103.53, 116.28, 123.675]], dtype=np.float32)
        img_rgb /= np.array([[57.375, 57.120003, 58.395]], dtype=np.float32)
        img_rgb /= 255.0

        # RGB -> BGR back
        img_bgr = img_rgb[:, :, ::-1].copy()

        # HWC -> CHW with batch dimension
        blob = img_bgr.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return blob.astype(np.float32), scale, pad_top, pad_left

    def _mogdecode(self, loc: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Decode location predictions to bounding boxes (adapted from modelscope mogdecode)."""
        boxes = np.zeros_like(loc)
        boxes[:, :2] = anchors[:, :2] + loc[:, :2] * anchors[:, 2:]
        boxes[:, 2:] = anchors[:, 2:] * np.exp(loc[:, 2:])

        boxes[:, 0] -= (boxes[:, 2] - 1) / 2
        boxes[:, 1] -= (boxes[:, 3] - 1) / 2
        boxes[:, 2] += boxes[:, 0] - 1
        boxes[:, 3] += boxes[:, 1] - 1
        return boxes

    def _py_cpu_nms(self, dets: np.ndarray, thresh: float) -> np.ndarray:
        """Pure Python NMS."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

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

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

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
        """
        self.confidence_threshold = threshold

        # Preprocess (returns padding offsets for non-multiple-of-32 dimensions)
        blob, scale, pad_top, pad_left = self._preprocess(img)

        # Run inference
        confidences, locations = self._sess.run(None, {self._input_name: blob})

        # Get actual image dimensions after preprocessing
        _, _, img_height, img_width = blob.shape

        # Generate priors for this size
        priors = self._get_priors(img_height, img_width)

        # Decode boxes
        loc = locations[0]
        conf = confidences[0, :, 0]
        boxes = self._mogdecode(loc, priors)

        # Remove padding offset from box coordinates
        if pad_top > 0:
            boxes[:, 1] -= pad_top
            boxes[:, 3] -= pad_top
        if pad_left > 0:
            boxes[:, 0] -= pad_left
            boxes[:, 2] -= pad_left

        # Filter by threshold
        inds = np.where(conf > self.confidence_threshold)[0]
        if len(inds) == 0:
            return [[]]

        boxes = boxes[inds]
        scores = conf[inds]

        # Keep top-k before NMS
        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # Scale back if image was resized
        if scale != 1.0:
            boxes /= scale

        # NMS
        dets = np.column_stack([boxes, scores]).astype(np.float32)
        dets = self._py_cpu_nms(dets, self.nms_threshold)

        # Keep top-k after NMS
        dets = dets[: self.keep_top_k]

        if len(dets) == 0:
            return [[]]

        # Clip to original image bounds and return
        H, W = img.shape[:2]
        faces = []
        for d in dets:
            x1, y1, x2, y2 = (
                max(0, int(d[0])),
                max(0, int(d[1])),
                min(W, int(d[2])),
                min(H, int(d[3])),
            )
            if x2 > x1 and y2 > y1:
                faces.append((float(x1), float(y1), float(x2), float(y2)))

        return [faces] if faces else [[]]
