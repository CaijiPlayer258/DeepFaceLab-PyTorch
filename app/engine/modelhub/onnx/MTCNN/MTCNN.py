from pathlib import Path
from modelhub import weights
from typing import List, Tuple
import math

import cv2
import numpy as np
from PIL import Image

from xlib.onnxruntime import (
    InferenceSession_with_device,
    ORTDeviceInfo,
    get_available_devices_info,
)


class MTCNN:
    """
    MTCNN (Multi-Task Cascaded Convolutional Networks) face detector.

    3-stage cascade: PNet -> RNet -> ONet
    Uses ONNX Runtime for inference of the three sub-networks.
    """

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in MTCNN.get_available_devices():
            raise Exception(f"device_info {device_info} not available for MTCNN")

        # 权重平铺在 weights/ 根下：mtcnn_pnet.onnx / mtcnn_rnet.onnx / mtcnn_onet.onnx
        self._pnet_path = weights.path('mtcnn_pnet.onnx')
        self._rnet_path = weights.path('mtcnn_rnet.onnx')
        self._onet_path = weights.path('mtcnn_onet.onnx')
        self._pnet_sess = InferenceSession_with_device(
            str(self._pnet_path), device_info
        )
        self._rnet_sess = InferenceSession_with_device(
            str(self._rnet_path), device_info
        )
        self._onet_sess = InferenceSession_with_device(
            str(self._onet_path), device_info
        )

        self._pnet_input = self._pnet_sess.get_inputs()[0].name
        self._rnet_input = self._rnet_sess.get_inputs()[0].name
        self._onet_input = self._onet_sess.get_inputs()[0].name

        # ── TRT BF16 加速 ──────────────────────
        for _attr, _fname in [('_pnet_sess', 'mtcnn_pnet.onnx'),
                              ('_rnet_sess', 'mtcnn_rnet.onnx'),
                              ('_onet_sess', 'mtcnn_onet.onnx')]:
            _p = weights.path(_fname)
            _trt_path = None
            try:
                from xlib.trt import find_trt_engine
                _trt_path = find_trt_engine(str(_p), _fname.replace('.onnx', ''))
            except Exception:
                pass
            if _trt_path:
                try:
                    from xlib.trt import TRTInferenceSession
                    setattr(self, _attr, TRTInferenceSession(_trt_path))
                except Exception as e:
                    import warnings as _w
                    _w.warn(f'MTCNN TRT fallback {_fname}: {e}')

        # Default parameters
        self.min_face_size = 20.0
        self.thresholds = [0.7, 0.8, 0.9]
        self.nms_thresholds = [0.7, 0.7, 0.7]
        self.factor = 0.709

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image: (HWC) -> (1, C, HW), normalize to [-1, 1]."""
        img = img.transpose((2, 0, 1))[np.newaxis, :, :, :]
        img = (img - 127.5) * 0.0078125
        return img.astype(np.float32)

    def _nms(self, boxes: np.ndarray, overlap_threshold: float = 0.5, mode: str = "union") -> np.ndarray:
        """Non-maximum suppression. Returns indices of kept boxes."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        pick = []
        x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
        area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        ids = np.argsort(score)

        while len(ids) > 0:
            last = len(ids) - 1
            i = ids[last]
            pick.append(i)

            ix1 = np.maximum(x1[i], x1[ids[:last]])
            iy1 = np.maximum(y1[i], y1[ids[:last]])
            ix2 = np.minimum(x2[i], x2[ids[:last]])
            iy2 = np.minimum(y2[i], y2[ids[:last]])

            w = np.maximum(0.0, ix2 - ix1 + 1.0)
            h = np.maximum(0.0, iy2 - iy1 + 1.0)
            inter = w * h

            if mode == "min":
                overlap = inter / np.minimum(area[i], area[ids[:last]])
            else:
                overlap = inter / (area[i] + area[ids[:last]] - inter)

            ids = np.delete(
                ids,
                np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]),
            )

        return np.array(pick, dtype=np.int64)

    def _calibrate_box(self, bboxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Transform bounding boxes with predicted offsets."""
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = np.expand_dims(x2 - x1 + 1.0, 1)
        h = np.expand_dims(y2 - y1 + 1.0, 1)
        translation = np.hstack([w, h, w, h]) * offsets
        bboxes[:, 0:4] = bboxes[:, 0:4] + translation
        return bboxes

    def _convert_to_square(self, bboxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes to square form."""
        square_bboxes = np.zeros_like(bboxes)
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = np.maximum(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
        square_bboxes[:, 4] = bboxes[:, 4]
        return square_bboxes

    def _get_image_boxes(self, bounding_boxes: np.ndarray, img: np.ndarray, size: int = 24) -> np.ndarray:
        """Crop and resize boxes from image."""
        num_boxes = len(bounding_boxes)
        h_img, w_img = img.shape[:2]

        x1, y1, x2, y2 = [bounding_boxes[:, i] for i in range(4)]
        w, h = x2 - x1 + 1.0, y2 - y1 + 1.0

        x, y, ex, ey = x1.copy(), y1.copy(), x2.copy(), y2.copy()

        dx = np.zeros((num_boxes,))
        dy = np.zeros((num_boxes,))
        edx = w.copy() - 1.0
        edy = h.copy() - 1.0

        ind = np.where(ex > w_img - 1.0)[0]
        edx[ind] = w[ind] + w_img - 2.0 - ex[ind]
        ex[ind] = w_img - 1.0

        ind = np.where(ey > h_img - 1.0)[0]
        edy[ind] = h[ind] + h_img - 2.0 - ey[ind]
        ey[ind] = h_img - 1.0

        ind = np.where(x < 0.0)[0]
        dx[ind] = 0.0 - x[ind]
        x[ind] = 0.0

        ind = np.where(y < 0.0)[0]
        dy[ind] = 0.0 - y[ind]
        y[ind] = 0.0

        dy, edy, dx, edx, y, ey, x, ex, w, h = [
            arr.astype("int32") for arr in [dy, edy, dx, edx, y, ey, x, ex, w, h]
        ]

        img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

        for i in range(num_boxes):
            img_box = np.zeros((h[i], w[i], 3), "uint8")
            img_array = img
            img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = (
                img_array[y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :]
            )

            img_box = cv2.resize(img_box, (size, size))
            img_box = img_box.astype("float32")
            img_box = (img_box - 127.5) * 0.0078125
            img_boxes[i, :, :, :] = img_box.transpose((2, 0, 1))

        return img_boxes.astype(np.float32)

    def _generate_bboxes(
        self, probs: np.ndarray, offsets: np.ndarray, scale: float, threshold: float
    ) -> np.ndarray:
        """Generate bounding boxes from PNet output."""
        stride, cell_size = 2, 12
        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        offsets_arr = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        bounding_boxes = np.vstack(
            [
                np.round((stride * inds[1] + 1.0) / scale),
                np.round((stride * inds[0] + 1.0) / scale),
                np.round((stride * inds[1] + 1.0 + cell_size) / scale),
                np.round((stride * inds[0] + 1.0 + cell_size) / scale),
                score,
                offsets_arr,
            ]
        )
        return bounding_boxes.T

    def extract(
        self, img: np.ndarray, threshold: float = 0.5, fixed_window: int = 0,
        input_mode: str = 'one_stage', resize_mode: str = 'letterbox', input_size: int = None
    ) -> List[List[float]]:
        """
        Detect faces in an image using the MTCNN cascade.

        Args:
            img: BGR image as numpy array (H, W, 3)
            threshold: detection confidence threshold (overrides all 3 stage thresholds)
            fixed_window: unused (for API compatibility)

        Returns:
            List of [x1, y1, x2, y2] face bounding boxes in original image space.
        """
        if threshold != 0.5:
            self.thresholds = [threshold * 1.4, threshold * 1.6, threshold * 1.8]
            self.thresholds = [min(t, 0.95) for t in self.thresholds]

        # Convert BGR to RGB (PIL format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        min_length = min(height, width)

        # Build image pyramid
        min_detection_size = 12
        m = min_detection_size / self.min_face_size
        min_length_scaled = min_length * m

        scales = []
        factor_count = 0
        while min_length_scaled > min_detection_size:
            scales.append(m * self.factor**factor_count)
            min_length_scaled *= self.factor
            factor_count += 1

        # STAGE 1: PNet
        all_boxes = []
        for s in scales:
            sw, sh = math.ceil(width * s), math.ceil(height * s)
            scaled_img = cv2.resize(img_rgb, (sw, sh))
            input_blob = self._preprocess(scaled_img)

            offsets, probs = self._pnet_sess.run(None, {self._pnet_input: input_blob})
            probs = probs[0, 1, :, :]  # face probability

            boxes = self._generate_bboxes(probs, offsets, s, self.thresholds[0])
            if len(boxes) > 0:
                keep = self._nms(boxes[:, 0:5], 0.5)
                all_boxes.append(boxes[keep])

        if not all_boxes:
            return [[]]

        bounding_boxes = np.vstack(all_boxes)
        keep = self._nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = self._calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = self._convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        if len(bounding_boxes) == 0:
            return [[]]

        # STAGE 2: RNet
        img_boxes = self._get_image_boxes(bounding_boxes, img_rgb, size=24)
        if len(img_boxes) > 0:
            offsets_rnet, probs_rnet = self._rnet_sess.run(
                None, {self._rnet_input: img_boxes}
            )
            face_probs = probs_rnet[:, 1]

            keep = np.where(face_probs > self.thresholds[1])[0]
            if len(keep) == 0:
                return [[]]

            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = face_probs[keep].reshape(-1)
            offsets_rnet = offsets_rnet[keep]

            keep = self._nms(bounding_boxes, self.nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = self._calibrate_box(bounding_boxes, offsets_rnet[keep])
            bounding_boxes = self._convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        if len(bounding_boxes) == 0:
            return [[]]

        # STAGE 3: ONet
        img_boxes = self._get_image_boxes(bounding_boxes, img_rgb, size=48)
        if len(img_boxes) == 0:
            return [[]]

        landmarks, offsets_onet, probs_onet = self._onet_sess.run(
            None, {self._onet_input: img_boxes}
        )
        face_probs = probs_onet[:, 1]

        keep = np.where(face_probs > self.thresholds[2])[0]
        if len(keep) == 0:
            return [[]]

        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = face_probs[keep]
        offsets_onet = offsets_onet[keep]
        landmarks = landmarks[keep]

        # Compute landmark points
        bb_width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        bb_height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(bb_width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(bb_height, 1) * landmarks[:, 5:10]

        bounding_boxes = self._calibrate_box(bounding_boxes, offsets_onet)
        keep = self._nms(bounding_boxes, self.nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]

        if len(bounding_boxes) == 0:
            return [[]]

        # Return as list of [x1, y1, x2, y2]
        H, W = img.shape[:2]
        faces = []
        for d in bounding_boxes:
            x1, y1, x2, y2 = (
                max(0, int(d[0])),
                max(0, int(d[1])),
                min(W, int(d[2])),
                min(H, int(d[3])),
            )
            if x2 > x1 and y2 > y1:
                faces.append((float(x1), float(y1), float(x2), float(y2)))

        return [faces] if faces else [[]]
