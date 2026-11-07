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
from ..det_preprocess import preprocess, map_back, sliding_window_detect

# SCRFD-style evaluation uses score_thr=0.02 for candidate selection.
# Default extract threshold is higher for practical use.
SCORE_THRESHOLD = 0.02
DEFAULT_EXTRACT_THRESHOLD = 0.5
TOP_K_BEFORE_NMS = 5000
MAX_FACES_PER_IMAGE = 100


class DamoFD:
    """
    DamoFD / SCRFD face detector (ICLR 2023).

    Uses ONNX Runtime for inference. Supports both:
    - Standard interleaved ONNX format (ours): [cls_0, bbox_0, kps_0, cls_1, ...]
    - Insightface grouped format: [cls_0, cls_1, cls_2, bbox_0, bbox_1, bbox_2, kps_0, kps_1, kps_2]

    Input: BGR image
    Output: face bounding boxes in original image space [x1, y1, x2, y2]
    """

    INPUT_SIZE = 640

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo):
        if device_info not in DamoFD.get_available_devices():
            raise Exception(f"device_info {device_info} not available for DamoFD")

        model_path = weights.path("damofd.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"DamoFD ONNX model not found at {model_path}")

        self._sess = InferenceSession_with_device(str(model_path), device_info)

        # Detect output format from model
        outputs = self._sess.get_outputs()
        self._num_outputs = len(outputs)

        # Check for insightface-style grouped format
        # cls[0:3], bbox[3:6], kps[6:9]
        # In grouped format, the first 3 outputs have shape (N, 1)
        # In interleaved format, outputs alternate cls/bbox/kps
        if self._num_outputs >= 3:
            first_shape = outputs[0].shape
            # If first output has 3 dims (batch), it's our interleaved format
            # If first output has 2 dims (no batch), it's insightface grouped format
            self._batched = len(first_shape) == 3
            self._grouped_format = not self._batched
        else:
            self._batched = True
            self._grouped_format = False

        # SCRFD params
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.fmc = 3  # number of FPN levels
        self.use_kps = True
        self.nms_thresh = 0.4
        self.det_thresh = SCORE_THRESHOLD
        self.input_mean = 127.5
        self.input_std = 128.0
        self._top_k = TOP_K_BEFORE_NMS

        # Cache for anchor centers (performance optimization)
        self._center_cache = {}

        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(model_path), 'DamoFD')
        except Exception:
            pass
        self._use_trt = False
        if _trt_path:
            try:
                from xlib.trt import TRTInferenceSession
                self._sess = TRTInferenceSession(_trt_path)
                self._use_trt = True
            except Exception as e:
                import warnings as _w
                _w.warn(f'TRT fallback: {e}')

    def _get_outs(self, net_outs, idx, stride):
        """Extract scores, bbox_preds, kps_preds for a given FPN level.

        Handles both output formats transparently.
        """
        if self._grouped_format:
            # Insightface format: cls[0:3], bbox[3:6], kps[6:9]
            scores = net_outs[idx]                     # (N, 1)
            bbox_preds = net_outs[idx + self.fmc] * stride       # (N, 4)
            kps_preds = net_outs[idx + self.fmc * 2] * stride    # (N, 10)
        else:
            # Our interleaved format: cls_0, bbox_0, kps_0, cls_1, ...
            if self._batched:
                scores = net_outs[idx * 3][0]           # (B, N, 1) -> (N, 1)
                bbox_preds = net_outs[idx * 3 + 1][0] * stride
                kps_preds = net_outs[idx * 3 + 2][0] * stride
            else:
                scores = net_outs[idx * 3]              # (N, 1)
                bbox_preds = net_outs[idx * 3 + 1] * stride
                kps_preds = net_outs[idx * 3 + 2] * stride
        return scores, bbox_preds, kps_preds

    def _distance2bbox(self, points, distance):
        """Decode distance prediction to bounding box."""
        return np.stack([
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ], axis=-1)

    def _distance2kps(self, points, distance):
        """Decode distance prediction to keypoints."""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _forward(self, img):
        """Run model on a preprocessed image."""
        input_size = tuple(img.shape[0:2][::-1])  # (W, H) 从实际处理图推导
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                      (self.input_mean,) * 3, swapRB=True)

        net_outs = self._sess.run(None, {self._sess.get_inputs()[0].name: blob})
        input_height, input_width = blob.shape[2:]

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores, bbox_preds, kps_preds = self._get_outs(net_outs, idx, stride)

            height = input_height // stride
            width = input_width // stride

            # Generate or retrieve anchor centers
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self._center_cache) < 100:
                    self._center_cache[key] = anchor_centers

            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            kpss = self._distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))

            pos_inds = np.where(scores.ravel() >= self.det_thresh)[0]
            scores_list.append(scores.ravel()[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def _detect_window(self, win):
        """单窗口检测（滑窗用），返回窗口内坐标 boxes (N,5)。"""
        scores_list, bboxes_list, kpss_list = self._forward(win)
        if not scores_list or sum(s.size for s in scores_list) == 0:
            return np.empty((0, 5))
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list, axis=0)
        if len(scores) > self._top_k:
            top_k_inds = np.argpartition(scores, -self._top_k)[-self._top_k:]
            scores = scores[top_k_inds]
            bboxes = bboxes[top_k_inds]
        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)
        return pre_det[order, :]

    def _detect_candidates(self, img, input_size=None, input_mode='one_stage', resize_mode='letterbox'):
        """按 input_mode 检测：one_stage 整图缩放 / sliding_window 滑窗扫描。"""
        # TRT engine 固定 640×640 输出，强制 640 尺寸
        if self._use_trt:
            input_size = self.INPUT_SIZE
        if input_size is None:
            input_size = self.INPUT_SIZE

        if input_mode == 'sliding_window':
            boxes = sliding_window_detect(img, int(input_size), resize_mode,
                                          self._detect_window, pad_value=0)
            return boxes, np.empty((0, 5, 2))

        det_img, meta = preprocess(img, resize_mode, int(input_size), pad_value=0)
        scores_list, bboxes_list, kpss_list = self._forward(det_img)
        if not scores_list or sum(s.size for s in scores_list) == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        scores = np.concatenate(scores_list)
        bboxes_list_concat = np.concatenate(bboxes_list, axis=0)
        kpss_list_concat = np.concatenate(kpss_list, axis=0)
        # Keep top-K by score before NMS (performance safeguard)
        if len(scores) > self._top_k:
            top_k_inds = np.argpartition(scores, -self._top_k)[-self._top_k:]
            scores = scores[top_k_inds]
            bboxes = bboxes_list_concat[top_k_inds]
            kpss = kpss_list_concat[top_k_inds]
        else:
            bboxes = bboxes_list_concat
            kpss = kpss_list_concat

        bboxes = map_back(bboxes, meta)
        kpss = map_back(kpss, meta)

        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)
        pre_det = pre_det[order, :]
        kpss = kpss[order, :, :]
        return pre_det, kpss

    def _nms(self, dets):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        return keep

    def extract(self, img, threshold=DEFAULT_EXTRACT_THRESHOLD, fixed_window=0, min_face_size=20,
                input_mode='one_stage', resize_mode='letterbox', input_size=None):
        """Detect faces, return list of [l,t,r,b].

        Internal candidate selection uses a low threshold (0.02) as per SCRFD
        evaluation practice. The `threshold` parameter controls the final
        confidence filter applied after NMS.
        """
        self.det_thresh = SCORE_THRESHOLD

        pre_det, kpss = self._detect_candidates(img, input_size, input_mode, resize_mode)
        if len(pre_det) == 0:
            return [[]]

        keep = self._nms(pre_det)
        det = pre_det[keep]

        # Apply final score threshold after NMS
        score_mask = det[:, 4] >= threshold
        det = det[score_mask]

        if len(det) == 0:
            return [[]]

        # Limit max faces
        det = det[:MAX_FACES_PER_IMAGE]

        H, W = img.shape[:2]
        faces = []
        for d in det:
            x1, y1, x2, y2 = max(0, int(d[0])), max(0, int(d[1])), min(W, int(d[2])), min(H, int(d[3]))
            if min(x2 - x1, y2 - y1) >= min_face_size:
                faces.append((float(x1), float(y1), float(x2), float(y2)))
        return [faces]

    def extract_with_kps(self, img, threshold=DEFAULT_EXTRACT_THRESHOLD, fixed_window=0, min_face_size=20,
                         input_mode='one_stage', resize_mode='letterbox', input_size=None):
        """Detect faces returning both boxes and 5-point keypoints.
        Returns: [(box, kps_or_none), ...] where box=[l,t,r,b], kps=[[x,y],...x5]
        Keypoints order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        self.det_thresh = SCORE_THRESHOLD
        pre_det, kpss = self._detect_candidates(img, input_size, input_mode, resize_mode)
        if len(pre_det) == 0:
            return []
        keep = self._nms(pre_det)
        det = pre_det[keep]
        # 滑窗模式 kpss 为空（boxes-only），逐人返回 kps=None
        if len(kpss) > 0:
            kpss = kpss[keep]
        score_mask = det[:, 4] >= threshold
        det = det[score_mask]
        if len(kpss) > 0:
            kpss = kpss[score_mask]
        if len(det) == 0:
            return []
        det = det[:MAX_FACES_PER_IMAGE]
        if len(kpss) > 0:
            kpss = kpss[:MAX_FACES_PER_IMAGE]
        H, W = img.shape[:2]
        results = []
        for i, d in enumerate(det):
            x1, y1, x2, y2 = max(0, int(d[0])), max(0, int(d[1])), min(W, int(d[2])), min(H, int(d[3]))
            if min(x2 - x1, y2 - y1) >= min_face_size:
                kps = kpss[i].copy() if len(kpss) > 0 and kpss[i] is not None else None
                results.append(((float(x1), float(y1), float(x2), float(y2)), kps))
        return results
