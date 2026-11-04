from pathlib import Path
from modelhub import weights
from typing import List, Tuple
import cv2
import numpy as np
from xlib import math as lib_math
from xlib.onnxruntime import (InferenceSession_with_device, ORTDeviceInfo,
                              get_available_devices_info)
from ..det_preprocess import preprocess, map_back, sliding_window_detect


class RetinaFace:
    MODEL_SIZES = {
        'det_10g':  {'path': 'det_10g.onnx', 'scale': 1.0},
        'det_500m': {'path': 'det_500m.onnx', 'scale': 0.5},
    }

    @staticmethod
    def get_available_devices() -> List[ORTDeviceInfo]:
        return get_available_devices_info()

    def __init__(self, device_info: ORTDeviceInfo, model_name: str = 'det_10g'):
        if device_info not in RetinaFace.get_available_devices():
            raise Exception(f'device_info {device_info} not available for RetinaFace')
        if model_name not in self.MODEL_SIZES:
            raise Exception(f'Unknown model: {model_name}')
        path = weights.path('onnx/RetinaFace/' + self.MODEL_SIZES[model_name]['path'])
        self._sess = InferenceSession_with_device(str(path), device_info)
        self._input_name = self._sess.get_inputs()[0].name
        self.model_name = model_name
        # ── TRT BF16 加速 ──────────────────────
        _trt_path = None
        try:
            from xlib.trt import find_trt_engine
            _trt_path = find_trt_engine(str(path), self.MODEL_SIZES[model_name]['path'].replace('.onnx', ''))
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
        # InsightFace params
        self.input_mean = 127.5
        self.input_std = 128.0
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.fmc = 3  # number of FPN levels = output_groups / 3

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
        """Run model on a single image (letterboxed to input_size)."""
        input_size = tuple(img.shape[0:2][::-1])  # (W, H)
        # Preprocessing: (val - 127.5) / 128, BGR->RGB
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                      (self.input_mean,) * 3, swapRB=True)

        net_outs = self._sess.run(None, {self._input_name: blob})
        input_height, input_width = blob.shape[2:]

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width

            # Generate anchor centers (grid * stride, no +0.5)
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

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
        order = scores.argsort()[::-1]
        bboxes = np.concatenate(bboxes_list, axis=0)
        pre_det = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)
        return pre_det[order, :]

    def _detect_candidates(self, img, input_size=640, input_mode='one_stage', resize_mode='letterbox'):
        """按 input_mode 检测：one_stage 整图缩放 / sliding_window 滑窗扫描。"""
        # TRT engine 固定 640×640 输出，强制 640 尺寸
        if self._use_trt:
            input_size = 640

        if input_mode == 'sliding_window':
            boxes = sliding_window_detect(img, int(input_size), resize_mode,
                                          self._detect_window, pad_value=0)
            return boxes, np.empty((0, 5, 2))

        # one_stage
        det_img, meta = preprocess(img, resize_mode, int(input_size), pad_value=0)
        scores_list, bboxes_list, kpss_list = self._forward(det_img)
        if not scores_list or sum(s.size for s in scores_list) == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))
        scores = np.concatenate(scores_list)
        order = scores.argsort()[::-1]
        bboxes = map_back(np.concatenate(bboxes_list, axis=0), meta)
        kpss = map_back(np.concatenate(kpss_list, axis=0), meta)
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

    def extract(self, img, threshold=0.5, fixed_window=0, min_face_size=20,
                input_mode='one_stage', resize_mode='letterbox', input_size=640):
        """Detect faces, return list of [l,t,r,b]."""
        if threshold != 0.5:
            self.det_thresh = threshold
        pre_det, kpss = self._detect_candidates(img, input_size, input_mode, resize_mode)
        if len(pre_det) == 0:
            return [[]]
        keep = self._nms(pre_det)
        det = pre_det[keep]
        # Filter by min face size
        faces = []
        H, W = img.shape[:2]
        for d in det:
            x1, y1, x2, y2 = max(0, int(d[0])), max(0, int(d[1])), min(W, int(d[2])), min(H, int(d[3]))
            if min(x2 - x1, y2 - y1) >= min_face_size:
                faces.append((float(x1), float(y1), float(x2), float(y2)))
        return [faces]

    def extract_with_landmarks(self, img, threshold=0.5, fixed_window=0, min_face_size=20,
                               input_mode='one_stage', resize_mode='letterbox', input_size=640):
        """Detect faces with 5-point landmarks."""
        if threshold != 0.5:
            self.det_thresh = threshold
        pre_det, kpss = self._detect_candidates(img, input_size, input_mode, resize_mode)
        if len(pre_det) == 0:
            return [[]]
        keep = self._nms(pre_det)
        det = pre_det[keep]
        kpss = kpss[keep]
        faces = []
        H, W = img.shape[:2]
        for i in range(len(det)):
            x1, y1, x2, y2 = det[i, :4].astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if min(x2 - x1, y2 - y1) >= min_face_size:
                lm = [(float(kpss[i][j][0]), float(kpss[i][j][1])) for j in range(5)]
                faces.append((float(x1), float(y1), float(x2), float(y2), lm))
        return [faces]

    def extract_with_kps(self, img, threshold=0.5, fixed_window=0, min_face_size=20,
                         input_mode='one_stage', resize_mode='letterbox', input_size=640):
        """Detect faces returning both boxes and 5-point keypoints.
        Returns: [(box, kps_or_none), ...] where box=[l,t,r,b], kps=[[x,y],...x5]
        Keypoints order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        raw = self.extract_with_landmarks(img, threshold, fixed_window, min_face_size,
                                          input_mode, resize_mode, input_size)
        if not raw or not raw[0]:
            return []
        results = []
        for item in raw[0]:
            x1, y1, x2, y2, lm = item
            kps = np.array([[p[0], p[1]] for p in lm[:5]], dtype=np.float32) if isinstance(lm, list) and len(lm) >= 5 else None
            results.append(((float(x1), float(y1), float(x2), float(y2)), kps))
        return results
