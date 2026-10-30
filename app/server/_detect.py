"""RetinaFace 检测器（精简版：整图检测 + 5 点关键点，不含 TRT / 滑窗）。

从 4.0 的 modelhub/onnx/RetinaFace/RetinaFace.py 摘出核心解码逻辑，
去掉对 xlib / det_preprocess 的依赖，只保留 onnxruntime + numpy + cv2。

关键点是必须的：DFL 的对齐靠 5 点相似变换，只有框的话裁出来的脸对不上
DFL 的对齐空间（眼睛/嘴巴位置会飘），所以 _forward 把 kps 分支一并解出来。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class RetinaFaceDet:
    needs_canvas = False      # detect(img) 内部自己做 letterbox
    """det_10g / det_500m 通用。输入 BGR ndarray，输出人头框列表。"""

    STEPS = (8, 16, 32)

    def __init__(self, model_path, input_size=640, thresh=0.5, nms_thresh=0.4,
                 workspace=None, engine_name=None):
        import onnxruntime as ort
        try:
            from ._ortcuda import providers as _prov
            _providers = _prov()          # 装了 onnxruntime-gpu 就用 CUDA
        except Exception:
            _providers = ["CPUExecutionProvider"]
        # 优先用编译好的 TRT 引擎（引擎名按 onnx 的 md5 + SM 命名）
        self.sess = None
        if workspace is not None:
            try:
                from ._engines import try_engine
                self.sess = try_engine(workspace, engine_name or Path(model_path).stem, model_path)
            except Exception:
                self.sess = None
        if self.sess is None:
            self.sess = ort.InferenceSession(str(model_path), providers=_providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(input_size)
        self.thresh = float(thresh)
        self.nms_thresh = float(nms_thresh)
        self.fmc = 3
        self.num_anchors = 2

    # ---------- 解码 ----------
    @staticmethod
    def _distance2bbox(points, distance):
        return np.stack([
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ], axis=-1)

    def _forward(self, img):
        """img: 已经 letterbox 到 (S,S) 的 BGR 图。"""
        S = self.input_size
        # 一次拷贝：直接写进 float32 的 NCHW 缓冲，再原地减均值/缩放。
        # 原来的写法 (astype -> -127.5 -> /128 -> 通道翻转 -> transpose+ascontiguousarray)
        # 一共 5 遍全图遍历，实测 3.48 ms；这里 1.05 ms，结果逐位一致。
        blob = np.empty((1, 3, S, S), np.float32)
        blob[0] = img[:, :, ::-1].transpose(2, 0, 1)
        blob -= 127.5
        blob *= (1.0 / 128.0)

        outs = self.sess.run(None, {self.input_name: blob})

        scores_list, bboxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self.STEPS):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + self.fmc].reshape(-1, 4) * stride
            kps_preds = outs[idx + 2 * self.fmc].reshape(-1, 10) * stride

            h = S // stride
            w = S // stride
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)

            bboxes = self._distance2bbox(centers, bbox_preds)
            kps = self._distance2kps(centers, kps_preds)
            keep = np.where(scores >= self.thresh)[0]
            scores_list.append(scores[keep])
            bboxes_list.append(bboxes[keep])
            kps_list.append(kps[keep])

        if not scores_list or sum(s.size for s in scores_list) == 0:
            return np.empty((0,)), np.empty((0, 4)), np.empty((0, 10))
        return (np.concatenate(scores_list),
                np.concatenate(bboxes_list, axis=0),
                np.concatenate(kps_list, axis=0))

    @staticmethod
    def _distance2kps(points, distance):
        """5 个关键点（10 个数）从中心点 + 偏移还原。"""
        out = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            out.append(np.stack([px, py], axis=-1))
        return np.stack(out, axis=1)      # (N, 5, 2)

    @staticmethod
    def _nms(dets, thresh):
        if dets.shape[0] == 0:
            return []
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = dets[:, 4].argsort()[::-1]
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
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep

    # ---------- 对外 ----------
    def detect(self, img_bgr, min_face=16):
        """返回 [{'box': [x1,y1,x2,y2], 'score': float}, ...]（原图坐标）。"""
        import cv2
        H, W = img_bgr.shape[:2]
        S = self.input_size
        scale = min(S / float(W), S / float(H))
        nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        canvas = np.zeros((S, S, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        fwd = self._forward(canvas)
        if len(fwd) == 3:
            scores, boxes, kpss = fwd
        else:
            scores, boxes = fwd
            kpss = np.zeros((scores.shape[0], 5, 2), np.float32)
        if scores.size == 0:
            return []

        order = scores.argsort()[::-1]
        dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
        kpss = kpss[order]
        keep = self._nms(dets, self.nms_thresh)

        out = []
        for i in keep:
            d = dets[i]
            x1 = max(0.0, d[0] / scale)
            y1 = max(0.0, d[1] / scale)
            x2 = min(float(W), d[2] / scale)
            y2 = min(float(H), d[3] / scale)
            if min(x2 - x1, y2 - y1) >= min_face:
                kp = kpss[i] / scale
                out.append({"box": [round(float(x1), 1), round(float(y1), 1),
                                    round(float(x2), 1), round(float(y2), 1)],
                            "kps": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp],
                            "score": round(float(d[4]), 4)})
        out.sort(key=lambda x: -x["score"])
        return out


class TinyMogDet:
    needs_canvas = True       # detect_batch 吃的是预先 letterbox 好的 640 canvas
    """TinyMog（SCRFD + MobileNetV1）—— 快检测器。

    与 RetinaFaceDet 同一套对外接口（detect(img) -> [{box, kps, score}]），
    区别是模型只有 2.7 MB、输入 640、batch 维是**动态**的（可以批量推理）。

    预处理与 4.0 的 modelhub/onnx/TinyMog 一致：letterbox 到 640（padding=128）、
    blobFromImage(1/128, mean=127.5, swapRB=True)。输出 9 个张量：每个 stride
    一组 (cls, bbox, kps)，anchor 顺序与 RetinaFace 相同（2 anchors，8/16/32）。
    """

    STEPS = (8, 16, 32)
    INPUT_SIZE = 640

    def __init__(self, model_path, input_size=640, thresh=0.5, nms_thresh=0.4,
                 providers=None, workspace=None, engine_name=None):
        import onnxruntime as ort

        if providers is None:
            try:
                from ._ortcuda import providers as _prov
                providers = _prov()
            except Exception:
                providers = ["CPUExecutionProvider"]
        # 优先用编译好的 TRT 引擎（引擎名按 onnx 的 md5 + SM 命名）
        self.sess = None
        if workspace is not None:
            try:
                from ._engines import try_engine
                self.sess = try_engine(workspace, engine_name or Path(model_path).stem, model_path)
            except Exception:
                self.sess = None
        if self.sess is None:
            self.sess = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(input_size)
        self.thresh = float(thresh)
        self.nms_thresh = float(nms_thresh)
        self._centers_cache = {}

    # ---- 解码 ----
    @staticmethod
    def _centers(stride, S, num_anchors=2):
        h = w = S // stride
        c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        c = (c * stride).reshape(-1, 2)
        if num_anchors > 1:
            c = np.stack([c] * num_anchors, axis=1).reshape(-1, 2)
        return c

    def _forward(self, canvas):
        S = self.input_size
        outs = None
        # 优先走 GPU 预处理：640² 的 BGR→RGB + CHW + 归一化在 numpy 里要 1ms 以上
        if getattr(self.sess, "run_tensor", None) is not None:
            try:
                import torch
                t = torch.from_numpy(np.ascontiguousarray(canvas)).cuda()
                t = t.permute(2, 0, 1).flip(0).float()
                t = (t - 127.5) * (1.0 / 128.0)
                outs = self.sess.run_tensor({self.input_name: t})
            except Exception:
                outs = None
        if outs is None:
            blob = np.empty((1, 3, S, S), np.float32)
            blob[0] = canvas[:, :, ::-1].transpose(2, 0, 1)
            blob -= 127.5
            blob *= (1.0 / 128.0)
            outs = self.sess.run(None, {self.input_name: blob})

        scores_list, boxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self.STEPS):
            scores = outs[idx * 3].reshape(-1)
            key = stride
            if key not in self._centers_cache:
                self._centers_cache[key] = self._centers(stride, S)
            centers = self._centers_cache[key]

            # 只在命中阈值的索引上取值：不再整表 stack / where 两遍
            keep = np.flatnonzero(scores >= self.thresh)
            if keep.size == 0:
                continue
            n = min(len(scores), len(centers))
            keep = keep[keep < n]
            if keep.size == 0:
                continue
            bboxes = outs[idx * 3 + 1].reshape(-1, 4)[keep] * stride
            kps = outs[idx * 3 + 2].reshape(-1, 10)[keep] * stride
            c = centers[keep]

            b = np.empty((keep.size, 4), np.float32)
            b[:, 0] = c[:, 0] - bboxes[:, 0]
            b[:, 1] = c[:, 1] - bboxes[:, 1]
            b[:, 2] = c[:, 0] + bboxes[:, 2]
            b[:, 3] = c[:, 1] + bboxes[:, 3]

            k = kps.reshape(-1, 5, 2)
            k[:, :, 0] += c[:, 0:1]
            k[:, :, 1] += c[:, 1:2]

            scores_list.append(scores[keep])
            boxes_list.append(b)
            kps_list.append(k)
        if not scores_list:
            return np.empty((0,)), np.empty((0, 4)), np.empty((0, 5, 2))
        return (np.concatenate(scores_list),
                np.concatenate(boxes_list, axis=0),
                np.concatenate(kps_list, axis=0))

    _nms = staticmethod(RetinaFaceDet._nms)

    # ---------- 批量 ----------
    def forward_batch(self, canvases):
        """canvases: list of (640,640,3) BGR uint8，返回每张的 (scores, boxes, kps)。"""
        n = len(canvases)
        S = self.input_size
        blob = np.empty((n, 3, S, S), np.float32)
        for i, c in enumerate(canvases):
            blob[i] = c[:, :, ::-1].transpose(2, 0, 1)
        blob -= 127.5
        blob *= (1.0 / 128.0)
        outs = self.sess.run(None, {self.input_name: blob})

        results = []
        for b in range(n):
            scores_list, boxes_list, kps_list = [], [], []
            for idx, stride in enumerate(self.STEPS):
                scores = outs[idx * 3][b].reshape(-1)
                bboxes = outs[idx * 3 + 1][b].reshape(-1, 4) * stride
                kps = outs[idx * 3 + 2][b].reshape(-1, 10) * stride
                if stride not in self._centers_cache:
                    self._centers_cache[stride] = self._centers(stride, S)
                centers = self._centers_cache[stride]
                m = min(len(scores), len(centers))
                scores, bboxes, kps, centers = scores[:m], bboxes[:m], kps[:m], centers[:m]
                keep = np.where(scores >= self.thresh)[0]
                if keep.size == 0:
                    continue
                c = centers[keep]
                bb = bboxes[keep]
                kk = kps[keep].reshape(-1, 5, 2).copy()
                kk[:, :, 0] += c[:, 0][:, None]
                kk[:, :, 1] += c[:, 1][:, None]
                scores_list.append(scores[keep])
                boxes_list.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1],
                                            c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], axis=-1))
                kps_list.append(kk)
            if not scores_list:
                results.append((np.empty(0), np.empty((0, 4)), np.empty((0, 5, 2))))
            else:
                results.append((np.concatenate(scores_list),
                                np.concatenate(boxes_list, 0),
                                np.concatenate(kps_list, 0)))
        return results

    def detect_batch(self, items, min_face=16):
        """items: [(img_bgr, canvas, scale)] → 每张的 [{box,kps,score}]（原图坐标）。"""
        canvases = [it[1] for it in items]
        res = self.forward_batch(canvases)
        out = []
        for (img, canvas, scale), (scores, boxes, kpss) in zip(items, res):
            H, W = img.shape[:2]
            if scores.size == 0:
                out.append([])
                continue
            order = scores.argsort()[::-1]
            dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
            kpss = kpss[order]
            keep = self._nms(dets, self.nms_thresh)
            faces = []
            for i in keep:
                d = dets[i]
                x1 = max(0.0, d[0] / scale); y1 = max(0.0, d[1] / scale)
                x2 = min(float(W), d[2] / scale); y2 = min(float(H), d[3] / scale)
                if min(x2 - x1, y2 - y1) < min_face:
                    continue
                kp = kpss[i] / scale
                faces.append({"box": [round(float(x1), 1), round(float(y1), 1),
                                      round(float(x2), 1), round(float(y2), 1)],
                              "kps": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp],
                              "score": round(float(d[4]), 4)})
            faces.sort(key=lambda x: -x["score"])
            out.append(faces)
        return out

    def detect(self, img_bgr, min_face=16):
        import cv2
        H, W = img_bgr.shape[:2]
        S = self.input_size
        scale = min(S / float(W), S / float(H))
        nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        canvas = np.full((S, S, 3), 128, np.uint8)          # letterbox 用 128 填充
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

        scores, boxes, kpss = self._forward(canvas)
        if scores.size == 0:
            return []
        order = scores.argsort()[::-1]
        dets = np.hstack([boxes[order], scores[order, None]]).astype(np.float32)
        kpss = kpss[order]
        keep = self._nms(dets, self.nms_thresh)

        out = []
        for i in keep:
            d = dets[i]
            x1 = max(0.0, d[0] / scale); y1 = max(0.0, d[1] / scale)
            x2 = min(float(W), d[2] / scale); y2 = min(float(H), d[3] / scale)
            if min(x2 - x1, y2 - y1) < min_face:
                continue
            kp = kpss[i] / scale
            out.append({"box": [round(float(x1), 1), round(float(y1), 1),
                                round(float(x2), 1), round(float(y2), 1)],
                        "kps": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in kp],
                        "score": round(float(d[4]), 4)})
        out.sort(key=lambda x: -x["score"])
        return out
