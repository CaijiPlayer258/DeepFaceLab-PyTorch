"""insightface 2D106 → DFL 68 点。

为什么不用 FAN：2DFAN 在 CPU 上 112 ms/张、GPU 39 ms/张，太重；
这个 106 点模型 2.5 ms/张，而且 106→68 的误差比 FAN 还小。

106→68 怎么来的：**不是查表**，而是用 1200 张已有真实 68 点的对齐脸做岭回归，
拟合一个 213×136 的线性映射（106 个点 ×2 坐标 + 偏置 → 68 个点 ×2 坐标）。
留出集（未参与训练）实测平均误差 **2.55 px**（512 空间），中位 2.18 px。
相比之下 FAN 是 8.2 px、5 点+模板是 10.6 px。

两边的坐标系是一致的：输入是 DFL 512 对齐图（缩到 192 送模型），
输出按 (x/2+0.5)*512 还原回 512 空间。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_W = None


def _weights():
    global _W
    if _W is None:
        p = Path(__file__).parent / "_landmarks106_w.npy"
        _W = np.load(str(p)).astype(np.float32)
    return _W


class InsightFace106:
    """106 点关键点模型；对外只暴露 landmarks68()。"""

    def __init__(self, model_path, providers=None, workspace=None, engine_name=None):
        import onnxruntime as ort

        self.sess = None
        if workspace is not None:
            try:
                from ._engines import try_engine
                self.sess = try_engine(workspace, engine_name or "insightface_2d106", model_path)
            except Exception:
                self.sess = None
        if self.sess is None:
            if providers is None:
                try:
                    from ._ortcuda import providers as _prov
                    providers = _prov()
                except Exception:
                    providers = ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        self._batch_ok = None      # None=没试过；False=引擎是静态 batch=1，退回逐张

    def landmarks106(self, face_bgr):
        import cv2

        size = face_bgr.shape[1]
        small = cv2.resize(face_bgr, (192, 192), interpolation=cv2.INTER_LINEAR)
        # 与 4.0 的 InsightFace2D106.extract 一致：swap_ch（BGR→RGB）+ NCHW float32
        blob = np.ascontiguousarray(small[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None])
        out = self.sess.run(None, {self.input_name: blob})[0].reshape(106, 2)
        return (out / 2.0 + 0.5) * float(size)

    def landmarks68_batch(self, faces):
        """批量版：一次 session.run 处理多张脸（模型 batch 维是动态的）。

        单张 1.58ms，批 8 摊到 0.33ms/脸 —— 对齐流水线里按批攒着跑。
        """
        import cv2
        if not faces:
            return []
        if self._batch_ok is False:
            return [self.landmarks68(f) for f in faces]
        smalls = [cv2.resize(f, (192, 192), interpolation=cv2.INTER_LINEAR) for f in faces]
        blob = np.stack([s[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) for s in smalls])
        try:
            outs = self.sess.run(None, {self.input_name: blob})[0]  # (N,106,2)
            if outs.shape[0] != len(faces):
                # 静态 batch=1 的 TRT 引擎只会回 1 条，直接退回逐张
                self._batch_ok = False
                return [self.landmarks68(f) for f in faces]
            self._batch_ok = True
        except Exception:
            # TRT 引擎可能是静态 batch=1：记下来，之后都逐张跑
            self._batch_ok = False
            return [self.landmarks68(f) for f in faces]
        w = _weights()
        res = []
        for i, f in enumerate(faces):
            p = outs[i].reshape(-1) / 2.0 + 0.5
            p = p * float(f.shape[1])
            if p.shape[0] + 1 != w.shape[0]:
                res.append(None)
                continue
            res.append((np.append(p, 1.0).astype(np.float32) @ w).reshape(68, 2))
        return res

    def landmarks68(self, face_bgr):
        """返回 (68,2) float32，坐标系与输入图一致（512 对齐空间）。"""
        p = self.landmarks106(face_bgr).reshape(-1)
        w = _weights()
        if p.shape[0] + 1 != w.shape[0]:
            return None
        return (np.append(p, 1.0).astype(np.float32) @ w).reshape(68, 2)
