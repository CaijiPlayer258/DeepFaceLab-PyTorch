"""XSeg 遮罩推理（4.1 侧，纯 onnxruntime）。

对齐时顺手生成 xseg_mask 写进 DFLJPG —— 真 DFL 提取的每张脸都带 (256,256,1) 遮罩，
而 4.1 之前写出去的 DFLJPG 没有，训练端要么退化到用 landmarks 算 hull（而 4.1 的
68 点是从 5 点刚体投影出来的，形状是死的），要么根本没有遮罩。

预处理与 MergeStudio 的 _phase_mask_worker 完全一致：
  BGR → 256×256 → RGB → /255 → CHW → [1,3,256,256]；输出 0..1，按需缩放回原尺寸。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class XSegMasker:
    def __init__(self, model_path, providers=None, threads: int = 2, workspace=None,
                 engine_name: str | None = None):
        # 先看有没有编译好的 TensorRT 引擎（MD5 + SM 都对才算数）
        self.sess = None
        if workspace is not None:
            try:
                from ._engines import try_engine
                self.sess = try_engine(workspace, engine_name or Path(model_path).stem,
                                       model_path)
            except Exception:
                self.sess = None
        if self.sess is not None:
            self.input_name = self.sess.get_inputs()[0].name
            return
        import onnxruntime as ort

        if providers is None:
            from ._ortcuda import providers as _prov
            providers = _prov()          # 有 CUDA 走 CUDA，否则 CPU
        self.sess = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name
        try:
            self.sess.set_providers(list(providers), [{}] * len(providers))
        except Exception:
            pass

    def mask(self, face_bgr, out_size: int = 256):
        """ face_bgr: 任意尺寸的 BGR 对齐脸；返回 (out_size, out_size) float32，0..1。"""
        import cv2

        if face_bgr is None:
            return None
        x = face_bgr
        if x.shape[0] != 256 or x.shape[1] != 256:
            x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LINEAR)
        # 一次拷贝进 NCHW float32（BGR→RGB 顺手在索引里做），再原地 /255
        inp = np.empty((1, 3, 256, 256), np.float32)
        inp[0] = x[:, :, ::-1].transpose(2, 0, 1)
        inp *= (1.0 / 255.0)
        out = self.sess.run(None, {self.input_name: inp})[0]
        m = np.squeeze(out).astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
        if m.shape[0] != out_size or m.shape[1] != out_size:
            m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return m
