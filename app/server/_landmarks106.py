"""insightface 2D106 → DFL 68 点（**纯索引映射**）。

为什么用这个模型：2DFAN 在 CPU 上 112 ms/张、GPU 39 ms/张，太重；这个 106 点
模型 2.5 ms/张，够快。

106 → 68 怎么来的：**不是回归、不是拟合**，而是从 106 个点里**按下标挑**出 68 个，
顺序就是 iBUG-68（下巴 17 / 眉 10 / 鼻 9 / 眼 12 / 嘴 20）。映射表原样抄自引擎
`app/engine/Extractor/Extractor.py: landmark106to68`（用户确认这份是对的，**勿改**）。

为什么是挑点而不是回归：
  * 68 个点永远落在模型真正检测到的位置上，不会被线性映射平滑、拉偏；
  * 不用再维护一份拟合权重文件（以前那个 .npy 已经删掉，见 git 历史）；
  * 出错时能一眼看出是哪一段（下巴/眉/眼）歪了，而不是"整体偏 2px"。

坐标系：`landmarks106()` 返回 106×2、`landmarks68()` 返回 68×2，都是**输入图**
的像素坐标。输入约定仍是 DFL 512 对齐图（内部缩到 192 送模型），输出按
(x/2+0.5)*输入边长 还原。
"""
from __future__ import annotations

import numpy as np

# 来源：app/engine/Extractor/Extractor.py:1171 landmark106to68（索引映射，勿改）
# 分组：下巴 9 + 下巴 8 = 17；左眉 5；右眉 5；鼻 9；左眼 6；右眼 6；嘴 20 = 68
LM106_TO_68 = (
    1, 10, 12, 14, 16, 3, 5, 7, 0,                 # 下巴 9
    23, 21, 19, 32, 30, 28, 26, 17,                # 下巴 8（补齐 17 个轮廓点）
    43, 48, 49, 51, 50,                            # 左眉 5
    102, 103, 104, 105, 101,                       # 右眉 5
    72, 73, 74, 86, 78, 79, 80, 85, 84,            # 鼻 9
    35, 41, 42, 39, 37, 36,                        # 左眼 6
    89, 95, 96, 93, 91, 90,                        # 右眼 6
    52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54,  # 嘴 20
)


def landmark106to68(pt106) -> np.ndarray:
    """106 点 → 68 点：只按下标挑，不插值、不加权、不拟合。

    pt106 可以是任意坐标单位（模型原始输出 / 像素 / 归一化），函数只做下标选择；
    点数不是 106 时按老规矩截断（兼容别的关键点模型）。
    """
    arr = np.asarray(pt106, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] != 106:
        return arr[:68] if arr.shape[0] >= 68 else arr
    return arr[np.asarray(LM106_TO_68, dtype=np.int64)]


class InsightFace106:
    """106 点关键点模型；对外 landmarks106()（调试）与 landmarks68()/landmarks68_batch()。"""

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
        """原始 106 点（输入图像素坐标）。调试 / 校验用，不参与对齐。"""
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
        返回的 68 点是「106 点按索引挑出来」的，坐标系与输入图一致。
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
        res = []
        for i, f in enumerate(faces):
            p = outs[i].reshape(106, 2) / 2.0 + 0.5
            p = p * float(f.shape[1])
            res.append(landmark106to68(p))
        return res

    def landmarks68(self, face_bgr):
        """返回 (68,2) float64，坐标系与输入图一致（68 点 = 106 点的下标子集）。"""
        return landmark106to68(self.landmarks106(face_bgr))
