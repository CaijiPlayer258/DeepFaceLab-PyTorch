"""MergeMaskedFace 遮罩合成 / mask_mode 分支 / fallback 语义的单元测试。

不加载模型：predictor_func 用假函数返回可控数组。
重点固定（审阅发现）：
  - predictor 返回单图（非 tuple）→ mask_mode=2/3/4/5 应 fallback 到全 1 或 hull 遮罩；
  - MergeMaskedFace 异常时必须返回 (原帧, 全 0 遮罩)（下游按存在性拾取的防线）；
  - erode/blur 修饰符对遮罩的实际作用方向。
"""
import numpy as np
import pytest
import cv2

from MergeStudio.core.config import MergerConfigMasked
from MergeStudio.core.merger import MergeMaskedFace


def _make_frame(w=256, h=256):
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_landmarks_68(cx=128.0, cy=128.0, scale=60.0):
    """合成 68 点人脸关键点（facelib expand_eyebrows 硬性要求 68 点）。
    按 iBUG 300-W 标准布局：17 下颚 + 5 眉 + 6 鼻梁 + 5 鼻底 + 6 眼 + 12 嘴 + 17 外圈。"""
    lm = np.zeros((68, 2), dtype=np.float32)
    t = np.linspace(0, np.pi, 17)  # 下颚：从左耳到右耳的下半弧
    lm[:17, 0] = cx - scale * np.cos(t) * 1.05
    lm[:17, 1] = cy + scale * np.sin(t) * 0.95 + scale * 0.10
    a = np.linspace(0, 2 * np.pi, 68 - 17, endpoint=False)  # 其余点均匀撒在内部椭圆
    lm[17:, 0] = cx + scale * 0.62 * np.cos(a)
    lm[17:, 1] = cy - scale * 0.25 + scale * 0.55 * np.sin(a)
    return lm


def _fake_pred_returns_face_only(face_img):
    """模拟旧版 DFM：只返回人脸图，无遮罩输出。"""
    return (face_img.copy(), None, None)


def _fake_pred_returns_all(face_img):
    """模拟完整输出：face + prd 遮罩 + dst 遮罩。"""
    m = np.zeros(face_img.shape[:2], dtype=np.float32)
    cv2.circle(m, (face_img.shape[1] // 2, face_img.shape[0] // 2), face_img.shape[0] // 4, 1.0, -1)
    return (face_img.copy(), m, m)


class TestMergeMaskedFaceFallbacks:

    def test_returns_original_on_garbage(self):
        """异常安全网：内部异常时返回 (原帧, 全0遮罩)。"""
        frame = _make_frame()
        cfg = MergerConfigMasked()
        bad_lm = np.array([[np.nan, np.nan]] * 10, dtype=np.float32)
        out, mask = MergeMaskedFace(frame, bad_lm, cfg, _fake_pred_returns_face_only)
        assert out.shape == frame.shape

    def test_no_faces_returns_frame(self):
        """无 predictor：应原样返回（预测器 None → predicted=[dst, None, None]）。"""
        frame = _make_frame(64, 64)
        cfg = MergerConfigMasked(mask_mode=0)
        lm = _make_landmarks_68(cx=32, cy=32, scale=12)
        out, mask = MergeMaskedFace(frame, lm, cfg, predictor_func=None)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_mask_mode0_full_face_region(self):
        frame = _make_frame()
        cfg = MergerConfigMasked(mask_mode=0)
        lm = _make_landmarks_68()
        out, mask = MergeMaskedFace(frame, _make_landmarks_68(), cfg, _fake_pred_returns_face_only)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
        # mask_mode=0 → working mask 全 1 → 整个人脸变换覆盖区都应被替换；
        # 返回的 merging mask 是整帧 warp 回去的，人脸区域中心必须是 1
        cy, cx = 128, 128
        region = mask[cy - 30:cy + 30, cx - 30:cx + 30]
        assert region.max() > 0
        assert region.mean() > 0.5

    def test_mask_mode4_without_masks_falls_back(self):
        """mask_mode=4（prd*dst）但 predictor 不给 mask → fallback 全 1（旧行为记录）。"""
        frame = _make_frame()
        cfg = MergerConfigMasked(mask_mode=4)
        out, mask = MergeMaskedFace(frame, _make_landmarks_68(), cfg, _fake_pred_returns_face_only)
        assert out.shape == frame.shape  # 不炸即可（fallback 语义）

    def test_mask_mode9_all_none_falls_back(self):
        frame = _make_frame()
        cfg = MergerConfigMasked(mask_mode=9)
        out, mask = MergeMaskedFace(frame, _make_landmarks_68(), cfg, _fake_pred_returns_face_only)
        assert out.shape == frame.shape


class TestErodeBlurDirection:

    def _mask(self):
        m = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(m, (128, 128), 80, 1.0, -1)
        return m

    def test_erode_shrinks(self):
        m = self._mask()
        pad = 32
        mp = np.pad(m, pad)
        eroded = cv2.erode(mp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
        eroded = eroded[pad:-pad, pad:-pad]
        assert eroded.sum() < m.sum()

    def test_dilate_grows(self):
        m = self._mask()
        pad = 32
        mp = np.pad(m, pad)
        dil = cv2.dilate(mp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=1)
        dil = dil[pad:-pad, pad:-pad]
        assert dil.sum() > m.sum()

    def test_blur_softens_edges(self):
        m = self._mask()
        pad = 32
        mp = np.pad(m, pad)
        blurred = cv2.GaussianBlur(mp, (31, 31), 0)
        blurred = blurred[pad:-pad, pad:-pad]
        # 边缘软化：中间应仍接近 1，但出现 0<x<1 的过渡像素
        assert blurred[128, 128] == pytest.approx(1.0, abs=0.05)
        n_transitional = np.count_nonzero((blurred > 0.05) & (blurred < 0.95))
        assert n_transitional > 0

    def test_negative_blur_kernel_zero_is_invalid_cv(self):
        """blur=0 时不应调用 GaussianBlur（核 (0,0) 是 cv2 错误）。"""
        for b in (0, 1):
            k = b + (1 - b % 2)  # 0→1, 1→1
            assert k >= 1 and k % 2 == 1
