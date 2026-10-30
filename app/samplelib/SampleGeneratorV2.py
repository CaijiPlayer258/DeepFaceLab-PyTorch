"""
SampleGeneratorV2 - V2 样本生成器

从 SampleLoaderV4 获取批次，应用所有 SampleProcessor 数据增强，
输出与原始 pipeline 兼容的批次数据。

设计要点：
1. 输入：SampleLoaderV4.get_batch() 返回的批次（已解码图片，list of dicts）
2. 处理：完全复现 SampleProcessor.process() 的变换逻辑
3. 输出：与原始 SampleGeneratorFace 格式一致的 numpy 数组列表
4. 不包含偏航角均衡采样（由 SampleLoaderV4 完成）
"""

import cv2
import numpy as np

from core import imagelib
from facelib import LandmarksProcessor
from samplelib.SampleGeneratorBase import SampleGeneratorBase
from samplelib.SampleProcessor import SampleProcessor


class SampleGeneratorV2(SampleGeneratorBase):
    """
    V2 样本生成器，封装 SampleLoaderV4 并应用完整的样本处理管线。

    继承自 SampleGeneratorBase，可直接用于训练 pipeline。

    Args:
        loader: SampleLoaderV4 实例（或实现了 get_batch() 的对象）
        sample_process_options: SampleProcessor.Options 实例
        output_sample_types: list[dict]，与 SampleProcessor 格式一致
        resolution: int，输出分辨率
        debug: bool，是否启用调试模式（跳过 normalize_tanh）
        ct_loader: SampleLoaderV4 | None，用于色彩迁移的配对加载器。
                   当输出类型使用 ct_mode 时，生成器会自动从 ct_loader
                   获取配对批次用于色彩迁移。

    Note:
        色彩迁移（ct_mode）使用时通过 ct_loader 获取配对样本，
        不同于旧的 SampleGeneratorFace 的 random_ct_samples_path 方式。
    """

    def __init__(self, loader, sample_process_options, output_sample_types,
                 resolution=256, debug=False, ct_loader=None, xseg_augment=False):
        batch_size = getattr(loader, 'batch_size', 1)
        self.initialized = False
        super().__init__(debug, batch_size=batch_size)
        self.loader = loader
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.resolution = resolution
        self.debug = debug
        self.xseg_augment = xseg_augment
        self.ct_loader = ct_loader
        self._last_filenames = []

        if getattr(loader, 'total_file_count', 0) > 0:
            self.initialized = True

    # ------------------------------------------------------------------
    # SampleGeneratorBase 接口
    # ------------------------------------------------------------------

    def is_initialized(self):
        return self.initialized

    def close(self):
        if hasattr(self, 'loader') and self.loader is not None:
            try:
                self.loader.shutdown()
            except Exception:
                pass
        if hasattr(self, 'ct_loader') and self.ct_loader is not None:
            try:
                self.ct_loader.shutdown()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_last_filenames(self):
        """返回最近一批的样本文件名列表。"""
        return self._last_filenames

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def __next__(self):
        return self.generate_next()

    def generate_next(self, paired_batch=None):
        """
        获取下一个批次并处理。

        Args:
            paired_batch: list[dict] | None，配对的原始批次数据（覆盖 ct_loader）。
                          当提供此参数时，优先使用它而非 ct_loader。

        Returns:
            list[np.ndarray] | None:
                每个输出类型对应一个数组，形状为 (batch_size, H, W, C) 或
                (batch_size, C, H, W)（当 data_format='NCHW' 时）。
                若加载器无数据则返回 None。
        """
        if not self.is_initialized():
            return None

        batch = self.loader.get_batch()
        if not batch or len(batch) == 0:
            return None

        # 自动从 ct_loader 获取配对批次用于色彩迁移
        if paired_batch is None and self.ct_loader is not None:
            paired_batch = self.ct_loader.get_batch()

        return self._process_batch(batch, paired_batch=paired_batch)

    # ------------------------------------------------------------------
    # 批处理核心
    # ------------------------------------------------------------------

    def _process_batch(self, batch, paired_batch=None):
        """处理整个批次，返回每输出类型堆叠后的数组列表。

        Args:
            paired_batch: list[dict] | None，配对批次的原始数据。
                          用于色彩迁移参考。
        """
        self._last_filenames = [item.get('filename', '') for item in batch]

        SPST = SampleProcessor.SampleType
        SPFMT = SampleProcessor.FaceMaskType

        per_type = [[] for _ in self.output_sample_types]

        for idx, item in enumerate(batch):
            sample_rnd_seed = np.random.randint(0x80000000)

            # 预计算 warp_params（同一样本所有输出类型复用）
            rnd_state_shared = np.random.RandomState(sample_rnd_seed)
            warp_rnd_state_shared = np.random.RandomState(sample_rnd_seed)
            shared_warp_params = imagelib.gen_warp_params(
                self.resolution,
                self.sample_process_options.random_flip,
                rotation_range=self.sample_process_options.rotation_range,
                scale_range=self.sample_process_options.scale_range,
                tx_range=self.sample_process_options.tx_range,
                ty_range=self.sample_process_options.ty_range,
                rnd_state=rnd_state_shared,
                warp_rnd_state=warp_rnd_state_shared,
            )

            # 延迟预计算 hull mask（FULL_FACE 和 EYES_MOUTH 复用）
            shared_hull_mask = None

            # 获取配对图像用于色彩迁移
            paired_img = None
            if paired_batch is not None and idx < len(paired_batch):
                pb_item = paired_batch[idx]
                if isinstance(pb_item, dict):
                    paired_img = pb_item.get('image')

            for out_idx, opts in enumerate(self.output_sample_types):
                sample_type = opts.get('sample_type')

                if sample_type == SPST.FACE_IMAGE:
                    xseg_mask = item.get('xseg_mask')
                    out = self._process_face_image(item, opts, sample_rnd_seed,
                                                    shared_warp_params,
                                                    paired_img=paired_img,
                                                    xseg_mask=xseg_mask)
                elif sample_type == SPST.FACE_MASK:
                    f_mask_type = opts.get('face_mask_type', SPFMT.NONE)
                    if f_mask_type in (SPFMT.FULL_FACE, SPFMT.EYES_MOUTH) \
                            and shared_hull_mask is None:
                        shared_hull_mask = self._compute_hull_mask(item)
                    out = self._process_face_mask(item, opts, sample_rnd_seed,
                                                   shared_warp_params, shared_hull_mask)
                else:
                    continue

                per_type[out_idx].append(out)

        return [np.stack(arr, axis=0) for arr in per_type]

    # ------------------------------------------------------------------
    # FACE_IMAGE 处理
    # ------------------------------------------------------------------

    def _process_face_image(self, item, opts, sample_rnd_seed,
                             precomputed_warp_params=None, paired_img=None,
                             xseg_mask=None):
        """处理 FACE_IMAGE 类型：色彩迁移 → HSV → warp/flip → 通道转换。"""
        SPCT = SampleProcessor.ChannelType

        img_u8 = item['image']

        # 转为 float32 [0, 1]（融合转换与缩放，单次遍历）
        img = np.multiply(img_u8, 1.0/255.0, dtype=np.float32)

        # 随机参数
        rnd_shift = opts.get('rnd_seed_shift', 0)
        warp_rnd_shift = opts.get('warp_rnd_seed_shift', rnd_shift)
        rnd_state = np.random.RandomState(sample_rnd_seed + rnd_shift)

        # 使用预计算的 warp_params（所有输出类型共享）
        if precomputed_warp_params is not None:
            warp_params = precomputed_warp_params
        else:
            warp_rnd_state = np.random.RandomState(sample_rnd_seed + warp_rnd_shift)
            warp_params = imagelib.gen_warp_params(
                self.resolution,
                self.sample_process_options.random_flip,
                rotation_range=self.sample_process_options.rotation_range,
                scale_range=self.sample_process_options.scale_range,
                tx_range=self.sample_process_options.tx_range,
                ty_range=self.sample_process_options.ty_range,
                rnd_state=rnd_state,
                warp_rnd_state=warp_rnd_state,
            )

        # 色彩迁移 (ct_mode) —— 使用配对图像作为参考
        ct_mode = opts.get('ct_mode', None)
        if ct_mode is not None and paired_img is not None:
            # paired_img 是 uint8 格式，转 float32 [0,1]
            ct_ref = np.multiply(paired_img, 1.0/255.0, dtype=np.float32)
            ct_resized = cv2.resize(
                ct_ref, (self.resolution, self.resolution),
                interpolation=cv2.INTER_LINEAR,
            )
            img = imagelib.color_transfer(ct_mode, img, ct_resized)

        # 随机 HSV 偏移
        hsv_amount = opts.get('random_hsv_shift_amount', 0)
        if hsv_amount != 0:
            a = hsv_amount
            h_amount = max(1, int(360 * a * 0.5))
            img_h, img_s, img_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            img_h = (img_h + rnd_state.randint(-h_amount, h_amount + 1)) % 360
            img_s = np.clip(img_s + (rnd_state.random() - 0.5) * a, 0, 1)
            img_v = np.clip(img_v + (rnd_state.random() - 0.5) * a, 0, 1)
            img = np.clip(cv2.cvtColor(cv2.merge([img_h, img_s, img_v]),
                                        cv2.COLOR_HSV2BGR), 0, 1)

        # 随机偏色（定义重建目标的 C；输入项与目标项用同一 seed，结果一致）
        _cc = opts.get('random_color_cast', 0.0)
        if _cc:
            img = imagelib.apply_random_color_cast(img, _cc, seed=sample_rnd_seed)

        # 扭曲之前的随机增强（固定顺序：先噪点，再遮挡）；与 random_warp 相互独立
        img = imagelib.apply_pre_warp_augmentations(
            img,
            do_noise=opts.get('random_noise', False),
            do_occlusion=opts.get('random_occlusion', False),
            rnd_state=rnd_state,
        )

        # 弹性变形 / 随机变换 / 随机翻转
        border_replicate = opts.get('border_replicate', True)
        can_warp = opts.get('warp', False)
        can_transform = opts.get('transform', False)
        img = imagelib.warp_by_params(
            warp_params, img, can_warp, can_transform,
            can_flip=True, border_replicate=border_replicate,
            cv2_inter=cv2.INTER_LINEAR,
        )

        # ---- XSeg 独享数据增强 ----
        if self.xseg_augment:
            r = self.resolution
            # Face flare（mask 区域模糊）
            if xseg_mask is not None and np.random.randint(2) == 0:
                krn = np.random.randint(r // 4, r)
                krn = krn - krn % 2 + 1
                mask_resized = cv2.resize(xseg_mask, (r, r)) if xseg_mask.shape[:2] != (r, r) else xseg_mask
                if mask_resized.ndim == 3 and mask_resized.shape[2] > 1:
                    mask_resized = mask_resized[..., :1]
                img = img + cv2.GaussianBlur(img * mask_resized, (krn, krn), 0)
            # BG flare（背景区域模糊）
            if xseg_mask is not None and np.random.randint(2) == 0:
                krn = np.random.randint(r // 4, r)
                krn = krn - krn % 2 + 1
                mask_resized = cv2.resize(xseg_mask, (r, r)) if xseg_mask.shape[:2] != (r, r) else xseg_mask
                if mask_resized.ndim == 3 and mask_resized.shape[2] > 1:
                    mask_resized = mask_resized[..., :1]
                img = img + cv2.GaussianBlur(img * (1.0 - mask_resized), (krn, krn), 0)
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_hsv_shift(img)
            else:
                img = imagelib.apply_random_rgb_levels(img)
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_sharpen(img, 25, 5)
            else:
                img = imagelib.apply_random_motion_blur(img, 25, 5)
                img = imagelib.apply_random_gaussian_blur(img, 25, 5)
            if np.random.randint(2) == 0:
                img = imagelib.apply_random_nearest_resize(img, 25, 75)
            else:
                img = imagelib.apply_random_bilinear_resize(img, 25, 75)
            img = np.clip(img, 0, 1)
            img = imagelib.apply_random_jpeg_compress(img, 25)

        # 通道转换
        channel_type = opts.get('channel_type', SPCT.BGR)
        if channel_type == SPCT.BGR:
            out = img
        elif channel_type == SPCT.G:
            out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        elif channel_type == SPCT.GGG:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = np.repeat(gray[..., None], 3, -1)
        else:
            out = img

        # Nearest-resize 增强（压缩伪影模拟）
        nearest_to = opts.get('nearest_resize_to', None)
        if nearest_to is not None:
            out = cv2.resize(out, (nearest_to, nearest_to),
                             interpolation=cv2.INTER_NEAREST)

        # 归一化
        np.clip(out, 0, 1, out=out)
        if opts.get('normalize_tanh', False) and not self.debug:
            np.subtract(np.multiply(out, 2.0, out=out), 1.0, out=out)

        # 数据格式
        if opts.get('data_format', 'NHWC') == 'NCHW':
            out = np.transpose(out, (2, 0, 1))

        return out

    # ------------------------------------------------------------------
    # FACE_MASK 处理
    # ------------------------------------------------------------------

    def _process_face_mask(self, item, opts, sample_rnd_seed,
                            precomputed_warp_params=None,
                            precomputed_hull_mask=None):
        """处理 FACE_MASK 类型：生成掩码 → warp/flip → 归一化。

        优先使用 item 中 XSeg 遮罩（如存在），否则回退到特征点凸包。
        """
        SPFMT = SampleProcessor.FaceMaskType

        xseg_mask = item.get('xseg_mask')
        if xseg_mask is not None:
            # 确保缩放到 self.resolution
            oh, ow = xseg_mask.shape[:2]
            if oh != self.resolution or ow != self.resolution:
                xseg_mask = cv2.resize(xseg_mask, (self.resolution, self.resolution),
                                        interpolation=cv2.INTER_LINEAR)

        landmarks = item.get('landmarks')
        orig_shape = item.get('original_shape')
        oh, ow = orig_shape if orig_shape else (self.resolution, self.resolution)

        # 将 landmarks 缩放到分辨率空间
        lm_res = self._scale_landmarks(landmarks, oh, ow, self.resolution) if landmarks is not None else None

        shape_1c = (self.resolution, self.resolution, 1)
        face_mask_type = opts.get('face_mask_type', SPFMT.NONE)

        if face_mask_type == SPFMT.FULL_FACE:
            if xseg_mask is not None:
                mask = xseg_mask  # use XSeg directly
            elif precomputed_hull_mask is not None:
                mask = precomputed_hull_mask
            elif lm_res is not None:
                mask = LandmarksProcessor.get_image_hull_mask(shape_1c, lm_res)
            else:
                mask = np.zeros(shape_1c, dtype=np.float32)
        elif face_mask_type == SPFMT.EYES:
            if lm_res is not None:
                mask = LandmarksProcessor.get_image_eye_mask(shape_1c, lm_res)
            else:
                mask = np.zeros(shape_1c, dtype=np.float32)
        elif face_mask_type == SPFMT.EYES_MOUTH:
            base_mask = xseg_mask  # use XSeg as base if available
            if base_mask is None:
                if precomputed_hull_mask is not None:
                    base_mask = precomputed_hull_mask
                elif lm_res is not None:
                    base_mask = LandmarksProcessor.get_image_hull_mask(shape_1c, lm_res)
                else:
                    base_mask = np.zeros(shape_1c, dtype=np.float32)
            # 确保 3D (H, W, 1)，避免 em * base_mask 广播错误（2D × 3D → (H, W, H)）
            if base_mask.ndim == 2:
                base_mask = base_mask[..., None]
            if lm_res is not None:
                base_mask = base_mask.copy()
                if base_mask.max() != 0.0:
                    base_mask[base_mask != 0.0] = 1.0
                em = LandmarksProcessor.get_image_eye_mask(shape_1c, lm_res)
                em += LandmarksProcessor.get_image_mouth_mask(shape_1c, lm_res)
                em = np.clip(em, 0, 1)
                mask = em * base_mask
            else:
                mask = np.zeros(shape_1c, dtype=np.float32)
        else:
            mask = np.zeros(shape_1c, dtype=np.float32)

        # 随机参数（与同一样本的 FACE_IMAGE 用不同的 rnd_seed_shift 时保证独立性）
        rnd_shift = opts.get('rnd_seed_shift', 0)
        warp_rnd_shift = opts.get('warp_rnd_seed_shift', rnd_shift)

        # 使用预计算的 warp_params
        if precomputed_warp_params is not None:
            warp_params = precomputed_warp_params
        else:
            rnd_state = np.random.RandomState(sample_rnd_seed + rnd_shift)
            warp_rnd_state = np.random.RandomState(sample_rnd_seed + warp_rnd_shift)
            warp_params = imagelib.gen_warp_params(
                self.resolution,
                self.sample_process_options.random_flip,
                rotation_range=self.sample_process_options.rotation_range,
                scale_range=self.sample_process_options.scale_range,
                tx_range=self.sample_process_options.tx_range,
                ty_range=self.sample_process_options.ty_range,
                rnd_state=rnd_state,
                warp_rnd_state=warp_rnd_state,
            )

        # 变形 / 变换 / 翻转
        border_replicate = opts.get('border_replicate', False)
        can_warp = opts.get('warp', False)
        can_transform = opts.get('transform', False)
        mask = imagelib.warp_by_params(
            warp_params, mask, can_warp, can_transform,
            can_flip=True, border_replicate=border_replicate,
            cv2_inter=cv2.INTER_LINEAR,
        )

        # EYES_MOUTH 后归一化（warp 后插值衰减）
        if face_mask_type == SPFMT.EYES_MOUTH:
            div = mask.max()
            if div != 0.0:
                mask /= div

        if len(mask.shape) == 2:
            mask = mask[..., None]

        out = mask.astype(np.float32)

        if opts.get('data_format', 'NHWC') == 'NCHW':
            out = np.transpose(out, (2, 0, 1))

        return out

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_landmarks(landmarks, orig_h, orig_w, resolution):
        """将 landmarks 从原始图像坐标空间缩放到分辨率空间。"""
        if landmarks is None:
            return None
        scaled = landmarks.copy().astype(np.float32)
        scaled[:, 0] *= resolution / orig_w
        scaled[:, 1] *= resolution / orig_h
        return scaled

    def _compute_hull_mask(self, item):
        """预计算 hull mask（供 FULL_FACE 和 EYES_MOUTH 复用）。

        优先使用 XSeg 遮罩（如存在），否则计算特征点凸包。
        始终保证输出尺寸为 (self.resolution, self.resolution, 1)。
        """
        xseg_mask = item.get('xseg_mask')
        if xseg_mask is not None:
            # 确保缩放到 self.resolution
            oh, ow = xseg_mask.shape[:2]
            if oh != self.resolution or ow != self.resolution:
                xseg_mask = cv2.resize(xseg_mask, (self.resolution, self.resolution),
                                        interpolation=cv2.INTER_LINEAR)
            if xseg_mask.dtype != np.float32:
                xseg_mask = xseg_mask.astype(np.float32)
            if xseg_mask.ndim == 2:
                xseg_mask = xseg_mask[..., None]
            return xseg_mask

        landmarks = item.get('landmarks')
        if landmarks is None:
            return None
        orig_shape = item.get('original_shape')
        oh, ow = orig_shape if orig_shape else (self.resolution, self.resolution)
        lm_res = self._scale_landmarks(landmarks, oh, ow, self.resolution)
        if lm_res is None:
            return None
        return LandmarksProcessor.get_image_hull_mask(
            (self.resolution, self.resolution, 1), lm_res)
