"""
Configuration classes for merge parameters.
Transplanted from merger/MergerConfig.py, stripped of terminal interactivity.
"""
import copy
from facelib import FaceType

mode_dict = {
    0: 'original', 1: 'overlay', 2: 'hist-match', 3: 'seamless',
    4: 'seamless-hist-match', 5: 'raw-rgb', 6: 'raw-predict',
    7: 'debug'
}

mode_str_dict = {v: k for k, v in mode_dict.items()}

mask_mode_dict = {
    0: 'full', 1: 'dst', 2: 'learned-prd', 3: 'learned-dst',
    4: 'learned-prd*learned-dst', 5: 'learned-prd+learned-dst',
    6: 'XSeg-prd', 7: 'XSeg-dst', 8: 'XSeg-prd*XSeg-dst',
    9: 'learned-prd*learned-dst*XSeg-prd*XSeg-dst'
}

ctm_dict = {
    0: "None", 1: "rct", 2: "lct", 3: "mkl", 4: "mkl-m",
    5: "idt", 6: "idt-m", 7: "sot-m", 8: "mix-m",
    # 9/10 与训练预览侧对齐：mt = 遮罩加权均值/方差（RGB）；lut = 拟合式 3D LUT
    9: "mt", 10: "lut"
}

ctm_str_dict = {v: k for k, v in ctm_dict.items()}


class MergerConfig:
    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0, sharpen_mode=0, blursharpen_amount=0):
        self.type = type
        self.sharpen_mode = sharpen_mode
        self.blursharpen_amount = blursharpen_amount

    def copy(self):
        return copy.copy(self)

    def get_config(self):
        d = self.__dict__.copy()
        d.pop('type', None)
        return d

    def __eq__(self, other):
        if isinstance(other, MergerConfig):
            return (self.sharpen_mode == other.sharpen_mode and
                    self.blursharpen_amount == other.blursharpen_amount)
        return False


class MergerConfigMasked(MergerConfig):
    def __init__(self, face_type=FaceType.FULL, default_mode='overlay',
                 mode='overlay', masked_hist_match=True,
                 hist_match_threshold=238, mask_mode=4, seg_mode='model',
                 erode_mask_modifier=0, blur_mask_modifier=0,
                 motion_blur_power=0, output_face_scale=0,
                 super_resolution_power=0, color_transfer_mode=1,
                 image_denoise_power=0, bicubic_degrade_power=0,
                 color_degrade_power=0, show_debug=False, **kwargs):
        # P1-2b: swallow unknown keys (UI adds fields independently of the two
        # whitelists) into .extra instead of raising TypeError.
        self.extra = {k: v for k, v in kwargs.items()
                      if k not in ('type', 'sharpen_mode', 'blursharpen_amount')}
        kwargs = {k: v for k, v in kwargs.items() if k not in self.extra}
        super().__init__(type=MergerConfig.TYPE_MASKED, **kwargs)
        self.face_type = face_type
        self.default_mode = default_mode
        self.mode = mode if mode in mode_str_dict else mode_dict[1]
        self.show_debug = show_debug
        # P1-2: respect the caller; default stays True (matches 3d68 original).
        self.masked_hist_match = masked_hist_match
        self.hist_match_threshold = hist_match_threshold
        self.mask_mode = mask_mode
        self.seg_mode = seg_mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        self.motion_blur_power = motion_blur_power
        self.output_face_scale = output_face_scale
        self.super_resolution_power = super_resolution_power
        self.color_transfer_mode = color_transfer_mode
        self.image_denoise_power = image_denoise_power
        self.bicubic_degrade_power = bicubic_degrade_power
        self.color_degrade_power = color_degrade_power

    def copy(self):
        return copy.copy(self)

    def set_mode(self, mode):
        self.mode = mode_dict.get(mode, self.default_mode)

    def add_hist_match_threshold(self, diff):
        if self.mode in ('hist-match', 'seamless-hist-match'):
            self.hist_match_threshold = max(0, min(255, self.hist_match_threshold + diff))

    def __eq__(self, other):
        if isinstance(other, MergerConfigMasked):
            return (super().__eq__(other) and
                    self.mode == other.mode and
                    self.mask_mode == other.mask_mode and
                    self.erode_mask_modifier == other.erode_mask_modifier and
                    self.blur_mask_modifier == other.blur_mask_modifier and
                    self.motion_blur_power == other.motion_blur_power and
                    self.output_face_scale == other.output_face_scale and
                    self.color_transfer_mode == other.color_transfer_mode and
                    self.super_resolution_power == other.super_resolution_power and
                    self.image_denoise_power == other.image_denoise_power and
                    self.bicubic_degrade_power == other.bicubic_degrade_power and
                    self.color_degrade_power == other.color_degrade_power)
        return False


class MergerConfigFaceAvatar(MergerConfig):
    def __init__(self, temporal_face_count=0, add_source_image=False):
        super().__init__(type=MergerConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count
        self.add_source_image = add_source_image

    def copy(self):
        return copy.copy(self)

    def __eq__(self, other):
        if isinstance(other, MergerConfigFaceAvatar):
            return super().__eq__(other) and self.add_source_image == other.add_source_image
        return False
