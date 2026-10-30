import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

class TanhPolar(nn.LayerBase):
    """
    RoI Tanh-极坐标变换网络，用于人脸解析
    https://github.com/hhj1897/roi_tanh_warping
    """

    def __init__(self, width, height, angular_offset_deg=270, name=None, **kwargs):
        self.width = width
        self.height = height

        super().__init__(name=name, **kwargs)

        warp_gridx, warp_gridy = TanhPolar._get_tanh_polar_warp_grids(width,height,angular_offset_deg=angular_offset_deg)
        restore_gridx, restore_gridy = TanhPolar._get_tanh_polar_restore_grids(width,height,angular_offset_deg=angular_offset_deg)

        # 将网格转换为PyTorch tensor
        self.register_buffer('warp_gridx_t', torch.from_numpy(warp_gridx[None, ...]))
        self.register_buffer('warp_gridy_t', torch.from_numpy(warp_gridy[None, ...]))
        self.register_buffer('restore_gridx_t', torch.from_numpy(restore_gridx[None, ...]))
        self.register_buffer('restore_gridy_t', torch.from_numpy(restore_gridy[None, ...]))

    def warp(self, inp_t):
        batch = inp_t.shape[0]
        warp_gridx_t = self.warp_gridx_t.repeat(batch, 1, 1)
        warp_gridy_t = self.warp_gridy_t.repeat(batch, 1, 1)

        # PyTorch grid_sample需要grid格式为(N, H, W, 2)，值域为[-1, 1]
        # 归一化坐标到[-1, 1]范围
        grid_x = 2.0 * warp_gridx_t / (self.width - 1) - 1.0
        grid_y = 2.0 * warp_gridy_t / (self.height - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        out_t = F.grid_sample(inp_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return out_t

    def restore(self, inp_t):
        batch = inp_t.shape[0]
        restore_gridx_t = self.restore_gridx_t.repeat(batch, 1, 1)
        restore_gridy_t = self.restore_gridy_t.repeat(batch, 1, 1)

        # 对称填充
        inp_t = F.pad(inp_t, (0, 0, 1, 1), mode='reflect')

        # 归一化坐标
        grid_x = 2.0 * restore_gridx_t / (self.width + 1 - 1) - 1.0
        grid_y = 2.0 * restore_gridy_t / (self.height - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        out_t = F.grid_sample(inp_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return out_t

    @staticmethod
    def _get_tanh_polar_warp_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W),np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1)
        radii = normalised_dest_indices[..., 0]
        orientation_x = np.cos(normalised_dest_indices[..., 1])
        orientation_y = np.sin(normalised_dest_indices[..., 1])

        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)

        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float)
        normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                                [sin_offset, cos_offset]]))
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                        roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]

        src_radii = np.tanh(radii)


        src_x_indices = src_radii * W + 1.0
        src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                                2.0 / np.pi) * H, H) + 1.0

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)


nn.TanhPolar = TanhPolar