"""LIAELarge Model — LIAE variant of DeepFakeLarge.

Architecture: Encoder → Bottleneck_AB + Bottleneck_B → single Decoder.
Same convolutional backbone as DeepFakeLarge but with dual bottleneck
modules and a shared decoder, enabling better identity separation.
"""

import copy
import math
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import SampleGeneratorV2, SampleLoaderV4, SampleProcessor

# =============================================================================
# Network building blocks (shared with DeepFakeLarge)
# =============================================================================


class ConvAct(torch.nn.Module):
    """Conv2d + InstanceNorm + LeakyReLU(0.2)."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.c = torch.nn.Conv2d(in_ch, out_ch, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.n = torch.nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x):
        return F.leaky_relu(self.n(self.c(x)), 0.2)


class Encoder(torch.nn.Module):
    """Simple encoder: 5 stages of ConvAct + avg_pool, no Inception.

    Like SAEHD's simple strided-conv approach — just stacked ConvAct blocks
    with 2× downsampling between stages. No multi-scale input branches.
    Output: e_dims * 8 channels at resolution/32 spatial size.
    """

    def __init__(self, resolution, e_dims):
        super().__init__()
        self.resolution = resolution
        ed = e_dims

        # 5 simple stages: ConvAct(k=3) → avg_pool(2)
        self.d1 = ConvAct(3, ed, 3)
        self.d2 = ConvAct(ed, ed * 2, 3)
        self.d3 = ConvAct(ed * 2, ed * 4, 3)
        self.d4 = ConvAct(ed * 4, ed * 8, 3)
        self.d5 = ConvAct(ed * 8, ed * 8, 3)

    def forward(self, x):
        shapes = []
        x = self.d1(x);  shapes.append(('d1', x.shape));  x = F.avg_pool2d(x, 2)
        x = self.d2(x);  shapes.append(('d2', x.shape));  x = F.avg_pool2d(x, 2)
        x = self.d3(x);  shapes.append(('d3', x.shape));  x = F.avg_pool2d(x, 2)
        x = self.d4(x);  shapes.append(('d4', x.shape));  x = F.avg_pool2d(x, 2)
        x = self.d5(x);  shapes.append(('d5', x.shape));  x = F.avg_pool2d(x, 2)
        shapes.append(('enc_out', x.shape))
        return x, shapes


class InterBottleneck(torch.nn.Module):
    """LIAE intermediate bottleneck: Flatten → Linear → LayerNorm → Linear → Reshape.

    Unlike DeepFakeLarge's single Bottleneck, LIAE uses two inter modules
    (inter_AB and inter_B) with configurable output channels.
    """

    def __init__(self, enc_ch, resolution, ae_dims, out_ch):
        super().__init__()
        self.out_ch = out_ch
        self.spatial = resolution // 32
        in_dim = enc_ch * self.spatial * self.spatial
        out_dim = out_ch * self.spatial * self.spatial
        self.fc1 = torch.nn.Linear(in_dim, ae_dims)
        self.n1 = torch.nn.LayerNorm(ae_dims)
        self.fc2 = torch.nn.Linear(ae_dims, out_dim)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, -1)
        z = self.n1(self.fc1(x))
        x = self.fc2(z).reshape(b, self.out_ch, self.spatial, self.spatial)
        return x, z


class ResidualBlock(torch.nn.Module):
    """1x1 → 3x3 → 3x3 → 1x1 bottleneck residual."""

    def __init__(self, c, mid):
        super().__init__()
        self.c0 = ConvAct(c, mid, 1)
        self.c1 = ConvAct(mid, mid, 3)
        self.c2 = ConvAct(mid, mid, 3)
        self.c3 = ConvAct(mid, c, 1)

    def forward(self, x):
        return x + self.c3(self.c2(self.c1(self.c0(x))))


class BasicResidualBlock(torch.nn.Module):
    """3x3 → 3x3 simple residual."""

    def __init__(self, c):
        super().__init__()
        self.c1 = ConvAct(c, c, 3)
        self.c2 = ConvAct(c, c, 3)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class Decoder(torch.nn.Module):
    """Decoder with 4 upsampling stages and ensemble output heads.

    Args:
        d_dims: Base channel count.
        width:  Channel scaling factor.
        in_ch:  Input channel count from the bottleneck(s).
    """

    def __init__(self, d_dims=64, width=1.0, in_ch=None):
        super().__init__()
        scale = width
        c0 = max(16, int(d_dims * 8 * scale))
        c1 = max(16, int(d_dims * 8 * scale))
        c2 = max(16, int(d_dims * 4 * scale))
        c3 = max(16, int(d_dims * 2 * scale))
        rmid = max(16, int(d_dims * 4 * scale))

        self.proj = ConvAct(in_ch, c0, 1)

        self.u0_c1 = ConvAct(c0, c0, 1)
        self.u0_c2 = ConvAct(c0, c0, 3)
        self.u0_c3 = ConvAct(c0, c0, 3)
        self.u0_c4 = ConvAct(c0, c0, 3)

        self.u1_c1 = ConvAct(c0, c1, 1)
        self.u1_c2 = ConvAct(c1, c1, 3)
        self.u1_c3 = ConvAct(c1, c1, 3)
        self.u1_c4 = ConvAct(c1, c1, 3)
        self.u1_res0 = ResidualBlock(c1, rmid)
        self.u1_res1 = ResidualBlock(c1, rmid)

        self.u2_c1 = ConvAct(c1, c2, 1)
        self.u2_c2 = ConvAct(c2, c2, 3)
        self.u2_c3 = ConvAct(c2, c2, 3)
        self.u2_c4 = ConvAct(c2, c2, 3)
        self.u2_res = BasicResidualBlock(c2)

        self.u3_c1 = ConvAct(c2, c3, 1)
        self.u3_c2 = ConvAct(c3, c3, 3)
        self.u3_c3 = ConvAct(c3, c3, 3)
        self.u3_c4 = ConvAct(c3, c3, 3)
        self.u3_res = BasicResidualBlock(c3)

        self.out_face0 = ConvAct(c3, 3, 1)
        self.out_face1 = ConvAct(c3, 3, 3)
        self.out_face2 = ConvAct(c3, 3, 3)
        self.out_face3 = ConvAct(c3, 3, 3)

        self.out_m2x0 = ConvAct(c3, 16, 3)
        self.out_m2x1 = ConvAct(c3, 16, 3)
        self.out_m2x2 = ConvAct(c3, 16, 3)
        self.out_m2x3 = ConvAct(c3, 16, 1)
        self.out_mask = torch.nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def up2(x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x):
        shapes = []
        x = self.proj(x)

        x = self.up2(x)
        x = self.u0_c4(self.u0_c3(self.u0_c2(self.u0_c1(x))))
        shapes.append(('u0', x.shape))

        x = self.up2(x)
        x = self.u1_c4(self.u1_c3(self.u1_c2(self.u1_c1(x))))
        x = self.u1_res1(self.u1_res0(x))
        shapes.append(('u1+res', x.shape))

        x = self.up2(x)
        x = self.u2_c4(self.u2_c3(self.u2_c2(self.u2_c1(x))))
        x = self.u2_res(x)
        shapes.append(('u2+res', x.shape))

        x = self.up2(x)
        x = self.u3_c4(self.u3_c3(self.u3_c2(self.u3_c1(x))))
        x = self.u3_res(x)
        shapes.append(('u3+res', x.shape))

        x2 = self.up2(x)

        face = torch.stack([
            self.out_face0(x2), self.out_face1(x2),
            self.out_face2(x2), self.out_face3(x2),
        ], dim=0).mean(0)

        m2x = torch.stack([
            self.out_m2x0(x2), self.out_m2x1(x2),
            self.out_m2x2(x2), self.out_m2x3(x2),
        ], dim=0).mean(0)

        mask = self.out_mask(m2x)
        return face, mask, shapes

    def get_mask_param_ids(self):
        """Return set of id(param) for mask-specific parameters.

        These can be excluded from optimizer updates when mask branch is frozen.
        Mask params: out_m2x0..3 (ensemble mask heads) + out_mask (1×1 final).
        """
        ids = set()
        for name, param in self.named_parameters():
            if name.startswith('out_m2x') or name == 'out_mask.weight' or name == 'out_mask.bias':
                ids.add(id(param))
        return ids


class LIAELarge(torch.nn.Module):
    """LIAE-style network: Encoder → Bottleneck_AB + Bottleneck_B → single Decoder.

    Bottleneck_AB learns shared identity features (bottleneck).
    Bottleneck_B learns dst-specific features (bottleneck).
    Decoder reconstructs from concatenated bottleneck codes.
    """

    def __init__(self, resolution=256, ae_dims=256, e_dims=64, d_dims=64):
        super().__init__()
        self.encoder = Encoder(resolution, e_dims)
        enc_ch = e_dims * 8
        inter_out = ae_dims
        self.bottleneck_AB = InterBottleneck(enc_ch, resolution, ae_dims, out_ch=inter_out)
        self.bottleneck_B = InterBottleneck(enc_ch, resolution, ae_dims, out_ch=inter_out)
        self.decoder = Decoder(d_dims, width=1.0, in_ch=inter_out * 2)

    def forward(self, x):
        enc, s1 = self.encoder(x)
        ab, z_ab = self.bottleneck_AB(enc)
        b, z_b = self.bottleneck_B(enc)
        z = torch.cat([ab, b], dim=1)
        face, mask, s2 = self.decoder(z)
        return {
            'face': face, 'mask': mask,
            'z_ab': z_ab, 'z_b': z_b,
            'bottleneck_AB': ab, 'bottleneck_B': b,
            'shapes': s1 + [('Bottleneck_AB', ab.shape), ('Bottleneck_B', b.shape),
                            ('cat', z.shape)] + [('D.' + n, sh) for n, sh in s2],
        }


# =============================================================================
# AdaBelief optimizer
# =============================================================================


class AdaBelief(torch.optim.Optimizer):
    """AdaBelief — Adam with belief in observed gradient."""

    def __init__(self, params, lr=5e-5, betas=(0.5, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']
                state['step'] += 1
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_diff = grad - m
                v.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m, denom, value=-step_size)
        return loss


# =============================================================================
# DFL Model class
# =============================================================================


class LIAELargeModel(ModelBase):
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()
        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb
        suggest_batch_size = 8 if lowest_vram >= 4 else 4
        min_res = 64
        max_res = 640

        default_resolution = self.options['resolution'] = self.load_or_def_option('resolution', 128)
        default_face_type = self.options['face_type'] = self.load_or_def_option('face_type', 'f')
        default_ae_dims = self.options['ae_dims'] = self.load_or_def_option('ae_dims', 256)
        default_e_dims = self.options['e_dims'] = self.load_or_def_option('e_dims', 64)
        default_d_dims = self.options['d_dims'] = self.load_or_def_option('d_dims', 64)
        default_masked_training = self.options['masked_training'] = self.load_or_def_option('masked_training', True)
        default_eyes_mouth_prio = self.options['eyes_mouth_prio'] = self.load_or_def_option('eyes_mouth_prio', False)
        default_blur_out_mask = self.options['blur_out_mask'] = self.load_or_def_option('blur_out_mask', False)
        default_random_warp = self.options['random_warp'] = self.load_or_def_option('random_warp', True)
        default_random_hsv_power = self.options['random_hsv_power'] = self.load_or_def_option('random_hsv_power', 0.0)
        default_uniform_yaw = self.options['uniform_yaw'] = self.load_or_def_option('uniform_yaw', False)
        default_ct_mode = self.options['ct_mode'] = self.load_or_def_option('ct_mode', 'none')
        default_adabelief = self.options['adabelief'] = self.load_or_def_option('adabelief', True)
        default_lr = self.options['lr'] = self.load_or_def_option('lr', 5e-5)
        lr_policy = self.load_or_def_option('lr_policy', 'CosineAnnealingLR')
        lr_policy = {'n': 'None', 'y': 'CosineAnnealingLR'}.get(lr_policy, lr_policy)
        default_lr_policy = self.options['lr_policy'] = lr_policy
        default_clipgrad = self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)
        default_pretrain = self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False
        default_gan_power = self.options['gan_power'] = self.load_or_def_option('gan_power', 0.0)
        default_use_bf16 = self.options['use_bf16'] = self.load_or_def_option('use_bf16', False)
        default_gradient_checkpointing = self.options['gradient_checkpointing'] = self.load_or_def_option('gradient_checkpointing', False)
        default_freeze_decoder_mask = self.options['freeze_decoder_mask'] = self.load_or_def_option('freeze_decoder_mask', False)
        default_freeze_encoder = self.options['freeze_encoder'] = self.load_or_def_option('freeze_encoder', False)
        default_freeze_bottleneck_AB = self.options['freeze_bottleneck_AB'] = self.load_or_def_option('freeze_bottleneck_AB', False)
        default_freeze_bottleneck_B = self.options['freeze_bottleneck_B'] = self.load_or_def_option('freeze_bottleneck_B', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_random_dst_flip()
            self.ask_batch_size(suggest_batch_size)
            self.ask_backup_interval()
            self.ask_max_backups()

        if self.is_first_run():
            resolution = io.input_int('分辨率', default_resolution, add_info='64-640')
            resolution = np.clip((resolution // 16) * 16, min_res, max_res)
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str(
                '人脸类型', default_face_type, ['h', 'mf', 'f', 'wf', 'head']).lower()
            self.options['ae_dims'] = int(np.clip(
                io.input_int('AE dims', default_ae_dims, add_info='32-1024'), 32, 1024))
            e_dims = int(np.clip(
                io.input_int('E dims', default_e_dims, add_info='16-256'), 16, 256))
            self.options['e_dims'] = e_dims + e_dims % 2
            d_dims = int(np.clip(
                io.input_int('D dims', default_d_dims, add_info='16-256'), 16, 256))
            self.options['d_dims'] = d_dims + d_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] in ('wf', 'head'):
                self.options['masked_training'] = io.input_bool('Masked training', default_masked_training)
            self.options['eyes_mouth_prio'] = io.input_bool('眼睛与嘴巴优先', default_eyes_mouth_prio)
            self.options['random_warp'] = io.input_bool('随机扭曲样本', default_random_warp)
            self.options['random_hsv_power'] = float(
                np.clip(io.input_number('随机 HSV 力度', default_random_hsv_power, add_info='0.0-0.3'), 0.0, 0.3))
            self.options['ct_mode'] = io.input_str('颜色迁移模式', default_ct_mode,
                                                    ['none', 'rct', 'lct', 'mkl', 'idt', 'sot'])
            self.options['gan_power'] = float(
                np.clip(io.input_number('GAN 力度', default_gan_power, add_info='0.0-10.0'), 0.0, 10.0))
            self.options['adabelief'] = io.input_bool('使用 AdaBelief 优化器（否则 Adam）', default_adabelief)
            self.options['lr'] = float(
                np.clip(io.input_number('学习率', default_lr, add_info='1e-6 ~ 1e-3'), 1e-6, 1e-3))
            self.options['lr_policy'] = io.input_str(
                '学习率策略', default_lr_policy, ['None', 'CosineAnnealingLR'])
            self.options['clipgrad'] = io.input_bool('启用梯度裁剪', default_clipgrad)
            self.options['use_bf16'] = io.input_bool('启用 BF16', default_use_bf16)
            self.options['gradient_checkpointing'] = io.input_bool('启用梯度检查点', default_gradient_checkpointing)
            self.options['freeze_decoder_mask'] = io.input_bool(
                '冻结解码器 Mask 分支（共享解码器）', default_freeze_decoder_mask)
            self.options['freeze_encoder'] = io.input_bool(
                '冻结编码器（Encoder）', default_freeze_encoder)
            self.options['freeze_bottleneck_AB'] = io.input_bool(
                '冻结共享瓶颈（Bottleneck_AB）', default_freeze_bottleneck_AB)
            self.options['freeze_bottleneck_B'] = io.input_bool(
                '冻结目标瓶颈（Bottleneck_B）', default_freeze_bottleneck_B)

    def _select_device(self):
        if getattr(nn, 'device', None) is not None:
            return nn.device
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        if len(devices) > 0 and torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    def _np_to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x
        # torch.tensor 强制创建副本，防止与 numpy 共享内存导致数据污染
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _move_leras_model_to_device(self, model):
        try:
            layers = model.get_layers()
        except Exception:
            return
        for layer in layers:
            try:
                layer.to(self.device)
            except Exception:
                pass

    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.model_data_format = 'NCHW'
        nn.initialize(device_config, data_format=self.model_data_format)

        self.device = self._select_device()
        self.resolution = resolution = int(self.options['resolution'])
        self.face_type = {
            'h': FaceType.HALF, 'mf': FaceType.MID_FULL, 'f': FaceType.FULL,
            'wf': FaceType.WHOLE_FACE, 'head': FaceType.HEAD,
        }[self.options['face_type']]

        self.eyes_mouth_prio = bool(self.options['eyes_mouth_prio'])
        self.masked_training = bool(self.options['masked_training'])
        self.blur_out_mask = bool(self.options['blur_out_mask'])
        self.use_bf16 = bool(self.load_or_def_option('use_bf16', True))  # None（旧模型未存）时回退 True，避免误用 FP32

        ae_dims = int(self.options['ae_dims'])
        e_dims = int(self.options['e_dims'])
        d_dims = int(self.options['d_dims'])

        self.pretrain = bool(self.options['pretrain'])
        if getattr(self, 'pretrain_just_disabled', False):
            self.set_iter(0)

        self.gan_power = gan_power = 0.0 if self.pretrain else float(self.options['gan_power'])
        random_warp = False if self.pretrain else bool(self.options['random_warp'])
        random_src_flip = True if self.pretrain else bool(self.random_src_flip)
        random_dst_flip = True if self.pretrain else bool(self.random_dst_flip)
        random_hsv_power = 0.0 if self.pretrain else float(self.options['random_hsv_power'])

        if self.pretrain:
            self.options_show_override['random_warp'] = False
            self.options_show_override['gan_power'] = 0.0
            self.options_show_override['random_hsv_power'] = 0.0

        ct_mode = self.options['ct_mode']
        if ct_mode == 'none':
            ct_mode = None

        # -- Build network --
        self.net = LIAELarge(resolution=resolution, ae_dims=ae_dims,
                              e_dims=e_dims, d_dims=d_dims).to(self.device)

        # GAN discriminator
        if self.is_training and gan_power != 0.0:
            self.D_src = nn.UNetPatchDiscriminator(
                patch_size=int(self.options.get('gan_patch_size', resolution // 8)),
                in_ch=3, base_ch=int(self.options.get('gan_dims', 16)), name='D_src',
            )
            self._move_leras_model_to_device(self.D_src)

        # -- Optimizers --
        if self.is_training:
            lr = float(self.options.get('lr', 5e-5))
            adabelief = bool(self.options.get('adabelief', True))
            OptimizerClass = AdaBelief if adabelief else torch.optim.Adam

            frozen_ids = set()
            if bool(self.options.get('freeze_decoder_mask', False)):
                frozen_ids |= self.net.decoder.get_mask_param_ids()
            if bool(self.options.get('freeze_encoder', False)):
                frozen_ids |= {id(p) for p in self.net.encoder.parameters()}
            if bool(self.options.get('freeze_bottleneck_AB', False)):
                frozen_ids |= {id(p) for p in self.net.bottleneck_AB.parameters()}
            if bool(self.options.get('freeze_bottleneck_B', False)):
                frozen_ids |= {id(p) for p in self.net.bottleneck_B.parameters()}
            if frozen_ids:
                train_params = [p for p in self.net.parameters() if id(p) not in frozen_ids]
                io.log_info(f'  {sum(p.numel() for p in train_params):,} trainable params')
            else:
                train_params = self.net.parameters()
            self.optimizer = OptimizerClass(
                train_params, lr=lr, betas=(0.5, 0.999),
            )
            if gan_power != 0.0:
                self.D_optimizer = torch.optim.Adam(
                    self.D_src.parameters(), lr=lr, betas=(0.5, 0.999),
                )

            lr_policy = self.options.get('lr_policy', 'CosineAnnealingLR')
            self.lr_scheduler = None
            self.lr_cos = 0
            if lr_policy == 'CosineAnnealingLR' and not self.pretrain:
                self.lr_cos = self.options.get('lr_cos', 500)
                if self.lr_cos <= 0:
                    self.lr_cos = 500
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_cos, eta_min=lr * 0.01,
                )

        # -- Load weights + build module info --
        self._module_info_list = []

        def _load_pth(component, module, display_name):
            path = self._pth_path(component)
            loaded = False
            if Path(path).exists() and not self.is_first_run():
                try:
                    sd = torch.load(path, map_location=self.device)
                    module.load_state_dict(sd)
                    loaded = True
                    io.log_info(f'  loaded {Path(path).name}')
                except Exception as e:
                    io.log_info(f'  skip {Path(path).name}: {e}')
            params = sum(p.numel() for p in module.parameters())
            status = '已加载' if loaded else ('新初始化' if self.is_first_run() else '新初始化')
            self._module_info_list.append((display_name, params, status))

        _load_pth('encoder.pth', self.net.encoder, 'Encoder')
        _load_pth('bottleneck_AB.pth', self.net.bottleneck_AB, 'Bottleneck_AB')
        _load_pth('bottleneck_B.pth', self.net.bottleneck_B, 'Bottleneck_B')
        _load_pth('decoder.pth', self.net.decoder, 'Decoder')

        # Split decoder mask into separate display entry + annotate freeze status
        extra_entries = []
        for i, (name, params, status) in list(enumerate(self._module_info_list)):
            if name == 'Encoder' and self.options.get('freeze_encoder', False):
                self._module_info_list[i] = (name, params, '已冻结')
            elif name == 'Bottleneck_AB' and self.options.get('freeze_bottleneck_AB', False):
                self._module_info_list[i] = (name, params, '已冻结')
            elif name == 'Bottleneck_B' and self.options.get('freeze_bottleneck_B', False):
                self._module_info_list[i] = (name, params, '已冻结')
            elif name == 'Decoder':
                mask_ids = self.net.decoder.get_mask_param_ids()
                m_p = sum(p.numel() for p in self.net.decoder.parameters() if id(p) in mask_ids)
                f_p = params - m_p
                mask_frozen = self.options.get('freeze_decoder_mask', False)
                self._module_info_list[i] = (name, f_p, status)
                extra_entries.append((f'{name}_Mask', m_p,
                                      '已冻结' if mask_frozen else status))
        self._module_info_list.extend(extra_entries)

        if self.is_training:
            opt_path = self._pth_path('opt.pth')
            if Path(opt_path).exists() and not self.is_first_run():
                try:
                    opt_sd = torch.load(opt_path, map_location=self.device)
                    if 'optimizer' in opt_sd:
                        self.optimizer.load_state_dict(opt_sd['optimizer'])
                    else:
                        self.optimizer.load_state_dict(opt_sd)
                    if self.lr_scheduler is not None and opt_sd.get('scheduler') is not None:
                        self.lr_scheduler.load_state_dict(opt_sd['scheduler'])
                    io.log_info(f'  loaded {Path(opt_path).name}')
                except Exception as e:
                    io.log_info(f'  skip {Path(opt_path).name}: {e}')

        if self.is_training and gan_power != 0.0:
            gan_path = self._pth_path('GAN.pth')
            if Path(gan_path).exists() and not self.is_first_run():
                try:
                    sd = torch.load(gan_path, map_location=self.device)
                    self.D_src.load_state_dict(sd)
                    io.log_info(f'  loaded {Path(gan_path).name}')
                except Exception as e:
                    io.log_info(f'  GAN load: {e}')

        # -- Data generators (快速加载器 V4 + V2) --
        if self.is_training:
            ts = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            td = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            src_loader = SampleLoaderV4(
                aligned_path=ts, batch_size=self.get_batch_size(),
                resolution=resolution,
                use_yaw_sampling=bool(self.options['uniform_yaw']) or self.pretrain,
            )
            dst_loader = SampleLoaderV4(
                aligned_path=td, batch_size=self.get_batch_size(),
                resolution=resolution,
                use_yaw_sampling=bool(self.options['uniform_yaw']) or self.pretrain,
            )
            src_ct_loader = dst_loader

            def _oty(ct):
                return [
                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                     'warp': random_warp, 'transform': True,
                     'channel_type': SampleProcessor.ChannelType.BGR,
                     'ct_mode': ct, 'random_hsv_shift_amount': random_hsv_power,
                     'face_type': self.face_type, 'data_format': nn.data_format,
                     'resolution': resolution},
                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                     'warp': False, 'transform': True,
                     'channel_type': SampleProcessor.ChannelType.BGR,
                     'ct_mode': ct, 'face_type': self.face_type,
                     'data_format': nn.data_format, 'resolution': resolution},
                    {'sample_type': SampleProcessor.SampleType.FACE_MASK,
                     'warp': False, 'transform': True,
                     'channel_type': SampleProcessor.ChannelType.G,
                     'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                     'face_type': self.face_type, 'data_format': nn.data_format,
                     'resolution': resolution},
                    {'sample_type': SampleProcessor.SampleType.FACE_MASK,
                     'warp': False, 'transform': True,
                     'channel_type': SampleProcessor.ChannelType.G,
                     'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
                     'face_type': self.face_type, 'data_format': nn.data_format,
                     'resolution': resolution},
                ]

            self.set_training_data_generators([
                SampleGeneratorV2(
                    loader=src_loader,
                    sample_process_options=SampleProcessor.Options(
                        scale_range=[-0.15, 0.15], random_flip=random_src_flip),
                    output_sample_types=_oty(ct_mode),
                    resolution=resolution, ct_loader=src_ct_loader,
                ),
                SampleGeneratorV2(
                    loader=dst_loader,
                    sample_process_options=SampleProcessor.Options(
                        scale_range=[-0.15, 0.15], random_flip=random_dst_flip),
                    output_sample_types=_oty(None), resolution=resolution,
                ),
            ])

    # ---- Weight management ----
    def _pth_rel(self, component):
        return f'{self.model_class_name}_{component}'

    def _pth_path(self, component):
        name = f'{self.get_model_name()}_{self.model_class_name}_{component}'
        return self.get_model_root_path() / name

    def get_model_filename_list(self):
        files = ['encoder.pth', 'bottleneck_AB.pth', 'bottleneck_B.pth', 'decoder.pth']
        if self.is_training and hasattr(self, 'optimizer'):
            files.append('opt.pth')
        if self.gan_power != 0.0 and hasattr(self, 'D_src'):
            files.append('GAN.pth')
        return [[self, self._pth_rel(f)] for f in files]

    def _save_pth(self, component, state_dict):
        torch.save(state_dict, self._pth_path(component))

    def onSave(self):
        self._save_pth('encoder.pth', self.net.encoder.state_dict())
        self._save_pth('bottleneck_AB.pth', self.net.bottleneck_AB.state_dict())
        self._save_pth('bottleneck_B.pth', self.net.bottleneck_B.state_dict())
        self._save_pth('decoder.pth', self.net.decoder.state_dict())
        if self.is_training and hasattr(self, 'optimizer'):
            opt_sd = {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            }
            self._save_pth('opt.pth', opt_sd)
        if self.gan_power != 0.0 and hasattr(self, 'D_src'):
            self._save_pth('GAN.pth', self.D_src.state_dict())

    # ── DFM export (ONNX) ──────────────────────────────────────────

    def export_dfm(self):
        """Export LIAELarge model to ONNX .dfm for DeepFaceLab merger."""
        import torch, os, warnings
        output_path = self.get_strpath_storage_for_file('model.dfm')
        io.log_info(f'Dumping .dfm to {output_path}')

        def _disable_grad(m):
            try:
                for w in m.get_weights():
                    if hasattr(w, 'requires_grad'):
                        w.requires_grad_(False)
            except Exception:
                pass

        _disable_grad(self.net.encoder)
        _disable_grad(self.net.bottleneck_AB)
        _disable_grad(self.net.bottleneck_B)
        _disable_grad(self.net.decoder)

        class _DFMWrapper(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.encoder = parent.net.encoder
                self.bottleneck_AB = parent.net.bottleneck_AB
                self.bottleneck_B = parent.net.bottleneck_B
                self.decoder = parent.net.decoder

            def forward(self, in_face):
                x = in_face.permute(0, 3, 1, 2).contiguous()
                if hasattr(self, 'model_dtype') and x.dtype != self.model_dtype:
                    x = x.to(self.model_dtype)
                code = self.encoder(x)
                inter_b = self.bottleneck_B(code)
                inter_ab = self.bottleneck_AB(code)
                code_dst = torch.cat([inter_b, inter_ab], dim=1)
                code_src_dst = torch.cat([inter_ab, inter_ab], dim=1)
                out_celeb_face, out_celeb_face_mask, _ = self.decoder(code_src_dst)
                _, out_face_mask, _ = self.decoder(code_dst)

                out_celeb_face = out_celeb_face.to(torch.float32)
                out_celeb_face_mask = out_celeb_face_mask.to(torch.float32)
                out_face_mask = out_face_mask.to(torch.float32)

                out_face_mask = out_face_mask.permute(0, 2, 3, 1).contiguous()
                out_celeb_face = out_celeb_face.permute(0, 2, 3, 1).contiguous()
                out_celeb_face_mask = out_celeb_face_mask.permute(0, 2, 3, 1).contiguous()
                return out_face_mask, out_celeb_face, out_celeb_face_mask

        wrapper = _DFMWrapper(self)
        wrapper.eval()

        _export_prec = os.environ.get("DFM_EXPORT_PRECISION", "fp32")
        use_fp16_export = _export_prec == "fp16"
        if use_fp16_export:
            wrapper.model_dtype = torch.float16
            for sub in ["encoder", "bottleneck_AB", "bottleneck_B", "decoder"]:
                m = getattr(wrapper, sub, None)
                if m is not None:
                    m.to(torch.float16)

        dummy = torch.randn(1, self.resolution, self.resolution, 3)
        if use_fp16_export:
            dummy = dummy.half()

        warnings.filterwarnings(
            'ignore',
            message=r'Constant folding - Only steps=1 can be constant folded.*onnx::Slice.*',
            category=UserWarning,
        )

        torch.onnx.export(
            wrapper, dummy, output_path,
            input_names=['in_face:0'],
            output_names=['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0'],
            dynamic_axes={
                'in_face:0': {0: 'batch'},
                'out_face_mask:0': {0: 'batch'},
                'out_celeb_face:0': {0: 'batch'},
                'out_celeb_face_mask:0': {0: 'batch'},
            },
            opset_version=20 if use_fp16_export else 12,
        )
        io.log_info(f'✓ .dfm exported to {output_path}')

    # ---- Forward (LIAE) ----

    def _forward(self, warped_src, warped_dst):
        # LIAE forward: encoder → bottleneck_AB + bottleneck_B → decoder
        src_enc, _ = self.net.encoder(warped_src)
        src_ab, _ = self.net.bottleneck_AB(src_enc)
        src_cat = torch.cat([src_ab, src_ab], dim=1)

        dst_enc, _ = self.net.encoder(warped_dst)
        dst_b, _ = self.net.bottleneck_B(dst_enc)
        dst_ab, _ = self.net.bottleneck_AB(dst_enc)
        dst_cat = torch.cat([dst_b, dst_ab], dim=1)

        src_dst_cat = torch.cat([dst_ab, dst_ab], dim=1)

        pred_src_src, pred_src_srcm, _ = self.net.decoder(src_cat)
        pred_dst_dst, pred_dst_dstm, _ = self.net.decoder(dst_cat)
        pred_src_dst, pred_src_dstm, _ = self.net.decoder(src_dst_cat)
        pred_src_dst_no_code_grad, _, _ = self.net.decoder(src_dst_cat.detach())

        return {
            'src_code': src_cat, 'dst_code': dst_cat,
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
            'pred_src_dst_no_code_grad': pred_src_dst_no_code_grad,
        }

    # ---- Training ----

    def train_one_step(self, warped_src, target_src, target_srcm, target_srcm_em,
                       warped_dst, target_dst, target_dstm, target_dstm_em):
        warped_src = self._np_to_torch(warped_src)
        warped_dst = self._np_to_torch(warped_dst)
        target_src = self._np_to_torch(target_src)
        target_dst = self._np_to_torch(target_dst)
        target_srcm = self._np_to_torch(target_srcm)
        target_srcm_em = self._np_to_torch(target_srcm_em)
        target_dstm = self._np_to_torch(target_dstm)
        target_dstm_em = self._np_to_torch(target_dstm_em)

        if self.blur_out_mask:
            sigma = float(self.resolution) / 128.0
            for t, tm in [(target_src, target_srcm), (target_dst, target_dstm)]:
                anti = 1.0 - tm
                x = nn.gaussian_blur(t * anti, sigma)
                y = 1.0 - nn.gaussian_blur(tm, sigma)
                y = torch.where(y == 0, torch.ones_like(y), y)
                t[:] = t * tm + (x / y) * anti

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
            if self.options.get('gradient_checkpointing', False) and self.is_training:
                fw = torch.utils.checkpoint.checkpoint(
                    self._forward, warped_src, warped_dst, use_reentrant=False)
            else:
                fw = self._forward(warped_src, warped_dst)

        if self.use_bf16:
            fw = {k: (v.float() if isinstance(v, torch.Tensor) else v)
                  for k, v in fw.items()}

        ps = fw['pred_src_src']; psm = fw['pred_src_srcm']
        pd = fw['pred_dst_dst']; pdm = fw['pred_dst_dstm']
        psd = fw['pred_src_dst']; psdm = fw['pred_src_dstm']
        psd_ncg = fw['pred_src_dst_no_code_grad']

        k = max(1, self.resolution // 32)
        tsm_b = torch.clamp(nn.gaussian_blur(target_srcm, k), 0.0, 0.5) * 2.0
        tdm_b = torch.clamp(nn.gaussian_blur(target_dstm, k), 0.0, 0.5) * 2.0
        tsm_ab = 1.0 - tsm_b

        style_mb = torch.clamp(tsm_b.detach(), 0.0, 1.0)
        style_mab = 1.0 - style_mb

        td_m = target_dst * tdm_b
        ts_am = target_src * tsm_ab
        ps_am = ps * tsm_ab

        ts_mo = target_src * tsm_b if self.masked_training else target_src
        td_mo = td_m if self.masked_training else target_dst
        ps_mo = ps * tsm_b if self.masked_training else ps
        pd_mo = pd * tdm_b if self.masked_training else pd

        def dssim(a, b, fs, w):
            return float(w) * nn.dssim(a, b, max_val=1.0, filter_size=fs).mean(dim=[1, 2, 3])
        def mse(a, b, w):
            return float(w) * ((a - b) ** 2).mean(dim=[1, 2, 3])
        def l1(a, b, w):
            return float(w) * (a - b).abs().mean(dim=[1, 2, 3])

        fs1 = max(1, int(self.resolution / 11.6))
        fs2 = max(1, int(self.resolution / 23.2))

        if self.resolution < 256:
            sl = dssim(ts_mo, ps_mo, fs1, 10)
            dl = dssim(td_mo, pd_mo, fs1, 10)
        else:
            sl = dssim(ts_mo, ps_mo, fs1, 5) + dssim(ts_mo, ps_mo, fs2, 5)
            dl = dssim(td_mo, pd_mo, fs1, 5) + dssim(td_mo, pd_mo, fs2, 5)

        sl = sl + mse(ts_mo, ps_mo, 10)
        dl = dl + mse(td_mo, pd_mo, 10)

        if self.eyes_mouth_prio:
            sl = sl + l1(target_src * target_srcm_em, ps * target_srcm_em, 300)
            dl = dl + l1(target_dst * target_dstm_em, pd * target_dstm_em, 300)

        sl = sl + mse(target_srcm, psm, 10)
        dl = dl + mse(target_dstm, pdm, 10)

        fsp = float(self.options.get('face_style_power', 0.0)) / 100.0
        bsp = float(self.options.get('bg_style_power', 0.0)) / 100.0
        esl = torch.tensor(0.0, device=self.device)

        if fsp != 0.0 and not self.pretrain:
            esl = esl + nn.style_loss(
                psd_ncg * psdm.detach(), pd.detach() * pdm.detach(),
                gaussian_blur_radius=self.resolution // 8, loss_weight=10000.0 * fsp)

        if bsp != 0.0 and not self.pretrain:
            tds_am = target_dst * style_mab
            psd_am = psd * style_mab
            esl = esl + (10.0 * bsp * nn.dssim(psd_am, tds_am, max_val=1.0, filter_size=fs1).mean()
                         + (10.0 * bsp) * ((psd_am - tds_am) ** 2).mean())

        emgl = torch.tensor(0.0, device=self.device)
        if self.masked_training and self.gan_power != 0.0:
            emgl = emgl + 0.000001 * nn.total_variation_mse(ps)
            emgl = emgl + 0.02 * ((ps_am - ts_am) ** 2).mean()

        G_loss = sl.mean() + dl.mean() + esl + emgl
        self._last_loss_per_sample = (sl + dl).detach().cpu().tolist()

        D_gan_loss = None
        if self.gan_power != 0.0:
            ts_gan = target_src * target_srcm if self.masked_training else target_src
            ps_gan = ps * target_srcm if self.masked_training else ps
            pd1, pd2 = self.D_src(ps_gan)
            td1, td2 = self.D_src(ts_gan)
            o1 = torch.ones_like(td1); z1 = torch.zeros_like(pd1)
            o2 = torch.ones_like(td2); z2 = torch.zeros_like(pd2)
            D_gan_loss = (0.5 * (F.binary_cross_entropy_with_logits(td1, o1)
                                 + F.binary_cross_entropy_with_logits(pd1.detach(), z1))
                          + 0.5 * (F.binary_cross_entropy_with_logits(td2, o2)
                                   + F.binary_cross_entropy_with_logits(pd2.detach(), z2)))
            G_loss = G_loss + self.gan_power * (
                F.binary_cross_entropy_with_logits(pd1, o1)
                + F.binary_cross_entropy_with_logits(pd2, o2))

        self.optimizer.zero_grad()
        G_loss.backward()
        if bool(self.options.get('clipgrad', False)):
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if D_gan_loss is not None:
            self.D_optimizer.zero_grad()
            D_gan_loss.backward()
            self.D_optimizer.step()

        return sl.detach().mean().item(), dl.detach().mean().item()

    def onTrainOneIter(self):
        ((warped_src, target_src, target_srcm, target_srcm_em),
         (warped_dst, target_dst, target_dstm, target_dstm_em)) = self.generate_next_samples()
        src_loss, dst_loss = self.train_one_step(
            warped_src, target_src, target_srcm, target_srcm_em,
            warped_dst, target_dst, target_dstm, target_dstm_em)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return (('src_loss', src_loss), ('dst_loss', dst_loss))

    # ---- Preview ----

    def AE_view(self, target_src, target_dst):
        target_src = self._np_to_torch(target_src)
        target_dst = self._np_to_torch(target_dst)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                fw = self._forward(target_src, target_dst)
        return (fw['pred_src_src'].detach().cpu().float().numpy(),
                fw['pred_dst_dst'].detach().cpu().float().numpy(),
                fw['pred_dst_dstm'].detach().cpu().float().numpy(),
                fw['pred_src_dst'].detach().cpu().float().numpy(),
                fw['pred_src_dstm'].detach().cpu().float().numpy())

    def onGetPreview(self, samples, for_history=False):
        ((warped_src, target_src, target_srcm, _),
         (warped_dst, target_dst, target_dstm, _)) = copy.deepcopy(samples)
        S, D, SS, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0)
            for x in ([target_src, target_dst] + list(self.AE_view(target_src, target_dst)))]
        tgt_srcm_nhwc = nn.to_data_format(target_srcm, 'NHWC', self.model_data_format).copy()
        tgt_dstm_nhwc = nn.to_data_format(target_dstm, 'NHWC', self.model_data_format).copy()

        ddm_1ch = DDM.copy(); sdm_1ch = SDM.copy()
        DDM = np.repeat(DDM, 3, axis=-1); SDM = np.repeat(SDM, 3, axis=-1)
        n_samples = min(4, self.get_batch_size())
        WS = np.clip(nn.to_data_format(warped_src, 'NHWC', self.model_data_format), 0.0, 1.0)
        WD = np.clip(nn.to_data_format(warped_dst, 'NHWC', self.model_data_format), 0.0, 1.0)
        result = []

        st = []
        for i in range(n_samples):
            st.append(np.concatenate((S[i], SS[i], D[i], DD[i], SD[i]), axis=1))
        result.append(('原图预览', np.concatenate(st, axis=0)))

        st = []
        for i in range(n_samples):
            sd_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
            st.append(np.concatenate(
                (S[i] * tgt_srcm_nhwc[i], SS[i], D[i] * tgt_dstm_nhwc[i],
                 DD[i] * DDM[i], SD[i] * sd_mask), axis=1))
        result.append(('遮罩下', np.concatenate(st, axis=0)))

        st = []
        for i in range(n_samples):
            st.append(np.concatenate((WS[i], SS[i], WD[i], DD[i], SD[i]), axis=1))
        result.append(('原始输入', np.concatenate(st, axis=0)))

        st = []
        for i in range(n_samples):
            if self.face_type < FaceType.HEAD:
                dm = tgt_dstm_nhwc[i] * ddm_1ch[i]
                sm = tgt_srcm_nhwc[i] * sdm_1ch[i]
            else:
                dm = tgt_dstm_nhwc[i]; sm = tgt_srcm_nhwc[i]
            st.append(np.concatenate(
                (S[i], SS[i] * sm + S[i] * (1.0 - sm),
                 D[i], DD[i] * dm + D[i] * (1.0 - dm),
                 SD[i] * dm + D[i] * (1.0 - dm)), axis=1))
        result.append(('合并预览', np.concatenate(st, axis=0)))

        self._preview_masks = {
            'col0': tgt_srcm_nhwc[:n_samples], 'col2': tgt_dstm_nhwc[:n_samples],
            'col3': ddm_1ch[:n_samples],
            'col4': np.stack([
                ddm_1ch[j] * sdm_1ch[j] if self.face_type < FaceType.HEAD else sdm_1ch[j]
                for j in range(n_samples)], axis=0),
        }
        return result

    # ---- Merge (LIAE) ----

    def AE_merge(self, warped_dst):
        warped_dst = self._np_to_torch(warped_dst)
        with torch.no_grad():
            enc, _ = self.net.encoder(warped_dst)
            ab, _ = self.net.bottleneck_AB(enc)
            b, _ = self.net.bottleneck_B(enc)
            src_dst_cat = torch.cat([ab, ab], dim=1)
            dst_cat = torch.cat([b, ab], dim=1)
            pred_src_dst, pred_src_dstm, _ = self.net.decoder(src_dst_cat)
            _, pred_dst_dstm, _ = self.net.decoder(dst_cat)
        return (pred_src_dst.detach().cpu().numpy(),
                pred_dst_dstm.detach().cpu().numpy(),
                pred_src_dstm.detach().cpu().numpy())

    def predictor_func(self, face=None):
        import cv2
        face = nn.to_data_format(face[None, ...], self.model_data_format, 'NHWC')
        bgr, mdd, msd = [nn.to_data_format(x, 'NHWC', self.model_data_format).astype(np.float32)
                         for x in self.AE_merge(face)]
        bgr0, ms0, md0 = bgr[0], msd[0, ..., 0], mdd[0, ..., 0]
        res = self.resolution
        nx = float(self.options.get('output_shift_x', 0.100))
        ny = float(self.options.get('output_shift_y', 0.100))
        dx = nx * res / 2.0; dy = ny * res / 2.0
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            bgr0 = cv2.warpAffine(bgr0, M, (res, res), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ms0 = cv2.warpAffine(ms0, M, (res, res), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            md0 = cv2.warpAffine(md0, M, (res, res), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return bgr0, ms0, md0

    def get_MergerConfig(self):
        import merger
        return (self.predictor_func,
                (self.options['resolution'], self.options['resolution'], 3),
                merger.MergerConfigMasked(face_type=self.face_type, default_mode='overlay'))


Model = LIAELargeModel
