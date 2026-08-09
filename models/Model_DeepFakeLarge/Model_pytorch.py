"""DeepFakeLarge Model — PyTorch training script (pure torch modules).

Architecture: Encoder + Linear Bottleneck + dual Decoder (A=wide, B=narrow).
Based on the HyperDF design: brute-force decoder, dual-branch output, linear bottleneck.
All operators are pure PyTorch, no TensorFlow compatibility.

Saved model naming:
    - DeepFakeLarge_data.dat         (options/iter via ModelBase)
    - DeepFakeLarge_<res>.pth        (network + optimizer state)
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
from samplelib import SampleGeneratorFace, SampleGeneratorV2, SampleLoaderV4, SampleProcessor

# =============================================================================
# Network layer classes (pure torch.nn.Module)
# =============================================================================


class ConvAct(torch.nn.Module):
    """Conv2d + InstanceNorm + LeakyReLU(0.2) building block.

    Normalization is mandatory and hidden — baked into every convolution
    for training stability without user configuration.
    """

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


class Bottleneck(torch.nn.Module):
    """Linear bottleneck: Flatten -> Linear -> LayerNorm -> Linear -> Reshape.

    LayerNorm stabilizes the ae_dims latent space without depending on batch size.
    """

    def __init__(self, enc_ch, resolution, ae_dims):
        super().__init__()
        self.enc_ch = enc_ch
        self.spatial = resolution // 32
        self.latent_dim = enc_ch * self.spatial * self.spatial
        self.fc1 = torch.nn.Linear(self.latent_dim, ae_dims)
        self.n1 = torch.nn.LayerNorm(ae_dims)
        self.fc2 = torch.nn.Linear(ae_dims, self.latent_dim)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, -1)
        z = self.n1(self.fc1(x))
        x = self.fc2(z).reshape(b, self.enc_ch, self.spatial, self.spatial)
        return x, z


class ResidualBlock(torch.nn.Module):
    """1x1 -> 3x3 -> 3x3 -> 1x1 bottleneck residual block."""

    def __init__(self, c, mid):
        super().__init__()
        self.c0 = ConvAct(c, mid, 1)
        self.c1 = ConvAct(mid, mid, 3)
        self.c2 = ConvAct(mid, mid, 3)
        self.c3 = ConvAct(mid, c, 1)

    def forward(self, x):
        return x + self.c3(self.c2(self.c1(self.c0(x))))


class BasicResidualBlock(torch.nn.Module):
    """3x3 -> 3x3 simple residual block."""

    def __init__(self, c):
        super().__init__()
        self.c1 = ConvAct(c, c, 3)
        self.c2 = ConvAct(c, c, 3)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class Decoder(torch.nn.Module):
    """Decoder branch with 4 upsampling stages and ensemble output heads.

    Args:
        d_dims: Base channel count.
        width:   Channel scaling factor (1.0 for full, 0.6875 for narrow).
        in_ch:   Input channel count from the bottleneck.
    """

    def __init__(self, d_dims=64, width=1.0, in_ch=None):
        super().__init__()
        scale = width
        c0 = max(16, int(d_dims * 8 * scale))
        c1 = max(16, int(d_dims * 8 * scale))
        c2 = max(16, int(d_dims * 4 * scale))
        c3 = max(16, int(d_dims * 2 * scale))
        rmid = max(16, int(d_dims * 4 * scale))

        # Input projection
        self.proj = ConvAct(in_ch, c0, 1)

        # Stage u0
        self.u0_c1 = ConvAct(c0, c0, 1)
        self.u0_c2 = ConvAct(c0, c0, 3)
        self.u0_c3 = ConvAct(c0, c0, 3)
        self.u0_c4 = ConvAct(c0, c0, 3)

        # Stage u1
        self.u1_c1 = ConvAct(c0, c1, 1)
        self.u1_c2 = ConvAct(c1, c1, 3)
        self.u1_c3 = ConvAct(c1, c1, 3)
        self.u1_c4 = ConvAct(c1, c1, 3)
        self.u1_res0 = ResidualBlock(c1, rmid)
        self.u1_res1 = ResidualBlock(c1, rmid)

        # Stage u2
        self.u2_c1 = ConvAct(c1, c2, 1)
        self.u2_c2 = ConvAct(c2, c2, 3)
        self.u2_c3 = ConvAct(c2, c2, 3)
        self.u2_c4 = ConvAct(c2, c2, 3)
        self.u2_res = BasicResidualBlock(c2)

        # Stage u3
        self.u3_c1 = ConvAct(c2, c3, 1)
        self.u3_c2 = ConvAct(c3, c3, 3)
        self.u3_c3 = ConvAct(c3, c3, 3)
        self.u3_c4 = ConvAct(c3, c3, 3)
        self.u3_res = BasicResidualBlock(c3)

        # Ensemble output heads — 4 parallel Conv heads -> averaged
        self.out_face0 = ConvAct(c3, 3, 1)
        self.out_face1 = ConvAct(c3, 3, 3)
        self.out_face2 = ConvAct(c3, 3, 3)
        self.out_face3 = ConvAct(c3, 3, 3)

        # Mask path — 4 parallel heads -> averaged -> 1x1 conv -> 1ch
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

        # Input projection
        x = self.proj(x)

        # u0: up -> 4x conv
        x = self.up2(x)
        x = self.u0_c4(self.u0_c3(self.u0_c2(self.u0_c1(x))))
        shapes.append(('u0', x.shape))

        # u1: up -> 4x conv -> 2x residual
        x = self.up2(x)
        x = self.u1_c4(self.u1_c3(self.u1_c2(self.u1_c1(x))))
        x = self.u1_res1(self.u1_res0(x))
        shapes.append(('u1+res', x.shape))

        # u2: up -> 4x conv -> residual
        x = self.up2(x)
        x = self.u2_c4(self.u2_c3(self.u2_c2(self.u2_c1(x))))
        x = self.u2_res(x)
        shapes.append(('u2+res', x.shape))

        # u3: up -> 4x conv -> residual
        x = self.up2(x)
        x = self.u3_c4(self.u3_c3(self.u3_c2(self.u3_c1(x))))
        x = self.u3_res(x)
        shapes.append(('u3+res', x.shape))

        # Final up -> ensemble output
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


class DeepFakeLarge(torch.nn.Module):
    """Hyper Decoder Fusion Autoencoder — the full network.

    Composes Encoder, Bottleneck, and dual Decoders (A=wide/src, B=narrow/dst).

    Args:
        resolution: Input spatial size (128/256/512).
        ae_dims:    Bottleneck latent dimension.
        e_dims:     Encoder base channel count.
        d_dims:     Decoder base channel count.
    """

    def __init__(self, resolution=256, ae_dims=256, e_dims=64, d_dims=64):
        super().__init__()
        self.encoder = Encoder(resolution, e_dims)
        self.bottleneck = Bottleneck(e_dims * 8, resolution, ae_dims)
        enc_ch = e_dims * 8
        self.decoderA = Decoder(d_dims, width=1.0, in_ch=enc_ch)
        self.decoderB = Decoder(d_dims, width=0.6875, in_ch=enc_ch)

    def forward(self, x):
        enc, s1 = self.encoder(x)
        lat, z = self.bottleneck(enc)
        a_face, a_mask, s2 = self.decoderA(lat)
        b_face, b_mask, s3 = self.decoderB(lat)
        return {
            'A_face': a_face,
            'A_mask': a_mask,
            'B_face': b_face,
            'B_mask': b_mask,
            'z': z,
            'shapes': (s1
                       + [('latent_z', z.shape), ('latent_out', lat.shape)]
                       + [('A.' + n, sh) for n, sh in s2]
                       + [('B.' + n, sh) for n, sh in s3]),
        }


# =============================================================================
# AdaBelief optimizer (pure torch implementation)
# =============================================================================


class AdaBelief(torch.optim.Optimizer):
    """AdaBelief optimizer — Adam with 'belief' in the observed gradient.

    v_t = beta2 * v_{t-1} + (1-beta2) * (g_t - m_t)^2 + eps
    Uses (g - m)^2 instead of g^2 for more adaptive step sizes.
    """

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

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second 'belief' estimate (g - m)^2
                grad_diff = grad - m
                v.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1

                # Update parameters
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m, denom, value=-step_size)

        return loss


# =============================================================================
# DFL Model class
# =============================================================================


class DeepFakeLargeModel(ModelBase):
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
        # backward compat: old 'n'/'y' convention
        lr_policy = {'n': 'None', 'y': 'CosineAnnealingLR'}.get(lr_policy, lr_policy)
        default_lr_policy = self.options['lr_policy'] = lr_policy

        default_clipgrad = self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)
        default_pretrain = self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False
        default_gan_power = self.options['gan_power'] = self.load_or_def_option('gan_power', 0.0)
        default_use_bf16 = self.options['use_bf16'] = self.load_or_def_option('use_bf16', False)
        default_gradient_checkpointing = self.options['gradient_checkpointing'] = self.load_or_def_option('gradient_checkpointing', False)
        default_freeze_decoderA_mask = self.options['freeze_decoderA_mask'] = self.load_or_def_option('freeze_decoderA_mask', False)
        default_freeze_decoderB_mask = self.options['freeze_decoderB_mask'] = self.load_or_def_option('freeze_decoderB_mask', False)
        # backward compat: old single freeze_decoder_mask maps to both
        if not self.is_first_run() and self.load_or_def_option('freeze_decoder_mask', False):
            self.options['freeze_decoderA_mask'] = True
            self.options['freeze_decoderB_mask'] = True
        default_freeze_encoder = self.options['freeze_encoder'] = self.load_or_def_option('freeze_encoder', False)
        default_freeze_bottleneck = self.options['freeze_bottleneck'] = self.load_or_def_option('freeze_bottleneck', False)
        default_freeze_decoderB = self.options['freeze_decoderB'] = self.load_or_def_option('freeze_decoderB', False)  # 冻结整个 dst 解码器（DF 架构专用）

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
                '人脸类型', default_face_type, ['h', 'mf', 'f', 'wf', 'head']
            ).lower()

            self.options['ae_dims'] = int(np.clip(
                io.input_int('自编码器维度（AE dims）', default_ae_dims, add_info='32-1024'), 32, 1024))

            e_dims = int(np.clip(
                io.input_int('编码器维度（E dims）', default_e_dims, add_info='16-256'), 16, 256))
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = int(np.clip(
                io.input_int('解码器维度（D dims）', default_d_dims, add_info='16-256'), 16, 256))
            self.options['d_dims'] = d_dims + d_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] in ('wf', 'head'):
                self.options['masked_training'] = io.input_bool(
                    '启用 Masked training', default_masked_training)
            self.options['eyes_mouth_prio'] = io.input_bool(
                '眼睛与嘴巴优先', default_eyes_mouth_prio)
            self.options['random_warp'] = io.input_bool(
                '随机扭曲样本', default_random_warp)
            self.options['random_hsv_power'] = float(
                np.clip(io.input_number('随机 HSV 力度', default_random_hsv_power,
                                        add_info='0.0-0.3'), 0.0, 0.3))
            ct_choices = ['none', 'rct', 'lct', 'mkl', 'idt', 'sot']
            self.options['ct_mode'] = io.input_str(
                '颜色迁移模式', default_ct_mode, ct_choices)
            self.options['gan_power'] = float(
                np.clip(io.input_number('GAN 力度', default_gan_power,
                                        add_info='0.0-10.0'), 0.0, 10.0))
            self.options['adabelief'] = io.input_bool(
                '使用 AdaBelief 优化器（否则用 Adam）', default_adabelief)
            self.options['lr'] = float(
                np.clip(io.input_number('学习率', default_lr,
                                        add_info='1e-6 ~ 1e-3'), 1e-6, 1e-3))
            self.options['lr_policy'] = io.input_str(
                '学习率策略', default_lr_policy,
                ['None', 'CosineAnnealingLR'])
            self.options['clipgrad'] = io.input_bool(
                '启用梯度裁剪（防模型崩溃）', default_clipgrad)
            self.options['use_bf16'] = io.input_bool(
                '启用 BF16 训练（需 Ampere+ GPU）', default_use_bf16)
            self.options['gradient_checkpointing'] = io.input_bool(
                '启用梯度检查点（省显存，略慢）', default_gradient_checkpointing)
            self.options['freeze_decoderA_mask'] = io.input_bool(
                '冻结解码器 A Mask 分支（src/dst交叉用）', default_freeze_decoderA_mask)
            self.options['freeze_decoderB_mask'] = io.input_bool(
                '冻结解码器 B Mask 分支（dst重建用）', default_freeze_decoderB_mask)
            self.options['freeze_encoder'] = io.input_bool(
                '冻结编码器（Encoder）', default_freeze_encoder)
            self.options['freeze_bottleneck'] = io.input_bool(
                '冻结瓶颈层（Bottleneck）', default_freeze_bottleneck)

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
            'h': FaceType.HALF,
            'mf': FaceType.MID_FULL,
            'f': FaceType.FULL,
            'wf': FaceType.WHOLE_FACE,
            'head': FaceType.HEAD,
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

        # -- Build pure torch network --
        self.net = DeepFakeLarge(
            resolution=resolution, ae_dims=ae_dims,
            e_dims=e_dims, d_dims=d_dims,
        ).to(self.device)

        # GAN discriminator (leras-based, used only when gan_power > 0)
        if self.is_training and gan_power != 0.0:
            self.D_src = nn.UNetPatchDiscriminator(
                patch_size=int(self.options.get('gan_patch_size', resolution // 8)),
                in_ch=3,
                base_ch=int(self.options.get('gan_dims', 16)),
                name='D_src',
            )
            self._move_leras_model_to_device(self.D_src)

        # -- Optimizers --
        if self.is_training:
            lr = float(self.options.get('lr', 5e-5))
            adabelief = bool(self.options.get('adabelief', True))
            OptimizerClass = AdaBelief if adabelief else torch.optim.Adam

            frozen_ids = set()
            if bool(self.options.get('freeze_decoderA_mask', False)):
                frozen_ids |= self.net.decoderA.get_mask_param_ids()
            if bool(self.options.get('freeze_decoderB_mask', False)):
                frozen_ids |= self.net.decoderB.get_mask_param_ids()
            if bool(self.options.get('freeze_encoder', False)):
                frozen_ids |= {id(p) for p in self.net.encoder.parameters()}
            if bool(self.options.get('freeze_bottleneck', False)):
                frozen_ids |= {id(p) for p in self.net.bottleneck.parameters()}
            if bool(self.options.get('freeze_decoderB', False)):
                frozen_ids |= {id(p) for p in self.net.decoderB.parameters()}
                # 联动：同时冻结 encoder + bottleneck（用户习惯一起开，只精修 src 解码器）
                frozen_ids |= {id(p) for p in self.net.encoder.parameters()}
                frozen_ids |= {id(p) for p in self.net.bottleneck.parameters()}
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

            # -- LR scheduler --
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

        # -- Load weights + build module info for summary --
        # Per-component files: {model_name}_{component} via get_strpath_storage_for_file
        self._module_info_list = []

        def _load_pth(component, module, display_name):
            path = self._pth_path(component)
            loaded = False
            if Path(path).exists() and not self.is_first_run():
                try:
                    sd = torch.load(path, map_location=self.device)
                    module.load_state_dict(sd)
                    io.log_info(f'  loaded {Path(path).name}')
                    loaded = True
                except Exception as e:
                    io.log_info(f'  skip {Path(path).name}: {e}')
            params = sum(p.numel() for p in module.parameters())
            status = '已加载' if loaded else ('新初始化' if self.is_first_run() else '新初始化')
            self._module_info_list.append((display_name, params, status))

        _load_pth('encoder.pth', self.net.encoder, 'Encoder')
        _load_pth('inter.pth', self.net.bottleneck, 'Bottleneck')
        _load_pth('decoder_src.pth', self.net.decoderA, 'Decoder_A')
        _load_pth('decoder_dst.pth', self.net.decoderB, 'Decoder_B')

        # Split decoder mask branches into separate display entries
        # Also annotate freeze status for each module
        extra_entries = []
        for i, (name, params, status) in list(enumerate(self._module_info_list)):
            if name == 'Encoder' and self.options.get('freeze_encoder', False):
                self._module_info_list[i] = (name, params, '已冻结')
            elif name == 'Bottleneck' and self.options.get('freeze_bottleneck', False):
                self._module_info_list[i] = (name, params, '已冻结')
            elif name in ('Decoder_A', 'Decoder_B', 'Decoder'):
                decoder_attr = {'Decoder_A': 'decoderA', 'Decoder_B': 'decoderB',
                                'Decoder': 'decoder'}[name]
                decoder = getattr(self.net, decoder_attr, None)
                if decoder is not None:
                    mask_ids = decoder.get_mask_param_ids()
                    m_p = sum(p.numel() for p in decoder.parameters() if id(p) in mask_ids)
                    f_p = params - m_p
                    mask_opt = {'Decoder_A': 'freeze_decoderA_mask',
                                'Decoder_B': 'freeze_decoderB_mask',
                                'Decoder': 'freeze_decoder_mask'}[name]
                    mask_frozen = self.options.get(mask_opt, False)
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
            opt_params = sum(p.numel() for p in self.optimizer.param_groups[0]['params'])
            self._module_info_list.append(('Optimizer', opt_params, '训练中'))

        if self.is_training and gan_power != 0.0:
            gan_path = self._pth_path('GAN.pth')
            if Path(gan_path).exists() and not self.is_first_run():
                try:
                    sd = torch.load(gan_path, map_location=self.device)
                    self.D_src.load_state_dict(sd)
                    io.log_info(f'  loaded {Path(gan_path).name}')
                except Exception as e:
                    io.log_info(f'  GAN load: {e}')
            gan_params = sum(p.numel() for p in self.D_src.parameters())
            self._module_info_list.append(('Discriminator', gan_params, '训练中'))

        # -- Data generators (强制快速加载器 V4 + V2) --
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            src_loader = SampleLoaderV4(
                aligned_path=training_data_src_path,
                batch_size=self.get_batch_size(),
                resolution=resolution,
                use_yaw_sampling=bool(self.options['uniform_yaw']) or self.pretrain,
            )
            dst_loader = SampleLoaderV4(
                aligned_path=training_data_dst_path,
                batch_size=self.get_batch_size(),
                resolution=resolution,
                use_yaw_sampling=bool(self.options['uniform_yaw']) or self.pretrain,
            )
            src_ct_loader = dst_loader

            def _output_types(ct_mode_val):
                return [
                    {
                        'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                        'warp': random_warp, 'transform': True,
                        'channel_type': SampleProcessor.ChannelType.BGR,
                        'ct_mode': ct_mode_val,
                        'random_hsv_shift_amount': random_hsv_power,
                        'face_type': self.face_type,
                        'data_format': nn.data_format,
                        'resolution': resolution,
                    },
                    {
                        'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                        'warp': False, 'transform': True,
                        'channel_type': SampleProcessor.ChannelType.BGR,
                        'ct_mode': ct_mode_val,
                        'face_type': self.face_type,
                        'data_format': nn.data_format,
                        'resolution': resolution,
                    },
                    {
                        'sample_type': SampleProcessor.SampleType.FACE_MASK,
                        'warp': False, 'transform': True,
                        'channel_type': SampleProcessor.ChannelType.G,
                        'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                        'face_type': self.face_type,
                        'data_format': nn.data_format,
                        'resolution': resolution,
                    },
                    {
                        'sample_type': SampleProcessor.SampleType.FACE_MASK,
                        'warp': False, 'transform': True,
                        'channel_type': SampleProcessor.ChannelType.G,
                        'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
                        'face_type': self.face_type,
                        'data_format': nn.data_format,
                        'resolution': resolution,
                    },
                ]

            self.set_training_data_generators([
                SampleGeneratorV2(
                    loader=src_loader,
                    sample_process_options=SampleProcessor.Options(
                        scale_range=[-0.15, 0.15], random_flip=random_src_flip,
                    ),
                    output_sample_types=_output_types(ct_mode),
                    resolution=resolution,
                    ct_loader=src_ct_loader,
                ),
                SampleGeneratorV2(
                    loader=dst_loader,
                    sample_process_options=SampleProcessor.Options(
                        scale_range=[-0.15, 0.15], random_flip=random_dst_flip,
                    ),
                    output_sample_types=_output_types(None),
                    resolution=resolution,
                ),
            ])

    # ---- Weight management (per-component .pth files) ----
    # Naming: {model_name}_{class_name}_{component}.pth
    # e.g. test_DeepFakeLarge_encoder.pth

    def _pth_rel(self, component):
        """Relative filename for get_strpath_storage_for_file (backup compat)."""
        return f'{self.model_class_name}_{component}'

    def _pth_path(self, component):
        """Full save path: {root}/{name}_{class}_{component}.pth"""
        name = f'{self.get_model_name()}_{self.model_class_name}_{component}'
        return self.get_model_root_path() / name

    def get_model_filename_list(self):
        files = ['encoder.pth', 'inter.pth', 'decoder_src.pth', 'decoder_dst.pth']
        if self.is_training and hasattr(self, 'optimizer'):
            files.append('opt.pth')
        if self.gan_power != 0.0 and hasattr(self, 'D_src'):
            files.append('GAN.pth')
        return [[self, self._pth_rel(f)] for f in files]

    def _save_pth(self, component, state_dict):
        torch.save(state_dict, self._pth_path(component))

    def onSave(self):
        self._save_pth('encoder.pth', self.net.encoder.state_dict())
        self._save_pth('inter.pth', self.net.bottleneck.state_dict())
        self._save_pth('decoder_src.pth', self.net.decoderA.state_dict())
        self._save_pth('decoder_dst.pth', self.net.decoderB.state_dict())
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
        """Export DeepFakeLarge model to ONNX .dfm for DeepFaceLab merger."""
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
        _disable_grad(self.net.bottleneck)
        _disable_grad(self.net.decoderA)
        _disable_grad(self.net.decoderB)

        class _DFMWrapper(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.encoder = parent.net.encoder
                self.bottleneck = parent.net.bottleneck
                self.decoderA = parent.net.decoderA
                self.decoderB = parent.net.decoderB

            def forward(self, in_face):
                x = in_face.permute(0, 3, 1, 2).contiguous()
                if hasattr(self, 'model_dtype') and x.dtype != self.model_dtype:
                    x = x.to(self.model_dtype)
                code = self.bottleneck(self.encoder(x))
                out_celeb_face, out_celeb_face_mask, _ = self.decoderA(code)
                _, out_face_mask, _ = self.decoderB(code)

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
            for sub in ["encoder", "bottleneck", "decoderA", "decoderB"]:
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

    # ---- Forward ----

    def _forward(self, warped_src, warped_dst):
        src_enc, _ = self.net.encoder(warped_src)
        src_lat, _ = self.net.bottleneck(src_enc)
        dst_enc, _ = self.net.encoder(warped_dst)
        dst_lat, _ = self.net.bottleneck(dst_enc)

        pred_src_src, pred_src_srcm, _ = self.net.decoderA(src_lat)
        pred_dst_dst, pred_dst_dstm, _ = self.net.decoderB(dst_lat)
        pred_src_dst, pred_src_dstm, _ = self.net.decoderA(dst_lat)
        pred_src_dst_no_code_grad, _, _ = self.net.decoderA(dst_lat.detach())

        return {
            'src_code': src_lat,
            'dst_code': dst_lat,
            'pred_src_src': pred_src_src,
            'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst,
            'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst,
            'pred_src_dstm': pred_src_dstm,
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

        # blur-out-mask preprocessing
        if self.blur_out_mask:
            sigma = float(self.resolution) / 128.0
            srcm_anti = 1.0 - target_srcm
            x = nn.gaussian_blur(target_src * srcm_anti, sigma)
            y = 1.0 - nn.gaussian_blur(target_srcm, sigma)
            y = torch.where(y == 0, torch.ones_like(y), y)
            target_src = target_src * target_srcm + (x / y) * srcm_anti
            dstm_anti = 1.0 - target_dstm
            x = nn.gaussian_blur(target_dst * dstm_anti, sigma)
            y = 1.0 - nn.gaussian_blur(target_dstm, sigma)
            y = torch.where(y == 0, torch.ones_like(y), y)
            target_dst = target_dst * target_dstm + (x / y) * dstm_anti

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
            if self.options.get('gradient_checkpointing', False) and self.is_training:
                fw = torch.utils.checkpoint.checkpoint(
                    self._forward, warped_src, warped_dst, use_reentrant=False,
                )
            else:
                fw = self._forward(warped_src, warped_dst)

        if self.use_bf16:
            fw = {k: (v.float() if isinstance(v, torch.Tensor) else v)
                  for k, v in fw.items()}

        # ===== Reconstruction losses =====
        pred_src_src = fw['pred_src_src']
        pred_src_srcm = fw['pred_src_srcm']
        pred_dst_dst = fw['pred_dst_dst']
        pred_dst_dstm = fw['pred_dst_dstm']
        pred_src_dst = fw['pred_src_dst']
        pred_src_dstm = fw['pred_src_dstm']
        pred_src_dst_no_code_grad = fw['pred_src_dst_no_code_grad']

        k_blur = max(1, self.resolution // 32)
        target_srcm_blur = nn.gaussian_blur(target_srcm, k_blur)
        target_srcm_blur = torch.clamp(target_srcm_blur, 0.0, 0.5) * 2.0
        target_dstm_blur = nn.gaussian_blur(target_dstm, k_blur)
        target_dstm_blur = torch.clamp(target_dstm_blur, 0.0, 0.5) * 2.0
        target_srcm_anti_blur = 1.0 - target_srcm_blur

        style_mask_blur = target_srcm_blur.detach()
        style_mask_blur = torch.clamp(style_mask_blur, 0.0, 1.0)
        style_mask_anti_blur = 1.0 - style_mask_blur

        target_dst_masked = target_dst * target_dstm_blur
        target_src_anti_masked = target_src * target_srcm_anti_blur
        pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur

        target_src_masked_opt = target_src * target_srcm_blur if self.masked_training else target_src
        target_dst_masked_opt = target_dst_masked if self.masked_training else target_dst
        pred_src_src_masked_opt = pred_src_src * target_srcm_blur if self.masked_training else pred_src_src
        pred_dst_dst_masked_opt = pred_dst_dst * target_dstm_blur if self.masked_training else pred_dst_dst

        def dssim_loss(a, b, fs, w):
            return float(w) * nn.dssim(a, b, max_val=1.0, filter_size=fs).mean(dim=[1, 2, 3])

        def mse(a, b, w):
            return float(w) * ((a - b) ** 2).mean(dim=[1, 2, 3])

        def l1(a, b, w):
            return float(w) * (a - b).abs().mean(dim=[1, 2, 3])

        fs1 = max(1, int(self.resolution / 11.6))
        fs2 = max(1, int(self.resolution / 23.2))

        if self.resolution < 256:
            src_loss = dssim_loss(target_src_masked_opt, pred_src_src_masked_opt, fs1, 10)
            dst_loss = dssim_loss(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1, 10)
        else:
            src_loss = (dssim_loss(target_src_masked_opt, pred_src_src_masked_opt, fs1, 5)
                        + dssim_loss(target_src_masked_opt, pred_src_src_masked_opt, fs2, 5))
            dst_loss = (dssim_loss(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1, 5)
                        + dssim_loss(target_dst_masked_opt, pred_dst_dst_masked_opt, fs2, 5))

        src_loss = src_loss + mse(target_src_masked_opt, pred_src_src_masked_opt, 10)
        dst_loss = dst_loss + mse(target_dst_masked_opt, pred_dst_dst_masked_opt, 10)

        if self.eyes_mouth_prio:
            src_loss = src_loss + l1(target_src * target_srcm_em, pred_src_src * target_srcm_em, 300)
            dst_loss = dst_loss + l1(target_dst * target_dstm_em, pred_dst_dst * target_dstm_em, 300)

        src_loss = src_loss + mse(target_srcm, pred_src_srcm, 10)
        dst_loss = dst_loss + mse(target_dstm, pred_dst_dstm, 10)

        # style losses
        face_style_power = float(self.options.get('face_style_power', 0.0)) / 100.0
        bg_style_power = float(self.options.get('bg_style_power', 0.0)) / 100.0
        extra_style_loss = torch.tensor(0.0, device=self.device)

        if face_style_power != 0.0 and not self.pretrain:
            extra_style_loss = extra_style_loss + nn.style_loss(
                pred_src_dst_no_code_grad * pred_src_dstm.detach(),
                pred_dst_dst.detach() * pred_dst_dstm.detach(),
                gaussian_blur_radius=self.resolution // 8,
                loss_weight=10000.0 * face_style_power,
            )

        if bg_style_power != 0.0 and not self.pretrain:
            target_dst_style_anti_masked = target_dst * style_mask_anti_blur
            psd_style_anti_masked = pred_src_dst * style_mask_anti_blur
            extra_style_loss = extra_style_loss + (
                10.0 * bg_style_power
                * nn.dssim(psd_style_anti_masked, target_dst_style_anti_masked,
                           max_val=1.0, filter_size=fs1).mean())
            extra_style_loss = extra_style_loss + (
                (10.0 * bg_style_power)
                * ((psd_style_anti_masked - target_dst_style_anti_masked) ** 2).mean())

        extra_masked_gan_loss = torch.tensor(0.0, device=self.device)
        if self.masked_training and self.gan_power != 0.0:
            extra_masked_gan_loss = extra_masked_gan_loss + 0.000001 * nn.total_variation_mse(pred_src_src)
            extra_masked_gan_loss = extra_masked_gan_loss + 0.02 * (
                (pred_src_src_anti_masked - target_src_anti_masked) ** 2).mean()

        G_loss = src_loss.mean() + dst_loss.mean() + extra_style_loss + extra_masked_gan_loss
        self._last_loss_per_sample = (src_loss + dst_loss).detach().cpu().tolist()

        # GAN
        D_gan_loss = None
        if self.gan_power != 0.0:
            target_src_masked_opt_gan = target_src * target_srcm if self.masked_training else target_src
            pred_src_src_masked_opt_gan = pred_src_src * target_srcm if self.masked_training else pred_src_src

            pred_d1, pred_d2 = self.D_src(pred_src_src_masked_opt_gan)
            tgt_d1, tgt_d2 = self.D_src(target_src_masked_opt_gan)

            ones1 = torch.ones_like(tgt_d1)
            zeros1 = torch.zeros_like(pred_d1)
            ones2 = torch.ones_like(tgt_d2)
            zeros2 = torch.zeros_like(pred_d2)

            D_gan_loss = (0.5 * (F.binary_cross_entropy_with_logits(tgt_d1, ones1)
                                 + F.binary_cross_entropy_with_logits(pred_d1.detach(), zeros1))
                          + 0.5 * (F.binary_cross_entropy_with_logits(tgt_d2, ones2)
                                   + F.binary_cross_entropy_with_logits(pred_d2.detach(), zeros2)))

            G_loss = G_loss + self.gan_power * (
                F.binary_cross_entropy_with_logits(pred_d1, torch.ones_like(pred_d1))
                + F.binary_cross_entropy_with_logits(pred_d2, torch.ones_like(pred_d2)))

        # Backward
        self.optimizer.zero_grad()
        G_loss.backward()

        # Gradient clipping (防模型崩溃)
        if bool(self.options.get('clipgrad', False)):
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        self.optimizer.step()

        if D_gan_loss is not None:
            self.D_optimizer.zero_grad()
            D_gan_loss.backward()
            self.D_optimizer.step()

        return src_loss.detach().mean().item(), dst_loss.detach().mean().item()

    def onTrainOneIter(self):
        ((warped_src, target_src, target_srcm, target_srcm_em),
         (warped_dst, target_dst, target_dstm, target_dstm_em)) = self.generate_next_samples()

        src_loss, dst_loss = self.train_one_step(
            warped_src, target_src, target_srcm, target_srcm_em,
            warped_dst, target_dst, target_dstm, target_dstm_em,
        )

        # Step LR scheduler (余弦退火)
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

        return (
            fw['pred_src_src'].detach().cpu().float().numpy(),
            fw['pred_dst_dst'].detach().cpu().float().numpy(),
            fw['pred_dst_dstm'].detach().cpu().float().numpy(),
            fw['pred_src_dst'].detach().cpu().float().numpy(),
            fw['pred_src_dstm'].detach().cpu().float().numpy(),
        )

    def onGetPreview(self, samples, for_history=False):
        # Deep copy to break generator's memory reuse (view 链导致数据被后续批次覆盖)
        ((warped_src, target_src, target_srcm, target_srcm_em),
         (warped_dst, target_dst, target_dstm, target_dstm_em)) = copy.deepcopy(samples)

        S, D, SS, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0)
            for x in ([target_src, target_dst] + list(self.AE_view(target_src, target_dst)))
        ]

        tgt_srcm_nhwc = nn.to_data_format(target_srcm, 'NHWC', self.model_data_format).copy()
        tgt_dstm_nhwc = nn.to_data_format(target_dstm, 'NHWC', self.model_data_format).copy()
        n_samples = min(4, self.get_batch_size())

        ddm_1ch = DDM.copy()
        sdm_1ch = SDM.copy()
        DDM = np.repeat(DDM, 3, axis=-1)
        SDM = np.repeat(SDM, 3, axis=-1)
        pad_to = 4  # Trainer assumes 4 rows; pad if less

        def _pad_rows(img, n):
            """Pad image to n rows by repeating the last row if needed."""
            if img.shape[0] >= n:
                return img[:n]
            pad = np.repeat(img[-1:], n - img.shape[0], axis=0)
            return np.concatenate([img, pad], axis=0)
        WS = np.clip(nn.to_data_format(warped_src, 'NHWC', self.model_data_format), 0.0, 1.0)
        WD = np.clip(nn.to_data_format(warped_dst, 'NHWC', self.model_data_format), 0.0, 1.0)

        result = []

        # Tab 1: 原图预览
        st = []
        for i in range(n_samples):
            ar = (S[i], SS[i], D[i], DD[i], SD[i])
            st.append(np.concatenate(ar, axis=1))
        result.append(('原图预览', np.concatenate(st, axis=0)))

        # Tab 2: 遮罩下
        st = []
        for i in range(n_samples):
            SD_mask = DDM[i] * SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
            ar = (S[i] * tgt_srcm_nhwc[i], SS[i],
                  D[i] * tgt_dstm_nhwc[i], DD[i] * DDM[i], SD[i] * SD_mask)
            st.append(np.concatenate(ar, axis=1))
        result.append(('遮罩下', np.concatenate(st, axis=0)))

        # Tab 3: 原始输入
        st = []
        for i in range(n_samples):
            ar = (WS[i], SS[i], WD[i], DD[i], SD[i])
            st.append(np.concatenate(ar, axis=1))
        result.append(('原始输入', np.concatenate(st, axis=0)))

        # Tab 4: 合并预览
        st = []
        for i in range(n_samples):
            if self.face_type < FaceType.HEAD:
                dst_merge_mask = tgt_dstm_nhwc[i] * ddm_1ch[i]
                src_merge_mask = tgt_srcm_nhwc[i] * sdm_1ch[i]
            else:
                dst_merge_mask = tgt_dstm_nhwc[i]
                src_merge_mask = tgt_srcm_nhwc[i]
            ss_composite = SS[i] * src_merge_mask + S[i] * (1.0 - src_merge_mask)
            dd_composite = DD[i] * dst_merge_mask + D[i] * (1.0 - dst_merge_mask)
            sd_composite = SD[i] * dst_merge_mask + D[i] * (1.0 - dst_merge_mask)
            ar = (S[i], ss_composite, D[i], dd_composite, sd_composite)
            st.append(np.concatenate(ar, axis=1))
        result.append(('合并预览', np.concatenate(st, axis=0)))

        self._preview_masks = {
            'col0': tgt_srcm_nhwc[:n_samples],
            'col2': tgt_dstm_nhwc[:n_samples],
            'col3': ddm_1ch[:n_samples],
            'col4': np.stack([
                ddm_1ch[j] * sdm_1ch[j] if self.face_type < FaceType.HEAD else sdm_1ch[j]
                for j in range(n_samples)
            ], axis=0),
        }

        return result

    # ---- Merge ----

    def AE_merge(self, warped_dst):
        warped_dst = self._np_to_torch(warped_dst)
        with torch.no_grad():
            dst_enc, _ = self.net.encoder(warped_dst)
            dst_lat, _ = self.net.bottleneck(dst_enc)
            pred_src_dst, pred_src_dstm, _ = self.net.decoderA(dst_lat)
            _, pred_dst_dstm, _ = self.net.decoderB(dst_lat)

        return (
            pred_src_dst.detach().cpu().numpy(),
            pred_dst_dstm.detach().cpu().numpy(),
            pred_src_dstm.detach().cpu().numpy(),
        )

    def predictor_func(self, face=None):
        face = nn.to_data_format(face[None, ...], self.model_data_format, 'NHWC')
        bgr, mask_dst_dstm, mask_src_dstm = [
            nn.to_data_format(x, 'NHWC', self.model_data_format).astype(np.float32)
            for x in self.AE_merge(face)
        ]
        bgr0, ms0, md0 = bgr[0], mask_src_dstm[0, ..., 0], mask_dst_dstm[0, ..., 0]
        res = self.resolution
        nx = float(self.options.get('output_shift_x', 0.100))
        ny = float(self.options.get('output_shift_y', 0.100))
        dx = nx * res / 2.0
        dy = ny * res / 2.0
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            import cv2
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            bgr0 = cv2.warpAffine(bgr0, M, (res, res), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            ms0 = cv2.warpAffine(ms0, M, (res, res), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            md0 = cv2.warpAffine(md0, M, (res, res), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return bgr0, ms0, md0

    def get_MergerConfig(self):
        import merger
        return (
            self.predictor_func,
            (self.options['resolution'], self.options['resolution'], 3),
            merger.MergerConfigMasked(face_type=self.face_type, default_mode='overlay'),
        )


Model = DeepFakeLargeModel
