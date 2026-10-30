"""XSegLite Model — PyTorch training script (standard torch, torch.compile ready).

XSegLite: Ghost Attention XSeg binary segmentation model.
Uses GhostV2 blocks + ECA attention for efficient face mask prediction.

Saved model naming:
    - XSegLite_data.dat       (options/iter)
    - XSegLite_256.pth         (network + optimizer state)
"""

import math
import multiprocessing
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop as TorchRMSprop

# 抑制 torch.compile 缓存的 cubin 文件缺失警告（换机器/temp 目录变化时常见）
warnings.filterwarnings('ignore', message='.*Cubin file saved by TritonBundler.*')
warnings.filterwarnings('ignore', message='.*Failed to reload cubin file.*')

from core.interact import interact as io
from core.leras import nn          # still needed for dssim, to_data_format
from core.xseglite_torch import XSegLiteTorch
from facelib import FaceType
from models import ModelBase
from samplelib import SampleGeneratorFace, SampleGeneratorV2, SampleProcessor, SampleLoaderV4


class XSegLiteModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_model_class_name='XSegLite', **kwargs)

    # override
    def on_initialize_options(self):
        ask_override = self.ask_override()

        if not self.is_first_run() and ask_override:
            if io.input_bool(
                "Re-start training?",
                False,
                help_message="Reset model weights and start from scratch.",
            ):
                self.set_iter(0)

        default_face_type = self.options['face_type'] = self.load_or_def_option('face_type', 'wf')
        default_pretrain = self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False
        default_use_eca = self.options['use_eca'] = self.load_or_def_option('use_eca', True)
        default_use_bf16 = self.options['use_bf16'] = self.load_or_def_option('use_bf16', False)
        default_loader_skip = self.options['loader_skip'] = self.load_or_def_option('loader_skip', False)
        default_use_compile = self.options['use_compile'] = self.load_or_def_option('use_compile', False)

        if self.is_first_run():
            self.options['face_type'] = io.input_str(
                "Face type",
                default_face_type,
                ['h', 'mf', 'f', 'wf', 'head'],
                help_message="half / mid-full / full / whole-face / head. Match your swap model.",
            ).lower()

        if self.is_first_run() or ask_override:
            self.options['resolution'] = io.input_int(
                "Resolution", 256,
                valid_range=[128, 1024],
                help_message="256 is standard. 512/1024 for production. Model is fully convolutional.",
            )
            self.ask_batch_size(4, range=None)
            # pretrain 已停用：强制 False，不再提供询问入口
            self.options['pretrain'] = False
            self.options['use_bf16'] = io.input_bool("Use BF16 mixed precision training", default_use_bf16)
            self.options['use_edge'] = io.input_bool(
                "Use edge-enhanced input (Sobel edge as 4th channel)", False,
                help_message="Adds Laplacian edge map as extra input channel. Better boundaries, +5% params.")
            self.options['loader_skip'] = io.input_bool(
                "Skip original loader (fast, no validation)", default_loader_skip,
                help_message="Yes = load all images instantly. No = original validated loader.")

        if not self.is_exporting and (self.options['pretrain'] and self.get_pretraining_data_path() is None):
            raise Exception("pretraining_data_path is not defined")

        self.pretrain_just_disabled = (default_pretrain is True and self.options['pretrain'] is False)

    # override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()

        self.model_data_format = 'NCHW'
        nn.initialize(device_config, data_format=self.model_data_format)
        self.device = nn.device

        self.resolution = resolution = int(self.load_or_def_option('resolution', 256))

        self.face_type = {
            'h': FaceType.HALF,
            'mf': FaceType.MID_FULL,
            'f': FaceType.FULL,
            'wf': FaceType.WHOLE_FACE,
            'head': FaceType.HEAD,
        }[self.options['face_type']]

        self.pretrain = bool(self.options['pretrain'])
        if self.pretrain_just_disabled:
            self.set_iter(0)

        use_eca = bool(self.options.get('use_eca', True))
        self.use_edge = bool(self.options.get('use_edge', False))
        in_ch = 4 if self.use_edge else 3

        # -- Build standard-torch model --
        self._model = XSegLiteTorch(in_ch=in_ch, base_ch=32, use_eca=use_eca).to(self.device)

        self._optimizer = TorchRMSprop(self._model.parameters(), lr=0.0001,
                                       alpha=0.99, eps=1e-8, weight_decay=0.0)
        # weight-decay via Adam-style decay; RMSprop in DFL applies lr_dropout per batch.
        # We approximate with a small weight_decay at 0.3 * lr.

        # -- Load weights --
        self._save_prefix = 'XSegLite'
        self._model_file = Path(self.get_model_root_path()) / f'{self._save_prefix}_{resolution}.pth'

        if self._model_file.exists() and not self.is_first_run():
            try:
                ckpt = torch.load(str(self._model_file), map_location=self.device)
                self._model.load_state_dict(ckpt['model'])
                self._optimizer.load_state_dict(ckpt['optimizer'])
                io.log_info(f'Loaded weights from {self._model_file.name}')
            except Exception as e:
                io.log_info(f'Failed to load weights: {e}, starting fresh')

        self._compiled = None
        use_compile = self.options.get('use_compile', False)
        if self.is_first_run():
            use_compile = io.input_bool("使用 torch.compile 加速? (首次需编译，后续会缓存)", use_compile)
            self.options['use_compile'] = use_compile
        if use_compile:
            io.log_info('正在编译模型，请稍候...')
            try:
                self._compiled = torch.compile(self._model, mode='default', dynamic=True)
                io.log_info('torch.compile 完成')
            except Exception:
                pass

        # -- Data generators --
        if self.is_training:
            cpu_count = min(multiprocessing.cpu_count(), 8)

            if self.pretrain:
                pretrain_gen = SampleGeneratorFace(
                    self.get_pretraining_data_path(),
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=True),
                    output_sample_types=[
                        {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                         'warp': True, 'transform': True,
                         'channel_type': SampleProcessor.ChannelType.BGR,
                         'face_type': self.face_type,
                         'data_format': nn.data_format, 'resolution': resolution},
                        {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                         'warp': True, 'transform': True,
                         'channel_type': SampleProcessor.ChannelType.G,
                         'face_type': self.face_type,
                         'data_format': nn.data_format, 'resolution': resolution},
                    ],
                    uniform_yaw_distribution=False,
                    generators_count=cpu_count,
                )
                self.set_training_data_generators([pretrain_gen])
            else:
                # V4 快速加载器 + V2 生成器（同 SAEHD 快速路径）
                cpu_count = min(multiprocessing.cpu_count(), 8)
                loader = SampleLoaderV4(
                    aligned_path=self.training_data_src_path,
                    batch_size=self.get_batch_size(),
                    resolution=resolution,
                )
                _opts = SampleProcessor.Options(random_flip=True)
                _out = [
                    {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                     'warp': True, 'transform': True,
                     'channel_type': SampleProcessor.ChannelType.BGR,
                     'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
                    {'sample_type': SampleProcessor.SampleType.FACE_MASK,
                     'warp': True, 'transform': True,
                     'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                     'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
                ]
                self.set_training_data_generators([
                    SampleGeneratorV2(
                        loader=loader,
                        sample_process_options=_opts,
                        output_sample_types=_out,
                        resolution=resolution,
                        debug=self.is_debug(),
                        xseg_augment=True,
                    ),
                ])

    # ---- Weight management ----
    def get_model_filename_list(self):
        res = str(self.resolution)
        return [[self, f'{self._save_prefix}_{res}.pth']]

    def onSave(self):
        ckpt = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'iter': self.get_iter(),
        }
        torch.save(ckpt, str(self._model_file))

    # ---- Forward helpers ----
    def _flow(self, x):
        x = self._add_edge(x)
        if self._compiled is not None:
            return self._compiled(x)
        return self._model(x)

    def _dice_loss(self, pred, y, smooth=1.0):
        pred_flat = pred.reshape(pred.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        intersection = (pred_flat * y_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + y_flat.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    # ---- Training step ----
    def _add_edge(self, x):
        """Append Sobel edge channel to NCHW input. x: (N,3,H,W) tensor on device."""
        if not self.use_edge:
            return x
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gy = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
        gx = torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        edge = torch.zeros_like(gray)
        edge[:, :, :-1] += gy; edge[:, :, 1:] += gy
        edge[:, :-1, :] += gx; edge[:, 1:, :] += gx
        edge = torch.clamp(edge, 0, 1).unsqueeze(1)
        return torch.cat([x, edge], dim=1)

    def _train_step(self, input_np: np.ndarray, target_np: np.ndarray) -> float:
        use_bf16 = bool(self.load_or_def_option('use_bf16', True)) and self.device.type == 'cuda'  # None（旧模型未存）时回退 True
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        x = torch.from_numpy(input_np).to(self.device, dtype=dtype)
        y = torch.from_numpy(target_np).to(self.device, dtype=dtype)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, pred = self._flow(self._add_edge(x))

        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        bce = torch.mean(bce, dim=[1, 2, 3])
        dice = self._dice_loss(pred, y)
        loss_per = 0.5 * bce + 0.5 * dice
        loss = torch.mean(loss_per)
        per_sample = loss_per.detach().cpu().tolist()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._last_loss_per_sample = per_sample
        return float(loss.item())

    def _view(self, input_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(input_np).to(self.device, dtype=torch.float32)
        x = self._add_edge(x)
        with torch.no_grad():
            _, pred = self._model(x)
        return pred.detach().cpu().numpy()

    # override
    def onTrainOneIter(self):
        image_np, target_np = self.generate_next_samples()[0]
        loss = self._train_step(image_np, target_np)
        return (('loss', loss),)

    # override
    def onGetPreview(self, samples, for_history=False):
        n_samples = min(self.get_batch_size(), 4)

        # Guard: no samples yet (before first training iteration)
        if not samples or not samples[0]:
            return []

        if self.pretrain:
            (srcdst_samples,) = samples
            image_np, mask_np = srcdst_samples
        else:
            srcdst_samples = samples[0]
            image_np, mask_np = srcdst_samples

        I = np.clip(nn.to_data_format(image_np, 'NHWC', self.model_data_format), 0.0, 1.0)
        M = np.clip(nn.to_data_format(mask_np, 'NHWC', self.model_data_format), 0.0, 1.0)
        IM = np.clip(nn.to_data_format(self._view(image_np), 'NHWC', self.model_data_format), 0.0, 1.0)

        M_3 = np.repeat(M, 3, axis=-1)
        IM_3 = np.repeat(IM, 3, axis=-1)

        green_bg = np.tile(
            np.array([0, 1, 0], dtype=np.float32)[None, None, ...],
            (self.resolution, self.resolution, 1),
        )

        # Mask data for GUI gradient compositing
        self._preview_masks = {
            'col3': M[:n_samples],
            'col4': IM[:n_samples],
        }

        # Filenames from generator
        src_fnames = (self.last_filenames[0] if getattr(self, 'last_filenames', None)
                      and len(self.last_filenames) > 0 else [])

        # Load original faces from disk for col1 & col6
        Raw = []
        for i in range(n_samples):
            raw_face = I[i].copy()
            if i < len(src_fnames) and src_fnames[i]:
                fname = src_fnames[i]
                if not Path(fname).exists():
                    fname = str(Path(self.training_data_src_path) / fname)
                orig = cv2.imread(str(fname))
                if orig is not None:
                    orig = cv2.resize(orig, (self.resolution, self.resolution))
                    raw_face = orig.astype(np.float32) / 255.0
            Raw.append(raw_face)

        # Run inference on original faces for column 6
        Raw_np = np.zeros((n_samples, 3, self.resolution, self.resolution), dtype=np.float32)
        for i in range(n_samples):
            Raw_np[i] = np.transpose(Raw[i], (2, 0, 1))
        IM_raw = np.clip(nn.to_data_format(self._view(Raw_np), 'NHWC', self.model_data_format), 0.0, 1.0)
        IM_raw_3 = np.repeat(IM_raw, 3, axis=-1)

        # Per-sample loss values
        psl = getattr(self, '_last_loss_per_sample', [0.0] * n_samples)

        result = []
        st = []
        for i in range(n_samples):
            col1 = Raw[i]
            col2 = I[i]
            col3 = M_3[i]
            col4 = IM_3[i]

            # Col 5: pred mask on warped input
            col5 = I[i] * IM_3[i] + 0.5 * I[i] * (1 - IM_3[i]) + 0.5 * green_bg * (1 - IM_3[i])

            # Col 6: pred mask on original face
            col6 = Raw[i] * IM_raw_3[i] + 0.5 * Raw[i] * (1 - IM_raw_3[i]) + 0.5 * green_bg * (1 - IM_raw_3[i])

            st.append(np.concatenate([col1, col2, col3, col4, col5, col6], axis=1))
        result += [('XSegLite training faces', np.concatenate(st, axis=0))]

        return result


Model = XSegLiteModel
