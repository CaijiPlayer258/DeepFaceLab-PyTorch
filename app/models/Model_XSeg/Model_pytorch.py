"""XSeg model - PyTorch implementation.

Restores DeepFaceLab XSeg training (pretrain + segmentation) without TensorFlow.

Saved model naming:
    - XSeg_data.dat (options/iter)
    - XSeg_256.pth (network weights)
    - XSeg_256_opt.pth (optimizer state)
"""

import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType, XSegNet
from models import ModelBase
from samplelib import SampleGeneratorFace, SampleGeneratorFaceXSeg, SampleProcessor


class XSegModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, force_model_class_name='XSeg', **kwargs)

    # override
    def on_initialize_options(self):
        ask_override = self.ask_override()

        if not self.is_first_run() and ask_override:
            if io.input_bool(
                "重新开始训练？",
                False,
                help_message="重置模型权重并从头开始训练。",
            ):
                self.set_iter(0)

        default_face_type = self.options['face_type'] = self.load_or_def_option('face_type', 'wf')
        default_pretrain = self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False

        if self.is_first_run():
            self.options['face_type'] = io.input_str(
                "人脸类型",
                default_face_type,
                ['h', 'mf', 'f', 'wf', 'head'],
                help_message="Half / mid face / full face / whole face / head。请与深度换脸模型的 face_type 保持一致。",
            ).lower()

        if self.is_first_run() or ask_override:
            self.ask_batch_size(4, range=[2, 16])
            # pretrain 已停用：强制 False，不再提供询问入口
            self.options['pretrain'] = False

        if not self.is_exporting and (self.options['pretrain'] and self.get_pretraining_data_path() is None):
            raise Exception("未定义 pretraining_data_path")

        self.pretrain_just_disabled = (default_pretrain is True and self.options['pretrain'] is False)

    # override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()

        # PyTorch training uses NCHW.
        self.model_data_format = 'NCHW'
        nn.initialize(device_config, data_format=self.model_data_format)
        self.device = nn.device

        self.resolution = resolution = 256

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

        # Build XSeg network + optimizer wrapper.
        self.model = XSegNet(
            name='XSeg',
            resolution=resolution,
            load_weights=not self.is_first_run(),
            weights_file_root=self.get_model_root_path(),
            training=True,
            place_model_on_cpu=(self.device.type == 'cpu'),
            optimizer=nn.RMSprop(lr=0.0001, lr_dropout=0.3, name='opt'),
            data_format=nn.data_format,
        )

        # Initialize sample generators.
        if self.is_training:
            cpu_count = min(multiprocessing.cpu_count(), 8)

            if self.pretrain:
                pretrain_gen = SampleGeneratorFace(
                    self.get_pretraining_data_path(),
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=True),
                    output_sample_types=[
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': True,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution,
                        },
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': True,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.G,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution,
                        },
                    ],
                    uniform_yaw_distribution=False,
                    generators_count=cpu_count,
                )
                self.set_training_data_generators([pretrain_gen])
            else:
                src_dst_generators_count = max(1, cpu_count // 2)
                src_generators_count = max(1, cpu_count // 2)
                dst_generators_count = max(1, cpu_count // 2)

                srcdst_generator = SampleGeneratorFaceXSeg(
                    [self.training_data_src_path, self.training_data_dst_path],
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    resolution=resolution,
                    face_type=self.face_type,
                    generators_count=src_dst_generators_count,
                    data_format=nn.data_format,
                )

                src_generator = SampleGeneratorFace(
                    self.training_data_src_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=False),
                    output_sample_types=[
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': False,
                            'transform': False,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'border_replicate': False,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution,
                        },
                    ],
                    generators_count=src_generators_count,
                    raise_on_no_data=False,
                )

                dst_generator = SampleGeneratorFace(
                    self.training_data_dst_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(random_flip=False),
                    output_sample_types=[
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': False,
                            'transform': False,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'border_replicate': False,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution,
                        },
                    ],
                    generators_count=dst_generators_count,
                    raise_on_no_data=False,
                )

                self.set_training_data_generators([srcdst_generator, src_generator, dst_generator])

    # override
    def get_model_filename_list(self):
        return self.model.model_filename_list

    # override
    def onSave(self):
        self.model.save_weights()

    def _train_step(self, input_np: np.ndarray, target_np: np.ndarray) -> float:
        x = torch.from_numpy(input_np).to(self.device, dtype=torch.float32)
        y = torch.from_numpy(target_np).to(self.device, dtype=torch.float32)

        logits, pred = self.model.flow(x, pretrain=self.pretrain)

        if self.pretrain:
            # Structural + pixel reconstruction losses (match TF formulas).
            fs1 = int(self.resolution / 11.6)
            fs2 = int(self.resolution / 23.2)
            d1 = nn.dssim(y, pred, max_val=1.0, filter_size=fs1).mean(dim=[1, 2, 3])
            d2 = nn.dssim(y, pred, max_val=1.0, filter_size=fs2).mean(dim=[1, 2, 3])
            mse = torch.mean((y - pred) ** 2, dim=[1, 2, 3])
            loss_per = 5.0 * d1 + 5.0 * d2 + 10.0 * mse
        else:
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss_per = torch.mean(bce, dim=[1, 2, 3])

        loss = torch.mean(loss_per)

        self.model.opt.zero_grad()
        loss.backward()
        self.model.opt.step()

        return float(loss.item())

    def _view(self, input_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(input_np).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            _, pred = self.model.flow(x, pretrain=self.pretrain)
        return pred.detach().to('cpu').numpy()

    # override
    def onTrainOneIter(self):
        image_np, target_np = self.generate_next_samples()[0]
        loss = self._train_step(image_np, target_np)
        return (('loss', loss),)

    # override
    def onGetPreview(self, samples, for_history=False):
        n_samples = self.get_batch_size()

        if self.pretrain:
            (srcdst_samples,) = samples
            image_np, mask_np = srcdst_samples
        else:
            srcdst_samples, src_samples, dst_samples = samples
            image_np, mask_np = srcdst_samples

        I = np.clip(nn.to_data_format(image_np, 'NHWC', self.model_data_format), 0.0, 1.0)
        M = np.clip(nn.to_data_format(mask_np, 'NHWC', self.model_data_format), 0.0, 1.0)
        IM = np.clip(nn.to_data_format(self._view(image_np), 'NHWC', self.model_data_format), 0.0, 1.0)

        M = np.repeat(M, 3, axis=-1)
        IM = np.repeat(IM, 3, axis=-1)

        green_bg = np.tile(np.array([0, 1, 0], dtype=np.float32)[None, None, ...], (self.resolution, self.resolution, 1))

        result = []
        st = []
        for i in range(n_samples):
            if self.pretrain:
                ar = (I[i], IM[i])
            else:
                ar = (
                    I[i] * M[i] + 0.5 * I[i] * (1 - M[i]) + 0.5 * green_bg * (1 - M[i]),
                    IM[i],
                    I[i] * IM[i] + 0.5 * I[i] * (1 - IM[i]) + 0.5 * green_bg * (1 - IM[i]),
                )
            st.append(np.concatenate(ar, axis=1))
        result += [('XSeg training faces', np.concatenate(st, axis=0))]

        if not self.pretrain and len(src_samples) != 0:
            (src_np,) = src_samples
            D = np.clip(nn.to_data_format(src_np, 'NHWC', self.model_data_format), 0.0, 1.0)
            DM = np.clip(nn.to_data_format(self._view(src_np), 'NHWC', self.model_data_format), 0.0, 1.0)
            DM = np.repeat(DM, 3, axis=-1)

            st = []
            for i in range(n_samples):
                ar = (D[i], DM[i], D[i] * DM[i] + 0.5 * D[i] * (1 - DM[i]) + 0.5 * green_bg * (1 - DM[i]))
                st.append(np.concatenate(ar, axis=1))
            result += [('XSeg src faces', np.concatenate(st, axis=0))]

        if not self.pretrain and len(dst_samples) != 0:
            (dst_np,) = dst_samples
            D = np.clip(nn.to_data_format(dst_np, 'NHWC', self.model_data_format), 0.0, 1.0)
            DM = np.clip(nn.to_data_format(self._view(dst_np), 'NHWC', self.model_data_format), 0.0, 1.0)
            DM = np.repeat(DM, 3, axis=-1)

            st = []
            for i in range(n_samples):
                ar = (D[i], DM[i], D[i] * DM[i] + 0.5 * D[i] * (1 - DM[i]) + 0.5 * green_bg * (1 - DM[i]))
                st.append(np.concatenate(ar, axis=1))
            result += [('XSeg dst faces', np.concatenate(st, axis=0))]

        return result

    def export_dfm(self):
        """Export XSeg to ONNX for DeepFaceLive usage (matches original DFL behavior).

        Original TF version exports `model.onnx` with:
          input:  in_face  (NHWC, float32)
          output: out_mask (NHWC, float32)
        """

        output_path = self.get_strpath_storage_for_file('model.onnx')
        io.log_info(f'Dumping .onnx to {output_path}')

        class _XSegONNXWrapper(torch.nn.Module):
            def __init__(self, xseg_module: torch.nn.Module):
                super().__init__()
                self.xseg = xseg_module

            def forward(self, in_face: torch.Tensor) -> torch.Tensor:
                # NHWC -> NCHW
                x = in_face.permute(0, 3, 1, 2).contiguous()
                _, pred = self.xseg(x, pretrain=False)
                # NCHW -> NHWC
                return pred.permute(0, 2, 3, 1).contiguous()

        # Export on CPU for maximum compatibility.
        wrapper = _XSegONNXWrapper(self.model.model)
        wrapper.eval()
        wrapper_cpu = wrapper.to('cpu')

        dummy_face = torch.zeros(1, self.resolution, self.resolution, 3, dtype=torch.float32)
        with torch.no_grad():
            _ = wrapper_cpu(dummy_face)

        # 原版 DFL(tf2onnx) 导出的 XSeg ONNX 使用 ':0' 张量名：
        #   input_names=['in_face:0']
        #   output_names=['out_mask:0']
        # 为了保持 DeepFaceLive 侧加载一致，这里仅对齐命名。
        export_kwargs = dict(
            input_names=['in_face:0'],
            output_names=['out_mask:0'],
            dynamic_axes={
                'in_face:0': {0: 'batch'},
                'out_mask:0': {0: 'batch'},
            },
            opset_version=13,
        )

        # Force legacy exporter when available (Torch 2.5+ default may use dynamo).
        try:
            torch.onnx.export(
                wrapper_cpu,
                dummy_face,
                output_path,
                dynamo=False,
                **export_kwargs,
            )
        except TypeError:
            torch.onnx.export(
                wrapper_cpu,
                dummy_face,
                output_path,
                **export_kwargs,
            )


Model = XSegModel
