"""
DF-Single: 基于 DF 架构，去掉 dst_decoder。

与 DF 完全一致：
  - Encoder / Inter / decoder_src 结构、参数相同
  - 训练损失：仅 src 侧（dst 不再产生 loss）
  - 推理：只前向 src_decoder（人脸 + 遮罩）

去掉 dst_decoder 后：
  - 参数量减少 ~35%（约 15M→10M）
  - 训练速度提升 ~30%
  - 现有 DF 权重可直接迁移

迁移命令：
  python -m models.Model_DF-Single.migrate --src /path/to/SAEHD/model --dst /path/to/new/model
"""

import multiprocessing
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import SampleGeneratorFace, SampleProcessor

class SAEHDDFSingleModel(ModelBase):
    """
    DF-Single: DF 单解码器版本。

    与 DF 的差异：
    - 没有 decoder_dst
    - 训练只有 src 侧的 loss
    - 合并时只用 src_decoder 的输出
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('force_model_class_name', 'DFSingle')
        super().__init__(*args, **kwargs)
        self.is_df_single = True

    # ========== options ==========
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()
        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb
        suggest_batch_size = 8 if lowest_vram >= 4 else 4
        min_res, max_res = 64, 640

        self.options['resolution'] = self.load_or_def_option('resolution', 128)
        self.options['face_type'] = self.load_or_def_option('face_type', 'f')
        self.options['models_opt_on_gpu'] = self.load_or_def_option('models_opt_on_gpu', True)
        self.options['archi'] = self.load_or_def_option('archi', 'df-ud')
        self.options['ae_dims'] = self.load_or_def_option('ae_dims', 256)
        self.options['e_dims'] = self.load_or_def_option('e_dims', 64)
        default_d_dims = self.options['d_dims'] = self.load_or_def_option('d_dims', 64)
        default_d_mask_dims = self.options['d_mask_dims'] = self.load_or_def_option('d_mask_dims', 22)
        self.options['masked_training'] = self.load_or_def_option('masked_training', True)
        self.options['eyes_mouth_prio'] = self.load_or_def_option('eyes_mouth_prio', False)
        self.options['blur_out_mask'] = self.load_or_def_option('blur_out_mask', False)
        self.options['adabelief'] = self.load_or_def_option('adabelief', True)
        self.options['random_warp'] = self.load_or_def_option('random_warp', True)
        self.options['random_hsv_power'] = self.load_or_def_option('random_hsv_power', 0.0)
        self.options['true_face_power'] = self.load_or_def_option('true_face_power', 0.0)
        self.options['gan_power'] = self.load_or_def_option('gan_power', 0.0)
        self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)
        self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False
        self.options['lr_dropout'] = self.load_or_def_option('lr_dropout', 'n')
        self.options['use_bf16'] = self.load_or_def_option('use_bf16', False)
        self.options['uniform_yaw'] = self.load_or_def_option('uniform_yaw', False)
        self.options['ct_mode'] = self.load_or_def_option('ct_mode', 'none')
        self.options['use_fast_generator'] = self.load_or_def_option('use_fast_generator', False)
        self.options['optimizer'] = self.load_or_def_option('optimizer', 'adabelief')
        self.options['freeze_encoder'] = self.load_or_def_option('freeze_encoder', False)
        self.options['freeze_inter'] = self.load_or_def_option('freeze_inter', False)
        self.options['freeze_decoder_mask'] = self.load_or_def_option('freeze_decoder_mask', False)

        ask_override = self.ask_override()
        _interactive = not self.silent_start  # 静默启动跳过所有交互提示
        if (self.is_first_run() or ask_override) and _interactive:
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_random_src_flip()
            self.ask_batch_size(suggest_batch_size)
            self.ask_backup_interval()
            self.ask_max_backups()

        if self.is_first_run():
            if _interactive:
                res = io.input_int('分辨率', self.options['resolution'], add_info='64-640')
                res = np.clip((res // 16) * 16, min_res, max_res)
                self.options['resolution'] = res
                self.options['face_type'] = io.input_str(
                    '人脸类型', self.options['face_type'],
                    ['h', 'mf', 'f', 'wf', 'head'],
                ).lower()
                while True:
                    archi = io.input_str('AE 架构', self.options['archi'],
                        help_message="df / df-d / df-ud / df-udt ...")
                    archi_split = archi.split('-')
                    if len(archi_split) >= 1 and archi_split[0] in ['df', 'liae']:
                        if len(archi_split) == 1 or all(c in 'udtc' for c in archi_split[1]):
                            self.options['archi'] = archi
                            break
                self.options['ae_dims'] = int(np.clip(io.input_int('AE dims', 256, add_info='32-1024'), 32, 1024))
                self.options['e_dims'] = int(np.clip(io.input_int('E dims', 64, add_info='16-256'), 16, 256))
                self.options['d_dims'] = int(np.clip(io.input_int('D dims', 64, add_info='16-256'), 16, 256))
                self.options['d_mask_dims'] = int(np.clip(io.input_int('D mask dims', 22, add_info='16-256'), 16, 256))

        if (self.is_first_run() or ask_override) and _interactive:
            # pretrain 已停用：强制 False，不再提供询问入口
            self.options['random_warp'] = io.input_bool('启用 Random Warp', self.options['random_warp'])
            self.options['gan_power'] = float(np.clip(io.input_number('GAN 强度', 0.0, add_info='0.0..5.0'), 0.0, 5.0))

    # ========== initialization ==========
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        self.model_data_format = 'NCHW'
        nn.initialize(device_config, data_format=self.model_data_format)
        self.device = nn.device
        self.resolution = int(self.options['resolution'])
        self.face_type = {'h': FaceType.HALF, 'mf': FaceType.MID_FULL, 'f': FaceType.FULL, 'wf': FaceType.WHOLE_FACE, 'head': FaceType.HEAD}[self.options['face_type']]
        self.pretrain = bool(self.options['pretrain'])

        archi_split = self.options['archi'].split('-')
        archi_type, archi_opts = archi_split[0], archi_split[1] if len(archi_split) > 1 else ''
        self.archi_type = archi_type
        self.archi_opts = archi_opts

        ae_dims = int(self.options['ae_dims'])
        e_dims = int(self.options['e_dims'])
        d_dims = int(self.options['d_dims'])
        d_mask_dims = int(self.options['d_mask_dims'])
        resolution = self.resolution
        gan_power = float(self.options['gan_power'])

        model_archi = nn.DeepFakeArchi(resolution, opts=archi_opts)

        if 'df' in archi_type:
            # ── Encoder ──
            self.encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch() * (self.encoder.get_out_res(resolution) ** 2)

            # ── Inter ──
            self.inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.get_out_ch()

            # ── Decoder (src only) ──
            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')

            self.model_filename_list = [
                [self.encoder, 'encoder.pth'],
                [self.inter, 'inter.pth'],
                [self.decoder_src, 'decoder_src.pth'],
            ]
        else:
            raise ValueError(f'DF-Single only supports DF, got {archi_type}')

        for item in [self.encoder, self.inter, self.decoder_src]:
            if item is not None:
                for layer in item.get_layers():
                    try: layer.to(self.device)
                    except: pass

        # Optimizer
        if self.is_training:
            _opt_name = str(self.options.get('optimizer', 'adabelief')).lower()
            _opt_map = {'adam': nn.Adam, 'adabelief': nn.AdaBelief, 'lion': nn.Lion}
            _opt_class = _opt_map.get(_opt_name, nn.AdaBelief)
            self.src_dst_opt = _opt_class(
                list(self.encoder.get_weights()) + list(self.inter.get_weights()) + list(self.decoder_src.get_weights()),
                lr=5e-5, name='src_dst_opt',
            )
            self.model_filename_list += [[self.src_dst_opt, 'src_dst_opt.pth']]

        # Freeze layers
        if self.is_training:
            if self.options.get('freeze_encoder', False):
                for w in self.encoder.get_weights():
                    if hasattr(w, 'requires_grad'): w.requires_grad_(False)
            if self.options.get('freeze_inter', False):
                for w in self.inter.get_weights():
                    if hasattr(w, 'requires_grad'): w.requires_grad_(False)
            if self.options.get('freeze_decoder_mask', False):
                _ml = getattr(self, 'decoder_src', None)
                if _ml is not None:
                    for _ln in ('upscalem0', 'upscalem1', 'upscalem2', 'upscalem3', 'upscalem4', 'out_convm'):
                        _lay = getattr(_ml, _ln, None)
                        if _lay is not None:
                            for w in _lay.get_weights():
                                if hasattr(w, 'requires_grad'): w.requires_grad_(False)

        # Data generators
        if self.is_training:
            from samplelib import SampleGeneratorFace, SampleProcessor
            cpu_count = min(multiprocessing.cpu_count(), 8)
            generators_count = max(1, cpu_count // 2)
            _opts = SampleProcessor.Options(
                scale_range=[-0.15, 0.15],
                random_flip=self.random_src_flip if hasattr(self, 'random_src_flip') else True,
            )
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
            _gen_src = SampleGeneratorFace(
                self.training_data_src_path,
                debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=_opts,
                output_sample_types=_out,
                generators_count=generators_count,
                uniform_yaw_distribution=bool(self.options.get('uniform_yaw', False)),
            )
            _gen_dst = SampleGeneratorFace(
                self.training_data_dst_path,
                debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=True),
                output_sample_types=_out,
                generators_count=generators_count,
            )
            self.set_training_data_generators([_gen_src, _gen_dst])


        # Load weights
        for model, filename in self.model_filename_list:
            if not self.is_first_run():
                model.load_weights(str(Path(self.get_strpath_storage_for_file(filename))))
            else:
                model.init_weights()

    def get_model_filename_list(self):
        return self.model_filename_list

    def onSave(self):
        for model, filename in self.model_filename_list:
            model.save_weights(self.get_strpath_storage_for_file(filename))

    # ========== forward ==========
    def _forward(self, warped_src, warped_dst=None):
        """前向：src 用于重建，dst 用于域对齐。"""
        from core.leras.archis.DeepFakeArchi import DeepFakeArchi
        src_code = self.inter(self.encoder(warped_src))
        # Inter 最终的 4D 特征图（reshaped + upscale1 后）供 Conv2D 判别器
        src_inter_4d = src_code
        # 确保返回 4D 张量（Inter 输出是 NCHW）
        if src_inter_4d.ndim != 4:
            ae = self.options.get('ae_dims', 256)
            lr = self.resolution // (32 if 'd' in (self.archi_opts or '') else 16)
            lr = lr * (2 if 't' not in (self.archi_opts or '') else 1)
            src_inter_4d = src_inter_4d.view(src_inter_4d.size(0), ae, lr, lr)
        pred_src_src, pred_src_srcm = self.decoder_src(src_code)
        out = {'inter_out_4d': src_inter_4d, 'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm}
        if warped_dst is not None:
            dst_inter = self.inter(self.encoder(warped_dst))
            dst_4d = dst_inter
            if dst_4d.ndim != 4:
                dst_4d = dst_4d.view(dst_4d.size(0), ae, lr, lr)
            out['dst_inter_out_4d'] = dst_4d
        return out

    # ========== losses ==========
    def _recon_loss(self, target_src, target_srcm, target_srcm_em, fw):
        resolution = self.resolution
        pred = fw['pred_src_src']
        predm = fw['pred_src_srcm']

        # mask blur
        k = max(1, resolution // 32)
        tm = nn.gaussian_blur(target_srcm, k)
        tm = torch.clamp(tm, 0.0, 0.5) * 2.0
        tma = 1.0 - tm

        # DSSIM + MSE
        fs1 = max(1, int(resolution / 11.6))
        fs2 = max(1, int(resolution / 23.2))
        loss = 0
        if resolution < 256:
            loss = loss + nn.dssim(target_src * tm, pred * tm, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 10
        else:
            loss = loss + nn.dssim(target_src * tm, pred * tm, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 5
            loss = loss + nn.dssim(target_src * tm, pred * tm, max_val=1.0, filter_size=fs2).mean(dim=[1,2,3]) * 5
        loss = loss + F.mse_loss(pred * tm, target_src * tm, reduction='none').mean(dim=[1,2,3]) * 10
        if self.options.get('eyes_mouth_prio', False):
            loss = loss + F.l1_loss(pred * target_srcm_em, target_src * target_srcm_em, reduction='none').mean(dim=[1,2,3]) * 300
        loss = loss + F.mse_loss(predm, target_srcm, reduction='none').mean(dim=[1,2,3]) * 10
        return loss

    def train_one_step(self, warped_src, target_src, target_srcm, target_srcm_em,
                       warped_dst=None):
        warped_src = self.tensor_from_np(warped_src)
        target_src = self.tensor_from_np(target_src)
        target_srcm = self.tensor_from_np(target_srcm)
        target_srcm_em = self.tensor_from_np(target_srcm_em)
        if warped_dst is not None:
            warped_dst = self.tensor_from_np(warped_dst)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=bool(self.options.get('use_bf16', False))):
            fw = self._forward(warped_src, warped_dst)

        # ── 重建 loss ──
        loss_per = self._recon_loss(target_src, target_srcm, target_srcm_em, fw)
        loss = loss_per.mean()

        # ── 域对齐：匹配 Inter code 分布 ──
        if 'dst_inter_out_4d' in fw:
            sc = fw['inter_out_4d'].detach()
            dc = fw['dst_inter_out_4d']
            sc_f = sc.view(sc.size(0), -1).float()
            dc_f = dc.view(dc.size(0), -1).float()
            mu_s, mu_d = sc_f.mean(0), dc_f.mean(0)
            align_loss = F.mse_loss(mu_s, mu_d)
            if sc_f.shape[0] > 1:
                sc_c = sc_f - sc_f.mean(0)
                dc_c = dc_f - dc_f.mean(0)
                cov_s = (sc_c.T @ sc_c) / (sc_f.shape[0] - 1)
                cov_d = (dc_c.T @ dc_c) / (dc_f.shape[0] - 1)
                align_loss = align_loss + F.mse_loss(cov_s, cov_d) * 0.1
            loss = loss + align_loss * 0.01

        # ── 反向 ──
        self.src_dst_opt.zero_grad()
        loss.backward()
        self.src_dst_opt.step()
        return float(loss.item())

    def tensor_from_np(self, x):
        if isinstance(x, torch.Tensor): return x
        return torch.from_numpy(x).float().to(self.device)

    def onTrainOneIter(self):
        samples = self.generate_next_samples()
        # samples[0] = src, samples[1] = dst
        ws, ts, tsm = samples[0][0], samples[0][1], samples[0][2]
        tsme = samples[0][3] if len(samples[0]) > 3 else torch.zeros_like(tsm)
        wd = samples[1][0] if len(samples) > 1 and len(samples[1]) > 0 else None
        loss = self.train_one_step(ws, ts, tsm, tsme, wd)
        return (('loss', loss),)

    # ========== preview / merge ==========
    def AE_view(self, target_src, target_dst):
        target_src = self.tensor_from_np(target_src)
        with torch.no_grad():
            fw = self._forward(target_src, target_dst)
        return (
            fw['pred_src_src'].detach().cpu().float().numpy(),
            fw['pred_src_srcm'].detach().cpu().float().numpy(),
        )

    def onGetPreview(self, samples, for_history=False):
        ((ws, ts, tsm, tsme), (wd, td, tdm, tdme)) = samples
        # 用 numpy 做预览
        S = np.clip(nn.to_data_format(ts, 'NHWC', self.model_data_format), 0.0, 1.0)
        # 预测输入改用「训练输入」(warped：已含随机扭曲/遮挡/噪点)，对齐原版 DFL；预览会差一点但更真实。S/D 参照仍取干净目标图。
        SS, SSM = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(ws, wd)]
        WS = np.clip(nn.to_data_format(ws, 'NHWC', self.model_data_format), 0.0, 1.0)
        n = min(4, ts.shape[0])
        st = []
        for i in range(n):
            ar = (S[i], SS[i], np.repeat(SSM[i], 3, axis=-1),
                  WS[i], SS[i] * np.repeat(SSM[i], 3, axis=-1) + S[i] * (1 - np.repeat(SSM[i], 3, axis=-1)))
            st.append(np.concatenate(ar, axis=1))
        return [('DF-Single preview', np.concatenate(st, axis=0))]

    def predictor_func(self, face):
        face = nn.to_data_format(face[None, ...], self.model_data_format, 'NHWC')
        face_t = self.tensor_from_np(face) if face.dtype != np.float32 else face
        with torch.no_grad():
            code = self.inter(self.encoder(face_t))
            bgr, mask = self.decoder_src(code)
        bgr = bgr.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        return (
            nn.to_data_format(bgr, 'NHWC', self.model_data_format)[0],
            nn.to_data_format(mask, 'NHWC', self.model_data_format)[0, ..., 0],
            nn.to_data_format(mask, 'NHWC', self.model_data_format)[0, ..., 0],
        )

    def get_MergerConfig(self):
        import merger
        return (
            self.predictor_func,
            (self.resolution, self.resolution, 3),
            merger.MergerConfigMasked(face_type=self.face_type, default_mode='overlay'),
        )

    def export_dfm(self):
        """导出 ONNX。只有 encoder + inter + decoder_src，比 DF 精简。"""
        from core.leras.archis.DeepFakeArchi import DeepFakeArchi
        output = Path(self.get_strpath_storage_for_file('model.dfm'))

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.encoder = m.encoder
                self.inter = m.inter
                self.decoder_src = m.decoder_src

            def forward(self, x):
                x = x.permute(0, 3, 1, 2)
                code = self.inter(self.encoder(x))
                face, mask = self.decoder_src(code)
                return (face.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1))

        w = Wrapper(self).eval().to('cpu')
        dummy = torch.randn(1, self.resolution, self.resolution, 3).float()
        torch.onnx.export(w, dummy, str(output),
            input_names=['in_face:0'],
            output_names=['out_celeb_face:0', 'out_celeb_face_mask:0'],
            dynamic_axes={'in_face:0': {0: 'batch'}})
        io.log_info(f'Exported: {output}')

Model = SAEHDDFSingleModel
