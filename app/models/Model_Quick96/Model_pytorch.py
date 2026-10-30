"""
Quick96模型 - 完整PyTorch实现

这是Quick96模型的完整训练和推理实现，包含所有训练功能。
"""

import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


class QModel(ModelBase):
    """Quick96模型 - 快速96x96分辨率的deepfake模型"""
    
    def on_initialize(self):
        """初始化模型"""
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        
        # PyTorch总是使用NCHW格式
        self.model_data_format = "NCHW"
        # 不要覆盖 ModelBase 已经选定的 device_config（CPU/CUDA/MPS）
        nn.initialize(device_config, data_format=self.model_data_format)
        
        # 模型配置
        resolution = self.resolution = 96
        self.face_type = FaceType.FULL
        ae_dims = 128
        e_dims = 64
        d_dims = 64
        d_mask_dims = 16
        self.pretrain = False
        self.pretrain_just_disabled = False
        
        masked_training = True
        
        # 设备配置
        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])
        self.models_opt_on_gpu = models_opt_on_gpu and self.is_training
        optimizer_vars_on_cpu = not self.models_opt_on_gpu

        # 选择设备：跟随 leras nn.initialize 的结果，保证输入/权重在同一设备
        self.device = nn.device
        
        input_ch = 3
        self.model_filename_list = []
        
        # 创建模型架构
        model_archi = nn.DeepFakeArchi(resolution, opts='ud')
        
        # 初始化模型组件
        self.encoder = model_archi.Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
        encoder_out_res = self.encoder.get_out_res(resolution)
        encoder_out_ch = self.encoder.get_out_ch() * encoder_out_res * encoder_out_res
        
        self.inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
        inter_out_ch = self.inter.get_out_ch()
        
        self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
        self.decoder_dst = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

        # 将所有权重移动到选定设备（leras Layer 默认在 CPU 上创建权重）
        for _m in (self.encoder, self.inter, self.decoder_src, self.decoder_dst):
            try:
                _m.to(self.device)
            except Exception:
                pass
        
        self.model_filename_list += [
            [self.encoder, 'encoder.pth'],
            [self.inter, 'inter.pth'],
            [self.decoder_src, 'decoder_src.pth'],
            [self.decoder_dst, 'decoder_dst.pth']
        ]
        
        # 存储masked_training标志
        self.masked_training = masked_training
        self.resolution = resolution
        
        if self.is_training:
            # 收集可训练参数
            self.src_dst_trainable_weights = (
                list(self.encoder.get_weights()) + 
                list(self.inter.get_weights()) + 
                list(self.decoder_src.get_weights()) + 
                list(self.decoder_dst.get_weights())
            )
            
            # 初始化优化器
            self.src_dst_opt = nn.RMSprop(
                self.src_dst_trainable_weights,
                lr=2e-4,
                lr_dropout=0.3,
                name='src_dst_opt'
            )
            
            self.model_filename_list += [(self.src_dst_opt, 'src_dst_opt.pth')]
            
            # 调整batch size
            gpu_count = max(1, len(devices))
            bs_per_gpu = max(1, 4 // gpu_count)
            self.set_batch_size(gpu_count * bs_per_gpu)
            
            # 如果使用多GPU，包装模型
            if gpu_count > 1 and torch.cuda.is_available():
                self.encoder = torch.nn.DataParallel(self.encoder)
                self.inter = torch.nn.DataParallel(self.inter)
                self.decoder_src = torch.nn.DataParallel(self.decoder_src)
                self.decoder_dst = torch.nn.DataParallel(self.decoder_dst)
        
        # 加载/初始化权重
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if model == self.inter:
                    do_init = True
            else:
                do_init = self.is_first_run()
            
            if not do_init:
                filepath = self.get_strpath_storage_for_file(filename)
                if Path(filepath).exists():
                    try:
                        model.load_weights(filepath)
                        do_init = False
                    except:
                        do_init = True
                else:
                    do_init = True
            
            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                if pretrained_filepath.exists():
                    try:
                        model.load_weights(str(pretrained_filepath))
                        do_init = False
                    except:
                        do_init = True
            
            if do_init:
                model.init_weights()

        # Ensure all layers end up on the selected device (MPS/CUDA/CPU).
        def _move_leras_tree(m):
            try:
                if isinstance(m, torch.nn.DataParallel):
                    m = m.module
            except Exception:
                pass

            try:
                layers = m.get_layers()
            except Exception:
                layers = []
            for layer in layers:
                try:
                    layer.to(self.device)
                except Exception:
                    pass

        for _m in (self.encoder, self.inter, self.decoder_src, self.decoder_dst):
            _move_leras_tree(_m)
        
        # 初始化数据生成器
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()
            
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
            
            self.set_training_data_generators([
                SampleGeneratorFace(
                    training_data_src_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(
                        random_flip=True if self.pretrain else False
                    ),
                    output_sample_types=[
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': True,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        },
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': False,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        },
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_MASK,
                            'warp': False,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.G,
                            'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        }
                    ],
                    generators_count=src_generators_count
                ),
                
                SampleGeneratorFace(
                    training_data_dst_path,
                    debug=self.is_debug(),
                    batch_size=self.get_batch_size(),
                    sample_process_options=SampleProcessor.Options(
                        random_flip=True if self.pretrain else False
                    ),
                    output_sample_types=[
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': True,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        },
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                            'warp': False,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.BGR,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        },
                        {
                            'sample_type': SampleProcessor.SampleType.FACE_MASK,
                            'warp': False,
                            'transform': True,
                            'channel_type': SampleProcessor.ChannelType.G,
                            'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                            'face_type': self.face_type,
                            'data_format': nn.data_format,
                            'resolution': resolution
                        }
                    ],
                    generators_count=dst_generators_count
                )
            ])
            
            self.last_samples = None
    
    def get_model_filename_list(self):
        """获取模型文件名列表"""
        return self.model_filename_list
    
    def onSave(self):
        """保存模型权重"""
        for model, filename in io.progress_bar_generator(
            self.get_model_filename_list(), "Saving", leave=False
        ):
            model.save_weights(self.get_strpath_storage_for_file(filename))
    
    def train_one_step(self, warped_src, target_src, target_srcm,
                       warped_dst, target_dst, target_dstm):
        """执行一步训练
        
        Args:
            warped_src: 扭曲的源图像
            target_src: 目标源图像
            target_srcm: 源掩码
            warped_dst: 扭曲的目标图像
            target_dst: 目标图像
            target_dstm: 目标掩码
            
        Returns:
            src_loss, dst_loss: 源和目标损失
        """
        # 转换为PyTorch张量并移到设备
        device = self.device
        warped_src = torch.from_numpy(warped_src).float().to(device)
        target_src = torch.from_numpy(target_src).float().to(device)
        target_srcm = torch.from_numpy(target_srcm).float().to(device)
        warped_dst = torch.from_numpy(warped_dst).float().to(device)
        target_dst = torch.from_numpy(target_dst).float().to(device)
        target_dstm = torch.from_numpy(target_dstm).float().to(device)
        
        # 前向传播
        src_code = self.inter(self.encoder(warped_src))
        dst_code = self.inter(self.encoder(warped_dst))
        
        pred_src_src, pred_src_srcm = self.decoder_src(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
        
        # 应用高斯模糊到掩码
        target_srcm_blur = nn.gaussian_blur(target_srcm, max(1, self.resolution // 32))
        target_dstm_blur = nn.gaussian_blur(target_dstm, max(1, self.resolution // 32))
        
        # 准备masked版本
        if self.masked_training:
            target_src_masked_opt = target_src * target_srcm_blur
            target_dst_masked_opt = target_dst * target_dstm_blur
            pred_src_src_masked_opt = pred_src_src * target_srcm_blur
            pred_dst_dst_masked_opt = pred_dst_dst * target_dstm_blur
        else:
            target_src_masked_opt = target_src
            target_dst_masked_opt = target_dst
            pred_src_src_masked_opt = pred_src_src
            pred_dst_dst_masked_opt = pred_dst_dst
        
        def _batch_mean(x: torch.Tensor) -> torch.Tensor:
            """Reduce any per-sample tensor to shape [B]."""
            return x.reshape(x.shape[0], -1).mean(dim=1)

        # 计算损失（每项先变成 [B] 再相加，避免广播导致的 in-place 错误）
        # 源损失
        src_loss = (
            _batch_mean(
                10 * nn.dssim(
                    target_src_masked_opt,
                    pred_src_src_masked_opt,
                    max_val=1.0,
                    filter_size=int(self.resolution / 11.6),
                )
            )
            + _batch_mean(10 * torch.square(target_src_masked_opt - pred_src_src_masked_opt))
            + _batch_mean(10 * torch.square(target_srcm - pred_src_srcm))
        )
        
        # 目标损失
        dst_loss = (
            _batch_mean(
                10 * nn.dssim(
                    target_dst_masked_opt,
                    pred_dst_dst_masked_opt,
                    max_val=1.0,
                    filter_size=int(self.resolution / 11.6),
                )
            )
            + _batch_mean(10 * torch.square(target_dst_masked_opt - pred_dst_dst_masked_opt))
            + _batch_mean(10 * torch.square(target_dstm - pred_dst_dstm))
        )
        
        # 总损失
        total_loss = src_loss + dst_loss
        total_loss = torch.mean(total_loss)
        
        # 反向传播
        self.src_dst_opt.zero_grad()
        total_loss.backward()
        self.src_dst_opt.step()
        
        # 返回标量损失
        return src_loss.mean().item(), dst_loss.mean().item()
    
    def onTrainOneIter(self):
        """训练一次迭代"""
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            # 每3次迭代使用未扭曲的图像
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = self.last_samples
            warped_src = target_src
            warped_dst = target_dst
        else:
            # 生成新样本
            samples = self.last_samples = self.generate_next_samples()
            ((warped_src, target_src, target_srcm),
             (warped_dst, target_dst, target_dstm)) = samples
        
        # 训练一步
        src_loss, dst_loss = self.train_one_step(
            warped_src, target_src, target_srcm,
            warped_dst, target_dst, target_dstm
        )
        
        return (('src_loss', src_loss), ('dst_loss', dst_loss),)
    
    def AE_view(self, target_src, target_dst):
        """推理视图函数
        
        Args:
            target_src: 源图像
            target_dst: 目标图像
            
        Returns:
            pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm
        """
        device = self.device
        
        # 转换为张量
        target_src = torch.from_numpy(target_src).float().to(device)
        target_dst = torch.from_numpy(target_dst).float().to(device)
        
        with torch.no_grad():
            # 前向传播
            src_code = self.inter(self.encoder(target_src))
            dst_code = self.inter(self.encoder(target_dst))
            
            pred_src_src, _ = self.decoder_src(src_code)
            pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
            pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
        
        # 转换回numpy
        pred_src_src = pred_src_src.cpu().numpy()
        pred_dst_dst = pred_dst_dst.cpu().numpy()
        pred_dst_dstm = pred_dst_dstm.cpu().numpy()
        pred_src_dst = pred_src_dst.cpu().numpy()
        pred_src_dstm = pred_src_dstm.cpu().numpy()
        
        return pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm
    
    def onGetPreview(self, samples, for_history=False):
        """生成预览图像"""
        ((warped_src, target_src, target_srcm),
         (warped_dst, target_dst, target_dstm)) = samples
        
        # 获取预测结果
        S, D, SS, DD, DDM, SD, SDM = [
            np.clip(nn.to_data_format(x, "NHWC", self.model_data_format), 0.0, 1.0)
            # 预测输入改用「训练输入」(warped：已含随机扭曲/遮挡/噪点)，对齐原版 DFL；预览会差一点但更真实。S/D 参照仍取干净目标图。
            for x in ([target_src, target_dst] + list(self.AE_view(warped_src, warped_dst)))
        ]
        
        # 重复掩码通道
        DDM = np.repeat(DDM, 3, axis=-1)
        SDM = np.repeat(SDM, 3, axis=-1)
        
        # 转换掩码格式
        target_srcm = nn.to_data_format(target_srcm, "NHWC", self.model_data_format)
        target_dstm = nn.to_data_format(target_dstm, "NHWC", self.model_data_format)
        
        n_samples = self.get_batch_size()
        result = []
        
        # 生成基础预览
        st = []
        for i in range(n_samples):
            ar = [S[i], SS[i], D[i], DD[i], SD[i]]
            st.append(np.concatenate(ar, axis=1))
        result.append(('Quick96', np.concatenate(st, axis=0)))
        
        # 生成masked预览
        st_m = []
        for i in range(n_samples):
            ar = [
                S[i] * target_srcm[i],
                SS[i],
                D[i] * target_dstm[i],
                DD[i] * DDM[i],
                SD[i] * (DDM[i] * SDM[i])
            ]
            st_m.append(np.concatenate(ar, axis=1))
        result.append(('Quick96 masked', np.concatenate(st_m, axis=0)))
        
        return result
    
    def AE_merge(self, warped_dst):
        """合并函数（用于推理）
        
        Args:
            warped_dst: 扭曲的目标图像
            
        Returns:
            bgr, mask_dst_dstm, mask_src_dstm
        """
        device = self.device
        warped_dst = torch.from_numpy(warped_dst).float().to(device)
        
        with torch.no_grad():
            dst_code = self.inter(self.encoder(warped_dst))
            pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
            _, pred_dst_dstm = self.decoder_dst(dst_code)
        
        # 转换回numpy
        pred_src_dst = pred_src_dst.cpu().numpy()
        pred_dst_dstm = pred_dst_dstm.cpu().numpy()
        pred_src_dstm = pred_src_dstm.cpu().numpy()
        
        return pred_src_dst, pred_dst_dstm, pred_src_dstm
    
    def predictor_func(self, face=None):
        """预测函数"""
        face = nn.to_data_format(face[None, ...], self.model_data_format, "NHWC")
        
        bgr, mask_dst_dstm, mask_src_dstm = [
            nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32)
            for x in self.AE_merge(face)
        ]
        
        return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]
    
    def get_MergerConfig(self):
        """获取合并配置"""
        import merger
        return (
            self.predictor_func,
            (self.resolution, self.resolution, 3),
            merger.MergerConfigMasked(
                face_type=self.face_type,
                default_mode='overlay',
            )
        )


Model = QModel
