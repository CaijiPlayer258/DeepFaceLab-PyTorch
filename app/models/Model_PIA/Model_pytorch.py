"""
PIA: Per-Identity Adapter —— 把"身份"从主干里搬到一个极小的适配块里。

主干与 DF-Single 完全一致（Encoder / Inter / decoder_src），但**全程冻结**：
只有两个 IdentityAdapter（adapter_src / adapter_dst）参与训练和更新，
参数量约为 DF 主干 decoder_src 的 1/100。

  - 训练：每分支自己的重建损失 + 循环一致损失（dst->src->dst），全自监督
  - 推理：decoder_src(adapter_src(inter(encoder(dst))))，单路，不需要参考图
  - 主干冻结 -> 不存在"换谁都能重建"的平凡解，身份只能挤进适配块

archi 只支持 df 主干；第二段 u/d/t/c 通用，第三段 -l* 走 DFLiteArchi。
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
# 导入顺序有讲究（已实测）：core.leras 单独 import 不会填充聚合对象 nn，
# 必须先显式导入 core.leras.layers（它把 Saveable 挂上去），
# 之后 core.leras.models 里的层基类才能正常定义。
import core.leras          # noqa: F401
import core.leras.layers   # noqa: F401
from core.leras import nn
# 层基类：注意它和训练模型的 models/ModelBase.py 是【两个同名的类】，
# 这里要的是 core.leras.models.ModelBase（网络层基类，带 Saveable / get_weights / save_weights）。
from core.leras.models.ModelBase import ModelBase as LayerModelBase
from facelib import FaceType
from models import ModelBase
from samplelib import SampleGeneratorFace, SampleProcessor
# archi 字符串解析：三段式（base-opts-mod）里的 mod 决定用哪个 Archi 类。
# archi_parse 会自举 core.leras.layers / core.leras.models，模块级只依赖
# core.archi_names（不碰 torch），放这里 import 是安全的。
from core.leras.archis.archi_parse import parse_archi_string, build_archi_classes, archi_is_lite

class IdentityAdapter(LayerModelBase):
    """逐身份适配块 —— PIA 的"身份"就装在这里。

    只承载"观感"（颜色 / 质感 / 细节倾向），不承载几何：
    几何由 encoder 从输入图提供，所以换谁都成立。

    结构：1x1 通道混合 -> 全局池化出 gamma/beta (FiLM 逐通道仿射) -> 1x1 -> 残差
    参数量约 3*C*mix（C=256 / mix=128 时约 10 万），
    对比 DF 的 decoder_src 约 2500 万 —— 小两个数量级。
    """

    def __init__(self, ch, mix_ch=None, **kwargs):
        super().__init__(**kwargs)
        self.ch = int(ch)
        self.mix_ch = int(mix_ch) if mix_ch else max(32, int(ch) // 2)

    def on_build(self):
        self.conv1 = nn.Conv2D(self.ch, self.mix_ch, kernel_size=1, padding='SAME')
        # 池化后的 1x1 特征上做一次 1x1 卷积 = 等价于全连接，产生 gamma / beta
        self.fc = nn.Conv2D(self.mix_ch, self.mix_ch * 2, kernel_size=1, padding='SAME')
        self.conv2 = nn.Conv2D(self.mix_ch, self.ch, kernel_size=1, padding='SAME')

    def forward(self, f):
        # ★ 去风格（关键，不能省）：
        #   把输入特征自身的逐通道统计量抹掉，只留下空间结构。
        #   否则共享 decoder 能直接从 encoder 特征里恢复"原图外观"，
        #   于是"适配块=恒等"能同时满足重建项和循环项（平凡解），
        #   整个 PIA 就退化成普通自编码器，适配块学不到任何东西。
        #   抹掉之后 ⇒ 外观只能由适配块自己提供。
        # 统计量用 fp32 算再回投：bf16 只有 8 位尾数，逐通道 std 在 bf16 下会明显失真，
        # 而这一步正是"抹掉外观统计量"的核心，误差会直接串到训练信号里。
        _f32 = f.float()
        m = _f32.mean(dim=(2, 3), keepdim=True)
        s = _f32.std(dim=(2, 3), keepdim=True) + 1e-5
        f = ((_f32 - m) / s).to(f.dtype)

        x = self.conv1(f)
        p = x.mean(dim=(2, 3), keepdim=True)          # 全局池化 -> (N, mix, 1, 1)
        gb = self.fc(p)
        g, b = torch.chunk(gb, 2, dim=1)              # gamma / beta
        x = x * (1.0 + g) + b                         # FiLM：逐通道仿射
        return self.conv2(x) + f                      # 残差


class PIAModel(ModelBase):
    """
    DF-Single: DF 单解码器版本。

    与 DF 的差异：
    - 没有 decoder_dst
    - 训练只有 src 侧的 loss
    - 合并时只用 src_decoder 的输出
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('force_model_class_name', 'PIA')
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
        # PIA：循环一致强度（无配对自监督的核心，0 = 关闭）
        self.options['pia_cycle_power'] = self.load_or_def_option('pia_cycle_power', 1.0)
        self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)
        # 预训练模式。
        # PIA 平时主干是冻结的（只训适配块），但【预训练阶段必须解冻主干】——
        # 否则"预训练"根本没有可训练对象：适配块才 20 万参数，撑不起一个换脸器。
        # 语义就是：
        #   预训练：主干解冻，在通用多身份人脸集上把 encoder/inter/decoder_src 练出来
        #   正训练：主干冻结，只训逐身份适配块
        self.options['pretrain'] = self.load_or_def_option('pretrain', False)
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
                        help_message="df / df-ud / df-udt / df-udt-l / df-udt-l3 ...")
                    # 统一用 parse_archi_string 校验：顺手把第三段 DFLite 修饰符
                    # (-l/-l3/-l5/-lm/-lx...) 也一起验了，且与 GUI 走同一套规则。
                    if parse_archi_string(archi) is not None:
                        self.options['archi'] = str(archi).lower()
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
        # ---- 下面这几项以前只躺在 options 里、从没有消费点，现在真正接进损失/样本 ----
        self.masked_training = bool(self.options.get('masked_training', True))
        self.blur_out_mask = bool(self.options.get('blur_out_mask', False))
        self.edge_enhance_power = float(self.options.get('edge_enhance_power', 0.0) or 0.0)
        _ct = str(self.options.get('ct_mode', 'none') or 'none').strip().lower()
        self.ct_mode = None if _ct in ('none', '') else _ct
        if self.pretrain and self.is_training:
            # 预训练集解析（与 SAEHD 同一套约定）：
            #   1) GUI / CLI 未指定时，默认用 dst 训练集；
            #   2) CLI/GUI 可用 --pretraining-data-dir 覆盖（支持 faceset.pak 目录或散装对齐目录）。
            if self.get_pretraining_data_path() is None:
                _default_pt = getattr(self, 'training_data_dst_path', None)
                if _default_pt is not None:
                    self.pretraining_data_path = _default_pt
                    io.log_info(f'[预训练] 未指定预训练集，默认使用 dst 训练集：{_default_pt}')
            _pt = self.get_pretraining_data_path()
            if _pt is not None and Path(_pt).suffix.lower() == '.pak':
                self.pretraining_data_path = Path(_pt).parent
                io.log_info(f'[预训练] 预训练集指向 pak 文件，已改用其所在目录：{self.pretraining_data_path}')
            if self.get_pretraining_data_path() is None:
                raise Exception('预训练模式已开启，但未定义 pretraining_data_path')
            io.log_info('[预训练] 理想上使用多身份人脸集；用 dst 单一身份也能跑，但收益会打折。')
            io.log_info('[预训练] 主干已解冻（encoder + inter + decoder_src 均参与训练）。')

        # 统一走 parse_archi_string：三段式（df-udt-l / -l3 / -l5 / -lm / -lx …）
        # 和两段式旧写法都支持；第三段 mod 交给 build_archi_classes 决定用
        # DeepFakeArchi 还是 DFLiteArchi。以前这里只取 split[1]，会把 -l*
        # 静默丢掉（用户以为在跑 DFLite，实际跑的是原版 5x5 稠密）。
        _parsed = parse_archi_string(self.options['archi'])
        if _parsed is None:
            io.log_info(f"[archi] 无法解析 {self.options['archi']!r}，回退到 df-ud")
            archi_type, archi_opts, archi_mod = 'df', 'ud', ''
        else:
            archi_type, archi_opts, archi_mod = _parsed
        archi_opts = archi_opts or ''

        self.archi_type = archi_type
        self.archi_opts = archi_opts
        self.archi_mod = archi_mod
        self.is_lite = archi_is_lite(archi_mod)

        ae_dims = int(self.options['ae_dims'])
        e_dims = int(self.options['e_dims'])
        d_dims = int(self.options['d_dims'])
        d_mask_dims = int(self.options['d_mask_dims'])
        resolution = self.resolution
        gan_power = float(self.options['gan_power'])

        # mod 为空 -> DeepFakeArchi（与改动前完全一致）；
        # mod 含 'l' -> DFLiteArchi。DFLite 里 ae_dims 同时决定瓶颈宽度和解码器
        # 输入通道，必须与下面 Inter 的 ae_out_ch 一致，所以显式传进去。
        _ArchiClass, _archi_kwargs = build_archi_classes(
            archi_mod, {'ae_dims': ae_dims} if archi_is_lite(archi_mod) else None)
        model_archi = _ArchiClass(resolution, opts=archi_opts, **_archi_kwargs)
        self.archi_class_name = _ArchiClass.__name__

        if 'df' in archi_type:
            # ── Encoder ──
            self.encoder = model_archi.Encoder(in_ch=3, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.get_out_ch() * (self.encoder.get_out_res(resolution) ** 2)

            # ── Inter ──
            self.inter = model_archi.Inter(in_ch=encoder_out_ch, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.get_out_ch()

            # ── Decoder (src only) ──
            self.decoder_src = model_archi.Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')

            # ── PIA：两个逐身份适配块（身份就装在这里）──
            self.adapter_src = IdentityAdapter(ch=inter_out_ch, name='adapter_src')
            self.adapter_dst = IdentityAdapter(ch=inter_out_ch, name='adapter_dst')

            self.model_filename_list = [
                [self.encoder, 'encoder.pth'],
                [self.inter, 'inter.pth'],
                [self.decoder_src, 'decoder_src.pth'],
                [self.adapter_src, 'adapter_src.pth'],
                [self.adapter_dst, 'adapter_dst.pth'],
            ]
        else:
            raise ValueError(f'PIA only supports the DF backbone, got {archi_type!r}')

        for item in [self.encoder, self.inter, self.decoder_src, self.adapter_src, self.adapter_dst]:
            if item is not None:
                for layer in item.get_layers():
                    try: layer.to(self.device)
                    except: pass

        # Optimizer
        if self.is_training:
            _opt_name = str(self.options.get('optimizer', 'adabelief')).lower()
            _opt_map = {'adam': nn.Adam, 'adabelief': nn.AdaBelief, 'lion': nn.Lion}
            _opt_class = _opt_map.get(_opt_name, nn.AdaBelief)
            # 常规：优化器【只】拿适配器参数 —— 主干完全冻结，不会被更新。
            # 预训练：主干也进优化器（encoder + inter + decoder_src），
            #         这一段的目的就是把它从随机初始化学成一个通用换脸器。
            _trainable = list(self.adapter_src.get_weights()) + list(self.adapter_dst.get_weights())
            if self.pretrain:
                _trainable = (list(self.encoder.get_weights()) + list(self.inter.get_weights())
                              + list(self.decoder_src.get_weights()) + _trainable)
            # ★ 梯度裁剪：跟 SAEHD / AMP 一样，clipgrad → 优化器 clipnorm=1.0
            #   （leras 的 Adam/AdaBelief/Lion 在 step 里按全局 grad norm 缩放）。
            #   以前这里只是把 clipgrad 读进 options 就再没人读它 ——
            #   界面上的「梯度裁剪」是装饰品，dat 里 clipgrad=True 也完全没裁剪。
            _clipnorm = 1.0 if bool(self.options.get('clipgrad', False)) else 0.0
            # ★ 学习率相关的开关以前全是死的：这里把 lr 写死成 5e-5，
            #   界面上的「学习率 / 学习率策略(lr_dropout) / 余弦退火周期(lr_cos) /
            #   总步数(lr_total_steps) / 平台期自动降 lr」一个都没传进优化器 ——
            #   优化器其实全都支持，只是没人喂。
            _lr = float(self.options.get('lr', 5e-5) or 5e-5)
            _lr_dropout = 0.3 if (str(self.options.get('lr_dropout', 'n')) in ('y', 'cpu')
                                  and not self.pretrain) else 1.0
            self.src_dst_opt = _opt_class(
                _trainable, lr=_lr, name='src_dst_opt',
                lr_dropout=_lr_dropout,
                lr_cos=int(self.options.get('lr_cos', 0) or 0),
                lr_total_steps=int(self.options.get('lr_total_steps', 0) or 0),
                lr_plateau_factor=float(self.options.get('lr_plateau_factor', 1.0) or 1.0),
                lr_plateau_patience=int(self.options.get('lr_plateau_patience', 0) or 0),
                lr_plateau_min_ratio=float(self.options.get('lr_plateau_min_ratio', 0.1) or 0.1),
                clipnorm=_clipnorm,
            )
            # 优化器参数表随模式变化（2 个模块 vs 5 个模块），两种模式各存一份，
            # 避免来回切换时按位置读档把 Adam/AdaBelief 的 m/v 串到一起。
            self.model_filename_list += [[
                self.src_dst_opt,
                'src_dst_opt_pretrain.pth' if self.pretrain else 'src_dst_opt.pth',
            ]]

        # ★ 主干冻结：反向不产生任何主干梯度（省显存 + 身份只存在适配块里）。
        #   预训练阶段不冻结 —— 那时主干就是唯一要练的东西。
        if not (self.is_training and self.pretrain):
            for _m in (self.encoder, self.inter, self.decoder_src):
                try:
                    for _p in _m.get_weights():
                        _p.requires_grad_(False)
                except Exception:
                    pass

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
            # 旋转 / 缩放 / 平移范围以前是写死的（scale_range 固定 ±0.15，
            # 旋转吃 SampleProcessor 的默认 ±10°），界面上那三个框改了没反应。
            _rot = float(self.options.get('rotation_range', 10) or 10)
            _scl = float(self.options.get('scale_range', 0.15) or 0.15)
            _trn = float(self.options.get('t_range', 0.05) or 0.05)
            _opts = SampleProcessor.Options(
                rotation_range=[-_rot, _rot],
                scale_range=[-_scl, _scl],
                tx_range=[-_trn, _trn],
                ty_range=[-_trn, _trn],
                random_flip=self.random_src_flip if hasattr(self, 'random_src_flip') else True,
            )
            # 4 项：输入(扭曲) / 目标(干净) / 全脸遮罩 / 嘴眼遮罩
            # 训练代码（onTrainOneIter）就是按这个顺序读的。
            # 这几项以前只是躺在 dat 里，样本规格里一个都没接 —— 界面开了也不生效。
            _aug_occlusion = bool(self.options.get('random_occlusion', False))
            _aug_noise     = bool(self.options.get('random_noise', False))
            _aug_colorcast = float(self.options.get('random_color_power', 0.0) or 0.0)
            # 预训练阶段按 SAEHD 的惯例关掉色彩类增强
            _aug_hsv       = 0.0 if self.pretrain else float(self.options.get('random_hsv_power', 0.0) or 0.0)
            _out = [
                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                 # 预训练时关掉随机形变（SAEHD 同款：预训练先学结构，不学形变）
                 'warp': False if self.pretrain else bool(self.options.get('random_warp', True)),
                 'random_occlusion': _aug_occlusion,
                 'random_noise': _aug_noise,
                 'random_color_cast': _aug_colorcast,
                 'random_hsv_shift_amount': _aug_hsv,
                 'ct_mode': self.ct_mode,
                 'transform': True,
                 'channel_type': SampleProcessor.ChannelType.BGR,
                 'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                 'warp': False, 'transform': True,
                 'channel_type': SampleProcessor.ChannelType.BGR,
                 'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
                {'sample_type': SampleProcessor.SampleType.FACE_MASK,
                 'warp': False, 'transform': True,
                 'channel_type': SampleProcessor.ChannelType.G,
                 'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
                 'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
                {'sample_type': SampleProcessor.SampleType.FACE_MASK,
                 'warp': False, 'transform': True,
                 'channel_type': SampleProcessor.ChannelType.G,
                 'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
                 'face_type': self.face_type, 'data_format': nn.data_format, 'resolution': resolution},
            ]
            # 预训练：src / dst 都指向预训练集（多身份人脸集），与 SAEHD 一致
            _train_src_path = self.get_pretraining_data_path() if self.pretrain else self.training_data_src_path
            _train_dst_path = self.get_pretraining_data_path() if self.pretrain else self.training_data_dst_path
            # 色彩迁移（ct_mode）：把 src 的颜色往 dst 的分布拉，需要一份 dst 样本源
            _ct_samples_path = _train_dst_path if (self.ct_mode is not None and not self.pretrain) else None
            _gen_src = SampleGeneratorFace(
                _train_src_path,
                random_ct_samples_path=_ct_samples_path,
                debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=_opts,
                output_sample_types=_out,
                generators_count=generators_count,
                uniform_yaw_distribution=bool(self.options.get('uniform_yaw', False)) or self.pretrain,
            )
            _gen_dst = SampleGeneratorFace(
                _train_dst_path,
                debug=self.is_debug(),
                batch_size=self.get_batch_size(),
                sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=True),
                output_sample_types=_out,
                generators_count=generators_count,
            )
            self.set_training_data_generators([_gen_src, _gen_dst])

            # ── 快速加载器（use_fast_generator）──
            # 走 SampleLoaderV4 + SampleGeneratorV2：图片一次解码进内存池，
            # 不再每步重新读盘解码，加载 5 万张时差距是"1.5 分钟 vs 几秒"。
            # 以前 PIA 只实现了老的 SampleGeneratorFace 这条路，界面上的
            # 「快速生成器」开关拨了没反应，所以之前把它藏了 —— 现在补上。
            if bool(self.options.get('use_fast_generator', False)):
                try:
                    from samplelib import SampleLoaderV4, SampleGeneratorV2
                    _yaw = bool(self.options.get('uniform_yaw', False)) or self.pretrain
                    _bs = self.get_batch_size()
                    _src_loader = SampleLoaderV4(aligned_path=_train_src_path, batch_size=_bs,
                                                 resolution=resolution, use_yaw_sampling=_yaw)
                    _dst_loader = SampleLoaderV4(aligned_path=_train_dst_path, batch_size=_bs,
                                                 resolution=resolution, use_yaw_sampling=_yaw)
                    _opts_dst = SampleProcessor.Options(
                        rotation_range=[-_rot, _rot], scale_range=[-_scl, _scl],
                        tx_range=[-_trn, _trn], ty_range=[-_trn, _trn], random_flip=True)
                    _fast_src = SampleGeneratorV2(
                        loader=_src_loader, sample_process_options=_opts,
                        output_sample_types=_out, resolution=resolution,
                        debug=self.is_debug(),
                        # ct_mode 在 V2 里由配对 loader 提供，而不是 random_ct_samples_path
                        ct_loader=(_dst_loader if (_ct_samples_path is not None) else None))
                    _fast_dst = SampleGeneratorV2(
                        loader=_dst_loader, sample_process_options=_opts_dst,
                        output_sample_types=_out, resolution=resolution, debug=self.is_debug())
                    self.set_training_data_generators([_fast_src, _fast_dst])
                    io.log_info('[加载器] 已启用快速生成器（SampleLoaderV4 + SampleGeneratorV2）')
                except Exception as _fe:
                    # 回退到老生成器，不让"打开一个加速开关"把训练搞挂
                    io.log_info(f'[加载器] 快速生成器不可用，回退到 SampleGeneratorFace：{_fe}')


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
        ae = int(self.options.get('ae_dims', 256))
        lr = self.resolution // (32 if 'd' in (self.archi_opts or '') else 16)
        lr = lr * (2 if 't' not in (self.archi_opts or '') else 1)

        def _to_4d(x):
            # Inter 输出本来就是 NCHW；只有兜底形状推断时才需要 view
            return x if x.ndim == 4 else x.view(x.size(0), ae, lr, lr)

        src_code = self.inter(self.encoder(warped_src))
        # Inter 最终的 4D 特征图（reshaped + upscale1 后）供 Conv2D 判别器 / 域对齐
        src_inter_4d = _to_4d(src_code)
        # PIA：用 src 适配器调制后再解码（换脸时把这里换成 adapter_dst 的那份 code 即可）
        pred_src_src, pred_src_srcm = self.decoder_src(self.adapter_src(src_code))
        out = {'inter_out_4d': src_inter_4d, 'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm}

        if warped_dst is not None:
            # ★ dst 支路只前向一次：这份 code 同时供
            #     ① dst 重建（pred_dst_dst / pred_dst_dstm）
            #     ② 域对齐（dst_inter_out_4d）
            #   原来这里 inter(encoder(warped_dst)) 被算了**两遍**（一次给重建、一次给对齐），
            #   白烧一整套 encoder+inter 的算力和激活 —— 416 分辨率下这一份就是几百 MB。
            dst_code = self.inter(self.encoder(warped_dst))
            pred_dst_dst, pred_dst_dstm = self.decoder_src(self.adapter_dst(dst_code))
            out['pred_dst_dst'] = pred_dst_dst
            out['pred_dst_dstm'] = pred_dst_dstm
            out['dst_inter_out_4d'] = _to_4d(dst_code)
        return out

    # ========== losses ==========
    @staticmethod
    def _blur_out(target, mask, sigma):
        """把遮罩外区域换成"从脸内向外扩散的模糊值"（SAEHD 的 blur_out_mask 同款）。

        目的：遮罩外的目标值本来是"原图那一块"，模型却永远被遮罩挡住看不到，
        两边打架。换成模糊扩散后，遮罩外的惩罚项变成平滑的外推，不会产生硬边。
        """
        anti = 1.0 - mask
        x = nn.gaussian_blur(target * anti, sigma)
        y = 1.0 - nn.gaussian_blur(mask, sigma)
        y = torch.where(y == 0, torch.ones_like(y), y)
        return target * mask + (x / y) * anti

    def _cycle_loss(self, target_src, target_dst, weight=1.0):
        """循环一致（无配对自监督）—— 保证适配块"只改外观、不改几何"。

            dst --[src适配器]--> F --[dst适配器]--> 应还原 dst
            src --[dst适配器]--> G --[src适配器]--> 应还原 src

        若适配块擅自改了五官位置，反向就还原不回来，这一项会炸。
        注意：它必须和"各自重建"一起用 —— 单用也有平凡解（恒等）。
        """
        def _recode(img, ad):
            return self.decoder_src(ad(self.inter(self.encoder(img))))[0]

        # ★ 显存：循环一致天然要跑 4 条完整 pipeline（去程 f/g + 回程 back_d/back_s）。
        # 4 条全留在计算图里，加上主前向的 2 条 = 每步 6 条 encoder+decoder，
        # 416 分辨率 / batch 8 会直接把 20G 卡打爆（实测炸在本函数第二次 _recode）。
        # 去程的语义只是"把外观换成另一个身份"，把它当固定变换（no_grad）完全够用：
        # 回程仍然对 encoder / inter / decoder / adapter 完整求导，
        # "适配块不能改几何"这个约束一点没丢，只是省掉一半激活。
        with torch.no_grad():
            # 正向去程：dst 换成 src 外观
            f = _recode(target_dst, self.adapter_src)
            # 反向去程：src 换成 dst 外观
            g = _recode(target_src, self.adapter_dst)
        back_d = _recode(f, self.adapter_dst)
        back_s = _recode(g, self.adapter_src)

        return (F.l1_loss(back_d, target_dst) + F.l1_loss(back_s, target_src)) * weight

    def _recon_loss(self, target_src, target_srcm, target_srcm_em, fw,
                    target_dst=None, target_dstm=None, target_dstm_em=None):
        resolution = self.resolution
        pred = fw['pred_src_src']
        predm = fw['pred_src_srcm']

        # 遮罩羽化（blur_out_mask）：把遮罩外的目标值换成扩散外推，避免硬边
        if self.blur_out_mask:
            target_src = self._blur_out(target_src, target_srcm,
                                        float(resolution) / 128.0)

        # mask blur
        k = max(1, resolution // 32)
        tm = nn.gaussian_blur(target_srcm, k)
        tm = torch.clamp(tm, 0.0, 0.5) * 2.0
        # 遮罩训练：关掉时整图参与像素损失（SAEHD 的 masked_training 语义）
        _m = tm if self.masked_training else torch.ones_like(tm)
        t_s, p_s = target_src * _m, pred * _m

        # DSSIM + MSE
        fs1 = max(1, int(resolution / 11.6))
        fs2 = max(1, int(resolution / 23.2))
        loss = 0
        if resolution < 256:
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 10
        else:
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 5
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs2).mean(dim=[1,2,3]) * 5
        loss = loss + F.mse_loss(p_s, t_s, reduction='none').mean(dim=[1,2,3]) * 10
        if self.options.get('eyes_mouth_prio', False):
            loss = loss + F.l1_loss(pred * target_srcm_em, target_src * target_srcm_em, reduction='none').mean(dim=[1,2,3]) * 300
        loss = loss + F.mse_loss(predm, target_srcm, reduction='none').mean(dim=[1,2,3]) * 10

        # ── 边缘强化 ──
        # 用模型自己的遮罩把 pred 合成回原图再比：comp - tgt = m*(pred - tgt)，
        # 于是 0<m<1 的过渡带（合成后的脸边缘）会被专门惩罚。
        if self.edge_enhance_power > 0.0:
            _comp = pred * predm + target_src * (1.0 - predm)
            loss = loss + F.mse_loss(_comp, target_src, reduction='none').mean(dim=[1,2,3]) * self.edge_enhance_power

        # ── dst 支路重建（给共享 decoder 泛化，防止过拟合 src）──
        if target_dst is not None and 'pred_dst_dst' in fw:
            loss = loss + self._recon_only(fw['pred_dst_dst'], fw['pred_dst_dstm'],
                                           target_dst, target_dstm, target_dstm_em)

        # ── 循环一致（无配对自监督，核心）──
        _cyc_p = float(self.options.get('pia_cycle_power', 1.0))
        if target_dst is not None and _cyc_p > 0.0:
            loss = loss + self._cycle_loss(target_src, target_dst, weight=_cyc_p)

        return loss

    def _recon_only(self, pred, predm, target, targetm, targetm_em):
        """单个分支的重建项（与 src 支路同一套配方：dssim + mse + 嘴眼优先 + 遮罩）。"""
        resolution = self.resolution
        if self.blur_out_mask:
            target = self._blur_out(target, targetm, float(resolution) / 128.0)
        k = max(1, resolution // 32)
        tm = nn.gaussian_blur(targetm, k)
        tm = torch.clamp(tm, 0.0, 0.5) * 2.0
        _m = tm if self.masked_training else torch.ones_like(tm)
        t_s, p_s = target * _m, pred * _m
        fs1 = max(1, int(resolution / 11.6))
        fs2 = max(1, int(resolution / 23.2))
        loss = 0
        if resolution < 256:
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 10
        else:
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs1).mean(dim=[1,2,3]) * 5
            loss = loss + nn.dssim(t_s, p_s, max_val=1.0, filter_size=fs2).mean(dim=[1,2,3]) * 5
        loss = loss + F.mse_loss(p_s, t_s, reduction='none').mean(dim=[1,2,3]) * 10
        if self.options.get('eyes_mouth_prio', False):
            loss = loss + F.l1_loss(pred * targetm_em, target * targetm_em,
                                    reduction='none').mean(dim=[1,2,3]) * 300
        loss = loss + F.mse_loss(predm, targetm, reduction='none').mean(dim=[1,2,3]) * 10
        return loss

    def train_one_step(self, warped_src, target_src, target_srcm, target_srcm_em,
                       warped_dst=None, target_dst=None, target_dstm=None, target_dstm_em=None):
        warped_src = self.tensor_from_np(warped_src)
        target_src = self.tensor_from_np(target_src)
        target_srcm = self.tensor_from_np(target_srcm)
        target_srcm_em = self.tensor_from_np(target_srcm_em)
        if warped_dst is not None:
            warped_dst = self.tensor_from_np(warped_dst)
        # dst 的三个目标也要转成 torch —— 它们是在 PIA 里新增的参数，
        # 原来 DFSingle 的转换语句里没有它们（少了这一步，loss 会拿到 numpy 而报错）。
        if target_dst is not None:
            target_dst = self.tensor_from_np(target_dst)
        if target_dstm is not None:
            target_dstm = self.tensor_from_np(target_dstm)
        if target_dstm_em is not None:
            target_dstm_em = self.tensor_from_np(target_dstm_em)

        # ★ autocast 现在覆盖**整个 step**（前向 + 所有损失）。
        #   以前只有 _forward 在 autocast 里，循环一致那两条回程和域对齐都是在
        #   fp32 下算的 —— 半张卡的时间白花。反向仍在 autocast 外（标准用法）。
        _amp_on = bool(self.options.get('use_bf16', False))
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=_amp_on):
            fw = self._forward(warped_src, warped_dst)

            # ── 重建 loss ──
            loss_per = self._recon_loss(target_src, target_srcm, target_srcm_em, fw,
                                        target_dst=target_dst, target_dstm=target_dstm,
                                        target_dstm_em=target_dstm_em)
            loss = loss_per.mean()

            # ── 域对齐：匹配 Inter code 分布（只在预训练阶段有意义）──
            # ① 只有主干可训练时这一项才产生梯度：常规训练里 encoder/inter 被冻结，
            #    而 inter_out_4d 取自它们（且 src 侧 detach 了），上游没有可训练参数。
            # ② ★ 原来是「把 code 拉平成 (B, C*H*W) 再算完整协方差」：
            #      cov = sc_c.T @ sc_c 的形状是 (C*H*W)² ——
            #      416 分辨率 / ae 320 时 (320*26*26)² = 216320² ≈ 174 GiB，
            #      batch ≥ 2 必然 OOM（batch=1 时被 shape[0] > 1 挡住才侥幸没炸）。
            #    改成逐通道一阶/二阶矩匹配：内存 O(C)，同样是分布对齐，量级完全可控。
            if self.pretrain and 'dst_inter_out_4d' in fw:
                sc = fw['inter_out_4d'].detach().float()
                dc = fw['dst_inter_out_4d'].float()
                sc_mu, sc_var = sc.mean(dim=(0, 2, 3)), sc.var(dim=(0, 2, 3), unbiased=False)
                dc_mu, dc_var = dc.mean(dim=(0, 2, 3)), dc.var(dim=(0, 2, 3), unbiased=False)
                align_loss = (F.mse_loss(sc_mu, dc_mu)
                              + F.mse_loss(torch.sqrt(sc_var + 1e-6), torch.sqrt(dc_var + 1e-6)))
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
        # PIA：dst 的 target 也要取出来 —— 它承担两件事：
        #   ① dst 重建 → 给共享 decoder 泛化，防止过拟合 src
        #   ② 循环一致 → 无配对自监督的核心
        td   = samples[1][1] if wd is not None and len(samples[1]) > 1 else None
        tdm  = samples[1][2] if wd is not None and len(samples[1]) > 2 else None
        tdme = samples[1][3] if wd is not None and len(samples[1]) > 3 else None
        loss = self.train_one_step(ws, ts, tsm, tsme, wd, td, tdm, tdme)
        return (('loss', loss),)

    # ========== preview / merge ==========
    def AE_view(self, target_src, target_dst):
        target_src = self.tensor_from_np(target_src)
        # ★ dst 也必须转成 tensor：_forward 会把两张图一起送进 encoder，
        #   漏转的话 F.pad 会收到 numpy，预览直接抛
        #   "pad(): argument 'input' must be Tensor, not numpy.ndarray"。
        if target_dst is not None:
            target_dst = self.tensor_from_np(target_dst)
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
        return [('PIA preview', np.concatenate(st, axis=0))]

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

Model = PIAModel
