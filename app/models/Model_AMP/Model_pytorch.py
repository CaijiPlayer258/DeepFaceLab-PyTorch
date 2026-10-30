"""\
AMP模型 - 完整PyTorch训练/推理实现

目标：在不简化功能的前提下，将原 TF/graph 版 AMP 训练逻辑迁移到 PyTorch eager。

对齐特性：
- Encoder / Inter_src / Inter_dst / Decoder 结构
- morph_factor 的 inter mixing（binomial shuffle）
- blur_out_mask 逻辑
- loss 组成：dssim(两尺度) + pixel mse + eyes/mouth prio + mask mse + dst background weak + total_variation_mse
- 可选 GAN：UNetPatchDiscriminator + AdaBelief
- preview / merger predictor / export_dfm(ONNX)
"""

import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.interact import interact as io
from core.leras import nn
from core.imagelib.text import _get_pil_font
from facelib import FaceType
from models import ModelBase
from samplelib import SampleGeneratorFace, SampleProcessor, SampleLoaderV4, SampleGeneratorV2


class AMPModel(ModelBase):
	# --- options ---
	def on_initialize_options(self):
		fast_smoke = str(os.environ.get('DFL_SMOKE_FAST', '')).strip().lower() in ('1', 'y', 'yes', 'true', 'on')

		default_resolution = self.options['resolution'] = self.load_or_def_option('resolution', 64 if fast_smoke else 224)
		default_face_type = self.options['face_type'] = self.load_or_def_option('face_type', 'wf')
		default_models_opt_on_gpu = self.options['models_opt_on_gpu'] = self.load_or_def_option('models_opt_on_gpu', True)

		default_ae_dims = self.options['ae_dims'] = self.load_or_def_option('ae_dims', 64 if fast_smoke else 256)
		default_inter_dims = self.options['inter_dims'] = self.load_or_def_option('inter_dims', 256 if fast_smoke else 1024)

		default_e_dims = self.options['e_dims'] = self.load_or_def_option('e_dims', 32 if fast_smoke else 64)
		default_d_dims = self.options['d_dims'] = self.options.get('d_dims', None)
		default_d_mask_dims = self.options['d_mask_dims'] = self.options.get('d_mask_dims', None)
		default_morph_factor = self.options['morph_factor'] = self.options.get('morph_factor', 0.5)
		default_uniform_yaw = self.options['uniform_yaw'] = self.load_or_def_option('uniform_yaw', False)
		default_blur_out_mask = self.options['blur_out_mask'] = self.load_or_def_option('blur_out_mask', False)
		default_lr_dropout = self.options['lr_dropout'] = self.load_or_def_option('lr_dropout', 'n')
		default_random_warp = self.options['random_warp'] = self.load_or_def_option('random_warp', False if fast_smoke else True)
		default_ct_mode = self.options['ct_mode'] = self.load_or_def_option('ct_mode', 'none')
		default_clipgrad = self.options['clipgrad'] = self.load_or_def_option('clipgrad', False)

		# Freeze layer options
		default_freeze_encoder = self.options['freeze_encoder'] = self.load_or_def_option('freeze_encoder', False)
		default_freeze_inter_src = self.options['freeze_inter_src'] = self.load_or_def_option('freeze_inter_src', False)
		default_freeze_inter_dst = self.options['freeze_inter_dst'] = self.load_or_def_option('freeze_inter_dst', False)
		default_freeze_decoder_mask = self.options['freeze_decoder_mask'] = self.load_or_def_option('freeze_decoder_mask', False)

		ask_override = self.ask_override()
		if self.is_first_run() or ask_override:
			self.ask_backup_interval()
			self.ask_max_backups()
			self.ask_crash_threshold()
			self.ask_write_preview_history()
			self.ask_target_iter()
			self.ask_random_src_flip()
			self.ask_random_dst_flip()
			self.ask_batch_size(1 if fast_smoke else 8)

		if self.is_first_run():
			resolution = io.input_int(
				"分辨率",
				default_resolution,
				add_info="64-640",
				help_message="更高分辨率需要更多显存和训练时间。该值会自动调整为 32 的倍数。",
			)
			resolution = np.clip((resolution // 32) * 32, 64, 640)
			self.options['resolution'] = int(resolution)
			self.options['face_type'] = io.input_str("人脸类型", default_face_type, ['f', 'wf', 'head'], help_message="wf=whole_face（整脸） / head（头部）").lower()

		default_d_dims = self.options['d_dims'] = self.load_or_def_option('d_dims', 64)

		default_d_mask_dims = default_d_dims // 3
		default_d_mask_dims += default_d_mask_dims % 2
		default_d_mask_dims = self.options['d_mask_dims'] = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

		if self.is_first_run():
			self.options['ae_dims'] = int(
				np.clip(
					io.input_int(
						"自编码器维度（AE dims）",
						default_ae_dims,
						add_info="32-1024",
						help_message="所有人脸信息会被压缩到 AE dims 中。维度越大越好，但更占显存。",
					),
					32,
					1024,
				)
			)
			self.options['inter_dims'] = int(
				np.clip(
					io.input_int(
						"中间层维度（Inter dims）",
						default_inter_dims,
						add_info="32-2048",
						help_message="应大于等于自编码器维度（AE dims）。",
					),
					32,
					2048,
				)
			)

			e_dims = int(
				np.clip(
					io.input_int(
						"编码器维度（E dims）",
						default_e_dims,
						add_info="16-256",
						help_message="维度越大越容易学习更多面部特征并获得更锐利的结果，但更占显存。",
					),
					16,
					256,
				)
			)
			self.options['e_dims'] = e_dims + e_dims % 2

			d_dims = int(
				np.clip(
					io.input_int(
						"解码器维度（D dims）",
						default_d_dims,
						add_info="16-256",
						help_message="维度越大越容易学习更多面部特征并获得更锐利的结果，但更占显存。",
					),
					16,
					256,
				)
			)
			self.options['d_dims'] = d_dims + d_dims % 2

			d_mask_dims = int(
				np.clip(
					io.input_int(
						"解码器 Mask 维度（D mask dims）",
						default_d_mask_dims,
						add_info="16-256",
						help_message="通常 mask 维度 = 解码器维度 / 3。",
					),
					16,
					256,
				)
			)
			self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

			morph_factor = float(np.clip(io.input_number("形变系数（Morph factor）", default_morph_factor, add_info="0.1 .. 0.5", help_message="常用值为 0.5"), 0.1, 0.5))
			self.options['morph_factor'] = morph_factor

		if self.is_first_run() or ask_override:
			self.options['uniform_yaw'] = io.input_bool(
				"样本 yaw 均匀分布",
				default_uniform_yaw,
				help_message='当 faceset 侧脸样本较少导致侧脸模糊时，该选项有帮助。',
			)
			self.options['blur_out_mask'] = io.input_bool(
				"模糊 Mask 外围",
				default_blur_out_mask,
				help_message=(
					'对训练样本的人脸 mask 外侧邻近区域进行模糊处理，使换脸后脸部边缘附近背景更平滑、更不显眼。'
					'需要 src/dst faceset 都具备准确的 XSeg mask。'
				),
			)
			self.options['lr_dropout'] = io.input_str(
				"使用学习率 dropout",
				default_lr_dropout,
				['n', 'y', 'cpu'],
				help_message=(
					"当人脸训练足够后，可启用以获得额外锐度。\n"
					"n - 关闭\n"
					"y - 开启\n"
					"cpu - 在 CPU 上启用（节省显存）。"
				),
			)

			default_lr = self.options['lr'] = self.load_or_def_option('lr', 5e-5)
			self.options['lr'] = float(
				np.clip(
					io.input_number('学习率', default_lr, add_info='1e-5 ~ 1e-4', help_message='训练学习率。Adam默认5e-5。'),
					1e-6, 1e-3,
				)
			)

			default_lr_scheduler = self.options['lr_scheduler'] = self.load_or_def_option('lr_scheduler', 'none')
			self.options['lr_scheduler'] = io.input_str(
				'学习率调度器', default_lr_scheduler, ['none', 'cosine'],
				help_message='cosine: 余弦退火·热重启（SGDR）—— 每次周期结束把 lr 弹回峰值重开一轮，'
				             '不是单调退火。none: 固定学习率。',
			)
			if self.options['lr_scheduler'] == 'cosine':
				default_cos = self.options['lr_cos_period'] = self.load_or_def_option('lr_cos_period', 500)
				self.options['lr_cos_period'] = io.input_int(
					'余弦退火·热重启单轮长度(迭代)', default_cos, add_info='建议 500~2000',
					help_message='一轮热重启的迭代数：走完一轮 lr 弹回峰值重新开始。'
					             '注意「余弦退火」与「余弦退火·热重启」是两个不同的东西 —— '
					             '前者单调下降不重启（PyTorch 的 CosineAnnealingLR），'
					             '后者周期性重启（CosineAnnealingWarmRestarts）。这里用的是后者。',
				)

		default_gan_power = self.options['gan_power'] = self.load_or_def_option('gan_power', 0.0)
		default_gan_patch_size = self.options['gan_patch_size'] = self.load_or_def_option('gan_patch_size', self.options['resolution'] // 8)
		default_gan_dims = self.options['gan_dims'] = self.load_or_def_option('gan_dims', 16)

		if self.is_first_run() or ask_override:
			self.options['models_opt_on_gpu'] = io.input_bool(
				"将模型与优化器放在 GPU",
				default_models_opt_on_gpu,
				help_message=(
					"单 GPU 训练时，默认会把模型与优化器权重放在 GPU 上以加速。"
					"也可放到 CPU 以释放额外显存。"
				),
			)

			self.options['random_warp'] = io.input_bool(
				"启用样本随机形变（random warp）",
				default_random_warp,
				help_message=(
					"随机形变有助于泛化表情。"
					"当人脸训练足够后，可关闭以获得额外锐度。"
				),
			)

			self.options['gan_power'] = float(
				np.clip(
					io.input_number(
						"GAN 强度（GAN power）",
						default_gan_power,
						add_info="0.0 .. 5.0",
						help_message=(
							"迫使网络学习更细小的人脸细节。仅建议在关闭 random_warp 后开启。"
							"常用值 0.1。"
						),
					),
					0.0,
					5.0,
				)
			)

			if self.options['gan_power'] != 0.0:
				gan_patch_size = int(
					np.clip(
						io.input_int(
							"GAN patch 大小",
							default_gan_patch_size,
							add_info="3-640",
							help_message="常用值为 分辨率/8。",
						),
						3,
						640,
					)
				)
				self.options['gan_patch_size'] = gan_patch_size

				gan_dims = int(
					np.clip(
						io.input_int(
							"GAN 维度（GAN dims）",
							default_gan_dims,
							add_info="4-512",
							help_message="常用值 16。",
						),
						4,
						512,
					)
				)
				self.options['gan_dims'] = gan_dims

			self.options['ct_mode'] = io.input_str(
				"src faceset 的颜色迁移模式",
				default_ct_mode,
				['none', 'rct', 'lct', 'mkl', 'idt', 'sot'],
				help_message=(
					"将 src 样本的颜色分布调整得更接近 dst。"
					"若 src faceset 足够多样，多数情况下 lct 就很好用。"
				),
			)
			self.options['clipgrad'] = io.input_bool(
				"启用梯度裁剪（clipgrad）",
				default_clipgrad,
				help_message="梯度裁剪可降低模型崩溃概率，但会牺牲训练速度。",
			)
			self.ask_gradient_checkpointing()

		default_fast_gen = self.load_or_def_option('use_fast_generator', False)
		if self.is_first_run() or ask_override:
			self.options['use_fast_generator'] = io.input_bool(
				'启用快速生成器（实验性，V4/V2）',
				default_fast_gen,
				help_message='使用新的多进程预取加载器，可显著提升数据加载速度。实验性功能。',
			)

		default_use_bf16 = self.load_or_def_option('use_bf16', False)
		if self.is_first_run() or ask_override:
			self.options['use_bf16'] = io.input_bool(
				'启用 BF16 混合精度训练',
				default_use_bf16,
				help_message='在支持 BF16 的 GPU（Ampere+）上可节省显存并加速训练。权重仍为 FP32。',
			)

		self.gan_model_changed = (default_gan_patch_size != self.options['gan_patch_size']) or (default_gan_dims != self.options['gan_dims'])

	# --- init ---
	def on_initialize(self):
		device_config = nn.getCurrentDeviceConfig()
		devices = device_config.devices
		self.model_data_format = "NCHW"
		nn.initialize(device_config, data_format=self.model_data_format)
		self.device = nn.device

		input_ch = 3
		resolution = self.resolution = int(self.options['resolution'])
		e_dims = int(self.options['e_dims'])
		ae_dims = int(self.options['ae_dims'])
		inter_dims = self.inter_dims = int(self.options['inter_dims'])
		inter_res = self.inter_res = resolution // 32
		d_dims = int(self.options['d_dims'])
		d_mask_dims = int(self.options['d_mask_dims'])
		self.face_type = {
			'f': FaceType.FULL,
			'wf': FaceType.WHOLE_FACE,
			'head': FaceType.HEAD,
		}[self.options['face_type']]
		self.morph_factor = float(self.options['morph_factor'])
		self.gan_power = float(self.options['gan_power'])
		random_warp = bool(self.options['random_warp'])
		self.blur_out_mask = bool(self.options['blur_out_mask'])
		self.use_bf16 = bool(self.options.get('use_bf16', False))

		ct_mode = self.options['ct_mode']
		if ct_mode == 'none':
			ct_mode = None

		use_fp16 = False
		use_bf16 = False
		if self.is_exporting:
			_prec = os.environ.get('DFM_EXPORT_PRECISION', '')
			if _prec == 'bf16':
				use_bf16 = True

		if use_bf16:
			conv_dtype = torch.bfloat16
		elif use_fp16:
			conv_dtype = torch.float16
		else:
			conv_dtype = torch.float32

		class Downscale(nn.ModelBase):
			def on_build(self, in_ch, out_ch, kernel_size=5):
				self.conv1 = nn.Conv2D(in_ch, out_ch, kernel_size=kernel_size, strides=2, padding='SAME', dtype=conv_dtype)

			def forward(self, x):
				return F.leaky_relu(self.conv1(x), 0.1)

		class Upscale(nn.ModelBase):
			def on_build(self, in_ch, out_ch, kernel_size=3):
				self.conv1 = nn.Conv2D(in_ch, out_ch * 4, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

			def forward(self, x):
				x = F.leaky_relu(self.conv1(x), 0.1)
				x = nn.depth_to_space(x, 2)
				return x

		class ResidualBlock(nn.ModelBase):
			def on_build(self, ch, kernel_size=3):
				self.conv1 = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
				self.conv2 = nn.Conv2D(ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

			def forward(self, inp):
				x = self.conv1(inp)
				x = F.leaky_relu(x, 0.2)
				x = self.conv2(x)
				x = F.leaky_relu(inp + x, 0.2)
				return x

		class Encoder(nn.ModelBase):
			def on_build(self):
				self.down1 = Downscale(input_ch, e_dims, kernel_size=5)
				self.res1 = ResidualBlock(e_dims)
				self.down2 = Downscale(e_dims, e_dims * 2, kernel_size=5)
				self.down3 = Downscale(e_dims * 2, e_dims * 4, kernel_size=5)
				self.down4 = Downscale(e_dims * 4, e_dims * 8, kernel_size=5)
				self.down5 = Downscale(e_dims * 8, e_dims * 8, kernel_size=5)
				self.res5 = ResidualBlock(e_dims * 8)
				self.dense1 = nn.Dense(((resolution // (2**5)) ** 2) * e_dims * 8, ae_dims)

			def forward(self, x):
				if use_bf16:
					x = x.to(torch.bfloat16)
				elif use_fp16:
					x = x.to(torch.float16)
				x = self.down1(x)
				x = self.res1(x)
				x = self.down2(x)
				x = self.down3(x)
				x = self.down4(x)
				x = self.down5(x)
				x = self.res5(x)
				if use_bf16 or use_fp16:
					x = x.to(torch.float32)
				x = nn.pixel_norm(nn.flatten(x), axes=-1)
				x = self.dense1(x)
				return x

		class Inter(nn.ModelBase):
			def on_build(self):
				self.dense2 = nn.Dense(ae_dims, inter_res * inter_res * inter_dims)

			def forward(self, inp):
				x = self.dense2(inp)
				x = nn.reshape_4D(x, inter_res, inter_res, inter_dims)
				return x

		class Decoder(nn.ModelBase):
			def on_build(self):
				self.upscale0 = Upscale(inter_dims, d_dims * 8, kernel_size=3)
				self.upscale1 = Upscale(d_dims * 8, d_dims * 8, kernel_size=3)
				self.upscale2 = Upscale(d_dims * 8, d_dims * 4, kernel_size=3)
				self.upscale3 = Upscale(d_dims * 4, d_dims * 2, kernel_size=3)

				self.res0 = ResidualBlock(d_dims * 8, kernel_size=3)
				self.res1 = ResidualBlock(d_dims * 8, kernel_size=3)
				self.res2 = ResidualBlock(d_dims * 4, kernel_size=3)
				self.res3 = ResidualBlock(d_dims * 2, kernel_size=3)

				self.upscalem0 = Upscale(inter_dims, d_mask_dims * 8, kernel_size=3)
				self.upscalem1 = Upscale(d_mask_dims * 8, d_mask_dims * 8, kernel_size=3)
				self.upscalem2 = Upscale(d_mask_dims * 8, d_mask_dims * 4, kernel_size=3)
				self.upscalem3 = Upscale(d_mask_dims * 4, d_mask_dims * 2, kernel_size=3)
				self.upscalem4 = Upscale(d_mask_dims * 2, d_mask_dims * 1, kernel_size=3)
				self.out_convm = nn.Conv2D(d_mask_dims * 1, 1, kernel_size=1, padding='SAME', dtype=conv_dtype)

				self.out_conv = nn.Conv2D(d_dims * 2, 3, kernel_size=1, padding='SAME', dtype=conv_dtype)
				self.out_conv1 = nn.Conv2D(d_dims * 2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
				self.out_conv2 = nn.Conv2D(d_dims * 2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)
				self.out_conv3 = nn.Conv2D(d_dims * 2, 3, kernel_size=3, padding='SAME', dtype=conv_dtype)

			def forward(self, z):
				if use_bf16:
					z = z.to(torch.bfloat16)
				elif use_fp16:
					z = z.to(torch.float16)

				x = self.upscale0(z)
				x = self.res0(x)
				x = self.upscale1(x)
				x = self.res1(x)
				x = self.upscale2(x)
				x = self.res2(x)
				x = self.upscale3(x)
				x = self.res3(x)

				# 4x RGB head + depth_to_space
				y = torch.cat((self.out_conv(x), self.out_conv1(x), self.out_conv2(x), self.out_conv3(x)), dim=nn.conv2d_ch_axis)
				y = torch.sigmoid(nn.depth_to_space(y, 2))

				m = self.upscalem0(z)
				m = self.upscalem1(m)
				m = self.upscalem2(m)
				m = self.upscalem3(m)
				m = self.upscalem4(m)
				m = torch.sigmoid(self.out_convm(m))

				if use_bf16 or use_fp16:
					y = y.to(torch.float32)
					m = m.to(torch.float32)
				return y, m

		# Build modules
		self.encoder = Encoder(name='encoder')
		self.inter_src = Inter(name='inter_src')
		self.inter_dst = Inter(name='inter_dst')
		self.decoder = Decoder(name='decoder')

		# Ensure on device
		for _m in (self.encoder, self.inter_src, self.inter_dst, self.decoder):
			try:
				_m.to(self.device)
			except Exception:
				pass

		self.model_filename_list = [
			[self.encoder, 'encoder.pth'],
			[self.inter_src, 'inter_src.pth'],
			[self.inter_dst, 'inter_dst.pth'],
			[self.decoder, 'decoder.pth'],
		]

		# Optimizers / GAN
		if self.is_training:
			lr = self.options.get('lr', 5e-5)
			clipnorm = 1.0 if bool(self.options['clipgrad']) else 0.0
			if self.options['lr_dropout'] in ['y', 'cpu']:
				lr_cos = self.options.get('lr_cos_period', 500)
				lr_dropout = 0.3
			else:
				lr_cos = self.options.get('lr_cos_period', 0)
				lr_dropout = 1.0

			self.G_weights = self.encoder.get_weights() + self.decoder.get_weights() + self.inter_src.get_weights() + self.inter_dst.get_weights()
			self.src_dst_opt = nn.AdaBelief(self.G_weights, lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='src_dst_opt')
			self.model_filename_list += [(self.src_dst_opt, 'src_dst_opt.pth')]

			if self.gan_power != 0.0:
				self.GAN = nn.UNetPatchDiscriminator(
					patch_size=int(self.options['gan_patch_size']),
					in_ch=input_ch,
					base_ch=int(self.options['gan_dims']),
					name='GAN',
				)
				try:
					self.GAN.to(self.device)
				except Exception:
					pass
				self.GAN_opt = nn.AdaBelief(self.GAN.get_weights(), lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, name='GAN_opt')
				self.model_filename_list += [
					[self.GAN, 'GAN.pth'],
					[self.GAN_opt, 'GAN_opt.pth'],
				]

		# Load weights
		self._module_info_list = []
		for model, filename in io.progress_bar_generator(self.model_filename_list, 'Initializing models'):
			do_init = self.is_first_run()
			if self.is_training and self.gan_power != 0.0 and getattr(self, 'GAN', None) is model and getattr(self, 'gan_model_changed', False):
				do_init = True
			load_ok = False
			if not do_init:
				filepath = self.get_strpath_storage_for_file(filename)
				if Path(filepath).exists():
					try:
						load_ok = model.load_weights(filepath)
					except Exception:
						load_ok = False
					do_init = not load_ok
				else:
					do_init = True

			if do_init:
				try:
					model.init_weights()
				except Exception:
					pass
				status = '已重置'
			else:
				status = '已加载'

			weights = model.get_weights()
			param_count = sum(w.numel() for w in weights) if weights else 0
			module_name = filename.replace('.pth', '')
			self._module_info_list.append((module_name, param_count, status))

		# Split decoder entry: add separate mask sub-entry in module info
		_MASK_LAYER_NAMES = ('upscalem0', 'upscalem1', 'upscalem2', 'upscalem3', 'upscalem4', 'out_convm')
		_new_module_list = []
		for nm, pc, st in self._module_info_list:
			_new_module_list.append((nm, pc, st))
			if nm == 'decoder':
				_dec = getattr(self, nm, None)
				if _dec is not None:
					_mask_pc = 0
					for _mln in _MASK_LAYER_NAMES:
						_ml = getattr(_dec, _mln, None)
						if _ml is not None:
							_mask_pc += sum(w.numel() for w in _ml.get_weights())
					if _mask_pc > 0:
						_new_module_list.append(('decoder_mask', _mask_pc, st))
						_new_module_list[-2] = ('decoder', pc - _mask_pc, st)
		self._module_info_list = _new_module_list

		# User-controlled layer freeze
		if self.is_training:
			freeze_enc = bool(self.options.get('freeze_encoder', False))
			freeze_inter_src_val = bool(self.options.get('freeze_inter_src', False))
			freeze_inter_dst_val = bool(self.options.get('freeze_inter_dst', False))
			freeze_dm = bool(self.options.get('freeze_decoder_mask', False))
			if freeze_enc:
				for w in self.encoder.get_weights():
					if hasattr(w, 'requires_grad'):
						w.requires_grad_(False)
			if freeze_inter_src_val:
				for w in self.inter_src.get_weights():
					if hasattr(w, 'requires_grad'):
						w.requires_grad_(False)
			if freeze_inter_dst_val:
				for w in self.inter_dst.get_weights():
					if hasattr(w, 'requires_grad'):
						w.requires_grad_(False)
			if freeze_dm:
				_dec = getattr(self, 'decoder', None)
				if _dec is not None:
					for _mln in ('upscalem0', 'upscalem1', 'upscalem2', 'upscalem3', 'upscalem4', 'out_convm'):
						_ml = getattr(_dec, _mln, None)
						if _ml is not None:
							for w in _ml.get_weights():
								if hasattr(w, 'requires_grad'):
									w.requires_grad_(False)
			# Update module info for user-frozen layers
			for i, (nm, pc, st) in enumerate(self._module_info_list):
				if st != '已冻结':
					user_frozen = (
						(freeze_enc and nm == 'encoder')
						or (freeze_inter_src_val and nm == 'inter_src')
						or (freeze_inter_dst_val and nm == 'inter_dst')
						or (freeze_dm and nm == 'decoder_mask')
					)
					if user_frozen:
						self._module_info_list[i] = (nm, pc, '已冻结')

		# Sample generators
		if self.is_training:
			training_data_src_path = self.training_data_src_path
			training_data_dst_path = self.training_data_dst_path

			use_fast = self.options.get('use_fast_generator', False)

			if use_fast:
				# === 快速生成器路径（V4 loader + V2 generator）===
				src_loader = SampleLoaderV4(
					aligned_path=training_data_src_path,
					batch_size=self.get_batch_size(),
					resolution=resolution,
					use_yaw_sampling=bool(self.options['uniform_yaw']),
				)
				dst_loader = SampleLoaderV4(
					aligned_path=training_data_dst_path,
					batch_size=self.get_batch_size(),
					resolution=resolution,
					use_yaw_sampling=bool(self.options['uniform_yaw']),
				)

				src_ct_loader = dst_loader if ct_mode is not None else None

				gen_src_outputs = [
					{
						'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
						'warp': random_warp,
						'transform': True,
						'channel_type': SampleProcessor.ChannelType.BGR,
						'ct_mode': ct_mode,
						'face_type': self.face_type,
						'data_format': nn.data_format,
						'resolution': resolution,
					},
					{
						'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
						'warp': False,
						'transform': True,
						'channel_type': SampleProcessor.ChannelType.BGR,
						'ct_mode': ct_mode,
						'face_type': self.face_type,
						'data_format': nn.data_format,
						'resolution': resolution,
					},
					{
						'sample_type': SampleProcessor.SampleType.FACE_MASK,
						'warp': False,
						'transform': True,
						'channel_type': SampleProcessor.ChannelType.G,
						'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
						'face_type': self.face_type,
						'data_format': nn.data_format,
						'resolution': resolution,
					},
					{
						'sample_type': SampleProcessor.SampleType.FACE_MASK,
						'warp': False,
						'transform': True,
						'channel_type': SampleProcessor.ChannelType.G,
						'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
						'face_type': self.face_type,
						'data_format': nn.data_format,
						'resolution': resolution,
					},
				]

				gen_dst_outputs = [
					{
					'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
					'warp': random_warp,
					'transform': True,
					'channel_type': SampleProcessor.ChannelType.BGR,
					'face_type': self.face_type,
					'data_format': nn.data_format,
					'resolution': resolution,
					},
					{
					'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
					'warp': False,
					'transform': True,
					'channel_type': SampleProcessor.ChannelType.BGR,
					'face_type': self.face_type,
					'data_format': nn.data_format,
					'resolution': resolution,
					},
					{
					'sample_type': SampleProcessor.SampleType.FACE_MASK,
					'warp': False,
					'transform': True,
					'channel_type': SampleProcessor.ChannelType.G,
					'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
					'face_type': self.face_type,
					'data_format': nn.data_format,
					'resolution': resolution,
					},
					{
					'sample_type': SampleProcessor.SampleType.FACE_MASK,
					'warp': False,
					'transform': True,
					'channel_type': SampleProcessor.ChannelType.G,
					'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
					'face_type': self.face_type,
					'data_format': nn.data_format,
					'resolution': resolution,
					},
				]

				self.set_training_data_generators(
				[
					SampleGeneratorV2(
						loader=src_loader,
						sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_src_flip),
						output_sample_types=gen_src_outputs,
						resolution=resolution,
						debug=self.is_debug(),
						ct_loader=src_ct_loader,
					),
					SampleGeneratorV2(
						loader=dst_loader,
						sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_dst_flip),
						output_sample_types=gen_dst_outputs,
						resolution=resolution,
						debug=self.is_debug(),
					),
				]
				)
			else:
				# === 原始生成器路径（SampleGeneratorFace）===
				random_ct_samples_path = training_data_dst_path if ct_mode is not None else None

				forced_gens = os.environ.get('DFL_GENERATORS_COUNT', None)
				if forced_gens is not None and str(forced_gens).strip() != '':
					try:
						forced_gens_i = max(1, int(forced_gens))
					except Exception:
						forced_gens_i = None
				else:
					forced_gens_i = None

				cpu_count = multiprocessing.cpu_count()
				src_generators_count = cpu_count // 2
				dst_generators_count = cpu_count // 2
				if ct_mode is not None:
					src_generators_count = int(src_generators_count * 1.5)

				if forced_gens_i is not None:
					src_generators_count = forced_gens_i
					dst_generators_count = forced_gens_i

					self.set_training_data_generators(
					[
						SampleGeneratorFace(
							training_data_src_path,
							random_ct_samples_path=random_ct_samples_path,
							debug=self.is_debug(),
							batch_size=self.get_batch_size(),
							sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_src_flip),
							output_sample_types=[
								{
								'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
								'warp': random_warp,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.BGR,
								'ct_mode': ct_mode,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.BGR,
								'ct_mode': ct_mode,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_MASK,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.G,
								'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_MASK,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.G,
								'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
						],
						uniform_yaw_distribution=bool(self.options['uniform_yaw']),
						generators_count=src_generators_count,
						rnd_seed=int(os.environ.get('DFL_INDEX_SEED', '0')) if os.environ.get('DFL_INDEX_SEED', '').strip() != '' else None,
					),
					SampleGeneratorFace(
						training_data_dst_path,
						debug=self.is_debug(),
						batch_size=self.get_batch_size(),
						sample_process_options=SampleProcessor.Options(scale_range=[-0.15, 0.15], random_flip=self.random_dst_flip),
						output_sample_types=[
							{
								'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
								'warp': random_warp,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.BGR,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.BGR,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_MASK,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.G,
								'face_mask_type': SampleProcessor.FaceMaskType.FULL_FACE,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
							{
								'sample_type': SampleProcessor.SampleType.FACE_MASK,
								'warp': False,
								'transform': True,
								'channel_type': SampleProcessor.ChannelType.G,
								'face_mask_type': SampleProcessor.FaceMaskType.EYES_MOUTH,
								'face_type': self.face_type,
								'data_format': nn.data_format,
								'resolution': resolution,
							},
						],
						uniform_yaw_distribution=bool(self.options['uniform_yaw']),
						generators_count=dst_generators_count,
						rnd_seed=int(os.environ.get('DFL_INDEX_SEED', '0')) if os.environ.get('DFL_INDEX_SEED', '').strip() != '' else None,
					),
				]
			)

	# --- helpers ---
	def _np_to_torch(self, arr: np.ndarray) -> torch.Tensor:
		if isinstance(arr, torch.Tensor):
			t = arr
		else:
			t = torch.from_numpy(arr)
		if t.dtype != torch.float32:
			t = t.float()
		return t.to(self.device, non_blocking=True)

	def _random_inter_binomial_mask(self, batch: int, inter_dims: int, inter_dims_bin: int) -> torch.Tensor:
		# TF 版：每个样本拼接 [1...1,0...0] 然后 random.shuffle。
		# 这里用同等的“随机置换”实现。
		if inter_dims_bin <= 0:
			mask = torch.zeros((batch, inter_dims), dtype=torch.float32)
			return mask
		if inter_dims_bin >= inter_dims:
			mask = torch.ones((batch, inter_dims), dtype=torch.float32)
			return mask

		base = torch.cat(
			[
				torch.ones((inter_dims_bin,), dtype=torch.float32),
				torch.zeros((inter_dims - inter_dims_bin,), dtype=torch.float32),
			],
			0,
		)
		masks = []
		for _ in range(batch):
			idx = torch.randperm(inter_dims)
			masks.append(base[idx])
		mask = torch.stack(masks, 0)
		return mask

	def _combine_src_dst_code(self, dst_inter_src_code: torch.Tensor, dst_inter_dst_code: torch.Tensor, morph_value):
		# 输出：concat([dst_inter_src[:k], dst_inter_dst[k:]]) along channel.
		inter_dims = int(self.inter_dims)
		inter_res = int(self.inter_res)
		if torch.is_tensor(morph_value):
			mv = morph_value.reshape(-1)[0].float()
			k_f = torch.clamp(mv, 0.0, 1.0) * float(inter_dims)
			k = torch.clamp(k_f.floor().long(), 0, inter_dims)
			# mask: (1, C, 1, 1)
			c_idx = torch.arange(inter_dims, device=dst_inter_src_code.device).view(1, inter_dims, 1, 1)
			m = (c_idx < k).to(dst_inter_src_code.dtype)
			return dst_inter_src_code * m + dst_inter_dst_code * (1.0 - m)

		k = int(inter_dims * float(morph_value))
		k = int(np.clip(k, 0, inter_dims))
		if k == 0:
			return dst_inter_dst_code
		if k == inter_dims:
			return dst_inter_src_code
		return torch.cat((dst_inter_src_code[:, :k, :, :], dst_inter_dst_code[:, k:, :, :]), dim=1)

	# --- forward ---
	def _forward(self, warped_src: torch.Tensor, warped_dst: torch.Tensor, morph_value=None):
		# warped_*: NCHW float32
		# Encode
		src_code = self.encoder(warped_src)
		dst_code = self.encoder(warped_dst)

		src_inter_src = self.inter_src(src_code)
		src_inter_dst = self.inter_dst(src_code)
		dst_inter_src = self.inter_src(dst_code)
		dst_inter_dst = self.inter_dst(dst_code)

		# morph_factor mixing (src)
		inter_dims_bin = int(self.inter_dims * float(self.morph_factor))
		mask = self._random_inter_binomial_mask(src_inter_src.shape[0], int(self.inter_dims), inter_dims_bin)
		mask = mask.to(device=src_inter_src.device, dtype=src_inter_src.dtype).detach()[:, :, None, None]
		src_mixed_code = src_inter_src * mask + src_inter_dst * (1.0 - mask)

		dst_mixed_code = dst_inter_dst

		# morph_value merge code
		if morph_value is None:
			morph_value = 0.0
		src_dst_code = self._combine_src_dst_code(dst_inter_src, dst_inter_dst, morph_value)

		pred_src_src, pred_src_srcm = self.decoder(src_mixed_code)
		pred_dst_dst, pred_dst_dstm = self.decoder(dst_mixed_code)
		pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)

		return {
			'pred_src_src': pred_src_src,
			'pred_src_srcm': pred_src_srcm,
			'pred_dst_dst': pred_dst_dst,
			'pred_dst_dstm': pred_dst_dstm,
			'pred_src_dst': pred_src_dst,
			'pred_src_dstm': pred_src_dstm,
			'dst_inter_dst': dst_inter_dst,
			'dst_inter_src': dst_inter_src,
		}

	
	def _forward_and_losses(self, warped_src, target_src, target_srcm, target_srcm_em,
							warped_dst, target_dst, target_dstm, target_dstm_em):
		"""Combined forward + losses."""
		with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
			if self.options.get('gradient_checkpointing', False):
				fw = torch.utils.checkpoint.checkpoint(self._forward, warped_src, warped_dst, 0.0, use_reentrant=False)
			else:
				fw = self._forward(warped_src, warped_dst, morph_value=0.0)

		if self.use_bf16:
			fw = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in fw.items()}
		pred_src_src = fw["pred_src_src"]
		pred_src_srcm = fw["pred_src_srcm"]
		pred_dst_dst = fw["pred_dst_dst"]
		pred_dst_dstm = fw["pred_dst_dstm"]

		# mask blur
		k_blur = max(1, int(self.resolution // 32))
		target_srcm_gblur = nn.gaussian_blur(target_srcm, k_blur)
		target_dstm_gblur = nn.gaussian_blur(target_dstm, k_blur)
		target_srcm_blur = torch.clamp(target_srcm_gblur, 0.0, 0.5) * 2.0
		target_dstm_blur = torch.clamp(target_dstm_gblur, 0.0, 0.5) * 2.0
		target_srcm_anti_blur = 1.0 - target_srcm_blur
		target_dstm_anti_blur = 1.0 - target_dstm_blur

		# blur_out_mask (modifies targets in-place)
		if self.blur_out_mask:
			sigma = float(self.resolution) / 128.0
			target_srcm_anti = 1.0 - target_srcm
			target_dstm_anti = 1.0 - target_dstm
			x = nn.gaussian_blur(target_src * target_srcm_anti, sigma)
			y = 1.0 - nn.gaussian_blur(target_srcm, sigma)
			y = torch.where(y == 0, torch.ones_like(y), y)
			target_src = target_src * target_srcm + (x / y) * target_srcm_anti
			x = nn.gaussian_blur(target_dst * target_dstm_anti, sigma)
			y = 1.0 - nn.gaussian_blur(target_dstm, sigma)
			y = torch.where(y == 0, torch.ones_like(y), y)
			target_dst = target_dst * target_dstm + (x / y) * target_dstm_anti

		# masked/anti-masked
		target_src_masked = target_src * target_srcm_blur
		target_dst_masked = target_dst * target_dstm_blur
		target_src_anti_masked = target_src * target_srcm_anti_blur
		target_dst_anti_masked = target_dst * target_dstm_anti_blur
		pred_src_src_masked = pred_src_src * target_srcm_blur
		pred_dst_dst_masked = pred_dst_dst * target_dstm_blur
		pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur
		pred_dst_dst_anti_masked = pred_dst_dst * target_dstm_anti_blur

		# Losses
		fs1 = max(1, int(self.resolution / 11.6))
		fs2 = max(1, int(self.resolution / 23.2))
		src_loss_vec = torch.mean(5.0 * nn.dssim(target_src_masked, pred_src_src_masked, max_val=1.0, filter_size=fs1), dim=1).view(-1)
		src_loss_vec = src_loss_vec + torch.mean(5.0 * nn.dssim(target_src_masked, pred_src_src_masked, max_val=1.0, filter_size=fs2), dim=1).view(-1)
		dst_loss_vec = torch.mean(5.0 * nn.dssim(target_dst_masked, pred_dst_dst_masked, max_val=1.0, filter_size=fs1), dim=1).view(-1)
		dst_loss_vec = dst_loss_vec + torch.mean(5.0 * nn.dssim(target_dst_masked, pred_dst_dst_masked, max_val=1.0, filter_size=fs2), dim=1).view(-1)
		src_loss_vec = src_loss_vec + 10.0 * torch.mean((target_src_masked - pred_src_src_masked) ** 2, dim=[1, 2, 3])
		dst_loss_vec = dst_loss_vec + 10.0 * torch.mean((target_dst_masked - pred_dst_dst_masked) ** 2, dim=[1, 2, 3])
		src_loss_vec = src_loss_vec + 300.0 * torch.mean(torch.abs(target_src * target_srcm_em - pred_src_src * target_srcm_em), dim=[1, 2, 3])
		dst_loss_vec = dst_loss_vec + 300.0 * torch.mean(torch.abs(target_dst * target_dstm_em - pred_dst_dst * target_dstm_em), dim=[1, 2, 3])
		src_loss_vec = src_loss_vec + 10.0 * torch.mean((target_srcm - pred_src_srcm) ** 2, dim=[1, 2, 3])
		dst_loss_vec = dst_loss_vec + 10.0 * torch.mean((target_dstm - pred_dst_dstm) ** 2, dim=[1, 2, 3])

		G_loss = src_loss_vec.mean() + dst_loss_vec.mean()
		G_loss = G_loss + 0.1 * torch.mean((pred_dst_dst_anti_masked - target_dst_anti_masked) ** 2)
		G_loss = G_loss + 0.000001 * nn.total_variation_mse(pred_dst_dst_anti_masked)
		return G_loss, src_loss_vec, dst_loss_vec, fw

	# --- losses / train ---
	def train_one_step(self, warped_src, target_src, target_srcm, target_srcm_em, warped_dst, target_dst, target_dstm, target_dstm_em):
		warped_src = self._np_to_torch(warped_src)
		target_src = self._np_to_torch(target_src)
		target_srcm = self._np_to_torch(target_srcm)
		target_srcm_em = self._np_to_torch(target_srcm_em)
		warped_dst = self._np_to_torch(warped_dst)
		target_dst = self._np_to_torch(target_dst)
		target_dstm = self._np_to_torch(target_dstm)
		target_dstm_em = self._np_to_torch(target_dstm_em)

		# === Forward + Losses ===
		G_loss, src_loss_vec, dst_loss_vec, fw = self._forward_and_losses(
			warped_src, target_src, target_srcm, target_srcm_em,
			warped_dst, target_dst, target_dstm, target_dstm_em,
		)

		pred_src_src = fw['pred_src_src']
		pred_src_srcm = fw['pred_src_srcm']
		pred_dst_dst = fw['pred_dst_dst']
		pred_dst_dstm = fw['pred_dst_dstm']

		# mask blur (shared/GAN)
		k_blur = max(1, int(self.resolution // 32))
		target_srcm_gblur = nn.gaussian_blur(target_srcm, k_blur)
		target_dstm_gblur = nn.gaussian_blur(target_dstm, k_blur)
		target_srcm_blur = torch.clamp(target_srcm_gblur, 0.0, 0.5) * 2.0
		target_dstm_blur = torch.clamp(target_dstm_gblur, 0.0, 0.5) * 2.0
		target_srcm_anti_blur = 1.0 - target_srcm_blur
		target_dstm_anti_blur = 1.0 - target_dstm_blur

		# blur_out_mask (shared/GAN)
		if self.blur_out_mask:
			sigma = float(self.resolution) / 128.0
			target_srcm_anti = 1.0 - target_srcm
			target_dstm_anti = 1.0 - target_dstm
			x = nn.gaussian_blur(target_src * target_srcm_anti, sigma)
			y = 1.0 - nn.gaussian_blur(target_srcm, sigma)
			y = torch.where(y == 0, torch.ones_like(y), y)
			target_src = target_src * target_srcm + (x / y) * target_srcm_anti
			x = nn.gaussian_blur(target_dst * target_dstm_anti, sigma)
			y = 1.0 - nn.gaussian_blur(target_dstm, sigma)
			y = torch.where(y == 0, torch.ones_like(y), y)
			target_dst = target_dst * target_dstm + (x / y) * target_dstm_anti

		# masked/anti-masked (shared/GAN)
		target_src_masked = target_src * target_srcm_blur
		target_dst_masked = target_dst * target_dstm_blur
		target_src_anti_masked = target_src * target_srcm_anti_blur
		target_dst_anti_masked = target_dst * target_dstm_anti_blur
		pred_src_src_masked = pred_src_src * target_srcm_blur
		pred_dst_dst_masked = pred_dst_dst * target_dstm_blur
		pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur
		pred_dst_dst_anti_masked = pred_dst_dst * target_dstm_anti_blur


		# GAN
		D_gan_loss = None
		if self.gan_power != 0.0:
			# logits losses
			pred_src_src_d1, pred_src_src_d2 = self.GAN(pred_src_src_masked)
			pred_dst_dst_d1, pred_dst_dst_d2 = self.GAN(pred_dst_dst_masked)
			tgt_src_d1, tgt_src_d2 = self.GAN(target_src_masked)
			tgt_dst_d1, tgt_dst_d2 = self.GAN(target_dst_masked)

			def dloss_ones(logits):
				return torch.mean(F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none'), dim=[1, 2, 3])

			def dloss_zeros(logits):
				return torch.mean(F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none'), dim=[1, 2, 3])

			D_gan_loss_vec = (
				dloss_ones(tgt_src_d1)
				+ dloss_ones(tgt_src_d2)
				+ dloss_zeros(pred_src_src_d1.detach())
				+ dloss_zeros(pred_src_src_d2.detach())
				+ dloss_ones(tgt_dst_d1)
				+ dloss_ones(tgt_dst_d2)
				+ dloss_zeros(pred_dst_dst_d1.detach())
				+ dloss_zeros(pred_dst_dst_d2.detach())
			) * (1.0 / 8.0)
			D_gan_loss = D_gan_loss_vec.mean()

			G_loss = G_loss + self.gan_power * (
				dloss_ones(pred_src_src_d1).mean()
				+ dloss_ones(pred_src_src_d2).mean()
				+ dloss_ones(pred_dst_dst_d1).mean()
				+ dloss_ones(pred_dst_dst_d2).mean()
			)

			# Minimal src-src-bg rec + tv
			G_loss = G_loss + 0.000001 * nn.total_variation_mse(pred_src_src)
			G_loss = G_loss + 0.02 * torch.mean((pred_src_src_anti_masked - target_src_anti_masked) ** 2)

		# Update generator
		self.src_dst_opt.zero_grad()
		G_loss.backward()
		self.src_dst_opt.step()

		# Update GAN
		if D_gan_loss is not None:
			self.GAN_opt.zero_grad()
			D_gan_loss.backward()
			self.GAN_opt.step()

		# save per-sample vectors for WebUI preview
		self._last_src_loss_per_sample = src_loss_vec.detach().cpu()
		self._last_dst_loss_per_sample = dst_loss_vec.detach().cpu()
		return float(src_loss_vec.mean().detach().cpu()), float(dst_loss_vec.mean().detach().cpu())

	# --- ModelBase hooks ---
	def get_model_filename_list(self):
		return self.model_filename_list

	def onSave(self):
		for model, filename in io.progress_bar_generator(self.get_model_filename_list(), 'Saving', leave=False):
			model.save_weights(self.get_strpath_storage_for_file(filename))

	def should_save_preview_history(self):
		return (not io.is_colab() and self.iter % (10 * (max(1, self.resolution // 64))) == 0) or (io.is_colab() and self.iter % 100 == 0)

	def onTrainOneIter(self):
		((warped_src, target_src, target_srcm, target_srcm_em), (warped_dst, target_dst, target_dstm, target_dstm_em)) = self.generate_next_samples()
		src_loss, dst_loss = self.train_one_step(
			warped_src,
			target_src,
			target_srcm,
			target_srcm_em,
			warped_dst,
			target_dst,
			target_dstm,
			target_dstm_em,
		)
		return (('src_loss', src_loss), ('dst_loss', dst_loss))

	# --- preview / merge ---
	def AE_view(self, warped_src, warped_dst, morph_value):
		warped_src_t = self._np_to_torch(warped_src)
		warped_dst_t = self._np_to_torch(warped_dst)
		with torch.no_grad():
			with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
				fw = self._forward(warped_src_t, warped_dst_t, morph_value=float(morph_value))
			out = [
				fw['pred_src_src'],
				fw['pred_dst_dst'],
				fw['pred_dst_dstm'],
				fw['pred_src_dst'],
				fw['pred_src_dstm'],
			]
			out = [t.detach().cpu().float().numpy() for t in out]
		return out

	def AE_merge(self, warped_dst, morph_value):
		warped_dst_t = self._np_to_torch(warped_dst)
		with torch.no_grad():
			with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
				dst_code = self.encoder(warped_dst_t)
				dst_inter_src = self.inter_src(dst_code)
				dst_inter_dst = self.inter_dst(dst_code)
				src_dst_code = self._combine_src_dst_code(dst_inter_src, dst_inter_dst, float(morph_value))
				pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)
				_, pred_dst_dstm = self.decoder(dst_inter_dst)
			out = [pred_src_dst, pred_dst_dstm, pred_src_dstm]
			out = [t.detach().cpu().float().numpy() for t in out]
		return out

	@staticmethod
	def _draw_dynamic_text(cell, text, iter):
		"""在 cell (H,W,3) float [0,1] 上绘制动态蓝紫渐变文字（底左），原地修改。
		渐变色沿 45°（左上→右下）随时间流动。"""
		if not text:
			return
		h, w = cell.shape[:2]
		font_size = max(10, h // 25)
		pil_font = _get_pil_font(None, font_size)

		from PIL import Image, ImageDraw
		tmp = Image.new('RGB', (1, 1))
		bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=pil_font)
		tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

		pad = 3
		tx, ty = pad, h - th - pad

		# 用 PIL 渲染文字遮罩（白字黑底）
		mask = Image.new('L', (w, h), 0)
		ImageDraw.Draw(mask).text((tx, ty), text, font=pil_font, fill=255)
		mask_np = np.asarray(mask, dtype=np.float32) / 255.0

		# 45° 梯度位置（左上→右下）
		yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
		d = ((xx - tx) + (yy - ty)).astype(np.float32) / max(1, tw + th)
		d = np.clip(d, 0, 1)

		t = (iter or 0) * 0.015

		# HSV：蓝→紫渐变，随时间波动
		hue = 0.62 + d * 0.23 + np.sin(t + d * np.pi * 2) * 0.04
		sat = 0.85 + np.sin(t * 0.7 + d) * 0.15
		val = 0.75 + np.sin(t * 0.5 + d * 1.5) * 0.25

		# HSV → RGB 矢量转换
		hsv_h = hue % 1.0
		hi = (hsv_h * 6).astype(np.int32) % 6
		f = hsv_h * 6 - np.floor(hsv_h * 6)
		p = val * (1.0 - sat)
		q = val * (1.0 - f * sat)
		t_ = val * (1.0 - (1.0 - f) * sat)

		r = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
					  [val, q, p, p, t_, val])
		g = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
					  [t_, val, val, q, p, p])
		b = np.select([hi==0, hi==1, hi==2, hi==3, hi==4, hi==5],
					  [p, p, t_, val, val, q])

		# 逐文字像素写入
		for c_idx, color in enumerate([b, g, r]):
			cell[:, :, c_idx] = np.where(mask_np > 0, color, cell[:, :, c_idx])

	@staticmethod
	def _draw_adaptive_text(cell, text):
		"""在 cell (H,W,3) float [0,1] 上绘制反色文字（底左），原地修改。
		WCAG 亮度感知 L = 0.2126R + 0.7152G + 0.0722B，文字灰度 = 1 - L。"""
		if not text:
			return
		h, w = cell.shape[:2]
		font_size = max(10, h // 25)
		pil_font = _get_pil_font(None, font_size)

		from PIL import Image, ImageDraw
		tmp = Image.new('RGB', (1, 1))
		bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=pil_font)
		tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

		pad = 3
		tx, ty = pad, h - th - pad

		mask = Image.new('L', (w, h), 0)
		ImageDraw.Draw(mask).text((tx, ty), text, font=pil_font, fill=255)
		mask_np = np.asarray(mask, dtype=np.float32) / 255.0

		lum = 0.2126 * cell[:, :, 2] + 0.7152 * cell[:, :, 1] + 0.0722 * cell[:, :, 0]
		inv_lum = 1.0 - lum

		for c in range(3):
			cell[:, :, c] = np.where(mask_np > 0, inv_lum, cell[:, :, c])

	def onGetPreview(self, samples, for_history=False):
		((warped_src, target_src, target_srcm, target_srcm_em), (warped_dst, target_dst, target_dstm, target_dstm_em)) = samples

		S, D, SS, DD, DDM_000, SD_000, SDM_000 = [
			np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0)
			for x in ([target_src, target_dst] + self.AE_view(target_src, target_dst, 0.0))
		]
		_ddm_1ch = DDM_000
		_sdm_1ch = SDM_000

		WS = np.clip(nn.to_data_format(warped_src, 'NHWC', self.model_data_format), 0.0, 1.0)
		WD = np.clip(nn.to_data_format(warped_dst, 'NHWC', self.model_data_format), 0.0, 1.0)

		_, _, DDM_025, SD_025, SDM_025 = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(target_src, target_dst, 0.25)]
		_, _, DDM_050, SD_050, SDM_050 = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(target_src, target_dst, 0.50)]
		_, _, DDM_065, SD_065, SDM_065 = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(target_src, target_dst, 0.65)]
		_, _, DDM_075, SD_075, SDM_075 = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(target_src, target_dst, 0.75)]
		_, _, DDM_100, SD_100, SDM_100 = [np.clip(nn.to_data_format(x, 'NHWC', self.model_data_format), 0.0, 1.0) for x in self.AE_view(target_src, target_dst, 1.00)]
		_ddm_100_1ch = DDM_100
		_sdm_100_1ch = SDM_100

		(
			DDM_000,
			DDM_025,
			SDM_025,
			DDM_050,
			SDM_050,
			DDM_065,
			SDM_065,
			DDM_075,
			SDM_075,
			DDM_100,
			SDM_100,
		) = [
			np.repeat(x, (3,), -1)
			for x in (
				DDM_000,
				DDM_025,
				SDM_025,
				DDM_050,
				SDM_050,
				DDM_065,
				SDM_065,
				DDM_075,
				SDM_075,
				DDM_100,
				SDM_100,
			)
		]
		n_samples = self.get_batch_size()
		# (text labels rendered as HTML overlays in WebUI — not drawn on image)
		result = []

		# tab 1: 原始图片
		st = []
		for j in range(n_samples):
			st.append(np.concatenate((S[j], SS[j], D[j], DD[j], SD_100[j]), axis=1))
		result.append(("原始图片", np.concatenate(st, axis=0)))
		# tab 2: 遮罩下
		st = []
		for j in range(n_samples):
			ar = (S[j] * target_srcm[j], SS[j],
			      D[j] * target_dstm[j], DD[j] * DDM_000[j],
			      SD_100[j] * DDM_100[j] * SDM_100[j])
			st.append(np.concatenate(ar, axis=1))
		result.append(("遮罩下", np.concatenate(st, axis=0)))
		# Store per-column masks for GUI gradient compositing
		_col4_list = []
		for j in range(n_samples):
			_col4_list.append(_ddm_100_1ch[j] * _sdm_100_1ch[j])
		self._preview_masks = {
			'col0': target_srcm[:n_samples],
			'col2': target_dstm[:n_samples],
			'col3': _ddm_1ch[:n_samples],
			'col4': np.stack(_col4_list, axis=0),
		}
		# tab 3: 原始输入
		st = []
		for j in range(n_samples):
			st.append(np.concatenate((WS[j], SS[j], WD[j], DD[j], SD_100[j]), axis=1))
		result.append(("原始输入", np.concatenate(st, axis=0)))

		# morph list (keep original)
		st = [np.concatenate((DD[i], SD_025[i], SD_050[i]), axis=1)]
		st += [np.concatenate((SD_065[i], SD_075[i], SD_100[i]), axis=1)]
		result += [('AMP morph list', np.concatenate(st, axis=0))]

		st = [np.concatenate((DD[i], SD_025[i] * DDM_025[i] * SDM_025[i], SD_050[i] * DDM_050[i] * SDM_050[i]), axis=1)]
		st += [
			np.concatenate(
				(
					SD_065[i] * DDM_065[i] * SDM_065[i],
					SD_075[i] * DDM_075[i] * SDM_075[i],
					SD_100[i] * DDM_100[i] * SDM_100[i],
				),
				axis=1,
			)
		]
		result += [('AMP morph list masked', np.concatenate(st, axis=0))]

		return result

	def predictor_func(self, face, morph_value):
		face = nn.to_data_format(face[None, ...], self.model_data_format, 'NHWC')
		bgr, mask_dst_dstm, mask_src_dstm = [
			nn.to_data_format(x, 'NHWC', self.model_data_format).astype(np.float32) for x in self.AE_merge(face, morph_value)
		]
		return bgr[0], mask_src_dstm[0][..., 0], mask_dst_dstm[0][..., 0]

	def get_MergerConfig(self):
		morph_factor = float(np.clip(io.input_number("Morph factor", 1.0, add_info="0.0 .. 1.0"), 0.0, 1.0))

		def predictor_morph(face):
			return self.predictor_func(face, morph_factor)

		import merger

		return predictor_morph, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode='overlay')

	def export_dfm(self):
		"""Export model to .dfm (ONNX) compatible with DeepFaceLab merger pipeline."""
		output_path = self.get_strpath_storage_for_file('model.dfm')
		io.log_info(f'Dumping .dfm to {output_path}')

		class _DFMWrapper(torch.nn.Module):
			def __init__(self, parent):
				super().__init__()
				self.encoder = parent.encoder
				self.inter_src = parent.inter_src
				self.inter_dst = parent.inter_dst
				self.decoder = parent.decoder
				self.inter_dims = int(parent.inter_dims)
				self.inter_res = int(parent.inter_res)

			def forward(self, in_face, morph_value):
				# in_face: NHWC float32 [0..1]
				x = in_face.permute(0, 3, 1, 2).contiguous()
				if hasattr(self, 'model_dtype') and x.dtype != self.model_dtype:
					x = x.to(self.model_dtype)
				code = self.encoder(x)
				dst_inter_src = self.inter_src(code)
				dst_inter_dst = self.inter_dst(code)

				# dynamic morph_value -> channel mask
				mv = morph_value.reshape(-1)[0].float()
				k_f = torch.clamp(mv, 0.0, 1.0) * float(self.inter_dims)
				k = torch.clamp(k_f.floor().long(), 0, self.inter_dims)
				c_idx = torch.arange(self.inter_dims, device=dst_inter_src.device).view(1, self.inter_dims, 1, 1)
				m = (c_idx < k).to(dst_inter_src.dtype)
				src_dst_code = dst_inter_src * m + dst_inter_dst * (1.0 - m)

				out_celeb_face, out_celeb_face_mask = self.decoder(src_dst_code)
				_, out_face_mask = self.decoder(dst_inter_dst)

				# Cast outputs to FP32 for ONNX Runtime compatibility
				out_celeb_face = out_celeb_face.to(torch.float32)
				out_celeb_face_mask = out_celeb_face_mask.to(torch.float32)
				out_face_mask = out_face_mask.to(torch.float32)

				out_face_mask = out_face_mask.permute(0, 2, 3, 1)
				out_celeb_face = out_celeb_face.permute(0, 2, 3, 1)
				out_celeb_face_mask = out_celeb_face_mask.permute(0, 2, 3, 1)
				return out_face_mask, out_celeb_face, out_celeb_face_mask

		wrapper = _DFMWrapper(self)
		wrapper.eval()

		# 检测导出精度
		_export_prec = os.environ.get("DFM_EXPORT_PRECISION", "fp32")
		use_fp16_export = _export_prec == "fp16"
		if _export_prec == "fp16":
			# BF16 不被 ONNX Conv 支持，改用 FP16（同为 16 位，ONNX 原生支持）
			wrapper.model_dtype = torch.float16
			for sub in ["encoder", "inter_src", "inter_dst", "decoder"]:
				m = getattr(wrapper, sub, None)
				if m is not None:
					try:
						m.to("cpu", dtype=torch.float16)
					except Exception:
						pass
			io.log_info("FP16 导出模式（权重转 FP16）")
		else:
			# FP32 导出：强制所有权重转为 FP32
			for sub in ["encoder", "inter_src", "inter_dst", "decoder"]:
				m = getattr(wrapper, sub, None)
				if m is not None:
					try:
						m.to("cpu", dtype=torch.float32)
					except Exception:
						pass

		# BF16 on CPU is extremely slow; use GPU if available
		export_device = torch.device("cuda:0") if (_export_prec != "fp32" and torch.cuda.is_available()) else torch.device("cpu")
		wrapper = wrapper.to(export_device)
		dummy_face = torch.zeros(1, self.resolution, self.resolution, 3, dtype=torch.float32, device=export_device)
		dummy_morph = torch.tensor([1.0], dtype=torch.float32, device=export_device)
		with torch.no_grad():
			_ = wrapper(dummy_face, dummy_morph)

		# DeepFaceLive 侧通常用 TF 风格的 ':0' 张量名进行喂入。
		# 这里对齐导出的 ONNX I/O 命名（只改名字，不改计算/结构）。
		export_kwargs = dict(
			input_names=['in_face:0', 'morph_value:0'],
			output_names=['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0'],
			dynamic_axes={
				'in_face:0': {0: 'batch'},
				'out_face_mask:0': {0: 'batch'},
				'out_celeb_face:0': {0: 'batch'},
				'out_celeb_face_mask:0': {0: 'batch'},
			},
			opset_version=20 if use_fp16_export else 12,
		)

		# Torch 2.5+ 默认可能走 dynamo 导出（依赖 onnxscript）；
		# 在 Python 3.14 环境下 onnxscript 可能出现 typing 兼容问题。
		# 因此这里强制走 legacy exporter。
		try:
			torch.onnx.export(
				wrapper,
				(dummy_face, dummy_morph),
				output_path,
				dynamo=False,
				**export_kwargs,
			)
		except TypeError:
			# 兼容更老的 torch.onnx.export（不支持 dynamo 参数）
			torch.onnx.export(
				wrapper,
				(dummy_face, dummy_morph),
				output_path,
				**export_kwargs,
			)


Model = AMPModel
