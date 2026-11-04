"""
FaceRestoreLite 训练脚本。
条件式 GAN（LSGAN）+ L1 + VGG 感知 loss。
"""
import multiprocessing, os, pickle, re
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from core.interact import interact as io
from core.leras import nn
from core.facerestorelite_torch import Generator, Discriminator, degrade_np, augment_hsv_np
from models import ModelBase
from samplelib import SampleLoaderV4, SampleGeneratorV2, SampleProcessor


# ── VGG 感知 loss ─────────────────────────────────────

class VGGFeatureExtractor(torch.nn.Module):
    """VGG16 特征提取器（固定权重，不训练）。"""
    def __init__(self, layer_ids=(3, 8, 13, 19)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        self.layers = torch.nn.ModuleList([features[:i+1] for i in layer_ids])
        # 归一化参数（ImageNet）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x 在 [-1, 1] → 转到 [0, 1] → ImageNet 归一化
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        return [layer(x) for layer in self.layers]  # 每个从原始输入重新跑


class FaceRestoreLiteModel(ModelBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('force_model_class_name', 'FaceRestoreLite')
        super().__init__(*args, **kwargs)

    def on_initialize_options(self):
        self.options['resolution'] = self.load_or_def_option('resolution', 256)
        self.options['use_bf16'] = self.load_or_def_option('use_bf16', False)
        self.options['pretrain'] = False  # pretrain 已停用：无论读到什么一律强制 False
        self.options['lr_cos'] = self.load_or_def_option('lr_cos', 10000)

        if self.is_first_run():
            self.options['resolution'] = io.input_int('分辨率', 256, add_info='128-512')
            self.options['use_bf16'] = io.input_bool('BF16 混合精度', False)

        # 每次启动都可改
        self.ask_batch_size(8)

        # 退化参数 —— 每次启动都可改，按 Enter 沿用当前值
        self.options['jpeg_range'] = io.input_str(
            'JPEG 质量范围 (min-max)', self.options.get('jpeg_range', '30-80'))
        self.options['blur_k_range'] = io.input_str(
            '模糊核范围 (min-max, 奇数)', self.options.get('blur_k_range', '3-11'))
        self.options['blur_sigma_range'] = io.input_str(
            '模糊 sigma 范围 (min-max)', self.options.get('blur_sigma_range', '0.5-3.0'))
        self.options['noise_range'] = io.input_str(
            '随机噪点强度范围 (min-max)', self.options.get('noise_range', '0.0-0.05'))
        # HSV 光照增强（面部亮、背景暗），每次可改
        self.options['hue_range'] = io.input_str(
            '色调偏移范围 (min-max)', self.options.get('hue_range', '-20-20'))
        self.options['sat_range'] = io.input_str(
            '饱和度缩放范围 (min-max)', self.options.get('sat_range', '0.6-1.4'))
        self.options['val_face_range'] = io.input_str(
            '面部亮度变化 (min-max)', self.options.get('val_face_range', '0.05-0.25'))
        self.options['val_bg_range'] = io.input_str(
            '背景亮度变化 (min-max)', self.options.get('val_bg_range', '-0.15-0.0'))

    @staticmethod
    def _parse_range(s, dtype=int, allowed_fn=None):
        """解析 'min-max' 字符串为 (min, max) 元组。"""
        m = re.match(r'^\s*(-?[.\d]+)\s*[-–]\s*(-?[.\d]+)\s*$', s)
        if not m:
            raise ValueError(f'格式无效，需要 "min-max"，得到: {s}')
        lo, hi = dtype(m.group(1)), dtype(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if allowed_fn:
            lo, hi = allowed_fn(lo), allowed_fn(hi)
        return lo, hi

    def on_initialize(self):
        nn.initialize(nn.getCurrentDeviceConfig(), data_format='NCHW')
        self.device = nn.device
        self.resolution = int(self.options['resolution'])

        # ── 退化参数 ──
        raw_jpeg = str(self.options.get('jpeg_range', '30-80'))
        raw_k = str(self.options.get('blur_k_range', '3-11'))
        raw_sigma = str(self.options.get('blur_sigma_range', '0.5-3.0'))
        raw_noise = str(self.options.get('noise_range', '0.0-0.05'))
        raw_hue = str(self.options.get('hue_range', '-20-20'))
        raw_sat = str(self.options.get('sat_range', '0.6-1.4'))
        raw_val_f = str(self.options.get('val_face_range', '0.05-0.25'))
        raw_val_b = str(self.options.get('val_bg_range', '-0.15-0.0'))
        try:
            self.jpeg_min, self.jpeg_max = self._parse_range(raw_jpeg, int)
            self.blur_k_min, self.blur_k_max = self._parse_range(raw_k, int)
            self.blur_sigma_min, self.blur_sigma_max = self._parse_range(raw_sigma, float)
            self.noise_min, self.noise_max = self._parse_range(raw_noise, float)
            self.hue_min, self.hue_max = self._parse_range(raw_hue, int)
            self.sat_min, self.sat_max = self._parse_range(raw_sat, float)
            self.val_f_min, self.val_f_max = self._parse_range(raw_val_f, float)
            self.val_b_min, self.val_b_max = self._parse_range(raw_val_b, float)
        except ValueError:
            io.log_info(f'退化参数解析失败，使用默认值')
            self.jpeg_min, self.jpeg_max = 30, 80
            self.blur_k_min, self.blur_k_max = 3, 11
            self.blur_sigma_min, self.blur_sigma_max = 0.5, 3.0
            self.noise_min, self.noise_max = 0.0, 0.05
            self.hue_min, self.hue_max = -20, 20
            self.sat_min, self.sat_max = 0.6, 1.4
            self.val_f_min, self.val_f_max = 0.05, 0.25
            self.val_b_min, self.val_b_max = -0.15, 0.0

        # Build models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.vgg = VGGFeatureExtractor().to(self.device)

        # Optimizers
        self.opt_g = Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = Adam(self.discriminator.parameters(), lr=5e-5, betas=(0.5, 0.999))

        # 余弦退火
        self.lr_cos = int(self.options.get('lr_cos', 10000))
        self.lr_scheduler = CosineAnnealingLR(self.opt_g, T_max=self.lr_cos, eta_min=1e-6)

        self.model_filename_list = [
            [self, 'FaceRestoreLite.pth'],
        ]

        # Load
        pth = Path(self.get_strpath_storage_for_file('FaceRestoreLite.pth'))
        if pth.exists() and not self.is_first_run():
            try:
                ckpt = torch.load(str(pth), map_location=self.device)
                self.generator.load_state_dict(ckpt['generator'], strict=False)
                if 'discriminator' in ckpt:
                    self.discriminator.load_state_dict(ckpt['discriminator'])
                self.opt_g.load_state_dict(ckpt['opt_g'])
                for pg in self.opt_g.param_groups:
                    pg['lr'] = 2e-4
                if 'opt_d' in ckpt:
                    self.opt_d.load_state_dict(ckpt['opt_d'])
                    for pg in self.opt_d.param_groups:
                        pg['lr'] = 5e-5
                if 'scheduler' in ckpt:
                    self.lr_scheduler.load_state_dict(ckpt['scheduler'])
                io.log_info("Loaded weights (iter=%d)" % ckpt.get("iter", 0))
            except Exception as e:
                io.log_info(f'Load failed: {e}')

        # Data generator (V4 fast loader)
        if self.is_training:
            _loader = SampleLoaderV4(
                aligned_path=self.training_data_src_path,
                batch_size=self.get_batch_size(),
                resolution=self.resolution,
            )
            _out = [{'sample_type': SampleProcessor.SampleType.FACE_IMAGE,
                      'warp': False, 'transform': True,
                      'channel_type': SampleProcessor.ChannelType.BGR,
                      'face_type': 'wf',
                      'data_format': nn.data_format, 'resolution': self.resolution}]
            _gen = SampleGeneratorV2(
                loader=_loader,
                sample_process_options=SampleProcessor.Options(random_flip=False),
                output_sample_types=_out,
                resolution=self.resolution,
            )
            self.set_training_data_generators([_gen])

    def get_model_filename_list(self):
        return self.model_filename_list

    def onSave(self):
        ckpt = {
            'generator': self.generator.state_dict(),
            'opt_g': self.opt_g.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'iter': self.get_iter(),
        }
        if self.discriminator:
            ckpt['discriminator'] = self.discriminator.state_dict()
            ckpt['opt_d'] = self.opt_d.state_dict()
        torch.save(ckpt, self.get_strpath_storage_for_file('FaceRestoreLite.pth'))

    def train_one_step(self, hq_batch, xseg_masks=None):
        B = hq_batch.shape[0]
        # 先 HSV 光照增强（面部亮、背景暗），再退化
        hq_hsv = np.array([augment_hsv_np(hq_batch[i].transpose(1,2,0),
                                  xseg_mask=xseg_masks[i] if xseg_masks is not None else None,
                                  hue_range=(self.hue_min, self.hue_max),
                                  sat_range=(self.sat_min, self.sat_max),
                                  val_face_range=(self.val_f_min, self.val_f_max),
                                  val_bg_range=(self.val_b_min, self.val_b_max))
                   for i in range(B)])
        lq_np = np.array([degrade_np(hq_hsv[i],
                              jpeg_min=self.jpeg_min, jpeg_max=self.jpeg_max,
                              blur_k_min=self.blur_k_min, blur_k_max=self.blur_k_max,
                              blur_sigma_min=self.blur_sigma_min, blur_sigma_max=self.blur_sigma_max,
                              xseg_mask=xseg_masks[i] if xseg_masks is not None else None)
                   for i in range(B)])
        lq = torch.from_numpy(lq_np.transpose(0, 3, 1, 2).copy()).float().to(self.device) * 2 - 1
        target = torch.from_numpy(hq_hsv.transpose(0, 3, 1, 2).copy()).float().to(self.device) * 2 - 1
        use_bf16 = bool(self.options.get('use_bf16', False))

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            pred = self.generator(lq)

            # ── 条件式 GAN（LSGAN）──
            logits_fake = self.discriminator(lq, pred)
            loss_gan = F.mse_loss(logits_fake, torch.ones_like(logits_fake))

            # ── L1 锚定 ──
            loss_l1 = F.l1_loss(pred, target)

            # ── VGG 感知 loss ──
            pred_feats = self.vgg(pred)
            target_feats = self.vgg(target)
            loss_vgg = sum(F.l1_loss(pf, tf) for pf, tf in zip(pred_feats, target_feats))

            # ── 总 loss ──
            loss_g = loss_gan + loss_l1 * 0.05 + loss_vgg * 0.1

        # ── ÿ?ÿ?±¾ loss ──
        with torch.no_grad():
            l1_per = F.l1_loss(pred, target, reduction='none').mean(dim=[1,2,3])
            g_per = F.mse_loss(logits_fake, torch.ones_like(logits_fake), reduction='none').mean(dim=[1,2,3])
            vgg_per = sum(F.l1_loss(pf, tf, reduction='none').mean(dim=[1,2,3]) for pf, tf in zip(pred_feats, target_feats))
            per_sample = g_per + l1_per * 0.05 + vgg_per * 0.1
            self._last_src_loss_per_sample = per_sample.cpu().numpy()
            self._last_dst_loss_per_sample = per_sample.cpu().numpy()

        self.opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10.0)
        self.opt_g.step()

        # Discriminator（条件式 + LSGAN + 标签平滑 0.9）
        loss_d_val = 0.0
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
            logits_real = self.discriminator(lq, target.detach())
            logits_fake = self.discriminator(lq, pred.detach())
            loss_d = F.mse_loss(logits_real, torch.full_like(logits_real, 0.9)) * 0.5
            loss_d = loss_d + F.mse_loss(logits_fake, torch.zeros_like(logits_fake)) * 0.5
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        loss_d_val = loss_d.item()

        return float(loss_g.item()), loss_d_val

    def onTrainOneIter(self):
        batch = self.generate_next_samples()
        hq_batch = batch[0][0]

        # 从生成器获取文件名，加载 XSeg 遮罩
        masks = None
        try:
            gen = self.get_training_data_generators()[0]
            fnames = gen.get_last_filenames()
            if fnames:
                from DFLIMG import DFLJPG
                aligned_dir = Path(str(self.training_data_src_path))
                masks = []
                for f in fnames:
                    # fname 可能已经是完整路径
                    fpath = Path(f) if Path(f).is_absolute() else aligned_dir / f
                    if not fpath.exists():
                        io.log_info(f'[XSeg] 文件不存在: {fpath}')
                        masks.append(None)
                        continue
                    dfl = DFLJPG.load(str(fpath))
                    m = dfl.get_xseg_mask() if dfl and dfl.has_xseg_mask() else None
                    if m is not None and m.ndim > 2:
                        m = m.squeeze()
                    masks.append(m)
                    if m is None:
                        io.log_info(f'[XSeg] {fpath.name}: 无遮罩')
        except Exception as e:
            io.log_info(f'[XSeg] 加载失败: {e}')


        g_loss, d_loss = self.train_one_step(hq_batch, xseg_masks=masks)
        self.lr_scheduler.step()
        return (('G_loss', g_loss), ('D_loss', d_loss))

    def onGetPreview(self, samples, for_history=False):
        hq_batch = samples[0][0]
        self.generator.eval()

        # ── 注册 hook 捕获注意力图 ──
        from core.facerestorelite_torch import SpatialAttention
        attn_maps = {}
        def make_hook(name):
            def hook(m, inp, out):
                with torch.no_grad():
                    x = inp[0]
                    avg = x.mean(dim=1, keepdim=True)
                    mx = x.max(dim=1, keepdim=True)[0]
                    a = m.conv(torch.cat([avg, mx], dim=1)).sigmoid()
                    attn_maps[name] = a[:, 0].float().detach().cpu().numpy()  # (B, H, W)
            return hook
        handles = []
        for i, attn_mod in enumerate(self.generator.enc_attns):
            handles.append(attn_mod.register_forward_hook(make_hook(f'enc_{i}')))
        handles.append(self.generator.middle_attn.sa.register_forward_hook(make_hook('mid')))

        with torch.no_grad():
            n_preview = min(4, hq_batch.shape[0])
            # load XSeg masks for preview
            pv_masks = None
            try:
                gen = self.get_training_data_generators()[0]
                fnames = gen.get_last_filenames()
                if fnames:
                    from DFLIMG import DFLJPG
                    adir = Path(str(self.training_data_src_path))
                    pv_masks = []
                    for f in fnames[:n_preview]:
                        fp = Path(f) if Path(f).is_absolute() else adir / f
                        if fp.exists():
                            dfl = DFLJPG.load(str(fp))
                            m = dfl.get_xseg_mask() if dfl and dfl.has_xseg_mask() else None
                            if m is not None and m.ndim > 2:
                                m = m.squeeze()
                            pv_masks.append(m)
                        else:
                            pv_masks.append(None)
            except: pass
            hq_hsv = np.array([augment_hsv_np(hq_batch[i].transpose(1,2,0),
                                               hue_range=(self.hue_min, self.hue_max),
                                               sat_range=(self.sat_min, self.sat_max),
                                               val_face_range=(self.val_f_min, self.val_f_max),
                                               val_bg_range=(self.val_b_min, self.val_b_max))
                                for i in range(n_preview)])
            lq_np = np.array([degrade_np(hq_hsv[i],
                                          jpeg_min=self.jpeg_min, jpeg_max=self.jpeg_max,
                                          blur_k_min=self.blur_k_min, blur_k_max=self.blur_k_max,
                                          blur_sigma_min=self.blur_sigma_min, blur_sigma_max=self.blur_sigma_max,
                                          noise_min=self.noise_min, noise_max=self.noise_max,
                                          xseg_mask=pv_masks[i] if pv_masks is not None else None)
                               for i in range(n_preview)])
            lq = torch.from_numpy(lq_np.transpose(0, 3, 1, 2).copy()).float().to(self.device) * 2 - 1
            pred = self.generator(lq)
            # per-sample L1 loss for WebUI (matches preview filenames)
            target_t = torch.from_numpy(hq_hsv.transpose(0, 3, 1, 2).copy()).float().to(self.device) * 2 - 1
            per_l1 = torch.nn.functional.l1_loss(pred, target_t, reduction='none').mean(dim=[1,2,3])
            self._last_src_loss_per_sample = per_l1.cpu().numpy()
            self._last_dst_loss_per_sample = per_l1.cpu().numpy()
        for h in handles:
            h.remove()
        self.generator.train()

        n = n_preview
        res = self.resolution
        gap = 4
        # 列: 原图 | 退化 | 预测 | enc0注意力 | enc1 | enc2 | enc3 | mid = 8
        ncols = 9  # orig | lq | pred | diff | 5 attn
        vis = np.zeros((n * res + (n - 1) * gap, res * ncols + gap * (ncols - 1), 3), dtype=np.float32)

        for i in range(n):
            y = i * (res + gap)
            hq_rgb = hq_hsv[i]  # HSV 增强后的原图（模型的目标）
            lq_rgb = np.clip(lq_np[i], 0, 1).astype(np.float32)
            pr_rgb = np.clip((nn.to_data_format(pred[i:i+1].cpu().numpy(), 'NHWC', 'NCHW')[0] + 1) / 2, 0, 1).astype(np.float32)

            vis[y:y+res, :res] = hq_rgb
            vis[y:y+res, res+gap:2*res+gap] = lq_rgb
            vis[y:y+res, 2*res+2*gap:3*res+2*gap] = pr_rgb

            # Diff 热图（GAN 编辑区域）
            diff = np.abs(hq_rgb - pr_rgb).max(axis=2)
            mx = diff.max()
            diff_norm = (diff / mx * 255).astype(np.uint8) if mx > 0 else np.zeros((res, res), dtype=np.uint8)
            diff_jet = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
            vis[y:y+res, 3*res+3*gap:4*res+3*gap] = diff_jet.astype(np.float32) / 255.0

            # 注意力图列（每个样本取对应的批次索引）
            for j, key in enumerate(['enc_0', 'enc_1', 'enc_2', 'enc_3', 'mid']):
                if key in attn_maps and i < attn_maps[key].shape[0]:
                    a = attn_maps[key][i]
                    if a.size > 1:
                        a_rgb = cv2.resize(a, (res, res), interpolation=cv2.INTER_NEAREST)
                    else:
                        a_rgb = np.zeros((res, res))
                    a_rgb = np.clip(a_rgb, 0, 1)
                    a_jet = cv2.applyColorMap((a_rgb * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    x = (4 + j) * (res + gap)
                    vis[y:y+res, x:x+res] = a_jet.astype(np.float32) / 255.0

        # 注意力层标注（在每个注意力列底部加文字）
        for j, key in enumerate(['enc_0', 'enc_1', 'enc_2', 'enc_3', 'mid']):
            if key in attn_maps:
                x = (4 + j) * (res + gap) + 8
                y_last = (n - 1) * (res + gap) + res - 20
                cv2.putText(vis, f'enc_{j}' if j < 4 else 'mid',
                            (x, y_last), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1)

        return [('FaceRestoreLite+Attn', vis)]


Model = FaceRestoreLiteModel
