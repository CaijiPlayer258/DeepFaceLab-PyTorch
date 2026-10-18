# -*- coding: utf-8 -*-
"""PIA —— Per-Identity Adapter（逐身份适配器）核心模块。

设计要点（与 DF / LIAE 的差别）：
    1. 主干（Encoder + Decoder）**共享且冻结**，由预训练得到，不再改动；
    2. 身份 = 一个极小的 **FiLM 适配块**（逐通道仿射调制），每个身份一份；
    3. 适配块只承载"观感"（颜色 / 质感 / 细节倾向），**不承载几何** ——
       几何由 encoder 从输入图提供，所以换谁都成立；
    4. 训练只用现成对齐人脸集，靠"各自重建 + 循环一致"自监督，**不需要配对真值**。

与 LIAE 的关系：结构同形（共享 encoder + 两个小模块 + 单 decoder），
差别是那两个模块从"完整卷积块"缩成了"FiLM 级小块"，且主干冻结。
"""

import torch
import torch.nn as nn


class IdentityAdapter(nn.Module):
    """逐身份适配块：1×1 通道混合 + FiLM(逐通道仿射) + 残差。

    参数量（ch=256, mix=128 时）：
        conv1: 256*128 + 128      = 32,896
        fc   : 128*256 + 256      = 33,024
        conv2: 128*256 + 256      = 33,024
                                  ≈ 99,000   ← 对比 DF 的 decoder_src 约 2500 万
    """

    def __init__(self, ch, mix_ch=None):
        super().__init__()
        mix_ch = mix_ch or max(32, ch // 2)
        self.conv1 = nn.Conv2d(ch, mix_ch, 1)
        self.fc = nn.Linear(mix_ch, mix_ch * 2)      # 产生 gamma / beta
        self.conv2 = nn.Conv2d(mix_ch, ch, 1)
        # 初始化成"近似恒等"：gamma≈0, beta≈0 → 训练初期不破坏主干输出
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, f):
        x = self.conv1(f)
        gb = self.fc(x.mean(dim=(2, 3)))             # 全局池化 → (N, 2*mix)
        g, b = gb.chunk(2, dim=1)
        x = x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]   # FiLM
        return self.conv2(x) + f                     # 残差


class PIANet(nn.Module):
    """PIA 网络骨架：共享主干 + 两个逐身份适配器。

    主干以**依赖注入**方式传入（外部用现有 archi 构建好），
    这样本模块不依赖具体的 archi 构建 API，也方便单独测试。
    """

    def __init__(self, encoder, decoder, adapter_src, adapter_dst, freeze_backbone=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adapter_src = adapter_src
        self.adapter_dst = adapter_dst
        if freeze_backbone:
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                p.requires_grad_(False)

    # ---- 基础 ----
    def encode(self, img):
        return self.encoder(img)

    def decode_with(self, code, adapter):
        return self.decoder(adapter(code))

    # ---- 三个用途 ----
    def self_src(self, src_img):
        """src 自身的重建（有真值，用于学 src 的外观）"""
        return self.decode_with(self.encode(src_img), self.adapter_src)

    def self_dst(self, dst_img):
        """dst 自身的重建（有真值，用于学 dst 的外观、并给 decoder 泛化）"""
        return self.decode_with(self.encode(dst_img), self.adapter_dst)

    def swap(self, dst_img):
        """换脸：dst 的几何 + src 的外观 —— 推理时只用这一条路径"""
        return self.decode_with(self.encode(dst_img), self.adapter_src)

    def cycle_back(self, img):
        """把一张图用 dst 适配器还原回 dst 外观（循环一致用）"""
        return self.decode_with(self.encode(img), self.adapter_dst)

    # ---- 损失 ----
    def losses(self, src_img, dst_img, w_recon=1.0, w_cycle=1.0):
        """重建 + 循环一致，全部自监督（不需要任何配对真值）。

        循环一致是保证"适配块只改外观、不改几何"的唯一约束：
        若适配块擅自改了五官位置，反向就还原不回来，该项会炸。
        """
        rec_s = torch.nn.functional.l1_loss(self.self_src(src_img), src_img)
        rec_d = torch.nn.functional.l1_loss(self.self_dst(dst_img), dst_img)

        # dst -> src 外观 -> 再用 dst 适配器还原，应回到 dst
        f = self.swap(dst_img)
        cyc_d = torch.nn.functional.l1_loss(self.cycle_back(f.detach()), dst_img)
        # src -> dst 外观 -> 再用 src 适配器还原，应回到 src
        g = self.decode_with(self.encode(src_img), self.adapter_dst)
        cyc_s = torch.nn.functional.l1_loss(
            self.decode_with(self.encode(g.detach()), self.adapter_src), src_img)

        total = w_recon * (rec_s + rec_d) + w_cycle * (cyc_d + cyc_s)
        return total, dict(rec_s=float(rec_s), rec_d=float(rec_d),
                           cyc_d=float(cyc_d), cyc_s=float(cyc_s))

    def adapter_params(self):
        return sum(p.numel() for p in self.adapter_src.parameters()) + \
               sum(p.numel() for p in self.adapter_dst.parameters())

    def backbone_params(self):
        return sum(p.numel() for p in self.encoder.parameters()) + \
               sum(p.numel() for p in self.decoder.parameters())
