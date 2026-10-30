"""
FaceRestoreLite: NAFNet 架构（匹配官方 GoPro-width64 预训练权重）。
Conv2d + PixelShuffle + SimpleGate + LayerNorm，无 BN。
"""
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """官方 NAFBlock: 5 conv, 2 LN, 2 Gate, SCA = Sequential(GAP, Conv1×1)"""
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1, bias=True)
        self.conv2 = nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2, bias=True)
        self.gate = SimpleGate()
        self.conv3 = nn.Conv2d(c, c, 1, bias=True)
        self.sca = nn.Sequential(  # SCA: GAP → Conv1×1, sigmoid 在 forward 里
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=True),
        )
        self.conv4 = nn.Conv2d(c, c * 2, 1, bias=True)
        self.conv5 = nn.Conv2d(c, c, 1, bias=True)
        self.norm2 = nn.LayerNorm(c)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1); x = self.norm1(x); x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)           # c→2c
        x = self.conv2(x)           # 2c depthwise
        x = self.gate(x)            # 2c→c
        x = self.conv3(x)           # c→c
        x = x * self.sca(x).sigmoid()  # SCA
        x = self.conv4(x)           # c→2c
        x = self.gate(x)            # 2c→c
        x = self.conv5(x)           # c→c
        x = x.permute(0, 2, 3, 1); x = self.norm2(x); x = x.permute(0, 3, 1, 2)
        return shortcut + x * self.gamma + self.beta


class SpatialAttention(nn.Module):
    """空间注意力：Conv7×7 → Sigmoid，让模型关注需要恢复的区域。"""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=True)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.conv(torch.cat([avg, mx], dim=1)).sigmoid()


class CBAM(nn.Module):
    """轻量 CBAM：通道注意力 + 空间注意力。"""
    def __init__(self, c):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=True),
        )
        self.sa = SpatialAttention(c)

    def forward(self, x):
        x = x * self.ca(x).sigmoid()
        x = self.sa(x)
        return x


class NAFNet(nn.Module):
    """
    NAFNet 架构（匹配官方 GoPro-width64）。

    架构: 4 encoder + 1 extra downsample + 1 middle + 4 decoder
    通道: 64→128→256→512 →1024(extra down) →512→256→128→64
    跳接: 加法（不是 concat）
    """
    def __init__(self, in_ch=3, width=32,
                 enc_blks=(2, 2, 4, 6),
                 middle_blk=2):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1, bias=True)

        # ── Encoder（每级后接空间注意力）──
        self.encoders = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for i, num_blk in enumerate(enc_blks):
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(num_blk)]))
            self.enc_attns.append(SpatialAttention(c))
            self.downs.append(nn.Conv2d(c, c * 2, 3, 2, 1, bias=True))
            c *= 2

        # ── Middle（接 CBAM 全局注意力）──
        self.middle_blks = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blk)])
        self.middle_attn = CBAM(c)

        # ── Decoder（级数 = 下采样次数 = len(enc_blks)）──
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(enc_blks)):
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(c // 2, c // 2, 3, 1, 1, groups=c // 2, bias=True),
            ))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(1)]))

        self.ending = nn.Conv2d(c, in_ch, 3, 1, 1, bias=True)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, x):
        inp = x
        x = self.intro(x)

        # Encoder（每级后空间注意力 → 存跳连 → 下采样）
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            x = self.enc_attns[i](x)
            skips.append(x)
            x = self.downs[i](x)

        # Middle（CBAM 全局注意力）
        x = self.middle_blks(x)
        x = self.middle_attn(x)

        # Decoder（加法跳连，跳连已经是注意力增强的）
        for i in range(len(self.decoders)):
            x = self.ups[i](x)
            x = x + skips[-(i + 1)]
            x = self.decoders[i](x)

        return inp + self.ending(x)  # 全局残差连接


# ── Discriminator ──────────────────────────────────────

class Discriminator(nn.Module):
    """条件式 PatchGAN：输入 concat(退化图, 目标/生成图) → 6通道。"""
    def __init__(self, in_ch=6, base_ch=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1, bias=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, cond, img):
        """cond=退化图, img=目标或生成图（都在 [-1,1]）"""
        x = torch.cat([cond, img], dim=1)
        return self.net(x)


Generator = NAFNet


# ── 在线退化 ──────────────────────────────────────────

def _face_mask(h, w):
    """生成椭圆人脸遮罩（对齐图的人脸在中心）。"""
    cy, cx = h // 2, w // 2
    axes = (int(w * 0.38), int(h * 0.45))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1, -1)
    return cv2.GaussianBlur(mask, (max(h//8*2+1, 3),)*2, h//16)



def augment_hsv_np(img, xseg_mask=None,
                   hue_range=(-20, 20),
                   sat_range=(0.6, 1.4),
                   val_face_range=(0.05, 0.25),
                   val_bg_range=(-0.15, 0.0)):
    """HSV 光照增强：面部亮、背景暗。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(*hue_range)) % 180
    hsv[..., 1] = hsv[..., 1] * random.uniform(*sat_range)
    m = cv2.resize(xseg_mask.astype(np.float32), (w, h)) if xseg_mask is not None else _face_mask(h, w)
    dv = m * random.uniform(*val_face_range) + (1 - m) * random.uniform(*val_bg_range)
    hsv[..., 2] = hsv[..., 2] * (1 + dv)
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def degrade_np(img, jpeg_min=30, jpeg_max=80,
               blur_k_min=3, blur_k_max=11,
               blur_sigma_min=0.5, blur_sigma_max=3.0,
               xseg_mask=None,
               noise_min=0.0, noise_max=0.05):
    """xseg_mask: HxW float32 [0,1]，遮罩区域 blur+noise+JPEG。
       noise_min/max: 随机高斯噪点强度。"""
    h, w = img.shape[:2]

    # 构建遮罩
    if xseg_mask is not None:
        mask = cv2.resize(xseg_mask.astype(np.float32), (w, h))
    else:
        mask = _face_mask(h, w)
    choice = random.random()
    if choice < 0.33:
        cutoff = random.uniform(0.3, 0.6)
        mask[:int(h * (1 - cutoff)), :] = 0
    elif choice < 0.66:
        cutoff = random.uniform(0.6, 0.9)
        mask[:int(h * (1 - cutoff)), :] = 0
    mask_s = cv2.GaussianBlur(mask, (max(h//32*2+1, 3),)*2, h//64)

    # blur
    img_out = img.copy()
    ks = [k for k in range(blur_k_min, blur_k_max + 1) if k % 2 == 1]
    if ks:
        k = random.choice(ks)
        s = random.uniform(blur_sigma_min, blur_sigma_max)
        full_blur = cv2.GaussianBlur(img, (k, k), s)
        img_out = img * (1 - mask_s[..., None]) + full_blur * mask_s[..., None]

    # noise（遮罩区域）
    ns = random.uniform(noise_min, noise_max)
    if ns > 0:
        noise = np.random.randn(h, w, 3).astype(np.float32) * ns
        noisy = (img_out + noise).clip(0, 1)
        img_out = img_out * (1 - mask_s[..., None]) + noisy * mask_s[..., None]

    # JPEG（遮罩区域）
    q = random.randint(jpeg_min, jpeg_max)
    _, enc = cv2.imencode('.jpg', (img_out * 255).astype(np.uint8),
                          [int(cv2.IMWRITE_JPEG_QUALITY), q])
    jpeg_full = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img_out = img_out * (1 - mask_s[..., None]) + jpeg_full * mask_s[..., None]

    return np.clip(img_out, 0, 1).astype(np.float32)

if __name__ == '__main__':
    for cfg, label in [({}, 'Lite(32)')]:
        g = NAFNet(**cfg)
        x = torch.randn(2, 3, 256, 256)
        out = g(x)
        print(f'{label}: {sum(p.numel() for p in g.parameters())/1e6:.2f}M | {out.shape}')
