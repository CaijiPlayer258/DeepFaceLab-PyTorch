import torch
import torch.nn.functional as F
from core.leras import nn


class GAXSeg(nn.ModelBase):
    """
    GA-XSeg v3: Mixed Conv + 5-stage + optional torch.compile.

    - Shallow (enc0-1, dec0-1): standard Conv2D 3×3 blocks (sharp edges)
    - Deep (enc2-4, dec2-4, bridge): GhostV2 blocks (efficient at high channels)
    - 5 encoder stages → 8×8 bottleneck (removed 4×4)
    - Deep supervision at dec4/dec3 (16×16, 32×32)
    """

    def on_build(self, in_ch=3, base_ch=32, out_ch=1, use_eca=True,
                 ghost_ratio=0.5):

        # ---- Standard ConvBlock (used in shallow layers) ----
        class StdBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, **kwargs):
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn  = nn.FRNorm2D(in_ch=out_ch)
                self.tlu  = nn.TLU(in_ch=out_ch)
            def forward(self, x):
                return self.tlu(self.frn(self.conv(x)))

        # ---- GhostV2 Block (used in deep layers) ----
        class GhostBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, g_ratio=0.5):
                self.ghost = nn.GhostModule(in_ch, out_ch, g_ratio)
                self.dfc   = nn.DFC(out_ch)
            def forward(self, x):
                return self.dfc(self.ghost(x))

        # ---- Spatial attention gate ----
        class SpatialGate(nn.LayerBase):
            def __init__(self, in_ch, name=None, **kwargs):
                self.in_ch = int(in_ch)
                super().__init__(name=name, **kwargs)
            def build_weights(self):
                self.conv = nn.Conv2D(2, 1, kernel_size=1, padding='SAME')
            def forward(self, x):
                ch_dim = 3 if nn.data_format == "NHWC" else 1
                avg = torch.mean(x, dim=ch_dim, keepdim=True)
                maxv, _ = torch.max(x, dim=ch_dim, keepdim=True)
                pooled = torch.cat([avg, maxv], dim=ch_dim)
                return x * torch.sigmoid(self.conv(pooled))

        # ---- Encoder stage ----
        class EncStage(nn.ModelBase):
            def on_build(self, in_ch, out_ch, n_blocks=2, block_type='ghost'):
                self.blocks = []
                ci = in_ch
                mk_block = StdBlock if block_type == 'standard' else GhostBlock
                for i in range(n_blocks):
                    co = out_ch if i == n_blocks - 1 else ci
                    b = mk_block(ci, co, g_ratio=ghost_ratio)
                    setattr(self, f'b{i}', b)
                    self.blocks.append(b)
                    ci = co
                self.pool = nn.DSCDownsample(out_ch, out_ch)
            def forward(self, x):
                s = x
                for b in self.blocks:
                    s = b(s)
                return self.pool(s), s

        # ---- Decoder stage ----
        class DecStage(nn.ModelBase):
            def on_build(self, in_ch, out_ch, skip_ch, n_blocks=2, block_type='ghost'):
                self.up_conv  = nn.Conv2DTranspose(in_ch, in_ch, kernel_size=3, padding='SAME')
                self.up_frn   = nn.FRNorm2D(in_ch=in_ch)
                self.up_tlu   = nn.TLU(in_ch=in_ch)
                self.skip_gate = SpatialGate(skip_ch)
                concat_ch = in_ch + skip_ch
                self.blocks = []
                mk_block = StdBlock if block_type == 'standard' else GhostBlock
                ci = concat_ch
                b0 = mk_block(ci, in_ch, g_ratio=ghost_ratio)
                setattr(self, 'b0', b0)
                ci = in_ch
                self.blocks.append(b0)
                for i in range(1, n_blocks):
                    co = out_ch if i == n_blocks - 1 else ci
                    b = mk_block(ci, co, g_ratio=ghost_ratio)
                    setattr(self, f'b{i}', b)
                    self.blocks.append(b)
                    ci = co
            def forward(self, x, skip, pretrain=False):
                if pretrain:
                    skip = torch.zeros_like(skip)
                x = self.up_tlu(self.up_frn(self.up_conv(x)))
                skip = self.skip_gate(skip)
                ch_axis = nn.conv2d_ch_axis
                x = torch.cat([x, skip], dim=ch_axis)
                for b in self.blocks:
                    x = b(x)
                return x

        self.base_ch = base_ch

        # ---- Encoder (5 stages) ----
        #    shallow (standard conv): enc0, enc1
        #    deep (ghost):            enc2, enc3, enc4
        self.enc0 = EncStage(in_ch,         base_ch,   n_blocks=2, block_type='standard')
        self.enc1 = EncStage(base_ch,       base_ch*2, n_blocks=2, block_type='standard')
        self.enc2 = EncStage(base_ch*2,     base_ch*4, n_blocks=2, block_type='ghost')
        self.enc3 = EncStage(base_ch*4,     base_ch*8, n_blocks=3, block_type='ghost')
        self.enc4 = EncStage(base_ch*8,     base_ch*8, n_blocks=3, block_type='ghost')

        # ---- Bottleneck at 8×8 (256ch) ----
        self.bridge = GhostBlock(base_ch*8, base_ch*8, g_ratio=ghost_ratio)
        self.bridge_attn = nn.ECA(base_ch*8) if use_eca else None

        # ---- Decoder (5 stages, symmetric) ----
        self.dec4 = DecStage(base_ch*8,  base_ch*8,  base_ch*8, n_blocks=3, block_type='ghost')
        self.dec3 = DecStage(base_ch*8,  base_ch*4,  base_ch*8, n_blocks=3, block_type='ghost')
        self.dec2 = DecStage(base_ch*4,  base_ch*2,  base_ch*4, n_blocks=2, block_type='ghost')
        self.dec1 = DecStage(base_ch*2,  base_ch,    base_ch*2, n_blocks=2, block_type='standard')
        self.dec0 = DecStage(base_ch,    base_ch,    base_ch,   n_blocks=2, block_type='standard')

        # ---- Output ----
        self.out_conv  = nn.Conv2D(base_ch, out_ch, kernel_size=3, padding='SAME')
        self.aux_head3 = nn.Conv2D(base_ch*4, out_ch, kernel_size=1, padding='SAME')  # dec3 → 32²
        self.aux_head2 = nn.Conv2D(base_ch*2, out_ch, kernel_size=1, padding='SAME')  # dec2 → 64²

    def forward(self, inp, pretrain=False):
        x = inp

        x, x0 = self.enc0(x)
        x, x1 = self.enc1(x)
        x, x2 = self.enc2(x)
        x, x3 = self.enc3(x)
        x, x4 = self.enc4(x)

        x = self.bridge(x)
        if self.bridge_attn is not None:
            x = self.bridge_attn(x)

        x = self.dec4(x, x4, pretrain)
        x = self.dec3(x, x3, pretrain)   # 32² 128ch
        f3  = x
        x = self.dec2(x, x2, pretrain)   # 64² 64ch
        f2  = x
        x = self.dec1(x, x1, pretrain)
        x = self.dec0(x, x0, pretrain)

        logits = self.out_conv(x)
        pred = torch.sigmoid(logits)

        if pretrain:
            return logits, pred

        aux3 = self.aux_head3(f3)
        aux2 = self.aux_head2(f2)
        return logits, pred, aux3, aux2


nn.GAXSeg = GAXSeg
