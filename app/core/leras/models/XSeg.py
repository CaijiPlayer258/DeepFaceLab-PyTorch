import torch
import torch.nn.functional as F
from core.leras import nn

class XSeg(nn.ModelBase):
    """
    XSeg分割模型 - PyTorch版本
    """
    def on_build(self, in_ch, base_ch, out_ch):
        
        class ConvBlock(nn.ModelBase):
            """卷积块"""
            def on_build(self, in_ch, out_ch):              
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(in_ch=out_ch)
                self.tlu = nn.TLU(in_ch=out_ch)

            def forward(self, x):                
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        class UpConvBlock(nn.ModelBase):
            """上采样卷积块（VALID + crop 匹配 TF 原始行为）"""
            def on_build(self, in_ch, out_ch):
                self.conv = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=3, padding='VALID')
                self.frn = nn.FRNorm2D(in_ch=out_ch)
                self.tlu = nn.TLU(in_ch=out_ch)

            def forward(self, x):
                x = self.conv(x)
                # VALID padding 使输出比预期大 1 像素，裁剪掉
                x = x[:, :, :-1, :-1]
                x = self.frn(x)
                x = self.tlu(x)
                return x
                
        self.base_ch = base_ch

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = nn.BlurPool (filt_size=4)

        self.conv11 = ConvBlock(base_ch, base_ch*2)
        self.conv12 = ConvBlock(base_ch*2, base_ch*2)
        self.bp1 = nn.BlurPool (filt_size=3)

        self.conv21 = ConvBlock(base_ch*2, base_ch*4)
        self.conv22 = ConvBlock(base_ch*4, base_ch*4)
        self.bp2 = nn.BlurPool (filt_size=2)

        self.conv31 = ConvBlock(base_ch*4, base_ch*8)
        self.conv32 = ConvBlock(base_ch*8, base_ch*8)
        self.conv33 = ConvBlock(base_ch*8, base_ch*8)
        self.bp3 = nn.BlurPool (filt_size=2)

        self.conv41 = ConvBlock(base_ch*8, base_ch*8)
        self.conv42 = ConvBlock(base_ch*8, base_ch*8)
        self.conv43 = ConvBlock(base_ch*8, base_ch*8)
        self.bp4 = nn.BlurPool (filt_size=2)
        
        self.conv51 = ConvBlock(base_ch*8, base_ch*8)
        self.conv52 = ConvBlock(base_ch*8, base_ch*8)
        self.conv53 = ConvBlock(base_ch*8, base_ch*8)
        self.bp5 = nn.BlurPool (filt_size=2)
        
        self.dense1 = nn.Dense ( 4*4* base_ch*8, 512)
        self.dense2 = nn.Dense ( 512, 4*4* base_ch*8)
                
        self.up5 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv53 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv52 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv51 = ConvBlock(base_ch*8, base_ch*8)
        
        self.up4 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv43 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv42 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv41 = ConvBlock(base_ch*8, base_ch*8)

        self.up3 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv33 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv32 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv31 = ConvBlock(base_ch*8, base_ch*8)

        self.up2 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv22 = ConvBlock(base_ch*8, base_ch*4)
        self.uconv21 = ConvBlock(base_ch*4, base_ch*4)

        self.up1 = UpConvBlock (base_ch*4, base_ch*2)
        self.uconv12 = ConvBlock(base_ch*4, base_ch*2)
        self.uconv11 = ConvBlock(base_ch*2, base_ch*2)

        self.up0 = UpConvBlock (base_ch*2, base_ch)
        self.uconv02 = ConvBlock(base_ch*2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2D (base_ch, out_ch, kernel_size=3, padding='SAME')
    
        
    def forward(self, inp, pretrain=False):
        """前向传播"""
        x = inp

        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)
        
        x = nn.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.reshape_4D(x, 4, 4, self.base_ch*8)
                          
        x = self.up5(x)
        if pretrain:
            x5 = torch.zeros_like(x5)
        x = self.uconv53(torch.cat([x, x5], dim=1))  # PyTorch使用dim=1表示通道维度
        x = self.uconv52(x)
        x = self.uconv51(x)
        
        x = self.up4(x)
        if pretrain:
            x4 = torch.zeros_like(x4)
        x = self.uconv43(torch.cat([x, x4], dim=1))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        if pretrain:
            x3 = torch.zeros_like(x3)
        x = self.uconv33(torch.cat([x, x3], dim=1))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        if pretrain:
            x2 = torch.zeros_like(x2)
        x = self.uconv22(torch.cat([x, x2], dim=1))
        x = self.uconv21(x)

        x = self.up1(x)
        if pretrain:
            x1 = torch.zeros_like(x1)
        x = self.uconv12(torch.cat([x, x1], dim=1))
        x = self.uconv11(x)

        x = self.up0(x)
        if pretrain:
            x0 = torch.zeros_like(x0)
        x = self.uconv02(torch.cat([x, x0], dim=1))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)

nn.XSeg = XSeg