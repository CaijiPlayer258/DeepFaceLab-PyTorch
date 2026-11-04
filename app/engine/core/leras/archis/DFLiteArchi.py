"""DFLite —— 类 DF 的轻量同构架构（5x5 -> 3x3 / DW5x5+PW1x1 重设计）。

对外接口与 DeepFakeArchi 对齐：
    a = DFLiteArchi(resolution, opts='udt', ae_dims=512, ...)
    enc  = a.Encoder(in_ch=3, e_ch=64, name='encoder')
    inter= a.Inter(in_ch=..., ae_ch=..., ae_out_ch=..., name='inter')
    dec  = a.Decoder(in_ch=..., d_ch=64, d_mask_ch=32, name='decoder')
    x, m = dec(codes, mask_hint)

优化项（每项独立可开关，便于消融）：
    kernel     5x5 稠密 -> 3x3 稠密 / DW5x5+PW1x1（感受野不变，算力降一个量级）
    repvgg     残差块 = conv3x3 + conv1x1 + identity 三分支，推理期可折叠成单个 3x3
    bottleneck 宽度可配；可选 stride 投影，省掉 Dense2 的 g*g 空间放大
    ladder     解码器宽度阶梯改为逐级 1/2，不再在第一级就顶到 d_ch*8
    padding    全部显式 pad 数值，规避 leras SAME+dilation 的 off-by-2
"""
import math
import torch
import torch.nn.functional as F

from core.leras.nn import nn
from core.leras.archis.ArchiBase import ArchiBase


def _dev():
    return getattr(nn, 'device', None)


def _use_dw_blocks(lite_mod):
    """哪些 kernel 变体用 DWRepBlock（可折的 DW5x5 + identity）做残差块。"""
    return lite_mod in ('dw5', 'repdw5', 'ms5')


def _block_kernels(lite_mod):
    """残差块里 depthwise 分支的核尺寸列表（多尺度时可多个，同 groups 故可折）。"""
    if lite_mod == 'ms5':
        return (5, 3)
    return (5,)


# LeakyReLU(0.1) 的方差增益：sqrt(2/(1+a^2))。徒手写 init 时必须带上，
# 否则默认 bound=1/sqrt(fan_in) 会让激活逐层衰减（实测 0.29 -> 0.003）。
_LEAK = 0.1
GAIN = math.sqrt(2.0 / (1.0 + _LEAK * _LEAK))


# ============================================================================
# 基础算子
# ============================================================================

class PWConv(nn.LayerBase):
    """1x1 pointwise。"""
    def __init__(self, in_ch, out_ch, use_bias=True, dtype=None, name=None, **kwargs):
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_bias, self.dtype = use_bias, dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(
            torch.empty(self.out_ch, self.in_ch, 1, 1, dtype=self.dtype, device=_dev()))
        # 按 LeakyReLU(0.1) 的增益修：kaiming_uniform_(a) 的 bound 少乘了 sqrt(2/(1+a^2))，
        # 而 xavier 在分组卷积下 fan_out 语义与 PyTorch 不一致 —— 两者都不直接用。
        torch.nn.init.uniform_(self.weight, -GAIN / math.sqrt(self.in_ch), GAIN / math.sqrt(self.in_ch))
        self.bias = (torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype, device=_dev()))
                     if self.use_bias else None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)


class DWConv(nn.LayerBase):
    """depthwise kxk（groups = channels）。k=5 时感受野等价 5x5 稠密卷积，算力约 1/ch。"""
    def __init__(self, ch, kernel_size=5, strides=1, use_bias=True, dtype=None, name=None, **kwargs):
        self.ch, self.kernel_size, self.strides = ch, int(kernel_size), int(strides)
        self.use_bias, self.dtype = use_bias, dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(torch.empty(
            self.ch, 1, self.kernel_size, self.kernel_size, dtype=self.dtype, device=_dev()))
        # depthwise 的 fan_in 就是 kernel_size^2（每通道只连自己）
        b = GAIN / math.sqrt(self.kernel_size * self.kernel_size)
        torch.nn.init.uniform_(self.weight, -b, b)
        self.bias = (torch.nn.Parameter(torch.zeros(self.ch, dtype=self.dtype, device=_dev()))
                     if self.use_bias else None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.strides,
                        padding=(self.kernel_size - 1) // 2, groups=self.ch)


class ConvBn(nn.LayerBase):
    """conv kxk（可分组）+ 可选 BatchNorm，是 RepVGG 折叠的最小单元。"""
    def __init__(self, in_ch, out_ch, kernel_size=3, strides=1, pad=None, groups=1,
                 use_bn=True, dtype=None, name=None, **kwargs):
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size, self.strides, self.groups = int(kernel_size), int(strides), int(groups)
        self.pad = (self.kernel_size - 1) // 2 if pad is None else int(pad)
        self.use_bn, self.dtype = use_bn, dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        self.weight = torch.nn.Parameter(torch.empty(
            self.out_ch, self.in_ch // self.groups, self.kernel_size, self.kernel_size,
            dtype=self.dtype, device=_dev()))
        fan_in = (self.in_ch // self.groups) * self.kernel_size * self.kernel_size
        b = GAIN / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.weight, -b, b)
        if self.use_bn:
            self.bias = None
            self.bn_w = torch.nn.Parameter(torch.ones(self.out_ch, dtype=self.dtype, device=_dev()))
            self.bn_b = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype, device=_dev()))
            self.bn_m = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=torch.float32, device=_dev()), requires_grad=False)
            self.bn_v = torch.nn.Parameter(torch.ones(self.out_ch, dtype=torch.float32, device=_dev()), requires_grad=False)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_ch, dtype=self.dtype, device=_dev()))
            self.bn_w = self.bn_b = self.bn_m = self.bn_v = None

    def bn_affine(self):
        """返回 (scale, shift)，满足 y = x*scale + shift。"""
        s = (self.bn_w.float() / torch.sqrt(self.bn_v + 1e-5)).to(self.dtype)
        b = (self.bn_b.float() - self.bn_m.float() * s.float()).to(self.dtype)
        return s, b

    def forward(self, x):
        x = F.conv2d(x, self.weight, None, stride=self.strides, padding=self.pad, groups=self.groups)
        if self.use_bn:
            if self.training:
                # 必须用 .data 原地更新：bn_m / bn_v 虽然 requires_grad=False，
                # 但仍是 nn.Parameter，带 version counter。下面 bn_affine() 会读它们
                # 并参与 autograd 图，此时若原地改（即使套了 no_grad），
                # backward 就会报 "variables needed for gradient computation has been
                # modified by an inplace operation"。
                # 实测：df-udt-l 和 liae-udt-l 都在第一次 backward 挂掉，训练完全跑不起来。
                with torch.no_grad():
                    self.bn_m.data.mul_(1 - 0.1).add_(0.1 * x.detach().mean(dim=(0, 2, 3)).float())
                    self.bn_v.data.mul_(1 - 0.1).add_(0.1 * x.detach().var(dim=(0, 2, 3), unbiased=False).float())
            s, b = self.bn_affine()
            x = x * s.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
        else:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


class IdBn(nn.LayerBase):
    """identity 分支上的 BatchNorm（RepVGG 的 id 分支）。"""
    def __init__(self, ch, use_bn=True, dtype=None, name=None, **kwargs):
        self.ch, self.use_bn, self.dtype = ch, use_bn, dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        if self.use_bn:
            self.weight = torch.nn.Parameter(torch.ones(self.ch, dtype=self.dtype, device=_dev()))
            self.bias = torch.nn.Parameter(torch.zeros(self.ch, dtype=self.dtype, device=_dev()))
            self.running_mean = torch.nn.Parameter(torch.zeros(self.ch, dtype=torch.float32, device=_dev()), requires_grad=False)
            self.running_var = torch.nn.Parameter(torch.ones(self.ch, dtype=torch.float32, device=_dev()), requires_grad=False)
        else:
            self.weight = self.bias = None
            self.running_mean = torch.nn.Parameter(torch.zeros(self.ch, dtype=torch.float32, device=_dev()), requires_grad=False)
            self.running_var = torch.nn.Parameter(torch.ones(self.ch, dtype=torch.float32, device=_dev()), requires_grad=False)

    def affine(self):
        s = (self.weight.float() / torch.sqrt(self.running_var + 1e-5)).to(self.dtype)
        b = (self.bias.float() - self.running_mean.float() * s.float()).to(self.dtype)
        return s, b

    def forward(self, x):
        if not self.use_bn:
            return x
        if self.training:
            # 同 ConvBn：必须走 .data，否则原地更新会让 autograd 的 version 校验失败
            with torch.no_grad():
                self.running_mean.data.mul_(1 - 0.1).add_(0.1 * x.detach().mean(dim=(0, 2, 3)).float())
                self.running_var.data.mul_(1 - 0.1).add_(0.1 * x.detach().var(dim=(0, 2, 3), unbiased=False).float())
        s, b = self.affine()
        return x * s.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)


class RepBlock(nn.LayerBase):
    """RepVGG 残差块：conv kxk + conv 1x1 + identity 三分支，推理期可折叠成单 kxk。

    折叠后在数值上与原三分支等价（误差 ~1e-5 量级），参数量约降为 1/3。
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, strides=1, use_bn=True,
                 dtype=None, name=None, **kwargs):
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size, self.strides = int(kernel_size), int(strides)
        self.use_bn, self.dtype = use_bn, dtype
        self.has_id = (in_ch == out_ch and self.strides == 1)
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        k = self.kernel_size
        self.conv_kxk = ConvBn(self.in_ch, self.out_ch, k, self.strides, (k - 1) // 2,
                               use_bn=self.use_bn, dtype=self.dtype)
        self.conv_1x1 = ConvBn(self.in_ch, self.out_ch, 1, self.strides, 0,
                               use_bn=self.use_bn, dtype=self.dtype)
        if self.has_id:
            self.idbn = IdBn(self.out_ch, use_bn=self.use_bn, dtype=self.dtype)
            self.rk = None
        else:
            self.idbn = None
            self.rk = ConvBn(self.in_ch, self.out_ch, 1, self.strides, 0, use_bn=False, dtype=self.dtype)
        self.fused_w = None
        self.fused_b = None

    @property
    def folded(self):
        return self.fused_w is not None

    @torch.no_grad()
    def fold(self):
        if self.folded:
            return
        k = self.kernel_size
        pad = (k - 1) // 2

        Wk = self.conv_kxk.weight.detach().clone()
        if self.conv_kxk.use_bn:
            s, b = self.conv_kxk.bn_affine()
            Wk = Wk * s.view(-1, 1, 1, 1)
            bk = b
        else:
            bk = self.conv_kxk.bias.detach().clone()

        W1 = self.conv_1x1.weight.detach().clone()
        if self.conv_1x1.use_bn:
            s, b = self.conv_1x1.bn_affine()
            W1 = W1 * s.view(-1, 1, 1, 1)
            b1 = b
        else:
            b1 = self.conv_1x1.bias.detach().clone()
        W1p = F.pad(W1, [pad, pad, pad, pad])

        if self.has_id:
            s, b = self.idbn.affine()
            Wi = torch.zeros_like(Wk)
            idx = torch.arange(self.out_ch, device=Wk.device)
            Wi[idx, idx, pad, pad] = 1.0
            Wi = Wi * s.view(-1, 1, 1, 1)
            bi = b
        else:
            Wi = self.rk.weight.detach().clone()
            bi = self.rk.bias.detach().clone()

        self.fused_w = torch.nn.Parameter((Wk + W1p + Wi).to(self.dtype), requires_grad=False)
        self.fused_b = torch.nn.Parameter((bk + b1 + bi).to(self.dtype), requires_grad=False)

    def forward(self, x):
        if self.folded:
            return F.conv2d(x, self.fused_w, self.fused_b, stride=self.strides,
                            padding=(self.kernel_size - 1) // 2)
        out = self.conv_kxk(x) + self.conv_1x1(x)
        if self.has_id:
            out = out + self.idbn(x)
        else:
            out = out + self.rk(x)
        return out


class DWRepBlock(nn.LayerBase):
    """【dw5 x repvgg 的叠加】Conv1x1( DW5x5(x) + BN_id(x) )，括号内推理期可折叠。

    为什么是这个结构 —— 代数约束（用 float64 逐分支验证过）：

      * DW5x5 与 identity 的核形状相同（identity = 只在中心位置系数为 1 的 depthwise 核），
        所以两者【可以】直接相加，折叠成单个 depthwise 5x5。
      * pointwise 1x1 必须【混合通道】，也就是 groups=1。
        它不能折进 depthwise —— 两个逐元素(diagonal)算子的复合不是逐元素算子，
        实测 pw(dw(x)) != dw(x)+pw(x)（误差 5.1，量级同信号本身）。
      * 于是正确的组合是：把可折的 part 放进括号，pointwise 留在括号外。

    训练期：DW5x5 + BN_id + Conv1x1  = 3 个算子
    推理期：单个 depthwise 5x5 + 单个 1x1 = 2 个算子（且第一个仍是 depthwise，
            所以 5x5 感受野和 ~C*25 的参数量都保住）

    与稠密 RepBlock 的差别：稠密版折出来是稠密 3x3（感受野掉到 3x3），
    这里折出来仍是 depthwise 5x5。
    """
    def __init__(self, ch, kernel_size=5, use_bn=True, dtype=None, name=None,
                 kernels=None, **kwargs):
        self.ch = ch
        # kernels: depthwise 分支的核尺寸序列。给多个就做多尺度（同 groups，仍可折）。
        self.kernels = tuple(kernels) if kernels else (int(kernel_size),)
        self.kernel_size = max(self.kernels)
        self.use_bn, self.dtype = use_bn, dtype
        super().__init__(name=name, **kwargs)

    def build_weights(self):
        # 可折分支：多个 depthwise（不同核尺寸）+ identity。
        # groups 都等于 ch，所以能各自 pad 到 max(k) 后相加。
        #
        # 必须用 ModuleList：普通 Python list 不会被 torch.nn.Module 注册为子模块，
        # 那些 ConvBn 的权重就不会出现在 parameters() 里，也不会被 .cuda()/.to()
        # 搬设备（实测 get_weights 只返回 6 个而不是 14 个）。
        self.dws = torch.nn.ModuleList()
        for i, k in enumerate(self.kernels):
            self.dws.append(ConvBn(self.ch, self.ch, k, 1, (k - 1) // 2, groups=self.ch,
                                   use_bn=self.use_bn, dtype=self.dtype))
        self.idbn = IdBn(self.ch, use_bn=self.use_bn, dtype=self.dtype)
        # 不可折分支：真正的 pointwise（混合通道，groups=1）
        self.pw = PWConv(self.ch, self.ch, use_bias=True, dtype=self.dtype)
        self.fused_w = None
        self.fused_b = None

    @property
    def folded(self):
        return self.fused_w is not None

    @torch.no_grad()
    def fold(self):
        if self.folded:
            return
        kmax = self.kernel_size
        pad = (kmax - 1) // 2

        Wf = None
        bf = None
        for dw in self.dws:
            W = dw.weight.detach().clone()            # (C,1,k,k)
            if dw.use_bn:
                s, b = dw.bn_affine()
                W = W * s.view(-1, 1, 1, 1)
            else:
                b = dw.bias.detach().clone()
            p = (kmax - dw.kernel_size) // 2
            W = F.pad(W, [p, p, p, p])                # 小核补零到 kmax，形状即可相加
            Wf = W if Wf is None else Wf + W
            bf = b if bf is None else bf + b

        s2, bid = self.idbn.affine()
        Wi = torch.zeros_like(Wf)
        Wi[:, 0, pad, pad] = 1.0
        Wf = Wf + Wi * s2.view(-1, 1, 1, 1)
        bf = bf + bid

        self.fused_w = torch.nn.Parameter(Wf.to(self.dtype), requires_grad=False)
        self.fused_b = torch.nn.Parameter(bf.to(self.dtype), requires_grad=False)

    def forward(self, x):
        if self.folded:
            h = F.conv2d(x, self.fused_w, self.fused_b,
                         padding=(self.kernel_size - 1) // 2, groups=self.ch)
        else:
            h = None
            for dw in self.dws:
                v = dw(x)
                h = v if h is None else h + v
            h = h + self.idbn(x)
        return self.pw(h)


# ============================================================================
# 架构
# ============================================================================

class DFLiteArchi(ArchiBase):
    """opts 与 DeepFakeArchi 同义：u/pixel_norm, d/半分辨率瓶颈, t/5 次下采样。

    额外关键字（不属于 opts 字符串，避免污染 Model.py 的 opts 白名单）：
        ae_dims          瓶颈宽度，默认 512
        kernel           'dw5'（默认）用 DW5x5+PW1x1；'3x3' 用稠密 3x3
        use_bn           RepBlock / ConvBn 是否带 BatchNorm，默认 True
        inter_stride     瓶颈投影用 stride 而非 g*g 空间放大（省参数）
    """

    def __init__(self, resolution, use_fp16=False, use_bf16=False, mod=None, opts=None,
                 ae_dims=512, kernel='dw5', use_bn=True, inter_stride=False):
        super().__init__()
        if opts is None:
            opts = ''
        self.resolution = int(resolution)
        self.opts = opts
        self.kernel = kernel
        self.use_bn = use_bn
        self.inter_stride = inter_stride
        self.ae_dims = int(ae_dims)
        if use_bf16:
            conv_dtype = torch.bfloat16
        elif use_fp16:
            conv_dtype = torch.float16
        else:
            conv_dtype = torch.float32
        self.conv_dtype = conv_dtype
        self.use_fp16 = use_fp16

        lite_mod = kernel  # 让闭包捕获

        # ---------------- 下采样 ----------------
        class DownLite(nn.ModelBase):
            def __init__(self, in_ch, out_ch, use_dw=True, **kwargs):
                self.in_ch, self.out_ch, self.use_dw = in_ch, out_ch, use_dw
                super().__init__(**kwargs)

            def on_build(self):
                if self.use_dw:
                    # DW 5x5 stride2 + PW 1x1：感受野与 5x5 稠密一致
                    self.dw = DWConv(self.in_ch, 5, strides=2, dtype=conv_dtype)
                    # PW 用 BN 压住逐层放大的激活（实测没有 BN 时 absmax 会涨到 28）
                    self.pw = ConvBn(self.in_ch, self.out_ch, 1, 1, 0,
                                     use_bn=True, dtype=conv_dtype)
                else:
                    self.conv = ConvBn(self.in_ch, self.out_ch, 3, 2, 1,
                                       use_bn=use_bn, dtype=conv_dtype)

            def forward(self, x):
                if self.use_dw:
                    return self.pw(self.dw(x))
                return self.conv(x)

            def get_out_ch(self):
                return self.out_ch

        # ---------------- 残差块 ----------------
        class ResLite(nn.ModelBase):
            def __init__(self, ch, **kwargs):
                self.ch = ch
                super().__init__(**kwargs)

            def on_build(self):
                # 'repdw5' 与 'dw5' 用同一个块（DWRepBlock 内部已含 identity 残差分支），
                # 区别只在下采样层是否也用 depthwise。
                if _use_dw_blocks(lite_mod):
                    self.blk = DWRepBlock(self.ch, 5, use_bn=use_bn, dtype=conv_dtype,
                                          kernels=_block_kernels(lite_mod))
                else:
                    self.blk = RepBlock(self.ch, self.ch, 3, 1, use_bn=use_bn, dtype=conv_dtype)

            def _act(self, x):
                return x * torch.cos(x) if 'c' in opts else F.leaky_relu(x, 0.2)

            def forward(self, inp):
                return self._act(inp + self.blk(inp))

        # ---------------- 编码器 ----------------
        class Encoder(nn.ModelBase):
            def __init__(self, in_ch, e_ch, **kwargs):
                self.in_ch, self.e_ch = in_ch, e_ch
                super().__init__(**kwargs)

            def on_build(self):
                # 'repdw5' 的下采样层用稠密 3x3（保守），只有残差块用可折的 DW 叠加块
                use_dw = (lite_mod == 'dw5')
                if 't' in opts:
                    self.down1 = DownLite(self.in_ch, self.e_ch, use_dw)
                    self.res1 = ResLite(self.e_ch)
                    self.down2 = DownLite(self.e_ch, self.e_ch * 2, use_dw)
                    self.down3 = DownLite(self.e_ch * 2, self.e_ch * 4, use_dw)
                    self.down4 = DownLite(self.e_ch * 4, self.e_ch * 8, use_dw)
                    self.down5 = DownLite(self.e_ch * 8, self.e_ch * 8, use_dw)
                    self.res5 = ResLite(self.e_ch * 8)
                else:
                    self.down1 = DownLite(self.in_ch, self.e_ch, use_dw)
                    self.down2 = DownLite(self.e_ch, self.e_ch * 2, use_dw)
                    self.down3 = DownLite(self.e_ch * 2, self.e_ch * 4, use_dw)
                    self.down4 = DownLite(self.e_ch * 4, self.e_ch * 8, use_dw)

            def forward(self, x):
                if use_fp16:
                    x = x.to(torch.float16)
                if 't' in opts:
                    x = self.res1(self.down1(x))
                    x = self.down2(x); x = self.down3(x); x = self.down4(x)
                    x = self.res5(self.down5(x))
                else:
                    x = self.down1(x); x = self.down2(x)
                    x = self.down3(x); x = self.down4(x)
                x = nn.flatten(x)
                if 'u' in opts:
                    x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
                if use_fp16:
                    x = x.to(torch.float32)
                return x

            def get_out_res(self, res):
                # 必须不依赖 on_build（它是惰性的）。't' 有 5 次下采样，否则 4 次。
                return res // (2 ** (5 if 't' in opts else 4))

            def get_out_ch(self):
                return self.e_ch * 8

        # ---------------- 瓶颈 ----------------
        # DFLite 自己的网格 = baseline inter 网格的一半。
        # 实测 baseline 的 inter 网格：
        #   基础 res/16；只有带 '-d' 时才变 res/32（'-u' 和 '-t' 都不改它）
        # 所以 DFLite 的网格：带 -d -> res/32，其余 -> res/16。
        lowest_dense_res = self.resolution // (32 if 'd' in opts else 16)

        class Inter(nn.ModelBase):
            def __init__(self, in_ch, ae_ch, ae_out_ch, **kwargs):
                self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
                super().__init__(**kwargs)

            def on_build(self):
                in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
                self.dense1 = nn.Dense(in_ch, ae_ch, dtype=conv_dtype)
                # 摊到空间网格的边长。实测 baseline 的 inter 输出张量：
                #   无 -d : res/16（无论有没有 -t / -u）
                #   有 -d : res/32
                # 注意：这里是 DFLite 自己的网格！它的 UpLite 自带 pixel_shuffle，
                # 所以网格值直接是输出分辨率，不需要再乘 2。
                g = lowest_dense_res
                # stride 路径：瓶颈再摊到 g/2 网格，用 1x1 + pixel_shuffle 升回 g
                self.g = (g // 2) if inter_stride else g
                self.expand_by_shuffle = bool(inter_stride)
                self.dense2 = nn.Dense(ae_ch, self.g * self.g * ae_out_ch, dtype=conv_dtype)
                if self.expand_by_shuffle:
                    self.proj = ConvBn(ae_out_ch, ae_out_ch * 4, 1, 1, 0,
                                       use_bn=False, dtype=conv_dtype)
                    self.upscale1 = None
                else:
                    self.proj = None
                    # 不需要 upscale1。
                    # 实测：DFLite 的解码器阶梯已经自带足够的上采样 ——
                    #   -t   : 网格 res/32，4 级 = 16x  -> 到 res
                    #   非-t : 网格 res/16，3 级 = 8x   -> 到 res
                    # 再加一次 2x 就会全部变成 2 倍尺寸（实测：加上它时非 -t 的输出
                    # 是 512 而不是 256）。baseline 的 inter 输出网格比 DFLite 小一半，
                    # 所以它需要 upscale1 补这一级，DFLite 不需要。
                    self.upscale1 = None

            def forward(self, inp):
                x = self.dense2(self.dense1(inp))
                x = nn.reshape_4D(x, self.g, self.g, self.ae_out_ch)
                # 注意：nn.reshape_4D 返回的就是 NCHW（x.reshape(-1, c, h, w)），
                # 不要再来一次 permute —— 那会把 (N,C,H,W) 打乱成 (N,W,C,H)。
                if self.expand_by_shuffle:
                    return F.pixel_shuffle(self.proj(x), 2)
                if use_fp16:
                    x = x.to(torch.float16)
                if self.upscale1 is not None:
                    x = F.pixel_shuffle(self.upscale1(x), 2)
                return x

            def get_out_res(self):
                # 与 forward 实际输出的空间尺寸保持一致（baseline 的这个 getter 是不准的，
                # 但 DFLite 里把它改成真值，避免误用）
                return self.g * 2 if inter_stride else self.g

            def get_out_ch(self):
                return self.ae_out_ch

        class DecoderWrapper(nn.ModelBase):
            """把 _make_decoder 返回的类包一层，使 a.Decoder(in_ch, d_ch, d_mask_ch) 与 DFL 调用一致。"""
            def __init__(self, in_ch, d_ch, d_mask_ch, **kwargs):
                self.in_ch, self.d_ch, self.d_mask_ch = in_ch, d_ch, d_mask_ch
                super().__init__(**kwargs)

            def on_build(self):
                self.impl = _make_decoder(self.in_ch, self.d_ch, self.d_mask_ch,
                                          opts, conv_dtype, lite_mod, use_bn, use_fp16, resolution)()

            def get_weights(self):
                # 必须覆盖：ModelBase.get_weights() 会沿着 self.layers 走，
                # 而 self.impl 同时被 ModelBase 的 layers 和 torch.nn.Module 的
                # _modules 两条路注册，导致同一张量被返回两次（实测 65 个 vs 实际 41 个）。
                # 重复会让 optimizer 重复更新同一参数、也会让存档/读档数量对不上。
                if not self.built:
                    self.build()
                seen, out = set(), []
                for p in self.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
                return out

            def forward(self, z, mask_hint=None):
                return self.impl(z, mask_hint)

        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = DecoderWrapper


def _make_decoder(in_ch, d_ch, d_mask_ch, opts, conv_dtype, lite_mod, use_bn, use_fp16, resolution):
    """按 opts 生成 Decoder 类，层级与通道阶梯**严格对齐** DeepFakeArchi 的 Decoder。

    对齐依据（实测 baseline 的 _modules 得到，res=256 / d_ch=64 / d_mask_ch=32 / ae_dims=64）：

      opts          图像 upscale 阶梯                        末端 shuffle
      (空)/u/ud      3 级: d*8 -> d*4 -> d*2 -> 3ch         无
      t/ut           4 级: d*8 -> d*8 -> d*4 -> d*2 -> 3ch  无
      d/ud           3 级 + 末端: d*8 -> d*4 -> d*2 -> 12ch 有
      dt/udt         4 级 + 末端: ... -> d*2 -> 12ch        有

    规则：n_up = 3 + (1 if 't' in opts else 0)，若带 -d 则**再加一级**（该级不翻倍，
    只把通道扩到 4*C 交给末端 depth_to_space(2) 完成最后一次 2x）。
    mask 侧同一套规则，宽度基数换成 d_mask_ch。
    """
    has_d = ('d' in opts)
    # 图像侧通道阶梯（从 in_ch 出发逐级 1/2）。
    # baseline 的 raw ladder：无 -t -> [8,4,2]；有 -t -> [8,8,4,2]。
    # -d 不追加图像侧层级 —— 它只是把末端 out_conv 的通道扩到 4x，
    # 由 depth_to_space(2) 完成最后一次 2x（mask 侧的 upscalem4 是纯 1x1，
    # 不改变空间尺寸）。多加一级会让输出变成 2 倍尺寸。
    # 图像侧：'-t' 起始网格 res/32，需要 4 级（前两级都停在 8 保证通道阶梯不断层）；
    #         非 '-t' 起始网格 res/16，需要 3 级。
    # 图像阶梯的 unit 序列。实测 baseline（8 个 opts 全量）：
    #   -t   -> [8, 8, 4, 2]（4 级 = 16x，网格 res/32 -> res）
    #   非-t -> [8, 4, 2, 2]（4 级；前 3 级做 2x=8x，末级是纯 1x1，
    #                          网格 res/16 -> res）
    # 注意「级数」都是 4，差别在最后一级做不做 2x。
    units = [8, 8, 4, 2] if 't' in opts else [8, 4, 2, 2]

    class ResLite(nn.ModelBase):
        def __init__(self, ch, **kwargs):
            self.ch = ch
            super().__init__(**kwargs)

        def on_build(self):
            if _use_dw_blocks(lite_mod):
                self.blk = DWRepBlock(self.ch, 5, use_bn=use_bn, dtype=conv_dtype,
                                      kernels=_block_kernels(lite_mod))
            else:
                self.blk = RepBlock(self.ch, self.ch, 3, 1, use_bn=use_bn, dtype=conv_dtype)

        def _act(self, x):
            return x * torch.cos(x) if 'c' in opts else F.leaky_relu(x, 0.2)

        def forward(self, inp):
            return self._act(inp + self.blk(inp))

    class UpLite(nn.ModelBase):
        """conv3x3 -> act -> pixel_shuffle(2)；纯 1x1 权重时与 baseline 的 Upscale 等价。"""
        def __init__(self, i, o, **kwargs):
            self.i, self.o = i, o
            super().__init__(**kwargs)

        def on_build(self):
            self.conv1 = ConvBn(self.i, self.o * 4, 3, 1, 1, use_bn=False, dtype=conv_dtype)

        def _act(self, x):
            return x * torch.cos(x) if 'c' in opts else F.leaky_relu(x, 0.1)

        def forward(self, x):
            return nn.depth_to_space(self._act(self.conv1(x)), 2)

        def get_out_ch(self):
            return self.o

    class Decoder(nn.ModelBase):
        def on_build(self):
            # 通道阶梯：严格按 baseline 实测规则 —— 每级输出宽度 = d_ch * unit，
            # unit 序列 [-t -> 8,8,4,2；无 -t -> 8,4,2]（实测 d_ch=32/48/64/96/... 全对）。
            # 关键：第一层的输入是 in_ch（= ae_out_ch），输出是 d_ch*8 —— 两者无关，
            # 不能把 in_ch 当成阶梯基数（那会让 d_ch=64 时第一层变成 4096 通道）。
            n_up = 4
            img_units = [8, 8, 4, 2] if 't' in opts else [8, 4, 2, 2]
            img_w = [d_ch * u for u in img_units]
            self.img_w = img_w
            self.img_ups = []
            for i, w in enumerate(img_w):
                src = in_ch if i == 0 else img_w[i - 1]
                setattr(self, 'upscale%d' % i, UpLite(src, w))
                self.img_ups.append(getattr(self, 'upscale%d' % i))
                if i < len(img_w) - 1:
                    setattr(self, 'res%d' % i, ResLite(w))
            out_ch = 3 * 4 if has_d else 3
            self.out_conv = ConvBn(img_w[-1], out_ch, 1, 1, 0,
                                   use_bn=False, dtype=conv_dtype)

            # mask 侧：实测 baseline 的完整规则（8 个 opts 全对）
            #   (-d 或 -t) : [8, 8, 4, 2]   （4 级；-d 时末级 64->32 是纯 1x1）
            #   其余       : [8, 4, 2]      （3 级）
            # 宽度 = d_mask_ch * unit，与图像侧同一套 unit 语义。
            # 实测补齐后的规则：
            #   有 -d : [8, 8, 4, 2]（4 级，末级 64->32 是纯 1x1）
            #   无 -d : [8, 8, 4, 2] 的「前 3 级」—— 但因为起始网格已经是 res/16 或
            #           res/32，3 次 2x 配末端 shuffle 正好到 res
            if has_d:
                mask_units = [8, 8, 4, 2]
            else:
                mask_units = [8, 8, 4]
            mask_w = [d_mask_ch * u for u in mask_units]
            self.mask_w = mask_w
            for i, w in enumerate(mask_w):
                src = in_ch if i == 0 else mask_w[i - 1]
                setattr(self, 'upscalem%d' % i, UpLite(src, w))
            self.out_convm = ConvBn(mask_w[-1], 4, 1, 1, 0,
                                    use_bn=False, dtype=conv_dtype)
        def forward(self, z, mask_hint=None):
            x = z
            for i, _w in enumerate(self.img_w):
                x = getattr(self, 'upscale%d' % i)(x)
                if i < len(self.img_w) - 1:
                    x = getattr(self, 'res%d' % i)(x)
            if use_fp16:
                x = x.to(torch.float16)
            out = self.out_conv(x)
            y = nn.depth_to_space(out, 2) if has_d else out
            y = torch.sigmoid(y)

            m = z
            for i, _w in enumerate(self.mask_w):
                m = getattr(self, 'upscalem%d' % i)(m)
            m = torch.sigmoid(nn.depth_to_space(self.out_convm(m), 2))

            if use_fp16:
                y = y.to(torch.float32)
                m = m.to(torch.float32)
            return y, m

    return Decoder
