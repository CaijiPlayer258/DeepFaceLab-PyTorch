import numpy as np
import torch

def get_nn():
    """延迟导入nn以避免循环依赖"""
    from core.leras import nn
    return nn

def get_torch():
    """获取PyTorch模块"""
    return torch

def torch_get_value(tensor):
    t = get_torch()
    if isinstance(tensor, t.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def batch_set_value(tuples):
    if len(tuples) != 0:
        for param, value in tuples:
            if isinstance(value, type(param)):
                param.data.copy_(value.data)
            else:
                if isinstance(value, np.ndarray):
                    param.data.copy_(get_torch().from_numpy(value))
                else:
                    param.data.copy_(get_torch().tensor(value))

def init_weights(weights):
    # PyTorch initializes weights when creating the module
    # This function is here for compatibility
    pass

def torch_gradients(loss, vars):
    # PyTorch uses autograd, gradients are computed automatically
    # This is here for compatibility
    return [(None, v) for v in vars]

def average_gv_list(grad_var_list, device_string=None):
    if len(grad_var_list) == 1:
        return grad_var_list[0]
    
    t = get_torch()
    result = []
    for i, (gv) in enumerate(grad_var_list):
        for j,(g,v) in enumerate(gv):
            if g is None:
                continue
            g = g.unsqueeze(0)
            if i == 0:
                result += [ [[g], v]  ]
            else:
                result[j][0] += [g]

    for i,(gs,v) in enumerate(result):
        result[i] = (t.cat(gs, 0).mean(0), v)
    return result

def average_tensor_list(tensors_list, device_string=None):
    if len(tensors_list) == 1:
        return tensors_list[0]
    
    t = get_torch()
    return t.stack(tensors_list, 0).mean(0)

def concat(tensors_list, axis):
    """
    Better version.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    return get_torch().cat(tensors_list, axis)

def gelu(x):
    import math
    t = get_torch()
    return x * 0.5 * (1.0 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))

def upsample2d(x, size=2):
    t = get_torch()
    import torch.nn.functional as F
    return F.interpolate(x, scale_factor=size, mode='nearest')

def resize2d_bilinear(x, size=2):
    t = get_torch()
    import torch.nn.functional as F
    h = x.shape[2]
    w = x.shape[3]
    
    if size > 0:
        new_size = (h*size, w*size)
    else:
        new_size = (h//-size, w//-size)
    
    return F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

def resize2d_nearest(x, size=2):
    if size in [-1,0,1]:
        return x
    
    t = get_torch()
    import torch.nn.functional as F
    
    if size > 0:
        raise Exception("")
    else:
        x = x[:,:,::-size,::-size]
    return x

def flatten(x):
    # 原版 iperov：NHWC→NCHW 后 reshape（通道优先展平），与 DFL 权重布局一致
    return x.reshape(x.size(0), -1)

def max_pool(x, kernel_size=2, strides=2):
    t = get_torch()
    import torch.nn.functional as F
    return F.max_pool2d(x, kernel_size=kernel_size, stride=strides, padding=0)

def reshape_4D(x, w,h,c):
    # 原版 iperov：展平按 (c,h,w) 通道优先解释（与原版 NCHW 分支一致）
    return x.reshape(-1, c, h, w)

def random_normal(shape, mean=0.0, stddev=1.0, dtype=None):
    t = get_torch()
    if dtype is None:
        dtype = nn.floatx
    return t.randn(shape, dtype=dtype) * stddev + mean

def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None):
    t = get_torch()
    if dtype is None:
        dtype = nn.floatx
    return t.rand(shape, dtype=dtype) * (maxval - minval) + minval

def random_binomial(shape, p=0.0, dtype=None):
    t = get_torch()
    if dtype is None:
        dtype = nn.floatx
    return t.bernoulli(t.full(shape, p, dtype=dtype))

def tf_get_value(tensor):
    return torch_get_value(tensor)

def depth_to_space(x, block_size):
    """
    TF-equivalent depth_to_space for NCHW tensors.

    TF depth_to_space:
      output[n, c, h*r+i, w*r+j] = input[n, (i*r+j)*C_out + c, h, w]
    PT pixel_shuffle:
      output[n, c, h*r+i, w*r+j] = input[n, c*r² + i*r + j, h, w]

    We permute input channels so that pixel_shuffle produces TF-equivalent
    spatial layout — matching what the converted Conv2D weights expect.
    """
    import torch.nn.functional as F
    import torch

    C_in = x.shape[1]
    r = block_size
    C_out = C_in // (r * r)

    idx = torch.arange(C_in, device=x.device)
    c = idx // (r * r)       # output channel index
    ij = idx % (r * r)       # sub-pixel position (i*r+j) packed
    i = ij // r
    j = ij % r
    perm = (i * r + j) * C_out + c

    return F.pixel_shuffle(x[:, perm, :, :], r)

def space_to_depth(x, block_size):
    """
    空间到深度的转换
    将NCHW格式的tensor从(N, C, H*r, W*r)转换为(N, C*r^2, H, W)
    其中r是block_size
    """
    t = get_torch()
    import torch.nn.functional as F
    return F.pixel_unshuffle(x, block_size)
def gaussian_blur(x, kernel_size):
    """
    应用高斯模糊（可分离 1D 两遍——数学等价于 2D 高斯卷积，计算量 k²→2k，GPU 更快）
    
    Args:
        x: 输入张量 (N, C, H, W)
        kernel_size: 内核大小
        
    Returns:
        模糊后的张量
    """
    # 兼容：允许传入sigma(float)或kernel_size(int)
    if kernel_size is None:
        return x

    # sigma(float) -> 推导一个合理的kernel_size
    if isinstance(kernel_size, float):
        sigma = max(0.0, float(kernel_size))
        if sigma == 0.0:
            return x
        # 经验：覆盖约3*sigma，两侧各3*sigma
        ks = int(round(sigma * 6.0))
        kernel_size = max(3, ks + (1 - ks % 2))

    if kernel_size <= 0:
        return x
    
    t = get_torch()
    import torch.nn.functional as F
    
    # 确保kernel_size是奇数
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 计算sigma
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # 创建1D高斯核（可分离）
    coords = t.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= kernel_size // 2
    
    g = t.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    # 为每个通道创建1D核（可分离：先列后行，等价于 2D 外积核卷积）
    channels = x.shape[1]
    gv = g.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)  # (C,1,k,1)
    gh = g.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)  # (C,1,1,k)
    
    # 应用卷积（1D 两遍，SAME padding 与 2D 等价——列核只在 H 维 pad，行核只在 W 维 pad）
    padding = kernel_size // 2
    blurred = F.conv2d(x, gv, padding=(padding, 0), groups=channels)
    blurred = F.conv2d(blurred, gh, padding=(0, padding), groups=channels)
    
    return blurred


def total_variation_mse(x):
    """Total variation loss (MSE form) for NCHW tensor."""
    t = get_torch()
    if x.ndim != 4:
        raise ValueError('total_variation_mse expects NCHW 4D tensor')
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    return t.mean(dx * dx) + t.mean(dy * dy)


def style_loss(x, y, gaussian_blur_radius=0, loss_weight=1.0):
    """Simple gram-matrix style loss.

    Args:
        x, y: (N,C,H,W)
        gaussian_blur_radius: int kernel_size or float sigma (passed to gaussian_blur)
        loss_weight: scalar multiplier
    """
    t = get_torch()

    if gaussian_blur_radius not in (None, 0, 0.0):
        x = gaussian_blur(x, gaussian_blur_radius)
        y = gaussian_blur(y, gaussian_blur_radius)

    n, c, h, w = x.shape
    fx = x.view(n, c, h * w)
    fy = y.view(n, c, h * w)

    gx = t.bmm(fx, fx.transpose(1, 2)) / (c * h * w)
    gy = t.bmm(fy, fy.transpose(1, 2)) / (c * h * w)

    return float(loss_weight) * t.mean((gx - gy) ** 2)

def dssim(x, y, max_val=1.0, filter_size=11, k1=0.01, k2=0.03):
    """
    计算结构相似性指数的差异 (DSSIM)
    
    DSSIM = (1 - SSIM) / 2
    
    Args:
        x: 第一个输入张量 (N, C, H, W)
        y: 第二个输入张量 (N, C, H, W)
        max_val: 像素值的最大值
        filter_size: 高斯窗口大小
        k1, k2: SSIM常数
        
    Returns:
        DSSIM值张量 (N, C, 1, 1)
    """
    t = get_torch()
    import torch.nn.functional as F
    
    # 确保filter_size是奇数
    filter_size = int(filter_size)
    if filter_size % 2 == 0:
        filter_size += 1
    
    # 创建高斯窗口（可分离：1D 两遍，等价于 2D 外积核——VALID 输出形状相同，k²→2k 计算量）
    sigma = 1.5
    channels = x.shape[1]
    # 缓存 1D 核（按 filter_size/sigma/channels/device——省每次生成）
    cache_key = (filter_size, sigma, channels, str(x.device))
    _cache = globals().setdefault('_dssim_kernel_cache', {})
    gv_gh = _cache.get(cache_key)
    if gv_gh is None:
        coords = t.arange(filter_size, dtype=x.dtype, device=x.device)
        coords -= filter_size // 2
        g = t.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        gv = g.view(1, 1, filter_size, 1).repeat(channels, 1, 1, 1)  # (C,1,k,1)
        gh = g.view(1, 1, 1, filter_size).repeat(channels, 1, 1, 1)  # (C,1,1,k)
        gv_gh = (gv, gh)
        if len(_cache) > 64:
            _cache.clear()
        _cache[cache_key] = gv_gh
    gv, gh = gv_gh
    
    # SSIM常数
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    # 计算均值（padding=VALID，与原版 iperov DFL 一致——滤波器不越界，
    # 巨大饱和输入下 SAME 补 0 会扭曲边缘 dssim 数值/梯度导致训练发散）
    def _sep(xx):
        v = F.conv2d(xx, gv, padding=0, groups=channels)
        return F.conv2d(v, gh, padding=0, groups=channels)
    mu_x = _sep(x)
    mu_y = _sep(y)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    # 计算方差和协方差
    sigma_x_sq = _sep(x ** 2) - mu_x_sq
    sigma_y_sq = _sep(y ** 2) - mu_y_sq
    sigma_xy = _sep(x * y) - mu_xy
    
    # 计算SSIM（数值稳定：巨大饱和输入下分母接近 0 会导致 luminance/cs 巨大 → 梯度爆炸发散）
    eps = 1e-6
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x_sq + mu_y_sq + c1 + eps) * (sigma_x_sq + sigma_y_sq + c2 + eps))
    ssim_map = ssim_map.clamp(-1.0, 1.0)
    
    # 计算DSSIM
    dssim_map = (1 - ssim_map) / 2
    
    # 在空间维度上取平均
    dssim_val = t.mean(dssim_map, dim=[2, 3], keepdim=True)
    
    return dssim_val

def _masked_mean_std(x, mask):
    """遮罩加权的逐通道均值/标准差（mask: (N,1,H,W)，与 x:(N,C,H,W) 广播）。"""
    t = get_torch()
    dims = [2, 3]
    wsum = mask.sum(dim=dims, keepdim=True) + 1e-8
    m = (x * mask).sum(dim=dims, keepdim=True) / wsum
    var = ((x - m) ** 2 * mask).sum(dim=dims, keepdim=True) / wsum
    return m, t.sqrt(var + 1e-8)


def color_transfer_torch(alg, img, ref, mask=None):
    """把 img 的**全局色彩**迁移到 ref 的色彩分布上（可微，NCHW）。

    用途：随机偏色训练时，loss 比较的是 CT(pred, C) 与 C，
    于是全局色彩被中性化，模型只需对结构负责 —— 输入色彩异常也不会导致解码异常。

    alg: 'rct' -> 均值/标准差匹配，但**在去相关空间**（Y / R-Y / B-Y）里做。
                  这对应 numpy 侧 Reinhard 用 Lab 的那一层语义：
                  亮度与色度解耦后再各自匹配，不会因为缩放某一通道而扭曲色相/饱和度。
         'mt'  -> 均值/标准差匹配，直接在**原生通道空间**做（遮罩加权）。
                  对应 numpy 侧的 match_tone_np（截图那一版）。
         'lct' -> 逐样本线性最小二乘（img --affine--> ref 的分布）
         其它 / None / 'none' -> 原样返回
    mask: 可选 (N,1,H,W)。给了就**只在该区域内统计**均值/方差。
          全图统计会被背景/头发带偏（脸只占一部分），所以训练时应当传脸部遮罩。
    """
    t = get_torch()
    if alg in (None, '', 'none'):
        return img

    if alg in ('rct', 'mt'):
        dims = [2, 3]
        if alg == 'rct':
            # BGR -> [ Y , R-Y , B-Y ]：一个线性去相关变换（BT.601 亮度权重）。
            # 常数偏移被均值匹配吸收，所以这里不需要 0.5 偏置，也不会引入误差。
            # 在去相关空间里各自匹配 -> 不会因为缩放单通道而改色相/饱和度，
            # 这正是 Reinhard 原论文要求用 Lab 的那个性质（YCbCr 是它的线性近似）。
            ycc = t.tensor([[0.114,  0.587,  0.299],
                            [-0.114, -0.587, 0.701],
                            [0.886, -0.587, -0.299]], device=img.device, dtype=img.dtype)
            ycc_inv = t.linalg.inv(ycc)
            a = t.einsum('ij,njhw->nihw', ycc, img)
            b = t.einsum('ij,njhw->nihw', ycc, ref)
        else:
            a, b = img, ref

        if mask is None:
            m_a = a.mean(dim=dims, keepdim=True)
            s_a = a.std(dim=dims, keepdim=True) + 1e-5
            m_b = b.mean(dim=dims, keepdim=True)
            s_b = b.std(dim=dims, keepdim=True) + 1e-5
        else:
            m_a, s_a = _masked_mean_std(a, mask); s_a = s_a + 1e-5
            m_b, s_b = _masked_mean_std(b, mask); s_b = s_b + 1e-5

        out = (a - m_a) / s_a * s_b + m_b
        if alg == 'rct':
            out = t.einsum('ij,njhw->nihw', ycc_inv, out)
        return out

    if alg == 'lct':
        n, c, hh, ww = img.shape
        x = img.permute(0, 2, 3, 1).reshape(n, -1, c)
        y = ref.permute(0, 2, 3, 1).reshape(n, -1, c)
        xa = t.cat([x, t.ones_like(x[..., :1])], dim=-1)          # (N, HW, C+1)
        xt = xa.transpose(1, 2)                                     # (N, C+1, HW)
        eye = t.eye(xa.shape[-1], device=img.device, dtype=img.dtype) * 1e-4
        if mask is None:
            a_mat = t.matmul(xt, xa) + eye
            b_mat = t.matmul(xt, y)
        else:
            wm = mask.permute(0, 2, 3, 1).reshape(n, -1, 1)         # (N, HW, 1)
            a_mat = t.matmul(xt, xa * wm) + eye
            b_mat = t.matmul(xt, y * wm)
        sol = t.linalg.solve(a_mat, b_mat)                          # (N, C+1, C)
        out = t.matmul(xa, sol).reshape(n, hh, ww, c).permute(0, 3, 1, 2)
        return out

    raise ValueError('unknown color_transfer_torch alg: %r' % (alg,))


def register_ops():
    """注册所有ops函数到nn模块，避免循环依赖"""
    nn = get_nn()
    
    # 基础操作
    nn.torch_get_value = torch_get_value
    nn.batch_set_value = batch_set_value
    nn.init_weights = init_weights
    nn.gradients = torch_gradients
    nn.average_gv_list = average_gv_list
    nn.average_tensor_list = average_tensor_list
    nn.concat = concat
    nn.gelu = gelu
    nn.upsample2d = upsample2d
    nn.resize2d_bilinear = resize2d_bilinear
    nn.resize2d_nearest = resize2d_nearest
    nn.flatten = flatten
    nn.max_pool = max_pool
    nn.reshape_4D = reshape_4D
    nn.random_normal = random_normal
    nn.random_uniform = random_uniform
    nn.random_binomial = random_binomial
    nn.tf_get_value = tf_get_value
    nn.depth_to_space = depth_to_space
    nn.space_to_depth = space_to_depth
    nn.gaussian_blur = gaussian_blur
    nn.dssim = dssim
    nn.total_variation_mse = total_variation_mse
    nn.style_loss = style_loss
    nn.color_transfer_torch = color_transfer_torch
