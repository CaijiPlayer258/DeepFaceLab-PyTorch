import cv2
import numexpr as ne
import numpy as np
import scipy as sp
from numpy import linalg as npla


def color_transfer_sot(src,trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer

    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h,w,c = src.shape
    new_src = src.copy()

    advect = np.empty ( (h*w,c), dtype=src_dtype )
    for step in range (steps):
        advect.fill(0)
        for batch in range (batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = np.sum( trg*dir, axis=-1).reshape ((h*w))

            idSource = np.argsort (projsource)
            idTarget = np.argsort (projtarget)

            a = projtarget[idTarget]-projsource[idSource]
            for i_c in range(c):
                advect[idSource,i_c] += a * dir[i_c]
        new_src += advect.reshape( (h,w,c) ) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src-src
        src_diff_filt = cv2.bilateralFilter (src_diff, 0, reg_sigmaV, reg_sigmaXY )
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[...,None]
        new_src = src + src_diff_filt
    return new_src

def color_transfer_mkl(x0, x1):
    eps = np.finfo(float).eps

    h,w,c = x0.shape
    h1,w1,c1 = x1.shape

    x0 = x0.reshape ( (h*w,c) )
    x1 = x1.reshape ( (h1*w1,c1) )

    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    result = np.dot(x0-mx0, t) + mx1
    return np.clip ( result.reshape ( (h,w,c) ).astype(x0.dtype), 0, 1)

def color_transfer_idt(i0, i1, bins=256, n_rot=20):
    import scipy.stats
    
    relaxation = 1 / n_rot
    h,w,c = i0.shape
    h1,w1,c1 = i1.shape

    i0 = i0.reshape ( (h*w,c) )
    i1 = i1.reshape ( (h1*w1,c1) )

    n_dims = c

    d0 = i0.T
    d1 = i1.T

    for i in range(n_rot):

        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):

            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _     = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return np.clip ( d0.T.reshape ( (h,w,c) ).astype(i0.dtype) , 0, 1)

def reinhard_color_transfer(target : np.ndarray, source : np.ndarray, target_mask : np.ndarray = None, source_mask : np.ndarray = None, mask_cutoff=0.5) -> np.ndarray:
    """
    Transfer color using rct method.

        target      np.ndarray H W 3C   (BGR)   np.float32
        source      np.ndarray H W 3C   (BGR)   np.float32

        target_mask(None)   np.ndarray H W 1C  np.float32
        source_mask(None)   np.ndarray H W 1C  np.float32
        
        mask_cutoff(0.5)    float

    masks are used to limit the space where color statistics will be computed to adjust the target

    reference: Color Transfer between Images https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    """
    # OpenCV 的 cvtColor 不支持 float64(CV_64F)。
    # 在合成链路里（例如与 python float 混合运算）可能把 float32 结果提升为 float64，
    # 这里按函数注释约定（输入应为 np.float32）做一次强制转换，以避免潜在的问题。
    if target.dtype != np.float32:
        target = np.asarray(target, dtype=np.float32)
    if source.dtype != np.float32:
        source = np.asarray(source, dtype=np.float32)
    target = np.ascontiguousarray(target)
    source = np.ascontiguousarray(source)

    if target_mask is not None and target_mask.dtype != np.float32:
        target_mask = np.asarray(target_mask, dtype=np.float32)
    if source_mask is not None and source_mask.dtype != np.float32:
        source_mask = np.asarray(source_mask, dtype=np.float32)

    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # 统计量**只在遮罩区域内**求。
    # 旧写法是把遮罩外置 0、再对【整幅图】求 mean/std —— 那等于把统计量按"遮罩覆盖率"稀释：
    #     mean_算出来 = Σ(遮罩内) / (H*W) = 覆盖率 * 真实均值
    # 于是 out = (x - m_t*k)/(s_t*k) * (s_s*k) + m_s*k
    #   -> 斜率被比值抵消（正常），但**偏移量算错** -> 合成后残留一层洗不掉的色偏。
    def _mask_stats(x, m):
        if m is None:
            return ([x[..., i].mean() for i in range(3)],
                    [x[..., i].std() for i in range(3)])
        sel = m[..., 0] >= mask_cutoff
        if int(sel.sum()) < 8:
            return ([x[..., i].mean() for i in range(3)],
                    [x[..., i].std() for i in range(3)])
        return ([x[..., i][sel].mean() for i in range(3)],
                [x[..., i][sel].std() for i in range(3)])

    (target_l_mean, target_a_mean, target_b_mean), \
        (target_l_std, target_a_std, target_b_std) = _mask_stats(target, target_mask)

    (source_l_mean, source_a_mean, source_b_mean), \
        (source_l_std, source_a_std, source_b_std) = _mask_stats(source, source_mask)
    
    # not as in the paper: scale by the standard deviations using reciprocal of paper proposed factor
    target_l = target[...,0]
    target_l = ne.evaluate('(target_l - target_l_mean) * source_l_std / target_l_std + source_l_mean')

    target_a = target[...,1]
    target_a = ne.evaluate('(target_a - target_a_mean) * source_a_std / target_a_std + source_a_mean')
    
    target_b = target[...,2]
    target_b = ne.evaluate('(target_b - target_b_mean) * source_b_std / target_b_std + source_b_mean')

    np.clip(target_l,    0, 100, out=target_l)
    np.clip(target_a, -127, 127, out=target_a)
    np.clip(target_b, -127, 127, out=target_b)

    out = cv2.cvtColor(np.stack([target_l, target_a, target_b], -1), cv2.COLOR_LAB2BGR)
    # 保持与调用方的 float32 约定一致。
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape( t.shape[-1],-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape( s.shape[-1],-1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)

def lab_image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr

def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h,w,c = src_im.shape
    matched_R = channel_hist_match(src_im[:,:,0], tar_im[:,:,0], hist_match_threshold, None)
    matched_G = channel_hist_match(src_im[:,:,1], tar_im[:,:,1], hist_match_threshold, None)
    matched_B = channel_hist_match(src_im[:,:,2], tar_im[:,:,2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += ( src_im[:,:,i],)


    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched

def color_transfer_mix(img_src,img_trg):
    img_src = np.clip(img_src*255.0, 0, 255).astype(np.uint8)
    img_trg = np.clip(img_trg*255.0, 0, 255).astype(np.uint8)

    img_src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    img_trg_lab = cv2.cvtColor(img_trg, cv2.COLOR_BGR2LAB)

    rct_light = np.clip ( linear_color_transfer(img_src_lab[...,0:1].astype(np.float32)/255.0,
                                                img_trg_lab[...,0:1].astype(np.float32)/255.0 )[...,0]*255.0,
                          0, 255).astype(np.uint8)

    img_src_lab[...,0] = (np.ones_like (rct_light)*100).astype(np.uint8)
    img_src_lab = cv2.cvtColor(img_src_lab, cv2.COLOR_LAB2BGR)

    img_trg_lab[...,0] = (np.ones_like (rct_light)*100).astype(np.uint8)
    img_trg_lab = cv2.cvtColor(img_trg_lab, cv2.COLOR_LAB2BGR)

    img_rct = color_transfer_sot( img_src_lab.astype(np.float32), img_trg_lab.astype(np.float32) )
    img_rct = np.clip(img_rct, 0, 255).astype(np.uint8)

    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_BGR2LAB)
    img_rct[...,0] = rct_light
    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_LAB2BGR)


    return (img_rct / 255.0).astype(np.float32)

def color_transfer(ct_mode, img_src, img_trg, src_mask=None, trg_mask=None):
    """
    color transfer for [0,1] float32 inputs
    """
    if ct_mode == 'lct':
        out = linear_color_transfer (img_src, img_trg)
    elif ct_mode == 'rct':
        out = reinhard_color_transfer(img_src, img_trg)
    elif ct_mode == 'mkl':
        out = color_transfer_mkl (img_src, img_trg)
    elif ct_mode == 'idt':
        out = color_transfer_idt (img_src, img_trg)
    elif ct_mode == 'sot':
        out = color_transfer_sot (img_src, img_trg)
        out = np.clip( out, 0.0, 1.0)
    elif ct_mode == 'lut':
        out = color_transfer_lut (img_src, img_trg)
    elif ct_mode == 'mt':
        # 截图那一版：遮罩加权、RGB 空间
        out = match_tone_np (img_src, img_trg, src_mask=src_mask, trg_mask=trg_mask)
    else:
        raise ValueError(f"unknown ct_mode {ct_mode}")
    return out


# ---------------------------------------------------------------------------
# 拟合式 3D LUT 色彩迁移
# ---------------------------------------------------------------------------
_LUT_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)   # BGR 亮度权重


def _lut_build_basis(x, size):
    """在 size^3 网格上，对每个像素做三线性插值 -> 稀疏设计矩阵 (N, size^3)。

    每个像素只触及 8 个节点，所以非常稀疏。
    """
    import scipy.sparse as sps
    n = len(x)
    g = np.clip(x, 0.0, 1.0) * (size - 1)
    i0 = np.clip(np.floor(g).astype(np.int32), 0, size - 2)
    f = (g - i0).astype(np.float32)

    rows, cols, vals = [], [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = ((f[:, 0] if dx else 1.0 - f[:, 0]) *
                       (f[:, 1] if dy else 1.0 - f[:, 1]) *
                       (f[:, 2] if dz else 1.0 - f[:, 2]))
                lin = ((i0[:, 0] + dx) * size + (i0[:, 1] + dy)) * size + (i0[:, 2] + dz)
                rows.append(np.arange(n, dtype=np.int32))
                cols.append(lin)
                vals.append(wgt)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    return sps.csr_matrix((vals, (rows, cols)), shape=(n, size ** 3), dtype=np.float32)


def _lut_laplacian(size):
    """size^3 网格上的 6 邻接图拉普拉斯（用于平滑先验）。"""
    import scipy.sparse as sps
    n = size ** 3
    idx = np.arange(n, dtype=np.int32).reshape(size, size, size)
    rows, cols = [], []
    for ax in range(3):
        a = np.take(idx, np.arange(size - 1), axis=ax)
        b = np.take(idx, np.arange(1, size), axis=ax)
        rows.append(a.ravel()); cols.append(b.ravel())
        rows.append(b.ravel()); cols.append(a.ravel())
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    vals = -np.ones(len(rows), dtype=np.float32)
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    L = sps.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L + sps.diags(deg)


def _lut_apply(img, lut, size):
    """三线性查表。img 为 HWC [0,1]，lut 为 (size,size,size,3)。"""
    h, w, c = img.shape
    x = img.reshape(-1, c)
    g = np.clip(x, 0.0, 1.0) * (size - 1)
    i0 = np.clip(np.floor(g).astype(np.int32), 0, size - 2)
    f = (g - i0).astype(np.float32)
    out = np.zeros_like(x)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = ((f[:, 0] if dx else 1.0 - f[:, 0]) *
                       (f[:, 1] if dy else 1.0 - f[:, 1]) *
                       (f[:, 2] if dz else 1.0 - f[:, 2]))[:, None]
                out += wgt * lut[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return np.clip(out.reshape(h, w, c), 0.0, 1.0)


def color_transfer_lut(src, trg, size=17, n_samples=40000, lam_smooth=0.05,
                       lam_ident=0.002, sot_steps=10, sot_batch=5, seed=0):
    """拟合式 3D LUT 色彩迁移（自适应 / 有界 / 非线性 / 确定性）。

    与 rct / lct / mkl 这类**仿射**方法的根本差别：
      * 输出只能取 LUT 表内插出来的值 -> **不会外推**，饱和区域（嘴唇、纯色）不会被顶爆；
      * 三维非线性查表 -> 暗部与高光可以各自重分布（**阴影可调**）；
      * 同一输入必得同一输出 -> **逐帧稳定，不会闪烁**；
      * 拟合完就是查表 -> 逐帧应用极快。

    做法：
      1) 先用 color_transfer_sot 求出**像素级对应**的目标 y（SOT 是分位数传输，有界、非线性）
      2) 在 size^3 网格上用三线性插值建立稀疏设计矩阵，最小二乘拟合 lut 使 lut(src) ≈ y
         正则项：拉普拉斯平滑（保证曲线光滑）+ 恒等先验（不过度偏离原图）
      3) 三线性查表输出

    src / trg : HWC float32 [0,1]（BGR）
    size      : LUT 分辨率（17 是 .cube 常见规格；越大越精细，代价是拟合稍慢）
    """
    import scipy.sparse.linalg as spla

    if src.shape != trg.shape:
        raise ValueError("src and trg must have the same shape")
    if src.shape[2] != 3:
        raise ValueError("only 3-channel images are supported")

    # ---- 1) SOT 求像素级对应 ----
    # SOT 内部用 np.random 抽投影方向，这里临时播种，保证**同一输入必得同一结果**
    # （确定性正是 LUT 相对 SOT 的核心优势之一，不能因为内部调 SOT 就丢掉）。
    _rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        y = color_transfer_sot(src, trg, steps=sot_steps, batch_size=sot_batch)
    finally:
        np.random.set_state(_rng_state)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)

    x = src.reshape(-1, 3).astype(np.float32)
    yy = y.reshape(-1, 3).astype(np.float32)

    # 抽样（按亮度分层：暗部也要有样本，否则阴影会被平滑先验压平）
    n = len(x)
    if n > n_samples:
        rng = np.random.RandomState(seed)
        lum = x @ _LUT_LUMA
        order = np.argsort(lum)
        # 均匀抽，保证整条亮度轴都有覆盖
        sel = order[np.linspace(0, n - 1, n_samples).astype(np.int64)]
        sel = np.unique(np.concatenate([sel, rng.choice(n, n_samples // 4, replace=False)]))
        x_s, y_s = x[sel], yy[sel]
    else:
        x_s, y_s = x, yy

    Phi = _lut_build_basis(x_s, size)                      # (N, size^3)
    L = _lut_laplacian(size)                               # 平滑先验
    n_node = size ** 3

    # 恒等映射作为先验目标
    grid = np.stack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size),
                                np.linspace(0, 1, size), indexing='ij'), axis=-1)
    ident = grid.reshape(-1, 3).astype(np.float32)

    A = (Phi.T @ Phi).tocsr() + lam_smooth * L + lam_ident * n_node *         __import__('scipy.sparse', fromlist=['eye']).eye(n_node, format='csr', dtype=np.float32)
    B = np.asarray(Phi.T @ y_s) + lam_ident * n_node * ident

    lut = np.zeros((n_node, 3), dtype=np.float32)
    for ch in range(3):
        sol, info = spla.cg(A, B[:, ch], rtol=1e-5, maxiter=500)
        lut[:, ch] = sol
    lut = np.clip(lut, 0.0, 1.0).reshape(size, size, size, 3)

    return _lut_apply(src, lut, size)


def match_tone_np(src, trg, src_mask=None, trg_mask=None):
    """遮罩加权的逐通道均值/标准差匹配（**RGB/原生通道空间，不转 Lab**）。

    对应「截图那一版 match_tone」：
        mean = Σ(x·m) / Σm
        var  = Σ((x-mean)^2·m) / Σm
        out  = (x - mean_src) * (std_trg/std_src) + mean_trg     再 clamp 到 [0,1]

    与 reinhard_color_transfer 的两点差别：
      1) 不转 Lab —— 直接在 BGR 里逐通道缩放，更快，但会连带影响色相/饱和度；
      2) 遮罩是**软加权**（按 mask 的数值加权求均值/方差），而不是按 mask_cutoff 阈值截断。
         羽化过的遮罩因此能被正确利用。
    """
    src = np.ascontiguousarray(np.asarray(src, np.float32))
    trg = np.ascontiguousarray(np.asarray(trg, np.float32))

    def _stats(x, m):
        flat = x.reshape(-1, x.shape[2])
        if m is None:
            return flat.mean(axis=0), flat.std(axis=0)
        w = np.asarray(m, np.float32)
        if w.ndim == 2:
            w = w[..., None]
        if w.shape[2] == 1:
            w = np.repeat(w, x.shape[2], axis=2)
        w = w.reshape(-1, x.shape[2])
        ws = w.sum(axis=0) + 1e-8
        mean = (flat * w).sum(axis=0) / ws
        var = ((flat - mean) ** 2 * w).sum(axis=0) / ws
        return mean, np.sqrt(var + 1e-6)

    s_mean, s_std = _stats(src, src_mask)
    t_mean, t_std = _stats(trg, trg_mask)
    scale = t_std / (s_std + 1e-6)
    out = (src - s_mean.reshape(1, 1, -1)) * scale.reshape(1, 1, -1) + t_mean.reshape(1, 1, -1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
