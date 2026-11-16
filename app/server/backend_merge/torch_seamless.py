"""GPU (torch) re-implementation of cv2.seamlessClone(NORMAL_CLONE).

Mirrors opencv/modules/photo/src/seamless_cloning_impl.cpp exactly:

  grad = forward difference (reflect101), divergence = backward difference
  mask erosion = 3 iterations of 3x3 rect (== 7x7 rect)
  composite field v = grad(P)*mask + grad(D)*(1-mask)
  rhs = div(v) - Laplacian(D * ring)        (ring = 1px border of the ROI)
  solve Delta f = rhs , f = 0 on the ROI ring   -> truncated/clamped into dst ROI

Solvers (selectable):
  'dst'  : exact DST-I diagonalisation (sine matrices, matmul)  == OpenCV's maths
  'fft'  : same operator built from rfft of the odd extension (O(N log N))
  'cg'   : conjugate gradient with Jacobi preconditioner, tol/iter controlled
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- primitives
def _fwd_x(a):
    """computeGradientX: kernel [0,-1,+1] -> a(j+1)-a(j), BORDER_REFLECT_101."""
    return torch.cat([a[..., :, 1:], a[..., :, -2:-1]], dim=-1) - a


def _fwd_y(a):
    return torch.cat([a[..., 1:, :], a[..., -2:-1, :]], dim=-2) - a


def _bwd_x(a):
    """computeLaplacianX: kernel [-1,+1] -> a(j)-a(j-1), BORDER_REFLECT_101."""
    return a - torch.cat([a[..., :, 1:2], a[..., :, :-1]], dim=-1)


def _bwd_y(a):
    return a - torch.cat([a[..., 1:2, :], a[..., :-1, :]], dim=-2)


def lap5(a):
    """cv2.Laplacian(ksize=1), BORDER_REFLECT_101."""
    p = F.pad(a, (1, 1, 1, 1), mode="reflect")
    return (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] +
            p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4.0 * a)


def erode_rect3x3_iter(mask, iterations=3):
    """cv2.erode(mask, ones(3,3), iterations=3); border treats outside as +inf."""
    m = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    for _ in range(iterations - 1):
        m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return m


def dst_matrices(n, device, dtype):
    k = torch.arange(1, n + 1, device=device, dtype=dtype)
    return torch.sin(torch.pi * k[:, None] * k[None, :] / (n + 1))


def _dst2(x, Sh, Sw):
    """2D DST-I:  Sh x Sw  (contracts rows with Sh, cols with Sw)."""
    b, c, h, w = x.shape
    t = torch.matmul(Sh, x.reshape(b, c, h, w)).reshape(b * c, h, w)
    t = torch.matmul(t, Sw.t())
    return t.reshape(b, c, h, w)


def _idst2(x, Sh, Sw):
    b, c, h, w = x.shape
    t = torch.matmul(Sh.t(), x).reshape(b * c, h, w)
    t = torch.matmul(t, Sw)
    return t.reshape(b, c, h, w) * (4.0 / ((h + 1) * (w + 1)))


def dst_solve(rhs, Sh=None, Sw=None):
    b, c, h, w = rhs.shape
    dev, dt = rhs.device, rhs.dtype
    if Sh is None:
        Sh = dst_matrices(h, dev, dt)
        Sw = dst_matrices(w, dev, dt)
    lam_y = 2.0 * torch.cos(torch.pi * torch.arange(1, h + 1, device=dev, dtype=dt) / (h + 1))
    lam_x = 2.0 * torch.cos(torch.pi * torch.arange(1, w + 1, device=dev, dtype=dt) / (w + 1))
    lam = (lam_y[:, None] + lam_x[None, :] - 4.0)[None, None]
    return _idst2(_dst2(rhs, Sh, Sw) / lam, Sh, Sw)


def _dst1_fft(x, dim):
    """DST-I along one axis via rfft of the odd extension [0, x, 0, -flip(x)]."""
    n = x.shape[dim]
    zero_shape = list(x.shape)
    zero_shape[dim] = 1
    zero = torch.zeros(zero_shape, device=x.device, dtype=x.dtype)
    seq = torch.cat([zero, x, zero, -torch.flip(x, [dim])], dim=dim)
    spec = torch.fft.rfft(seq, dim=dim)
    sl = [slice(None)] * spec.ndim
    sl[dim] = slice(1, n + 1)
    return torch.imag(spec[tuple(sl)]).mul(-0.5)


def dst_solve_fft(rhs):
    """Same operator as dst_solve, built from FFTs (O(N log N)) instead of matmuls."""
    b, c, h, w = rhs.shape
    dev, dt = rhs.device, rhs.dtype
    f = _dst1_fft(_dst1_fft(rhs, -1), -2)
    lam_y = 2.0 * torch.cos(torch.pi * torch.arange(1, h + 1, device=dev, dtype=dt) / (h + 1))
    lam_x = 2.0 * torch.cos(torch.pi * torch.arange(1, w + 1, device=dev, dtype=dt) / (w + 1))
    f = f / (lam_y[:, None] + lam_x[None, :] - 4.0)[None, None]
    # inverse DST-I == 2/(n+1) * DST-I  (sine matrix is symmetric)
    sol = _dst1_fft(_dst1_fft(f, -2), -1)
    return sol * (4.0 / ((h + 1) * (w + 1)))


def cg_solve(rhs, iters=200, tol=1e-6, x0=None, return_info=False):
    """Batched CG on the masked 5-point Laplacian with zero Dirichlet on the ROI ring.

    Unknowns = interior nodes (1..h-2, 1..w-2); RHS = rhs.
    """
    b, c, h, w = rhs.shape
    dev, dt = rhs.device, rhs.dtype

    def A(x):
        # x: interior only -> embed with zero ring, apply 5-point Laplacian, crop
        z = F.pad(x, (1, 1, 1, 1))
        return (z[..., :-2, 1:-1] + z[..., 2:, 1:-1] +
                z[..., 1:-1, :-2] + z[..., 1:-1, 2:] - 4.0 * x)

    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - A(x)
    # Jacobi preconditioner: diagonal = -4
    z = r / (-4.0)
    p = z.clone()
    rz = (r * z).sum(dim=(-1, -2), keepdim=True)
    bnorm = rhs.pow(2).sum(dim=(-1, -2), keepdim=True).clamp_min(1e-30)
    hist = []
    for it in range(iters):
        Ap = A(p)
        alpha = rz / (p * Ap).sum(dim=(-1, -2), keepdim=True).clamp_max(-1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        res = (r.pow(2).sum(dim=(-1, -2), keepdim=True) / bnorm).sqrt()
        if return_info:
            hist.append(res.max().item())
        if res.max().item() < tol:
            break
        z = r / (-4.0)
        rz_new = (r * z).sum(dim=(-1, -2), keepdim=True)
        p = z + (rz_new / rz.clamp_max(-1e-30)) * p
        rz = rz_new
    if return_info:
        return x, it + 1, hist
    return x


# ---------------------------------------------------------------- pipeline
def poisson_blend_roi(D, P, M, mode="dst", cg_iters=200, cg_tol=1e-6, return_info=False):
    """D,P: [B,3,h,w] float (D = destination ROI, P = source patch already masked),
    M: [B,1,h,w] float in 0..255 (binary mask).

    Returns the ROI result as uint8-valued float [B,3,h,w] (exactly what
    Cloning::solve writes) plus optional solver info.
    """
    me = erode_rect3x3_iter(M, 3) / 255.0
    gx = _fwd_x(D) * (1.0 - me) + _fwd_x(P) * me
    gy = _fwd_y(D) * (1.0 - me) + _fwd_y(P) * me
    lap = _bwd_x(gx) + _bwd_y(gy)
    ring = torch.zeros_like(M)
    ring[..., 0, :] = 1.0
    ring[..., -1, :] = 1.0
    ring[..., :, 0] = 1.0
    ring[..., :, -1] = 1.0
    rhs = (lap - lap5(D * ring))[..., 1:-1, 1:-1]

    info = None
    if mode == "dst":
        sol = dst_solve(rhs)
    elif mode == "fft":
        sol = dst_solve_fft(rhs)
    elif mode == "cg":
        sol, nit, hist = cg_solve(rhs, iters=cg_iters, tol=cg_tol, return_info=True)
        info = {"iters": nit, "hist": hist}
    else:
        raise ValueError(mode)

    out = D.clone()
    out[..., 1:-1, 1:-1] = torch.clamp(sol, 0.0, 255.0).trunc()
    if return_info:
        return out, info
    return out


def seamless_roi_from_frames(dst, src, mask, center, mode="dst", **kw):
    """Full cv2.seamlessClone semantics on batches of *full* frames.

    dst, src: [B,3,H,W] uint8 ; mask: [B,H,W] uint8 ; center: list of (x,y).
    Returns blended [B,3,H,W] uint8 and a dict of per-sample ROI rects.
    """
    B = dst.shape[0]
    outs = dst.clone()
    rects = []
    for b in range(B):
        m = mask[b].clone()
        # seamless_cloning.cpp:79-84  -> zero the 1px border, then bbox
        mm = m.clone()
        mm[0, :] = 0; mm[-1, :] = 0; mm[:, 0] = 0; mm[:, -1] = 0
        ys, xs = torch.nonzero(mm, as_tuple=True)
        if xs.numel() == 0:
            rects.append(None)
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        rw, rh = x1 - x0 + 1, y1 - y0 + 1
        l = center[b][0] - rw // 2
        t = center[b][1] - rh // 2
        D = dst[b:b + 1, :, t:t + rh, l:l + rw].float()
        S = src[b:b + 1, :, y0:y0 + rh, x0:x0 + rw].float()
        Mb = (m[y0:y0 + rh, x0:x0 + rw] > 0).float() * 255.0
        Pm = S * (Mb > 0).float() / 1.0
        out = poisson_blend_roi(D, Pm, Mb.unsqueeze(0), mode=mode, **kw)
        outs[b:b + 1, :, t:t + rh, l:l + rw] = out.round().clamp(0, 255).to(torch.uint8)
        rects.append((l, t, rw, rh))
    return outs, rects
