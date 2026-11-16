# -*- coding: utf-8 -*-
"""合成搬到 GPU：用 torch 逐段复刻 backend_merge.compose 的 ROI 流水线（**不含 seamless**）。

为什么是「逐段复刻」而不是「另写一版」：compose_frame() 的每一个数值选择都是一条验收过的
口径 —— ROI 拆法、cv2 的 warp 语义、ct 的统计量范围、融合后 u8 的截断方式。GPU 版只换
「在哪算」，不换「算什么」，所以每一段都能指回 CPU 那一行。

## 复刻 cv2 的两处关键（都是量出来的，不是猜的）

**① warpAffine(INTER_CUBIC) 在 float32 上不做 1/32 定点。**
探针（单点 delta 图 + 亚像素平移，见 app/tools/accept/compose_gpu_accept.py 的 cubic 项）：

    dx        观测[48,48]    连续精确值     1/32 定点预测
    0.00391   0.99996579     0.99996574    0.0            <- 只有连续值解释得通
    0.03125   0.99784088     0.99784088    0.99784088     <- 正好落在 1/32 上时才重合
    0.04375   0.99579793     0.99579803    0.99784088
    k/32 全扫： max|观测-连续| = 0.0   max|观测-1/32定点| = 1.35e-2

只有 **uint8** 输入才走定点（同一个探针：dx=0.0130 -> 255、dx=0.04375 -> 254）。
我们这条链上三处 warp 的输入全是 float32（roi/255、wrk、fo），所以这里对齐的是
「连续坐标 + A=-0.75 三次核 + 零边界」；真按 1/32 量化反而会差 1.35e-2 约 3.4 LSB。

**② cv2.resize(INTER_CUBIC) 在 float32 上同样是连续坐标。**
256x256 -> 416x416 白噪声：|cv2 - 连续精确参考| = 2.6e-7；|cv2 - 11bit 定点| = 9.9e-4；
|cv2 - 5bit 定点| = 5.9e-2。所以遮罩的 256->416 也用连续坐标 + 4 tap + 边缘钳位。

## 数据流（「只上传人脸框」）

    numpy ROI(u8)   --上传-->  src_f(3,rh,rw) --+
    pred / masks    --上传-->                   +--> 全在 GPU 上：warp / ct / 反贴 / 融合
                                                +
    合成后的框(u8)  <--下载--    （整帧不往返）

一帧多张脸时按 CPU 的语义来：**先合并成一张 float 的 work，最后只回一次 u8**
（compose_frame 的 len(layers)>1 分支）。所以 GPU 侧用「所有脸的 ROI 并集」当画布：
上传并集一次、下载并集一次，脸之间重叠处的先后顺序与 CPU 完全一致。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

# 只有这两类 ct 在 GPU 上复刻了；其它（rct/mkl/idt/sot/mix/lut）按「逐脸回退」处理：
# warp 仍在 GPU 上做，ct 那一步把 dw 拉回 CPU 调原实现，再把结果推上去。
_GPU_CT = (0, 9)
# 遮罩准备里 GPU 能复刻的 mask_mode（1 = dst hull 要栅格化 + uint8 定点 warp，回退 CPU；
# erode != 0 要椭圆结构元的形态学，也回退 CPU）。
_GPU_MASK_MODES = (0, 2, 3, 4, 5, 6, 7, 8, 9)

_ENV = "DFL_COMPOSE_GPU"


def env_on() -> bool:
    """DFL_COMPOSE_GPU 开关（默认关；1/on/true/yes 才开）。"""
    v = str(os.environ.get(_ENV) or "").strip().lower()
    return v not in ("", "0", "off", "false", "no", "none")


# ============================================================
# 设备上下文（懒加载 + 复用）
# ============================================================
class _Ctx:
    """一个进程一份：torch 模块 / device / 1-D 坐标缓存 / 统计。"""

    def __init__(self, device=None):
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.cuda = self.dev.type == "cuda"
        self.axes = {}
        self.stats = {"frames": 0, "faces": 0, "fallback_ct": 0, "fallback_mask": 0}

    def axis(self, ow, oh):
        """1-D 目标坐标（float64，按 (w,h) 缓存；两个向量，几十字节）。"""
        t = self.torch
        key = (int(ow), int(oh))
        v = self.axes.get(key)
        if v is None:
            v = (t.arange(int(ow), dtype=t.float64, device=self.dev),
                 t.arange(int(oh), dtype=t.float64, device=self.dev))
            self.axes[key] = v
        return v


_CTX = {"obj": None, "err": "", "lock": threading.RLock()}


def ctx(device=None):
    """拿（或建）本进程的 GPU 上下文；起不来就抛，调用方负责回退 CPU。"""
    with _CTX["lock"]:
        obj = _CTX["obj"]
        if obj is None:
            obj = _Ctx(device)
            _CTX["obj"] = obj
        return obj


def available() -> bool:
    """能不能用 GPU 合成（只查 torch + cuda，不碰显存账）。"""
    try:
        c = ctx()
        return bool(c.cuda)
    except Exception as e:                                     # noqa: BLE001
        _CTX["err"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        return False


def ready() -> bool:
    """能不能**真的跑起来**：除了 context，还真跑一次 8x8 的 warp + 同步。

    「torch 装了、cuda.is_available() 也是 True」不代表这条路上每个 kernel 都能跑
    （驱动版本 / 显存 / 编译缓存都可能在这里第一次炸）。在流水线开跑前用一次
    微型的真实调用把它试出来，比跑到第 300 帧才崩好。
    """
    try:
        c = ctx()
        if not c.cuda:
            _CTX["err"] = "torch.cuda 不可用（%s）" % c.dev
            return False
        src = c.torch.zeros((1, 8, 8), dtype=c.torch.float32, device=c.dev)
        out = warp_cubic(c, src, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 8, 8)
        c.torch.cuda.synchronize()
        if tuple(out.shape) != (1, 8, 8):
            _CTX["err"] = "试算形状不对：%r" % (tuple(out.shape),)
            return False
        return True
    except Exception as e:                                     # noqa: BLE001
        _CTX["err"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        return False


def last_error() -> str:
    return _CTX["err"]


# ============================================================
# 基元 ①：warpAffine(INTER_CUBIC) 的 torch 版
# ============================================================
def _sample_coords(c, m, out_w, out_h):
    """dst 像素中心 -> src 坐标（float64）。m 是**采样矩阵**（dst->src）的 2x3。

    坐标在 float64 里算（cv2 内部就是 double），最后一步才降到 float32 交给
    grid_sample —— 归一化那一下的舍入约 1e-5 px，对 416 画布是 0.005 LSB 量级。
    """
    xs, ys = c.axis(out_w, out_h)
    m = np.asarray(m, np.float64)
    m0, m1, m2, m3, m4, m5 = (float(v) for v in m.reshape(-1)[:6])
    sx = xs[None, :] * m0 + ys[:, None] * m1 + m2
    sy = xs[None, :] * m3 + ys[:, None] * m4 + m5
    return sx, sy


def warp_cubic(c, src, m, out_w, out_h):
    """src:(C,H,W) float32 张量 · m: dst->src 2x3（和 cv2 的 WARP_INVERSE_MAP 同一个语义）。

    grid_sample(mode='bicubic') 用的就是 A=-0.75 的三次卷积、align_corners=True 时像素中心
    落在整数坐标上、padding_mode='zeros' 时界外按 0 计入且**不重新归一化** —— 和 cv2 的
    remapBicubic(BORDER_CONSTANT, 0) 是同一套定义。
    """
    F = c.F
    ch, h, w = src.shape
    sx, sy = _sample_coords(c, m, out_w, out_h)
    gx = (sx * (2.0 / max(1, w - 1)) - 1.0).to(c.torch.float32)
    gy = (sy * (2.0 / max(1, h - 1)) - 1.0).to(c.torch.float32)
    grid = c.torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(src[None], grid, mode="bicubic",
                        padding_mode="zeros", align_corners=True)
    return out[0]


def warp_cubic_gather(c, src, m, out_w, out_h):
    """同一个 warp 的「逐 tap 直算」版：坐标与权重全程 float64，只有乘加是 float32。

    grid_sample 那条路的误差上限来自「归一化坐标要降到 float32 再还原」：|gx|<=1 的
    float32 舍入 6e-8，还原成像素是 6e-8*(W-1)/2 ≈ 1.2e-5 px，再乘三次核的斜率
    （<=1.5）就是 2e-5 的值误差（实测 3.7e-5）。这一版把这一步省掉（坐标不进 grid），
    权重按 cv2 的 interpolateCubic 在 float64 里算完再降 float32，误差回到 1e-7 量级
    —— 代价是 16 次 gather（见 accept 里的耗时对照，416 画布约 +0.15ms/张脸）。
    """
    t = c.torch
    ch, h, w = src.shape
    sx, sy = _sample_coords(c, m, out_w, out_h)
    fxi = t.floor(sx)
    fyi = t.floor(sy)
    # 小数部分降到 float32 再算权重 —— cv2 的 interpolateCubic 就是 float 里算的
    wx = _cubic_w((sx - fxi).to(t.float32))
    wy = _cubic_w((sy - fyi).to(t.float32))
    ix = fxi.to(t.int64)
    iy = fyi.to(t.int64)
    # 源四周补 2 圈 0（正好是三次核的半径）：界外只要把 tap 钳到补出来的那一圈
    # （值恒为 0）就等价于 cv2 的「界外按 0 计、不重新归一化」，不必逐 tap 建掩码。
    # 实测：逐 tap 掩码 16 组 int64 运算 3.4ms -> 这里 8 次钳位 + 16 次加。
    srcp = t.nn.functional.pad(src, (2, 2, 2, 2), mode="constant", value=0.0)
    wpad = w + 4
    flat = srcp.reshape(ch, (h + 4) * wpad)
    rows = [(iy + (i - 1)).clamp_(-2, h + 1).add_(2).mul_(wpad) for i in range(4)]
    cols = [(ix + (j - 1)).clamp_(-2, w + 1).add_(2) for j in range(4)]
    out = None
    for i in range(4):
        for j in range(4):
            idx = (rows[i] + cols[j]).reshape(-1)
            v = flat.index_select(1, idx).reshape(ch, out_h, out_w)
            term = v * (wy[i] * wx[j])
            out = term if out is None else out + term
    return out


_WARP = {"impl": ""}


def warp_impl() -> str:
    """DFL_COMPOSE_GPU_WARP=grid|gather，默认 grid。

    grid（grid_sample 一条 fused kernel）实测 0.13~0.33 ms/次，gather（16 次 index_select
    + float64 坐标/权重）2.7~4.3 ms/次 —— 后者把「离精确 float64 参考」的距离从
    1.9e-5 压到 1.1e-6，但 cv2 自己离精确参考就有 2.7e-5（见 accept 的 prims 段），
    所以那点精度在「与 cv2 对齐」这件事上收不回来，默认选快的。
    """
    v = _WARP.get("impl")
    if v:
        return v
    raw = str(os.environ.get("DFL_COMPOSE_GPU_WARP") or "").strip().lower()
    v = raw if raw in ("grid", "gather") else "grid"
    _WARP["impl"] = v
    return v


def warp(c, src, m, out_w, out_h):
    if warp_impl() == "grid":
        return warp_cubic(c, src, m, out_w, out_h)
    return warp_cubic_gather(c, src, m, out_w, out_h)


# ============================================================
# 基元 ②：resize(INTER_CUBIC)（遮罩 256 -> S）
# ============================================================
def _cubic_w(f):
    """OpenCV interpolateCubic（A=-0.75）：4 个邻居的权重，邻居是 ix-1..ix+2。"""
    A = -0.75
    g = 1.0 - f
    c0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
    c1 = ((A + 2) * f - (A + 3)) * f * f + 1
    c2 = ((A + 2) * g - (A + 3)) * g * g + 1
    c3 = 1.0 - c0 - c1 - c2
    return c0, c1, c2, c3


_TAPS = {}


def _resize_taps(n_in, n_out, c):
    """cv2.resize 的取样点：fx = (dx+0.5)*scale - 0.5，四邻居在边界处**钳位**（replicate）。

    按 (n_in, n_out) 缓存 —— 它只跟尺寸有关。生产里就是 256->416 这一个组合，
    不缓存的话每次都要在 float64 里重算一遍三次核（实测 1.8ms/次，是 mask 段的大头）。
    """
    t = c.torch
    key = (int(n_in), int(n_out), str(c.dev))
    got = _TAPS.get(key)
    if got is not None:
        return got
    scale = float(n_in) / float(n_out)
    d = t.arange(n_out, dtype=t.float64, device=c.dev)
    f = (d + 0.5) * scale - 0.5
    i0 = t.floor(f)
    fr = f - i0
    i0 = i0.to(t.int64)
    idx = []
    for k in (-1, 0, 1, 2):
        idx.append((i0 + k).clamp_(0, n_in - 1))
    w0, w1, w2, w3 = _cubic_w(fr)
    got = (idx, (w0.to(t.float32), w1.to(t.float32), w2.to(t.float32), w3.to(t.float32)))
    _TAPS[key] = got
    return got


def resize_cubic(c, src, out_w, out_h):
    """src:(C,H,W) -> (C,oh,ow)，复刻 cv2.resize(INTER_CUBIC) 的 float32 路径。"""
    t = c.torch
    ch, h, w = src.shape
    if (h, w) == (int(out_h), int(out_w)):
        return src
    ix, wx = _resize_taps(w, int(out_w), c)
    iy, wy = _resize_taps(h, int(out_h), c)
    tmp = None
    for k in range(4):
        part = src.index_select(2, ix[k]) * wx[k].view(1, 1, -1)
        tmp = part if tmp is None else tmp + part
    out = None
    for k in range(4):
        part = tmp.index_select(1, iy[k]) * wy[k].view(1, -1, 1)
        out = part if out is None else out + part
    return out


# ============================================================
# 基元 ③：GaussianBlur（遮罩的 blur 修饰）
# ============================================================
_GAUSS = {}


def gauss_kernel(ksize, sigma=0.0):
    """直接问 cv2 要核 —— sigma=0 时 OpenCV 会用**硬编码小核表**（这是版本相关的：

    cv2 5.0 里 ksize<=9 全部查表，实测 k=3 [1,2,1]/4、k=5 [1,4,6,4,1]/16、
    k=7 [2,7,14,18,14,7,2]/64、k=9 [1,3.25,7.5,12.75,15,...]/64；
    k>=11 才走 exp 公式（k=11 与公式差 1.4e-17）。自己按 0.3*((k-1)/2-1)+0.8 推
    在 k=5 上会差 2.7e-1（核权重本身），落到遮罩上就是 1.4e-3 的误差、1 LSB 的
    差异像素 —— 所以这里不自己算，直接取 cv2 的那一份。）
    """
    key = (int(ksize), float(sigma))
    k = _GAUSS.get(key)
    if k is not None:
        return k
    import cv2
    kk = np.asarray(cv2.getGaussianKernel(int(ksize), float(sigma), cv2.CV_32F),
                    np.float64).ravel()
    _GAUSS[key] = kk
    return kk


def _gauss_t(k, c):
    return c.torch.as_tensor(np.ascontiguousarray(k, np.float32), device=c.dev)


def gauss_blur(c, x, ksize, sigma=0.0, border="reflect"):
    """x:(C,H,W) 的可分离高斯。

    border="reflect" 复刻 cv2.GaussianBlur 的默认 BORDER_REFLECT_101（torch 的
    padding mode 就叫 reflect，定义一致：fedcba|abcdefgh|hgfedcb）。
    border="zero" 是「先 np.pad(wrk, 32) 再 blur」的等价写法 —— 遮罩那条路用的就是它
    （pad 出来的 32 圈 0 让反射边界看到的东西全是 0，核半径 <=32 时两者逐位相同）。
    """
    t = c.torch
    n = int(ksize)
    if n <= 1:
        return x
    k = _gauss_t(gauss_kernel(n, sigma), c)
    r = n // 2
    ch, h, w = x.shape
    mode = "reflect" if str(border) == "reflect" else "constant"
    pad = t.nn.functional.pad(x, (r, r, 0, 0), mode=mode, value=0.0)
    acc = None
    for i in range(n):
        part = pad[:, :, i:i + w] * k[i]
        acc = part if acc is None else acc + part
    pad2 = t.nn.functional.pad(acc, (0, 0, r, r), mode=mode, value=0.0)
    acc = None
    for i in range(n):
        part = pad2[:, i:i + h, :] * k[i]
        acc = part if acc is None else acc + part
    return acc


# ============================================================
# 基元 ④：色彩迁移（mt = match_tone）与直方图匹配
# ============================================================
def _mask_bbox_t(c, mask):
    """遮罩非零的包围盒（GPU 上算，一次同步）。返回 (y0,y1,x0,x1) 或 None。

    为什么非要这个盒子：CPU 的 _color_transfer(code 9) 是
        out = fo.copy(); out[bb] = clip(match_tone_roi(fo[bb], dw[bb], wrk[bb]))
    —— **盒子外一个像素都不动**。数学上「盒外权重 0」对统计量无所谓，但对**写回**有所谓：
    盒外按整画布算出来的值会被真的写进 fo。overlay 模式下那些像素乘的是 mw=0 看不出来，
    可 hist-match 的映射表是拿整张画布的分布算的 —— 盒外一动，整张脸的映射就变了
    （实测 max 42 LSB）。所以这里必须和 CPU 一样只写盒内。
    """
    t = c.torch
    m = mask[0] > 0
    rows = m.any(dim=1)
    cols = m.any(dim=0)
    if not bool(rows.any()):
        return None
    hh = int(rows.numel())
    ww = int(cols.numel())
    box = t.stack([t.argmax(rows.to(t.uint8)),
                   hh - 1 - t.argmax(rows.flip(0).to(t.uint8)),
                   t.argmax(cols.to(t.uint8)),
                   ww - 1 - t.argmax(cols.flip(0).to(t.uint8))]).cpu().tolist()
    return int(box[0]), int(box[1]) + 1, int(box[2]), int(box[3]) + 1


def mt_transfer(c, fo, dw, mask):
    """match_tone（ct=mt）的 GPU 版：统计量与写回都限制在遮罩包围盒里，和 CPU 逐行对应。"""
    t = c.torch
    ch = fo.shape[0]
    bb = _mask_bbox_t(c, mask)
    if bb is None:
        return fo
    y0, y1, x0, x1 = bb
    f = fo[:, y0:y1, x0:x1]
    g = dw[:, y0:y1, x0:x1]
    mk = mask[:, y0:y1, x0:x1]

    flat = f.permute(1, 2, 0).reshape(-1, ch)
    wf = mk.reshape(-1, 1).expand(-1, ch)
    ws = wf.sum(dim=0) + 1e-8
    s_mean = (flat * wf).sum(dim=0) / ws
    d = flat - s_mean
    d = d * d
    d = d * wf
    s_std = t.sqrt(d.sum(dim=0) / ws + 1e-6)

    tflat = g.permute(1, 2, 0).reshape(-1, ch)
    t_mean = (tflat * wf).sum(dim=0) / ws
    d2 = tflat - t_mean
    d2 = d2 * d2
    d2 = d2 * wf
    t_std = t.sqrt(d2.sum(dim=0) / ws + 1e-6)

    scale = t_std / (s_std + 1e-6)
    out = f.clone()
    out -= s_mean.view(ch, 1, 1)
    out *= scale.view(ch, 1, 1)
    out += t_mean.view(ch, 1, 1)
    res = fo.clone()
    res[:, y0:y1, x0:x1] = out.clamp_(0.0, 1.0)
    return res


def hist_match(c, src, tpl, thr=255):
    """color_hist_match 的逐通道版：np.unique + cumsum + np.interp + gather。"""
    t = c.torch
    ch = src.shape[0]
    outs = []
    for i in range(ch):
        s = src[i].reshape(-1)
        p = tpl[i].reshape(-1)
        sv, sc = t.unique(s, return_counts=True)
        tv, tc = t.unique(p, return_counts=True)
        if tv.numel() < 2 or sv.numel() < 1:
            # 退化：整条通道恒等于一个值（pred 被 clip 成全 0/全 1 时会这样）。
            # np.interp 在 xp 只有一个点时返回那个点；这里照做 —— 不照做会读到 tq[1] 越界。
            outs.append(t.full_like(src[i], float(tv[0].item())))
            continue
        sq = t.cumsum(sc.to(t.float64), 0)
        sq = float(thr) * sq / sq[-1]
        tq = t.cumsum(tc.to(t.float64), 0)
        tq = 255.0 * tq / tq[-1]
        # np.interp(sq, tq, tv)：两端钳位 + 中间线性
        j = (t.searchsorted(tq, sq, right=True) - 1).clamp_(0, max(0, tq.numel() - 2))
        x0 = tq[j]
        x1 = tq[j + 1]
        y0 = tv[j].to(t.float64)
        y1 = tv[j + 1].to(t.float64)
        slope = (y1 - y0) / (x1 - x0)
        interp = y0 + (sq - x0) * slope
        interp = t.where(sq <= tq[0], tv[0].to(t.float64), interp)
        interp = t.where(sq >= tq[-1], tv[-1].to(t.float64), interp)
        idx = t.searchsorted(sv, s.contiguous()).clamp_(0, sv.numel() - 1)
        outs.append(interp[idx].to(t.float32).view_as(src[i]))
    return t.stack(outs, dim=0)


# ============================================================
# seamless（cv2.seamlessClone 的 torch 版）—— 同目录的 torch_seamless.py
# ============================================================
_SEAM = {"mod": None, "err": "", "lock": threading.Lock()}


def seamless_mod():
    """按路径加载同目录的 torch_seamless.py（独立验收过的实现，含 DST/FFT/CG 三种解法）。

    以前放在 experiments/seamless/ 下（"还在演进，复制一份进 app/ 会分叉"，所以按路径加载）；
    实验目录清掉之后只剩这一份，收进 backend_merge/ 与调用点同目录，老路径留作兜底候选。
    加载失败不抛 —— 调用方按「这一脸回退 CPU seamless」处理。
    """
    with _SEAM["lock"]:            # 合成可能有多条线程（DFL_COMPOSE_GPU_THREADS）
        if _SEAM["mod"] is not None or _SEAM["err"]:
            return _SEAM["mod"]
        try:
            import importlib.util
            here = Path(__file__).resolve().parent
            root = Path(__file__).resolve().parents[3]
            cands = [here / "torch_seamless.py",                      # 现在的位置
                     root / "experiments" / "seamless" / "torch_seamless.py"]   # 老位置（已清）
            for path in cands:
                if not path.is_file():
                    continue
                spec = importlib.util.spec_from_file_location("_dfl_torch_seamless", str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _SEAM["mod"] = mod
                break
            else:
                raise FileNotFoundError("torch_seamless.py not found: %s" % (cands,))
        except Exception as e:                                 # noqa: BLE001
            _SEAM["err"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        return _SEAM["mod"]


def seamless_on() -> bool:
    """DFL_COMPOSE_GPU_SEAMLESS：**默认 1（seamless 也走 GPU，跟随合成总开关）**。

    历史：一开始默认 0，理由是 150 帧同段实测 seamless 走 GPU 会倒退 13.6%
    （seamless CPU 21.48 帧/秒 · 8.95 核 · det_wait 3%  vs  seamless GPU 18.55 · 14.12 核 · det_wait 43%），
    因为 Poisson 解（DST matmul）和 detect 抢同一块卡。所以曾经「选了 gpu 也会退回 cpu + 弹告警」。

    现在改成默认**允许**：让用户在界面上选 gpu + seamless 就真的跑 GPU（他们要看的是结果与手感，
    而不是我们替他们决定）。代价写在这里、也写进 docs/26：
      · 合成段本身快一倍多（107.87 → 47.62ms），但 detect 的 det_wait 会从 3% 涨到 ~43%，
        整条线在「1080p 单脸 + 我们这套流水线」下可能倒退十几个百分点；
      · GPU seamless 与 cv2 不是逐位一致（97.14% 逐位相同、2.53% 差 1 LSB、0.0038% 在 ROI 外框 ≤19px
        的窄带里 >16 LSB，见 docs/26）——画质取舍由用户自己判断。
    想要旧行为：DFL_COMPOSE_GPU_SEAMLESS=0（一行回退）。
    """
    v = str(os.environ.get("DFL_COMPOSE_GPU_SEAMLESS") or "").strip().lower()
    if v in ("", "auto"):
        return True
    return v not in ("0", "off", "false", "no", "none")


def _env_flag(name):
    return str(os.environ.get(name) or "").strip().lower() not in ("", "0", "off", "false", "no")


def seamless_merge(c, src_f, fw, mw, fo=None, fwd=None, wrk=None):
    """复刻 _blend_layer 的 seamless 分支（src_f / fw / mw 是同一块 ROI 的 float 张量）。

    src_f:(3,h,w) 当前画面（多脸时是累加中的 work）· fw:(3,h,w) 反贴回来的脸 ·
    mw:(1,h,w) 反贴回来的遮罩。返回 (3,h,w) float，语义与 CPU 的 out 相同。

    CPU 那边三步：mw/fw/dst 各自 clip(*255) 截断成 u8、按 u8 遮罩的包围盒定 center、
    seamlessClone 之后**只在 float 遮罩的包围盒里**按 float 遮罩叠回去。包围盒只是
    省时间（盒外 float 遮罩恒 0，叠上去等于没动），这里整块叠，数值等价。
    """
    t = c.torch
    import cv2
    # **遮罩的 u8 化必须在 CPU 上用和 CPU 版一模一样的表达式做**：
    #   mw_bin = np.clip(mw*255, 0, 255).astype(uint8)
    # 因为 cv2.seamlessClone 的 boundingRect/center 是从**这张 u8 遮罩**来的，
    # 而 mw 在软边上有 1e-5 量级的差（cv2 warp 的 float32 噪声）—— 落到 u8 上，
    # 边缘那一圈「0 还是 1」会翻掉（实测 246/57121 个像素），boundingRect 就可能长 1 px，
    # center 跟着偏 1 px，整块 Poisson 解平移一格 → 局部差到 50 LSB。
    # 把这一步搬回 CPU 之后遮罩与 CPU 逐位相同，剩下的只有 fw/dst 的 1e-5 量级差。
    if _env_flag("DFL_COMPOSE_GPU_SEAMLESS_EXACT") and wrk is not None and fwd is not None:
        # 「逐位对齐」模式第一步：**遮罩也在 CPU 上用 cv2 反贴**。
        # 为什么：Poisson 解是积分，u8 遮罩边缘那圈「0 还是 1」（实测 246/57121 个像素，
        # 来自 cv2 warp 自己 1.4e-5 的 float32 噪声）会让腐蚀后的边界挪一格，梯度沿边界
        # 累加成一条几十 LSB 的窄带（实测 max 51）。把遮罩和源片都换成 CPU 的 cv2 结果，
        # 送进解算器的 u8 与 CPU 逐位相同 —— 剩下的差异就只剩解算器自己（那份实现验收过
        # 与 cv2 ≤1 LSB）。
        wrk_np = wrk.detach()[0].to("cpu").numpy()
        mw_cpu = cv2.warpAffine(wrk_np, np.asarray(fwd, np.float64),
                                (int(mw.shape[2]), int(mw.shape[1])),
                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        mw_np = np.clip(mw_cpu, 0.0, 1.0)
    else:
        mw_np = mw[0].detach().to("cpu").numpy()
    mw_bin_np = np.clip(mw_np * 255.0, 0, 255).astype(np.uint8)
    mw_bin = t.as_tensor(mw_bin_np, device=c.dev)
    if _env_flag("DFL_COMPOSE_GPU_SEAMLESS_EXACT") and fo is not None and fwd is not None:
        # 「逐位对齐」模式：源片这一步用 cv2 在 CPU 上做，u8 输入与 CPU 逐位相同。
        # 为什么非要这样：Poisson 解是**积分**——源片在 u8 上差 1（由 cv2 自己 2e-5 的
        # float32 warp 噪声引起，实测 119/57121 个像素）会顺着梯度累加成一条几十 LSB 的窄带
        # （实测 max 51）。所以「换一份三次插值实现」这件事在 seamless 上不是 1 LSB 的事。
        # 代价：每张脸一次 CPU 的 cv2.warpAffine（269x269x3 约 3ms）。
        fo_np = fo.detach().permute(1, 2, 0).to("cpu").numpy()
        fw_cpu = cv2.warpAffine(fo_np, np.asarray(fwd, np.float64), (int(mw.shape[2]), int(mw.shape[1])),
                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
        fw_u8 = t.as_tensor(np.clip(fw_cpu * 255, 0, 255).astype(np.uint8),
                            device=c.dev).permute(2, 0, 1)
    else:
        fw_u8 = (fw * 255.0).clamp_(0.0, 255.0).to(t.uint8)
    dst_u8 = (src_f * 255.0).clamp_(0.0, 255.0).to(t.uint8)
    mb = mw_bin_np
    nz = cv2.findNonZero(mb)
    if nz is None:
        # u8 遮罩整块为 0（只有 mw < 1/255 的软边）：沿用整块混合，和 CPU 一样
        return src_f * (1.0 - mw) + fw * mw
    bx, by, bw, bh = cv2.boundingRect(nz)
    center = (bx + bw // 2, by + bh // 2)
    mod = seamless_mod()
    if mod is None:
        raise RuntimeError("torch_seamless 加载不了：%s" % _SEAM["err"])
    h_before = None
    if _env_flag("DFL_COMPOSE_GPU_ASSERT"):
        # cv2.seamlessClone 会就地改写传进去的 mask（erode/not 作用在 view 上）。
        # 我们的调用点是每次都新建数组所以没事，这里把「传进去的 mask 没被改」钉死。
        h_before = hash(mb.tobytes())
    out_u8, _rects = mod.seamless_roi_from_frames(
        dst_u8[None], fw_u8[None], mw_bin[None], [center])
    if h_before is not None:
        want = hash(mw_bin.contiguous().cpu().numpy().tobytes())
        if want != h_before:
            raise RuntimeError("seamless 改写了传进去的遮罩（cv2 那个就地 bug 的翻版）")
    sc = out_u8[0].to(t.float32).div_(255.0)
    return src_f * (1.0 - mw) + sc * mw


# ============================================================
# 遮罩准备（_prepare_mask 的 GPU 版）
# ============================================================
def _mask_native(c, arr):
    """u8/float 的 (H,W) 或 (H,W,1) 遮罩 -> (1,H,W) float32（**原尺寸**，按 CPU 的口径 /255）。

    不在这里 resize：CPU 是「先按 mask_mode 逐张相乘，再把**乘积** resize 到 S」
    （export_pipeline 840-893）。三次插值是线性的，但 resize(a*b) != resize(a)*resize(b)，
    顺序错了遮罩边缘会差 1e-3 量级 —— 实测就是 2 LSB 的成片差。
    """
    t = c.torch
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == np.uint8:
        return t.as_tensor(a, device=c.dev).to(t.float32).div_(255.0)[None]
    return t.as_tensor(np.ascontiguousarray(a, np.float32), device=c.dev)[None]


def _to_canvas(c, arr, size):
    """(H,W)->(1,size,size)：先原样上传，再 resize（单独一张遮罩用）。"""
    x = _mask_native(c, arr)
    if x is None:
        return None
    if x.shape[-2:] != (int(size), int(size)):
        x = resize_cubic(c, x, int(size), int(size))
    return x


def mask_prep(c, face, cfg, size, mat=None, frame_shape=None):
    """按 mask_mode 组出工作遮罩（S×S，0..1），再做 blur。返回 (wrk(1,1,S,S), note)。"""
    t = c.torch
    from .compose import _prepare_mask
    mask_mode = int(cfg.get("mask_mode", 6) or 0)
    erode = int(cfg.get("erode_mask_modifier", cfg.get("mask_erode", 0)) or 0)
    blur = int(cfg.get("blur_mask_modifier", cfg.get("mask_blur", 0)) or 0)
    if mask_mode not in _GPU_MASK_MODES or erode != 0:
        # 回退：形态学/凸包没在 GPU 上复刻（椭圆结构元的 erode 要 k^2 次移位，而默认
        # erode=0，热路径碰不到）。结果直接推上去，后面照旧走 GPU。
        wrk = _prepare_mask(face, frame_shape, cfg, int(size), {}, fmat=mat)
        x = t.as_tensor(np.ascontiguousarray(wrk, np.float32), device=c.dev)[None]
        c.stats["fallback_mask"] += 1
        return x[None], "cpu"
    wrk = None
    need = {2: ("prd_dfm",), 3: ("dst_dfm",), 4: ("prd_dfm", "dst_dfm"),
            5: ("prd_dfm", "dst_dfm"), 6: ("xseg_prd",), 7: ("xseg_dst",),
            8: ("xseg_prd", "xseg_dst"),
            9: ("prd_dfm", "dst_dfm", "xseg_prd", "xseg_dst")}.get(mask_mode, ())
    parts = [(k, _mask_native(c, face.get(k))) for k in need]
    parts = [(k, v) for (k, v) in parts if v is not None]
    if parts:
        shapes = set(tuple(v.shape[-2:]) for _k, v in parts)
        if len(shapes) > 1:
            # 不同尺寸的遮罩逐张相乘（mask_mode 9 的混搭）在 CPU 上 numpy 会直接抛，
            # 这里也不替它猜 —— 原样回退 CPU，让行为与 CPU 完全一致。
            c.stats["fallback_mask"] += 1
            m0 = _prepare_mask(face, frame_shape, cfg, int(size), {}, fmat=mat)
            return t.as_tensor(np.ascontiguousarray(m0, np.float32), device=c.dev)[None, None], "cpu"
        wrk = parts[0][1]
        if mask_mode in (4, 5):
            dst = parts[1][1]
            wrk = wrk * dst if mask_mode == 4 else (wrk + dst).clamp_(0, 1)
        elif mask_mode in (8, 9):
            for _k, v in parts[1:]:
                wrk = wrk * v
        if wrk.shape[-2:] != (int(size), int(size)):
            wrk = resize_cubic(c, wrk, int(size), int(size))
    if wrk is None:
        wrk = t.ones((1, int(size), int(size)), dtype=t.float32, device=c.dev)
    if blur > 0:
        # CPU 是 np.pad(wrk, 32) 再 GaussianBlur —— 32 远大于核半径，等价于零边界。
        b = int(blur) + (1 - int(blur) % 2)
        wrk = gauss_blur(c, wrk, b, 0.0, border="zero")
    return wrk.clamp_(0.0, 1.0)[None], ""


def _as_chw(c, arr):
    """(H,W,3) u8|float32 -> (3,H,W) float32 on device（u8 在 GPU 上 /255）。"""
    t = c.torch
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        x = t.as_tensor(arr, device=c.dev).permute(2, 0, 1).contiguous()
        return x.to(t.float32).div_(255.0)
    if arr.ndim == 2:
        x = t.as_tensor(np.ascontiguousarray(arr), device=c.dev)
        return x.to(t.float32)[None]
    return t.as_tensor(np.ascontiguousarray(arr.transpose(2, 0, 1)), device=c.dev).to(t.float32)


# ============================================================
# 主入口
# ============================================================
def compose_frame_gpu(frame_bgr, faces, cfg, timings=None, roi=True, aligned=False):
    """一帧的 GPU 合成：语义与 compose.compose_frame / compose_aligned_frame 逐位对齐。

    frame_bgr : uint8 (H,W,3)
    faces     : 同 compose_frame（pred/prd_dfm/... 可以是 0..1 的 float32，也可以是 uint8
                —— 后者省掉 CPU 上的 /255 与 4 次 astype，流水线直接给 u8）
    cfg       : 同 compose_frame
    aligned   : True 时用 face["mat"]（aligned 画布），False 用 68 点算 WHOLE_FACE
    """
    from .compose import (_roi_of_face, cv2_invert, _color_transfer,
                          ct_code, ensure_engine_path)
    timings = timings if isinstance(timings, dict) else {}
    if not faces:
        return frame_bgr
    mode = str(cfg.get("mode") or "overlay").strip().lower()
    if "seamless" in mode and not seamless_on():
        raise ValueError("seamless 的 GPU 版被 DFL_COMPOSE_GPU_SEAMLESS 关了")
    c = ctx()
    t = c.torch
    T = timings
    t_all = time.perf_counter()
    ct = ct_code(cfg.get("color_transfer_mode", cfg.get("ct_mode")))
    hm_thr = int(cfg.get("hist_match_threshold", 255) or 255)
    h, w = frame_bgr.shape[:2]

    ensure_engine_path()
    if not aligned:
        from facelib import FaceType
        from facelib.LandmarksProcessor import get_transform_mat

    t_p = time.perf_counter()
    # ---- ① 每张脸的几何：矩阵 / ROI / 并集 ----
    items = []
    for f in faces:
        fo_shape = np.shape(f["pred"])
        size = int(fo_shape[0])
        if aligned:
            m = np.asarray(f.get("mat"), np.float64)
            if m.shape != (2, 3) or not np.all(np.isfinite(m)):
                raise ValueError("mat 不是合法的 2x3 仿射矩阵：%r" % (np.shape(f.get("mat")),))
        else:
            lm = np.asarray(f["landmarks"], np.float32)
            m = np.asarray(get_transform_mat(lm, size, FaceType.WHOLE_FACE, scale=1.0),
                           np.float64)
        box = _roi_of_face(m, size, frame_bgr.shape) if roi else None
        if roi and box is None:
            continue                       # 脸整个在帧外（和 CPU 一样直接跳过）
        if box is None:
            box = (0, 0, w, h)
        items.append((f, m, size, box))
    if not items:
        return frame_bgr
    ux0 = min(b[3][0] for b in items)
    uy0 = min(b[3][1] for b in items)
    ux1 = max(b[3][2] for b in items)
    uy1 = max(b[3][3] for b in items)
    urw, urh = ux1 - ux0, uy1 - uy0

    # ---- ② 只上传并集这一块（整帧不往返）----
    roi_u8 = frame_bgr[uy0:uy1, ux0:ux1]
    src_f = _as_chw(c, roi_u8)             # (3,urh,urw)：ct 的 dw 取样源（原帧）
    acc = src_f.clone()                    # 多脸时就是 CPU 的 work（最后一次才回 u8）
    T["prep_ms"] = T.get("prep_ms", 0.0) + (time.perf_counter() - t_p) * 1000.0

    def _offset(mm, ox, oy):
        """M' = M·T(o)：把 ROI 原点折进平移项（和 CPU 的 fwd 同一个算法）。

        注意**两个原点不能混**：dw（帧 ROI -> 画布）的源是并集那一块，用并集原点；
        反贴（画布 -> 帧 ROI）的 dst 是这张脸自己的框，用它自己的原点。
        混用会让第二张脸的 mw/fw 取到别的位置（实测双脸 max 15 LSB）。
        """
        o = np.array(mm, np.float64)
        o[0, 2] += mm[0, 0] * ox + mm[0, 1] * oy
        o[1, 2] += mm[1, 0] * ox + mm[1, 1] * oy
        return o

    for (f, m, size, (x0, y0, x1, y1)) in items:
        t_p = time.perf_counter()
        fwd_u = _offset(m, ux0, uy0)        # 并集原点：dw 的取样矩阵
        fwd_f = _offset(m, x0, y0)          # 脸自己的框：反贴用
        wrk, note = mask_prep(c, f, cfg, size, mat=m, frame_shape=frame_bgr.shape)
        T["mask_ms"] = T.get("mask_ms", 0.0) + (time.perf_counter() - t_p) * 1000.0

        t_c = time.perf_counter()
        fo = _as_chw(c, f["pred"])
        if np.asarray(f["pred"]).dtype != np.uint8:
            fo = fo.clamp_(0.0, 1.0)
        need_dw = bool(ct) or ("hist-match" in mode)
        if need_dw:
            dw = warp(c, src_f, cv2_invert(fwd_u), size, size)
            if ct:
                if ct in _GPU_CT:
                    if ct == 9:
                        fo = mt_transfer(c, fo, dw, wrk[0])
                else:
                    # 逐脸回退：warp 留在 GPU，ct 用 CPU 原实现（rct/mkl/idt/sot/...）
                    c.stats["fallback_ct"] += 1
                    fo_np = fo.permute(1, 2, 0).cpu().numpy()
                    dw_np = dw.permute(1, 2, 0).cpu().numpy()
                    wrk_np = wrk[0, 0].cpu().numpy()
                    out = _color_transfer(fo_np, dw_np, wrk_np, ct, T)
                    fo = t.as_tensor(np.ascontiguousarray(out.transpose(2, 0, 1)),
                                     device=c.dev)
            if "hist-match" in mode:
                fo = hist_match(c, fo, dw, hm_thr)
        T["ct_ms"] = T.get("ct_ms", 0.0) + (time.perf_counter() - t_c) * 1000.0

        t_p = time.perf_counter()
        rw, rh = x1 - x0, y1 - y0
        mw = warp(c, wrk[0], fwd_f, rw, rh).clamp_(0.0, 1.0)
        fw = warp(c, fo, fwd_f, rw, rh).clamp_(0.0, 1.0)
        T["paste_ms"] = T.get("paste_ms", 0.0) + (time.perf_counter() - t_p) * 1000.0
        # ---- ⑤ 融合（overlay / hist-match 都按遮罩 alpha 叠；seamless 不进来）----
        t_b = time.perf_counter()
        ox0, oy0 = x0 - ux0, y0 - uy0
        sub = acc[:, oy0:oy0 + rh, ox0:ox0 + rw]
        if "seamless" in mode:
            acc[:, oy0:oy0 + rh, ox0:ox0 + rw] = seamless_merge(c, sub, fw, mw, fo=fo,
                                                                 fwd=fwd_f, wrk=wrk)
        else:
            # 与 CPU 的 _blend_layer 同一条式子；拆成 3 个 kernel（不成 FMA），
            # 保证和 numpy 的「先乘、再乘、最后加」逐位同序。
            acc[:, oy0:oy0 + rh, ox0:ox0 + rw] = sub * (1.0 - mw) + fw * mw
        T["blend_ms"] = T.get("blend_ms", 0.0) + (time.perf_counter() - t_b) * 1000.0

    t_p = time.perf_counter()
    q = acc.mul_(255.0).clamp_(0.0, 255.0).to(t.uint8)
    out_roi = q.permute(1, 2, 0).contiguous().cpu().numpy()
    out = frame_bgr if roi is False else frame_bgr.copy()
    out[uy0:uy1, ux0:ux1] = out_roi
    T["blend_ms"] = T.get("blend_ms", 0.0) + (time.perf_counter() - t_p) * 1000.0
    c.stats["frames"] += 1
    c.stats["faces"] += len(items)
    T["gpu_ms"] = T.get("gpu_ms", 0.0) + (time.perf_counter() - t_all) * 1000.0
    return out


def compose_aligned_frame_gpu(frame_bgr, faces, cfg, timings=None, roi=True):
    """aligned 原生画布那一路（项目模式）的 GPU 版。"""
    return compose_frame_gpu(frame_bgr, faces, cfg, timings, roi=roi, aligned=True)


def stats() -> dict:
    c = _CTX["obj"]
    return dict(c.stats) if c is not None else {}