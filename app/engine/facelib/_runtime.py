"""facelib 四个提取器的公共运行时。

S3FDExtractor / FANExtractor / XSegNet / FaceEnhancer 的 __init__ 开头、
extract 里的「入图 → torch → 前向 → 回 CPU」，原本是各抄一遍的同一段代码；
这里收成一处。判据与 compose_common 相同：抽出来之后每一步的张量形状、dtype、
设备、重采样都不变，所以推理出的数字逐位不变。

**不放**模型结构、权重映射、后处理（NMS / 关键点解码 / 遮罩阈值）—— 那些是各模型
自己的事，混进来反而更难查。

对外只有这几样：
    init_leras()      初始化 leras 主环境（幂等）
    pick_device()     按「要不要用 CPU」挑 torch.device
    to_nchw()         HWC / NHWC numpy → NCHW torch tensor
    to_numpy_hwc()    NCHW torch tensor → HWC numpy（CPU）
    forward()         no_grad 下跑一次前向
    fit_square()     缩到 size×size（已是这个尺寸就原样返回）
"""
from __future__ import annotations

import cv2
import numpy as np

from core.leras import nn


def init_leras(data_format="NCHW", cpu_only=False, probe_devices=True):
    """初始化 leras 主环境。四个提取器 __init__ 的第一件事。

    probe_devices=False 走 `set_data_format` —— 不重挑设备（XSegNet 的用法：
    它已经在 import 期由别处 initialize 过，再 initialize 一次会把用户的设备选择
    覆盖成 BestGPU）。

    cpu_only=True 挂一个空的 DeviceConfig，等于「别探测 GPU」—— 调用方明确要 CPU
    时用这条（FaceEnhancer），否则 nn.device 可能仍指向 cuda。
    """
    nn.initialize_main_env()
    if not probe_devices:
        nn.set_data_format(data_format)
    elif cpu_only:
        nn.initialize(device_config=nn.DeviceConfig([]), data_format=data_format)
    else:
        nn.initialize(data_format=data_format)


def pick_device(place_model_on_cpu=False, run_on_cpu=False):
    """要 CPU 就给 CPU，否则用 leras 当前主设备（与 nn.device 是同一个对象）。"""
    if run_on_cpu or place_model_on_cpu:
        return nn.torch.device("cpu")
    return nn.device


def to_nchw(arr, device=None, dtype=None):
    """HWC（3D）或 NHWC 批量（4D）的 numpy → NCHW torch tensor。

    3D 会补一个 batch 维。通道判定照 4.0 的老规矩：**末维是 1 或 3 就当 NHWC**，
    否则认为入参已经是 NCHW（XSegNet 的老实现就是这个启发式，别改成无条件的
    transpose —— 有人直接喂 NCHW 进来）。
    """
    t = nn.torch
    a = np.ascontiguousarray(arr)
    if a.ndim == 3:
        a = a[None, ...]
    if a.ndim != 4:
        raise ValueError(f"expected 3D/4D image array, got shape {np.shape(arr)}")
    if a.shape[-1] in (1, 3):
        a = np.transpose(a, (0, 3, 1, 2))
    x = t.from_numpy(a)
    if dtype is not None:
        x = x.to(dtype=dtype)
    if device is not None:
        x = x.to(device=device)
    return x


def to_numpy_hwc(x):
    """NCHW 4D torch tensor → HWC numpy（先回 CPU）。取 batch 的第 0 张。"""
    t = nn.torch
    if not isinstance(x, t.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if x.ndim != 4:
        raise ValueError(f"期望 NCHW 4D tensor，实际 ndim={x.ndim}")
    return x.detach().cpu()[0].permute(1, 2, 0).contiguous().numpy()


def forward(model, *args):
    """no_grad 下跑一次前向，返回**原始输出**（张量还在原设备上，调用方自己收）。"""
    with nn.torch.no_grad():
        return model(*args)


def fit_square(img, size, interpolation=cv2.INTER_LINEAR):
    """把图缩到 size×size；已经是这个尺寸就原样返回（不做无谓重采样）。"""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    return cv2.resize(img, (size, size), interpolation=interpolation)
