# XSegLite — 轻量 CNN 人脸分割

纯 Conv3x3 四阶 U-Net，无 Dense/GhostConv/SpatialGate。支持 ONNX 导出，任意分辨率。

## 与原版 XSeg 速度对比

| | PyTorch (GPU) | ONNX (GPU) | 参数量 | 文件大小 |
|---|---|---|---|---|
| 模型 | ONNX (CUDA) | TensorRT FP16 | 参数量 |
| 原版 XSeg (iperov) | ~10.4ms (96fps) | ~7.0ms (143fps) | 1.8M |
| **XSegLite** | **2.5ms (395fps)** | **1.46ms (684fps)** | **4.75M** |

*GPU: RTX 3080 (CUDA 13)，batch=1，eval 模式（BN 融合）。*

## 架构

```
256²×3
 enc0: Conv3x3(3→32)×2   + DSCPool → 128²×32
 enc1: Conv3x3(32→64)×2  + DSCPool → 64²×64
 enc2: Conv3x3(64→128)×2 + DSCPool → 32²×128
 enc3: Conv3x3(128→256)×3 + DSCPool → 16²×256
 bridge: Conv3x3(256→256) + ECA(256)
 dec3: ↑ConvT + Conv3x3(256→128)×3 → 32²×128 [aux]
 dec2: ↑ConvT + Conv3x3(128→64)×2  → 64²×64 [aux]
 dec1: ↑ConvT + Conv3x3(64→32)×2   → 128²×32
 dec0: ↑ConvT + Conv3x3(32→32)×2   → 256²×32
 out: Conv3x3(32→1) + Sigmoid
```

- 纯卷积：256/512/1024 均可，参数量始终 4.76M
- 损失函数：`0.5×BCE + 0.5×Dice + 0.3×AuxDice₃ + 0.2×AuxDice₂`
