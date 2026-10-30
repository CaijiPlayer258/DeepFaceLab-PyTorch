# XSegLite — Lightweight CNN XSeg

Pure Conv3x3 4-stage U-Net for face segmentation. No Dense, no GhostConv, no SpatialGate.
ONNX exportable, any resolution.

## Speed vs Original XSeg

| Model | ONNX (CUDA) | TensorRT FP16 | Params |
|---|---|---|---|
| Original XSeg (iperov) | ~10.4ms (96fps) | ~7.0ms (143fps) | N/A | 1.8M |
| **XSegLite** | **2.5ms (395fps)** | **1.46ms (684fps)** | **4.75M** |

*GPU: RTX 3080 (CUDA 13), batch=1, eval mode (BN fused).*

## Architecture

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

- Fully convolutional: 256/512/1024, same params (4.76M)
- Loss: `0.5×BCE + 0.5×Dice + 0.3×AuxDice₃ + 0.2×AuxDice₂`
