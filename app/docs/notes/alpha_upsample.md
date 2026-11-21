# 遮罩上采样：一次受控对比

XSeg 在 256² 出遮罩，合成时被放大到目标分辨率。这次比较五种上采样在 alpha 上的表现。

## 设置

```
GT      1024² 解析 alpha：椭圆软边（约 0.08 px）+ 三根 2~3 px 的发丝
降采样   INTER_AREA ×4 → 256²
上采样   bilinear / bicubic / lanczos4 / 引导滤波（He）/ 联合双边
指标     PSNR、IoU@0.5、边界定位误差（预测边界到 GT 边界的平均距离）
```

## 结果

| 方法 | PSNR | IoU@0.5 | 边界误差 (px) | 1024² 耗时 |
|---|---|---|---|---|
| bilinear | 28.25 | 0.9913 | 0.079 | 7.4 ms |
| bicubic | 29.39 | 0.9941 | 0.090 | — |
| **lanczos4** | **29.60** | **0.9952** | **0.073** | — |
| 引导滤波 | 28.66 | 0.9833 | 0.633 | 27 ms |
| 联合双边 | 29.37 | 0.9805 | 1.058 | 27.6 s（未优化） |

两个意料之外的点：

1. **引导滤波与联合双边都更差**。它们依赖「引导图的高频与 alpha 边缘真实对应」，
   本次引导图是人造的（alpha 的灰度化 + 噪声），没有这种对应关系，于是只在平滑 alpha、
   把边界推歪了 0.6~1.1 px。真实人脸图会不会更好，需要真实素材再验；但在没有真值的情况下，
   这条线无法给出可信结论。
2. **合成器本来就用 cubic**（`compose.py` 里的 `warpAffine / resize` 全部 `INTER_CUBIC`），
   并不是我先前以为的 bilinear。所以「换成更好的插值」这个改动的空间只有 cubic → lanczos 的 0.21 dB。

## 结论

```
改动不值得：cubic → lanczos 只有 +0.21 dB、边界误差 0.090 → 0.073 px，
而 lanczos 在二值 alpha 上会产生过冲（负值被截断后形成一圈浅晕）。
引导滤波方向在缺少真值的情况下无法验证，暂不投入。
现状保持 cubic。
```

脚本：`app/tools/bench/alpha_upsample_bench.py`　图：`alpha_upsample.png`（GT / bilinear / bicubic / lanczos / guided）