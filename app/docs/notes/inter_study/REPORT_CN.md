# Inter 权重换分辨率实验（df-udt @ 416）

![input / baseline](input_416.png)
![分辨率 × 方法](grid_matrix.png)
![bicubic vs 伪逆](adv_compare.png)
![XSegLite 遮罩](xseg_mask.png)

模型：workspace/model/New416_SAEHD（df-udt, 416, e_dims=80, ae_dims=320）。素材：app/docs/notes/target.jpg（已对齐）。
前向：仓库自带 leras（DeepFakeArchi, opts=udt）；预处理 BGR + [0,1]；
**所有指标都在 XSegLite 遮罩内统计**（workspace/model/XSegLite/xseglite.onnx，256×256 NCHW）。

## 1. 为什么 bicubic 视觉上更好

把 dense1 的权重看成一张 C×n×n 的“图”，换分辨率就是对它重采样。各核的频响：

| 核 | 频响 | 结果 |
|---|---|---|
| nearest | 常数（=保持） | 频谱镜像混叠，块状 |
| bilinear | sinc²，通带掉得厉害（Nyquist 附近约 -4 dB） | 权重被抹平，人脸发糊 |
| bicubic (Catmull-Rom, a=-0.5) | 通带更平，-1 dB 量级 | 细节保得住，且可分离、代价低 |
| area | 盒式平均 | 只在降采样时等价于抗混叠，升采样时过平 |

权重里高频是“身份细节”，所以**谁的通带平，谁就更像原模型** —— 这就是 bicubic 视觉上赢的原因，
不是玄学。它的代价是负瓣带来的振铃（overshoot），但在权重空间只表现为轻微对比过冲，
经 sigmoid 输出后几乎不可见。

## 2. 但插值本身不是最优（数学）

要保的是泛函 ⟨W, f⟩。换分辨率后 f' ≈ (L⊗L)f，于是需要

    (L⊗L)ᵀ W' = W          … (1)

- **升分辨率**（m>n）：(1) 欠定 → 最小范数解 W' = (L⊗L)[(L⊗L)ᵀ(L⊗L)]⁻¹ W 
- **降分辨率**（m<n）：(1) 超定 → 最小二乘解 W' = [(L⊗L)(L⊗L)ᵀ]⁻¹(L⊗L) W

插值给的是 (L⊗L)W，**缺了那一步逆滤波**（它相当于把权重先锐化一次再做插值）。
逆滤波在高频会放大噪声，所以加 Tikhonov：升用 (LᵀL+λI)⁻¹，降用 (LLᵀ+λI)⁻¹。
这就是我说的 **pinv 方法**：`app/tools/bench/inter_adv.py`。

## 3. 遮罩内结果（XSegLite，416 输出为基准）

| 目标 | 方法 | PSNR | SSIM | MAE | 潜码 cos |
|---|---|---|---|---|---|
| 320 | nearest | 18.77 | 0.749 | 23.89 | 0.622 |
| 320 | bilinear | 19.11 | 0.745 | 23.29 | 0.747 |
| 320 | **bicubic** | **19.32** | 0.740 | 23.06 | 0.779 |
| 320 | area | 18.83 | 0.736 | 23.51 | 0.748 |
| 320 | pinv λ=0.1 | 19.31 | 0.715 | **16.96** | **0.789** |
| 384 | bicubic | 23.32 | 0.813 | 13.37 | 0.906 |
| 384 | **pinv λ=0.1** | **24.12** | **0.866** | **11.97** | **0.939** |
| 384 | pinv λ=0.01 | 23.19 | 0.849 | 11.82 | 0.941 |
| 448 | bicubic | 23.41 | 0.834 | **8.80** | 0.939 |
| 448 | **pinv λ=1e-4** | **24.98** | 0.872 | 9.21 | **0.962** |
| 448 | pinv λ=0.01 | 24.77 | **0.875** | 10.13 | 0.963 |
| 512 | bicubic | 20.26 | 0.814 | **18.78** | 0.915 |
| 512 | **pinv λ=1e-4** | **21.19** | **0.837** | 19.00 | **0.931** |

**pinv 在升分辨率（384/448/512）全面超过 bicubic**：PSNR +0.8~1.6 dB、SSIM +0.02~0.05、
潜码余弦 +0.02~0.03，代价是 MAE 略高（高频更锐，逐像素差略增）。
320（降分辨率）时 bicubic 的 PSNR 更高，但 pinv λ=0.1 的 MAE 与潜码余弦更好；
降采样本来就不可逆，这里两种方法都只是热启动。

### 3.1 数据驱动修正：负结果

我还试了用真实激活做最小二乘修正（对偶岭回归，96 帧视频特征，见 inter_adv.py:ls_correct）：

| 目标 | bicubic | ls 修正后 |
|---|---|---|
| 384 | 23.32 / 0.813 / cos 0.906 | 19.61 / 0.730 / cos 0.785 |
| 448 | 23.41 / 0.834 / cos 0.939 | 22.36 / 0.809 / cos 0.817 |
| 512 | 20.26 / 0.814 / cos 0.915 | 14.02 / 0.755 / cos 0.541 |

**反而更差**。原因：F'（新分辨率的真实特征）与 W 之间不存在严格的线性对应，
回归把“图像内容差异”也拟合进去了；96 个样本对 C·m² 维（16k~164k）来说太少，
λ 稍小就过拟合。要它有用，得用上万张脸 + 强正则，成本远超收益。记录在此，避免重复踩。

## 4. 结论与建议

1. 视觉上 **bicubic 明显好于 bilinear/nearest**（通带更平）；这条被数据确认。
2. 但**更好的选择是 pinv**（保泛函的伪逆重采样）：升分辨率 λ≈1e-4~1e-2，降分辨率 λ≈0.1。
   384/448/512 上 PSNR、SSIM、潜码余弦全面优于 bicubic。
3. 数据驱动的 LS 修正目前是负结果，不要用。
4. 所有方法都只是热启动；换完分辨率仍需续训（优化器状态形状不兼容会被删）。

## 5. 复现

     runtime\engine\python.exe app/tools/bench/inter_adv_study.py     # 遮罩内 48 组对比（含 pinv / ls）
     runtime\engine\python.exe app/tools/bench/inter_study_run.py     # 原始 16 组（bilinear 系列）
     python app/tools/bench/inter_resample.py --src <inter.pth> --res 416 --to 448 --method bicubic --out <out.pth>

实现：app/tools/bench/inter_adv.py（插值矩阵 L、pinv、ls_correct）、xseg_mask.py（遮罩内指标）。
数据：results_adv.json（48 组）、results.json（16 组）、latent_metrics.json。