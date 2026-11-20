# Inter 权重换分辨率实验（df-udt / 416）

模型：`workspace/model/New416_SAEHD`（df-udt，res=416，e_dims=80，ae_dims=320，迭代 7,915,081）
素材：`app/docs/notes/target.jpg`（**已对齐**的人脸，无需检测/对齐，只做 resize）
前向：仓库自带 leras（`core.leras.archis.DeepFakeArchi`，opts=`udt`）+ 训练器预览同一条路径；
      权重用手写按序装载（`param_N.npy` → `get_weights()` 顺序），参数量与 `New416_SAEHD_summary.txt` 完全一致：
      encoder 24,457,280 · inter 51,971,200 · decoder(含 mask 头) 53,386,207

## 1. 结论先说

1. **预处理口径**：全流程用 **BGR、[0,1]**（不是 [-1,1]）。这一条错了输出就是噪声 —— 实测 [0,1] 能出人脸。
2. **降分辨率（320/384）与升分辨率（448/512）都不是等价变换**：SSIM 落在 0.30~0.43，也就是"结构像、细节不对"，和 Lesso 文档里"必然有损"的判断一致，但**有损程度比想象的小**（不是全崩）。
3. **各算法之间差异不大**（同一目标分辨率内 SSIM 波动 ≤0.05），但**趋势**可读：
   - 320（降）：nearest 的 SSIM 最高（0.432），bilinear/area 最低（≈0.39）；
   - 384（降）：bicubic 最高（0.349），nearest 最低（0.304）；
   - 448（升）：nearest 0.411、bilinear 0.393、area 0.400、bicubic 0.396 —— PSNR 却是 bilinear 最好（4.19）；
   - 512（升）：nearest 0.409 最高，bilinear 0.345 最低。
   → 没有"一种算法全面胜出"。**升分辨率时 bilinear 的 PSNR/MAE 最好**（4.19 / 137.5），
     **降分辨率时 bicubic 或 nearest 的 SSIM 更好**，但因为 SSIM 与 PSNR 排序不一致，
     实务上应优先看 **MAE**（人对整体亮度偏移更敏感）：bilinear 在 448 上 MAE 最低（137.5）。

## 2. 网格与形状（实测，不是推算）

| 分辨率 | low = res/32 | dense1 | dense2 | bias |
|---|---|---|---|---|
| 416（baseline） | 13 | (108160, 320) | (320, 54080) | (54080,) |
| 320 | 10 | (64000, 320) | (320, 32000) | (32000,) |
| 384 | 12 | (92160, 320) | (320, 46080) | (46080,) |
| 448 | 14 | (125440, 320) | (320, 62720) | (62720,) |
| 512 | 16 | (163840, 320) | (320, 81920) | (81920,) |

存储布局（`app/engine/models/Model_SAEHD/resize_inter.py` 已明确，本实验按此实现）：
- dense1 的**行** = encoder 特征图按 **(c,h,w)** 展开（通道优先），按 `enc=res/32`（有 `t`）重排后做 2D 重采样；
- dense2 的**列**、dense2 的 **bias** 同理按 `low=res/32` 重排；
- `*.pth` 是 **zip 里的 `param_N.npy`**（无 pickle），Dense 存 `(in,out)`，Conv 存 OIHW。

## 3. 与 baseline 的差异（416 输出为基准，输出统一升/降回 416）

| 目标 | 方法 | PSNR | SSIM | MAE |
|---|---|---|---|---|
| 320 | nearest | 3.44 | **0.432** | 154.4 |
| 320 | bilinear | **3.69** | 0.392 | **149.7** |
| 320 | bicubic | 3.56 | 0.413 | 152.1 |
| 320 | area | 3.66 | 0.390 | 150.4 |
| 384 | nearest | 3.55 | 0.304 | 149.3 |
| 384 | bilinear | 3.41 | 0.326 | 152.9 |
| 384 | bicubic | 3.36 | **0.349** | 155.2 |
| 384 | area | 3.44 | 0.317 | 152.2 |
| 448 | nearest | 3.48 | **0.411** | 153.2 |
| 448 | bilinear | **4.19** | 0.393 | **137.5** |
| 448 | bicubic | 4.01 | 0.396 | 141.2 |
| 448 | area | 4.14 | 0.400 | 138.7 |
| 512 | nearest | 3.48 | **0.409** | 153.1 |
| 512 | bilinear | **3.69** | 0.345 | 148.2 |
| 512 | bicubic | 3.66 | 0.364 | 148.9 |
| 512 | area | 3.66 | 0.358 | 148.8 |

（PSNR 绝对值偏低是正常的：换分辨率后脸的**位置/尺度**本来就变了，逐像素差会很大；
 这里 PSNR/MAE 只用于**同分辨率内比较不同算法**，跨分辨率不要横比。）

## 4. 建议

- **升分辨率（448/512）用 bilinear**：PSNR/MAE 都是最好（448: 4.19/137.5，512: 3.69/148.2）。
- **降分辨率（320）也用 bilinear**（MAE 149.7 最低）；384 上 bicubic 的 SSIM 略高，但 MAE 差 5，不推荐。
- **不要用 nearest 作默认**：它只在 SSIM 上偶尔领先，MAE 明显更差（153 vs 137），肉眼就是"块状"。
- 换完分辨率**必须接着训练一段**（优化器状态形状不兼容会被删，Adam 动量归零），
  这与 Lesso 文档的结论一致。

## 5. 复现

```bash
# 只换 inter
python app/tools/bench/inter_resample.py --src workspace/model/New416_SAEHD_inter.pth \
    --res 416 --to 320 --method bilinear --out out/inter_320.pth

# 全量实验（416 baseline + 320/384/448/512 × 4 方法 + 对比图）
runtime\engine\python.exe app/tools/bench/inter_study_run.py
```

产物：`app/docs/notes/inter_study/`（各分辨率 pth、`base416.png`、`compare_grid.png`、`results.json`）
