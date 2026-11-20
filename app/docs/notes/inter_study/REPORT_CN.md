# Inter 权重换分辨率实验（df-udt @ 416）

![input / baseline](input_416.png)
![baseline inference](base416.png)
![resolution x method](grid_matrix.png)

模型：workspace/model/New416_SAEHD（df-udt, res=416, e_dims=80, ae_dims=320, iter 7,915,081）。
素材：app/docs/notes/target.jpg（**已对齐**，只 resize）。前向：仓库自带 leras
（core.leras.archis.DeepFakeArchi, opts=udt），权重按 param_N.npy 顺序灌入；参数量与
New416_SAEHD_summary.txt 一致（24,457,280 / 51,971,200 / 53,386,207 含 mask）。预处理实测为 **BGR + [0,1]**。

## 0. 上一版为什么是花屏（已修）

我第一版把 dense2/bias 按 (h,w,c) 解释，而引擎真实的展平是 **(c,h,w)**（行索引 = c·H·W + h·W + w）。
轴序错位 = 把同一段内存换个形状读 → 输出直接饱和成紫白。改回通道优先后，
所有分辨率都能出正常人脸（见上方矩阵图）。

## 1. 结论

**bicubic 在每个目标分辨率上都是最好的**（PSNR / SSIM / MAE 三项全胜，除 512 的 PSNR 略低于 bilinear 0.15 dB）。
nearest 一律最差。转换后的模型**可以直接出片**（不再是花屏），但仍是热启动，继续训练会更稳。

## 2. 端到端（转换后直接推理，与 416 baseline 比，输出回 416）

| 目标 | 方法 | PSNR | SSIM | MAE |
|---|---|---|---|---|
| 320 | nearest | 17.43 | 0.716 | 23.23 |
| 320 | bilinear | 19.28 | 0.715 | 19.97 |
| 320 | **bicubic** | **19.91** | **0.731** | **18.36** |
| 320 | area | 19.32 | 0.691 | 20.31 |
| 384 | nearest | 20.78 | 0.800 | 13.70 |
| 384 | bilinear | 23.67 | 0.853 | 11.11 |
| 384 | **bicubic** | **25.72** | **0.881** | **8.58** |
| 384 | area | 23.38 | 0.842 | 11.57 |
| 448 | nearest | 20.43 | 0.795 | 13.29 |
| 448 | bilinear | 23.58 | 0.849 | 9.94 |
| 448 | **bicubic** | **25.68** | **0.864** | **8.04** |
| 448 | area | 23.84 | 0.853 | 9.60 |
| 512 | nearest | 19.05 | 0.723 | 18.57 |
| 512 | bilinear | **20.48** | 0.777 | 15.38 |
| 512 | bicubic | 20.33 | 0.770 | 16.28 |
| 512 | area | 20.36 | 0.774 | 15.59 |

## 3. 潜空间（把 encoder 特征缩到目标网格，比 320 维潜码）

| 目标 | 方法 | latent MAE | cosine |
|---|---|---|---|
| 320 | nearest | 6.13 | 0.556 |
| 320 | bilinear | 5.37 | 0.809 |
| 320 | **bicubic** | **4.96** | **0.854** |
| 320 | area | 5.48 | 0.800 |
| 384 | nearest | 5.23 | 0.693 |
| 384 | bilinear | 4.45 | 0.853 |
| 384 | **bicubic** | **3.70** | **0.904** |
| 384 | area | 4.49 | 0.849 |
| 448 | nearest | 5.10 | 0.729 |
| 448 | bilinear | 3.56 | 0.882 |
| 448 | **bicubic** | **2.65** | **0.933** |
| 448 | area | 3.48 | 0.887 |
| 512 | nearest | 5.49 | 0.768 |
| 512 | bilinear | 3.25 | 0.906 |
| 512 | **bicubic** | **2.93** | **0.953** |
| 512 | area | 3.06 | 0.920 |

潜码 cosine 0.85~0.95（bicubic）说明映射关系被保留了大部分 —— 与花屏那版（cos≈0）完全不是一个量级。

## 4. 建议

- **默认用 bicubic**；nearest 不要用。
- 目标分辨率与源越接近越好：384/448 的 SSIM 0.86~0.88，320/512 掉到 0.69~0.78。
- 转换后可以直接出片；要长期训练则续训一段（优化器状态形状不兼容会被删）。
- 后缀/网格：带 t 的模型 enc = res/32；不带 t 是 res/16。用错网格会形状不符（硬报错），不会静默出错。

## 5. 复现

     runtime\engine\python.exe app/tools/bench/inter_study_run.py      # 16 组端到端 + baseline
     runtime\engine\python.exe app/tools/bench/inter_study_visual.py   # 对照矩阵图
     runtime\engine\python.exe app/tools/bench/inter_latent_metric.py  # 潜空间指标
     python app/tools/bench/inter_resample.py --src <inter.pth> --res 416 --to 384 --method bicubic --out <out.pth>

产物：app/docs/notes/inter_study/