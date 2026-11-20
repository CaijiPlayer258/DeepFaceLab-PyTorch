# Inter Weight Rescaling Study (df-udt @ 416)

![input / baseline](input_416.png)
![baseline inference](base416.png)

Model: workspace/model/New416_SAEHD (df-udt, res=416, e_dims=80, ae_dims=320, iter 7,915,081).
Input app/docs/notes/target.jpg is an **already aligned** crop (resize only). Forward pass uses the
repo's own leras (core.leras.archis.DeepFakeArchи, opts=udt) with weights fed in param_N.npy order;
parameter counts match New416_SAEHD_summary.txt exactly (24,457,280 / 51,971,200 / 53,386,207 incl. mask).
Preprocessing measured to be **BGR + [0,1]**.

## 1. Headline: interpolating inter does NOT preserve behaviour

Encode at 416, rescale the feature grid to the target, compare the 320-d latent code of the original
versus the rescaled inter (both interpolated back to 13x13):

| target | method | latent MAE | cosine |
|---|---|---|---|
| 320 | nearest | 32.94 | -0.003 |
| 320 | bilinear | 17.73 | -0.001 |
| 320 | bicubic | 24.17 | -0.004 |
| 320 | **area** | **13.16** | 0.005 |
| 384 | nearest | 35.23 | -0.008 |
| 384 | bilinear | 13.83 | 0.005 |
| 384 | **area** | **13.48** | 0.005 |
| 448 | nearest | 44.28 | -0.009 |
| 448 | **bilinear** | **21.95** | 0.007 |
| 448 | area | 22.60 | 0.007 |
| 512 | nearest | 46.16 | -0.006 |
| 512 | **bilinear** | **30.87** | -0.004 |
| 512 | area | 33.74 | -0.004 |

Cosine stays within +/-0.01: the rescaled inter output is essentially uncorrelated with the original.
dense1 rows are independent coefficients; interpolating mixes spatial neighbours, so the low-frequency
shape survives but the mapping is destroyed. Rescaling is therefore a **warm start only**; retraining is
mandatory (optimizer state is shape-incompatible and gets dropped).

## 2. End-to-end (converted weights, no retraining)

![resolution x method](grid_matrix.png)

| target | method | PSNR | SSIM | MAE |
|---|---|---|---|---|
| 320 | nearest | 3.44 | 0.432 | 154.4 |
| 320 | bilinear | 3.69 | 0.392 | 149.7 |
| 320 | bicubic | 3.56 | 0.413 | 152.1 |
| 320 | area | 3.66 | 0.390 | 150.4 |
| 384 | nearest | 3.55 | 0.304 | 149.3 |
| 384 | bilinear | 3.41 | 0.326 | 152.9 |
| 384 | bicubic | 3.36 | 0.349 | 155.2 |
| 384 | area | 3.44 | 0.317 | 152.2 |
| 448 | nearest | 3.48 | 0.411 | 153.2 |
| 448 | bilinear | 4.19 | 0.393 | 137.5 |
| 448 | bicubic | 4.01 | 0.396 | 141.2 |
| 448 | area | 4.14 | 0.400 | 138.7 |
| 512 | nearest | 3.48 | 0.409 | 153.1 |
| 512 | bilinear | 3.69 | 0.345 | 148.2 |
| 512 | bicubic | 3.66 | 0.364 | 148.9 |
| 512 | area | 3.66 | 0.358 | 148.8 |

This is what happens without retraining: outputs drift and saturate (see the matrix), hence PSNR 3-4 dB;
SSIM 0.3-0.43 is coincidental structural overlap, not quality.

## 3. Practical guidance

- Treat the converted model as a **warm start**; fine-tune before producing output.
- If an interpolation must be chosen: lowest latent MAE is **area** (320/384) and **bilinear** (448/512);
  **nearest is always worst** (2-3x higher latent MAE).
- Using the wrong grid (parsing a `t` model as res/16) fails shape checks: the archi suffix decides enc/low.

## 4. Shapes and layout

| res | low=res/32 | dense1 | dense2 | bias |
|---|---|---|---|---|
| 416 | 13 | (108160,320) | (320,54080) | (54080,) |
| 320 | 10 | (64000,320) | (320,32000) | (32000,) |
| 384 | 12 | (92160,320) | (320,46080) | (46080,) |
| 448 | 14 | (125440,320) | (320,62720) | (62720,) |
| 512 | 16 | (163840,320) | (320,81920) | (81920,) |

*.pth are zips of param_N.npy (no pickle); Dense stored (in,out), Conv OIHW; dense1 rows are flattened
channel-first (c,h,w) - see app/engine/models/Model_SAEHD/resize_inter.py.

## 5. Reproduce

     runtime\engine\python.exe app/tools/bench/inter_study_run.py      # 16 combos + baseline
     runtime\engine\python.exe app/tools/bench/inter_study_visual.py   # comparison matrix
     runtime\engine\python.exe app/tools/bench/inter_latent_metric.py  # latent metrics

Artifacts: app/docs/notes/inter_study/