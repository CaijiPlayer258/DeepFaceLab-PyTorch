# Inter Weight Rescaling: derivation and conclusions

Model `New416_SAEHD` (df-udt, 416, e_dims=80, ae_dims=320); aligned face input; forward via repo leras
(`DeepFakeArchi`, opts=`udt`); preprocessing BGR + [0,1]; all metrics inside the **XSegLite mask**, baseline = 416 output.

![every resolution vs every method (baseline repeated on the left)](compare_all.png)

## 1. Derivation

### 1.1 Rescaling is a convolution of the weight field

Each row of dense1 maps to one spatial position of the encoder feature map (index = c*H*W + h*W + w), so
rescaling Inter is a 2-D resampling of a C x n x n coefficient field, operator L x L:

    W' = (L x L) W

The flatter the kernel passband, the less high-frequency content (identity detail) is lost:
lanczos-3 (-0.5 dB) > bicubic (-1 dB) > bilinear (-4 dB); nearest applies no low-pass but destroys phase.
Every interpolation kernel is a low-pass, so the family is always structurally right and slightly soft.

### 1.2 The real objective: preserve the functional

Downstream only sees <W, f>. Since f' ~ (LxL)f, we need

    (LxL)^T W' = W

- upscale (m>n): underdetermined -> minimum norm `W' = (LxL)[(LxL)^T(LxL)]^-1 W`
- downscale (m<n): overdetermined -> least squares `W' = [(LxL)(LxL)^T]^-1 (LxL) W`

Interpolation only computes `(LxL)W` and misses that inverse filter. That is **pinv**, damped by Tikhonov
(`(L^T L + lam I)^-1` up, `(L L^T + lam I)^-1` down) because full inversion amplifies noise.

### 1.3 Partial inverse filtering (the blend)

    W' = a*lanczos + (1-a)*pinv,  a = 0.5
    pinv kernel depends on direction: lanczos up, area down (box pre-filters aliasing)

### 1.4 Hard floor when downscaling (rank deficiency)

    rank(LxL) = m^2 < n^2  =>  (n^2-m^2)/n^2 of the space is discarded and unobservable
    n^2 = 169: 320 loses 41% (m^2=100), 384 loses 15% (144), 288 loses 52% (81)
    error floor ||(I-P)W|| is algorithm independent; upscale loses no dimension, hence higher scores

Sweeping lam from 0.1 to 10 at 288 makes PSNR worse (18.18 -> 14.70): the loss is structural, not a tuning issue.

## 2. Results (PSNR / SSIM / MAE, masked)

| target | grid | bicubic | lanczos | pinv | **blend (default)** |
|---|---|---|---|---|---|
| 288 | 13->9 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | 18.18 / 0.698 / **21.6** |
| 320 | 13->10 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.24 / 0.718 / 17.3** |
| 384 | 13->12 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **25.10 / 0.874 / 9.64** |
| 448 | 13->14 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **26.27 / 0.897 / 6.39** |
| 512 | 13->16 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **23.97 / 0.847 / 8.73** |
| 544 | 13->17 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **22.93 / 0.824 / 10.33** |

## 3. Conclusions

1. **lanczos-3 is the best interpolation kernel**; bicubic is usable; nearest only looks sharp at small ratios.
2. **pinv restores the missing inverse filter**; full inversion over-sharpens, so its MAE is worse alone.
3. **blend = 0.5*lanczos + 0.5*pinv (area for downscale)** is the best method on 5 of 6 targets and ships as default.
4. **Downscaling has a mathematical floor** set by the discarded dimension fraction (15% / 41% / 52%); only retraining helps.
5. **Practical bound**: move down at most 1-2 grid steps (~32-64 px); 416 -> 288 should be retrained instead.
6. Rescaling is a **warm start**: optimizer state is shape-incompatible and dropped, so fine-tune before use.

## 4. Shipping

    app/engine/models/Model_SAEHD/resize_inter.py
        _METHODS = ('blend','bicubic','bilinear','lanczos','pinv')     # default blend
        python -m models.Model_SAEHD.resize_inter --model-dir <dir> --new-res 448

    tool measurement (416->448, masked): PSNR 26.27 / SSIM 0.897 / MAE 6.39 / NCC 0.978

Scripts: `inter_study_run.py`, `inter_adv_study.py`, `inter_blend_study.py`, `inter_ratio_study.py`,
`inter_hybrid_study.py`, `inter_extreme_study.py`, `inter_lambda_study.py`, `xseg_mask.py`.
Engineering doc: [38-INTER-RESIZE.md](../38-INTER-RESIZE.md).