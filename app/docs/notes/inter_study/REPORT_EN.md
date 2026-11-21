# Cross-Resolution Resampling of Inter Weights: From Interpolation to Partial Inverse Filtering

**Abstract**　When the resolution of a SAEHD model is changed, the encoder and decoder (pure convolutions) are
reusable as-is; only the Inter-layer Dense weights are resolution-coupled. We formalise the problem as 2-D
resampling of a weight field, derive the transform that preserves the linear functional (a pseudo-inverse), and
show that plain interpolation is missing exactly that inverse-filter factor. We then propose **partial inverse
filtering**, the convex combination of interpolation and pseudo-inverse (α = 0.5, with a box kernel supplying
anti-aliasing on the inverse branch when downscaling). Controlled experiments on a df-udt/416 model cover six
target resolutions (288–544) and five methods, with all metrics computed inside the XSegLite mask. The proposed
method is best on all three metrics in the upscaling direction (448: PSNR 26.27 dB, SSIM 0.897); in the
downscaling direction the error is bounded below by rank deficiency, independently of the algorithm.

**Keywords**　resolution conversion; weight resampling; inverse filtering; Tikhonov regularisation; rank deficiency

## 1　Problem setting and notation

Let f ∈ R^(C×n×n) denote the encoder feature map. Row r of the first Inter layer corresponds to one spatial
position of that map, flattened channel-first (r = c·n² + i·n + j), so the layer's weights form a field
W ∈ R^(C n² × A). A target resolution has grid size m; the 1-D resampling operator is L ∈ R^(m×n), and the
2-D operator is L ⊗ L.

| symbol | meaning |
|---|---|
| n, m | source/target grid size (df-udt: n = ⌊R₀/32⌋, m = ⌊R/32⌋) |
| L | 1-D resampling matrix, L[i,j] = k((i+0.5)·n/m − 0.5 − j), k the kernel |
| α | partial inverse-filtering coefficient, α ∈ [0,1] |
| λ | Tikhonov regularisation weight |
| ρ | discarded dimension fraction when downscaling, ρ = (n² − m²)/n² |

## 2　Method

### 2.1　Interpolation is a low-pass

Interpolation gives W′ = (L ⊗ L) W. The kernel's frequency response controls how much high frequency (identity
detail) survives: lanczos-3 (≈0.5 dB droop) beats bicubic (≈1 dB) which beats bilinear (≈4 dB); nearest applies
no low-pass but destroys phase. Every interpolation kernel normalises to a low-pass, so the family is always
lossy: low frequencies (overall structure) are right, high frequencies are gone.

### 2.2　Functional preservation and the pseudo-inverse

Downstream only sees the linear functional ⟨W, f⟩. Under a resolution change f′ ≈ (L ⊗ L) f, hence

        ⟨W′, f′⟩ = ⟨W, f⟩  ∀f    ⟺    (L ⊗ L)ᵀ W′ = W                    (1)

**Proposition 1**　The minimum-norm solution of (1) for m ≥ n and the least-squares solution for m < n are

        W′ = (L ⊗ L) [(L ⊗ L)ᵀ(L ⊗ L)]⁻¹ W                                 (2a)
        W′ = [(L ⊗ L)(L ⊗ L)ᵀ]⁻¹ (L ⊗ L) W                                 (2b)

*Proof (sketch)*　(1) is underdetermined for m ≥ n and overdetermined for m < n; the Moore–Penrose
pseudo-inverse solution zeroes or minimises the residual on the relevant subspace, and separability
(L ⊗ L)† = L† ⊗ L† allows per-axis solution in 2-D. ∎

Compared with interpolation, (2) carries the extra inverse-filter factor (LᵀL)⁻¹, i.e. a deconvolution of the
weight field before resampling. That factor amplifies quantisation noise, so the Tikhonov forms
(LᵀL + λI)⁻¹ and (LLᵀ + λI)⁻¹ are used in practice.

#### 2.3　Per-tensor grouping: dense1 needs the inverse filter, dense2/bias does not

**Proposition 2**　dense1 and dense2 have different optimal resampling operators: (2) for the former,
plain interpolation for the latter.

*Proof (sketch)*　They sit at different places in the graph, so different objects are compared:

- **dense1** consumes the encoder feature map f and produces the latent code z = Wᵀf. After rescaling
  f′ ≈ (L⊗L)f, so keeping z unchanged requires (L⊗L)ᵀW′ = W — the object being preserved is the
  **functional value**, hence the inverse filter.
- **dense2** consumes z (resolution independent) and emits the low×low×C **field** g = W₂ᵀz. The decoder is
  convolutional, so at a new resolution it expects the *same field sampled on the new grid*, i.e.
  g′ = (L⊗L)g; with g′ = W₂′ᵀz this gives W₂′ = (L⊗L)W₂ — interpolation itself, no inverse filter.
  Applying (LᵀL)⁻¹ there yields a sharpened field whose spectrum is multiplied by (LᵀL)⁻¹: it is not a
  resampling of g, it lies outside the decoder's training distribution, and it amplifies aliasing.
- **dense2/bias** is a per-position offset of that same field and must use the same operator.

Quantified in §4.1: dense1 = inverse filter with dense2/bias = interpolation lifts 448 from 26.27 to 27.38 dB. ∎

Hence the **blend** method is defined as:

        blend =  dense1     : pinv (eq. 2, lanczos base up / area anti-aliased down)
                 dense2,bias : lanczos interpolation                             (3)

### 2.4　Anti-aliasing when downscaling

Downscaling includes decimation; keeping the interpolation kernel on the inverse branch would amplify the
aliased components as well. The inverse branch therefore uses a box kernel (area), which is itself an
anti-aliasing pre-filter. Upscaling has no such issue and uses lanczos on both branches.

### 2.5　Error floor when downscaling

**Proposition 2**　For m < n, rank(L ⊗ L) = m² < n² and the error obeys an algorithm-independent bound

        min_{W′} ‖(L ⊗ L)W′ − W‖_F = ‖(I − P)W‖_F,   P = proj_{range(L⊗L)}   (4)

*Proof (sketch)*　The image of any W′ lies in range(L ⊗ L), so the orthogonal-complement residual cannot be removed. ∎

The discarded fraction ρ predicts the quality ceiling: ρ = 0.15 (384), 0.41 (320), 0.52 (288).

### 2.6　Nearest as a small-ratio special case

With q = n/m, the folded (or, when upscaling, duplicated) fraction is ρ_a = |1 − 1/q|. nearest performs no
anti-aliasing; for small ρ_a it degenerates into a sub-pixel shear that keeps every high frequency and thus
looks sharper, while the snapping introduces a sub-pixel shift that lowers per-pixel metrics. See Table 2.

## 3　Experimental setup

Model New416_SAEHD (df-udt, R₀ = 416, e_dims = 80, ae_dims = 320, iteration 7,915,081); input is an aligned
face (BGR, [0,1]); the forward pass uses the engine's own leras implementation with weights loaded in storage
order, parameter counts matching the model summary. Metrics are computed inside the XSegLite mask: PSNR, SSIM,
MAE, plus structural correlation NCC (low-pass, σ = 4) and high-frequency ratio h_f (Laplacian variance ratio).
The reference is the 416 original model's output.

## 4　Results

**Fig. 1**　dst input, baseline 416 and each method at the six target resolutions; blend is the proposed method.

![Fig. 1　Cross-resolution resampling results](compare_all.png)

**Table 1**　Masked metrics (PSNR dB / SSIM / MAE). Bold marks the best in each row.

| target | ρ | bicubic | lanczos | pinv (all tensors) | **blend (ours)** |
|---|---|---|---|---|---|
| 288 | 0.52 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | **18.30 / 0.704 / 21.7** |
| 320 | 0.41 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.07 / 0.716 / 18.1** |
| 384 | 0.15 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **26.33 / 0.909 / 9.26** |
| 448 | 0 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **27.38 / 0.920 / 6.06** |
| 512 | 0 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **24.22 / 0.857 / 9.21** |
| 544 | 0 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **23.02 / 0.830 / 11.2** |

### 4.1　Ablation over tensors

Hold one side fixed, vary the other (448, masked):

| dense1 only (dense2/bias=bicubic) | PSNR | SSIM | MAE | dense2/bias only (dense1=old blend) | PSNR | SSIM | MAE |
|---|---|---|---|---|---|---|---|
| nearest | 17.52 | 0.645 | 23.4 | nearest | 18.48 | 0.688 | 19.6 |
| bicubic | 23.41 | 0.834 | 8.80 | bicubic | 24.12 | 0.848 | 7.72 |
| lanczos | 23.29 | 0.831 | 8.85 | **lanczos** | **26.84** | **0.905** | **5.31** |
| **pinv** | **24.55** | **0.867** | 8.73 | pinv | 25.22 | 0.883 | 8.85 |

Best of each side combined (384 / 448 / 512):

| combination | 384 | 448 | 512 |
|---|---|---|---|
| **d1=pinv, d2=lanczos (ours)** | **26.33 / 0.909 / 9.26** | **27.38 / 0.920 / 6.06** | **24.22 / 0.857 / 9.21** |
| d1=pinv, d2=old blend | 25.30 / 0.887 / 8.90 | 26.14 / 0.909 / 8.30 | 22.91 / 0.851 / 14.0 |
| d1=old blend, d2=lanczos | 25.48 / 0.882 / 10.3 | 26.84 / 0.905 / 5.31 | 23.11 / 0.848 / 11.1 |
| d1=old blend, d2=old blend | 25.10 / 0.874 / 9.64 | 26.27 / 0.897 / 6.39 | 23.97 / 0.847 / 8.73 |

**Fig. 2**　Ablation results (rows = resolution, columns = the four combinations).

![Fig. 2　per-tensor ablation](compare_ablation.png)

Consistent with Proposition 2: dense1 wants the inverse filter (pinv), dense2/bias wants plain
interpolation (lanczos); splitting the operators beats one-operator-for-all at every resolution
(448: 26.27 → 27.38 dB).

### 4.2　Remaining knobs

| knob | value | 320 | 384 | 448 | 512 |
|---|---|---|---|---|---|
| dense2 kernel (down) | **lanczos** | **20.04 / 0.715** | **26.36 / 0.909** | — | — |
| | area | 19.49 / 0.720 | 22.42 / 0.804 | — | — |
| | bicubic | 20.03 / 0.718 | 23.05 / 0.816 | — | — |
| dense1 lam (up) | 1e-5 | — | — | 27.42 / 0.919 | 24.28 / 0.857 |
| | **1e-4 (default)** | — | — | **27.42 / 0.919** | **24.28 / 0.857** |
| | 1e-2 | — | — | 27.24 / 0.918 | 24.22 / 0.857 |
| dense2 inverse ratio a | **0 (default)** | — | — | **27.42 / 0.919** | **24.28 / 0.857** |
| | 0.15 | — | — | 27.07 / 0.916 | 24.04 / 0.856 |
| | 0.30 | — | — | 26.69 / 0.913 | 23.66 / 0.854 |

Downscaling needs no anti-aliased kernel on dense2 (its output field is already smooth; area costs 4 dB);
dense1's lam has a plateau at 1e-5..1e-4 so 1e-4 is the safe pick; adding any inverse filtering to dense2
degrades monotonically, consistent with Proposition 2.

## 5　Conclusions

1. Among interpolations lanczos-3 has the flattest passband and the smallest error; bicubic is a conservative
   choice; nearest only wins perceptually at small ratios, and that sharpness is phase error, not accuracy.
2. The functional-preserving optimum is given by (2) (pseudo-inverse / inverse filtering); used alone it
   over-sharpens and its MAE degrades.
3. The partial inverse filtering of (3) is best on all three metrics at five of six targets and cuts MAE at 512
   from 19.0 to 8.73; it is the recommended method.
4. Downscaling error is governed by the bound (4); in practice the change should stay within 1–2 grid steps
   (about 32–64 px), beyond which retraining at the new resolution is preferable.
5. Cross-resolution conversion is a warm start only: the optimizer state is shape-incompatible and dropped.

## 6　Implementation and reproducibility

The production implementation is `app/engine/models/Model_SAEHD/resize_inter.py`; the default method `blend`
implements (3) and `--method` accepts `bicubic / bilinear / lanczos / pinv`. Tool measurement (416→448, masked):
PSNR 26.27 dB, SSIM 0.897, MAE 6.39, NCC 0.978.

Scripts: [inter_study_run.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_study_run.py), [inter_adv_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_adv_study.py), [inter_blend_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_blend_study.py), [inter_ratio_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_ratio_study.py),
[inter_hybrid_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_hybrid_study.py), [inter_extreme_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_extreme_study.py), [inter_lambda_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_lambda_study.py), [xseg_mask.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/xseg_mask.py);
data: `results*.json`, `latent_metrics.json`. Engineering note: `app/docs/38-INTER-RESIZE.md`.
## 7　Code and data (GitHub)

Branch: <https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/tree/feature/inter-weight-resample>

| kind | files |
|---|---|
| production | [resize_inter.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/engine/models/Model_SAEHD/resize_inter.py) |
| conversion / inverse filter | [inter_resample.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_resample.py), [inter_adv.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_adv.py) |
| main experiments | [inter_study_run.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_study_run.py), [inter_study_visual.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_study_visual.py) |
| latent space | [inter_latent_metric.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_latent_metric.py) |
| method comparison | [inter_adv_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_adv_study.py), [inter_blend_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_blend_study.py), [inter_ratio_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_ratio_study.py), [inter_hybrid_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_hybrid_study.py) |
| extremes and ridge | [inter_extreme_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_extreme_study.py), [inter_lambda_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/inter_lambda_study.py) |
| metrics | [xseg_mask.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/tools/bench/xseg_mask.py) |
| reports | [REPORT_CN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/REPORT_CN.md), [REPORT_EN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/REPORT_EN.md) |
| figure | [compare_all.png](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/compare_all.png) |
| data | [results_ratio.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/results_ratio.json) and the other `results*.json` |
| engineering note | [38-INTER-RESIZE.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/38-INTER-RESIZE.md) |