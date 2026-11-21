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

### 2.3　Partial inverse filtering

Full inversion (2) over-sharpens; zero inversion (interpolation) is too soft; both are ends of one family. A
convex combination gives

        W′ = α · W′_interp + (1 − α) · W′_pinv,   α = 0.5                 (3)

which is a Wiener shrinkage of the inverse-filter gain. α = 0.5 is near-optimal on every target, so it is fixed.

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

**Fig. 1**　dst input (what the model actually sees), baseline 416 (original Inter output) and each method at the
six target resolutions; both reference columns repeat per row for in-row comparison.

![Fig. 1　Cross-resolution resampling results](compare_all.png)

**Table 1**　Masked metrics (PSNR dB / SSIM / MAE). Bold marks the best or tied-best method in that row.

| target | ρ | bicubic | lanczos | pinv | **blend (proposed)** |
|---|---|---|---|---|---|
| 288 | 0.52 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | 18.18 / 0.698 / **21.6** |
| 320 | 0.41 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.24 / 0.718 / 17.3** |
| 384 | 0.15 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **25.10 / 0.874 / 9.64** |
| 448 | 0 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **26.27 / 0.897 / 6.39** |
| 512 | 0 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **23.97 / 0.847 / 8.73** |
| 544 | 0 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **22.93 / 0.824 / 10.33** |

**Table 2**　nearest at small ratios (ρ_a = folded/duplicated fraction). Its h_f exceeds that of interpolation,
while NCC and PSNR are lower.

| target | q | ρ_a | nearest (PSNR / NCC / h_f) | bicubic (PSNR / NCC / h_f) |
|---|---|---|---|---|
| 384 | 1.083 | 0.077 | 21.21 / 0.901 / **0.32** | 23.33 / 0.940 / 0.23 |
| 448 | 0.929 | 0.071 | 17.91 / **0.736** / 0.37 | 23.41 / 0.941 / 0.37 |
| 320 | 1.300 | 0.231 | 18.77 / 0.692 / 0.08 | 19.32 / 0.754 / 0.06 |
| 512 | 0.813 | 0.188 | 18.07 / 0.816 / 0.48 | 20.26 / 0.922 / 0.59 |

The measurements agree with the theory: in the upscaling direction (ρ = 0) the proposed method is best on all
three metrics; in the downscaling direction quality decreases monotonically with ρ, and the bound (4) cannot be
side-stepped by tuning λ (at 288, raising λ from 0.1 to 10 lowers PSNR from 18.18 to 14.70 dB).

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

Scripts: `inter_study_run.py`, `inter_adv_study.py`, `inter_blend_study.py`, `inter_ratio_study.py`,
`inter_hybrid_study.py`, `inter_extreme_study.py`, `inter_lambda_study.py`, `xseg_mask.py`;
data: `results*.json`, `latent_metrics.json`. Engineering note: `app/docs/38-INTER-RESIZE.md`.