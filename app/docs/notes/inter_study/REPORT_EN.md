# Cross-Resolution Resampling of Inter Weights

Model `New416_SAEHD` (df-udt, 416, e_dims=80, ae_dims=320); aligned face input (BGR, [0,1]); all metrics
inside the XSegLite mask, baseline = the 416 original model. Everything below (pseudo-inverse transform,
per-tensor grouping, anti-aliasing) is derived and implemented independently in this repository.

![Fig. 1　dst input / baseline 416 / each method at six target resolutions](compare_all.png)

## 1　Propositions and proofs

**Notation**　encoder feature map f ∈ R^(C×n²) (channel-first), Inter weight field W ∈ R^(C n²×A);
grid n → m; 1-D resampling operator L ∈ R^(m×n), 2-D is L⊗L; λ Tikhonov weight, ρ = (n²−m²)/n².

**Proposition 1 (interpolation lacks the inverse filter)**　Downstream only sees the functional <W,f>;
after rescaling f′ ≈ (L⊗L)f, hence

        (L⊗L)ᵀ W′ = W                                                      (1)

with minimum-norm solution (m ≥ n) and least-squares solution (m < n)

        W′ = (L⊗L)[(L⊗L)ᵀ(L⊗L)]⁻¹W ,   W′ = [(L⊗L)(L⊗L)ᵀ]⁻¹(L⊗L)W        (2)

*Proof*　(1) is underdetermined for m ≥ n and overdetermined for m < n; the Moore–Penrose solution zeroes or
minimises the residual on the relevant subspace, and separability (L⊗L)† = L†⊗L† allows per-axis solution.
Interpolation only gives (L⊗L)W and misses the factor (LᵀL)⁻¹ (a deconvolution of the weight field), which
amplifies quantisation noise; in practice (LᵀL+λI)⁻¹ and (LLᵀ+λI)⁻¹. ∎

**Proposition 2 (per-tensor grouping)**　dense1 wants (2), dense2/bias wants interpolation itself:

        blend = dense1 : pinv (lanczos base up / area anti-aliased down, λ = 1e-4 / 1e-1)
                dense2, bias : lanczos interpolation                          (3)

*Proof*　dense1 consumes f and emits z = Wᵀf, preserving a functional value → (2). dense2 consumes z
(resolution independent) and emits the field g = W₂ᵀz on the low grid; the decoder is convolutional, so it
expects the same field sampled on the new grid, g′ = (L⊗L)g, i.e. W₂′ = (L⊗L)W₂ — interpolation itself.
Applying (LᵀL)⁻¹ here sharpens the field, moves it off the training distribution and amplifies aliasing.
bias is a per-position offset of that field and must use the same operator. ∎

**Proposition 3 (downscaling floor)**　For m < n, rank(L⊗L) = m² < n² and

        min ‖(L⊗L)W′ − W‖_F = ‖(I − P)W‖_F ,   P = proj_{range(L⊗L)}       (4)

*Proof*　The image of any W′ lies in range(L⊗L); the orthogonal-complement residual cannot be removed. ∎
The discarded fraction bounds quality: ρ = 0.15 (384), 0.41 (320), 0.52 (288); upscaling has ρ = 0.

## 2　Comparison data

**Table 1**　Masked metrics at six target resolutions (PSNR dB / SSIM / MAE); bold = best per row.

| target | ρ | bicubic | lanczos | pinv (all tensors) | **blend (ours)** |
|---|---|---|---|---|---|
| 288 | 0.52 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | **18.30 / 0.704 / 21.7** |
| 320 | 0.41 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.07 / 0.716 / 18.1** |
| 384 | 0.15 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **26.33 / 0.909 / 9.26** |
| 448 | 0 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **27.38 / 0.920 / 6.06** |
| 512 | 0 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **24.22 / 0.857 / 9.21** |
| 544 | 0 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **23.02 / 0.830 / 11.2** |

**Table 2**　Ablation for Proposition 2 (448, PSNR / SSIM), the other side held fixed.

| dense1 only (dense2/bias = bicubic) | PSNR / SSIM | dense2/bias only (dense1 = earlier mix) | PSNR / SSIM |
|---|---|---|---|
| nearest | 17.52 / 0.645 | nearest | 18.48 / 0.688 |
| bicubic | 23.41 / 0.834 | bicubic | 24.12 / 0.848 |
| lanczos | 23.29 / 0.831 | **lanczos** | **26.84 / 0.905** |
| **pinv** | **24.55 / 0.867** | pinv | 25.22 / 0.883 |

Grouped: dense1 = pinv with dense2 = lanczos gives 26.33 / 27.38 / 24.22 dB at 384 / 448 / 512,
above the earlier one-operator-for-all 25.10 / 26.27 / 23.97 dB. Remaining knobs confirmed: area on dense2
costs 4 dB when downscaling, λ has a plateau over 1e-5..1e-4, and adding inverse filtering to dense2
(α = 0.15 / 0.30) degrades monotonically.

## 3　Implementation

`app/engine/models/Model_SAEHD/resize_inter.py`, default method `blend` (eq. 3); `--method` falls back to
`bicubic / bilinear / lanczos / pinv`. Tool measurement (416→448): PSNR 27.38 / SSIM 0.920 / MAE 6.06 /
NCC 0.983. Conversion is a warm start only — fine-tune before use (optimizer state is dropped).