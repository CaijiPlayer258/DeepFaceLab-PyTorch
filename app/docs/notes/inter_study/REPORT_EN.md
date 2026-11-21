# Cross-Resolution Resampling of Inter Weights

Model New416_SAEHD (df-udt, 416, e_dims=80, ae_dims=320). Input is an aligned face, BGR, values in [0,1].
Metrics are computed inside the XSegLite mask, against the output of the 416 original model. The pseudo-inverse
transform, per-tensor grouping and anti-aliasing treatment below are derived and implemented here.

blend means dense1 with inverse filtering and dense2 with its bias under plain interpolation. The two tensors
sit at different places in the graph, so their optimal operators differ; see Proposition 2.

![Fig. 1　dst input, baseline 416 and each method at six target resolutions](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/master/app/docs/notes/inter_study/compare_all.png)

## 1　Propositions and proofs

Notation. The encoder feature map is f ∈ R^(C×n²), flattened channel-first; the first Inter weight field is
W ∈ R^(C n²×A). The grid changes from n to m; the 1-D resampling operator is L ∈ R^(m×n), the 2-D one is L⊗L.
λ is the Tikhonov weight and ρ = (n²−m²)/n².

**Proposition 1 (interpolation lacks the inverse filter)**　Downstream uses only the functional <W,f>.
After rescaling f′ ≈ (L⊗L)f, so

        (L⊗L)ᵀ W′ = W                                                      (1)

with minimum-norm and least-squares solutions

        W′ = (L⊗L)[(L⊗L)ᵀ(L⊗L)]⁻¹W ,   W′ = [(L⊗L)(L⊗L)ᵀ]⁻¹(L⊗L)W        (2)

the first for m ≥ n, the second for m < n.

Proof. (1) is underdetermined for m ≥ n and overdetermined for m < n; the Moore–Penrose solution zeroes or
minimises the residual on the relevant subspace. Separability (L⊗L)† = L†⊗L† allows per-axis solution.
Interpolation gives only (L⊗L)W and misses the factor (LᵀL)⁻¹, a deconvolution of the weight field. That factor
amplifies quantisation noise, so (LᵀL+λI)⁻¹ and (LLᵀ+λI)⁻¹ are used in practice. ∎

**Proposition 2 (per-tensor grouping)**　The optimal operator for dense1 is (2); for dense2 and bias it is
interpolation itself.

        blend:  dense1  <- pinv, inverse filtering; lanczos base up, area anti-aliasing down; λ = 1e-4 / 1e-1
                dense2  <- lanczos, plain interpolation
                bias    <- lanczos, plain interpolation, same grid as dense2                (3)

Proof. dense1 consumes f and emits z = Wᵀf; the preserved quantity is a functional value, hence (2).
dense2 consumes z, which is resolution independent, and emits the field g = W₂ᵀz over the low grid. The decoder
is convolutional and at a new resolution expects the same field sampled on the new grid, g′ = (L⊗L)g, so
W₂′ = (L⊗L)W₂ and interpolation is exact in this sense. Applying (LᵀL)⁻¹ here sharpens the field, which is
neither a resampling of g nor inside the training distribution, and it amplifies aliasing. bias corresponds
position by position to that field and must use the same operator. ∎

**Proposition 3 (downscaling floor)**　For m < n, rank(L⊗L) = m² < n² and the error obeys a bound independent
of the algorithm,

        min ‖(L⊗L)W′ − W‖_F = ‖(I − P)W‖_F ,   P = proj_{range(L⊗L)}       (4)

Proof. The image of any W′ lies in range(L⊗L), so the orthogonal-complement residual cannot be removed. ∎

The discarded fraction ρ sets the ceiling: ρ = 0.15 at 384, 0.41 at 320, 0.52 at 288. Upscaling has ρ = 0
and no such loss.

## 2　Comparison data

Table 1　Masked metrics at six target resolutions, PSNR dB / SSIM / MAE; bold marks the best value in a row.

| target | ρ | bicubic | lanczos | pinv (all tensors) | **blend (ours)** |
|---|---|---|---|---|---|
| 288 | 0.52 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | **18.30 / 0.704 / 21.7** |
| 320 | 0.41 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.07 / 0.716 / 18.1** |
| 384 | 0.15 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **26.33 / 0.909 / 9.26** |
| 448 | 0 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **27.38 / 0.920 / 6.06** |
| 512 | 0 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **24.22 / 0.857 / 9.21** |
| 544 | 0 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **23.02 / 0.830 / 11.2** |

Table 2　Ablation for Proposition 2 at 448, PSNR / SSIM; the other side is held fixed in each pair of columns.

| dense1 only, dense2/bias = bicubic | PSNR / SSIM | dense2/bias only, dense1 = earlier mix | PSNR / SSIM |
|---|---|---|---|
| nearest | 17.52 / 0.645 | nearest | 18.48 / 0.688 |
| bicubic | 23.41 / 0.834 | bicubic | 24.12 / 0.848 |
| lanczos | 23.29 / 0.831 | **lanczos** | **26.84 / 0.905** |
| **pinv** | **24.55 / 0.867** | pinv | 25.22 / 0.883 |

With dense1 on pinv and dense2 on lanczos the scores are 26.33, 27.38 and 24.22 dB at 384, 448 and 512,
above the single-operator figures of 25.10, 26.27 and 23.97 dB. The remaining parameters were checked as well:
area on dense2 costs about 4 dB when downscaling, λ has a plateau between 1e-5 and 1e-4, and adding inverse
filtering to dense2 degrades monotonically at α = 0.15 and 0.30.

## 3　Implementation

Implementation: [resize_inter.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/master/app/engine/models/Model_SAEHD/resize_inter.py), default method blend; `--method` falls back to
bicubic, bilinear, lanczos or pinv. Measured through the tool, 416 to 448: PSNR 27.38, SSIM 0.920, MAE 6.06,
NCC 0.983. A converted model is a warm start only; fine-tune before use, since the optimizer state is dropped
for shape incompatibility.