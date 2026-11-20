# Inter Weight Rescaling Study (df-udt @ 416)

Model: `workspace/model/New416_SAEHD` (df-udt, res=416, e_dims=80, ae_dims=320, iter 7,915,081).
Input: `app/docs/notes/target.jpg` — an **already aligned** face crop (no detection/alignment, resize only).
Forward pass: the repo's own leras (`core.leras.archis.DeepFakeArchi`, opts=`udt`), i.e. the same path the
trainer's preview uses; weights assigned in storage order (`param_N.npy` → `get_weights()`).
Parameter counts match `New416_SAEHD_summary.txt` exactly:
encoder 24,457,280 · inter 51,971,200 · decoder incl. mask head 53,386,207.

## 1. Findings

1. **Preprocessing** must be **BGR, [0,1]** (not [-1,1]). With the wrong scaling the output is noise —
   verified by comparing probe images.
2. **Neither down- nor up-scaling is an equivalence transform**: SSIM 0.30–0.43.
   Structure survives, fine detail does not — consistent with the "necessarily lossy" claim, but the
   damage is smaller than expected (it does not collapse).
3. **Interpolation choice matters less than expected** (SSIM spread ≤0.05 within a target resolution),
   but the trend is readable:
   - 320 (down): nearest has the best SSIM (0.432); bilinear/area lowest (≈0.39);
   - 384 (down): bicubic best (0.349);
   - 448 (up): nearest best SSIM (0.411) but bilinear best PSNR (4.19) and best MAE (137.5);
   - 512 (up): nearest best SSIM (0.409), bilinear best PSNR (3.69).
   No single method wins on every metric. On **MAE** (closest to human perception of overall shift)
   **bilinear wins for upscaling** (448: 137.5).

## 2. Grids and shapes (measured)

| res | low = res/32 | dense1 | dense2 | bias |
|---|---|---|---|---|
| 416 (baseline) | 13 | (108160, 320) | (320, 54080) | (54080,) |
| 320 | 10 | (64000, 320) | (320, 32000) | (32000,) |
| 384 | 12 | (92160, 320) | (320, 46080) | (46080,) |
| 448 | 14 | (125440, 320) | (320, 62720) | (62720,) |
| 512 | 16 | (163840, 320) | (320, 81920) | (81920,) |

Layout (documented in `app/engine/models/Model_SAEHD/resize_inter.py`, implemented here):
dense1 **rows** are the encoder feature map flattened **channel-first (c,h,w)** and re-gridded at
`enc = res/32` (archi has `t`); dense2 **columns** and its **bias** re-gridded at the same rate.
`*.pth` files are zips of `param_N.npy` (no pickle); Dense is stored `(in,out)`, Conv is OIHW.

## 3. Difference vs baseline (all outputs normalised back to 416)

| target | method | PSNR | SSIM | MAE |
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

(Low absolute PSNR is expected: after rescaling the face position/scale changes, so per-pixel error is
large by construction. Compare **within** one target resolution only.)

## 4. Recommendation

- **Upscale (448/512) → bilinear** (best PSNR and MAE).
- **Downscale (320) → bilinear** (lowest MAE); 384 shows slightly better SSIM with bicubic but worse MAE.
- **Do not default to nearest**: only wins on SSIM, clearly worse on MAE (blocky in practice).
- Retrain a while after rescaling: the optimizer state is shape-incompatible and gets dropped
  (Adam moments reset) — same conclusion as the reference note.

## 5. Reproduce

```bash
python app/tools/bench/inter_resample.py --src workspace/model/New416_SAEHD_inter.pth \
    --res 416 --to 320 --method bilinear --out out/inter_320.pth
runtime\engine\python.exe app/tools/bench/inter_study_run.py
```

Artifacts: `app/docs/notes/inter_study/` (per-resolution pth, `base416.png`,
`compare_grid.png`, `results.json`).
