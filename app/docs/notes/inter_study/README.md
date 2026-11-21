# inter_study · Inter 权重跨分辨率重采样

本目录是「把已训练模型改成分辨率」这一功能的完整实验记录：报告、数据、图、脚本。

- 正式报告：[REPORT_CN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_CN.md) （学术体例）
- English：[REPORT_EN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_EN.md) 
- 工程说明：[app/docs/38-INTER-RESIZE.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/38-INTER-RESIZE.md)

## 结论速览

| 目标 | 网格 | bicubic | lanczos | pinv | **blend（默认）** |
|---|---|---|---|---|---|
| 288 | 13→9 | 18.20 / 0.728 / 25.7 | 17.87 / 0.723 / 27.3 | 16.21 / 0.705 / 28.7 | **18.30 / 0.704 / 21.7** |
| 320 | 13→10 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | 19.10 / 0.714 / 17.6 | **20.07 / 0.716 / 18.1** |
| 384 | 13→12 | 23.32 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | 23.62 / 0.856 / 13.8 | **26.33 / 0.909 / 9.26** |
| 448 | 13→14 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | 24.78 / 0.892 / 10.8 | **27.38 / 0.920 / 6.06** |
| 512 | 13→16 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | 20.83 / 0.838 / 20.1 | **24.22 / 0.857 / 9.21** |
| 544 | 13→17 | 18.46 / 0.774 / 24.5 | 18.88 / 0.808 / 23.9 | 19.35 / 0.810 / 24.4 | **23.02 / 0.830 / 11.2** |

（PSNR dB / SSIM / MAE，XSegLite 遮罩内，416 原模型输出为基准）

三句话：

1. 插值族里 **lanczos-3** 最好，bicubic 可用，nearest 只在 ±1 格时「看着锐」（相位错，不算更准）。
2. blend 为 dense1 取伪逆、dense2 与 bias 取 lanczos；降采样时伪逆基座改用 area 抗混叠。
   六个目标上均为最优，已设为默认，448 上比单一算子高 1.1 dB。
3. 降采样存在秩亏下界，丢弃维度比例为 15%、41% 与 52%，重采样无法突破，只能续训；
   实用边界是降分辨率不超过 1 至 2 个网格步长，约 32 至 64 px。

## 文件地图

| 文件 | 内容 |
|---|---|
| [REPORT_CN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_CN.md) /  | 中文报告（摘要 → 推导 → 实验 → 结论 → 可复现） |
| [REPORT_EN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_EN.md) /  | 英文版 |
| [compare_all.png](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/compare_all.png) | 总图：dst 输入 / baseline / 5 方法 × 6 档目标分辨率 |
| [results.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results.json) | 16 组（bilinear 系列 × 4 分辨率） |
| [results_adv.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results_adv.json) | 48 组（插值 / pinv / ls 修正） |
| [results_ratio.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results_ratio.json) | 32 组（核 × 逆滤波 × 抗混叠） |
| [results_hybrid.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results_hybrid.json) | AA 感知混合候选 |
| [results_extreme.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results_extreme.json) | 288 / 544（±128） |
| [results_lambda288.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/results_lambda288.json) | 288 上 λ 扫描（0.1 → 10） |
| [latent_metrics.json](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/latent_metrics.json) | 潜空间（320 维码）对照 |

图目录：`cand/`（旧混合）、`cand2/`（核对比）、`cand3/`（AA 混合）、`cand128/`（极端比值）。

## 复现

```
# 单次转换（生产工具，默认 blend）
python -m models.Model_SAEHD.resize_inter --model-dir <模型目录> --new-res 448

# 全量指标（需要 runtime\engine 的 python）
runtime\engine\python.exe app/tools/bench/inter_ratio_study.py    # 32 组核/逆滤波/抗混叠
runtime\engine\python.exe app/tools/bench/inter_extreme_study.py  # 288 / 544
runtime\engine\python.exe app/tools/bench/inter_hybrid_study.py   # AA 混合
runtime\engine\python.exe app/tools/bench/inter_lambda_study.py   # 288 λ 扫描
```

脚本集合：`app/tools/bench/inter_resample.py`（转换）、[inter_adv.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_adv.py)（插值矩阵/pinv/ls）、
[xseg_mask.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/xseg_mask.py)（掩码内指标 + ncc/hf）、[inter_study_run.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_study_run.py)、[inter_adv_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_adv_study.py)、
[inter_blend_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_blend_study.py)、[inter_ratio_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_ratio_study.py)、[inter_hybrid_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_hybrid_study.py)、[inter_extreme_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_extreme_study.py)、
[inter_lambda_study.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/tools/bench/inter_lambda_study.py)。

## 注意

- 指标都在 **XSegLite 遮罩内**统计；整图指标会被背景拉平，不要混用。
- 参照是 **baseline 416**（原 Inter 输出），不是 dst 输入 —— 该模型 dst 侧本身有风格偏移。
- 转换只是热启动：`data.dat` 的 optimizer state 形状不兼容会被丢弃，投产前需续训。
## 仓库链接

- 分支：<https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/tree/4.1-beta>
- 生产实现：[resize_inter.py](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/engine/models/Model_SAEHD/resize_inter.py)
- 工程说明：[38-INTER-RESIZE.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/38-INTER-RESIZE.md)
- 报告：[REPORT_CN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_CN.md) ／ [REPORT_EN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/4.1-beta/app/docs/notes/inter_study/REPORT_EN.md)