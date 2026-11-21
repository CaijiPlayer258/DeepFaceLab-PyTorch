# 38 · Inter 分辨率转换（模型改分辨率）

> 状态：生产默认 `blend`（按张量分组：dense1 = 伪逆，dense2/bias = lanczos 插值），2026-11 定稿。
> 实验与数据见 [inter_study/REPORT_CN.md](notes/inter_study/REPORT_CN.md)。

## 1. 它做什么

把一个**已训练**模型的 Inter 权重换到另一个分辨率，Encoder / Decoder 不动（纯卷积，分辨率无关）：

```
python -m models.Model_SAEHD.resize_inter --model-dir <模型目录> --new-res 448
#   --method blend|bicubic|bilinear|lanczos|pinv   （默认 blend）
#   --dry-run  只打印计划，不写盘
```

写盘规则：先 `*.bak.<时间戳>` 备份 → 写临时文件 → 形状校验 → `os.replace`；任一步失败整体回滚。
`data.dat` 只改 `options['resolution']`，并追加只读的 `_resize_history` 审计记录。

## 2. 网格语义（必须按引擎真实展平解析）

```
enc = res // 16   （archi 含 t 时 res // 32）   dense1 的输入网格
low = res // 32   （无 d 时 res // 16）          dense2 的输出网格

dense1.weight (e_dims*8*enc^2, ae_dims)   行 = c*H*W 通道优先
dense2.weight (ae_dims, ae_out*low^2)     列 = 同一展平次序
dense2.bias   (ae_out*low^2,)             同上（按 (h,w,c) 解析会因轴序错位而花屏）
```

`New416_SAEHD`（df-udt，416，e_dims=80，ae_dims=320）：enc = low = 13，dense1 `(108160, 320)`。

## 3. 方法：从插值到部分逆滤波

> 本节算法（伪逆变换、按张量分组、降采样抗混叠）由本项目独立推导与实现，不依赖外部实现的参数或经验。
>
> **按张量分组（关键）**：dense1 需要保泛函 → 用伪逆 (L⊗L)ᵀW'=W；dense2/bias 是**输出场重采样**，
> 直接用 lanczos 插值，做逆滤波会过锐化并把混叠放大。448 上分组比"一套算子通吃"高 1.1 dB（26.27 → 27.38）。

| 方法 | 做什么 | 何时用 |
|---|---|---|
| `nearest` | 只取样 | 几乎不用（混叠） |
| `bilinear` | 三角核插值 | 兼容旧行为 |
| `bicubic` | Catmull-Rom 插值 | 想稳、不想动时 |
| `lanczos` | Lanczos-3 插值 | 纯插值里最好（448：25.4 dB） |
| `pinv` | 保泛函伪逆（逆滤波） | 升分辨率单独用也不错 |
| **`blend`（默认）** | 按张量分组：**dense1 = pinn 逆滤波**（升 lanczos 基座 / 降 area 抗混叠），**dense2/bias = lanczos 插值** | 通用最优 |

### 为什么是这三个要素

1. **插值 = 低通**。要保的是线性泛函 ⟨W, f⟩，换分辨率后 f' ≈ (L⊗L)f，因此需要
   `(L⊗L)ᵀW' = W`。插值只算了 `(L⊗L)W`，缺一个逆滤波因子（锐化）。
2. **逆滤波是那个缺失项**：升采样取最小范数解 `L(LᵀL+λI)⁻¹`，降采样取最小二乘解
   `(LLᵀ+λI)⁻¹L`（λ 抑制噪声放大）。
3. **全量逆滤波会过锐**，所以取一半（Wiener 收缩，`α=0.5`）。
4. **降采样必须先抗混叠**：`pinv` 那一支用 box（`area`）核，否则混叠被一起放大
   （320 上 MAE 17.0 → 16.5，NCC 0.68 → 0.78）。

## 4. 实测（416 原模型 → 目标分辨率，XSegLite 遮罩内，与 416 baseline 比）

| 目标 | bicubic | lanczos | **blend（默认）** |
|---|---|---|---|
| 320 | 19.32 / 0.740 / 23.1 | 19.29 / 0.735 / 23.2 | **20.07 / 0.716 / 18.1** |
| 384 | 23.33 / 0.813 / 13.4 | 24.32 / 0.842 / 11.8 | **26.33 / 0.909 / 9.26** |
| 448 | 23.41 / 0.834 / 8.8 | 25.41 / 0.883 / 7.5 | **27.38 / 0.920 / 6.06** |
| 512 | 20.26 / 0.814 / 18.8 | 20.71 / 0.836 / 18.5 | **24.22 / 0.857 / 9.21** |

（PSNR / SSIM / MAE）。生产路径实测与实验一致：`--new-res 448` → 27.38 / 0.920 / 6.06，NCC 0.983。

## 5. 两个必须记住的边界

1. **换分辨率 ≠ 等价变换**，只能当热启动。要出片质量需续训；优化器状态形状不兼容会被丢弃
   （Adam 动量归零）。
2. **降分辨率有数学下界**。网格 n²→m² 是秩亏投影：169→100（320）丢 41%，169→144（384）丢 15%，
   丢的是正交补空间，低分辨率编码器根本不产生那些频率，任何插值/混合都无法挽回；
   升采样没有这个损失（m²>n²），所以指标明显更高。

## 6. 相关实现

```
app/engine/models/Model_SAEHD/resize_inter.py   生产工具（_interp_matrix / _blend_weight）
app/tools/bench/inter_adv.py                    实验：插值矩阵 / pinv / ls 修正
app/tools/bench/inter_ratio_study.py            32 组：核 × 逆滤波 × 抗混叠
app/tools/bench/inter_hybrid_study.py           AA 感知混合候选
app/tools/bench/xseg_mask.py                    XSegLite 遮罩内指标 + ncc / hf_ratio
```

注意：`--method` 的 **argparse 默认值**与 `_METHODS` 是两处，改默认值必须同时改，
并用工具跑一次、比对「实验 vs 工具」数值（曾因漏改 silent 回落到 bilinear）。
## 7. 仓库链接

- 分支：<https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/tree/feature/inter-weight-resample>
- 实验目录：[app/docs/notes/inter_study](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/README.md)
- 报告：[REPORT_CN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/REPORT_CN.md) ／ [REPORT_EN.md](https://github.com/CaijiPlayer258/DeepFaceLab-PyTorch/blob/feature/inter-weight-resample/app/docs/notes/inter_study/REPORT_EN.md)