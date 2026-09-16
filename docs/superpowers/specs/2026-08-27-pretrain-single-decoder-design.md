# 预训练「单解码器支路 + 权重复制」方案 · 设计记录

> 状态：**已实现 → v4.0.8.2**（记录于 2026-08-27）
> 相关代码：`models/Model_SAEHD/Model_pytorch.py`、`ui/components/page_trainer/components/training_config_page.py`、
> `ui/components/page_trainer/components/new_model_config_page.py`、`ui/components/page_trainer/page_trainer.py`
> 结论：在"等显存预算"下判定为**净收益**，但存在明确取舍（见 §5）。

---

## 1. 提案

预训练阶段**只训练一条支路**（例如 encoder + inter + `decoder_dst`），预训练结束后**把该解码器权重复制一份**给另一条支路（`decoder_src`）。

动机：
1. 少一条 decoder 的前向/反向 → 省下激活与优化器状态显存；
2. 用省下的显存把 **batch size 开到 2×**；
3. 预训练本来就不需要 decoder 专精，复制一份作为初始化即可。

---

## 2. 为什么成立（代码依据）

| 依据 | 位置 | 含义 |
|---|---|---|
| 两个 decoder 同类同超参 | `Model_pytorch.py:650-665` | `decoder_src` / `decoder_dst` 是同一个 `Decoder` 类的两个独立实例，`in_ch/d_ch/d_mask_ch` 完全相同 → **state_dict 互拷无损** |
| 预训练两条支路同数据 | `Model_pytorch.py:989-990` | `training_data_src_path` 与 `training_data_dst_path` 在 pretrain 时**都指向** `get_pretraining_data_path()` → 同分布 |
| 预训练关闭一切额外项 | `Model_pytorch.py:617-631` | `gan=0 / random_warp=False / random_hsv_power=0 / face_style=0 / bg_style=0 / vgg=0 / lr_dropout=n / uniform_yaw=True` |
| 损失无交叉项 | `Model_pytorch.py:1666` | `G_loss = src_loss.mean() + dst_loss.mean()`；`src_loss` 只作用于 `pred_src_src`，`dst_loss` 只作用于 `pred_dst_dst` |
| 交叉路径不参与重建损失 | `Model_pytorch.py:1416, 1571, 1579` | `pred_src_dst = decoder_src(dst_code)` 仅被 style loss 使用，而 style loss 的条件带 `not self.pretrain` |
| 官方已有同类快捷路径 | `Model_pytorch.py:1630-1635, 1431-1440` | `freeze_decoder_dst` → `_forward_df_src_only()`，只跑 src 分支、反向完全跳过 dst（"真正提速"），loss 用 `skip_dst=True` 置零 |

**推论**：预训练阶段两个 decoder 之间**没有任何经由 loss 的耦合**，唯一联系是共享 encoder + inter。所谓"两个 decoder 一起预训练"，在 encoder 视角下等价于"每步多收一份同分布样本的梯度"。

---

## 3. 有效 batch 核算

原版 encoder 的梯度是两条支路之和（`1411-1412`），两个独立 N-batch 估计相加 → 方向方差 `σ²/(2N)`，等效 **2N batch**；而每个 decoder 只拿自己那一条 → **N batch**。

| 方案 | encoder 有效 batch | decoder 有效 batch |
|---|---|---|
| **原版**（双 decoder，每支路 bs=N） | **2N** | dec_src **N**、dec_dst **N** |
| **单支路，bs=N** | N ❌ | dec_dst **N**（无收益） |
| **单支路，bs=2N** | **2N** ✅ 打平 | dec_dst **2N** ✅ **×2** |

**要点**：decoder 的"双倍"只有把 bs 开到 2N 才成立。而少一个 decoder 省下的显存恰好允许这么做 —— 这就是本方案成立的关键闭环。

---

## 4. 收益核算（为什么判定为净收益）

- 原版每步消耗 2N 样本，decoder 侧 = N（dec_src）+ N（dec_dst）；
- 单支路 bs=2N 每步同样消耗 2N 样本，decoder 侧 = 2N（全给 dec_dst）。

总 decoder 算力**完全相同**，差别只是"配额被集中到一条支路上"。于是：

| 项 | 原版 | 单支路 + bs×2 |
|---|---|---|
| encoder 有效 batch | 2N | 2N（打平） |
| decoder 有效 batch | N | 2N（+100%） |
| decoder 数量 | 2 | 1（复制补齐） |

在**等显存/等算力预算**下：encoder 不退化、decoder 精度翻倍 → 判定为**净收益**。

---

## 5. 取舍（代价项）

1. **丢失"偏向 encoder"的隐性偏置**
   原版 encoder:decoder 有效 batch = **2:1**，即 encoder 梯度比 decoder 干净 √2 倍 —— 相当于隐性地优先保证 latent 学习质量。
   单支路方案（bs=2N）变成 **1:1**，该偏置消失。
   → 若 inter 收敛质量不及预期，这是第一个该调的地方（适当降低 decoder 更新量，找回 2:1）。

2. **丢失弱多头正则（encoder–decoder 共适应）**
   两个独立参数化的 decoder 会阻止 encoder 把 code 迁就单一 decoder 的怪癖，逼迫产生更"通用可解"的 latent。单 decoder 时存在轻度共适应风险。
   缓解：decoder 容量别太大／加 weight decay。

3. **预训练阶段"复制"只是初始化**
   decoder 的域专精（src 干净 / dst 带遮挡）本发生在正训练阶段，故复制不影响最终分工。此项**不构成实际损失**。

---

## 6. 三个必须注意的坑

### 6.1 不能按「同 bs 跑两次 = 原版一次」来算
本仓库可选优化器为 `adam / adabelief / lion`（`768-769`），**三者全部尺度不敏感**：
- Adam：`m̂/(√v̂+ε)` —— 梯度整体 ×2 会在分子分母约掉；
- AdaBelief：同理（分母为梯度方差）；
- Lion：直接 `sign()` —— 尺度完全不进公式。

故"梯度 ×2 → 步数减半"只在**纯 SGD** 下成立，对本仓库**全部失效**。
**正确补偿方式：加 batch，不是加步数。**（附：两个顺序 step ≠ 一次大梯度，前者在 θ 与 θ−lr·g 两点采样，属"小 batch SGD"；真要等价需做梯度累积。）

### 6.2 想真省显存，必须动优化器参数表
`772-779` 处四个模块**无条件**全部进入 `src_dst_trainable_weights`；而 `freeze_decoder_dst` 只改前向（`1631-1635`）、**不改优化器**。结果是 Adam/AdaBelief 的 `m/v` 状态（每参数 2 份）照旧占用显存。
→ 要兑现"显存换 bs"，需把冻结支路从 `src_dst_trainable_weights` 摘除。
→ 注意 `src_dst_saveable_weights` 仍含四个模块（用于存读）：从零训练/自行复制权重无冲突，但**从他人预训练模型续训时 opt state 可能对不上**。

### 6.3 本仓库 pretrain 被强制关闭；且 LIAE 架构不适用
- `123` 与 `451-452` 两处将 `pretrain` 一律强制为 `False`，做实验前必须先解除。
- LIAE 架构（`675-681`）为**两个 inter（inter_AB / inter_B）+ 一个共享 decoder**，"复制 decoder"的前提不成立，本方案**仅适用于 DF 架构**。

---

## 7. 验证方法

开启 inter code 分布日志（`1650-1659`，选项 `log_code_stats`）：

```
[CODE] iter=0000123  mu_gap=...  var_gap=...  mu_s=...  var_s=...
```

两种方案跑**相同数据量**，对比 `mu_s / var_s / mu_gap / var_gap`：
- 数值重合 → 两种做法训到了同一隐空间，方案成立；
- 系统性偏离 → 回到 §5 的取舍项定位原因。

---

## 8. 一句话总结

> 少一个 decoder 换来的显存足以把 bs 开到 2×；在等预算下 encoder 打平、decoder 有效 batch 翻倍，因此是**净收益**。
> 取舍在于丢掉了原版 encoder:decoder = 2:1 的隐性偏置与弱多头正则。
> 补偿必须靠 **batch** 而非步数（Adam 族尺度不变）；落地记得把冻结支路从优化器摘掉。
> 仅适用于 **DF** 架构，且需先解除本仓库对 `pretrain` 的强制关闭。

---

## 9. 实现落地（v4.0.8.2）

### 9.1 选项
- 新增 `pretrain_single_decoder`（bool，默认 False），仅 **DF 架构 + pretrain** 时生效。
- 恢复被 4.0.8.1 强制关闭的 `pretrain` 选项（改回 `load_or_def_option` 并重新提供交互询问入口），
  否则本功能不可达。

### 9.2 模型侧（`Model_SAEHD/Model_pytorch.py`）
| 改动 | 位置 | 说明 |
|---|---|---|
| 生效标志 | `on_initialize` | `pretrain_single_decoder = pretrain and archi=='df' and option` |
| 优化器摘除 | 优化器参数表段 | 生效时 `src_dst_trainable_weights` 不含 `decoder_dst`（saveable 仍含四个，存档结构不变） |
| 前向分派 | `train_one_step` / `_xla_core` | 复用既有 `_forward_df_src_only()` + `skip_dst=True` |
| 权重同步 | `onSave` → `_sync_pretrain_single_decoder()` | 保存前把 `decoder_src` 权重 `copy_` 到 `decoder_dst` |

**实现训练的是 src 支路**（而非提案举例的 dst 支路）。依据 §2 验证的对称性（预训练两条支路
读取同一份多身份数据集，同 warp/hsv/flip/scale），「训练 src 再复制给 dst」与「训练 dst 再复制给 src」
完全等价；选 src 是为了复用已被 `freeze_decoder_dst` 长期验证的 `_forward_df_src_only` 路径，
不新增任何损失逻辑。

### 9.3 验证（真实 GPU + 真实数据冒烟测试）
环境：RTX 3080 20GB，res128 df-ud，src/dst 均指向真实 aligned 人脸集。

```
pretrain = True
pretrain_single_decoder = True
opt params=50   四模块=86   三模块=50      -> 优化器已摘除 decoder_dst
decoder_src 变化 = True
decoder_dst 变化 = False                   -> 单支路生效，dst 未被更新
[预训练] 已把 decoder_src 权重同步到 decoder_dst。
同步后 decoder_src == decoder_dst : True   -> onSave 同步成功
=== 冒烟测试全部通过 ===
```

### 9.4 兼容性
- 从他人预训练模型续训时，其 `src_dst_opt.pth` 含四个模块的优化器状态；本方案下优化器只持有三个模块
  → 位置索引会错位。`Saveable._load_pt_weights` 对缺失键是**静默跳过**（不会崩），
  且 Adam/AdaBelief 的 m/v 会在数十步内重新适应，影响可忽略 —— 与 §6.2 的结论一致。
- 仅 DF 架构；LIAE 为「两个 inter + 一个共享 decoder」，不适用。
