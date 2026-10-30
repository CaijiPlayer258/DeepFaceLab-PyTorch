# DFL TF → PyTorch 模型转换指南

## 概述

将原版 DeepFaceLab（TensorFlow/lErAs）训练的模型权重转换为本仓库（DeepFaceLab-Torch）可加载的 PyTorch `.pth` 格式。

**核心脚本**：`tools/convert_dfl_tf_to_torch.py`

---

## 权重格式对照

### Conv2D

| 框架 | 权重格式 | 示例 `shape` |
|------|---------|-------------|
| TF (lErAs) | `(kernel, kernel, in_ch, out_ch)` | `(5, 5, 3, 80)` |
| PyTorch | `(out_ch, in_ch, kernel, kernel)` | `(80, 3, 5, 5)` |

**转换**：`np.transpose(weight, (3, 2, 0, 1))`

### Conv2DTranspose

| 框架 | 权重格式 | 示例 |
|------|---------|------|
| TF (lErAs) | `(kernel, kernel, out_ch, in_ch)` | `(4, 4, 416, 512)` |
| PyTorch | `(in_ch, out_ch, kernel, kernel)` | `(416, 512, 4, 4)` |

**转换**：**同一个 transpose** `np.transpose(weight, (3, 2, 0, 1))` 对 Conv2D 和 Conv2DTranspose 都正确！

> 注意：TF lErAs 的 Conv2DTranspose **通道顺序和 Conv2D 相反**（源文件 `Conv2DTranspose.py:43`：`tf.get_variable("weight", (k, k, out_ch, in_ch))`）

### Dense

| 框架 | 权重格式 | 示例 |
|------|---------|------|
| TF (lErAs) | `(in_ch, out_ch)` | `(108160, 320)` |
| PyTorch (本仓库) | `(in_ch, out_ch)` | `(108160, 320)` |

**不需要转置**！本仓库的 `Dense` 层（`core/leras/layers/Dense.py:52`）存储格式与 TF 一致，前向传播中用 `weight.t()` 做 `F.linear`。

---

## SAEHD 架构说明

SAEHD（df / liae）的 `Upscale` 块使用的是 **Conv2D + depth_to_space(pixel_shuffle)**，**不是 Conv2DTranspose**。

- TF 版 `DeepFakeArchi.py:68`：`self.conv1 = nn.Conv2D(in_ch, out_ch*4, ...)` → `nn.depth_to_space(x, 2)`
- PT 版 `DeepFakeArchi.py:77`：`self.conv1 = nn.Conv2D(in_ch, out_ch*4, ...)` → `nn.depth_to_space(x, 2)`

Conv2DTranspose 只出现在 XSeg / GAN PatchDiscriminator 中，不在主模型。

---

## 已修复的关键 Bug

### Bug 1：权重存档顺序错误（最严重）

**症状**：转换后权重全是初始化值，模型输出灰色/随机噪声。

**原因**：`_convert_one_model` 先把 `.npy` 移到 `old/`，再去 `_find_existing_weight_file()` 找源文件——该函数不去 `old/` 找，返回 `None`，导致跳过加载保存初始化权重。

**修复**：把存档移 **到转换之后**（`convert → archive`）。

### Bug 2：名称映射优先级反了

**症状**：权重值虽然复制了但和层不匹配（名字匹配全失败，回退 shape 贪心匹配）。

**原因**：`_collect_tensor_name_maps` 优先用 `get_layers()` 的扁平名（如 `"conv1"`），缺少父级作用域（如 `"down1/conv1"`）。`named_parameters()` 能提供完整层级名但用了 `setdefault` 被覆盖。

**修复**：交换优先级——`named_parameters()` 优先（完整路径 `"down1/conv1/weight:0"`），`get_layers()` 做 `setdefault` fallback。

### Bug 3：build 前调用了 `named_parameters()`

**原因**：`_assign_from_tf_saveable_dict` 在 `get_weights()`（触发 build）**之前**调用 `_collect_tensor_name_maps`，此时 `named_parameters()` 返回空。

**修复**：先 `list(saveable.get_weights())` 再 `_collect_tensor_name_maps()`。

### Bug 4：Conv2D SAME padding 不对称 ⇒ 偏向左上角

**症状**：推理输出偏向左上角、模糊。

**原因**：TF 的 SAME padding 是固定对称的 `(kernel-1)//2` 每边。PT 实现在 stride>1 时计算**最小填充**，奇数总 padding 时变成不对称（如 kernel=5, stride=2 时 TF 用 2+2=4，PT 用 1+2=3），导致 1 像素空间偏移逐层累积。

**修复**：`core/leras/layers/Conv2D.py` 的 `_same_padding` 分支改用 TF 一致的 `pad = (kernel-1)//2` 每边对称填充。

---

## 推理注意事项

1. **输入颜色格式**：DFL 模型训练时图片由 OpenCV 加载，使用 **BGR 格式**，不需要转 RGB。直接 `cv2.imread()` → resize → 归一化 [0,1] → NCHW。

2. **人脸对齐**：DFL 使用 68 点 landmarks + umeyama 仿射变换对齐。推理时需传入已经对齐好的人脸，或用 `LandmarksProcessor.get_transform_mat()` 预处理。

3. **depth_to_space**：本仓库的实现（`core/leras/ops/__init__.py:145`）通过通道重排使 PyTorch 的 `pixel_shuffle` 输出和 TF 的 `depth_to_space` 一致。

---

## 使用方法

```bash
# 转换单个模型
python tools/convert_dfl_tf_to_torch.py \
  --src /path/to/DFL_saved_models \
  --dst /path/to/torch_saved_models \
  --model SAEHD --name MyModel_SAEHD

# 批量转换所有模型（含 XSeg）
python tools/convert_dfl_tf_to_torch.py \
  --src /path/to/DFL_saved_models \
  --dst /path/to/torch_saved_models \
  --all
```

---

## Runtime 加载

`Saveable.load_weights()` 自动检测 TF 格式（key 含 `:0`），调用 `_load_tf_weights`。该函数使用 `_collect_tf_weights` 递归构建层级名（`"down1/conv1/weight:0"`），**已经正确处理了 build 顺序和层级名**。

---

## 版本参考

- 原版 DFL：`iperov/DeepFaceLab`（TF 1.x / lErAs）
- 权重格式确认源文件：`Conv2D.py:61`、`Conv2DTranspose.py:43`、`Dense.py:47`
- 对应 DFL 版本：**2.3.0.1**
