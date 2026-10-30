# XSeg — Face Segmentation Model (PyTorch)

PyTorch reimplementation of DeepFaceLab's original TensorFlow XSeg, with weight-compatible ONNX export.

## TF → PyTorch / ONNX 导出注意事项

### 1. UpConvBlock padding 差异（关键！）

TF 原始实现的 `Conv2DTranspose` 使用 `padding='VALID'` + 手动裁剪 1 像素：

```python
# TF 原始行为
x = tf.nn.conv2d_transpose(x, ..., padding='VALID')
x = x[:, :, :-1, :-1]  # crop
```

PyTorch 实现必须**严格匹配**这个行为，不能直接用 `padding='SAME'`：

```python
# ✅ 正确（匹配 TF）
self.conv = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=3, padding='VALID')

def forward(self, x):
    x = self.conv(x)
    x = x[:, :, :-1, :-1]  # 必须裁剪
    x = self.frn(x)
    x = self.tlu(x)
    return x
```

`SAME` 和 `VALID+crop` 输出尺寸相同，但**卷积核对齐位置不同**，会导致整个解码路径像素偏移。若不匹配此行为，导出的 ONNX 遮罩与 TF 原始版有 ~14% 的像素差异。

### 2. 权重格式转换

`.npy` 权重文件是 pickle 格式的 Python dict，键名为 TF 风格（`conv01/conv/weight:0`），值 layout 为 TF 原生：

| 层类型 | TF 格式 | → PyTorch 格式 | 转置操作 |
|--------|---------|----------------|----------|
| Conv2D weight | `(K, K, in_ch, out_ch)` HWIO | `(out_ch, in_ch, K, K)` OIHW | `transpose(3,2,0,1)` |
| Conv2DTranspose weight | `(K, K, out_ch, in_ch)` HWOI | `(in_ch, out_ch, K, K)` IOHW | `transpose(3,2,0,1)` |
| Dense weight | `(in_dim, out_dim)` | `(in_dim, out_dim)` | 直接匹配（`.t()`） |
| Bias / FRN / TLU param | `(ch,)` | `(ch,)` | 直接匹配 |

权重转换逻辑见 `facelib/XSegNet.py` → `_load_model_weights_compat()` 和 `core/leras/models/ModelBase.py` → `_collect_tf_weights()`。

### 3. 参数收集完整性

`ModelBase._collect_tf_weights()` 必须收集**全部**参数属性：

```python
# 必须包含 eps (FRNorm2D) 和 tau (TLU)，否则 TF→Torch 映射不完整
for attr in ("weight", "bias", "running_mean", "running_var", "eps", "tau"):
    ...
```

XSeg 模型共 **222 个可训练参数 + 6 个 BlurPool buffer**（`bp0.k` ~ `bp5.k`）。TF `.npy` 文件有 222 个键，必须全部匹配。

### 4. ONNX 导出

```bash
# 从 .npy 加载权重 → PyTorch model → 导出 ONNX
python models/Model_XSeg/export_XSeg.py
```

- **输入**: NCHW `(batch, 3, 256, 256)`，float32，值域 [0, 1]
- **输出**: NCHW `(batch, 1, 256, 256)`，float32 → sigmoid(logits)，值域 [0, 1]
- **opset**: 18
- **动态 batch**: 是

### 5. 输入图片格式

模型接受 **BGR** 图片（OpenCV 原生格式），无需转 RGB。归一化到 [0, 1] 后送入 NCHW / NHWC tensor（取决于 ONNX 导出时的布局）。

### 6. 已知参考实现

桌面验证项目 `XSegNet2onnx-main` 使用 TF → tf2onnx 路径导出，其 ONNX 为 NHWC 格式。我们的 PyTorch 路径导出为 NCHW 格式，`DataAugmenter` 的 `_create_session()` 会自动检测输入布局（参考 `XSegAugmenter.py`）。
