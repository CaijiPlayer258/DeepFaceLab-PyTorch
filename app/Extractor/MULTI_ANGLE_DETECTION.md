# 多角度人脸检测功能使用说明

## 功能概述

Extractor.py 新增了**多角度旋转检测**功能，可以从多个角度（0°, 90°, 180°, 270°）检测人脸，提高侧脸、倒置人脸等特殊情况下的检测召回率。

## 核心特性

### 1. 多角度检测
- **工作原理**: 将输入图像旋转到指定角度，分别进行人脸检测
- **坐标转换**: 自动将检测到的人脸框坐标转换回原始图像坐标系
- **去重处理**: 使用 IoU (Intersection over Union) 算法去除重复检测的人脸框
- **智能 landmark**: 根据检测角度自动旋转图像，让特征点标记器更好地工作

### 2. 快速测试模式
- **仅处理第一张图片**: 快速验证参数配置
- **完整可视化**: 生成各阶段的调试图像
- **详细日志**: 输出每个步骤的详细信息

## 使用方法

### 命令行参数

#### `-a, --angles` - 检测角度
```bash
# 单角度检测（默认）
python Extractor.py -i "input" -o "output" -a "0"

# 多角度检测
python Extractor.py -i "input" -o "output" -a "0,90,180,270"

# 只检测 0° 和 90°
python Extractor.py -i "input" -o "output" -a "0,90"
```

**支持的角度**: 0, 90, 180, 270（顺时针旋转）

#### `--quick-test` - 快速测试模式
```bash
# 只处理第一张图片，并生成可视化调试图
python Extractor.py -i "input" -o "output" -a "0,90,180,270" --quick-test
```

### 完整示例

#### 示例 1: 标准多角度检测
```bash
python Extractor.py \
  -i "workspace/data_src/aligned" \
  -o "workspace/data_dst/aligned" \
  -d BlazeFace \
  -l insightface-2d106det \
  -a "0,90,180,270" \
  -r 1920
```

#### 示例 2: 快速测试
```bash
python Extractor.py \
  -i "test_images" \
  -o "test_output" \
  -d BlazeFace \
  -l insightface-2d106det \
  -a "0,90,180,270" \
  --quick-test
```

#### 示例 3: 交互式模式
```bash
python Extractor.py
# 按提示输入参数，在 angles 处输入: 0,90,180,270
# 启用 quick-test: y
```

## 可视化调试输出

使用 `--quick-test` 模式后，会在输出目录的 `debug/` 文件夹中生成以下图片：

### Stage 1: 人脸检测结果
**文件名**: `00000_stage1_detection.png`

- 显示预缩放后的图像
- 用不同颜色标注不同角度的检测结果：
  - 🟢 绿色: 0° 检测
  - 🔵 蓝色: 90° 检测
  - 🔴 红色: 180° 检测
  - 🟡 黄色: 270° 检测
- 标注检测角度

### Stage 2: 特征点标记
**文件名**: `00000_stage2_landmarks.png`

- 在检测框基础上显示 68 个特征点
- 每 10 个点标注序号
- 保持与 Stage 1 相同的颜色编码

### Stage 3: 原图上的检测结果
**文件名**: `00000_stage3_original_with_boxes.png`

- 在原始分辨率图像上显示检测框
- 显示映射到原图的特征点
- 白色多边形显示实际提取区域
- 标注每个人脸的索引和检测角度

### 单独的对齐人脸
**文件名**: 
- `00000_face_0_aligned.png` - 对齐后的人脸
- `00000_face_0_aligned_lm.png` - 带特征点的对齐人脸

## 技术细节

### 坐标转换逻辑

#### 90° 旋转
```python
# 旋转后: (x, y) -> (y, w-x)
# 转回原图: (x', y') -> (h-y', x')
orig_l = t
orig_t = w - r
orig_r = b
orig_b = w - l
```

#### 180° 旋转
```python
# 旋转后: (x, y) -> (w-x, h-y)
# 转回原图: (x', y') -> (w-x', h-y')
orig_l = w - r
orig_t = h - b
orig_r = w - l
orig_b = h - t
```

#### 270° 旋转
```python
# 旋转后: (x, y) -> (h-y, x)
# 转回原图: (x', y') -> (y', h-x')
orig_l = h - b
orig_t = l
orig_r = h - t
orig_b = r
```

### 去重算法

使用 **IoU (Intersection over Union)** 判断重复检测：

```python
IoU = 交集面积 / 并集面积

if IoU > 0.5:
    # 认为是同一个人脸，保留面积较大的那个
    移除重复检测
```

**策略**: 
1. 按检测框面积从大到小排序
2. 优先保留大面积的检测框
3. IoU > 0.5 视为重复

### Landmark 旋转

对于非 0° 角度检测到的人脸：

1. **旋转图像**: 将工作图像旋转到检测角度
2. **提取 landmark**: 在旋转后的图像上提取特征点
3. **反向旋转**: 将 landmark 坐标转回原始方向
4. **正常对齐**: 使用转换后的 landmark 进行人脸对齐

## 性能影响

### 检测速度
- **单角度 (0°)**: 基准速度 1x
- **双角度 (0°, 90°)**: 约 2x 时间
- **四角度 (0°, 90°, 180°, 270°)**: 约 4x 时间

### 内存占用
- 临时存储旋转图像，额外内存开销 < 10%

### 建议配置

| 场景 | 推荐角度 | 说明 |
|------|---------|------|
| 正常视频/图片 | `0` | 最快，适合大多数情况 |
| 有侧脸的情况 | `0,90` | 平衡速度和召回率 |
| 复杂场景 | `0,90,180,270` | 最高召回率，速度最慢 |
| 特殊角度拍摄 | `0,180` | 针对倒置视频 |

## 常见问题

### Q1: 为什么需要多角度检测？
A: 传统人脸检测器对正脸效果好，但对侧脸、倒置脸效果差。多角度检测可以显著提高这些情况下的检测率。

### Q2: 会影响最终的人脸质量吗？
A: 不会。多角度检测只是增加检测召回率，最终的人脸对齐仍然在原始图像上进行，质量不受影响。

### Q3: 去重会误删真实人脸吗？
A: IoU 阈值设为 0.5，通常不会误删。如果两个人脸非常接近（IoU > 0.5），它们很可能是同一个脸的重复检测。

### Q4: Quick-test 模式能用于生产吗？
A: 不能。Quick-test 只处理第一张图片，仅用于测试和调试。生产环境请去掉 `--quick-test` 参数。

### Q5: 如何选择合适的角度组合？
A: 
- 先试 `0,90`，观察 debug 图像
- 如果还有漏检，再加 `180,270`
- 如果 0° 已经很好，就用单角度

## 调试技巧

### 1. 检查检测覆盖率
```bash
# 运行 quick-test
python Extractor.py -i "test" -o "out" -a "0,90,180,270" --quick-test

# 查看 debug 图像
# - Stage 1: 看是否有漏检
# - Stage 3: 看提取区域是否合理
```

### 2. 对比不同角度
```bash
# 分别测试单角度
python Extractor.py -i "test" -o "out_0" -a "0" --quick-test
python Extractor.py -i "test" -o "out_90" -a "90" --quick-test

# 对比两个输出的 debug 图像
```

### 3. 调整 IoU 阈值
如果需要更严格或更宽松的去重，修改代码中的 `iou_threshold` 参数：

```python
# 在 detect_faces_multi_angle 函数中
unique_detections = remove_duplicate_detections(
    all_detections, 
    iou_threshold=0.5  # 调大更严格，调小更宽松
)
```

## 示例输出

```
====================================================================================================================
███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗ ██████╗ ██████╗     ████████╗ ██████╗  ██████╗ ██╗     
...
====================================================================================================================

Configuration:
  Input Path: test_images
  Output Path: test_output
  Detector: BlazeFace
  Landmarker: insightface-2d106det
  Output Size: Dynamic based on face bbox
  Pre-resize: 1920px
  Detection Angles: [0, 90, 180, 270]
  Quick Test: True

[DEBUG] Saved stage 1 (detection): test_output/debug/00000_stage1_detection.png
[DEBUG] Saved stage 2 (landmarks): test_output/debug/00000_stage2_landmarks.png
[DEBUG] Saved stage 3 (original with boxes): test_output/debug/00000_stage3_original_with_boxes.png
[DEBUG] Total faces detected: 3
[DEBUG] Debug images saved to: test_output/debug

Debug complete. Check debug directory for visualization.
```

## 总结

多角度检测功能显著提升了复杂场景下的人脸检测能力，配合快速测试模式和完整的可视化调试，让你能够轻松找到最佳的检测配置。

**最佳实践**:
1. 先用 `--quick-test` 测试不同角度组合
2. 查看 debug 图像确认检测效果
3. 选择最适合的角度配置
4. 在生产环境中使用该配置
