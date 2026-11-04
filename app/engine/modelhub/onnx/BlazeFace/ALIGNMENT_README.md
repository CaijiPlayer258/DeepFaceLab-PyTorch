# BlazeFace 人脸对齐功能说明

## 概述

BlazeFace 检测器现在支持基于5个面部特征点的人脸对齐功能。该功能利用检测到的眼睛、鼻子和嘴巴位置来计算人脸旋转角度，并生成经过对齐的边界框。

## 功能特点

### 1. 5个特征点
BlazeFace 模型输出以下5个面部特征点（归一化坐标 0-1）：
- **左眼** (left eye): `ley_x, ley_y`
- **右眼** (right eye): `rey_x, rey_y`
- **鼻子** (nose): `nose_x, nose_y`
- **左嘴角** (mouth left): `mou_x, mou_y`
- **右嘴角** (mouth right / left ear area): `lea_x, lea_y`

### 2. 对齐算法

#### 步骤 1: 计算旋转角度
```python
# 使用双眼连线计算人脸旋转角度
eye_dx = right_eye[0] - left_eye[0]
eye_dy = right_eye[1] - left_eye[1]
angle_rad = math.atan2(eye_dy, eye_dx)
```

#### 步骤 2: 确定人脸中心
```python
# 以双眼中点作为人脸中心
eye_center = (left_eye + right_eye) / 2.0
```

#### 步骤 3: 估算人脸尺寸
```python
# 基于瞳距估算完整人脸尺寸
eye_dist = np.linalg.norm(right_eye - left_eye)
face_size = eye_dist * 4.0  # 典型人脸约为瞳距的4倍
```

#### 步骤 4: 生成旋转后的边界框
```python
# 创建正方形边界框并绕眼中心旋转
# 旋转矩阵应用负角度以使人脸垂直对齐
cos_a = math.cos(-angle_rad)
sin_a = math.sin(-angle_rad)
rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

# 旋转四个角点
rotated_corners = corners @ rot_matrix.T + eye_center

# 从旋转后的角点提取轴对齐边界框
aligned_x1 = min(rotated_corners[:, 0])
aligned_y1 = min(rotated_corners[:, 1])
aligned_x2 = max(rotated_corners[:, 0])
aligned_y2 = max(rotated_corners[:, 1])
```

### 3. API 使用

```python
from modelhub.onnx.BlazeFace.BlazeFace import BlazeFace

# 初始化检测器
devices = BlazeFace.get_available_devices()
detector = BlazeFace(devices[0])

# 方法 1: 启用对齐（默认）
faces_aligned = detector.extract(img, threshold=0.5, enable_alignment=True)
# 返回经过旋转对齐的边界框

# 方法 2: 禁用对齐，返回原始检测框
faces_raw = detector.extract(img, threshold=0.5, enable_alignment=False)
# 返回原始的矩形检测框
```

### 4. 返回值格式

两种模式都返回相同格式的列表：
```python
[
    [x1, y1, x2, y2],  # 第一个人脸的边界框
    [x1, y1, x2, y2],  # 第二个人脸的边界框
    ...
]
```

其中：
- `x1, y1`: 左上角坐标
- `x2, y2`: 右下角坐标
- 坐标为整数像素值，已裁剪到图像边界内

## 优势

### 启用对齐的好处：
1. **更准确的裁剪**: 边界框跟随人脸旋转，减少背景区域
2. **更好的对齐效果**: 后续的人脸对齐和处理更加精确
3. **适应倾斜人脸**: 对侧脸或倾斜的人脸有更好的鲁棒性

### 禁用对齐的场景：
1. **性能优先**: 不需要额外的旋转计算
2. **简单场景**: 人脸基本正对摄像头
3. **兼容性**: 需要与旧版本保持一致的行为

## 技术细节

### 坐标系
- **输入**: BlazeFace 输出归一化坐标 (0-1 范围)
- **转换**: 直接乘以原图宽高得到像素坐标
- **输出**: 整数像素坐标，已在图像边界内裁剪

### 旋转方向
- 使用**负角度**进行旋转，使人脸垂直对齐
- 如果人脸向右倾斜（顺时针），则逆时针旋转校正

### 尺寸估算
- 基于瞳距（两眼之间的距离）估算完整人脸尺寸
- 系数 4.0 是经验值，适用于大多数成年人脸

## 测试

运行测试脚本验证功能：
```bash
python test_blazeface_alignment.py [image_path]
```

如果不提供图像路径，将使用随机图像进行基本测试。

## 注意事项

1. **最小检测尺寸**: 边界框宽度和高度必须至少为 5 像素
2. **边界处理**: 所有坐标都已裁剪到图像范围内 (0-W, 0-H)
3. ** landmark 质量**: 对齐效果依赖于 BlazeFace  landmark 检测的准确性
4. **极端角度**: 对于超过 ±45° 的极端旋转，对齐效果可能下降

## 与其他组件集成

此功能可以与 DeepFaceLive 的 FaceAligner 模块无缝集成：
- FROM_RECT 模式可以直接使用对齐后的边界框
- 提供更准确的人脸裁剪区域
- 改善后续人脸交换和对齐的质量

## 示例代码

完整的集成示例请参考 `test_blazeface_alignment.py` 文件。
