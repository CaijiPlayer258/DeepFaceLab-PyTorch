# ModelHub - DeepFaceLive 模型仓库

## 📋 概述

ModelHub 是 DeepFaceLive 的核心模型仓库，包含了实时人脸交换和处理所需的各种深度学习模型。该仓库按后端框架和用途组织，支持 ONNX Runtime、PyTorch 和 OpenCV 等多种推理引擎。

## 📁 目录结构

```
modelhub/
├── DFLive/          # DeepFaceLive 核心模型（DFM 格式）
├── onnx/            # ONNX Runtime 模型
├── torch/           # PyTorch 模型及转换工具
├── cv/              # OpenCV 传统计算机视觉模型
└── README.md        # 本说明文档
```

---

## 🔧 各模块详解

### 1. DFLive - 核心换脸模型

**位置**: `modelhub/DFLive/`

这是 DeepFaceLive 的核心模块，负责加载和执行 DFM（DeepFaceModel）格式的换脸模型。

#### 主要功能
- **DFMModel**: DFM 模型的加载和推理引擎
- 支持标准 DFM 模型（`.dfm` 文件）
- 支持轻量级模型（`.litedfm` 和 `.onnx` 文件）
- 支持多设备推理（CPU/GPU/TensorRT）
- 提供 GPU 加速预处理（CuPy/PyTorch）

#### 关键类和方法
```python
from modelhub.DFLive import (
    DFMModel_from_path,      # 从路径加载模型
    DFMModel_from_info,      # 从模型信息加载
    get_available_models_info, # 获取可用模型列表
    get_available_devices     # 获取可用推理设备
)
```

#### 模型类型
- **Type 1**: 基础换脸模型
- **Type 2**: 带形态控制（morph_value）的模型
- **Type 3**: 带掩码输出的高级模型

#### 性能优化
- ✅ CuPy GPU 加速预处理
- ✅ PyTorch CUDA 图像缩放
- ✅ OpenCV CUDA 加速
- ✅ TensorRT 推理加速（详见 [TENSORRT_ACCELERATION.md](DFLive/TENSORRT_ACCELERATION.md)）
- ✅ GPU 缓冲区预分配

---

### 2. ONNX - ONNX Runtime 模型

**位置**: `modelhub/onnx/`

包含各种用于人脸检测、对齐和处理的 ONNX 格式模型。

#### 可用模型

| 模型 | 用途 | 说明 |
|------|------|------|
| **BlazeFace** | 人脸检测 | 轻量级快速人脸检测器 |
| **CenterFace** | 人脸检测 | 高精度人脸检测和关键点定位 |
| **S3FD** | 人脸检测 | 单阶段尺度不变人脸检测器 |
| **YoloV5Face** | 人脸检测 | YOLOv5 架构的人脸检测 |
| **FaceMesh** | 面部网格 | 468 点面部 landmarks 检测 |
| **InsightFace2d106** | 面部关键点 | 106 点面部特征点检测 |
| **InsightFaceSwap** | 人脸交换 | InsightFace 换脸模型 |
| **XSeg** | 面部分割 | 面部区域语义分割 |
| **LIA** | 局部图像调整 | 面部属性编辑 |

#### 使用示例
```python
from modelhub.onnx import BlazeFace, CenterFace, XSeg

# 初始化检测器
detector = BlazeFace(device_info)
faces = detector.detect(image)

# 面部分割
xseg = XSeg(device_info)
mask = xseg.segment(face_image)
```

---

### 3. Torch - PyTorch 模型

**位置**: `modelhub/torch/`

包含 PyTorch 原生模型和 ONNX 转换工具。

#### 可用模块

| 模块 | 功能 | 说明 |
|------|------|------|
| **FaceAligner** | 人脸对齐网络 | 用于精确的人脸对齐 |
| **FaceMerger** | 人脸融合网络 | 高质量的人脸 blending |
| **CenterFace** | 转换工具 | CenterFace 转 ONNX |
| **S3FD** | S3FD 模型 | PyTorch 版本的人脸检测 |

#### 主要功能
```python
from modelhub.torch import FaceAlignerNet, CenterFace_to_onnx

# 人脸对齐网络
aligner = FaceAlignerNet()
aligned_face = aligner.align(face_image, landmarks)

# 模型转换
CenterFace_to_onnx(torch_model_path, onnx_output_path)
```

---

### 4. CV - OpenCV 传统模型

**位置**: `modelhub/cv/`

基于 OpenCV 的传统计算机视觉模型，无需深度学习推理引擎。

#### 可用模型

| 模型 | 用途 | 说明 |
|------|------|------|
| **FaceMarkerLBF** | 面部标记 | LBF 算法的面部特征点检测 |

#### 特点
- ⚡ 速度快，资源占用低
- 💻 仅依赖 OpenCV
- 🎯 适合轻量级应用

#### 使用示例
```python
from modelhub.cv import FaceMarkerLBF

marker = FaceMarkerLBF()
landmarks = marker.detect(image)
```

---

## 🚀 模型加载流程

### 标准工作流程

1. **选择检测设备**
   ```python
   from modelhub.DFLive import get_available_devices
   
   devices = get_available_devices()
   device = devices[0]  # 选择第一个可用设备
   ```

2. **加载换脸模型**
   ```python
   from modelhub.DFLive import DFMModel_from_path
   
   model = DFMModel_from_path('path/to/model.dfm', device)
   ```

3. **执行人脸交换**
   ```python
   output = model.convert(face_image, morph_value=0.5)
   ```

### 完整管道示例

```
视频帧 → 人脸检测(BlazeFace/CenterFace) 
       → 关键点检测(FaceMesh/InsightFace2d106)
       → 人脸对齐(FaceAligner)
       → 换脸推理(DFMModel)
       → 面部分割(XSeg)
       → 人脸融合(FaceMerger)
       → 输出帧
```

---

## ⚙️ 配置和优化

### 设备选择优先级

1. **TensorRT GPU** (最快，需要 NVIDIA GPU + TensorRT)
2. **CUDA GPU** (快速，需要 NVIDIA GPU)
3. **DirectML** (Windows GPU 加速)
4. **CPU** (通用，速度较慢)

### 性能调优建议

#### 对于实时应用
- 使用轻量级检测器（BlazeFace vs CenterFace）
- 启用 GPU 加速预处理（安装 CuPy）
- 使用 TensorRT 优化的 DFM 模型
- 降低输入分辨率（128x128 或 256x256）

#### 对于高质量输出
- 使用高精度检测器（CenterFace/S3FD）
- 启用 XSeg 面部分割
- 使用 FaceMerger 进行后处理
- 使用更高分辨率的 DFM 模型

### 环境变量

```bash
# 设置 ONNX Runtime 日志级别
export ORT_LOGGING_LEVEL=1

# 指定 CUDA 设备
export CUDA_VISIBLE_DEVICES=0
```

---

## 📦 模型文件格式

### DFM 格式
- **扩展名**: `.dfm`
- **描述**: DeepFaceLab 标准模型格式
- **特点**: 完整的模型权重和架构

### Lite DFM 格式
- **扩展名**: `.litedfm`
- **描述**: 轻量级 DFM 模型
- **特点**: 优化的模型，更快的推理速度

### ONNX 格式
- **扩展名**: `.onnx`
- **描述**: 开放神经网络交换格式
- **特点**: 跨平台兼容，支持多种推理引擎

---

## 🔍 调试和故障排除

### 常见问题

#### 1. 模型加载失败
```python
# 检查模型文件是否存在
from pathlib import Path
model_path = Path('path/to/model.dfm')
print(f"Model exists: {model_path.exists()}")

# 检查设备兼容性
from modelhub.DFLive import get_available_devices
print(get_available_devices())
```

#### 2. 性能问题
- 确认 GPU 驱动已正确安装
- 检查是否启用了 CUDA/CuDNN
- 验证模型是否与设备兼容
- 查看 [TENSORRT_ACCELERATION.md](DFLive/TENSORRT_ACCELERATION.md) 了解优化方法

#### 3. 内存不足
- 降低输入分辨率
- 减少批处理大小
- 使用轻量级模型
- 清理未使用的 GPU 缓存

### 调试模式

在开发过程中，可以启用调试输出来监控性能：

```python
# 在 DFMModel 中启用调试（需要修改源码）
model._debug_enabled = True
model._debug_per_frame = True
```

---

## 🛠️ 开发指南

### 添加新模型

1. **ONNX 模型**
   - 在 `modelhub/onnx/` 下创建新目录
   - 实现模型加载和推理类
   - 在 `__init__.py` 中导出

2. **PyTorch 模型**
   - 在 `modelhub/torch/` 下创建新目录
   - 实现模型定义和转换工具
   - 更新 `__init__.py`

3. **DFM 模型**
   - 直接使用 DFMModel 加载
   - 无需额外代码

### 模型转换

```python
# PyTorch 转 ONNX
from modelhub.torch import CenterFace_to_onnx
CenterFace_to_onnx('model.pth', 'model.onnx')

# 导出 XSeg 为 TensorRT
# 运行项目根目录下的脚本
python export_xseg_fp32_trt.py
```

---

## 📚 相关文档

- [TensorRT 加速指南](DFLive/TENSORRT_ACCELERATION.md) - TensorRT 优化详细说明
- [DeepFaceLive 主文档](../../README.md) - 项目总体介绍
- [许可证](../../LICENSE) - 使用和分发条款

---

## 🤝 贡献

欢迎贡献新的模型或改进现有模型！请遵循以下原则：

1. 保持代码风格一致
2. 添加必要的文档和注释
3. 测试不同设备和配置
4. 提供性能基准数据

---

## 📄 许可证

本项目遵循 DeepFaceLive 的许可证条款。详情请参阅 [LICENSE](../../LICENSE) 文件。

---

**最后更新**: 2026-04-11  
**维护者**: DeepFaceLive 团队
