# Extractor 模块使用示例

## 命令行模式（推荐用于测试和自动化）

### 基本用法

```bash
# 使用短参数
.\runExtractor.bat -i ".\workspace\data_dst.mp4" -o ".\workspace\data_dst\aligned" -d BlazeFace -l insightface-2d106det

# 使用长参数
.\runExtractor.bat --input ".\workspace\data_dst.mp4" --output ".\workspace\data_dst\aligned" --FaceDetector BlazeFace --FaceMarker insightface-2d106det

# 处理图片文件夹
.\runExtractor.bat -i ".\workspace\images" -o ".\workspace\faces" -d CenterFace -l Google-mediapipe
```

### 参数说明

| 参数 | 简写 | 说明 | 可选值 |
|------|------|------|--------|
| --input | -i | 输入路径（视频文件或图片文件夹） | 任意有效路径 |
| --output | -o | 输出路径（保存对齐后的人脸） | 任意有效路径 |
| --FaceDetector | -d | 人脸检测器 | 无, YOLOv8-Face, RetinaFace, MTCNN, BlazeFace, CenterFace, S3FD, YoloV5Face |
| --FaceMarker | -l | 特征点标记器 | insightface-2d106det, 2DFAN-4, Google-mediapipe |
| --size | -s | 输出图像尺寸（像素） | 整数，默认None（动态根据人脸大小计算） |

### 常用组合

```bash
# 快速测试（BlazeFace + InsightFace）
.\runExtractor.bat -i ".\workspace\data_dst.mp4" -o ".\workspace\test_output" -d BlazeFace -l insightface-2d106det

# 高精度提取（CenterFace + MediaPipe）
.\runExtractor.bat -i ".\workspace\data_dst.mp4" -o ".\workspace\hq_output" -d CenterFace -l Google-mediapipe -s 512

# 动态尺寸（根据人脸bbox自动计算，推荐保持原始质量）
.\runExtractor.bat -i ".\workspace\data_dst.mp4" -o ".\workspace\output" -d BlazeFace -l insightface-2d106det

# 仅指定必要参数（其他使用默认值：BlazeFace + insightface-2d106det + 动态尺寸）
.\runExtractor.bat -i ".\workspace\data_dst.mp4" -o ".\workspace\output"
```

## 交互式模式

如果不提供任何参数，将进入交互式模式：

```bash
.\runExtractor.bat
```

然后按照提示输入：
1. 输入路径
2. 输出路径
3. 选择检测器（输入数字）
4. 选择特征点标记器（输入数字）
5. 确认开始

## Python直接调用

```bash
# 直接使用Python
.\python\Scripts\python.exe Extractor\Extractor.py -i ".\workspace\data_dst.mp4" -o ".\workspace\aligned"

# 查看帮助
.\python\Scripts\python.exe Extractor\Extractor.py --help
```

## 输出格式

生成的文件命名规则：`{帧序号:05d}_{人脸序号}.jpg`

示例：
- `00000_0.jpg` - 第1帧的第1个人脸
- `00000_1.jpg` - 第1帧的第2个人脸
- `00001_0.jpg` - 第2帧的第1个人脸

### 元数据文件

提取完成后会生成 `metadata.json` 文件，包含每个人脸的详细信息：
- `face_type`: 人脸类型（WHOLE_FACE）
- `landmarks`: 对齐后的68个特征点坐标
- `source_landmarks`: 原始图像中的特征点坐标
- `source_rect`: 原始人脸边界框 [left, top, right, bottom]
- `image_to_face_mat`: 仿射变换矩阵
- `source_filename`: 源文件名

```json
{
  "00000_0.jpg": {
    "face_type": "whole_face",
    "landmarks": [[x1, y1], [x2, y2], ...],
    "source_landmarks": [[x1, y1], [x2, y2], ...],
    "source_rect": [100, 150, 300, 350],
    "image_to_face_mat": [[...], [...], [...]],
    "source_filename": "data_dst.mp4"
  }
}
```

## 注意事项
0. **有使用AI生成代码**
1. **路径格式**：建议使用相对路径，以 `.\` 开头
2. **检测器选择**：
   - **BlazeFace**：速度快，适合实时处理，CPU/GPU均可
   - **CenterFace**：精度高，适合高质量需求
   - **S3FD**：平衡速度和精度
   - **YoloV5Face**：YOLOv5架构，检测效果好
   - ⚠️ YOLOv8-Face、RetinaFace、MTCNN 暂未实现
3. **标记器选择**：
   - **insightface-2d106det**：106个特征点，转换为标准68点，精度高（推荐）
   - **Google-mediapipe**：468个特征点，转换为标准68点，速度较快但精度略低
   - ⚠️ 2DFAN-4 暂未实现
4. **输出尺寸**：
   - **默认None**：根据人脸边界框动态计算（原尺寸×1.2倍padding），保持最佳质量
   - **固定尺寸**：使用 `-s` 参数指定（如 `-s 256` 或 `-s 512`）
   - 动态计算范围：64px ~ 2048px，确保为偶数
5. **视频处理特性**：
   - 自动帧间人脸排序：通过欧氏距离匹配，保持连续帧内人脸序号一致性
   - GPU加速：优先使用CUDA，其次DirectML/DX12，最后CPU
6. **图片处理特性**：
   - 多线程并行处理：使用CPU核心数的线程池
   - 支持格式：.jpg, .jpeg, .png, .bmp, .tif, .tiff
7. **JPEG质量**：所有输出使用100%质量保存，无压缩损失
