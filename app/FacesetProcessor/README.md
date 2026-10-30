# FacesetProcessor - 人脸集处理工具集

高效的人脸数据集分析、过滤和排序工具，支持 HDF5 流式加载，内存占用降低 90%+，启动速度提升 30-60 倍。

## 🚀 快速开始

### 1. Analyzer - 元数据分析

```bash
# 分析所有特征
python Analyzer.py -i "workspace/data_dst/aligned"

# 只分析特定特征
python Analyzer.py -i "workspace/data_dst/aligned" --features phash,embedding

# 强制重新分析
python Analyzer.py -i "workspace/data_dst/aligned" --force

# 回写元数据到 JPG
python Analyzer.py -i "workspace/data_dst/aligned" --write-back

# 合并两个人脸集
python Analyzer.py -i "workspace/A" --merge "workspace/B"
```

**支持的特征**:
- `phash` - 感知哈希（去重）
- `histogram` - RGB/HSV 直方图
- `hue` - 色相分布
- `pose` - 人脸姿态 (pitch/yaw/roll)
- `embedding` - ArcFace 人脸嵌入向量
- `landmark` - 68点人脸特征点

### 2. Filter - 质量过滤和人脸分组

```bash
# 质量过滤（基于清晰度）
python Filter.py -i "workspace/data_dst/aligned" --mode quality    # 相对分位，自动标定
python Filter.py -i "workspace/data_dst/aligned" --mode quality --train-res 256 --absolute   # 绝对门槛

# 人脸ID分组（基于 embedding 聚类）
python Filter.py -i "workspace/data_dst/aligned" --mode face_id --eps 0.3

# 合并模式（不计算新 embedding）
python Filter.py -i "workspace/data_dst/aligned" --mode face_id --merge-only
```

### 3. Sorter - 多种特征排序

```bash
# 按感知哈希排序（相似图片排一起）
python Sorter.py -i "workspace/data_dst/aligned" --method phash

# 按其他特征排序
python Sorter.py -i "workspace/data_dst/aligned" --method hist        # 直方图
python Sorter.py -i "workspace/data_dst/aligned" --method blur         # 模糊度
python Sorter.py -i "workspace/data_dst/aligned" --method face_pose    # 人脸姿态
python Sorter.py -i "workspace/data_dst/aligned" --method resolution   # 分辨率
python Sorter.py -i "workspace/data_dst/aligned" --method color        # 颜色
python Sorter.py -i "workspace/data_dst/aligned" --method name         # 文件名
```

#### 按清晰度排序筛掉模糊图（推荐用法）

```bash
# --train-res 应与训练分辨率一致（默认 256）
python Sorter.py -i "workspace/data_dst/aligned" --method blur --train-res 256 --rename
```

排序结果：**分数越高 = 越清晰 = 序号越靠前**，重命名后形如
`sorted_00000_a3f7...jpg`（最清晰）… `sorted_01999_...jpg`（最糊）。

之后**直接打开文件夹看图删就行** —— 文件管理器按文件名排序，等于一张按清晰度
排好的缩略图长条。不需要任何 GUI，也不用真的执行"过滤"。

- 指标 = `E(b0)/E(b2)` 高频/低频能量比，与 `Filter.py` 是同一份实现（排序/过滤同一把尺子）
- 结果写进 `metadata.h5`，字段名 `sharpness_br{train_res}`；**第二次跑直接读缓存**
- 自动扣掉噪声贡献（噪声=高频=高分，不扣的话高 ISO 帧会排到最前面）
- `--motion-blur` 已废弃：新指标对运动模糊同样敏感，不再需要单独分支

## ✨ 核心特性

### HDF5 流式加载

**性能对比**（10万图片数据集）:

| 指标 | 传统方式 | 流式加载 | 提升 |
|------|---------|---------|------|
| 内存占用 | ~1GB | ~10-100MB | **90-99% ↓** |
| 启动时间 | 30-60秒 | <1秒 | **30-60倍 ↑** |
| 单字段IO | ~10KB | ~100字节 | **99% ↓** |

**工作原理**:
```python
# 旧方式：一次性加载所有元数据
metadata = load_all_metadata()  # 耗时30秒，占用1GB

# 新方式：按需流式读取
accessor = H5StreamingAccessor(file)
phash = accessor.get_field(filename, 'phash')  # 瞬间，~100字节
```

### 统一架构

三个模块共享同一个基类 `FacesetBaseProcessor`：

```python
from FacesetProcessor.Analyzer import MetadataAnalyzer
from FacesetProcessor.Filter import QualityFilter, FaceIDFilter
from FacesetProcessor.Sorter import FaceSorter

# 统一的 API
analyzer = MetadataAnalyzer(path)
filter = QualityFilter(path)
sorter = FaceSorter(path)

# 相同的方法
phash = analyzer.get_field(filename, 'phash')
phash = filter.get_field(filename, 'phash')
phash = sorter.get_field(filename, 'phash')
```

## 📖 使用示例

### Python API

```python
from FacesetProcessor.Analyzer import MetadataAnalyzer
from FacesetProcessor.Filter import QualityFilter, FaceIDFilter
from FacesetProcessor.Sorter import FaceSorter

# 1. 分析人脸集
analyzer = MetadataAnalyzer("path/to/faceset", features=['phash', 'embedding'])
analyzer.analyze_batch(force_reanalyze=False, workers=8)

# 2. 质量过滤
# 清晰度 = E(b0)/E(b2) 高频/低频能量比（不再是拉普拉斯方差）
#   train_res: 归一化分辨率，应与训练分辨率一致
#   absolute=False（默认）: 按本数据集 10/30/60 分位分组
#   absolute=True:          用合成高斯模糊定绝对门槛（比 σ=1/2 更糊才判低）
quality_filter = QualityFilter("path/to/faceset")
stats = quality_filter.filter_by_quality(workers=8, train_res=256, absolute=False)

# 3. 人脸ID分组
faceid_filter = FaceIDFilter("path/to/faceset")
groups = faceid_filter.filter_by_face_id(eps=0.3, min_samples=2)

# 4. 排序
sorter = FaceSorter("path/to/faceset")
sorted_list, new_features = sorter.sort_by_method('phash', workers=8)
sorter.rename_sorted_files(sorted_list, prefix="sorted", new_features=new_features)
```

### 高级功能

```python
# 流式访问元数据
for filename in analyzer.metadata_filenames:
    phash = analyzer.get_field(filename, 'phash')
    landmarks = analyzer.get_field(filename, 'landmarks')
    embedding = analyzer.get_field(filename, 'embedding')

# 预加载缓存
analyzer.preload_metadata(['img1.jpg', 'img2.jpg'])

# 清空缓存释放内存
analyzer.clear_metadata_cache()

# 校验数据库完整性
integrity = analyzer.verify_database_integrity()
print(f"一致性: {integrity['is_consistent']}")
```

## 🔧 安装依赖

```bash
pip install h5py opencv-python numpy pillow imagehash tqdm scikit-learn
```

**可选依赖**:
- `insightface` - 用于人脸检测和 embedding 提取（如果不用 ONNX 模型）
- `onnxruntime` - ONNX 模型推理（推荐）

## 📁 项目结构

```
FacesetProcessor/
├── Analyzer.py              # 元数据分析器
├── Filter.py                # 质量过滤和人脸分组
├── Sorter.py                # 多种特征排序
├── FacesetBaseProcessor.py  # 通用基类（流式加载核心）
├── H5StreamingAccessor.py   # HDF5 流式访问器
├── strings.py               # 多语言支持
└── README.md                # 本文档
```

## 💡 最佳实践

### 1. 选择合适的特征

```bash
# 去重场景：只需要 phash
python Analyzer.py -i "faceset" --features phash

# 人脸识别：需要 embedding
python Analyzer.py -i "faceset" --features embedding

# 完整分析：所有特征
python Analyzer.py -i "faceset" --features all
```

### 2. 多进程加速

```bash
# 自动检测 CPU 核心数
python Analyzer.py -i "faceset" --workers auto

# 指定进程数
python Analyzer.py -i "faceset" --workers 8
```

### 3. 增量分析

```bash
# 智能模式：只分析缺失的特征（默认）
python Analyzer.py -i "faceset"

# 强制模式：重新分析所有
python Analyzer.py -i "faceset" --force
```

### 4. 内存管理

对于超大数据集（>100万图片）：

```python
# 定期清空缓存
if len(accessor._cache) > 10000:
    accessor.clear_metadata_cache()
```

## 🎯 典型工作流

### 工作流 1: 人脸集清洗

```bash
# 1. 分析人脸集
python Analyzer.py -i "raw_faceset" --features phash,embedding

# 2. 质量过滤
python Filter.py -i "raw_faceset" --mode quality --threshold 20

# 3. 移动到高质量目录
mv raw_faceset/highquality cleaned_faceset

# 4. 人脸ID分组
python Filter.py -i "cleaned_faceset" --mode face_id --eps 0.3
```

### 工作流 2: 人脸集去重

```bash
# 1. 计算 phash
python Analyzer.py -i "faceset" --features phash

# 2. 按 phash 排序（相似图片排一起）
python Sorter.py -i "faceset" --method phash

# 3. 手动检查并删除重复
```

### 工作流 3: 多人脸集合并

```bash
# 合并 B 到 A（自动去重）
python Analyzer.py -i "faceset_A" --merge "faceset_B"

# 分析合并后的人脸集
python Analyzer.py -i "faceset_A" --features embedding

# 人脸ID分组
python Filter.py -i "faceset_A" --mode face_id
```

## ⚠️ 注意事项

### 1. HDF5 文件句柄

流式加载会保持 HDF5 文件打开状态。在写入前会自动关闭，但长时间运行建议手动管理：

```python
analyzer.close()  # 关闭 HDF5 文件
```

### 2. 向后兼容

保留了 `metadata` 属性，旧代码仍然有效：

```python
# 旧代码（仍然有效）
meta = analyzer.metadata.get(filename, {})

# 新代码（推荐，更高效）
meta = analyzer.get_metadata(filename)
field = analyzer.get_field(filename, 'phash')
```

### 3. 多进程环境

每个进程创建独立的 `H5StreamingAccessor` 实例。注意系统文件句柄限制：

```bash
# Linux: 增加文件句柄限制
ulimit -n 65536
```

## 🐛 常见问题

### Q1: 启动很慢？
A: 首次启动需要构建 HDF5 索引，之后会很快。确保使用流式加载（已默认启用）。

### Q2: 内存占用高？
A: 检查是否调用了 `get_metadata_snapshot()`，这会加载所有数据。改用 `get_field()` 按需读取。

### Q3: 找不到模型文件？
A: 确保 ONNX 模型在 `modelhub/` 目录下：
```
modelhub/
├── w600k_mbf.onnx
└── w600k_r50.onnx
```

### Q4: 如何恢复 landmarks？
A: 使用 landmark 模式重新分析：
```bash
python Analyzer.py -i "faceset" --features landmark --force
```

## 📊 性能基准

测试环境: Intel i7-10700K, 32GB RAM, SSD

| 数据集大小 | 启动时间 | 内存占用 | 分析速度 |
|-----------|---------|---------|---------|
| 1,000 张 | <0.1秒 | ~5MB | ~100 img/s |
| 10,000 张 | <0.5秒 | ~20MB | ~80 img/s |
| 100,000 张 | <1秒 | ~50MB | ~60 img/s |
| 1,000,000 张 | <2秒 | ~200MB | ~50 img/s |

*注: 分析速度取决于选择的特征和硬件配置*

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循原 DeepFaceLab 项目的许可证。

---

**更多详细信息**:
- [STREAMING_REFACTORING_COMPLETE.md](STREAMING_REFACTORING_COMPLETE.md) - 流式加载技术细节
- [FILTER_SORTER_MIGRATION_GUIDE.md](FILTER_SORTER_MIGRATION_GUIDE.md) - 迁移指南
