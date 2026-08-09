<p align="center">
  <img src="assets/screenshots/首图.png" width="800" alt="DeepFaceLab-Torch">
</p>

<h1 align="center">DeepFaceLab-Torch</h1>

<p align="center">
  <em>A powerful and modern DeepFake workflow based on PyTorch</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/中文-文档-blue" alt="中文"></a>
  <a href="README_EN.md"><img src="https://img.shields.io/badge/English-Docs-green" alt="English"></a>
</p>

> **⚠️ 维护状态：本项目已进入低维护模式**（v4.0.7 起）。核心功能（训练/合成/导出）保持可用，但不再积极开发新功能；严重 bug 仍会修复，欢迎 fork 继续维护。

<br>

<p align="center">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License">
</p>

<p align="center">
  <img src="assets/screenshots/主页面大图.png" width="900" alt="主界面">
</p>

---

## 快速开始

> **为什么要扫码登录？**  
> 首次启动会弹出哔哩哔哩扫码登录窗口，登录后会保存 Cookie 到本地，用于解锁 **B 站视频下载器** 功能。  
> 登录同时会自动关注 B 站账号 **菜级玩家**，如果不需要此功能，可以跳过登录：
> 
> 在 `ui/user/bilibili_cookie.txt` 中写入任意包含 `SESSDATA=` 的内容：
> ```bash
> echo "SESSDATA=skip" > ui/user/bilibili_cookie.txt
> ```
> 跳过登录不影响训练、提取、合成等核心功能。

### 环境要求
- Python 3.12+
- NVIDIA GPU（推荐 8GB+ 显存）
- CUDA 12.x + cuDNN 9.x

### 下载模型文件

模型文件（检测器、特征点、换脸模型等）需从网盘下载后放入对应目录：

**百度网盘：** https://pan.baidu.com/s/1u7WULEl2glQmVCQXEkpX_g?pwd=cjwj

下载后解压，将 `modelhub/` 目录复制到项目根目录覆盖即可。

### 下载预安装环境包

已配置好 Python 环境、依赖、PyQt-SiliconUI 的完整压缩包，解压后将 `python/` 目录放到项目根目录（与 `run.bat` 同级）即可直接运行。无痛解压即用：

**百度网盘：** https://pan.baidu.com/s/1CyqR8Hw7gp6i3Dcx56Nhzw?pwd=cjwj

### 安装依赖

```bash
pip install PyQt-SiliconUI  # https://github.com/ChinaIceF/PyQt-SiliconUI
pip install -r requirements.txt
```

### 运行

```bash
python main.py
```

或直接双击 `run.bat`。

---

## 项目结构

```
DeepFaceLab-Torch/
├── main.py                   # 主入口
├── run.bat                   # Windows 快捷启动
│
├── python/                   # 嵌入式 Python 环境（解压到项目根目录）
│   ├── python.exe            # Python 解释器
│   ├── Lib/site-packages/    # 预装依赖库
│   └── Scripts/              # 可执行工具
│
├── models/                   # 训练模型
│   ├── Model_SAEHD/          # SAEHD 训练器（主训练模型）
│   ├── Model_DeepFakeLarge/  # DFLarge 训练器
│   ├── Model_LIAELarge/      # LIAELarge 训练器
│   └── ModelBase.py          # 模型基类
│
├── modelhub/                 # AI 推理模型（需从网盘下载）
│   └── onnx/                 # ONNX 模型（检测/特征点/换脸等）
│
├── mainscripts/              # 核心脚本
│   ├── Trainer.py            # 训练器主逻辑
│   ├── Extractor.py          # 人脸提取（已弃用，用 Extractor/）
│   └── Merger.py             # 合成器
│
├── Extractor/                # 人脸提取器（独立模块）
│   ├── Extractor.py          # 提取主程序
│   └── strings.py            # 多语言字符串
│
├── MergeStudio/              # 合成器（Web UI）
│   ├── api/                  # FastAPI 后端接口
│   │   ├── routes_preview.py # 预览/渲染
│   │   ├── routes_export.py  # 视频导出
│   │   ├── routes_project.py # 项目管理
│   │   └── routes_timeline.py# 时间线
│   └── core/                 # 核心逻辑
│       ├── merger.py         # 人脸融合引擎
│       ├── export_pipeline.py# 导出管线（ffmpeg 编码）
│       └── arcface.py        # ArcFace 识别
│
├── DataAugmenter/            # 数据增强
│   ├── XSegAugmenter.py      # XSeg/XSegLite 遮罩批量应用
│   └── __main__.py           # CLI 入口
│
├── ui/                       # PyQt6 图形界面
│   ├── components/
│   │   ├── page_trainer/     # 训练器界面
│   │   ├── page_data_extraction/ # 数据提取界面
│   │   └── page_bilibili_downloader/ # B站下载器
│   └── img/                  # 图标资源
│
├── WebUI/                    # Web 管理界面
│   ├── page_settings.py      # 训练参数设置
│   └── index.html            # 前端页面
│
├── xlib/                     # 工具库
│   ├── trt.py                # TensorRT 推理封装
│   └── ...
│
├── tools/                    # 工具脚本
│   ├── tonemap_aligned_faces.py  # 已提取人脸色调映射（杜比视界）
│   ├── export_onnx_to_trt.py     # ONNX → TRT 引擎编译
│   └── pack_source.py            # 源码打包
│
├── samplelib/                # 样本加载器
├── facelib/                  # 人脸处理库
│   ├── FaceDetector.py       # 检测器封装
│   ├── FaceType.py           # 脸型枚举
│   └── XSegNet.py            # XSeg 遮罩网络
│
├── DFLIMG/                   # DFL 图片格式
│   ├── DFLJPG.py             # JPG + APP15 元数据
│   └── DFLIMG.py             # 图片加载器
│
├── core/                     # 核心运行时
│   ├── leras/                # 神经网络层与优化器
│   │   ├── nn.py             # 网络层
│   │   └── optimizers/       # 优化器（AdaBelief/Adam/Lion/RMSprop）
│   ├── interact.py           # 交互式 CLI
│   └── osex.py               # 系统工具
│
└── ffmpeg/                   # FFmpeg（需自行下载）
    └── ffmpeg.exe            # 推荐 gyan.dev full 静态版
```

---

## 模块详解

### 🏋️ 训练器（Trainer）

![训练器界面](assets/screenshots/trainer.png)

支持训练模型：

| 模型 | 说明 |
|---|---|
| **SAEHD** | 标准换脸模型（DF/LIAE 架构） |
| **DeepFakeLarge** | 大模型 |
| **LIAELarge** | LIAE 大模型 |
| **XSeg** | 高精度遮罩模型 |
| **XSegLite** | 轻量遮罩模型（原创） |

**训练特性：**
- BF16 混合精度
- 崩溃检测器（自动回滚异常迭代）
- 快速加载器（多线程异步样本预载）
- 余弦退火（Cosine Annealing LR）
- Lion 优化器
- VGG 感知损失（可配置权重）


---

### ⚡ XSegLite（原创）

本项目完全自研的轻量级遮罩模型，独立于原版 DeepFaceLab。

**性能对比（基于 RTX 3080）：**

| 模型 | ONNX (CUDA) | TensorRT FP16 |
|---|---|---|
| 原版 XSeg (iperov) | ~10.4ms (96fps) | ~7.0ms (143fps) |
| **XSegLite** | **2.5ms (395fps)** | **1.46ms (684fps)** |

**特点：**
- SimpleGate 激活函数 + FP32 精度锁定，保证遮罩质量
- 逐层精度控制（BF16 卷积 + FP32 敏感层）
- 与 XSeg 完全兼容，可替代原版 XSeg
- 模型体积仅 32MB（TRT engine）

**下载训练好的模型：**

| 模型 | 下载 | 镜像 |
|---|---|---|
| XSeg | [xseg.onnx](https://huggingface.co/thinkanameishard/xseg/resolve/main/xseg.onnx) | [镜像](https://hf-mirror.com/thinkanameishard/xseg/resolve/main/xseg.onnx) |
| XSegLite | [xseglite.onnx](https://huggingface.co/thinkanameishard/xseg/resolve/main/xseglite.onnx) | [镜像](https://hf-mirror.com/thinkanameishard/xseg/resolve/main/xseglite.onnx) |

放入 `workspace/model/XSeg/`（XSeg）和 `workspace/model/XSegLite/`（XSegLite）。

XSegLite 训练器位于 `models/Model_XSegLite/`，推理集成在 `DataAugmenter/XSegAugmenter.py` 中。

### 🔍 人脸提取（Extractor）

支持 14 种检测器 + 5 种特征点标记器，TensorRT 加速。

**检测器：** BlazeFace / CenterFace / DamoFD / LightweightFD / MogFace / MTCNN / RetinaFace(10g/500m) / S3FD / TinyMog / ULFD / YoloV5Face / YoloV8Face / YoloV11nFace

**特征点标记器：** insightface-2d106det / 2DFAN-4 / 3DFAN-4 / insightface-3d68 / Google-mediapipe / OpenSeeFace / PFLD / MobileFaceNet

**提取模式：**
- **标准模式（快速）** — bt709 直出，满速切脸，提取完可后处理色调映射
- **HDR 精确模式（慢速）** — 管道内 libplacebo 色调映射，4K 直出，颜色完全正确

![数据提取](assets/screenshots/data_extract.png)

### 🎬 合成器（MergeStudio）

Web UI 合成界面，支持多轨道时间线。

**功能：**
- 实时预览合成效果
- 多角度人脸检测与替换
- XSeg/XSegLite 遮罩
- 色彩融合模式（mkl/sot/rct/lct/idt）
- FFmpeg 硬件加速导出（NVENC）
- 杜比视界视频导入支持

![合成器界面](assets/screenshots/mergestudio.png)

### 📊 数据处理（Data Processing）

**帧提取（`ui/components/page_data_extraction/`）：**
- FFmpeg 提取视频帧（CUDA 加速解码）
- 像素格式 / 色彩范围 / 色彩空间控制
- 杜比视界转码（HDR→Rec.709 / Rec.2020）

**人脸切脸（`Extractor/`）：**
- 批量人脸检测与对齐（14 种检测器 + 5 种特征点标记器）
- 预缩放 / 跳帧 / 多角度检测
- TRT 加速推理
- 杜比视界后处理色调映射（对已提取人脸）

**数据集工具（`mainscripts/`）：**

| 工具 | 用途 |
|---|---|
| `FacesetResizer.py` | 人脸集批量缩放 |
| `FacesetEnhancer.py` | 人脸增强/修复 |
| `Sorter.py` | 人脸排序（按模糊/相似度/人脸质量等） |
| `VideoEd.py` | 视频裁剪/合并/切片 |
| `XSegUtil.py` | XSeg 多边形遮罩标注管理（复制/移除/查看） |
| `Util.py` | 数据集打包/解包/去重等工具 |
| `ExportDFM.py` | 导出 DFM 模型格式 |

![数据处理](assets/screenshots/data_process.png)

### 🎨 遮罩绘制与数据增强

**XSeg 遮罩编辑器（`MaskProcessor/`）：**
- Web UI 交互式多边形遮罩标注
- 遮罩复制 / 移除 / 查看
- SAM / BiseNet 等多种自动遮罩模型
- 适用于训练自定义 XSeg / XSegLite 模型

![遮罩绘制](assets/screenshots/mask_editor.png)

**DataAugmenter（`DataAugmenter/`）：**
- XSeg（高精度） / XSegLite（快速） 两种模型
- TensorRT / ONNX Runtime / CPU 三种后端
- 多线程并行处理（支持 GPU 加速）
- 遮罩反转、编码优化

### 🛠️ 工具脚本

| 脚本 | 用途 |
|---|---|
| `tools/tonemap_aligned_faces.py` | 对已提取的人脸批量做 HDR→SDR 色调映射 |
| `tools/export_onnx_to_trt.py` | ONNX 模型编译为 TRT engine |
| `tools/export_dfm.py` | 模型导出为 DFM 格式 |
| `tools/unpack_pak.py` | PAK 文件解包 |
| `tools/clear_Iter_str.py` | 模型 data.dat 清理工具 |
| `update_ffmpeg.bat` | 自动下载更新 FFmpeg |

---

## FFmpeg

推荐使用 **gyan.dev full 静态版**（含 libplacebo，支持 HDR 色调映射）。

运行 `update_ffmpeg.bat` 自动下载，或手动下载：
- **百度网盘：** https://pan.baidu.com/s/13enDtjFNudUZYvnFmGA1xQ?pwd=cjwj
- **gyan.dev 官方：** https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z

## TensorRT

编译 TRT engine 需要安装 TensorRT：
```bash
# 安装 TensorRT（Windows）
pip install tensorrt==10.0.1
# 编译模型
python tools/export_onnx_to_trt.py --name model_name --batch 1
```

## 常见问题

**Q: 模型从哪里下载？**
A: 百度网盘 https://pan.baidu.com/s/1u7WULEl2glQmVCQXEkpX_g?pwd=cjwj

**Q: 需要哪种 FFmpeg？**
A: gyan.dev 的 full 静态版（含 libplacebo）。运行 `update_ffmpeg.bat` 自动下载。

**Q: HDR 视频颜色不对？**
A: 在「人脸提取」→「HDR 模式」选择「HDR 精确模式」使用 libplacebo 管道内色调映射。

**Q: 训练时显存不足？**
A: 降低 batch_size、开启梯度检查点、使用 BF16 混合精度。


**Q: TensorRT 怎么用？**
A: 安装 TensorRT 后，在训练/提取界面勾选「启用 TRT」即可。

---

## 交流

- **QQ 群：** 191017993
- **Bilibili：** https://space.bilibili.com/500398541

---

## 许可

本项目采用 **GPL-3.0** 协议。详情见 [LICENSE](LICENSE)。

---

## 鸣谢

- 原版 DeepFaceLab (iperov) — https://github.com/iperov/DeepFaceLab
- UI 框架 PyQt-SiliconUI (ChinaIceF) — https://github.com/ChinaIceF/PyQt-SiliconUI
- PyTorch 社区
- 所有贡献者
