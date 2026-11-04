# DeepFaceLab 4.1（测试版）

> **状态：开发中，功能未定型。** 这个分支是 4.1 的开发线，接口与目录仍会调整。
> 稳定版见 master（4.0）。

**目标**：一个普通人打开就会用的人脸替换工具 —— 没有命令行窗口，没有一大堆用不上的模型，每个页面只有一个主按钮。

## 快速开始

```bat
RunGUI.bat          :: 双击启动（Electron 外壳，无控制台窗口）
```

只跑后端、用浏览器打开：

```bat
runtime\server\Scripts\python.exe app\serve.py
```

启动日志会打印实际端口（默认 127.0.0.1:8761，被占用时自动顺延）。

首次使用如果提示缺 Electron：

```bat
cd desktop && npm install
```

## 模型放在哪里

**这是最容易搞错的一节，按用途分三类：**

### 1. 自己训练出来的模型 → `workspace/model/`

训练器产出的东西全在这里，一个模型由一组文件组成：

```
workspace/model/
├─ 我的模型_SAEHD_data.dat          ← 结构 + 参数（必需）
├─ 我的模型_SAEHD_encoder.pth       ← 权重（必需）
├─ 我的模型_SAEHD_decoder.pth
├─ 我的模型_SAEHD_inter_AB.pth
├─ 我的模型_SAEHD_src_dst_opt.pth
├─ 我的模型_SAEHD_preview.jpg       ← 训练预览（可再生）
└─ 我的模型.dfm                     ← 合成用的导出件（可选）
```

- **训练**读取上面的 `.dat` + `.pth` 组
- **合成**只认 `.dfm`。只有训练文件、没有 `.dfm` 时，先在训练器里点「导出」
- 也支持别人给你的、只有 `.dfm` 没有训练目录的模型，直接丢进 `workspace/model/` 即可，合成器会列出来

### 2. 检测器 / 特征点 / 人脸识别等推理模型 → `weights/onnx/`

```
weights/onnx/
├─ RetinaFace/      检测器（质量优先）
├─ TinyMog/         检测器（速度优先）
├─ InsightFace2d106/ 68 点关键点
├─ ArcFace/         人脸识别（数据集按身份聚类用）
└─ …（共 20+ 个，见 weights/manifest.json）
```

也可以放在 `app/engine/modelhub/onnx/` —— 两处都会被搜索，前者优先。
想放到别的盘，用环境变量 `DFL_WEIGHTS_DIR` 指向新的根目录。

### 3. 遮罩 / 分割 / 增强等附加权重 → `weights/` 下的各个子目录

```
weights/xseg/         XSeg 遮罩模型
weights/bisenet/      人脸分割
weights/sam/  sam2/   交互式分割
weights/errface/      错脸检测
weights/grounding_dino/
```

完整清单和体积见 [weights/manifest.json](weights/manifest.json) 与 [weights/README.md](weights/README.md)（约 4.7 GB）。

### 不放模型的地方

- `app/web/`、`app/server/` —— 代码，别往里放权重
- 仓库根目录 —— 只会让 `git status` 变脏（`workspace/` 与 `weights/` 已加入忽略规则）

## 目录结构

```
desktop/          Electron 外壳（窗口、菜单、无控制台启动）
app/
├─ serve.py       FastAPI 入口
├─ server/        后端：提取 / 对齐 / 训练 / 合成 / 工具
├─ web/           静态前端（原生 JS + CSS，无构建步骤）
├─ engine/        4.1 引擎（PyTorch 训练 + 合成管线 + modelhub）
├─ tools/         迁移、测试、性能脚本
└─ docs/          规划、结构说明、界面讨论稿
weights/          推理模型（见上）—— 不入库
workspace/        用户数据：数据集 / 模型 / 输出 —— 不入库
runtime/          Python 运行时（server / engine 两个 venv）—— 不入库
legacy/           4.0 Qt 界面与素材（仅供对照）
```

界面是 Web 技术栈，跑在 Electron 里；后端是本地 FastAPI，不依赖前端框架，改完刷新即可。

## 需要什么

| 依赖 | 说明 |
|---|---|
| Python 3.12 | 内置在 `runtime/server`（后端）与 `runtime/engine`（训练/合成） |
| PyTorch (CUDA) | 训练与合成 |
| onnxruntime-gpu | 检测器、特征点、DFM 推理 |
| Node.js | 只在第一次装 Electron 时需要 |

显卡：8 GB 显存可跑 128 分辨率 + batch 8（实测峰值 5.7 GB，约 6.5 it/s）。

## 设计约定

配色、间距、圆角、字号收敛在 [UI设计规范.md](UI设计规范.md)；主题色只用于主按钮和进度条。

训练器见 [app/docs/07-TRAINER.md](app/docs/07-TRAINER.md)，
合成器见 [app/docs/06-MERGER.md](app/docs/06-MERGER.md)，
结构说明 [app/docs/02-STRUCTURE.md](app/docs/02-STRUCTURE.md)，
产品规划 [app/docs/01-PLAN.md](app/docs/01-PLAN.md)。

## 协议

GPL-3.0
