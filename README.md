# DeepFaceLab 4.1（测试版）

> **状态：开发中，功能未定型。** 这个分支是 4.1 的开发线，接口与目录仍会调整。
> 稳定版见 master（4.0）。

**目标**：一个普通人打开就会用的人脸替换工具 —— 没有命令行窗口，没有一大堆用不上的模型，每个页面只有一个主按钮。

## 快速开始

```bat
RunGUI.bat          :: 双击启动（Electron 外壳，无控制台窗口）
```

它做三件事：拉起 Python 后端 → 等后端打印端口 → 开一个窗口指向该端口。
后端默认 127.0.0.1:8761，被占用时自动顺延，窗口会跟着实际端口走。

只跑后端、用浏览器打开（不需要任何 Electron 依赖）：

```bat
runtime\server\Scripts\python.exe app\serve.py
:: 然后在浏览器打开日志里打印的 http://127.0.0.1:<PORT>/
```

### Electron 外壳怎么配

外壳代码在 `desktop/`（`main.js` + `preload.js` + `package.json`），
**`node_modules` 不入库**，所以第一次要装一次依赖（之后不用再装）：

```bat
cd desktop
npm install
```

| 项目 | 说明 |
|---|---|
| Node.js | 需要 18 或更高（只在这一步用到，运行时不需要 Node） |
| 下载量 | 约 100–150 MB（Electron 33 + 字体包 Inter / Noto Sans SC） |
| 装到哪 | `desktop/node_modules`，已被 `.gitignore` 忽略 |
| 用时 | 通常 1–3 分钟，取决于网络 |

装完回根目录双击 `RunGUI.bat` 即可。若它提示 `Electron is not installed`，
就是这一步没做——脚本会直接打印缺失路径和修复命令。

国内网络卡在 Electron 二进制下载时，加两个镜像再装：

```bat
npm config set registry https://registry.npmmirror.com
set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
cd desktop && npm install
```

### 三种启动方式

| 方式 | 需要什么 | 适合 |
|---|---|---|
| `RunGUI.bat` | Electron 装好 + runtime | 日常使用，无控制台窗口 |
| `app\serve.py` + 浏览器 | 只要 runtime | 调试前端，或 Electron 装不上时 |
| `cd desktop && npm start` | Electron 装好 | 改 `main.js` / `preload.js` 时 |

> Electron 只负责“窗口 + 拉后端”，界面本身是 `app/web/` 里的静态文件。
> 改界面不用重启外壳，刷新窗口即可；只有改外壳代码才需要 `npm start`。

## 模型放在哪里

分成两类，**别混**：

### 一、`weights/` —— 推理权重（不是用户数据）

**平铺**：所有 `.onnx / .pth / .pt / .ubj` 直接放在 `weights/` 根下，文件名 = 模型名 + 原文件名。
**这个目录里没有任何代码**，也没有按模型分的子目录（`weights/onnx/<模型>/` 那套只作兼容兜底）。
权重本身不能执行，代码可以；所以为了安全，代码一律在 `app/` 里，`weights/` 只接受权重。
用户往这里丢对应的权重，那个功能就可用了；不丢就用不了。

```
weights/
├── tinymog.onnx                 检测器（core）
├── retinaface_10g.onnx         ⬆ 同目录，平铺
├── insightface_2d106.onnx       68 点关键点
├── arcface_mbf.onnx             人脸识别 / 身份聚类
├── errface_xgb.ubj              错脸检测（core）
├── bisenet_model_final.pth      人脸解析（**sam 组**，不装 sam 就没有）
├── sam2.1_hiera_*.pt            SAM2（sam 组，四个变体平铺）
├── groundingdino_swint_ogc.pth  Grounded SAM2 的检测头
├── sam_vit_b_01ec64.pth         SAM 点击分割
└── bert-base-uncased/           ← **唯一保留的目录**：BERT tokenizer 必须和权重同目录
```

**按需下载，不用的可以不要：**

| 包 | 内容 | 大小 | 必需？ |
|---|---|---|---|
| `core` | 检测器（`retinaface_10g.onnx` / `tinymog.onnx`，另带 15 个备用检测器）、`insightface_2d106.onnx`、`arcface_*.onnx`、`errface_xgb.ubj` 等 40 个文件 | ~1.4 GB | ✅ 必需 |
| `sam` | 遮罩绘制的交互分割：`sam_vit_b_01ec64.pth` / `sam2.1_hiera_*.pt` / `groundingdino_*.pth` / `bert-base-uncased/` / **`bisenet_model_final.pth`** | **~3.8 GB** | ❌ 可选 |

不装 SAM 那一组会怎样：**提取、训练、合成、自动遮罩全部照常**，
只有「工具 → 遮罩绘制」里 SAM / Grounded SAM2 / BiSeNet 三块是灰的。
只想要"检测 + 换脸"的用户完全不需要它。

下载方式：

```bat
python -m modelhub.fetch --group core     :: 必需（检测器、特征点、识别、XSeg）
python -m modelhub.fetch --group sam      :: 可选（遮罩绘制扩展，约 3.4 GB）
```

清单与校验值见 [weights/manifest.json](weights/manifest.json)，其中 `packages` 段说明了每个包解锁哪些功能；
`path` 字段就是平铺文件名（唯一带目录的是 `bert-base-uncased/`）。
想放到别的盘，用环境变量 `DFL_WEIGHTS_DIR` 指向新的根目录（布局仍然是平铺；
指向老的分目录布局也能用，`modelhub/weights.py` 的 `LEGACY_LAYOUT` 兜底）。

### 二、`workspace/model/` —— 你自己的模型

**布局规则（两条，别混）：**

```
workspace/model/
├─ 我的模型_SAEHD_data.dat        ← 换脸模型：直接平铺在根下
├─ 我的模型_SAEHD_encoder.pth
├─ 我的模型_SAEHD_decoder.pth
├─ 我的模型.dfm                   ← 合成用的导出件（平铺）
├─ XSegLite/                      ← 可训练的遮罩模型：各自一个子目录
│  ├─ XSegLite.onnx
│  └─ …
└─ XSeg/
   └─ XSeg.onnx
```

- **换脸模型（SAEHD / AMP / Quick96）**：文件平铺，一组 `_data.dat` + `.pth`，外加可选的 `.dfm`
- **可训练的遮罩模型（XSeg / XSegLite）**：一个模型一个文件夹，和换脸模型分开
- **合成只认 `.dfm`**：只有训练文件没有 `.dfm` 时，先在训练器里点「导出」
- 别人给你的、只有 `.dfm` 没有训练目录的模型，直接丢进根下即可，合成器会列出来

### 不放模型的地方

- `app/`、`desktop/` —— 代码目录
- 仓库根目录 —— 会让 `git status` 变脏（`workspace/`、`weights/` 已忽略）

## 目录结构

```
desktop/          Electron 外壳（窗口、菜单、无控制台启动）
app/
├─ serve.py       FastAPI 入口
├─ server/        后端，**一个界面一个包**（12 个）
│  ├─ _common/  _media/  _vision/        基础设施：env/路径/作业/设备 · 编码器 · 检测/对齐/遮罩几何
│  ├─ backend_merge/      合成器：pipeline_fused（融合导出）· stage_*（六阶段）· pipeline_project
│  ├─ backend_extract/    提取页：align_stream（唯一写 DFLJPG 的路径）· resume · frames · media
│  ├─ backend_engines/    引擎编译页：/api/engines · /api/engines/build
│  ├─ backend_bench/      速度测试页：/api/perf/bench
│  ├─ backend_models/     dfm 的 torch/ORT 实现、模型元信息（**没有路由**）
│  └─ backend_{faces,mask,bili,train,dataset,oneshot}/
│  app.py 只有 ~300 行：路由表 + 静态挂载 + PEP 562 转发
├─ web/           静态前端（原生 JS + CSS，无构建步骤）
│  ├─ index.html  单页多视图
│  └─ js/app/     **23 个分片**，后端按文件名顺序拼成 /static/js/app.js（逐字节等价）
├─ engine/        4.0 引擎（vendored：PyTorch 训练 + 合成管线 + modelhub）
├─ tools/         验收与工具：accept / bench / migrate / audit / js
└─ docs/          设计文档 00~35（编号递增）· legacy-reference/ 归档 · notes/ 实验记录
weights/          推理权重（平铺，见上）—— 不入库
workspace/        用户数据：数据集 / 模型 / 输出 —— 不入库
runtime/          Python 运行时（server / engine 两个 venv）—— 不入库
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

显卡：8 GB 显存可跑 128 分辨率 + batch 8（实测峰值 5.7 GB，约 6.5 it/s）。

## 性能（RTX 5070 Laptop 8 GB · 1080×1920 · 单脸 · 1439 帧）

| 项 | 数 | 说明 |
|---|---|---|
| 融合导出 | **23.6 帧/秒** | 解码→检测→对齐→dfm→XSeg→合成→编码 全在内存里串起来 |
| GPU 段 | 32.3 ms/帧 | 其中 dfm 20.6、xseg 9.5、等上游 6.8 |
| 编码 | NVENC（auto 默认） | 1080p 20.2→23.4 帧/秒；4K 17→48 帧/秒 |
| dfm | torch bf16w + CUDA Graph | 提交环 23.5→1.06 ms；b8 39.5→20.0 ms/张 |

- 默认值：`DFL_DFM_BACKEND=torch`、`DFL_DFM_DTYPE=bf16w`、`DFL_DFM_GRAPH=1`、`DFL_DFM_GRAPH_U8=1`、
  `DFL_FUSED_ENCODER=auto`（有 NVENC 就 NVENC，没有自动落回 x264）。
- **dfm 不做 fp16 / TRT**（实测没收益，还要编译）。全部开关见 [app/docs/27-ENV-KNOBS.md](app/docs/27-ENV-KNOBS.md)。
- 这台笔记本的 GPU 功耗墙是 60 W（上限 115 W），dfm 期 SM 只有 2.15~2.28 GHz（boost 3.09）——
  **剩下的空间主要在硬件，不在代码**。

## 改了代码怎么验

```bat
:: 1. 语法/导入
runtime\server\Scripts\python.exe -m compileall -q app\server
runtime\server\Scripts\python.exe -c "import sys;sys.path.insert(0,'app');import server.app"
node app\tools\js\check_bundle.js                 :: 前端分片拼起来是否仍是合法 JS

:: 2. 验收（真跑；按改动挑）
runtime\server\Scripts\python.exe app\tools\accept\fused_export_accept.py        :: 融合导出 67 项
runtime\server\Scripts\python.exe app\tools\accept\stage6_accept.py             :: 六阶段 64 项
runtime\server\Scripts\python.exe app\tools\accept\progress_lit_accept.py --shots-only   :: 进度页 79 项
runtime\server\Scripts\python.exe app\tools\accept\merge_ui_accept.py           :: 合成器界面 33 项
runtime\server\Scripts\python.exe app\tools\audit\api_contract_audit.py         :: 前后端协议对账（A/C 非空=失败）

:: 3. 像素与性能基线（改动合成/推理时）
runtime\server\Scripts\python.exe app\tools\bench\fused_pipeline_bench.py --frames 30 --modes B
::   30 个 PNG 的 md5 必须与 app/docs/notes/fused-baseline-hashes.txt 逐行相同
```

口径与原因写在 [app/docs/28-PIXEL-PARITY.md](app/docs/28-PIXEL-PARITY.md)：
**整片 fused 不是确定性函数**（攒批时序变→bf16 归约顺序变，同代码两遍 150 帧只有 62 帧相同），
所以判像素用 mode B（串行、固定批），判速度用多轮中位。

## 设计约定

配色、间距、圆角、字号收敛在 [UI设计规范.md](UI设计规范.md)；主题色只用于主按钮和进度条。

主要文档（都在 [app/docs/](app/docs/)）：

| 文档 | 内容 |
|---|---|
| [04-ACTUAL-STRUCTURE](app/docs/04-ACTUAL-STRUCTURE.md) | 现状目录树与已知结构债 |
| [06-MERGER](app/docs/06-MERGER.md) / [07-TRAINER](app/docs/07-TRAINER.md) | 合成器 / 训练器 |
| [21-PROJ-PIPELINE](app/docs/21-PROJ-PIPELINE.md) | 项目导出流水线 |
| [27-ENV-KNOBS](app/docs/27-ENV-KNOBS.md) | 全部环境开关（含默认值） |
| [28-PIXEL-PARITY](app/docs/28-PIXEL-PARITY.md) | 像素对拍口径：什么能比、什么不能比 |
| [33-API-CONTRACT](app/docs/33-API-CONTRACT.md) | 前后端协议对账（132 operation / 124 path） |
| [34-EXTRACT-THREE-PATHS](app/docs/34-EXTRACT-THREE-PATHS.md) | 三条「提取」链路各在哪、共用什么 |
| [35-LARGE-FILES-AUDIT](app/docs/35-LARGE-FILES-AUDIT.md) | 大文件审查与处置进展 |
| [notes/](app/docs/notes/) | 实验记录 · fused 基线与地图 · 清理记录 |

设计决策里最该先读的一条：**「效果优先于速度，但速度不许靠静默回落换」**——
任何“界面写一套、实际跑另一套”都要在日志和 [eff] 回显里喊出来，这条在 32 篇文档里反复出现。

Qt 时代的旧规划（01-PLAN / 02-STRUCTURE / 05-RESTRUCTURE-PLAN）在
[app/docs/archive/4.0/](app/docs/archive/4.0/)。

## 协议

GPL-3.0
