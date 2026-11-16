# DeepFaceLab 4.1 — 规划

> 一句话：**4.0 是技术样机，4.1 是产品。**
> 4.0 证明「这些事都能做」，4.1 只回答「做哪几件、怎么让人敢用」。

讨论稿配套：`docs/00-ui-mockup.html`（用浏览器打开，右上角可切深浅色）。

---

## 0. 先钉死三条产品原则

| 原则 | 具体落地 |
|---|---|
| **只留一条主路径** | 每个页面只有一个实心主按钮。分支靠档位（高质量/快速），不靠选项。 |
| **零命令行** | 任何用户操作都不弹 cmd。长任务 = 后台线程 + 进度条 + 指标回传；WebUI 服务 = 同进程线程，端口可配。 |
| **模型是资产，界面是消耗品** | 4.0 训练出来的 `.pth` / `.npy` 必须能被 4.1 直接加载，权重格式一行不改。 |

---

## 1. 事实核验（写在这里避免以后又猜）

这四条我都在 4.0 源码里查过了，是规划的前提，不是假设：

| 结论 | 证据 |
|---|---|
| TinyMog 输入 640×640 | `modelhub/onnx/TinyMog/TinyMog.py:31` `INPUT_SIZE = 640` |
| RetinaFace_10g 输入 640×640 | `modelhub/onnx/RetinaFace/RetinaFace.py:124,128` `_detect_candidates(..., input_size=640)`，注释「TRT engine 固定 640×640，强制 640 尺寸」 |
| InsightFace2D106 是**特征点模型**不是检测器，输入 192×192 | `modelhub/onnx/InsightFace2d106/InsightFace2D106.py:50` `self._input_height = 192` |
| 遮罩绘制 / 合成器的端口输入框是**死的** | 界面把值写进 `webui_settings.json`（`page_maskprocessor.py:88-97`、`page_mergestudio.py:266-276`），但 `MaskProcessor/__main__.py:25` `port = 8000`、`MergeStudio/__main__.py:22` `PORT = 8000` 都是写死的，两个 `__main__` 都没有 argparse |

所以「三个模型」的准确说法是：**两个检测器（TinyMog / RetinaFace_10g）+ 一个特征点器（InsightFace2D106）**。高质量档 = RetinaFace_10g，快速档 = TinyMog，特征点器两档共用。

---

## 2. 目录结构

```
DeepFaceLab-PyTorch-4.1/
├── app/
│   ├── ui/                     # 只有视图，不含算法，绝不 import torch
│   │   ├── widgets/            # 自研控件：Card / Stat / Bar / Sidebar / Segmented / TitleBar
│   │   ├── theme/              # 设计令牌：颜色、圆角、间距、字号（tokens.py）
│   │   ├── pages/              # home / extractor / dataset / trainer / merger / tools / settings / about
│   │   └── bridge/             # Qt ←→ core：QThread worker、进度信号、取消令牌
│   ├── core/                   # 全部业务
│   │   ├── vision/             # 4.0 的 facelib：FaceType、align、mask、图像算子
│   │   ├── data/               # 4.0 的 samplelib：样本加载、增强
│   │   ├── models/             # 4.0 的 models：SAEHD / DF / LIAE 网络与训练循环
│   │   ├── extractor/
│   │   ├── processor/          # filter / sorter / angle / maskapply
│   │   ├── merger/
│   │   ├── tools/              # 遮罩绘制、B站下载、FFmpeg 色域迁移
│   │   └── runtime/            # 配置、端口、日志、进程与生命周期
│   ├── modelhub/               # 4.0 的 modelhub（只留需要的模型）
│   └── docs/                   # 给开发者/Agent 的文档
├── workspace/                  # 用户数据：data_src / data_dst / model / output（gitignore）
└── _python/                    # 内置运行时（gitignore）
```

**和 4.0 的映射**：`facelib→core/vision`、`samplelib→core/data`、`models→core/models`、`Extractor→core/extractor`、`FacesetProcessor→core/processor`、`MergeStudio+mainscripts→core/merger`、`MaskProcessor→core/tools`、`modelhub→app/modelhub`。

**硬规则**：`app/ui/**` 里出现 `import torch` 就算 bug。4.0 的 `WinError 1114 / c10.dll` 崩溃就是 UI 页面经 `core.leras` 把 torch 拖进 Qt 进程导致的，这条规则从结构上根除它。

---

## 3. 技术栈决定

- **GUI：PyQt5（沿用）或迁 PySide6。** 见开放问题 Q6。
- **不整体套用 PyQt-SiliconUI-main。** 只把它当控件参考。4.1 自研一套最小控件集（约 10 个），全部走 `theme/tokens.py` 里的变量，这样才能做到「改一个文件换整套配色」。
- **长任务模型：** 一律 `QThread` + `pyqtSignal`。后端函数形如 `def run(input, output, quality, angle, skip, on_progress, cancel_token)`，**不打印到控制台当唯一反馈**。
- **WebUI 服务：** uvicorn 在子线程里跑，`port` 由 settings 注入（`uvicorn.run(app, host="127.0.0.1", port=port)`），端口被占用时自动顺延到下一个空闲端口并在设置页显示实际端口。
- **子进程：** 只有 FFmpeg 一处，且必须带 `CREATE_NO_WINDOW`。

---

## 4. 模块规格

### 4.1 提取器

界面只有五项：**输入路径 / 输出路径 / 质量档 / 检测角度 / 帧跳跃**。

写死的东西：One-Stage、LetterBox、640×640、KPS 对齐开启、无 HDR、无预缩放、无数帧抽取 UI。

输入判定只有一条规则：**路径是文件 → 单文件；路径是目录 → 目录下所有图片和视频全收**，不再问模式。

混装目录的调度：**先图片，后视频**。理由你已经说了——一个视频比一张图慢两个数量级，把图片先清完，进度条前段推进得快，用户不会觉得卡住。

回传三个指标（一个进度条 + 三张卡片）：

| 指标 | 定义 |
|---|---|
| 检测帧速度 fps | 已解码并送进检测器的帧数 / 秒（**包含被帧跳跃跳过的不算**，跳过的帧不解码） |
| 输出脸速度 张/秒 | 实际写入磁盘的对齐图 / 秒 |
| 文件进度 | 已完成文件 / 总文件；混装时同时显示「图片 x/y · 视频 m/n」 |

> **需要拍板**：帧到底落不落盘？我的建议是**不落盘**——解码 → 检测 → 对齐 → 写脸，一遍过。代价是以后想换参数重新对齐，必须重新解码视频。如果你更看重「可反复重跑对齐」，就在 `workspace/cache/<视频名>/` 里静默存帧，界面上永远不提它。

### 4.2 数据处理

四个模块：**过滤器 / 排序器 / 角度可视化 / 遮罩应用**。数据分析器只留后端能力，不出现在任何界面。

排版建议见 mockup 第 03 节：**空状态用「打开一个文件夹以继续」，选中之后塌陷成顶部路径条 + 四模块工作台**（A 的直白 + B 的可参数化）。

### 4.3 训练器

不分 SAEHD / AMP / DF / LIAE，全部混排，**3 列瀑布流卡片**。卡片：大号模型名 → 架构一行小字 → 参数 tag → 操作按钮。点卡片弹出参数浮层；训练中的卡片显示预览缩略图和当前 iter。

### 4.4 合成器

启动是主按钮，导出收进各自模型卡片（mockup 第 05 节方案 B）。MergeStudio 以同进程线程启动，端口来自设置。

### 4.5 工具

遮罩绘制、B 站下载器合到「工具」栏目。FFmpeg 杜比/ HDR → SDR 转换脚本也放这里，做成一个小工具卡：选文件 → 选目标色域 → 跑 → 输出到同目录 `_sdr`。

### 4.6 设置

三项：三个 WebUI 端口、B 站 Cookie 管理（含清除/重新登录）、GitHub 代理地址。
端口必须真的接上（当前 4.0 是死的，见第 1 节）。

### 4.7 主页 / 关于

关于保持 4.0 风格。主页这一轮不改，等前面几块稳定了再单独做。

---

## 5. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0 骨架** | 目录、启动器、theme tokens、10 个自研控件、8 个空页面路由、自定义窗口按钮 | 能启动、能切页、**全程无 cmd 窗口**、UI 进程内 torch 未被导入 |
| **M1 提取器** | 三模型接入、混装调度、四指标回传、取消 | 混装目录跑通，指标跳动正常，取消能立刻停 |
| **M2 数据处理** | 四模块 + 路径条 | 四个模块都能对同一目录跑完并落盘 |
| **M3 训练器** | 卡片墙、参数浮层、训练线程桥、预览 | 能新建/续训/暂停/删除，loss 与预览实时出 |
| **M4 合成器** | 卡片墙、启动、DFM/ONNX 导出、端口参数化 | 端口改 8001 真生效 |
| **M5 工具 + 设置** | 遮罩绘制、B 站、Cookie、代理、端口 | 三项设置全部真实生效 |
| **M6 收尾** | 主页、打包（`_python`）、4.0 模型迁移自测 | 拿一个 4.0 训好的模型在 4.1 里能直接合成 |

M0 和 M1 是分水岭：这两个跑通，剩下的都是重复劳动。

---

## 6. 已定事项

| 问题 | 决定 |
|---|---|
| 施工顺序 | **目录结构 → UI 布局 → 后端接入**。4.0 的后端改接口就能用，所以后置。 |
| 数据处理排版 | **混合**：空状态是「打开一个文件夹以继续」，选中后用动画过渡到顶部路径条 + 四模块工作台。 |
| 帧落盘 | **不落盘**。解码 → 检测 → 对齐 → 写脸，一遍过。换参数重跑对齐需重新解码。 |
| 检测不到脸的帧 | **直接跳过**（沿用 4.0 行为，不生成占位图）。 |
| 训练架构 | **全留**：SAEHD / DF / LIAE / DFLite / AMP。 |
| GUI 框架 | **PySide6**。 |
| 打包 | **便携包 `_python`** 内嵌解释器。 |
| 合成器排版 | **定了**：模型在 WebUI 里才选，桌面端**没有「当前选择」这个概念**。页面 = 一张服务卡（状态点 + 启动/停止按钮）+ 一份「模型导出」列表（名字 + 架构 + 导出状态 + 导出按钮），列表行不可选中。 |
| 图标 | 一套 20 个线性 SVG，落在 `app/ui/assets/icons/`，24×24 / 1.6 描边 / currentColor。 |
| 提取器按钮位置 | **B：参数卡底部通栏**。点击后按钮原地转禁用态（底 #E7E7EA、字 ink-3、not-allowed），文字变「正在提取…」，进度条接管反馈。 |
| 页面说明文字 | **一律不写**。卡片头只留标题；参数项不加括号注解；页面标题上下不加面包屑与副标题。 |
| 设计令牌 | 统一收敛到根级 [UI设计规范.md](../../UI设计规范.md)，Qt 侧落到 `app/ui/theme/tokens.py`。 |

---

## 7. 明确砍掉的东西（给以后翻旧账的人看）

- 21 个 ONNX 模型里的 18 个：ArcFace、CenterFace、DamoFD、FAN、FaceMesh、GenderAge、InsightFace3D68、LIA、MTCNN、MogFace、S3FD、ULFD、YoloV11nFace、YoloV5Face、YoloV8Face、LightweightFD、PFLD、OpenSeeFace、BlazeFace、FastFaceAlign、FaceEnhancer、FaceParser（保留 TinyMog / RetinaFace / InsightFace2d106）
- 数帧抽取（作为独立功能）
- HDR 模式（改为独立 FFmpeg 工具）
- 滑动窗口 / One-Stage 选择
- LetterBox / WarpAffine 选择
- 预缩放参数
- KPS 对齐开关
- 数据分析器界面
- 每个页面的独立端口输入框（收进全局设置）
