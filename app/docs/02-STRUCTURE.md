> ⚠️ **这份文档已经过期。** 它描述的是 Qt 版（app/ui/ + app/core/）的规划，
> 那个方案在改用 Electron + FastAPI + 原生 Web 之后被放弃。`app/ui/` 从未存在，
> `app/core/` 只剩空目录。当前实际结构见 `04-ACTUAL-STRUCTURE.md`。

---

# 4.1 目录结构

## 树

```
DeepFaceLab-PyTorch-4.1/
├── app/
│   ├── main.py                 入口（唯一的进程起点）
│   │
│   ├── ui/                     界面层：只有视图，没有算法
│   │   ├── theme/tokens.py     设计令牌：颜色 / 圆角 / 间距 / 字号
│   │   ├── widgets/            自研控件（不依赖任何 UI 框架）
│   │   ├── pages/              8 个页面
│   │   └── bridge/             Qt ←→ core 的唯一通道：QThread + Signal
│   │
│   ├── core/                   后端：全部业务逻辑
│   │   ├── vision/             FaceType、对齐、遮罩、图像算子（4.0 的 facelib）
│   │   ├── data/               样本加载与增强（4.0 的 samplelib）
│   │   ├── models/             SAEHD / DF / LIAE / DFLite 网络与训练循环
│   │   ├── extractor/          提取器
│   │   ├── processor/          过滤器 / 排序器 / 角度 / 遮罩应用
│   │   ├── merger/             合成与导出
│   │   ├── tools/              遮罩绘制、B站下载、FFmpeg 色域迁移
│   │   └── runtime/            配置、端口、日志、生命周期
│   │
│   ├── modelhub/               检测器与特征点模型（只留 3 个）
│   └── docs/                   给开发者 / Agent 看的文档
│
├── workspace/                  用户数据，gitignore
│   ├── data_src/  data_dst/    对齐后的人脸集
│   ├── model/                  训练出来的模型
│   └── output/                 合成结果
│
└── _python/                    便携运行时，gitignore
```

## 已有文件（M0 完成时的状态）

```
app/main.py                      入口：QApplication + 主题 + 主窗口
app/ui/theme/tokens.py           设计令牌（颜色/圆角/间距/字号/动效）
app/ui/theme/style.py            令牌 → 全局 QSS
app/ui/widgets/icons.py          SVG 加载 + currentColor 染色
app/ui/widgets/base.py           Card / PageTitle / Hint / Tag / HLine
app/ui/widgets/buttons.py        Primary / Ghost / Mini / Text
app/ui/widgets/controls.py       Segmented / NumberStepper / Chip / ChipGroup / Bar
app/ui/widgets/panels.py         StatCard / ParamCell / ParamGrid / FieldRow / ModelRow / Toast
app/ui/shell/titlebar.py         无边框标题栏（三圆点 = 关闭/最小化/最大化）
app/ui/shell/sidebar.py          品牌 + 导航 + 工具分组
app/ui/shell/window.py           MainWindow：侧边栏 + QStackedWidget
app/ui/pages/home.py             主页（公告 + 三步 + 继续上次）
app/ui/pages/extractor.py        提取器（路径 / 参数 / 进度）
app/ui/pages/dataset.py          数据处理（模块 → 模式 → 参数 三级渐进）
app/ui/pages/trainer.py          训练器（3 列模型卡）
app/ui/pages/merge.py            合成器（服务开关 + 模型导出列表）
app/ui/pages/tools.py            工具（遮罩绘制 / B站 / 色域迁移）
app/ui/pages/settings.py         设置（端口 / Cookie / 代理）
app/ui/pages/about.py            关于
app/ui/assets/icons/*.svg        20 个线性图标
```

## 四条硬规则

**1. UI 层不许 import torch。**
`app/ui/**` 里出现 `import torch` 就算 bug。4.0 的 `WinError 1114 / c10.dll` 崩溃
就是页面模块经 `core.leras` 把 torch 拖进 Qt 进程炸掉的。torch 只在
`app/core/**` 内部、且只在工作线程真正开始干活时才导入。

**2. UI 与 core 之间只有一座桥。**
`app/ui/bridge/` 是唯一允许同时看到两边的地方。页面不直接调 core 的函数，
而是提交一个任务给 bridge，拿回一个信号流。

**3. 跨层依赖方向单向。**
`ui → bridge → core`，永不反向。core 不知道 Qt 的存在，
所以 core 可以被命令行、被脚本、被测试单独驱动。

**4. 根级 `docs/` 永久忽略。**
4.0 曾经把设计文档误传到公开仓库。4.1 里文档只在 `app/docs/`，
根级 `docs/` 写进了 `.gitignore`。

## 与 4.0 的映射

| 4.0 | 4.1 |
|---|---|
| `facelib/` | `app/core/vision/` |
| `samplelib/` | `app/core/data/` |
| `models/` | `app/core/models/` |
| `Extractor/` | `app/core/extractor/` |
| `FacesetProcessor/` | `app/core/processor/` |
| `MergeStudio/` + `mainscripts/` | `app/core/merger/` |
| `MaskProcessor/` | `app/core/tools/maskeditor/` |
| `modelhub/` | `app/modelhub/` |
| `ui/` (PyQt5 + siui) | `app/ui/` (PySide6 + 自研控件) |
| `WebUI/` | 后续迁进 `app/ui/pages/` |
