# DeepFaceLab 4.1（测试版）

> **状态：开发中，功能未定型，不建议当作日常工具使用。**
> 这个分支是 4.1 的开发线，随时会改接口、改目录、推翻交互。
> 稳定版请看 master（4.0）。

4.0 是技术样机，4.1 是产品。

**目标**：一个普通人打开就会用的人脸替换工具 —— 没有命令行窗口，没有一大堆用不上的模型，
每个页面只有一个主按钮。

## 架构

    desktop/        Electron 外壳（窗口、菜单、无控制台启动）
    app/
    ├── serve.py    FastAPI 后端入口
    ├── server/     后端：提取 / 对齐 / 训练 / 合成 / 工具
    ├── web/        静态前端（原生 JS + CSS，无构建步骤）
    ├── modelhub/   检测器与特征点模型
    └── docs/       规划、目录说明、界面讨论稿
    workspace/      用户数据（数据集 / 模型 / 输出）—— 不入库

界面是 Web 技术栈，跑在 Electron 里；后端是本地 FastAPI（默认端口 8764）。
不依赖任何前端框架，改完刷新即可，没有打包步骤。

## 快速开始

双击根目录的 **RunGUI.bat**（不会留下控制台窗口）。

首次使用如果提示缺 Electron：

    cd desktop
    npm install

只跑后端、用浏览器打开：

    python app/serve.py

## 需要的环境

- Python 3.12（内置运行时在 _python/，不入库）
- PyTorch（CUDA 版，训练与合成用）
- onnxruntime-gpu（检测器与特征点推理）
- Node.js（只在第一次装 Electron 时需要）

## 设计约定

界面配色、间距、圆角、字号统一收敛在 [UI设计规范.md](UI设计规范.md)，
主题色只用于主按钮和进度条，其余保持中性灰阶。

详细说明见 [app/docs/02-STRUCTURE.md](app/docs/02-STRUCTURE.md)，
产品规划见 [app/docs/01-PLAN.md](app/docs/01-PLAN.md)。

## 协议

GPL-3.0
