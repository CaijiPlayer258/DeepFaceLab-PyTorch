# 4.1 目录重构方案

> 现状见 `04-ACTUAL-STRUCTURE.md`。`02-STRUCTURE.md` 是 Qt 时代规划，已作废。
> 本文是**待执行**方案，不是已完成状态。

## 硬数字（核实过）

| 项 | 数字 |
|---|---|
| `app/server/app.py` | 71,991 B / 2045 行 / 54 个 `@app.*`（53 路由 + 1 middleware）|
| `app/web/js/app.js` | 196,446 B / 4904 行 / 4 个顶层 IIFE / 89 处 `window.X =` |
| `index.html` | 968 行 / 15 个 `.page` |
| `app.css` | 3997 行 / 78 个分节 |
| `app/core/` | 8 个空目录 / 0 文件 / 无 `__init__.py` |
| `app/web/_parts/` | 16 文件 / **0 处引用** |
| `/static/fonts` 绝对 URL | 210 处（noto.css 202 + app.css 4 + theme-41.css 4）|

## 八个结构问题

**P1 `app/core/` 是烂尾不是设计**
8 个空目录来自 `02-STRUCTURE.md`（那份文档自己都写着"已过期"），连 `__init__.py` 都没有 —— 从未被 import 过。文档里配套的 `app/ui/` 也从未存在（Qt 那套在 `legacy/qt-ui/`）。

**P2 `app/modelhub/` 空，但代码从别处找模型**
`app.py:301-315` 从 `ROOT.parent/"DeepFaceLab-PyTorch"/modelhub` 和硬编码 `D:/...` 找 RetinaFace —— **换机器必 404**。

**P3 `workspace.txt` 让 4.1 往 4.0 仓库写盘**
`settings.json` / `auth.json` / `_mask_progress.json` 全落在 4.0 那边。

**P4 `app.py` 一个文件扛 7 类职责**
静态挂载 + 中间件（915-930）、工作区（23-34）、4.0 子进程（323-380 / 530-660 / 1429-1520）、ONNX 检测（298-315）、B站（933-1035）、ffmpeg（41-79）、SSE。
5 个模块级可变字典：`_merge_cache` / `_export_job` / `_det_cache` / `_remote` / `_bili_session`。
还有 `threading as _threading / _th2 / _th3` 三个别名。

**P5 `app.js` 4904 行 89 个隐式 window 全局**
第一个 IIFE 独占 3843 行，连 `window.__cfgStartRetry` 都当 API 用。

**P6 `_parts/` 里 8 个是过期重复件（另 3 个性状待定）**
`merge-tracks.js` 头部写着"勿改 app.js"，实际已**逐字**内联进 `app.js:3848-4294`；`merge-pages.js` → `4297-4566`；`train-params.js` → `4570-4860`；CSS 也内联进 `app.css` 的 2239/2510/2717/2747 四节。只有 5 个 `.md` 有独有内容。

补测（200 字窗口逐字比对，见 `app/web/_parts/README.md`）：`mask-draw.html` 和
`mask-draw.js` 在 `app.js`/`app.css`/`index.html`/`mask-editor/index.html` 里都找不到逐字
副本，`merge-pages.html`/`merge-tracks.html` 只进去了一部分 —— 这 4 个不能直接删。

**P7 mask-editor 是"伪独立"**
反向依赖父应用的 `/static` 字体（`theme-41.css:24/31/38/45`），又被 `/static` 二次挂载（`app.py:920+921`），且 `api.js` 调 4 条后端不存在的接口。

**P8 三份旧东西并存 + gitignore 漏网**
`legacy/qt-ui/`（目录名带连字符不可 import、`bridge/` 空、`__pycache__` 还留着 `app.ui` 时代字节码）、`app/main.py`（pywebview，与 `serve.py` 端口逻辑重复，无人引用）、`app/server/_bili_mock.py`（独立 FastAPI，无人 import）。
`.gitignore` 缺 `node_modules/`；第 9 行 `/docs/` 会吞掉根级 docs。

## 目标结构

```
backend/                        ← 原 app/server/
├── serve.py                    后端入口（Electron fork 它）
├── config.py                   PROJECT_ROOT / FRONTEND_DIR / WORKSPACE / DFL40_ROOT / TRAINER_BOOT
├── api/                        只做请求解析 + 调用 service
│   ├── __init__.py             create_app() + 静态挂载 + middleware
│   ├── state.py  dataset.py  extract.py  train.py
│   ├── merge.py  models.py   mask.py     bili.py
├── services/                   业务逻辑，不碰 Request/Response
│   ├── workspace.py  jobs.py（5 个可变字典收这里）
│   ├── dfl40.py（所有 4.0 子进程）  video.py
│   ├── detect.py（ONNX）  trainer_boot.py  bili_auth.py
├── modelhub/                   ONNX 模型（从 4.0 复制过来，别指向外部）
├── tools/mock_bili.py, route_dump.py
└── tests/

frontend/                       ← 原 app/web/
├── index.html
├── css/{app,shadcn,noto}.css
├── js/
│   ├── core/{api,state,dom,toast,theme,window-chrome}.js
│   ├── pages/                  15 个页面，一个文件一个
│   ├── widgets/{merge-timeline,merge-pages,train-extra}.js
│   ├── i18n.js
│   └── bootstrap.js            统一加载顺序
├── img/  fonts/
└── mask-editor/

desktop/                        位置不动
docs/                           01-PLAN / 02-ACTUAL / 03-RESTRUCTURE + history/
legacy/                         qt-ui / pywebview-main.py / README
workspace/  _python/
```

**关键取舍：静态前缀保持 `/static` 不变。**
这样 `index.html` 10 处绝对引用 + 210 个字体 URL + `app.js:135/1358` 的 iframe src **全部零改动**。是在"前后端分离"和"不制造 210 处 404"之间的正确折中。

## 引用连带清单（迁移前必须背下来）

| 位置 | 改动 |
|---|---|
| `serve.py:12-14` | `sys.path` + `from server.app` → `from backend.api` |
| `serve.py:36` | `_remote_boot` → `api.state` |
| `app.py:20` | `WEB = parent.parent/"web"` → `config.FRONTEND_DIR` |
| `app.py:21` | `ROOT = parents[2]` → `config.PROJECT_ROOT`（在 `api/__init__.py` 里 `parents[2]` 恰好还对 —— 这正是必须删掉它的理由）|
| `app.py:301-315` | 检测器路径 → `config.MODELHUB_DIR` |
| `app.py:920-922` | 两个 mount（保留 `html=True`）|
| `app.py:1460` | `Path(__file__).parent/"_dfl_train.py"` → `config.TRAINER_BOOT` |
| `_dfl_train.py:32` | 硬编码 `DFL_ROOT` → `config.DFL40_ROOT` |
| `main.js:17` | `SERVE=ROOT/app/serve.py` → `ROOT/backend/serve.py` |
| `main.js:27` | `cwd=ROOT/app` → `ROOT`（代码不得依赖 cwd）|
| `RunGUI.bat:12/13/18` | 不用动 |
| `index.html` / `noto.css` / `app.js:135/1358` | 不用动（`/static` 不变）|

## 七步迁移（每步做完项目仍可运行）

**S0 安全网**
`git init`（现在**根本没有 .git**，不建仓无法回滚）+ 补 `.gitignore` + 用 `app.routes` 导出 54 条路由做基线快照 + TestClient 冒烟 10 条。不改结构。

**S1 抽 config.py**（纯新增）
`PROJECT_ROOT` / `BACKEND_DIR` / `FRONTEND_DIR` / `WORKSPACE` / `SETTINGS_FILE` / `DFL40_ROOT` / `TRAINER_BOOT`。`app.py` 顶部 4 个常量改为引用它。
验：`/api/state` 的 workspace 字段。

**S2 拆后端**（文件仍留 `app/` 下，路径不变）
`app.py` 退化为 `create_app` 的兼容壳，`serve.py` 一行不改。按路由建 `api/` 9 文件 + `services/` 6 文件，5 个可变字典收进 `services/jobs.py`，subprocess 全收进 `services/dfl40.py`。
验：路由快照 diff + 6 个主页面点一遍。

**S3 物理改名 `app/server` → `backend/`**（一次机械改）
同步改 `serve.py`、`main.js:17/27`、`api/__init__` 的配置引用、`trainer_boot` 路径。
验：`RunGUI.bat` 能起、`_electron_backend.log` 里有 `PORT=`。

**S4 `app/web` → `frontend/`**
只改 `config.FRONTEND_DIR` 与两个 mount 的 `directory`。
验：woff2 无 404、mask-editor iframe 打开。

**S5 拆前端**（保留 window 契约，不动调用点）
先 `core/api.js` 收口 39 处 fetch → `core/state.js` 收共享状态 → 按 15 个 page 拆 → widgets 三片段照搬 → `bootstrap.js` 统一顺序 + 收编 `app.js:4839-4859` 的 `openCfg` 包装。

**S6 清理**
删 `_parts/`、`core/`；`main.py` → `legacy/pywebview-main.py`；`_bili_mock` → `backend/tools/mock_bili.py`；`modelhub` 迁移；docs 归位；删 `_shots`/`_trash`/`workspace.txt`。

**S7（可选，别塞进结构迁移）**
mask-editor 4 条死接口、字体去绝对化、4.0 硬依赖收敛到 `config.DFL40_ROOT` 一处。

## 风险清单（症状 → 怎么验）

| # | 风险 | 验证 |
|---|---|---|
| R1 | `parents[N]` 巧合失效 → settings.json 写到 4.0、前端 404 | `/api/state` 的 workspace + 启动日志打印全量路径 |
| R2 | 迁移测试期仍写 4.0 workspace | 先删 `workspace.txt` 或 `DFL_WORKSPACE` 指向 4.1/workspace |
| R3 | 换静态前缀 → 210 个字体 URL 静默回落系统字体（**不报错**）| Network 过滤 woff2 计数 198+4，比对标题字重 |
| R4 | 拆 router 漏注册 / 漏 middleware | 路由快照 diff；`curl -I /` 看 no-store |
| R5 | 5 个共享状态被复制多份 → 进度永远 0、导出永远 running、扫码拿不到 key | 训练/导出/扫码各跑一次 |
| R6 | 脚本顺序错 → `window.TRAIN_EXTRA_GROUPS` 在 4857 才赋值而 1963 就 push（17 个训练参数永不渲染）| 训练参数页数出 69 个键 |
| R7 | `html=True` 丢失 → iframe 变目录列表 | 直接开 `/mask-editor/` |
| R8 | Electron 链路 → 窗口不弹 / backend timeout 25s | `_electron_backend.log` 有 `PORT=`、退出无残留 pythonw |
| R9 | 相对导入断 → **ImportError 让后端进程直接退出**（不是 500）| `python -c "import backend.api"` 必须先过 |
| R10 | trainer_boot 子进程命令行 → 训练点了没反应 | 看 `/api/train/start` 生成的 cmd 与进程表 |
| R11 | `git init` 首提交塞进 348MB node_modules | `git status --short` 行数、`git count-objects -vH` |
