# 08 · 后端路由地图（app.py 拆分进度）

> 目标：把 4241 行、67 个路由的 `app/server/app.py` 按域拆成 `routes_*.py`。
> 每拆一块都要跑一次「起服务 + 打关键接口」的断言，绿了才提交。

## 拆分约定

1. 每个域一个 `app/server/routes_<域>.py`，模块里只放自己的 `APIRouter` 和**自己的**模块级状态。
2. 共享的名字（`WORKSPACE`、`_get_detector` 等）用 `from .app import (...)` ——
   注意 `app.py` 里那行 `include_router` 必须放在所有 helper **之后**（文件末尾），否则半初始化导入会失败。
3. **自己在块里定义的名字不要从 .app 再 import**（第一次拆 face 就踩了这个坑，会 ImportError）。
4. 拆完在 `app.py` 原位置留一行注释 + 两行 include。

## 当前进度（拆分到 25/67）

| 模块 | 路由 | 内容 |
|---|---|---|
| `routes_bili.py` | 5 | B 站扫码 / 轮询 / 下载 / 账号 / 退出 |
| `routes_face.py` | 3 | 面部 68 点 |
| `routes_trt.py` | 8 | 引擎编译 + 提取（含 SSE `/api/extract/progress`） |
| `routes_account.py` | 4 | `/api/auth*` |
| `routes_oneshot.py` | 5 | `/api/dataset/{oneshot,scan,apply,restore}` |

`app.py`：4241 → 3679 行。OpenAPI 路径总数始终 **68**（每块拆分前后 diff 为空）。

## 剩余 42 条的准确边界（当前行号）

| 行范围 | 路由 | 内容 | 建议模块 |
|---|---|---|---|
| 1–1542 | 30 | 合成器（prepare/open/frame/export/detect/analyze）+ 遮罩（predict/save/progress）+ 模型 config/export | `routes_merge.py` + `routes_models.py`（要把夹在中间的模型 config/export 单独拎出来） |
| 1592–2767 | 13 | 模型库（model_spec/gpu/create）+ 训练控制（start/save/pause/resume/stop/preview/status） | `routes_models.py` + `routes_train.py` |
| 2768–2848 | 1 | 数据访问 | 并到 models 或独立 |
| 2849–3649 | 3 | 演示任务 / 提取（含大段 helper） | `routes_demo.py` |

## ⚠ 两个已经踩过、必须记住的坑

1. **验收指标**：`include_router` 之后 `app.routes` 里放的是 `_IncludedRouter` 容器，不再逐条列出子路由。
   用 `len(app.routes)` 判断会得到「少了 5 条」的假警报（为此白回滚过一次）。
   **唯一正确的等价判断：比对 `app.openapi()['paths']` 集合**。

2. **模块头必须 `import asyncio`**：SSE 路由（`/api/extract/progress`）里 `await asyncio.sleep()`，缺了直接 500。

## 拆分脚本要点

- 依赖用 `symtable` 求自由变量（不能用正则，字符串里的 `st["models"]` 会被误判）；
- 自由变量若定义在别的 `routes_*.py`，生成 `from .routes_x import ...`；
- `'... %s ... %s ...' % (n, n)` 这种占位符数量不匹配会让脚本**静默**退出（PowerShell 还会吞掉 traceback）——
  用 `*>` 重定向到文件再看。

## 前端为什么不拆（已定，别再来一次）

后端 `app.py` 拆成 9 个 `routes_*.py` 是净收益：FastAPI 的 router 有明确边界，
接口的输入输出就是契约，拆完能用 `app.openapi()['paths']` 做严格等价验证。

**前端不拆**（`app/web/js/app.js` 保持单文件，7179 行）。原因：

1. **没有构建步骤**——直接 `<script src>`，拆开后没有模块系统兜底，全靠加载顺序；
2. **全局作用域互相调用**——`window.pollTrain`、`window.openRun`、`window.TrainExtra`、
   事件委托（`document.addEventListener` 里按 id/class 分发）跨页面共享，
   拆开后很容易出现「A 还没定义 B 就用上了」的时序 bug；
3. **JS 没有类型**——不能像后端那样用 AST/OpenAPI 做严格等价检查，
   拆错的后果是「某页点一下没反应」，而这恰恰是最难发现、最容易漏测的一类；
4. 收益不匹配：后端拆完每块几百行、读起来是顺的；前端拆完每页也只有几百行，
   但多出十几个文件 + 一张隐式的加载顺序表，维护成本反而更高。

**如果只是觉得难读**，用零风险的办法：在 `app.js` 顶部维护一份分节目录（注释），
配合已有的 `/* ---------- 分区 ---------- */` 横幅，靠搜索跳转即可，不动代码结构。
