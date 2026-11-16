# 4.1 当前真实结构

> 这份文档描述**代码现在长什么样**，不是理想状态。理想方案见
> `archive/4.0/05-RESTRUCTURE-PLAN.md`；`archive/4.0/02-STRUCTURE.md` 是 Qt 时代的旧规划，已作废。

## 进程拓扑

```
Electron (desktop/main.js)
  └─ fork: pythonw app/serve.py          ← 只跑后端，stdout 打 PORT=xxxx
       └─ uvicorn: FastAPI (app/server/app.py)
            ├─ GET  /                      → app/web/index.html
            ├─ /static/*                   → app/web/{css,js,fonts,img}
            └─ /api/*                      → 业务接口

（pywebview 时代的 app/main.py 已经不存在了；4.0 的入口是 app/engine/main.py）
```

## 目录

```
app/
├── serve.py            后端独立入口（Electron fork 的就是它）
├── server/             FastAPI 业务
│   ├── app.py          单文件 70KB —— 所有路由 + 业务都在这
│   ├── _vision/        检测/对齐/XSeg 的实身（detect.py = RetinaFace ONNX）
│   ├── _dfl_train.py   训练包装：跑 4.0 的 main.py，禁掉自动弹浏览器
│   ├── backend_bili/   B 站扫码登录 / 投稿 / 下载（placeholder.py 是 DFL_BILI_MOCK 那套假数据）
│   └── backend_*/      其余业务按域分包（顶层 _xxx.py 都是 PEP 562 转发壳）
├── web/                前端（StaticFiles 挂这里）
│   ├── index.html      53KB 单页
│   ├── js/app.js       190KB 单文件
│   ├── js/i18n.js      中英词典 + 正则模板
│   ├── css/            app.css / shadcn.css
│   ├── mask-editor/    遮罩绘制（copied from 4.0，独立子应用）
├── core/               （已删：Qt 时代规划的空目录树，0 文件）
├── engine/             4.1 引擎（xlib / core / models / MergeStudio / facelib / modelhub …）
├── tools/              验收与迁移脚本（accept / bench / migrate）
└── docs/               设计文档（01/02/05 已移入 docs/archive/4.0/）

desktop/                Electron 壳
workspace/              用户数据
```

## 已知的结构债

1. `app/server/app.py` 70KB 单文件，2000 行，路由和业务混在一起
2. `app/web/js/app.js` 190KB 单文件，4898 行，靠 `window.*` 互相挂钩
3. ~~`app/core/` 整棵树是空的~~ —— 已删（8 个空目录，0 文件，从未被 import）
4. `app/tools/parts/` 里 11 个早期拆分稿（3 js + 5 css + 3 html）：8 个已逐字内联进
   `app.js` / `app.css`，`mask-draw.html` / `mask-draw.js` 两份还没有逐字副本、
   `merge-*.html` 只内联了一部分，删之前要逐个确认（清单在 `app/docs/notes/README.md`）
5. 4.0 那套 Qt 界面 `legacy/qt-ui/` 已归档到 `app/docs/legacy-reference/qt-ui/`
   （58 个跟踪文件，从未被 import；pywebview 时代的 `app/main.py` 早年就删了）
6. `archive/4.0/02-STRUCTURE.md` 描述的 Qt 方案与代码完全不符
