# 4.1 当前真实结构

> 这份文档描述**代码现在长什么样**，不是理想状态。理想方案见 `05-RESTRUCTURE-PLAN.md`。
> `02-STRUCTURE.md` 是 Qt 时代的旧规划，已作废。

## 进程拓扑

```
Electron (desktop/main.js)
  └─ fork: pythonw app/serve.py          ← 只跑后端，stdout 打 PORT=xxxx
       └─ uvicorn: FastAPI (app/server/app.py)
            ├─ GET  /                      → app/web/index.html
            ├─ /static/*                   → app/web/{css,js,fonts,img}
            └─ /api/*                      → 业务接口

app/main.py  ← pywebview 旧入口，已被 Electron 取代，保留未删
```

## 目录

```
app/
├── serve.py            后端独立入口（Electron fork 的就是它）
├── main.py             pywebview 入口（旧壳，未删）
├── server/             FastAPI 业务
│   ├── app.py          单文件 70KB —— 所有路由 + 业务都在这
│   ├── _detect.py      RetinaFace ONNX
│   ├── _dfl_train.py   训练包装：跑 4.0 的 main.py，禁掉自动弹浏览器
│   ├── biliauth.py     B 站扫码登录
│   └── _bili_mock.py   B 站假数据
├── web/                前端（StaticFiles 挂这里）
│   ├── index.html      53KB 单页
│   ├── js/app.js       190KB 单文件
│   ├── js/i18n.js      中英词典 + 正则模板
│   ├── css/            app.css / shadcn.css
│   ├── mask-editor/    遮罩绘制（copied from 4.0，独立子应用）
│   └── _parts/         早期拆分稿，**没有任何地方引用**（现状见 _parts/README.md）
├── core/               **空目录树**（data/extractor/merger/...），Qt 时代规划，烂尾
├── modelhub/           ONNX 模型
└── docs/               设计文档（01/02 已过期）

desktop/                Electron 壳
workspace/              用户数据
```

## 已知的结构债

1. `app/server/app.py` 70KB 单文件，2000 行，路由和业务混在一起
2. `app/web/js/app.js` 190KB 单文件，4898 行，靠 `window.*` 互相挂钩
3. ~~`app/core/` 整棵树是空的~~ —— 已删（8 个空目录，0 文件，从未被 import）
4. `app/web/_parts/` 里 8 个文件（3 js + 5 css）已逐字内联进 `app.js` / `app.css`，
   `mask-draw.html` / `mask-draw.js` 两份还没有逐字副本、`merge-*.html` 只内联了一部分，
   删之前要逐个确认（清单在 `_parts/README.md`）
5. `app/main.py` 是死代码（pywebview 时代留下的，与 `serve.py` 端口逻辑重复、无人引用）
   —— `legacy/qt-ui/` 已整目录删除（47 个文件 / 217 KB，从未被 import）
6. `02-STRUCTURE.md` 描述的 Qt 方案与代码完全不符
