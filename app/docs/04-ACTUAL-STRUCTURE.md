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
├── server/             FastAPI 业务（按域分包，顶层 _xxx.py 是 PEP 562 转发壳）
│   ├── app.py          组装（~300 行）：路由表 + 静态挂载 + PEP 562 转发，业务不在这里
│   ├── _common/        env / http / proc / paths / jobs / session / device / encoder_sel …
│   ├── _vision/        detect.py（RetinaFace/TinyMog）· align.py（5点→2D106→68→芯片）· xseg.py
│   ├── _media/         编码器选择 · 帧读取
│   ├── backend_merge/  合成器：pipeline_fused（融合导出）· stage_{faces,pred,mask,merge} · fused_io
│   │                   · pipeline_project（项目导出）· vram · preview · compose · knobs · frames_prep
│   ├── backend_extract/ 提取页：align_stream（唯一会写 DFLJPG 的路径）· resume · frames · media · pipeline
│   ├── backend_models/ dfm torch/ORT、模型元信息（**没有路由**了：原来的 trt.py 已拆空删除）
│   ├── backend_bench/  速度测试页：POST /api/perf/bench（只读）
│   ├── backend_engines/ 引擎编译页：/api/engines（状态）+ /api/engines/build（TRT 编译）
│   ├── backend_{faces,mask,bili,train,dataset,oneshot}/
│   └── routes_*.py     跨域路由（merge / 训练 / B 站）
├── web/                前端（StaticFiles 挂这里）
│   ├── index.html      单页多视图
│   ├── js/app/         **23 个分片**，后端拼成 /static/js/app.js（逐字节等价）
│   │                   00-core · 01-dataset · 02-merge-preview · 10-merge-settings · 20-progress-fused …
│   ├── js/i18n.js      中英词典 + 正则模板 · js/face_db.js 人脸库弹层
│   ├── css/            app.css / shadcn.css
│   └── mask-editor/    遮罩绘制（独立子应用）
├── engine/             4.0 引擎（vendored：xlib / core / models / MergeStudio / facelib / modelhub …）
├── tools/              验收与工具（accept / bench / migrate / audit / js / parts→docs）
└── docs/               设计文档 00~35（编号递增）· legacy-reference/ 归档区 · notes/ 实验记录

desktop/                Electron 壳
workspace/              用户数据
```

## 已知的结构债（2026-11 复核后更新）

1. ~~`app/server/app.py` 2000 行单文件~~ —— 已拆成组装层 + `_common/` + `backend_*/`；现在 ~300 行。
2. ~~`app/web/js/app.js` 单文件~~ —— 已按主题切成 `app/web/js/app/` 下 23 个分片（最大 2041 行），
   后端 `GET /static/js/app.js` 按文件名顺序拼接（逐字节等价）；改完跑 `node app/tools/js/check_bundle.js`。
3. `_fused_run` 仍是 1697 行 / 25 个闭包：拆它=改写成 ctx 对象（重写），且它是性能改动的落点，
   需要先钉 fps/帧号/钉批 md5 基线。周围可搬的 I/O 侧已搬到 `fused_io.py`。见 docs/35。
4. ~~提取器的路由被劈成两半~~ —— 已合并：`/api/extract/{status,start,cancel,progress}` 全在
   `backend_extract/routes.py`；引擎编译独立成 `backend_engines/`。
5. `app/docs/legacy-reference/` 是归档区（qt-ui / MaskProcessor / MergeStudio / parts / Extractor 等），
   不在运行路径上；其中 `parts/mask-draw.*` 是唯一副本，只移不删。
6. 4.0 的 `engine/Extractor/` 只剩 `_compat.py`（landmark106to68 的转发），其余已归档。