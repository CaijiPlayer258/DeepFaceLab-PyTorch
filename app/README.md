# app/ —— 4.1 的后端 + 前端 + 内置引擎

这个目录里混了两类东西：**我们自己写的 4.1 代码**，和**原样搬进来的 4.0 引擎**。
改之前先看一眼自己动的是哪一类。

## 一、4.1 自己的代码（可以改）

| 目录 | 是什么 |
|---|---|
| `server/` | FastAPI 后端。按界面分包：`backend_merge` / `backend_extract` / `backend_engines` / `backend_bench` / `backend_models` … + `_common`/`_media`/`_vision`；`app.py` 只剩组装（~300 行） |
| `web/` | 前端静态资源，挂在 `/static`：`index.html`、`css/`、`js/`、`img/`、`fonts/`、`mask-editor/`。**`js/app/` 是 23 个分片**，后端拼成 `js/app.js` |
| `tools/` | 运维脚本（迁移、导出、审计），不参与运行时 |
| `docs/` | 4.1 的设计与调研文档（`00` … `35` 编号递增）+ `notes/` 实验记录 + `legacy-reference/` 归档 |

## 二、`engine/` —— 4.0 引擎（原样搬入，尽量别改）

`engine/main.py` 是 4.0 的 CLI 入口，后端用子进程调它做训练/合成/导出。
下面这些目录**来自 iperov 的 DeepFaceLab 4.0-pytorch**，改它们等于改引擎：

```
engine/engine/core/           张量/层/优化器（leras）
engine/models/         各种模型：SAEHD / AMP / Quick96 / XSeg …
engine/samplelib/      训练用的样本加载器
engine/mainscripts/    Trainer / Extractor / Merger / Sorter / Util
engine/facelib/        人脸检测、关键点、对齐
engine/DFLIMG/         DFLJPG 格式（图片内嵌元数据）
engine/Extractor/  FacesetProcessor/  MaskProcessor/  MergeStudio/  DataAugmenter/
engine/WebUI/          4.0 的训练 WebUI（4.1 复用它传状态）
engine/localization/   多语言
engine/modelhub/       ONNX 模型（InsightFace2D106、errface 等）
engine/xlib/           外部库：onnxruntime 封装、SAM/GroundingDINO 权重
engine/assets/         引擎自带素材
```

## 三、跑起来

```bat
RunGUI.bat                      :: 桌面端（Electron + 本地后端）
runtime\server\Scripts\python.exe app\serve.py   :: 只起后端，浏览器开日志里那个端口
```

## 四、几处容易踩的约定

- **静态前缀固定 `/static`**：`index.html` 里有 200+ 处绝对路径（字体尤其多），改前缀会大面积 404。
- **`?v=` 由服务端按文件 mtime 重写**（`app.py` 的 `index()`），所以改完 js/css 刷新即可，不用手动清缓存。
- **登录凭证不进仓库**：`app/engine/ui/user/bilibili_cookie.txt`（运行期那份）与归档区里的
  `app/docs/legacy-reference/qt-ui/user/bilibili_cookie.txt` 都被 `.gitignore` 的 `**/` 规则拦住。
- **权重全部平铺在 `weights/` 根下**（`.onnx / .pth / .pt / .ubj`，只有 `bert-base-uncased/` 是个目录），`.gitignore` 已忽略，删了能重下但要重下。

## 五、从哪开始读

- 现状目录树与结构债：[docs/04-ACTUAL-STRUCTURE.md](docs/04-ACTUAL-STRUCTURE.md)
- 三条「提取」链路（提取页 / 快速导出 / 项目导出）各在哪：[docs/34-EXTRACT-THREE-PATHS.md](docs/34-EXTRACT-THREE-PATHS.md)
- 大文件审查与处置进度：[docs/35-LARGE-FILES-AUDIT.md](docs/35-LARGE-FILES-AUDIT.md)
- 改完怎么验：根目录 README 的「改了代码怎么验」（compileall / check_bundle / accept 套件 / mode B md5）

`app/docs/archive/4.0/05-RESTRUCTURE-PLAN.md` 里那套（`app/server` → `backend/`、`app/web` → `frontend/`）
**已决定不做**：要同步改 `serve.py`、`desktop/main.js` 的路径，收益只是目录名更短。

## 六、权重与代码是分开的

```
app/engine/modelhub/onnx/<模型>/*.py   ← 代码，随仓库分发
weights/<平铺名>.onnx | .pth | .pt     ← 权重，用户自己下载，不进 git
weights/bert-base-uncased/             ← 唯一保留的目录（tokenizer 必须和权重同目录）
weights/manifest.json                  ← 清单（路径/大小/sha256），入库
```

代码里用 `from modelhub import weights; weights.path("retinaface_10g.onnx")` 定位
（写老路径 `weights.path("onnx/RetinaFace/det_10g.onnx")` 也认，会自动折算成平铺名）；
缺失时抛出的异常里带着该执行的下载命令。详见 `weights/README.md`。
