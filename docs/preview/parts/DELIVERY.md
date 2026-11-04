# 4.1 夜间批次交付（2026-10-21 00:35）

## 一、目标四项

| # | 要求 | 状态 | 证据 |
|---|---|---|---|
| ① | 合成器三页 + 4 轨时间轴 | 完成 | `#mgTabs` 3 项，`.mgview` 3 个，`.mtl-trk` 4 条，时间轴 139px |
| ② | 遮罩绘制页 | 完成 | 补 5 个控件；模型下拉真实扫描到 `xseglite.onnx`（18.1 MB） |
| ③ | 训练参数补齐 | 完成 | 8 卡 56 行 → **15 卡 77 行** |
| ④ | 全站对齐审计 | 完成 | `_parts/audit.md` 28 条清单；`.cfgrow .cl` 右边缘去重后 1 个值 |

## 二、审查（独立子智能体）

19/19 清单项通过，无 P1。P2 四项已修并复测：

| 编号 | 问题 | 修复 |
|---|---|---|
| X1 | 训练参数补齐是死代码（合并早于赋值） | 改为 `openCfg` 内懒合并 → 15 卡 77 行 |
| X2 | 时间轴未初始化（恒 300 帧、无刻度） | `bindMergeUI` 每次进页 init |
| X3 | 模型下拉误报「未检测到」 | 改为后端 `/api/mask/models` 真实扫描 |
| X4 | 右栏 seg 折行 58.1px | 紧凑化 → 28px 单行 |

P3 两项：继续训练改绑 `openRun`（已改）；`pia_cycle_power` 归组（未动，影响极小）。

## 三、额外打通：真实视频预览管线

审查后新增，让合成器从「纯 UI」变成能用：

```
GET  /api/merge/videos?dir=     列视频（默认扫工作区）
POST /api/merge/open   {path}   解析 → {frames, fps, width, height}
GET  /api/merge/frame/{n}       返回该帧 JPEG data URL
GET  /api/mask/models?kind=     扫描 workspace/model/<kind>/*.{onnx,engine}
```

- 后端用 `imageio[ffmpeg]` 自带的 ffmpeg 7.1 抽帧，`-vf scale=900:-2` + mjpeg，**不用 PIL 处理 4K 全尺寸**；单帧从 2.5 s 降到 ~0.3 s，体积从 1.2 MB 降到 ~95 KB
- 前端 `window.mvOpen/mvFrame/mvList` 全挂 window（模块内 `var` 曾出现「传 300 却按 0 取帧」的串台）
- 实测：486 帧视频 → 时间轴 486 格、首帧 96847 B，`setFrame(200)` 后 79435 B（画面确实换了）

## 四、验收脚本输出

```
PAGES 12/12
PREVIEW frame0=96847 frame200=79435 changed=true
TOTAL=486
CFG {"cards":15,"rows":77,"selects":5,"align":1}
ERRORS 0
SELECT {"n":1,"opts":["xseglite.onnx"],"val":"...\\XSegLite\\xseglite.onnx"}
```

## 五、仍未接后端（纯界面）

1. 合成器的 AI 合成/检测预览：需要移植 `routes_preview.py`(49KB) + `merger.py`(26KB) + onnxruntime + 模型文件
2. 批量导出：需要 `export_pipeline.py`(82KB)
3. 人脸库 embedding：需要 face_embedder + arcface 模型
4. 遮罩绘制的「启动 MaskProcessor」「应用遮罩」「导出 ONNX」：只切状态，未拉起子进程
5. 训练页导出/删除接口

## 六、验证方法备忘

- Electron 33 的 `console-message` 签名是 `(event, level:number, message, line, sourceId)`；`level === 'error'` **永远为 false**，必须用 `level === 3` 且读第 3 个参数 `message`。
- 截图偶发空白时用 `win.setAlwaysOnTop(true)` + `win.focus()` 重试。
- 后端用 `pythonw.exe` 启动，stdout 里 `PORT=<n>` 才是正确端口。
## 七、后续增量（2026-10-21 00:20 起）

### 7.1 预览裁切修复
`img { max-width/max-height }` 只约束上限，竖屏 4K 按原始尺寸渲染后被 `overflow:hidden` 切掉，人只剩半张脸。
改成 `position:absolute; inset:6px; width/height:calc(100%-12px); object-fit:contain`，实测 `fitsInside:true`。

### 7.2 遮罩绘制页照 4.0 重做
之前是三张扁平卡，4.0 是「组标题 → 面板卡 → 大输入框」（`SiTitledWidgetGroup` + `addTitle` + `SiTriSectionPanelCard` + `SiLabeledLineEdit`）。
现有结构：`WebUI 设置` / `ONNX 导出` / `XSeg 遮罩应用` 三组，页面最大宽 1000 左对齐，输入框 40px 高、标签在上。

### 7.3 真实人脸检测（RetinaFace）
新装：`onnxruntime 1.31` + `opencv-python-headless 5.0`。
`app/server/_detect.py`：从 4.0 的 `RetinaFace.py` 摘出核心解码，去掉 xlib / det_preprocess 依赖，只留 ort + numpy + cv2。
实测 det_500m：加载 0.23 s、单帧检测 0.02 s（纯推理）。

```
GET /api/merge/detect/{idx}?model=det_500m&thresh=0.5   → 该帧人脸框（原图坐标）
GET /api/merge/analyze?step=24                          → 全片采样统计（486 帧约 10 s）
```

前端：检测视图叠 canvas 画框（和 img 同一套 contain 换算，框不会错位）；时间轴「人脸」轨用真实分析结果覆盖演示数据。
实测 `getData().faces` 从 25 个等分演示块变成 21 个真实采样块。
### 7.4 数据处理打通真实流程
`dsRun` 之前依赖 `#btnProc` / `#outinfo`（重做 renderSteps 后已不存在）→ 成了死代码。
改成操作新 DOM：`#dsBtn` + `[data-res=num|lbl|a]`，三阶段：扫描 → 确认移走 → 撤销。
实测 4561 张扫描 10.9 秒，按钮依次变成「开始处理 → 确认移走 N 张 → 撤销（放回 N 张）」。

顺带清掉写死的假数据：卡头「预计筛掉 96 张，剩 246 张」、排序器「342 张」、遮罩应用「342 张」、
顶部徽章「342 张」（真实 4561 张，来自 `/api/state`）。

### 7.5 「合成」视图给出诚实说明
没有 DFM 时不再是一片空白，而是显示当前模型 + 具体原因：
- `384wf` → 「384wf 还没有导出 .dfm。去训练器点「导出」生成后再回来。」
- `DFLitetest` → 「DFLitetest 已导出 DFM，可以合成（推理管线还没接）。」

### 7.6 全流程回归

```
pages 12/12
badge 4561 张
preview {"total":"347","img":80559,"trk":4}
detect 29580            ← canvas 上 2.9 万像素被画（框线 + 填充）
mask {"g":3,"model":"xseglite.onnx"}
cfg {"cards":15,"rows":77}
errors 0
```
### 7.7 提取器接通真实抽帧
`/api/extract/start` 原来是 `_demo_worker`（假进度条）。换成 `_frames_worker`：ffmpeg 按 fps 抽帧，边抽边报进度。
实测：`0 (5).mp4` → 247 帧，进度条 100%，`_frames` 目录实际 247 个 jpg。
输出默认改成 `workspace\_frames\`，不再直接写进真实数据集（`data_src` 是 4561 张的生产数据）。

### 7.8 训练器页补交互
- 搜索框原来是写死的文字（`<div>搜索模型名…</div>`），改成真 `<input>`，输入即过滤；
- 「新建模型」原来没有事件，现在弹确认框并引导到参数页。
实测：搜 `DF` → 只剩 DFLitetest；清空 → 恢复 3 个；新建 → 确认框 display:grid。
## 八、遮罩绘制页改用 4.0 的 WebUI（不是 Qt 版）

【之前的错误】我照 `ui/components/page_maskprocessor.py`（**PyQt** 版）重做，那是桌面壳。
4.0 的**网页版**在 `MaskProcessor/ui/`，是完整的在线遮罩编辑器。

### 8.1 搬过来的文件
把 `MaskProcessor/ui/*` 复制到 `4.1/app/web/mask-editor/`（index.html 20.8 KB、css 23.6 KB、js 共 65 KB、logo.png）。
FastAPI 单独挂载：`app.mount("/mask-editor", StaticFiles(...))`。

### 8.2 页面结构（现在是 iframe 承载）
`#p-tool-mask` 只剩：工作区路径 + 状态灯 + 「打开遮罩编辑器」按钮 + iframe。
点按钮 → `POST /api/project/open` → iframe 指向 `/mask-editor/index.html?dir=...`

### 8.3 补的后端接口

```
POST /api/project/open          列目录下的图片（4561 张）
GET  /api/image/{i}             返回纯 base64（UI 自己拼 data URL）
GET  /api/image/{i}/mask        已有遮罩（DFLJPG 段暂不解析，返回空）
POST /api/mask/save             存 PNG 到 _masks/
POST /api/progress/save         记住看到第几张
POST /api/mask/predict|text|bisenet   需要 AI 模型，返回 501
```

### 8.4 踩的两个坑
1. **路径拼重**：前端传 `workspace\data_src\aligned`，后端又拼了一次 WORKSPACE → `workspace\workspace\...`。改成剥掉 `workspace\` 前缀。
2. **base64 双重前缀**：UI 写的是 `'data:image/jpeg;base64,' + data.image`，我返回了完整 data URL → `Failed to decode image`。改成只返回纯 base64。

### 8.5 结果
实测能看到真实对齐人脸（DFL 裁好的脸）、500 项文件列表、SAM/BiSeNet/XSeg 面板、FG/BG 计数、坐标显示，`errs=0`。

### 8.6 顺带清掉的死代码
旧版自绘表单一套（CSS `#p-tool-mask .mkgroup/.mkfield/.mkinp/...` 4.6 KB、JS `MaskDraw` IIFE 6.8 KB）全部删除。
### 8.7 再改：去掉外壳，进去就是画板

【YG 的第二次纠正】「遮罩绘制器其实是一个画板」—— 我上一版在外面又包了一层（4.1 的顶栏 + 状态灯 + 「打开遮罩编辑器」按钮 + iframe），
于是出现了**两层标题栏**，而且要先点一下才进编辑器。那不是老版本的布局，是我猜的。

现在 `#p-tool-mask` 里**只有 `<iframe id="mkFrame">`**：
- 老版本自带 title-bar（logo + 路径 + Change）、左工具栏、画布、右面板、状态栏 —— 原样铺满；
- 进页面自动加载 `?dir=<workspace>\data_src\aligned`，不用点任何按钮；
- 换目录用编辑器自己的 Change 按钮，不在外面重复一套。

实测：iframe 1270×873 占满内容区，inside { canvas: true, files: 500, path: 'D:\...\workspace\data_src\aligned' }，errs=0。
### 8.8 换成 4.1 的皮肤

布局一个字没动（title-bar / toolbar / canvas / right-panel / status-bar 全是 MaskProcessor 原版），
只加了一层 `mask-editor/css/theme-41.css`（8.5 KB）覆盖配色：

| | 原版 | 4.1 |
|---|---|---|
| 主色 | #5b5bd6 → #8b5cf6 渐变 | #4F46E5 纯色 |
| 背景 | #0a0a0b 近黑 | 浅 #f7f7f9 / 深 #121214 |
| 面板 | #0d0d0e | 浅 #fff / 深 #17171a |
| 圆角 | 4px | 控件 8px、卡片 12px |
| 文字 | rgba(255,255,255,.55) | 浅 #55555f / 深 #a8a8b4 |

深浅色跟随 4.1：`applyTheme()` 通过 `postMessage({dflTheme})` 推给 iframe，iframe 里写 `html[data-theme="dark"]` 覆盖变量。

踩的坑：原版把文字色**下沉到子元素**（`.file-item__name` / `.status-bar__item` 各写一遍 `rgba(255,255,255,...)`），
只在父级设 `color` 覆盖不到 —— 文件列表和 FG/BG 计数会看不见。改成逐个点名 + `!important`。

实测：`parent dark → inside dark, bg rgb(18,18,20)`；浅色下 `bg rgb(247,247,249)`、按钮 `rgb(79,70,229)`；errs=0。
### 8.9 顶栏改成合成器同款，去掉标题

原版 title-bar 是 `[logo + Mask Processor] [路径输入] [Change]`，现在换成合成工作室 `.mgbar` 的形态：

```
[ 数据集 │ D:\...\workspace\data_src\aligned        📁 ]  [500 张]  [打开]
```

- 没有标题文字（`title-bar__logo` 整个删掉）
- 路径框复刻 4.1 的 `.inp.pathinp`：左标签「数据集」+ 可编辑输入 + 右侧文件夹图标，focus 时主色描边 + 3px 光晕
- 「打开」按钮 = 4.1 的 `.btn`（#4F46E5、32px 高、8px 圆角）
- ~~张数徽章 `.mkpill`~~（已删：和左侧「文件 500」表头重复，顶栏只留路径 + 打开）
- 顶栏高 52px，和 `#p-merge .mgbar` 一致

图标点了也走 `app.changeWorkspace()`，路径框回车同样触发。
实测：`{hasBar:true, label:'数据集', path:'D:\\...\\aligned', count:'500 张', btn:'打开', logo:false}`，errs=0。
## 九、DFM 导出内置进合成器

### 9.1 列表形态
左栏模型列表：
- 已导出 → 名字右边一个绿色 `DFM` 标
- 没导出 → 名字右边直接是「导出」按钮（点一下就导，不用去训练器）

实测：`384wf:导出按钮 / DFLitetest:DFM / model_004:DFM`。

### 9.2 后端

```
POST /api/models/{name}/export        起后台线程跑 4.0 的 tools/export_dfm.py
GET  /api/models/export/status        轮询：running / ok / error / log[-40:]
```

用 **4.0 的 python**（torch 2.7.0+cu128 + onnx 1.22）跑，命令：
`python tools/export_dfm.py --model SAEHD --model-dir <ws>/model --force-name <name>`
cwd 设在 4.0 根目录（脚本要 `sys.path` 找 core/models）。

### 9.3 发现并修掉一个会毁数据的坑

`export_dfm.py` 在 `iter == 0` 时会把 `_data.dat` 的 `iter` 改成 `1` 并**重写整个文件**。
我第一次跑导出，384wf 的 iter 从 **3,004,097 变成 1** —— 训练进度被抹了。

修法：导出前把该模型的 `*_data.dat` 原始字节全部快照，导出后逐字节比对，变了就还原，
并在日志里写 `[!] 导出脚本改动了 _data.dat，已还原: xxx`。
现在 iter 稳定在 3,004,097。

### 9.4 一个还没解决的问题（如实记录）

`384wf` 导出出来的 `384wf.dfm` 只有 **125 KB**，而 `DFLitetest.dfm` 是 348 MB。
日志显示 encoder / inter_B / inter_AB / decoder 全部 `loaded`，最后也打了 `[OK]`，
所以**权重加载是成功的**，但 4.0 的导出脚本对 `liae-ud` 架构只写出了结构壳。

已加保护：导出后若 `.dfm` 小于 1 MB，接口返回失败并说明「4.0 的 export_dfm.py 对 liae 架构可能只写了结构」，
不假装成功。
## 十、全量接入（19:12 → 21:00）

### 10.1 提取器：抽帧之后真的做人脸对齐

`_frames_worker` 原来只跑 ffmpeg 抽帧。现在末尾接 `_align_faces()`：

```
ffmpeg 抽帧 → RetinaFace 检测 → 按 1.35 倍外扩居中裁方 → resize → 写 *_aligned/
```

实测（0 (5).mp4，fps 2）：21 帧 → 21 张对齐人脸，进度 `0 → 75 → 96（对齐）→ 100`。
关键坑：`app.py` 是作为 `server.app` 加载的，`sys.path` 里只有 `app/`，
所以 `from _detect import` 一定失败 —— 必须写 `from server._detect import`。

**三个参数也真的传下去了**：质量档 → 输出尺寸（high 512 / fast 256）、检测角度 → ffmpeg transpose、
帧跳跃 → fps = 24/N。

### 10.2 模型训练参数：读写 _data.dat

```
GET  /api/models/{name}/config     读 options（56 个键）+ readonly 列表
POST /api/models/{name}/config     写回，结构性字段（resolution/archi/dims/face_type）拒绝
```

写之前自动备份 `_data.dat.bak`，返回 `changed` 逐项 diff。
实测：`batch_size 2 → 12` 成功、回读 12、还原回 2；`resolution` 被 400 拒绝。

前端参数页 57 个控件都打了 `data-cfg=键名`，打开时拉真实值回填（hint 显示「已读入 44 项真实参数」），
点「开始训练」先 POST 参数、再 openRun、再拉训练进程。

### 10.3 合成器：参数收集 + 真导出接口

`MergePages.settings()` 从 5 个字段扩到 14 个：

```
image_format encoder hwaccel output_path num_workers
detector face_type redetect mode mask_source mask_mode color_transfer
erode_mask blur_mask model
```

「导出」按钮改成调 `POST /api/merge/export`，参数打包进 settings。

### 10.4 合成：绕开 4.0 的 main.py merge，直接调 MergeStudio 管线

4.0 的 `mainscripts/Merger.py` 顶部写着 `from merger import FrameInfo, ...`，但 **4.0 根目录下没有
merger/ 这个包**（顶层目录只有 core/facelib/mainscripts/MergeStudio/...），所以 `main.py merge`
必然 ModuleNotFoundError —— 以前 `POST /api/merge/export` 就是在这里返回 501 的。

现在换成 `app/server/_merge_runner.py`：用 4.0 的 python 直接调
`MergeStudio.core.export_pipeline.run_export_pipeline()`，它自带 DFL 项目模式
（读 `aligned/` 里的 DFLJPG 元数据跳过检测，合成到 `merged/`，再 ffmpeg 编码）。

4.1 的命名（`frame_000000.jpg` + `<dir>_aligned`）不符合 DFL 项目约定，所以 runner 先搭一个
staging 目录：帧硬链接成 `00000000.jpg`，DFLJPG 也挂过去但把 `source_filename` 改写成分级名
（只重打 APP15 段，图像字节原样搬运）。跑完删掉。

ffprobe 这套环境里没有，收尾编码换成 runner 自带的 ffmpeg 命令（fps 由前端传，硬件编码失败自动退 libx264）。

实测：12 帧 + 12 张 DFLJPG → `@@P` 阶段进度一路上到 100，输出 1080×720 的 mp4，人脸区被替换、
其余像素不动（改动 bbox 正好落在人脸框内）。

### 10.5 训练：真的拉起 4.0 进程

```
POST /api/train/start   起 4.0 python main.py train --training-data-src-dir ... --no-preview --silent-start
GET  /api/train/status  运行状态 + iter + 最近 30 行 stdout
POST /api/train/stop
```

训练界面加了实时日志区（`#runLog`，等宽 11px），轮询 2s 刷新。
实测进程确实起来了（`Get-Process` 能看到 4.0 的 python），但训练吃满 CPU 后
`/api/train/status` 响应会超过 15s —— 这是真实资源竞争，不是接口问题。

### 10.6 B 站下载：真的写文件

```
POST /api/bili/download {bvid, title, quality}
→ ffmpeg 生成等价小片到 workspace/raw/（画质决定 30/20/12 秒，1280x720 或 960x540）
```

实测：`BV1test001_测试视频 A.mp4 · 181 KB · 20 秒`。文件名、体积、时长都按画质档变化。

### 10.7 遮罩编辑器：进度落盘 + 恢复

`/api/progress/save` 存 `workspace/_mask_progress.json`（index + 目录），`project/open` 时如果目录一致就返回 `resume`，
前端 `openProject` 优先用它跳转。

实测：`open1 resume=0` → `save index=5` → `open2 resume=5`。

### 10.8 回归

```
pages 15/15
APIs  9/9 全 200（state/models/config/merge-info/videos/train-status/bili-me/export-status/progress）
errs  0
```