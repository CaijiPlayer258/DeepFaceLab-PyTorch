# 17 · 项目模式导出的批量协议与裸字节通道

> 这一页记的是「DFL 项目导出」为了把 1080p 的合成从 3~4 帧/秒抬上去做的改造：
> **一条请求带 k 帧、帧走裸字节块（不再 base64）、worker 内 k 线程并行**。
> 结论先放这儿：本机（RTX 5070 Laptop 8GB / 8940HX）90~120 帧的真项目
> **3.8 → 7.3~7.9 帧/秒（+90~107%）**，产物与串行基准逐像素一致（mean ≤ 0.0001 LSB）。

## 一、为什么要把「进程数」和「线程数」分开

改造前是「一个进程一帧一请求」，参数只有 workers（进程数）。实测（416 的 dfm，
一份 session 2.07GB，8GB 卡）：

| 做法 | 帧/秒 | 说明 |
|---|---|---|
| 1 进程顺序（= 改造前默认路径） | 2.4~3.8 | 一帧 330~500ms，父进程的准备和 worker 的换脸完全串起来 |
| 1 进程 4 帧在飞（老代码的滑动窗口） | 4.0~4.6 | 准备与换脸重叠了，但 worker 内部一次只跑一帧 → 卡在 ~220ms/帧 |
| 2 进程 1 帧在飞 | 3.2 | 显存开始吃紧，进程之间只多出 33% |

也就是说：**再堆进程只能买到显存压力，买不到帧速**。worker 里一次只跑一帧，GPU 在
「整帧 float32 的遮罩/合成/seamlessClone/PNG 编码」那几段全程闲着；而那些段是 CPU 的、
会释放 GIL —— 所以真正的解是**在一个 worker 进程里同时跑 k 帧**。

## 二、协议形状（_PREVIEW_WORKER_SRC）

一条请求 = **一行 JSON 头 + 紧随其后的裸字节块**（父进程与 worker 都按头上的
"blobs": [n1, n2, …] 精确读满）：

    → {"id":7,"cmd":"swap","threads":4,"items":[{…item…},{…}],"blobs":[6220800,6220800,…]}
       <帧字节><帧字节>…
    ← {"id":7,"ok":true,"items":[{…meta…},…],"blobs":[728640,728640,…]}
       <出图字节>…

* **帧不再 base64**：base64 把 3 字节抬成 4 字节，父进程还要 ~5ms/帧编解码；头和块排在同一
  条管道上，谁也不需要"可打印字符"。旧的 "frame": "<b64>" 写法已经删掉
  （face_debug 那条命令两种都认，便于排障）。
* **item 里带什么**：enc（raw = 裸 BGR；jpg/png = 压缩字节）、h/w、
  fmt（raw/png/jpg = 出图怎么编）、landmarks/faces/box/config/xseg。
* **失败只影响那一项**：meta 里 ok:false，blobs 那一项是 0 字节，父进程靠下标对回帧名。
* **线程池按 k 建一次**（ThreadPoolExecutor），k 变了才重建。
* **超时按轮数放大**：k 帧分 ceil(k/threads) 轮，单帧上限 × 轮数。
* **模块级 _merger._model_input_size 不再被并发写**：MergeMaskedFace 加了显式
  input_size 参数（不传 = 老行为）。多模型、输入边长不同时这条是必须的。

## 三、父进程：三段流水 + 严格按序落盘

    准备池(prep_n)           换脸池(n_workers 线程，各持一个 worker 进程)        落盘池(enc_n)
    读帧/检测/68点/组帧  →   一条请求 k 帧（worker 里 k 线程）           →   PNG/JPEG 编码 + .part→replace
            └──────────────── 主线程按**提交顺序**收结果、记账 ────────────────┘

* **顺序**：批是连续切的、结果按批顺序取回、落盘按帧顺序收（writes 是有序队列）。
  所以「取消时 merged/ 里留下的一定是前缀」这条断点续跑语义原样保留（D 段验收：
  done=52 / merged=52 帧 / 编号连续）。
* **PNG 编码搬到父进程**（只对 .png 帧）：worker 里编 1080x1920 PNG（compression=3）
  要 132~145ms/帧，占那一路单帧的三成；挪到落盘池就能和别的帧的换脸重叠。像素无损
  （实测「PNG 回传 vs 裸字节回传」逐像素最大差 0），文件字节会变（换了编码器）。
  DFL_PROJ_OUT=png 可以让 .png 帧也交回 worker 编。
* **.jpg 帧不改编码**：那条路的产物就是 JPEG(q92)，改写成无损 PNG 会**改变像素**，
  所以 .jpg 帧仍旧由 worker 编 JPEG。DFL_PROJ_SEND=jpg 可以把传帧也退回 q95 JPEG。

## 四、默认值与开关

| 开关 | 默认 | 作用 |
|---|---|---|
| workers（请求体）/ DFL_PROJ_WORKERS | **2** | worker 进程数 = 几份 dfm session |
| batch（请求体）/ DFL_PROJ_BATCH | **4** | 一条请求带几帧 = worker 内几线程 |
| DFL_PROJ_SEND | raw | 传帧格式（jpg 退回 q95 JPEG，逐字节还原改造前的输入） |
| DFL_PROJ_OUT | raw | .png 帧的出图路径（png = 交回 worker 编） |

传帧改用裸 BGR 是**过了像素门**才切的：同一帧、同参数对拍（1080x1920 真项目帧）
q95 JPEG 往返 vs 裸字节 = mean 0.1506 LSB / 差异像素 12.60% / max 5 LSB，在 0.5 LSB 门内。

## 五、实测（1080x1920，416 的 dfm，90~120 帧真项目，app/tools/proj_pool_accept.py）

| workers×batch | 帧/秒 | 显存峰值 | worker 单帧 p50 | 落盘/失败 |
|---|---|---|---|---|
| 1×1 | 3.81 | 2471 MB | 334ms | 120/0 |
| 1×4 | 6.24 | 4703 MB | 765ms | 120/0 |
| 2×2 | 7.09 | 5888 MB | 674ms | 120/0 |
| **2×4** | **7.28~7.86** | 7652~7788 MB | 1187~1272ms | 120/0 |
| 2×8 | 1.57 | 7841 MB | 9.4s | 104/**16** |
| 3×4 | 1.5~1.9 | 7823 MB | 7.4~8.7s | 120/0 |

* **2×4 是这台机器的甜点**：8 帧在飞、显存 ~7.7GB（8GB 卡已经贴顶）。
* 再往上（2×8 / 3×4 = 12 帧在飞）显存 spill，单帧从 1.2s 蹦到 9s，掉到 1.5 帧/秒，
  而且 2×8 会有 16 帧直接超时失败 —— **不要用**。
* 逐像素：每档产物 vs 1×1 基准 mean ≤ 0.0001 LSB、差异像素 ≤0.013%、最大差 ≤8
  （CUDA 浮点累加顺序，老代码同样如此）。
* 断点续跑：跑一半取消（done=52 / merged=52 帧 / 编号连续）→ 续跑补齐 120 帧，
  和一次跑完逐像素一致（mean 0.000114 LSB）。
* 改造前后（同一台机器、同一份 90 帧 jpg 项目、都按「预热之后」计时）：
  2.45 → 7.41~7.86 帧/秒（改造前那档是 1×1 老代码）。

## 六、这台机器到不了 8~9 的原因（实测数据，不是猜）

* 1080p 单帧在 worker 里的固定开销 ≈ 330ms：其中 dfm 前向只有 ~15ms，
  seamlessClone ~156ms、色彩迁移 + 回 warp ~95ms，其余是整帧 float32 的遮罩/合成。
* 整帧 float32 的运算是**内存带宽**瓶颈：8 线程跑 numpy 逐元素只有 1.27× 聚合加速
  （单线程 ~3.9GB/s）。所以「再开线程/再开进程」都换不来线性增长。
* cv2.setNumThreads 默认按核数开池，多线程并发时会互相踩（seamlessClone 8 线程
  只有 1.19×；设成 1 是 3.73×）—— 这条只是让 k 线程别过度订阅，突破不了带宽上限。

## 七、怎么验收

    runtime\server\Scripts\python.exe app\tools\proj_pool_accept.py            # 全跑（约 8 分钟）
    runtime\server\Scripts\python.exe app\tools\proj_pool_accept.py --cases a,c  # 只跑协议/对拍

它自己组项目（/api/merge/stage/frames + /faces）、起后端、扫 workers×batch、逐像素对拍、
跑一次「取消 → 续跑」。全部 PASS 才算过；沙箱在 %TEMP%\dfl_pool_accept。
