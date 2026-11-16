# 解码腿：松开它（实测，2026-11）

> 台架 experiments/dec_lane/：micro.py（隔离解码车道）· parity.py（cv2 vs ffmpeg 画布逐位）·
> det_parity.py（换画布之后检测结果）· control.py（三条控制：TRT 确定性 / -threads 字节 /
> cv2 与步长采样）· prep.py（letterbox 四条实现）· e2e.py（真 _fused_run 端到端 A/B）。
> 机器 RTX 5070 Laptop 8GB · Ryzen 9 8940HX 16C/32T · 素材 workspace/9月29日 (5).mp4
> （HEVC 1080x1920 yuv420p 8bit 30fps）200–349 帧 · tinymog TRT bf16 · dfm 阿証証証証-416
> bf16w + CUDA Graph · seamless + mt + mask6 + blur4 · raw 管道。除注明外同进程交替、3 轮、中位。

## 0 口径修正

生产里那个「640 letterbox」不是 640x360：1080x1920 的短边缩放比 = 640/1920 = 1/3，
所以内容是 **360x640**，贴在 640x640 画布的**左上角**，其余用 **128** 填（RetinaFace 是 0）。
对拍与提速都按这个几何做。

## 1 现状复现（改动前）

| 环境 | fps | decode ms | u_dec | det_prep | detect | det_wait | compose | 本进程核 | dec ffmpeg | x264 |
|---|---|---|---|---|---|---|---|---|---|---|
| 有探针 | 18.94 | 38.89 | 75.8% | 5.16 | 17.49 | 29.04 | 179.7 | 13.7 | 27.5 CPU-s ≈ 3.4 核 | 29.5 CPU-s ≈ 3.6 核 |
| --noprobe | 22.35 | 31.59 | 76.5% | 4.24 | 12.01 | 24.12 | 186.5 | - | - | - |

用户报的 18.5~22 帧/秒、decode 36.3~44.5ms（71.7~86.3%）、两个 ffmpeg 各 3.5/3.6 核 —— 全部复现。
成片 md5 = `6c8aea1c46de53dd7b1027c7c6f14c62`（和 enc_unlock 26 次导出同一个）。

## 2 候选 1：让 ffmpeg 直接出检测画布 —— 像素门不过，且本身不更快

### 2.1 画布逐位对拍（30 帧，相对 cv2 现状；内容区 360x640，填边区另行核对）

| 滤镜链 | max 绝对差 | mean 绝对差 | 逐位相同 |
|---|---|---|---|
| format=bgr24,scale:flags=**neighbor**,pad | **7** | **0.00003** | **99.995%** |
| format=bgr24,scale:flags=area | 65 | 0.725 | 65.9% |
| format=bgr24,scale:flags=bicubic | 83 | 0.862 | 63.9% |
| format=bgr24,scale:flags=lanczos | 82 | 0.877 | 63.2% |
| format=bgr24,scale:flags=bilinear | 92 | 1.029 | 62.2% |
| yuv420p 上缩后转 BGR：neighbor / area / bicubic / bilinear | 36 / 66 / 85 / 93 | 0.41 / 1.42 / 1.18 / 1.54 | 78.9 / 26.5 / 44.8 / 31.9% |

**最接近的是 neighbor，而且是「几乎逐位」**：每帧只有 12~65 个像素不同（≤7 LSB，占 0.005%）。
原因不是插值算法选的不好，是 **cv2 在这条几何上根本不是「插值」**：1080→360 是精确 1/3，
目标像素的源坐标 (i+0.5)*3-0.5 = 3i+1 落在整数像素上，双线性权重退化成 1/0，于是 
cv2.resize(INTER_LINEAR) **恰好等于** frame[1::3, 1::3]（k=3 实测 0 像素不同；k=5 也成立；
k=2 不成立，max 79）。swscale 的 bilinear/area 是「真的抗锯齿重采样」（半径按 1/3 放大），
所以和 cv2 差一整个量级（mean 0.7~1.0、max 65~92、只有 62~66% 逐位相同）。
那 12~65 个不同像素来自 cv2 定点系数的边界舍入，ffmpeg 这边没有对应开关能关掉。

### 2.2 换了画布之后检测还对不对（tinymog TRT bf16，24 帧）

| flags | 面数变化帧 | box max 位移 px | 5 点 max 位移 px |
|---|---|---|---|
| neighbor | 0 | **0.1** | **0.1** |
| area | 0 | 1.3 | 0.9 |
| bilinear | 0 | 2.6 | 1.6 |

控制实验：同一张 canvas 连判 5 次，差 **0.000**（TRT bf16 + CUDA Graph 逐位确定）。
所以那 0.1px 是输入那几十个像素引起的，**不是噪声**。0.1px 会改 68 点 → 改 dfm 入参 → 
改合成 → 改 md5。按「boxes/5 点逐帧 max 位移 = 0」的门槛：**这条不采用**。

### 2.3 就算不看像素，它也不更快（隔离台架，150 帧 ×3 轮）

| 臂 | 帧/秒 | 车道 ms | read ms | prep ms | ffmpeg 核 |
|---|---|---|---|---|---|
| base（read+copy+cv2） | 101.1 | 9.21 | 5.62 | 3.55 | 3.34 |
| base_ri（readinto） | 118.8 | 7.76 | 5.80 | 1.87 | 3.89 |
| **tee**（1 ffmpeg、2 根管：全尺寸 + canvas） | 117.8 | 7.89 | 7.89 | 0 | 3.50 |
| canvas_only（只出 canvas） | **442.3** | 1.96 | 1.88 | 0 | 需求 18 核 |
| nvdec（-hwaccel cuda，参考） | 133.3 | 6.95 | 4.82 | 2.04 | 3.89 |

tee ≈ base_ri（多 0.69MB/帧的管子，刚好抵掉省下的 1.9ms cv2）。「只出 canvas」能跑 442 帧/秒，
说明 **瓶颈从来不是 letterbox，是 6.2MB/帧那条全尺寸管**。所以候选 1 两头都不成立，端到端没跑。

## 3 候选 2：det_prep 挪出解码线程 —— 这条是正的，采用

顺手做了两件事（都逐位不变）：
* `_prep_ctx(..., reuse=True)`：画布按线程复用，填边只做一次。**0.72 → 0.33 ms/帧**（prep.py）。
* 否掉一条看着聪明的路：既然 k=3 时 cv2 == 步长采样，能不能用 frame[1::3,1::3] 替掉 cv2？
  不能 —— 步长采样是 6.2MB 的跨步 gather，**1.15ms > cv2 的 0.72ms**。

端到端（150 帧 ×3 轮，同进程交替）：

| 臂 | 有探针 fps | decode ms | u_dec | 无探针 fps | decode ms | u_dec | 钉批无探针 fps |
|---|---|---|---|---|---|---|---|
| base（现状） | 18.94 | 38.89 | 75.8% | 22.35 | 31.59 | 76.5% | 22.39 (83.3%) |
| ri（只加 readinto） | 19.46 | 35.20 | 72.5% | 22.76 | 29.32 | 71.8% | - |
| **pool（prep 进 detect 线程 + 复用画布）** | **19.23** | **29.03** | **56.7%** | **22.97** | **25.83** | **63.1%** | **22.97** (71.0%) |

解码那一段 **-25%（探针 38.9→29.0）/ -18%（无探针 31.6→25.8）**，占用 76%→57~63%；
副作用是健康的：dec_wait 从 0→16~18%（解码开始被帧预算背压，不再自己饿着下游）。

## 4 候选 3：多条解码线程 / -threads / NVDEC

| 手段 | 隔离帧/秒 | ffmpeg 核 | 每帧 CPU | 结论 |
|---|---|---|---|---|
| 单进程 auto（现状） | 101（含 copy）/ 119（readinto） | 3.3~3.9 | 25~33 ms | 基线 |
| 单进程 **-threads 2** | **129.1** | 3.23 | **25.0 ms** | 隔离最快最省核；**端到端最差（-9%）** |
| 单进程 -threads 4 / 6 / 8 / 16 | 126.2 / 126.8 / 126.7 / 119.4 | 4.5 / 3.8 / 4.5 / 4.5 | - | 与 auto 同值 |
| **两进程各解一半**（两根管、两条读线程） | **49.9** | **8.11** | 163 ms | 明显更差 |
| NVDEC（-hwaccel cuda） | 133.3 | 3.89 | - | 只快 12%，且换像素（nv12→BGR mean 0.89/max 12） |

* `-threads` 不换像素：1/2/4/8/16/auto 输出的 rawvideo **逐帧 sha1 全等**（control.py，12 帧）。
* `-threads 2` 端到端反而更慢（探针 17.20、无探针 19.87 vs 22.35，整机核数 13.7→10.8）：
  它的「省核」是靠**解码自己变慢**换来的 —— 解码本来就只占 1/4 墙钟，省下来的核没人用得上，
  掉的那 20% 解码速度直接变成 fps。
* 两进程更差是因为 ffmpeg 每个进程 auto = 32 条解码线程：64 条线程抢 16 个物理核，
  每帧 CPU 从 25~33ms 涨到 163ms。要两路必须先钉 -threads，但**没必要**：单进程隔离能跑
  119~129 帧/秒，而生产这条车道只跑到 ~25 帧/秒。

## 5 候选 4：解码是不是到极限了？—— 对这条 CPU 车道是的

* 1080x1920 bgr24 = **6,220,800 B/帧 = 6.22 MB**。
* 隔离实测裸读这条管子：**5.2~5.8 ms/帧 → 1.07~1.20 GB/s**（Windows 匿名管道 + 单线程 readinto）。
  换算：**这条管子的上限 ≈ 172~192 帧/秒**；加上 cv2/resize 就是实测的 108~139 帧/秒。
* 同一个解码 ffmpeg，隔离下 25~33 ms CPU/帧能跑 119 帧/秒；生产里 27.5 CPU-s / 350 帧（含跳帧）=
  **78 ms CPU/帧**，只跑出 ~25 帧/秒 —— **它自己的效率也被抢核拉低了 2.4 倍**。
* 所以「解码已到极限」的正确读法是：**ffmpeg 不是不能再快，是这台机器没有空核给它**。
  这条腿的地板是那 6.2MB 的 BGR 管子（5.6ms/帧），天花板是整机 14/16 物理核。

## 6 像素门

* 钉批（batch=fused_batch=1、等待 0）对拍 base vs pool ×3 轮：**md5 全等 = 6c8aea1c46de5…**
  （与 enc_unlock 的 26 次导出、以及本次 32 次导出里的 30 次同一个）。
* 未钉批时有 **2/15** 次出现别的 md5（bb4c31eb / 8d058798，都在 -threads 臂）——那是
  「攒批张数随上游快慢变」引起的浮点差（enc_unlock 早已记录，故它的台架钉死批），不是解码改动；
  钉死后 6/6 全等。**结论：解码腿这三条改动的像素账是清的，但复现 md5 必须钉批。**
* 既有回归：app/tools/accept/det_prep_parity.py → box/kps/score max 差 = 0.0、canvas 复算 0；
  app/tools/accept/fused_decode_parity.py → PASS 25 / FAIL 0（8bit / 10bit / 12bit / 422 / 444 / rot90）。

## 7 改了什么（3 个文件，都在 env 开关后面）

`app/server/backend_merge/pipeline_fused.py`
* `DFL_FUSED_READINTO`（默认 **1**）：read 直接落进预分配环形缓冲（池 = 帧预算 + 4）。
* `DFL_FUSED_DET_PREP`：1 = decode 线程 / 0 = detect 线程 / **pool** = detect 线程 + 复用画布（默认 **pool**）。
* `DFL_FUSED_DEC_THREADS`（默认 0 = auto）：解码 ffmpeg 的 -threads（字节已验不变，但别开，见 §4）。
* 结果字典多三项：decode_threads / readinto / det_prep_where。

`app/server/_vision/detect.py`：`_prep_ctx(..., reuse=False)` + 两个检测器的 `prep(..., reuse=False)`。
`app/server/backend_merge/vram.py`：`_det_prep(det, frame, reuse=False)`（老实现没有 reuse 参数就退回原调用）。

复跑：runtime\server\Scripts\python.exe experiments\dec_lane\e2e.py --rounds 3 --pin-batch --noprobe

## 8 一句话

解码腿能松：隔离 101→119 帧/秒；真实流水线 decode 38.9→29.0ms/帧、占用 76%→57%，
端到端 +2.6%（22.39→22.97 无探针钉批）/ +1.5%（18.94→19.23 有探针）；松开的不是吞吐，
是把这条腿从刀尖上拿下来 —— **下一个墙是整机 14/16 物理核 + 那 6.2MB/帧的 BGR 管子，
要再快只能把解码与合成一起搬上 GPU（NVDEC + GPU 合成），而那一步开始换像素**。
