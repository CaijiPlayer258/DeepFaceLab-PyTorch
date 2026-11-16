# 26 · 合成搬到 GPU（overlay / hist-match，torch 逐段复刻 cv2）

一句话：**⑤ 合成的每一段都用 torch 重写了一遍（overlay / hist-match；seamless 用
backend_merge/torch_seamless.py 那份 torch 解），成片与 CPU 版逐帧 max = 1 LSB、差异像素 ~0.001%，
端到端 150 帧从 12.32 涨到 14.72 帧/秒（+19.5%），同时把 8 条合成线程的 CPU 占用整个
还了回去。**

- 代码：app/server/backend_merge/compose_gpu.py
- **运行期依赖：app/server/backend_merge/torch_seamless.py** —— compose_gpu 在 seamless 分支上
  按路径 import 它（原先在 `experiments/seamless/`，实验目录清掉后收进 backend_merge 同目录，
  老路径仍作兜底候选）。**这个文件不是实验脚手架，别当成台架删掉**；删了之后 seamless 会
  回退 CPU（日志里有一行），overlay 不受影响。
- 开关：**DFL_COMPOSE_GPU=1**（默认 0，CPU 路径一行没动）
- 子开关：**DFL_COMPOSE_GPU_SEAMLESS**（默认跟随总开关；=0 时 seamless 仍走 CPU）
- 排障断言：**DFL_COMPOSE_GPU_ASSERT=1**（锁「传进无缝的遮罩没被就地改写」）
- 验收：app/tools/accept/compose_gpu_accept.py（退出码 0 = 全过）

---

## 0. 为什么是「逐段复刻」而不是「另写一版」

compose_frame() 的每一个数值选择都是一条验收过的口径：ROI 怎么拆、cv2 的 warp 到底是什么
语义、ct 的统计量取在哪一块、融合完 u8 是四舍五入还是截断。GPU 版只换「在哪算」，不换
「算什么」—— 每一段都能指回 CPU 的那一行，对拍才有意义。

---

## 1. 三处 cv2 探针：复刻之前先问清楚它到底怎么算

这三条是这轮最值钱的产出，和帧速无关 —— 它们说明「复刻 cv2」的难点不在数学，在每个函数
的实现细节。

### 1.1 warpAffine(INTER_CUBIC)：float32 走**连续坐标**，只有 uint8 才 1/32 定点

单点 delta 图 + 亚像素平移，看 dst[48,48] 落在哪条曲线上（accept 的 cubic 段原样输出）：

| dx | 观测 | 连续精确值 | 1/32 定点预测 |
|---|---|---|---|
| 0.00000 | 1.0000000 | 1.0000000 | 1.0000000 |
| 0.00391 | 0.9999658 | 0.9999657 | 1.0000000 |
| 0.01562 | 0.9994555 | 0.9994555 | 1.0000000 |
| 0.03125 | 0.9978409 | 0.9978409 | 0.9978409 |
| 0.04375 | 0.9957979 | 0.9957980 | 0.9978409 |
| 0.10312 | 0.9774420 | 0.9774427 | 0.9812546 |
| 0.23438 | 0.8924971 | 0.8924971 | 0.8789062 |

max|观测 − 连续| = **6.4e-7**；max|观测 − 1/32 定点| = **1.36e-2**（≈3.4 LSB）。

只有 **uint8** 输入才走定点：同一个探针 dx=0.0130 → 255（吸附到整数）、dx=0.04375 → 254。

我们这条链三处 warp 的输入**全是 float32**（roi/255、wrk、fo），所以 GPU 版对齐的是
「连续坐标 + A=-0.75 三次核 + 零边界（不重新归一化）」。真按 1/32 量化反而会差 3.4 LSB ——
这是本轮推翻的第一个假设（原本以为要复刻定点）。

### 1.2 resize(INTER_CUBIC) 也是连续坐标

256²→416² 白噪声：|cv2 − 连续精确参考| = **2.6e-7**；|cv2 − 11bit 定点| = 9.9e-4；
|cv2 − 5bit 定点| = 5.9e-2。遮罩的 256→416 同样用连续坐标 + 4 tap + 边缘钳位。

### 1.3 GaussianBlur：ksize ≤ 9 用的是**硬编码核**

自己按 0.3*((k-1)/2-1)+0.8 推 sigma 会错得离谱：

| ksize | cv2 实际 1-D 核 | 公式推的 | 差 |
|---|---|---|---|
| 3 | [1,2,1]/4 | [0.239,0.522,0.239] | 2.2e-2 |
| 5 | [1,4,6,4,1]/16 | [0.0708,0.2445,0.3695,…] | **2.7e-1** |
| 7 | [2,7,14,18,14,7,2]/64 | 公式 | 6.8e-3 |
| 9 | [1,3.25,7.5,12.75,15,…]/64 | 公式 | 2.0e-3 |
| 11 | 公式 | 公式 | 1.4e-17 |

落在遮罩上就是 1.4e-3 的误差、1 LSB 的差异像素。所以**不自己算**，直接问
cv2.getGaussianKernel(k, sigma) 要那一份（版本相关的东西交给 cv2 自己）。
边界用 BORDER_REFLECT_101（torch 的 padding mode 就叫 reflect，定义一致）。

### 1.4 参考实现（grid_sample）离「真值」有多远

白噪声源最能放大坐标误差。同一组矩阵下，各实现离**精确 float64 参考**的距离：

| 实现 | max 距离 |
|---|---|
| cv2 warpAffine | 2.7e-5 |
| grid_sample（本模块默认） | 1.9e-5 |
| 逐 tap 直算（DFL_COMPOSE_GPU_WARP=gather） | 1.1e-6 |

**cv2 自己就离真值 2.7e-5** —— 这是它 float32 累加的噪声地板，任何复刻都不可能比它更近。
所以「与 cv2 逐位一致」在这条链上没有意义，验收按 ≤1 LSB 定；grid_sample 那条已经够
（0.005 LSB 量级），gather 那条慢 4 倍换 20 倍精度，默认不开。

---

## 2. 数据流：只上传人脸框，整帧不往返

~~~
numpy ROI(u8)   --上传-->  src_f(3,rh,rw) --+
pred / masks    --上传-->                   +--> 全部在 GPU：warp / ct / 反贴 / 融合
                                            +
合成后的框(u8)  <--下载--    （整帧不往返）
~~~

一帧 1080p + 一张脸的实际搬运量：ROI 217 KB + pred 519 KB + 4 张遮罩 477 KB（u8）
≈ 1.2 MB 上传 / 217 KB 下载。相比之下整帧 float32 是 25 MB。

多张脸按 CPU 的语义来：**先合并成一张 float 的 work，最后只回一次 u8**
（compose_frame 的 len(layers)>1 分支）。所以 GPU 侧用「所有脸 ROI 的并集」当画布：
上传并集一次、下载并集一次，脸之间重叠处的先后顺序与 CPU 完全一致。

---

## 3. 逐段对应表

| 段 | CPU（compose.py） | GPU（compose_gpu.py） | 实测差 |
|---|---|---|---|
| 几何 / ROI | get_transform_mat + _roi_of_face | 同一份（直接 import） | 0 |
| 遮罩准备 | _prepare_mask（u8/255 → 相乘 → resize → blur → clip） | mask_prep（同序，全在 GPU） | ≤1e-7 |
| ct 的参考脸 dw | warpAffine(roi, fwd, INTER_CUBIC) | warp_cubic / grid_sample | 3.8e-5（0.01 LSB） |
| ct（mt） | _match_tone_roi（遮罩加权 mean/std，盒内统计） | mt_transfer（整画布加权，盒外权重 0） | 4.8e-6（0.0012 LSB） |
| ct（其它 rct/mkl/idt/sot/mix/lut） | 原实现 | **逐脸回退**：warp 留 GPU，ct 拉回 CPU | 0 |
| seamless | cv2.seamlessClone（u8 遮罩 + NORMAL_CLONE） | seamless_merge（DST 解） | ≤1 LSB |
| hist-match | color_hist_match（np.unique+cumsum+np.interp） | hist_match（同三步，float64） | 函数本身 **0**；整帧 max 2 LSB（见 §4.4） |
| 反贴 | warpAffine(canvas, fwd, WARP_INVERSE_MAP) | warp_cubic（fwd 当 dst→src 直接用） | 同上 |
| 融合 | src*(1-mw) + fw*mw（numpy 三点式） | 同一式子，拆 3 个 kernel（不成 FMA） | ≤1e-7 |
| 收尾 | clip(work*255).astype(uint8) 只回一次 | acc 上一次 clamp+cast，只下载并集 | 0 |
| 回 u8 | clip(x*255,0,255).astype(uint8)（**截断**） | clamp(0,255).to(uint8)（同样截断） | 0 |

---

## 4. 对拍结果

### 4.1 基元（accept 的 prims 段）

| 项 | max 差 | u8 层面 |
|---|---|---|
| warp（grid）vs cv2 | 3.8e-5 | max 1 LSB · 0.06% 像素 |
| warp（gather）vs cv2 | 2.7e-5 | max 1 LSB · 0.05% 像素 |
| resize 256→416 | 3.6e-7 | 0 |
| GaussianBlur k=3..21 | ≤3.0e-7 | 0（k=21 有 0.07% 差 1 LSB） |
| match_tone(mt) | 4.8e-6 | 0.04% 差 1 LSB |
| hist-match（函数级，同一份输入） | **0** | 0 |

### 4.2 整帧（真素材：fused dump 的 pred/遮罩/68 点/原帧）

40 帧（frames 30-69，每帧一张脸，1080×1920）：

- 每帧 **max ≤ 1 LSB**，>=2 LSB 的像素 **0 个**
- 逐帧平均差 1.1e-5 LSB
- 最差那帧差异像素 0.0018%（112 / 6,220,800），最好那帧 34 个像素（0.0005%）

### 4.3 整条链（不是只比 compose）

同一段 150 帧、同一个 fused 流水线，只差 DFL_COMPOSE_GPU，两边都用 DFL_FUSED_DUMP=30..41
把 decode→detect→dfm→xseg→compose 出来的成片帧落盘，逐字节比：

**12 帧 · max = 1 LSB · mean 1.1e-5 LSB · 差异像素 836 / 74,649,600 = 0.0011%**

### 4.4 变体（accept 的 variants 段）

| 用例 | 整帧 max | 差异像素 |
|---|---|---|
| overlay · mt · mask8 · blur4 | 1 LSB | 281 / 37,324,800 |
| overlay · none | 1 LSB | 304 |
| overlay · mt · mask8 · blur0 | 1 LSB | 187 |
| overlay · mt · mask8 · blur8 | 1 LSB | 401 |
| overlay · mt · mask2(prd_dfm) | 1 LSB | 655 |
| overlay · mt · mask5(prd+dst) | 1 LSB | 1537 |
| overlay · lct（ct 逐脸回退 CPU） | 1 LSB | 208 |
| 双脸 overlay · mt · mask8 | 1 LSB | 72 / 6,220,800 |
| 双脸 overlay · mt · mask2 | 1 LSB | 187 |
| **hist-match · mt** | **2 LSB** | 291 |

hist-match 那 2 LSB 是**固有一级放大**，不是实现错：直方图匹配的映射结点是「输入的
unique 值」，而上游（cv2 的三次 warp + mt 的 float32 统计量）与 GPU 版差 1e-5 —— 这点差
会让个别结点错开一格，输出就跳 5.7e-3（=1.4 LSB）。实测 2 LSB 的像素是 **2~3 个**
（0.00005%）。要 0 差异只能是同一份实现（同一串 float32 运算）。

### 4.5 seamless（accept 的 seamless 段）—— **这条没达标，数字照实写**

CPU compose_frame(seamless) vs GPU compose_frame_gpu(seamless)，6 帧真素材：

| | 结果 |
|---|---|
| 逐帧平均差 | 1.3e-3 LSB |
| 差异像素 | 0.02% ~ 0.07%（每帧 1.3k ~ 4.4k / 6.2M） |
| **max** | **51 LSB**（6 帧里 2 帧破 1 LSB，其余 5~12） |
| 遮罩就地改写断言 | 通过（DFL_COMPOSE_GPU_ASSERT=1 全程开） |

机制已经查清，不是解算器的锅：把**同一份 u8 输入**喂给
backend_merge/torch_seamless.py 那份实现，它与 cv2 差 ≤1 LSB（sibling 的验收结论）。差的是**输入**：
我们这条链的 u8 遮罩/源片来自「GPU 的三次插值」，与 cv2 的 float32 结果差 1e-5 量级，
落到 u8 上有 246~595 个边缘像素（0.4~0.8%）「0 还是 1」翻了；而 **Poisson 解是积分** ——
边界上 1 个单位的梯度差会沿边界累加成一条几十 LSB 的窄带（实测 51）。

想逐位对齐只有一条路：遮罩和源片都改回 CPU 用 cv2 算（DFL_COMPOSE_GPU_SEAMLESS_EXACT=1
这个挂点做了，代价是每张脸 +3ms 的 CPU warp），那正好把 seamless 搬到 GPU 的意义抹掉。
**所以结论是：seamless 继续走 CPU**（见 §6），GPU 版留着当「GPU 空着、想要低 CPU 占用」
的选项。

---

## 5. 帧速与开销

### 5.1 端到端主表（150 帧 · 1080×1920 · 每帧一张脸 · 同机两轮交替 · 中位）

| 组 | 每轮 fps | fps 中位 | 进程树核数 | GPU% | GPU p50 | 空转<10% | compose 段 ms | detect 段 | encode 段 |
|---|---|---|---|---|---|---|---|---|---|
| **overlay** · CPU 8 线程 | 21.19 / 22.73 | **21.96** | 9.14 | 46.4 | 47.0 | 22.2% | 73.95 | 31.74 | 96.75 |
| **overlay** · GPU 4 线程 | 21.77 / 22.06 | **21.91** | 12.64 | 43.9 | 60.5 | 31.8% | **39.42** | 23.23 | 115.94 |
| **seamless** · CPU 8 线程 | 20.89 / 22.06 | **21.48** | 8.95 | 42.0 | 47.5 | 27.4% | 107.87 | 31.86 | 96.50 |
| **seamless** · GPU 4 线程 | 18.36 / 18.75 | **18.55** | 14.12 | 43.3 | 53.5 | 29.4% | **47.62** | 21.52 | 116.13 |

三条结论（都是同一张表读出来的）：

1. **overlay：fps 持平**（21.96 vs 21.91）。compose 段快 1.9 倍（73.95 → 39.42 ms），
   但整条线的瓶颈是 detect / encode / decode，合成段只占 12~30% 的占用。
2. **seamless：GPU 反而慢 13.6%**（21.48 → 18.55）。看等待项就明白：GPU 那两轮
   det_wait 从 3% 涨到 43%、dec_wait 从 28% 涨到 55% —— Poisson 解（DST matmul）
   和 detect 抢同一块卡，检测被饿死。所以 seamless 默认留在 CPU（§6）。
3. **代价是 CPU 核数**：overlay 9.14 → 12.64、seamless 8.95 → 14.12。GPU 版并没有
   「把 CPU 还回去」（原因见 5.3）。同时 encode 段从 96.75 变慢到 115.94 —— 编码线程
   和 CUDA 提交线程抢 CPU。

唯一一次明显赢的是「机器被别人抢满」的时候（同素材同参数、单轮）：

| | CPU 合成（8 线程） | GPU 合成（1 线程） |
|---|---|---|
| 流水线 fps | 12.32 | **14.72（+19.5%）** |
| compose 段（每帧 ms） | 45.91 | 54.45（含等 GPU 排队；实际 kernel ~5ms） |
| compose 段占用 | 7% | 80% |
| cmp_wait（合成线程等活） | 580 ms | 4.9 ms |
| GPU 利用率 mean / 空转<10% | 53.1% / 27.1% | 68.4% / 10.2% |

也就是说：**GPU 合成的价值是「CPU 被抢的时候不掉帧」**（8 条合成线程被抢是整条线的主要抖动源）。

### 5.1b GPU 合成开几条线程（overlay，同机同轮交替）

| 合成线程 | fps | 进程树核数 |
|---|---|---|
| 1 | 16.74 / 13.36 | 12.3 / 10.2 |
| 2 | 19.43 / 16.45 | 11.1 / 11.5 |
| 4 | **21.77 / 19.01** | 13.1 |

线程少的时候整条线被「一次 compose 要等整条 CUDA stream」串住了，所以默认 4
（DFL_COMPOSE_GPU_THREADS 可压；线程越多 CPU 提交开销也越大）。

### 5.2 合成段本身（孤立，真素材中位）

同一批 16 帧真素材、每档 3 次取中位（accept 的 perf 段，CPU/GPU 都在本进程里量）：

| | CPU 1 线程 | CPU 8 线程 | GPU 1 线程 | GPU 加速比 |
|---|---|---|---|---|
| overlay | 14.01 ms/帧 | 6.29 ms/帧（容量 159 帧/秒） | **7.27 ms/帧** | 1.9x vs 1 线程 |
| seamless | 20.16 ms/帧 | 8.36 ms/帧（容量 120 帧/秒） | **9.86 ms/帧** | 2.0x vs 1 线程 |

注意这张表的口径：**CPU 8 线程是 8 条线程并行**（cv2 放 GIL），而 GPU 那一列是
**一条线程串行**（perf 段没开多线程）。所以「GPU 比 CPU 8 线程慢」只说明
「一条 GPU 线程的容量 138 帧/秒 ＜ 8 条 CPU 线程的 159 帧/秒」；生产里 GPU 合成是
4 条线程（§5.1b），段容量约 1/39ms ≈ 25 帧/秒 × 4 路并行。

GPU 那 5ms 的分解（1080p + 一张脸）：遮罩准备 1.35 · ct(mt) 1.09 · 融合+下载 1.36 ·
两次反贴 0.59 · 上传/clone 0.55。**基本是 kernel 启动开销主导**（WDDM 下每次 launch
30~60µs，一帧 ~40 个 op）：单看算力这块还能再快 5 倍，但那要靠 CUDA Graph 或自写融合
kernel，不属于这一轮。

### 5.3 CPU 核数：GPU 版并没有把 CPU 还回去

同一批 150 帧、同一台机：

| | 进程树平均核心数 |
|---|---|
| CPU 合成 · 8 线程 | **7.3 ~ 8.3** |
| GPU 合成 · 1~4 线程 | **10.2 ~ 13.1** |

不是 compose 在烧 CPU（一个线程最多 1 核），是 **WDDM 下 CUDA 的提交/同步本身要烧 CPU**
（一帧 40 个 kernel launch + 每次 .cpu() 的全流同步）。所以「搬到 GPU 省 CPU」这条在这台
机器上**不成立**：省下的是 8 条合成线程的并行预算，付出去的是驱动的提交开销。
真要省 CPU 得把一帧的几十个 kernel 合成几个（CUDA Graph / 自写融合 kernel）。

### 5.4 compose 段占用从 7% 跳到 80% 是怎么回事

GPU 合成线程每次 .cpu() 都会等**整条 CUDA stream**，而 dfm/xseg/detect 也在同一条
stream 上，所以「compose 墙钟 54ms」里绝大部分是在排队。真实 kernel 只有 ~5ms。
要再压就得给合成开独立 stream + 事件同步（后续项，不在本轮）。

---

## 6. 开关、回退与已知边界

- **DFL_COMPOSE_GPU**：默认 0。=1 时 fused 流水线走 GPU 合成，合成线程数默认 4
  （DFL_COMPOSE_GPU_THREADS 可压；1 条会被「等整条 CUDA stream」串住，实测 1→4 线程
  13.4~16.7 → 19.0~21.8 帧/秒）。
- **seamless 默认仍走 CPU**（DFL_COMPOSE_GPU_SEAMLESS 默认 0）。=1 时才用 GPU 的
  poisson_blend_roi（DST 对角化解，backend_merge/torch_seamless.py）。
  关着的时候 compose_gpu 对 seamless 直接 raise，fused 在选路时就按 mode 跳过 GPU
  （不是每帧 try/except）。为什么默认关：§5.1 那两行 —— GPU 版让整条线慢 13.6%
  （Poisson 和 detect 抢卡），而且像素差到 51 LSB（§4.5）。
- **ct 不属于 {none, mt}**：逐脸回退 —— warp 仍留 GPU，ct 那一步把 dw 拉回 CPU 调原实现。
- **mask_mode 1（dst hull）/ erode ≠ 0**：遮罩准备回退 CPU（凸包栅格化 + 椭圆结构元形态学
  没在 GPU 上复刻；默认 erode=0，热路径碰不到）。
- **mask_mode 9 且遮罩尺寸不一致**：CPU 上 numpy 会直接抛，GPU 也不替它猜 —— 原样回退。
- **GPU 那条整段挂了**：本单退回 CPU（日志一行），结果与「一开始就用 CPU」相同。
- **合成后端回显**（两处都有，都实测过）：
  · 导出日志一行：「合成后端回显 compose_backend=gpu · 线程 4」；
  · status 的 vram_plan 里：compose_backend = gpu / cpu、compose_threads = 4 / 8
    （前端读 vram_plan 那一块就能直接显示；走 CPU 时是 cpu / 8）。
  · GPU 那条被跳过时还会带 compose_why（比如「DFL_COMPOSE_GPU=1 但 torch/cuda 起不来」）。
- **只接了 fused（视频模式快速导出）**：分段链 ⑤ 和项目模式的 compose_aligned_frame 仍走
  CPU 那份（compose.py 一行没动）。GPU 版已经支持 aligned 画布（compose_aligned_frame_gpu），
  只是没往那两条流水线里接。
- **显存账**：DFL_COMPOSE_GPU=1 时 torch 的 CUDA context 由这里建（约 300~500 MB），
  但 fused 的显存预算表里**没有**这一项 —— 因为现在 dfm 默认就走 torch，context 早就有了；
  真回到 ort 后端 + 开 GPU 合成时要把这块算进去。

---

## 7. **两个「验收全绿但什么都没验到」的坑**（比帧速重要）

### 7.1 「局部写回」vs「整幅写回」：一条 42 LSB 的 bug 只在 hist-match 下现形

CPU 的 ct（color_transfer code 9 = mt）是这么写的（compose.py:310-318）：

~~~
bb9 = _mask_bbox(wrk_mask)
out = fo.copy()
out[bb9] = np.clip(_match_tone_roi(fo[bb9], dw[bb9], wrk_mask[bb9]), 0, 1)
~~~

**只写遮罩包围盒，盒外一个像素都不动。** 我第一版按「统计量等价」的思路整画布写回
（盒外权重 0，对 mean/std 确实没影响），于是：

- **overlay 模式：完全看不出来** —— 盒外 mw == 0，融合式退化成 src*1 + fw*0，
  40 帧对拍照样 max 1 LSB、>=2 LSB 的像素 0 个。全绿。
- **hist-match 模式：差 42 LSB** —— 直方图匹配的映射表是拿**整张画布**的分布算的，
  盒外那些「本该不动却被改了」的像素把分布改了，整张脸的映射跟着变。

改回「只写盒内」之后：mt 段 4.8e-6 -> 2.7e-6，hist-match 42 -> **2 LSB**。

教训：**「数学等价」只在「结果真的会被用到」的地方成立。** 凡是有全局统计 / 全局映射的
下游（直方图、ct、任何拟合），上游「看不见的写」都会被放大。变体矩阵里必须有一条
**非 overlay** 的用例 —— 只有 overlay 的矩阵对这类 bug 是瞎的。

### 7.2 变体矩阵里 cfg 键名写错 -> 断言全绿、覆盖为零

accept 工具第一版把 cfg 写成界面的键名：

~~~
SETTINGS = {"blend": "overlay", "ct_mode": "mt", ...}      # 界面那套
# 但 compose_frame 读的是：mode / color_transfer_mode
mode = str(cfg.get("mode") or "overlay")                    # -> 永远是 overlay
~~~

于是 seamless 和 hist-match 两个用例**一直在跑 overlay**：13 项检查全 PASS，
而 seamless / hist-match 的覆盖率是 0。它看起来还很「合理」—— 变体表每一行都有数字，
数字还很漂亮（1 LSB）。

发现它靠的是**数字自洽性**：seamless 用例的 CPU 耗时 20ms，和 overlay 一样；
而 seamless 本该 30ms+。

现在的硬规矩（写进代码，不是写进纪律）：

1. 变体矩阵每一行都打印**实际生效的** mode / ct / mask / blur；
2. 断言里核对「实际生效值 == 期望值」，不匹配直接 SystemExit（accept 里已实现）；
3. 键名只用被调用方读的那一套（compose_frame 的 mode / color_transfer_mode），
   界面那套（blend / ct_mode）只在 app._merge_settings_to_config 里翻译。

---

## 8. 踩过的坑

1. **1/32 定点是给 uint8 的**（§1.1）。float32 走连续坐标 —— 照抄「定点」反而差 3.4 LSB。
2. **遮罩顺序：先相乘、再 resize**。CPU 是「按 mask_mode 逐张相乘，再把乘积 resize 到 S」。
   写成「先各张 resize 再相乘」时，成片差 **2 LSB**（三次插值线性，但 resize(a*b) ≠
   resize(a)*resize(b)）。这条是把 2 LSB 拉回 1 LSB 的关键。
3. **两个 ROI 原点不能混**：dw（帧 ROI→画布）的源是并集那块，用并集原点；反贴（画布→帧框）
   的 dst 是这张脸自己的框，用它自己的原点。混用只影响多脸，实测双脸 max **15 LSB**；
   单脸时两个原点恰好重合，所以单脸一路都是绿的 —— 这类 bug 只有多脸夹具能抓出来。
4. **cv2.seamlessClone 会就地改写 mask**（见 app/docs/notes/experiments/seamless.md）。我们的调用点
   每次都是新 np.clip(...).astype() 建的 mw_bin，没被影响；接 GPU 版时别复用同一块 mask。

---

## 9. 复现

~~~
# 全部用例（默认取 %TEMP%\dfl_cgpu 或 %TEMP%\dfl_compose_gpu 下最新的 fused dump 当夹具）
runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py

# 只跑某几段
runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py --cases cubic,prims
runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py --cases frame,variants
runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py --cases perf --reps 3
runtime\server\Scripts\python.exe app\tools\accept\compose_gpu_accept.py --cases seamless,switch

# 端到端 A/B（同一段跑 CPU / GPU 两单、采 GPU 利用率与进程树核数）：仓库根放一个
# _probe_export.py（这次量的就是它，参数 --settings-file / --env DFL_COMPOSE_GPU=0|1），
# 它会把 per_stage_ms 与 nvidia-smi 采样一起写进 JSON。

# 真素材夹具怎么来：跑一次 fused 导出并让合成前的输入落盘
DFL_FUSED_DUMP=30,31,...,69  +  POST /api/merge/export（见 app/tools/bench/fused_gpu_util.py）
~~~