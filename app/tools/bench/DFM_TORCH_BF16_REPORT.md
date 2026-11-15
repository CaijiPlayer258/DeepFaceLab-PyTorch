# dfm 换成 torch eager + bf16：两模型 × 四精度 × batch 1/4/8 实测

日期 2026-11-15 · RTX 5070 Laptop 8GB（sm_120）/ 16 物理核 · server venv torch 2.9.1+cu128 ·
ORT 1.22（CUDA EP）+ cuDNN 9 · 脚本：`app/tools/bench/dfm_backend_bench.py`（速度/显存/对拍）、
`app/tools/bench/dfm_bf16_quality.py`（真脸 + 成片的质量表与肉眼图）。
产物：`%TEMP%\dfm_backend_bench\<run>\cells.jsonl`、`%TEMP%\dfm_bf16_quality\*.npz|png`。

前置：`DFM_TORCH_REPORT.md` 已经证明「同一份 .dfm 可以翻成逐节点等价的 nn.Module，
torch eager + TF32 比 ONNX 快 3~13%」。那份表**没测 bf16**。这一份补上，并且把它做成生产后端。

---

## 0. 一句话结论

**bf16 不是「再快几个百分点」，是换了一个数量级的性价比：batch 8 上 416 从 39.5 → 20.0 ms/张
（−49%）、384 从 32.6 → 22.5 ms/张（−31%）；batch 4 上 −48% / −42%。**
代价是逐像素：**真脸上 face 输出 max 4.8 LSB、mean 0.38 LSB、p99 1.4 LSB**；
**过一遍真的 compose 之后 max 5 LSB、mean 0.009 LSB、p99 0 LSB（≥99% 的像素逐字节相同）、
>4 LSB 占 0.00%**。用户 2026-11 拍板：可接受 → **默认档 = torch + bf16w**
（权重也压 bf16：同进程交替 A/B 实测没有一档更慢，还少占 250MB~1.1GB —— 见 §3.1），
回退一条 env：`DFL_DFM_BACKEND=onnx`（fp32 老路）或 `DFL_DFM_DTYPE=fp32`（torch 严格 fp32）。
两档 bf16（`bf16` = 权重 fp32 + autocast / `bf16w` = 权重也 bf16）的**精度差 <1 LSB**，
选谁都不改像素，所以按显存挑。

---

## 1. 现在怎么用

| 变量 | 默认 | 说明 |
|---|---|---|
| `DFL_DFM_BACKEND` | **torch** | `onnx` = 回退到 onnxruntime（老行为，一行 env）；也认 `torch:bf16w` 简写 |
| `DFL_DFM_DTYPE` | **bf16w** | 只对 torch 生效：`fp32`（严格 fp32，比 ORT 慢 2.4 倍）/ `tf32` / `bf16`（权重 fp32 + autocast）/ **`bf16w`**（权重也压 bf16）/ `auto`（按权重体量选，旧口径） |
| `DFL_DFM_MB` | 8 | 预编译的 batch 上限（超了自动分块） |
| `DFL_DFM_CL` | 0 | 权重转 channels_last（384 那份 batch 8 上另有收益，见 `3 注） |
| `DFL_DFM_PIN` | 1 | 出参走 pinned 缓冲（关掉是为了对比 pageable 的代价） |
| `DFL_DFM_TRT` | 0 | 只影响 onnx 那条路（原生 TRT 引擎，仍默认关） |

torch 起不来（没装 / 没卡 / 编译失败）时**回落 onnx 并在日志里喊一行**，不静默换后端。

日志与 `[eff]` 都会回显（见 `7）：
`换脸：dfm 走 torch(cuda) · 精度=bf16（TF32=开）· 416×416 · 权重 172 张量/514MB · 预热单张 30.1 ms · 卡上 1185.0 MB`

---

## 2. 实现：三个文件

| 文件 | 干什么 |
|---|---|
| `app/server/backend_models/dfm_graph.py` | .dfm（ONNX ModelProto）→ nn.Module。**不依赖 onnx 包**（server venv 里没有）：自己按 protobuf 线格式扫（mmap，raw_data 零拷贝映射），再静态折叠 + 代码生成 |
| `app/server/backend_models/dfm_torch.py` | 会话：权重搬一次（`ParamStore`）、按 batch 分桶编译、精度档、pinned 出参、ORT 兼容的 `run(None, {name: x})` |
| `pipeline_stages._dfm_session` | 后端选择 + 日志；`_dfm_run_batch` **一行没改**（靠 ORT 风格的 shim） |

三个关键设计（都是被实测逼出来的）：

1. **权重只有一份**。折叠要把 batch 定死，所以 batch 1..8 各有一张图；如果每张图各自
   `register_buffer` 一份权重，384 那份 1567MB 会变成 12.5GB。现在常量放进共享
   `ParamStore`，各 bucket 的 forward 引用同一批张量对象 —— 实测「编译 8 张图」的显存
   增量 = **0**（日志 `batch=N 复用 111 个权重张量、另搬 46 个`，那 46 个是折叠出来的小常量）。
2. **逐位对齐旧编译器**。`dfm_torch_graph.py`（上一轮验证过的那个）与新编译器在
   416/384 × batch 1/8 上 **max|diff| = 0.0 / 0.0 / 0.0**（三个输出逐个比），
   而加载时间从 **3.5s → 0.33s（416）/ 0.82s（384）**（省掉 `onnx.load` 的 1.6GB protobuf 解析）。
3. **出参走 pinned**。416 batch 8 的一次前向要搬 27MB 回 CPU，pageable 拷贝实测多花 4~5ms；
   `run()` 里 `torch.empty(pin_memory=True)` + `copy_(non_blocking=True)` 把这段抹掉
   （`3 的数都是**含** H2D+D2H 的 `run()` 墙钟，和 ORT 的 `sess.run` 同口径）。

---

## 3. 速度表（run() 含 H2D+D2H，5 次取中位）

**416 · 阿証証証証-416.dfm（514MB 权重，NHWC 416）**

| 后端 / 精度 | b1 总 ms | b1 每张 | b4 每张 | b8 每张 | b8 相对 onnx |
|---|---|---|---|---|---|
| onnxruntime CUDA fp32（基线） | 35.78 | 35.78 | 41.02 | 39.51 | — |
| torch tf32 | 36.89 | 36.89 | 39.06 | 39.33 | −0.5% |
| torch **bf16**（autocast） | 28.31 | **28.31** | **21.10** | **20.03** | **−49.3%** |
| torch bf16w（权重也 bf16） | — | 27.3* | — | 27.3* | −31%* |
| torch fp32（关 TF32） | 85.36 | 85.36 | 137.25 | 109.71 | +178% |

**384 · 张元英.dfm（1567MB 权重，NHWC 384）**

| 后端 / 精度 | b1 总 ms | b1 每张 | b4 每张 | b8 每张 | b8 相对 onnx |
|---|---|---|---|---|---|
| onnxruntime CUDA fp32（基线） | 39.78 | 39.78 | 40.97 | 32.60 | — |
| torch tf32 | 38.00 | 38.00 | 35.09 | 32.90 | +0.9% |
| torch **bf16**（autocast） | 35.64 | **35.64** | **23.83** | **22.51** | **−30.9%** |
| torch bf16w（权重也 bf16） | — | — | — | **21.93** | −32.7% |
| torch fp32（关 TF32） | 95.99 | 95.99 | 71.86 | 62.81 | +93% |

\* 416 那两格 bf16w 是跨进程单点（首次 autotune 之后），被同进程 A/B 推翻了 —— **以 3.1 的 A/B 为准**
（它们只是同一份权重、同一张图，跨进程 5 次中位能差 6%，压不住）。
趋势比数字重要：**bf16 的收益随 batch 变大**（416：b1 −21% → b8 −49%），
因为它吃的是 tensor core；384 在 b1 只有 −10%，因为 1567MB 权重在 batch 1 时是**带宽受限**
（每张都要把权重读一遍）—— 这也正是 `bf16w`（权重减半）最占便宜的地方（见 3.1）。

### 3.1 bf16（权重 fp32 + autocast）vs bf16w（权重也 bf16）：同进程交替 A/B

跨进程量的「bf16w 慢 36%」是**假数**：同一格跑两次能差 6%，5 次中位压不住。所以改成
同进程、两个会话都常驻、交替采样 8 轮取中位 —— **比值可信，绝对值偏大**（两份权重同时在卡上，
分配器互相挤，别拿去和上面那张表对）：

| | bf16 | bf16w | bf16w 相对 |
|---|---|---|---|
| 416 b1 | 25.87 ms | **22.54** | **−12.9%** |
| 416 b8 | 182.59 ms（22.82/张） | 181.56（22.69/张） | −0.6% |
| 384 b1 | 46.52 ms | **27.16** | **−41.6%** |
| 384 b8 | 244.36（30.54/张） | **191.21（23.90/张）** | **−21.8%** |

**bf16w 没有一档更慢**；384 上快得离谱（b1 −42%：权重 1567MB→783MB，正打在带宽上）。
精度差 <1 LSB（§4/§5 两档是分开列的），显存省 250MB（416）/1.1GB（384）。所以默认取 bf16w。

> 这张表里 tf32 与 onnx 打平（±5%），和 `DFM_TORCH_REPORT.md` 的「−5.1% / −13.2%」不冲突：
> 那是 20 次中位、安静机器；这份是 5 次中位、且机器上有另一个 GPU 任务在跑（见 `11）。
> **bf16 与 onnx 的差是 2 倍，不是几个百分点，干扰改不了这个结论。**

---

## 4. 精度表 ①：pred 输出（相对 ONNX CUDA fp32）

基准 = 同一份输入、同一个 batch 跑 onnxruntime CUDA fp32 的输出。
LSB = 输出值 × 255（也就是「换成像素之后差几个色阶」）。

### 4.1 真脸输入（8 张真对齐脸，batch 8 —— 这才是用户看得见的分布）

| 模型 | 精度 | 输出 | max | mean | p99 | max(L) | p99(L) | >2LSB | >4LSB | >8LSB |
|---|---|---|---|---|---|---|---|---|---|---|
| 416 | tf32 | face | 0.00114 | 0.000090 | 0.000367 | 0.29 | 0.09 | 0% | 0% | 0% |
| 416 | **bf16** | face | **0.01886** | 0.001499 | 0.005315 | **4.81** | 1.36 | 0.12% | 0.001% | 0% |
| 416 | bf16w | face | 0.01897 | 0.001551 | 0.005445 | 4.84 | 1.39 | 0.11% | 0% | 0% |
| 416 | **bf16** | prd_mask | 0.02014 | 0.000632 | 0.007683 | 5.14 | 1.96 | 0.95% | 0.013% | 0% |
| 416 | **bf16** | dst_mask | 0.02326 | 0.000699 | 0.007428 | 5.93 | 1.89 | 0.85% | 0.028% | 0% |
| 384 | tf32 | face | 0.00150 | 0.000114 | 0.000487 | 0.38 | 0.12 | 0% | 0% | 0% |
| 384 | **bf16** | face | **0.01638** | 0.001453 | 0.005459 | **4.18** | 1.39 | 0.11% | 0% | 0% |
| 384 | bf16w | face | 0.01878 | 0.001527 | 0.005277 | 4.79 | 1.35 | 0.04% | 0% | 0% |
| 384 | **bf16** | dst_mask | 0.03375 | 0.000300 | 0.007190 | 8.61 | 1.83 | 0.85% | 0.139% | 0.001% |

读法：**脸的输出** max 4~5 LSB、mean 0.38~0.40 LSB、p99 1.35~1.4 LSB —— 也就是「99% 的像素
差不到 1.4 个色阶」。遮罩的 max 大一点（5~9 LSB）但那是**平滑渐变**，所以「>2 LSB 的比例」
反而更高（0.85%）而 mean 更低（0.08 LSB）：遮罩上 1 个 LSB 的变化会沿整条渐变带摊开。
两档 bf16 的差别 < 1 LSB，**bf16w 不额外付精度代价**。

### 4.2 均匀噪声输入（分布外的最坏情况，同样的 batch 8）

| 模型 | 精度 | face max | face max(L) | >2LSB | >4LSB |
|---|---|---|---|---|---|
| 416 | tf32 | 0.00355 | 0.91 | 0% | 0% |
| 416 | bf16 | **0.05148** | **13.1** | **3.27%** | 0.9% |
| 384 | bf16 | 0.03548 | 9.05 | 0.73% | 0.1% |

噪声是**训练分布之外**的输入（每个像素都在最大梯度上），bf16 的误差被放大 2.7 倍、
>2 LSB 的像素从 0.1% 涨到 3.3%。写在这里是为了诚实：**bf16 的误差上界是「十几个色阶」，
但那是在模型根本不会见到的输入上**；真脸是 4~5 个色阶。

---

## 5. 精度表 ②：成片（走过真的 compose_frame）

pred 差 5 LSB 不等于成片差 5 LSB —— 中间还有遮罩混合 + 色彩迁移 + 反贴回整帧（都是 CPU，
会把差异按遮罩权重缩一次）。所以拿**生产那个 `backend_merge.compose.compose_frame`**
真跑两遍：同一张 1080×1920 真帧、同一组 68 点、两套 pred/mask 各合成一次，成片（uint8）逐像素比。

| 模型 | 精度 | 脸 | 成片 max(L) | mean(L) | p99(L) | >2LSB | >4LSB | >8LSB |
|---|---|---|---|---|---|---|---|---|
| 416 | bf16 | #0 | **5** | 0.0089 | 0 | 0.11% | **0%** | 0% |
| 416 | bf16 | #1 | 4 | 0.0088 | 0 | 0.06% | 0% | 0% |
| 416 | bf16w | #0 | 4 | 0.0076 | 0 | 0.09% | 0% | 0% |
| 384 | bf16 | #0 | 4 | 0.0084 | 0 | 0.13% | 0% | 0% |
| 384 | bf16 | #1 | 3 | 0.0080 | 0 | 0.05% | 0% | 0% |
| 384 | bf16w | #0 | 3 | 0.0077 | 0 | 0.03% | 0% | 0% |

**结论（这就是「肉眼是否可辨」的那句话）**：
成片 max 差 **3~5 个色阶**、mean **0.009 LSB**、**p99 = 0**（99% 以上的像素逐字节相同）、
**没有任何一个像素差过 4 LSB**。差异**不是整脸均匀偏移**，而是集中在脸内的高对比边缘
（发丝 / 睫毛 / 鼻翼）上零星的 1~5 个色阶 —— 在那个量级上，同一个模型换一次 ORT 的
provider（CUDA vs CPU）都比它大（0.00146 = 1.5 LSB，见 `DFM_TORCH_REPORT.md` `1 那张表），
而且 8-bit 视频过一遍 libx264 的量化噪声就是 ±2~3。

肉眼图（并排：原帧 / onnx fp32 / bf16 / 差图放大）：
`%TEMP%\dfm_bf16_quality\face_416_bf16.png`（脸框放大，差图 ×16）、
`compose_416_bf16.png`（整帧）、`crops_416.png` / `crops_384.png`（脸中心 + 脸边缘 ×3）。

---

## 6. 显存

| | onnx（会话） | torch tf32 | torch **bf16** | torch bf16w |
|---|---|---|---|---|
| 416（权重 514MB） | 1240 MB | 1824 | **1184** | **935** |
| 384（权重 1567MB） | 2434 MB | 2684 | **2502** | **1360** |

（torch 那三列是 `torch.cuda.max_memory_allocated`：8 个 batch 档全预热后的峰值，
**含** CUDA context，进程私有、不受别的进程干扰；onnx 那列是 `vram.py` 里
`_VRAM_SESS_RUNTIME_MB` 那一段早就标定过的「权重 + context + 运行时」。）

- **bf16 与 onnx 同量级**（416 甚至更低）：省的是激活，权重仍是 fp32。这就是它能保住精度尾部的
  原因（`Div` / `Sqrt` / `ReduceMean` 那截 normalize 与 dense 都留在 fp32）。
- **bf16w 才真的把权重砍半**（514→257、1567→783），384 那份整张卡省 **1.1GB**。
  代价：没有（§3.1 同进程 A/B：−12.9% / −0.6% / −41.6% / −21.8%，一档不输）。
- `_vram_plan` 的处理：`bf16w` 时 `model_mb × 0.5`（明着算进 sess_mb）；
  `bf16` 时**不加正增量**（实测不比 onnx 高，凭空加一个数只会把能跑的单子判死）。
  计划里多两个键：`dfm_backend` / `dfm_dtype`，日志与 status 都能回显「这一单按哪个后端算的账」。

---

## 7. 日志 / [eff] 回显

`[stage]`（加载时，一行）：
`换脸：dfm 走 torch(cuda) · 精度=bf16（TF32=开）· 416×416 · NHWC · 权重 172 张量/514MB · 预热单张 30.1 ms · 卡上 1185.0 MB · DFL_DFM_BACKEND 未设（默认 torch+bf16；回退用 DFL_DFM_BACKEND=onnx）`
`[dfm-torch] 预热 batch [1..8] · 单张 24.0 ms · 合计 1224 ms`

`[eff]` 那一行的末尾多一个键：`"dfm": "torch · bf16"`（`DFL_DFM_BACKEND=onnx` 时是 `"onnx · fp32"`）。
预览 worker 也走同一个后端（日志 `[merge-preview] dfm 走 torch(cuda) 精度=bf16`）——
预览是用户拿眼睛判断成片的那扇窗，两边精度不一致会让人以为哪边坏了。

---

## 8. 这一轮修掉的三个真 bug

1. **ONNX 的 int64 是有符号的**，protobuf 的 varint 读出来是无符号。少了这一步，
   `axes=[-1]` 变成 18446744073709551615 → `torch.mean(dim=(1.8e19,))` 直接抛
   （416 那张图 46 个 Div / 30 个 Slice 全靠它）。
2. **Gather 的索引常量不能跟着权重压 bf16**。416 的 Gather 索引是 float 常量（0..2559），
   bf16 的整数精度只有 256 → 1500 被舍成 1504 → CUDA 断言 `index out of bounds`
   （要是舍进范围就是**静默取错通道**）。现在索引在**编译期**就转成 int64。
3. **预热会把 cuDNN 的 workspace 叠罗汉**。关 TF32（`fp32` 档）时 cuDNN 给 416 batch 8
   挑的算法要一大块 workspace，8 个 batch 档各留一块 → `reserved` 涨到 **8292MB**
   （比 8GB 卡还大，只能换页），实测 batch 1 从 85ms 掉到 147ms。修法：每档预热完
   `torch.cuda.empty_cache()`。

---

## 9. 回归

三份验收**全部通过**（默认档 = torch + bf16w，server venv，同一台机器同一时段；每条都跑真后端 +
真视频 1080×1920）：

| 验收 | 结果 | 用时 | 关键数字 |
|---|---|---|---|
| `app/tools/accept/fused_export_accept.py` | **PASS 64 · FAIL 0**（exit 0） | 137.2 s | dfm 段 **27.54 ms/帧**（onnx-fp32 基线 39.86，**−31%**）· 帧预算 30 不降档 · batch 8 · 降档/拒绝两档都在 |
| `app/tools/accept/stage6_accept.py` | **PASS 64 · FAIL 0**（exit 0） | 68.6 s | ③pred 批处理正常 · ⑤合成 33.5 帧/秒 · 与 4.0 逐像素 **mean 0.001 / max 1~2**（阈值 2/8）· 成片 vs 单帧预览 mean 3.6~4.0（阈值 8） |
| `app/tools/accept/proj_pool_accept.py` | **16 项通过 · 0 失败**（exit 0） | — | 帧速 38.3~41.5 · 显存峰值 2512 MB · 各档逐像素 **0.0000 LSB / 120 帧逐字节相同** · 续跑与一次跑完**逐字节一致** |

> **端到端帧/秒这一栏我没法给准数，原因写在下面**：四次 full fused 实测拿到 20.76 / 19.51 /
> 17.73 / 12.82 帧/秒，同一份代码 —— 因为**墙已经从 GPU 段挪到 CPU compose**（compose 159~199 ms/帧、
> cmp_wait 占墙钟 52~55%；dfm 只占 GPU 段的 52~54% 占用率）。只要机器上还有别人在吃 CPU，
> 这个数就掉，跟 dfm 后端没关系。**要引用端到端 fps 请用独占机器上的那次（B 那单）**；
> 这份报告只负责「dfm 段 ms」这一栏（26.6~29.4 ms/帧，四次之间差 ≤10%，不受 CPU 影响）。

三条验收里值得单独抄出来的数字：

- **fused b 段（fused vs 分段逐像素对拍）**：全片 mean **0.2962 LSB**，最差单帧 mean 0.8734 / max 25
  —— 阈值 1.0。这个数和 onnx 时代（0.30 / 0.54 / 0.66 / 0.68 / 0.70 五次实测）**在同一个量级**，
  也就是说 bf16 换进来的那 5 LSB 被「两条路径 batch 不同 → cuDNN 挑不同 kernel」这件事盖住了。
- **stage6 h 段（对 4.0 的逐像素回归）**：mean 0.001 / max 1~2 —— 与 onnx 时代**完全一致**（阈值 2/8）。
  这条最有说服力：4.0 自己跑的也是 onnx fp32，我们换了后端仍然落在它的 1 LSB 里。
- **proj_pool C 段**：攒批 4 与攒批 8 的产物 **逐字节相同**（0.0000 LSB / 120 帧全等）。
  bf16 在这个方向上甚至是**更确定**的：exp 的 bf16 kernel 不按 batch 挑不同算法。

### 9.1 动过的一条判据（唯一一处）

`fused_export_accept.py` 原来有一条「换快档检测器后**帧/秒**必须更高（说明瓶颈在检测模型）」。
dfm 从 40.7 ms 降到 27 ms 之后它**不成立了**：

| | 默认档 fps | fast 档 fps | fast 档 dfm |
|---|---|---|---|
| onnx fp32（改之前） | 16.57 | **18.29** ✓ | 38.57 ms |
| torch bf16（改之后） | **20.76** | 18.24 ✗ | 24.19 ms |

fast 档两次几乎一样（18.29 / 18.24）—— 因为它只动 detect，**根本没吃到 dfm 省下的那 14 ms**；
而默认档已经涨到 20.76。瓶颈已经从「GPU 段（detect+dfm 抢同一段）」挪到「CPU compose」
（cmp_wait 占墙钟 52~55%、compose 41~44%）。所以判据改成**因果那一半**：
换 fast 档之后 `per_stage_ms.detect` 必须真的更短（**47.8 → 25.2 ms/帧**），帧/秒只作信息打印。
理由和两组实测都写进脚本注释了（`fused_export_accept.py` 440~470 行）。

第二条：`case_c` 的伪造显存从 **1850 → 1600**。原因不是它坏了，是**它变便宜了**：
这一档断言「30 帧放不下 → 自动降到 16 帧并成功出片」，而 1600/1850 这两个数原本是照
onnx 的 1026MB 会话算的。bf16w 的会话只有 **769MB**（权重 257 + context 512），同 1850
空闲下 avail 1594 已经放得下 30 帧 / 批 2（1467）—— 于是「不降档」是**对的行为**。
要把 30 帧挤出去得把 avail 压到 1467 以下：1600 → avail 1344 → 16 帧 / 批 1（1276）采用。
拒绝那一档（1200 → avail 944 < 8 帧的 1212）两边都还成立，没动。

### 9.2 两个「看着对、实际漏」的坑（这一轮补的）

1. **`_vram_release_inproc` 只 pop 了 `_DFM_SESS` 元组**。torch 后端的会话还挂在
   `backend_models.dfm_torch._CACHE` 里 —— 元组一 pop，引用就断了，但那份权重**还在卡上**，
   `empty_cache()` 也收不回来（模块级缓存持有强引用）。后果是「导出收尾」之后显存不降，
   用户下一次预览/导出被自己的后端堵死（正是这一轮用户报过的那个病）。
   修法：`_vram_release_inproc` 里对 pop 掉的每个路径调 `dfm_torch.unload(path)`。
2. **预览 worker 与成片精度不一致**（我一开始想统一，实现完发现是错的，已回退并写清原因）：
   预览 worker 不是本进程的线程，它是**用 4.0 引擎的 python 起的独立脚本**
   （`_PREVIEW_WORKER_SRC` 落盘到 `%TEMP%\dfl41_merge_preview\`，import 的是 MergeStudio +
   引擎自带的 torch 2.7），**压根 import 不到 4.1 的 `backend_models`**。硬接会得到
   `NameError: _dfm_backend_sel`（stage6 就是这么红的）。所以预览继续走 ORT fp32，
   代价是同一帧预览 vs 成片差 ≤5 LSB —— stage6 g 段实测 mean 3.6~4.0（阈值 8），可以接受。
   等引擎那边的 torch 版本与这边对齐（或 worker 换用 4.1 的解释器）再谈统一。

---

## 10. 复现

```bat
:: 速度 + 显存 + 对拍（每格一个进程；reps 5 约 6 分钟）
runtime\server\Scripts\python.exe app\tools\bench\dfm_backend_bench.py --sweep --reps 5 ^
    --dtypes fp32 tf32 bf16 bf16w --batches 1 4 8 --run myrun

:: 真脸质量表 + 肉眼图（一进程一个 (model,dtype)）
for %m in (416 384) do for %d in (onnx tf32 bf16 bf16w) do ^
    runtime\server\Scripts\python.exe app\tools\bench\dfm_bf16_quality.py --dump %m:%d
runtime\server\Scripts\python.exe app\tools\bench\dfm_bf16_quality.py --crops
runtime\server\Scripts\python.exe app\tools\bench\dfm_bf16_quality.py --compose
```

---

## 11. 边界（没做 / 不该做）

- **xseg（TRT int8）与 detect（TRT bf16）一行没动**。这一单只改 dfm。
- **ONNX 那条路完整保留**（`DFL_DFM_BACKEND=onnx`），连磁盘上的 TRT 引擎逻辑都在。
- 速度表是在**机器上还有另一个 GPU 任务**的时候量的（5 次中位）。安静机器上 bf16 只会更快；
  这也是为什么表里 onnx 的绝对值比 `DFM_TORCH_REPORT.md` 差 5~8%（那份是独占机器量的）。
- `torch.compile` 没上：这台机器没有 triton（inductor 起不来），上次已经量过
  `cudagraphs` 的收益是 0~2.8%，不值得为它加一个编译期。
- bf16 的误差**不随 batch 单调**（416：b1 4.8 → b8 4.8 LSB；384：b1 5.5 → b8 4.2），
  所以没有「小批更准」这种说法，别拿 batch 当精度旋钮。
