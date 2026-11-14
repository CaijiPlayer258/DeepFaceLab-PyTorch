# dfm 能不能用 PyTorch 跑：四条后端 × 两个模型 × batch 1/8 实测

日期 2026-11-15 · 机器 RTX 5070 Laptop 8GB / 16 物理核 32 线程 / torch 2.7.0+cu128 / ORT 1.26
脚本：`app/tools/bench/dfm_torch_graph.py`（ONNX→torch 编译器）、`dfm_torch_bench.py`（四路对拍基准）、
`dfm_cudagraph_bench.py`（单配置 CUDA Graph）、`dfm_node_parity.py`（节点级对拍）、`fused_gpu_util.py`（生产流水线采样）、
`fused_pipeline_bench.py`（原型 C 模式，已有）。产物 JSON 在 `%TEMP%`。

---

## 0. 一句话结论

**能跑，而且略快**：同一份权重、同一个数值结果（max|diff| ≈ ORT 自己 CUDA-vs-CPU 的量级），
torch CUDA eager（TF32 开）在 batch1 快 3%、batch8 快 5~13%。但**速度差只有一个零头**，
gpu 利用率实测 69~73%、GPU 段空闲 11~16%，把空闲填掉最多 +13%，
而「让 CPU 参与推理」在这台机器上是**负收益**（CPU 已经 11/16 核在跑 compose/encode/align）。

---

## 1. dfm 里到底有没有 torch state_dict？

| | `张元英.dfm`（384） | `阿証証証証-416.dfm`（416） |
|---|---|---|
| 大小 | 1.64 GB | 539 MB |
| 文件头 | `08 07 12 07 tf2onnx 1a 05 1.9.3` | `08 07 12 07 pytorch 1a 06 2.12.1` |
| producer | tf2onnx 1.9.3（TF 转出来的老 DFM） | pytorch 2.12.1（本仓库 `export_dfm()` 导出） |
| `metadata_props` | 空 | 空 |
| 图节点 / initializer | 183 / 94 | 1065 / 78 |
| 权重命名 | `encoder/res5/conv2/weight/read__17`（TF 风格） | `encoder.down1.conv1.weight`（**torch 模块属性名**） |

- **没有 PyTorch state_dict**：两个 .dfm 都是纯 ONNX protobuf，权重只在 initializer 里，
  没有 metadata / 附挂 checkpoint。所以「引擎 Model_SAEHD 直接读 .dfm」不成立
  —— 引擎读的是 `.pth/.npy`（`ModelBase.save/load`），跟 .dfm 是两条路。
- **但它是可逆的**：416 那份的 initializer 名字就是 `Model_SAEHD` 子模块的属性名，
  权重可以直接灌回 torch 模型（只缺 `archi/ae_dims/e_dims/d_dims` 这些结构参数，
  从 `<name>_SAEHD_data.dat` 或从 conv 的通道数反推即可；本次不需要）。
- 本次走的是**更通用**的那条：把 ONNX 图静态折叠（Shape/Reshape/Slice/Pad/Mod/Range/Gather 全折成常量）
  再**代码生成**成 `nn.Module.forward`。416：1065 节点 → 156 个真算子；384：183 → 183。

### 等价性（不是「看起来像」，是逐节点对拍）
`dfm_node_parity.py` 用「前缀子图 + 二分」找第一个分叉节点，最终三路输出：

| 模型 | out_face_mask | out_celeb_face | out_celeb_face_mask |
|---|---|---|---|
| 416 | 0.00111 | 0.00248 | 0.00199 |
| 384 | 0.00035 | 0.00162 | 0.00082 |
| **参照：ORT CUDA vs ORT CPU** | 0.00146 | 0.00298 | 0.00182 |

即 torch 与 ORT 的差异 = ORT 自己换 provider 的差异。**同一个模型**。

> 过程中修掉的两个真 bug（都会让像素全错但形状不变）：
> `DepthToSpace` 的 DCR/CRD 通道序写反（416 是 CRD）；常量第二次引用时丢了 `self.`。

---

## 2. 表一 · 416（阿証証証証-416，539MB，df 架构）

输入 NHWC 1×416×416×3，float32 [0,1]；每格 = 20 次中位（ms）。**每格都是单配置独立进程**。

| 后端 | B=1 总 ms | B=1 每张 | B=8 总 ms | B=8 每张 | 相对 A |
|---|---|---|---|---|---|
| **A ORT CUDA EP**（现状基线） | 37.06 | 37.06 | 311.87 | 38.98 | — |
| **B torch CUDA eager**，TF32 开 | **35.84** | 35.84 | **295.96** | 37.00 | **−3.3% / −5.1%** |
| B 同上 + channels_last | 44.96* | 44.96 | — | — | +21% / — |
| B TF32 关（fp32 硬算） | 191~258* | — | — | — | +5x / — |
| B + TF32 关 + channels_last | 383~565* | — | — | — | +10x / — |
| **C torch.compile** | 不可用 | — | 不可用 | — | 见 §3 |
| C 替代 `backend="cudagraphs"` | 146.2** | 146.2 | — | — | — |
| **D torch CPU** 4 线程 | 876.96 | 876.96 | — | — | +23x |
| D torch CPU 8 线程 | 618.76 | 618.76 | — | — | +16x |
| D torch CPU 16 线程 | 542.46 | 542.46 | — | — | +14x |
| **Q4 手工 CUDA Graph**（固定 batch/地址） | 35.84 | 35.84 | 297.58 | 37.20 | −3.3% / −4.6% |
| 显存峰值 | 661 MB | | 1611 MB | | |

* 在同一进程里连跑四个 CL×TF32 变体会被上一个变体的显存碎片带偏（416 B1 同配置 35.8 ↔ 50.9），
所以「channels_last / TF32」两列只能**同进程内横向比**：CL 在 B1 反而更慢、在 B8（384 表）更快；
TF32 关掉直接 5 倍慢，这条没有歧义。
** 那次跑在 TF32=关 的上下文里，等价基准是 191ms（不是 35.8），所以它是「相对 fp32 eager −24%」，
不等于「相对最优 −24%」。

---

## 3. 表二 · 384（张元英，1.64GB，tf2onnx 老 DFM）

| 后端 | B=1 总 ms | B=1 每张 | B=8 总 ms | B=8 每张 | 相对 A |
|---|---|---|---|---|---|
| **A ORT CUDA EP** | 38.26 | 38.26 | 305.83 | 38.23 | — |
| **B torch CUDA eager**，TF32 开，contiguous | 38.96 | 38.96 | 273.16 | 34.15 | +1.8% / **−10.7%** |
| **B + channels_last**，TF32 开 | **38.27** | 38.27 | **265.42** | **33.18** | **0.0% / −13.2%** |
| B TF32 关 | 191.07 | 191.07 | 1945.7 | — | +5x / 显存溢出 |
| **D torch CPU** 4 / 8 / 16 线程 | 932.0 / 689.2 / **622.5** | 同 | — | — | +24x |
| **Q4 手工 CUDA Graph** | 38.29 | 38.29 | **266.22** | 33.28 | 0.0% / −12.9% |
| 显存峰值 | 1768 MB | | 2504 MB | | |

**batch 8 本身不划算**：两个模型、两个后端，B8 的「每张」都没比 B1 便宜
（416：37.06 → 38.98；384：38.26 → 38.23），差别全在 channels_last 与 TF32 上。
这台 5070 Laptop 的 dfm 是算力受限，不是 launch 受限。

---

## 4. 表三 · torch.compile / CUDA Graphs

| 项 | 结果 |
|---|---|
| `torch.compile(mode="default")`（inductor） | **起不来**：`TritonMissing: Cannot find a working triton installation`（Windows + 官方轮子不带 triton；engine venv 与 server venv 都没有） |
| `torch.compile(backend="cudagraphs")` | 可用。**首次编译 1.70 s**，稳态 146.2 ms（TF32 关上下文） |
| 手工 `torch.cuda.CUDAGraph` 捕获 | 成功，**捕获 0.12~0.17 s**，回放：416 B1 35.84（±0）、416 B8 297.58（**−0.5%**）、384 B1 38.29（+1.7%）、384 B8 266.22（**+2.8%**） |
| 结论 | **batch 固定 + 地址固定之后，CUDA Graph 基本不再省**：B8 的每张 37.0→37.2（416）、33.2→33.3（384）。没有 inductor 的算子融合，图里每个 kernel 本来就是大 kernel，launch 开销占比 <1%。 |

> 坑：在同一个进程里先跑 ORT 再跑 torch，8GB 显存会被 arena + 碎片吃满，驱动回退到系统内存，
> CUDA Graph 会量出 **7420 ms/张** 这种假数（显存峰值 9.07GB > 8.15GB 总量）。
> 所以 `dfm_cudagraph_bench.py` 一个进程只做一个配置；编译器还按「变量最后一次使用」回收了
> 中间张量名，416 batch8 的峰值从 9.07GB 降到 1.61GB。

---

## 5. 表四 · CPU/GPU 混合（Q3）

用实测两条曲线配比（GPU 跑 N 张 + CPU 跑 M 张，两边并行、互不抢核的上限）：

| 模型 | A（GPU）ms/张 | D（CPU 16 线程）ms/张 | GPU 单独 img/s | CPU 单独 img/s | 理想配比 img/s | 增益 |
|---|---|---|---|---|---|---|
| 416 | 37.06 | 542.46 | 26.98 | 1.84 | **28.82** | **+6.8%** |
| 384 | 38.26 | 622.54 | 26.14 | 1.61 | **27.75** | **+6.2%** |

CPU 4 线程时（876.96 / 932.0 ms）增益降到 +3.1% / +2.9%。

**但这是「CPU 核全闲着」的理想值，实测这台机器不是**：fused 流水线跑起来时
CPU 占用 **11.01 / 16 物理核**（compose 8 线程 102 ms/帧 + encode 2 线程 76 ms/帧 + align + decode）。
再塞 542 ms/张 的 dfm 前向进去，只会把 compose/encode 挤慢 —— 净效果**大概率是负的**。
所以「CPU 参与推理」在这台机器上值 **0 ~ +6%（上限），现实中为负**。

---

## 6. GPU 到底空转多少（生产流水线实测，150 帧 1080×1920，416 dfm）

`fused_pipeline_bench.py --modes C`（同一个 fused 结构、同一批 stage 函数）：

| | batch=1（`_pred_pick_batch` 自选） | batch=8（强制） |
|---|---|---|
| 帧/秒 | **17.06** | **16.22** |
| 每帧墙钟 | 58.61 ms | 61.65 ms |
| nvidia-smi GPU 均值 | 69.2% | 73.4% |
| **GPU 段空闲** | **11.1%** | **16.0%** |
| CPU 核 | 11.01 | 10.67 |
| decode | 19.76 ms / 33.7% | 21.51 / 34.9% |
| **detect**（retinaface_10g） | **33.10 / 56.5%** | **56.90 / 92.3%** |
| align | 29.30 / 12.5% | 32.78 / 13.3% |
| **dfm** | **39.86 / 69.4%** | **42.70 / 71.1%**（实际均值 5.56 张/批） |
| **xseg** | 11.18 / 19.5% | 7.73 / 12.9% |
| **compose**（8 线程） | **102.14 / 21.8%** | **137.49 / 27.9%** |
| **encode**（PNG level1, 2 线程） | **76.07 / 64.9%** | 77.08 / 62.5% |
| write | 4.86 / 8.3% | 4.92 / 8.0% |
| 帧缓冲 | 峰值 30（上限 30，**打满**） | 峰值 30（打满） |
| 显存峰 / RSS | 3020 MB / 2475 MB | 3078 MB / 2469 MB |

**读法**：
- GPU 侧天花板 = 1000/(39.86+11.18) = **19.6 帧/秒**；实际 17.06 → **只用掉 87%**，差 13%（正好对上 11.1% 空闲）。
- 攒到 batch 8 **更差**：dfm 每帧 39.86→42.70、detect 33.1→56.9、compose 102→137，
  帧率 17.06→16.22（**−4.9%**）。原因就是仓库注释里写的：GPU 段变成每 ~400ms 猛干一次，
  把并发跑在 CPU 上的 compose/encode 挤成一阵一阵的。**「攒批」不是免费的**。
- 所以「把 GPU 填满」的上限是 **+13%~+15%（17.06 → 19.6）**，
  而「让 CPU 参与推理」在这条线上是把已经吃紧的核再分出去 —— 方向反了。

---

## 7. 逐条回答

1. **dfm 能不能用 PyTorch 跑**：能。.dfm 里没有 torch state_dict，但用「ONNX 静态折叠 + 代码生成」
   可以拿到逐节点等价（max|diff| 与 ORT 自己换 provider 同量级）的 nn.Module；
   416 的 initializer 名就是 `Model_SAEHD` 的属性名，想走「原生 torch 模型」这条路也有据可依
   （缺的只是 `archi/ae_dims/e_dims/d_dims`，能从 `_data.dat` 或 conv 通道数反推）。

2. **四条路的 ms**：见表一/表二。最好的一格是 **384 + batch8 + channels_last + TF32 开 = 33.18 ms/张**
   （ORT 38.23，**−13.2%**）；416 是 37.00 vs 38.98（**−5.1%**）。

3. **CPU/GPU 混合**：理想上限 +6.8%（416）/ +6.2%（384），现实中因为 CPU 已占 11/16 核而为负。

4. **torch.compile + CUDA Graphs**：inductor 在这台机器上没有 triton，起不来；
   `backend="cudagraphs"` 与手工捕获都能成功，但**收益 0~2.8%**（B8 384 最好 +2.8%），
   首次编译 1.70 s、捕获 0.17 s。结论：**不值得为它改产品代码**。

**「CPU 参与推理」和「GPU 吃满」哪个解释更符合实测**：
都不是主因。实测 GPU 已经 69~73%（段空闲 11~16%），把空闲填掉最多 +13%；
CPU 侧 compose 102 ms/帧、encode 76 ms/帧才是真正的墙。传闻如果真是「CPU 帮推理」，
在这台机器上解释不通；如果是「别让 GPU 等批、别让 CPU 段饿着」，方向对，但**攒批反而是反效果**。

---

## 8. 我们比对方快的地方（交叉验证传闻）

- **检测/遮罩是现成的、且在关键路径上**：detect 33.1 ms/帧（56.5% 占用）、xseg 11.2 ms/脸（19.5%）。
  对手若用同一套 retinaface_10g + XSegLite，这两段没有优势；若他们也在 onnxruntime 上跑 detect，
  我们这边 detect 已经接到 `workspace/_engines` 的 TRT 引擎（本仓库 app 层没接，见 pipeline_fused 注释）。
- **dfm 我们并不慢**：batch1 37~39 ms、batch8 38~39 ms/张；torch 最好 33.2 ms/张（−13%）。
  也就是说「换个框架」能拿到的全部收益 ≤13%，落在 58.6 ms/帧的流水线上只值 **≈ +8%（17.1 → 18.5 帧/秒）**。
- 真正拉开差距的会是 **compose（102 ms/帧 × 8 线程）与 encode（76 ms/帧）**这两段纯 CPU 的活，
  以及 detect 的批量/引擎化 —— 不是 dfm 的后端。
