# 合成语义的边界：视频模式 vs DFL 项目模式

> 这一页是**语义约定**，不是实现细节。改合成/导出相关代码前先读它，别把两种模式的规则混在一起。

## 参考基线（别搞错）

- **视频模式** 对照 **深变 4.0**（`D:/MyProject/DeepFaceLab-PyTorch`）。上游原版**没有**视频直接换脸这条管线。
- **DFL 标准项目模式** 对照 **上游参考实现**（`D:/MyProject/dfl-reference`）。
- 两者语义冲突时：视频模式以 4.0 为准，项目模式以原版为准，并在文档里写明差异。

## 两种模式，两套规则

| | **视频模式**（我们设计的） | **DFL 项目模式**（复刻原版） |
|---|---|---|
| 模型数量 | **多 dfm**，可同时用 | **只能单模型**，一次换完所有脸 |
| 脸→模型的分配 | **embedding 自动判**（ArcFace，阈值 0.35），chip 可手动覆盖 | **用户自己决定**：在提取 / 删帧阶段选出要换的脸 |
| 不匹配的脸 | 保持原像素，**绝不兜给任何模型** | 不在 aligned 里，自然不换 |
| 管线 | 对照 **深变 4.0**（DeepFaceLab-PyTorch）：切帧 → 切脸 → dfm 推理 → 遮罩 → 合成(纯 CPU) → 编码 | **完全复刻原版**：遍历 aligned → 按 `original_name` 找帧图 → 推理 → 合成 → 写 merged |
| 优化空间 | 自由 | 只能性能优化（并行、CPU 预编译、零拷贝），**流程与原版一致** |

## DFL 项目的结构（别记错）

```
A/                 这段视频的帧图
A/aligned/         提取出来的人脸 DFLJPG
（等价写法：data_dst/frames + data_dst/aligned）
```

- aligned 里每个 DFLJPG 带 `original_name` 字段，值 = 对应帧图文件名 → **靠它把 aligned 与帧图配对**。
- 不是 `data_src/aligned + data_dst.mp4`，那是提取输入与输出视频，不是项目。

## 为什么项目模式不支持多模型

原版当年的设计背景：

1. 多人脸流程特别复杂；
2.  embedding 不成熟，没有可靠的预训练模型。

所以他把「换哪些脸」交给用户：一次只推理一个模型，把 aligned 全换掉，分批完成。**这是刻意的语义边界，不是缺陷**。

## 为什么视频模式可以多模型

我们现在有可用的 ArcFace embedding（同人 0.65–0.99，不同人 0.00–0.30），所以：

- 人脸库按组 = 按模型组织；
- 每次检测出的脸算 embedding，与各组均值比，≥ 0.35 归该组 → 用该组 dfm；
- 低于阈值 → 不换（保持原像素），提示用户手动拖到某个组；
- 合成器里 chip 显示自动判定结果，手动一改即覆盖。

## 预览 vs 导出：两条代码路径（视频模式）

| | 单帧预览 | 六阶段导出 |
|---|---|---|
| 代码 | `MergeMaskedFace`（worker 子进程，一次算完） | `stage ①-⑥`（pred/mask 落盘 + 纯 CPU `compose_frame`） |
| 中间产物 | 无（内存出图） | `<视频>_cache/` |
| 依赖 cache | **不依赖**（不读不写不建） | 就是 cache |
| 耗时 | ≈230-240 ms/帧（1080×1920 热态，改前 compose 版 497） | ⑤ 合成 32 帧/秒 + 推理/编码 |

- 两条路**算法等价、不是同一份代码**：像素差允许存在。实测：预览 vs 导出帧（都无损）
  mean 0.02~0.03 / max 20~33；成片 vs 预览 mean ≈ 3.6~4（编码往返），容差 8。
- **预览永远不许读/写 cache**：不许"发现有 cache 就复用"，也不许为「和成片逐像素一致」
  把预览接回 compose —— 那样预览会慢一倍，还会因为 cache 在不在出两张不同的图。
- 细节与验收见 `15-SIX-STAGES.md` 与 `app/tools/accept/preview_fast_accept.py`。

## 落地时的检查清单

- [ ] 项目导出分支里**不要**出现 embedding 归组、不要用 `models[]` 数组；
- [ ] 视频导出分支里**不要**退化成「挑第一个 dfm 换所有脸」；
- [ ] 无人脸的帧：复制原帧，不跑 dfm；
- [ ] 无人脸库时：用左栏选中的单个 dfm 换所有脸（这是合法的降级，不是兜底错误）；
- [ ] 人脸库 embedding 缓存：进程重启清空、一次导出结束后清空。
## 推理后端：TRT 必须用 TensorRT 原生库（硬规定）

> **TRT 推理用 TRT 库，不准使用 ONNX 库的 TRT 提供者，那不一样。**

| 路径 | 是什么 | 允许 |
|---|---|---|
| **原生 TRT** `TrtSession` + `*.engine` | 直接用 TensorRT 库加载 builder 生成的 engine，CUDA Graph / IO binding / 零拷贝可用 | ✅ 只允许这一条 |
| `onnxruntime` 的 `TensorrtExecutionProvider` | ort 内部自己 build、行为不同、可能每次重编 | ❌ 禁止 |
| `onnxruntime` 的 `CUDAExecutionProvider` | 没有 .engine 时的回落 | ✅ 回落，需日志 |
| `onnxruntime` 的 `CPUExecutionProvider` | 兜底 | ⚠️ 仅最后，必须打醒目 WARN |

**判定顺序**：原生 TRT（有 .engine）→ ort-CUDA → CPU。任何回落都要在 status / 日志里写明实际后端与原因，禁止静默。

> **⚠️ dfm 例外（用户硬规则，优先于上面的顺序）**：`.dfm` 的推理**默认只用 onnxruntime**
> （CUDA EP / fp32）—— **不编 TRT 引擎、不降精度**。磁盘上就算有 `_engines/dfm_*.engine` 也不加载：
> `DFL_DFM_TRT` 默认 **0**，并且必须打一行
>
> ```
> [dfm] 发现 TRT 引擎 dfm_416_af9afd7fb186_sm1200_trt101339.bf16.engine（457.7MB）但按 DFL_DFM_TRT=0忽略，使用 onnxruntime CUDA
> ```
>
> 要做 TRT 对照只能显式 `DFL_DFM_TRT=1`（仅供实验，**按用户要求默认不使用 TRT**，默认口径不算它）。
> **范围只限 dfm**：XSeg / 检测器（tinymog · retinaface_10g）/ 2D106 的引擎照旧命中，不受这条影响。
>
> **连带的语义变化（要知道）**：dfm 走引擎时，fused 攒批与分段逐张的 fp32 结果**逐字节相同**
> （验收里 `fused vs 分段 = 0.0000 LSB`）。换成 onnxruntime 后这条不再成立 —— ORT/cuDNN
> 按 batch 组成选卷积核，两条路实测差 **0.30 / 0.54 / 0.66 / 0.68 / 0.70 LSB**（5 次）。
> 那不是精度退化（两边都是 fp32、都是 ONNX）；`fused_export_accept` 的门槛因此从 0.5 调到
> **1.0 LSB**（理由写在该脚本 `TOL_MEAN` 的注释里）。`DFL_DFM_TRT=1` 时仍能看到 0.0000。

**报告口径**：涉及推理性能时，必须按「原生 TRT / ort-CUDA / ort-CPU」三分类给结论，并附 `session.get_providers()` 与引擎文件名；笼统说「用了 GPU」不算结论。**dfm 那一档永远报 ort-CUDA**（默认 ONNX only），除非报告里显式注明 `DFL_DFM_TRT=1`。

缺 `.engine` 时用 `POST /api/engines/build` 生成（`app/server/_trt_build.py`，Python Builder API）。
