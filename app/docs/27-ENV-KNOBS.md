# 27 · 合成链的 37 个环境变量（DFL_*）

> 一句话口径：**合成链上有 63 个 DFL_* 名字，其中 60 个真被读到；这 60 个里只有 2 个值得给用户选。**
> 本文主表 = 37 行（/api/merge/export 这条主链 + 合成 / dfm 两处旋钮，不含项目导出专用与 4.0 老名字），
> 附录 A/B/C 把剩下 26 个也归档 —— 加起来正好是 `grep -o 'DFL_[A-Z0-9_]*'` 数出来的 63。
>
> 4K 导出的编码器开关（`DFL_FUSED_ENCODER` 三件套）**不在上面这张表里** —— 它落在
> `app/server/_common/encoder_sel.py`，单独立一篇：[29-ENCODER.md](29-ENCODER.md)（4K 该不该换 NVENC）。
> 主表每行的「该不该暴露给用户」有四种取值，含义见下面第 1 节。

怎么自己数一遍（仓库根）：

```bash
grep -o 'DFL_[A-Z0-9_]*' app/server/backend_merge/*.py | cut -d: -f2 | sort -u | wc -l   # 63 个 token
```

---

## 1. 「该不该暴露给用户」四档

| 档 | 含义 | 界面上 |
|---|---|---|
| **用户可调** | 没有「一台机器上唯一正确」的答案；调它是在拿画质 / 显存 / 速度做取舍 | **给下拉**（设置页「性能」区） |
| **内部调优** | 已按实测定在最优；改了一般只会更慢或更抖（报告写在对应模块的注释里） | 不出现 |
| **排障与验收** | 伪造显存、注入 OOM、dump 中间结果、跳过回收……**生产别设**，设了跑出来的数就不是真实的 | 不出现 |
| **已禁用 / 有规则限制** | 代码里留着读取点，但产品口径已经把它关了（或只剩一个合法值） | 不出现，也没打算出现 |

为什么只放两个：其余 35 个都是「调坏了却不知道为什么慢」—— 用户改了 DFL_FUSED_CMP，
得到的是一整条流水线的吞吐变化，而不是一句话能说清的结果。这两个不一样：
精度是画质 / 显存的取舍，合成后端是「这张卡忙不忙」的取舍，两者都有正当的使用场景。

---

## 2. 用户可调（2 个）

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_DFM_DTYPE` | `bf16w` | dfm 在 **torch** 后端上的推理精度：`fp32`（关 TF32，与 onnx 对拍用，慢 2.4×）/ `tf32`（与 ORT 的 CUDA EP 同一数学模式，内部档）/ `bf16`（权重 fp32 + autocast）/ `bf16w`（权重也压 bf16，少占 250MB~1.1GB）/ `auto`（按权重体量选：≥900MB → bf16w，否则 bf16） | **给下拉**（自动 / fp32 / bf16 / bf16w；tf32 与 auto 只认 env） |
| `DFL_COMPOSE_GPU` | `0`（关） | 合成放 CPU（`compose_frame`，OpenCV）还是 GPU（`compose_frame_gpu`，torch CUDA）。8GB 卡实测：overlay 持平、seamless −13.6%（docs/26），所以**不默认开** | **给下拉**（自动 / cpu / gpu） |

这两行的界面形态、请求体口径与回显：

* 界面：设置页 →「性能」区两颗下拉，`data-cfg="dfm_dtype"` / `data-cfg="compose_backend"`，
  记忆存在 `localStorage["dfl-mg-cfg-memory"]`（与训练器参数页同一套键名约定，刷新后保持）。
* 请求体：`dfm_dtype=fp32|bf16|bf16w`、`compose_gpu=1|0`。**选「自动」时不带这两个键** ——
  带一个具体值等于替用户做了决定，以后他看到的是「自动」、跑的是当时那个值。
* 界面「自动」两级语义（**别混**）：
  · 界面「自动」= **不带这个键** = 走后端默认档：dfm 是 `bf16w`（今天的默认，速度最优）、
    合成是 CPU；
  · env 里的 `DFL_DFM_DTYPE=auto` 才是「按权重体量选」（≥900MB → bf16w，否则 bf16）——
    那一档没上界面，因为它在 8GB 卡上对 <900MB 的模型会退回 bf16（b1 实测 −12.9%），
    「自动反而更慢」比「自动选了个精度档」更难解释。
* 优先级：**env > 请求体 > 后端默认**。进程里显式设了这两个变量就让它赢（排障 / A-B 不用改界面），
  和请求体不一致会打一条 `[eff-warn]`。
* 回显：`[eff]` 那一行的 `dfm_dtype`（+ `dfm_dtype_src` = `ui|auto|default`）与
  `compose_backend`（+ `compose_backend_src`）；进度页「查看生效配置」抽屉里是 16 项。
  被跳过时（选了 gpu 但 torch/cuda 起不来 / seamless 不进 GPU / 分段链没有 GPU 合成）
  打 `[eff-warn]`：**选了 X、实际跑 Y、原因 Z**，并进 `status.eff_warnings[]` → 进度页横幅。

---

## 3. 内部调优（21 个）—— 已定最优，改了一般更慢

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_FUSED_PREP` | `4` | 快速导出的对齐 / 预处理线程数（B 配置；以前按 cpu//3 浮动，同一份代码在 8 核机跑出两种口径） | 不暴露 |
| `DFL_FUSED_CMP` | `8` | 快速导出 **CPU 合成**线程数。GPU 合成走 `DFL_COMPOSE_GPU_THREADS` | 不暴露 |
| `DFL_FUSED_BATCH` | `8` | dfm 攒批目标张数（还会被 session 原生上限与显存预算再压一次） | 不暴露 |
| `DFL_FUSED_BATCH_WAIT_MS` | `20` | 攒批超时。实测 20ms 是收益 / 延迟税的交点（8ms 太碎、40ms 憋气） | 不暴露 |
| `DFL_FUSED_BATCH_ADAPT` | `0` | 自适应攒批（投影攒不满就立刻送）。真素材上收益 ≈0（q_gpu 本身就是攒批器），留实现不留默认 | 不暴露 |
| `DFL_FUSED_DET_PREP` | `1` | 检测的 letterbox 在解码线程先算（关 = 回检测线程现算的老口径，两条路逐位一致） | 不暴露 |
| `DFL_FUSED_DECODE` | 空（= raw） | 解码管道：`raw`（rawvideo，软解，默认）/ `png`（ffmpeg→PNG→cv2） | 不暴露 |
| `DFL_CV2_THREADS` | 空（= 不动） | 进程级 OpenCV 内部线程数。孤立基准 4 比 1 好、整条线打平 → 默认谁都不动 | 不暴露 |
| `DFL_MERGE_WORKERS` | `8` | 六阶段链（分段导出）的合成 worker 数；`merge_workers` / `threads` 请求体字段优先 | 不暴露（设置页「并行进程」走请求体，0 = 自动） |
| `DFL_MERGE_CV2_THREADS` | `0` | 分段链每个 worker 里的 cv2 线程数 | 不暴露 |
| `DFL_MERGE_PNG_LEVEL` | `1` | 帧图 PNG 压缩级别（1 = 快、体积大一点） | 不暴露 |
| `DFL_PRED_BATCH` | `0`（自动） | 分段链 dfm 批大小（0 = 按显存 / 阶梯自动） | 不暴露 |
| `DFL_PRED_BATCH_HYST` | `1` | 批大小滞回 + 跨启动缓存。`0` = 逐字节回到「每次都重算」的旧行为 | 不暴露 |
| `DFL_DFM_BACKEND` | `torch` | dfm 推理后端：`torch`（默认）/ `onnx`（回退，含 TRT 引擎那条老路）。也认 `torch:bf16` 简写 | 不暴露（换了就是另一套数学，只能对拍用） |
| `DFL_COMPOSE_GPU_THREADS` | `4` | GPU 合成线程数。量出来的（1→4：overlay 14.7→21.0 帧/秒），再多只互相等 stream | 不暴露 |
| `DFL_COMPOSE_GPU_WARP` | `grid` | GPU 版 warp 实现：`grid` / `gather`（对拍用，两者逐位同结果） | 不暴露 |
| `DFL_COMPOSE_GPU_SEAMLESS` | `0` | seamless 是否也进 GPU。**默认 0**：Poisson 求解下 u8 遮罩边缘 0/1 翻转会放大到几十 LSB，且 seamless 挪 GPU 实测 −13.6% | 不暴露 |
| `DFL_VRAM_RESERVE_MODE` | 空（= 新口径） | 显存预留口径：`legacy` / `legacy20` = 老的「空闲 20% + 512MB」 | 不暴露 |
| `DFL_FUSED_TRIM_WS` | `1` | 会话释放后把工作集还给系统（`0` = 留 RSS，量内存曲线时用） | 不暴露 |
| `DFL_FUSED_FREE_PREVIEW` | `1` | 导出起手先卸掉常驻预览 worker 的模型（不卸的代价：帧预算 22→8，实测 22→15.8 帧/秒） | 不暴露 |
| `DFL_SESS_LEVEL` | 空 | 按 `low` / `mid` / `high` **伪造机器档位**来推导检测 / 遮罩会话配比（只影响推导，不改探测） | 不暴露（验收 / 复现用） |

---

## 4. 排障与验收（12 个）—— 生产别设

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_VRAM_FREE_MB` | 未设 | 伪造「可用显存 = N MB」（三条路共用）。设了之后所有显存决策都建在假数上 | 不暴露 |
| `DFL_FUSED_VRAM_FREE_MB` | 未设 | 快速导出专用的显存伪造（同上，历史名字，两处都读） | 不暴露 |
| `DFL_FUSED_BUDGET` | `0` | 钉住帧预算（不走显存探测），用来复现「固定 N 帧」的吞吐 | 不暴露 |
| `DFL_FUSED_DUMP` | 空 | `=41,80` 把这几帧的逐脸输入（原帧 / pred / 遮罩 / 几何）落盘，和分段 cache 逐位比 | 不暴露 |
| `DFL_ALLOW_CPU_FALLBACK` | `0` | 显存装不下时不拒绝、回落 CPU EP 照跑（慢约 17×）。默认拒绝才是对的 | 不暴露 |
| `DFL_VRAM_OOM_INJECT` | `0` | 在真加载前注入一次 OOM（验证回滚路径） | 不暴露 |
| `DFL_ALLOW_TEST_HOOK` | `0` | 上面那个注入的**总闸**：没有它，注入了也不生效（防止误开） | 不暴露 |
| `DFL_STAGE_KEEP` | `0` | 分段链成功后**不删** `<名字>_cache/`（数产物 / 续跑调试） | 不暴露 |
| `DFL_VRAM_NO_RECLAIM` | `0` | 回滚 / 对照：这一轮不卸预览模型（看「自己占着显存」对预算的影响） | 不暴露 |
| `DFL_VRAM_RECYCLE_WORKER` | `1` | 卸模型后回收 worker 进程（`0` = 只放张量，量「谁在占着」时用） | 不暴露 |
| `DFL_COMPOSE_GPU_ASSERT` | `0` | GPU 合成每次和 CPU 结果做断言（compose_gpu_accept 用；开着会慢一倍） | 不暴露 |
| `DFL_COMPOSE_GPU_SEAMLESS_EXACT` | `0` | seamless 走「精确但更慢」的求解路径（对拍用） | 不暴露 |

---

## 5. 已禁用 / 有规则限制（2 个）

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_DFM_TRT` | `0`（禁） | dfm 是否允许用原生 TensorRT 引擎。**规则：dfm 只走 ONNX / torch，不编 TRT**（用户口径，见 docs/26 与 pipeline_stages._stage_dfm 的注释）。显式写 1 只是给对照实验留的门 | 不暴露（这是产品规则，不是开关） |
| `DFL_DET_EP` | `auto` | 检测跑哪个 EP：`auto` / `gpu` / `cpu`。**实测把检测整段挪 CPU 是负收益**，所以 auto 的门②默认永远是「不挪」；只有显式写 `cpu` 才真挪 | 不暴露（只有一个合法答案） |

---

## 6. 附录 A：项目导出专用（15 个，/api/merge/export-project）

这条链是「逐帧 → <项目>/merged/ → 编码」，和视频导出的六段那条是两套并发模型，
旋钮也自成一组。**同样不暴露**：整套默认值是按模型体量 + 机器档位标定出来的。

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_PROJ_WORKERS` | 按模型体量 | 项目模式并行度（几帧同时在飞） | 不暴露 |
| `DFL_PROJ_READ` | `3` | 读帧 / 对齐线程数 | 不暴露 |
| `DFL_PROJ_SEND` | 空（= 裸 BGR） | 传帧格式：`jpg` 退回 q95 JPEG | 不暴露 |
| `DFL_PROJ_COMPOSE` | 自动 | 合成阶段是否并行 | 不暴露 |
| `DFL_PROJ_COMPOSE_THREADS` | `4` | 合成线程数 | 不暴露 |
| `DFL_PROJ_DFM_BATCH` | `max(计划 batch, 4)` | dfm 攒批目标 | 不暴露 |
| `DFL_PROJ_BATCH` | 计划值 | 覆盖「计划 batch」 | 不暴露 |
| `DFL_PROJ_BATCH_WAIT_MS` | `20` | 攒批超时 | 不暴露 |
| `DFL_PROJ_PNG_LEVEL` | `1` | .png 帧压缩级别 | 不暴露 |
| `DFL_PROJ_ENC` | `2` | 编码线程数 | 不暴露 |
| `DFL_PROJ_OUT` | 空 | 输出帧格式覆盖（png / jpg） | 不暴露 |
| `DFL_PROJ_BENCH` | `1` | `0` = 不起标定（只用缓存 / 保守默认） | 不暴露 |
| `DFL_PROJ_ADAPT_EVERY` | `200` | 自适应并发每 N 帧重算一次 | 不暴露 |
| `DFL_PROJ_ADAPT_PCT` | `88.0` | 自适应并发的占用阈值（%） | 不暴露 |

## 7. 附录 B：4.0 引擎口 / 老名字 / 兼容（8 个）

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_MERGE_THREADS` | `8` | `DFL_MERGE_WORKERS` 的老名字（只在前者没设时兜底） | 不暴露（用新名字） |
| `DFL_DFM_ENGINE` | 空 | `DFL_DFM_TRT` 的老名字，语义同上（只为兼容当年显式写 1 的人） | 不暴露 |
| `DFL_MERGE_ROI` | `1` | `0` = 退回整帧合成（慢一个数量级，只作对拍基准） | 不暴露 |
| `DFL_PREVIEW_WORK_TIMEOUT` | `15`（秒） | 单帧预览子进程的干活上限（超时杀 worker） | 不暴露 |
| `DFL_FORCE_CPU_EP` | `0` | 强制预览 worker 走 CPU EP（由 `DFL_ALLOW_CPU_FALLBACK` 在内部下传） | 不暴露 |
| `DFL_FACES_READ_THREADS` | `min(6, max(2, cpu//4))` | 切脸阶段读帧线程 | 不暴露 |
| `DFL_FACES_QUEUE` | `max(4, read×2)` | 切脸阶段在飞帧数 | 不暴露 |
| `DFL_FACES_BATCH` | `1` | 切脸阶段检测批次（>1 在真机上没有收益，默认关） | 不暴露 |

## 8. 附录 C：只在注释里出现、没有读取点（3 个）

| 变量 | 默认 | 作用 | 该不该暴露给用户 |
|---|---|---|---|
| `DFL_DET_EP_THREADS` | 无效 | 旧版「检测 EP 的线程数」想法，**没有任何读取点** | 不暴露（写了也不生效） |
| `DFL_SESS_PLAN` | 无效 | 旧版「按档位写死会话配比」想法，已被 `DFL_SESS_LEVEL` 取代 | 不暴露（写了也不生效） |
| `DFL_SESS_THREADS` | 无效 | 旧版「会话线程数」想法，已被自动推导取代 | 不暴露（写了也不生效） |
| `DFL_SR` | `off` | 人脸超分档位：`off` / `low`(0.35) / `mid`(0.6) / `high`(1.0)；见 [36-FACE-SR](36-FACE-SR.md) | 暴露 |
| `DFL_SR_BACKEND` | `auto` | `auto`（有 engine 用 engine）/ `trt` / `onnx` | 不暴露 |
| `DFL_SR_MODEL` | `weights/gpen_bfr_512.onnx` | 超分模型路径 | 不暴露 |

这三个只在注释 / 旧文档里被提到，代码里**没有任何读取点**。写它们不会报错，也不会生效；
名字留着是为了让人搜到「这条已经废弃」。
主表 37 + 附录 A 15 + 附录 B 8 + 附录 C 3 = **63**，与 `grep -o` 的结果逐个对得上。

---

## 9. 验收对照（改这两个旋钮时看这里）

| 断言 | 脚本 |
|---|---|
| 设置页两颗下拉存在、默认显示「自动」、刷新后保持（`data-cfg` 记忆） | `app/tools/accept/merge_ui_accept.py`（C / D 组） |
| 默认态请求体**不带** `dfm_dtype` / `compose_gpu`；选 fp32 / gpu 才带 | 同上（D2 / D3 / D4） |
| `[eff]` 有 `dfm_dtype`(+src) / `compose_backend`(+src)，16 个字段 | `app/tools/accept/eff_accept.py`（A4 / G 组） |
| 选了 gpu 但上不了 GPU → `[eff-warn]`「选了 X、实际跑 Y、原因 Z」 | `eff_accept.py`（G5 / G6），前端横幅在 F 组 |
| 进度页抽屉 16 项、来源标签含「自动」 | `app/tools/accept/progress_lit_accept.py`（B8 / B10） |
