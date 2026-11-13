# 16 · 显存预算与降级（_vram_plan）

> 「三条路」（融合视频导出 / 项目导出 / 预览）共用一份预算：**按预算跑，跑不了就明说**。
> 代码：`app/server/routes_merge.py` 的 `_vram_plan` 一段（搜 `显存预算（三条路共用）`）。
> 验收：`app/tools/vram_plan_accept.py`。

## 1. 为什么要有这一层

以前只有两道**事后**保险：

| 位置 | 以前的做法 | 问题 |
|---|---|---|
| `_fused_budget` | 只按帧预算 30/16/8 降档 | 不看**模型体量**：2GB 的 384 dfm 和 0.5GB 的 416 dfm 按同一个预算跑；连 8 帧都装不下时仍然「压到 8 帧照跑」，让 OOM 在流水线里炸 |
| `_pred_pick_batch` | 先跑，OOM 了 8→4→2→1 退档 | 退档本身已经在崩；而且 batch 的开销没进任何预算 |

现在：**加载模型之前先算账**。装得下 → 按算出来的帧预算/批/worker 数跑；装不下 →
HTTP 400（消息里带「需要 X GB / 可用 Y GB / 建议」）；用户显式要 CPU → 回落 CPU EP。

## 2. 公式

```
可用预算 avail = free × (1 - 0.20) - 512MB
                 ↑ 20% 是驱动/碎片/arena 增长余量，512MB 给桌面合成器
一份会话 sess  = model_mb(权重，≈ .dfm 文件体积) + 512MB(CUDA context)
固定开销 fixed = sess × 进程数 + 300MB(检测器 + 2D106 + XSeg)
帧缓冲         = frames × size²×3×4×4/1MB        （416 → 7.92MB/帧）
dfm 批         = f(batch)                         （查表，见下）
总需求 need    = fixed + 帧缓冲 + dfm 批
可行 ⟺ need ≤ avail
```

`dfm 批` 的实测台阶（416，RTX 5070 Laptop 8GB，ort-CUDA；数字是「整卡增量 − 会话本身」）：

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| +MB | 12 | 79 | 1036 | 1091 | 5117 |

batch 4 起 ORT 的 arena 换到另一个池子 → 台阶不是直线，所以 `_vram_batch_mb()` 是查表 +
分段线性；`size ≠ 416` 时按 `(size/416)²` 缩放。

会话本身的标定：**416 实测 1161MB**（= 514 权重 + 512 context + 135 arena）；
**384 实测 ≈ 2.07GB**（= 1567 权重 + 512 context），两处对得上。arena 不单列进公式，
由那 20% 预留吸收 —— 单列会把 8GB 卡实测「2 个 worker 行、3 个 spill」误判成「只装得下 1 个」。

## 3. 三条路怎么接

| 路径 | 旋钮 | 降档顺序 | 拒绝 |
|---|---|---|---|
| 融合视频导出 `_fused_run` | 帧预算 × dfm 批 | 先保帧预算（流水线重叠靠它），再降批：(30/16/8) × (8/4/2/1) | HTTP 400 |
| 项目导出 `_proj_run` | worker 进程数 | 3 → 2 → 1（每份 = 会话 + context） | HTTP 400 |
| 预览 `_PreviewWorker` | 只问「一份会话装不装得下」 | 不降档，装不下就拒绝 | HTTP 400 |

拒绝点都放在**路由里**（不是 worker 线程里）：线程里抛异常只会变成「任务失败」，
前端拿不到 400，也拿不到那句带数字的消息。

## 4. 开关

| 环境变量 | 作用 |
|---|---|
| `DFL_VRAM_FREE_MB` | 伪造「可用显存」，三条路共用（验收/排障；`DFL_FUSED_VRAM_FREE_MB` 仍认） |
| `DFL_ALLOW_CPU_FALLBACK=1` | 显存不够时不拒绝，回落 CPU EP 照跑（416 实测慢 ~17 倍） |

CPU 回退是**一处开关、全链路生效**：`_dfm_session` 建 CPU session、`_stage_xseg` 走 CPU
provider、预览 worker 收 `DFL_FORCE_CPU_EP=1`。日志与 status 都写「★ 当前在 CPU 上跑 ★」。
默认关 —— 宁可拒绝，也不悄悄变慢 17 倍。

## 5. status 字段

三个 status 端点都带 `vram_plan`（+ 顶层 `on_cpu`）：

```
{ok, cpu, mode, label, size_px,
 model_mb, sess_mb, ctx_mb, arena_mb, aux_mb, frame_mb, batch_mb,
 free_mb, total_mb, source, avail_mb, need_mb, peak_mb,
 frames, batch, procs, requested_frames, requested_batch, requested_procs,
 downgraded, steps[{frames,batch,procs,need_mb,avail_mb,fit}], reason, msg}
```

`steps` 是逐档记账（哪一档否决、哪一档采用、差多少 MB），排障时先看它。

## 6. 验收

```
runtime\server\Scripts\python.exe app\tools\vram_plan_accept.py
```

三档伪造显存（416 dfm）：1200 → 拒绝（需要 1.44GB / 可用 0.44GB）、3000 → 降档
（30 帧 / 批 8→2）、8000 → 正常（30 帧 / 批 8）；`DFL_ALLOW_CPU_FALLBACK=1 + 1200`
能出片且日志/状态写明 CPU；整轮后端日志断言「无 CUDA OOM」。
