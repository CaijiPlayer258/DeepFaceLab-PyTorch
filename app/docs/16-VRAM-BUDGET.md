# 16 · 显存预算与降级（_vram_plan）

> 「三条路」（融合视频导出 / 项目导出 / 预览）共用一份预算：**按预算跑，跑不了就明说**。
> 代码：`app/server/routes_merge.py` 的 `_vram_plan` 一段（搜 `显存预算（三条路共用）`）。
> 验收：`app/tools/vram_plan_accept.py`（三档伪造显存）、`app/tools/vram_reclaim_accept.py`
> （**预览常驻 → 导出**这条真实路径 + 「谁占的显存」）。

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
free          = min(torch.mem_get_info, nvidia-smi.memory.free)
可用预算 avail = free × (1 - 0.20) + 可回收 - 512MB
                 ↑ 20% 是驱动/碎片/arena 增长余量，512MB 给桌面合成器
                 ↑ 可回收 = 我们自己的预览 worker 里那份常驻会话（导出前会真回收掉）
一份会话 sess  = model_mb(权重，≈ .dfm 文件体积) + 512MB(CUDA context)
固定开销 fixed = sess × 进程数 + 300MB(检测器 + 2D106 + XSeg)
帧缓冲         = frames × size²×3×4×4/1MB        （416 → 7.92MB/帧）
dfm 批         = f(batch)                         （查表，见下）
总需求 need    = fixed + 帧缓冲 + dfm 批          （resident：会话那项不计，见 §2.2）
可行 ⟺ need ≤ avail
```

### 2.1 free 为什么是两个来源取小

Windows/WDDM 上 `torch.cuda.mem_get_info` 报的空闲**比 nvidia-smi 高**：本机空载
实测 6999 vs 5693 MiB。差的 1.3GB 不是谁算错 —— mem_get_info 报的是「驱动还愿意
答应多少」（不够就换页/回落共享内存），真分配下去立刻变成 nvidia-smi 的 used。
拿它当预算是**乐观到 OOM**；nvidia-smi 的 free 才是「这张卡上所有进程之外还剩
多少」。所以两边都探、取小，`source` 写成 `min(torch 6999 / smi 5693)`，计划里另带
`free_torch_mb` / `free_smi_mb` 两个键，排障时一眼看见差多少。

### 2.2 三种「容易把可跑的活判死」的口径

| 情形 | 以前 | 现在 |
|---|---|---|
| 预览过一帧（worker 里 dfm 常驻 2.0GB），再点导出 | 只看 free → 空闲 1.4GB → **拒绝** | 起算前先 `_vram_reclaim()` 把 worker 卸掉，等驱动真回收（轮询 ≤4s）再探 → 空闲 3.4GB |
| 导出过程中，加载完 dfm 后拿实测尺寸重算一次（文件名没带尺寸时必走） | 把「会话 1567+512MB」**又算一遍**，而它已经在卡上了 | `resident=True`：会话那项不计入需求，只判增量（检测/遮罩 + 帧 + 批） |
| 六阶段连着跑（pred → mask → merge） | 每个阶段的闸门都按「一份新会话」算 | 阶段要的模型已经在 `_DFM_SESS` / 检测器缓存 / `_XSEG_SESS` 里时 `resident=True`（`_stage_model_resident`）；真装不下且没有任务在跑 → `_vram_release_inproc()` 放掉旧会话再算一遍 |
| worker 忙/回收超时 | —— | 那 2GB 仍算进 `avail`（可回收），消息里注明「导出前会再收一次」 |

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

### 3.1 拒绝发生在「起 job 之前」（截图 2 那一轮）

用户点导出、显存不够时，**一个字节的任务都不许起**：不建 job、不加载模型、不开线程。
四条入口各自先算账：

| 入口 | 闸门 | 说明 |
|---|---|---|
| POST /api/merge/export（fused=1） | _vram_guard(mode=frames) | 帧预算 × dfm 批 |
| POST /api/merge/export（fused=0） | _vram_guard(mode=none) | 分段链一次一帧，只有一份会话 |
| POST /api/merge/stage/{name} | _stage_vram_gate | 按阶段点名要加载的模型（pred/merge=dfm、mask=XSeg、faces=检测器、frames/encode 不加载模型）；顺序在 _stage_precheck **之后** —— 缺上游该给 409，别拿显存背锅 |
| 预览 / 项目导出 | _preview_guard / _proj_run 的 _vram_guard | 老口径不变 |

拒绝响应 = HTTP 400 + **结构化字段**（不是让前端去正则抠中文）：

    {"ok": false, "error": "快速导出：显存不够 —— 需要 1.44 GB（拆项…），可用 0.59 GB（…）。建议：①…",
     "detail": "同上", "kind": "vram", "label": "快速导出",
     "need_mb": 1469.7, "avail_mb": 608.0, "free_mb": 1400, "total_mb": null,
     "procs": 1, "frames": 8, "batch": 1,
     "suggest": ["关掉占 GPU 的程序后重试", "换更小的 .dfm / 调低帧预算", "设 DFL_ALLOW_CPU_FALLBACK=1 用 CPU 跑"],
     "vram": { …完整 plan… }}

前端只弹**一条** toast（「显存不足：需要 2.44 GB，可用 0.60 GB」+ 一句建议）、进度页收起六步
只留一行错误 + 「返回导出设置」；同一条拒绝 4 秒内不重复弹（连点不叠 toast）。

两边都支持 dry_run=1：只回计划（vram_plan + need_mb/avail_mb/free_mb/suggest），不建 job ——
前端在起 job 之前先拿它做一次预检。

**重试不需要重启后端**：拒绝发生在加载之前，_DFM_SESS / 预览 worker 一个字都没动；
用户关掉占显存的程序后回设置页再点一次即可。验收里这条用
<WORKSPACE>/_vram_free_mb.txt（一行整数 MB）复现：它比 DFL_VRAM_FREE_MB 多一个能力 ——
**同一个后端进程里改得动**（env 换档必须重启，那等于把「缓存有没有坏」这件事测没了）。

### 3.2 导出收尾：把本进程的会话放掉（不是重载预览）

一单跑完（成功/失败/取消都算）立刻 `_vram_release_inproc()`：清 `_DFM_SESS` / `_XSEG_SESS`
＋ `gc` ＋ `torch.cuda.empty_cache()`，并把它从「本程序占用」的账上划掉。

为什么必须放：一份 384 会话建好 2.3GB，**跑完 12 帧之后 ORT 的 arena 还会再涨**
（本机实测整卡 used 7049/8151 → 空闲只剩 843MB）。留着它，用户下一次预览要新建
2.4GB 会话，会被自己的后端拒掉 —— 这就是用户那句「我什么都没跑」的另一半。

**预览 worker 不重载**（收起时它本来就是空的）：下一次预览/导出各自懒加载，
实测冷加载 3.5~4.2s；这比「导出完立刻把 2GB 重新占回去」抖来抖去强。

同一条规则也用在后端进程身上：预览闸门（`_preview_guard`）在装不下且**没有任务在跑**
时，会先 `_vram_release_inproc()` 把自己进程里那些「导出留下的」会话放掉再算一遍 ——
于是「六阶段跑完再回编辑页预览」不会被自己的后端堵死。

## 4. 开关

| 环境变量 | 作用 |
|---|---|
| `DFL_VRAM_FREE_MB` | 伪造「可用显存」，三条路共用（验收/排障；`DFL_FUSED_VRAM_FREE_MB` 仍认） |
| `<WORKSPACE>/_vram_free_mb.txt` | 同上，但**同一个后端进程里改得动**（验收用：验证「关掉占显存的程序后重试不用重启后端」）。文件不存在就当没有 |
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
runtime\server\Scripts\python.exe app\tools\vram_plan_accept.py      # 三档伪造显存
runtime\server\Scripts\python.exe app\tools\vram_reclaim_accept.py   # 预览常驻 → 导出
```

`vram_reclaim_accept.py` 验的是**真实路径**（真后端 + 真 GPU，不要伪造 free）：

| 案例 | 断言 |
|---|---|
| P0 | 我们的 free 与 nvidia-smi 原始行差 ≤256MB；`source` 写明两个来源 |
| A1 | 预览一帧之后 worker 里 `models` 非空、`used_mb` ≥500MB、整卡空闲明显下降 |
| A2 | 预览常驻着按下 dry_run：**ok=True**（不再 400）、`reclaimed_mb` ≥1GB、日志有「回收预览模型 X MB … 空闲 A → B」 |
| A3 | 真导出 12 帧 phase=done、成片 12 帧、全程无 OOM |
| A4 | 导出结束后 worker 不重载；再预览能出图（懒加载）、`spawns` 仍是 1 |
| B1 | 自己可回收 + 别的进程占 2GB → 仍然放行；`other_mb` 认得出那 2GB |
| B2 | 别的进程占 4.6GB → 400，消息里有「本程序 / 其它程序」两栏，结构化字段 `own_mb/other_mb` 有数 |

三档伪造显存（416 dfm）：1200 → 拒绝（需要 1.44GB / 可用 0.44GB）、3000 → 降档
（30 帧 / 批 8→2）、8000 → 正常（30 帧 / 批 8）；`DFL_ALLOW_CPU_FALLBACK=1 + 1200`
能出片且日志/状态写明 CPU；整轮后端日志断言「无 CUDA OOM」。
