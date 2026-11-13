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
预留 reserve  = min(768MB, max(256MB, free × 5%))      ← 2026-05 起（原来是 free×20%+512MB）
可用预算 avail = free - reserve + 可回收
                 ↑ 256MB 底线 = 探测抖动（见 §2.2）；5% 让大空闲自动放宽；768MB 封顶
                 ↑ 可回收 = 我们自己的预览 worker 里那份常驻会话（导出前会真回收掉）
一份会话 sess  = model_mb(权重，≈ .dfm 文件体积) + 512MB(CUDA context)
固定开销 fixed = sess × 进程数 + 300MB(检测器 + 2D106 + XSeg)
帧缓冲         = frames × size²×3×4×4/1MB        （416 → 7.92MB/帧）
dfm 批         = f(batch)                         （查表，见下）
总需求 need    = fixed + 帧缓冲 + dfm 批          （resident：会话那项不计，见 §2.3）
可行 ⟺ need ≤ avail
```

### 2.1 free 为什么是两个来源取小

Windows/WDDM 上 `torch.cuda.mem_get_info` 报的空闲**比 nvidia-smi 高**：本机空载
实测 6999 vs 5693 MiB。差的 1.3GB 不是谁算错 —— mem_get_info 报的是「驱动还愿意
答应多少」（不够就换页/回落共享内存），真分配下去立刻变成 nvidia-smi 的 used。
拿它当预算是**乐观到 OOM**；nvidia-smi 的 free 才是「这张卡上所有进程之外还剩
多少」。所以两边都探、取小，`source` 写成 `min(torch 6999 / smi 5693)`，计划里另带
`free_torch_mb` / `free_smi_mb` 两个键，排障时一眼看见差多少。

### 2.2 预留为什么从「20% + 512MB」改成「max(256MB, 空闲×5%)」

老口径在 8GB 卡、空闲 3.4GB 时预留 = 3.4×0.2 + 0.512 = **1.19GB**，再叠上「一份会话
+ 帧缓冲」就变成用户截图里那句「需要 2.46 GB / 可用 2.03 GB」—— 而那 1.19GB 里
**没有一个字节是实测需要的**：

| 会被预留「吸收」的东西 | 实测 | 已经在 need 里？ |
|---|---|---|
| CUDA context | 每份会话 512MB | 是：`_VRAM_CTX_MB` 明着算 |
| ORT CUDA arena（会话刚建好） | 416 +135MB / 384 +221MB | 由预留兜（很小） |
| arena 换池子那一步（batch 4 起） | +1036MB（416） | 是：已包含在 `_VRAM_BATCH_PTS` 的实测台阶里（那张表量的是「整卡增量 − 会话本身」） |
| 探测滞后 | 我们与 nvidia-smi 原始行差 ≤256MB；探测有 2~4s 缓存 | 否，真需要 |
| 别的程序插队 | 浏览器开个硬件加速几百 MB | 否，真需要 |

所以 20% 是**拍的**（出在 ceaf4f4 第一版，提交信息只有「avail=free×0.8-512MB」，
没有任何测量支撑），而且和 `_VRAM_BATCH_PTS` 对 arena 那一步**重复计账**。
新口径只覆盖表里最后两行：底线 256MB ≈ P0 校准出来的探测误差（我们 vs nvidia-smi
差 ≤256MB），5% 让大空闲时自动放宽，768MB 封顶（24GB 卡上别再长出 1.2GB 死重）。

实测（`app/tools/vram_force_accept.py`，本机 RTX 5070 Laptop 8GB / 384 dfm）：

| 场景 | 预留 | 可用 | 结果 |
|---|---|---|---|
| 空闲 3.24GB（新口径） | 256MB | 2.91GB | 放行 → 30 帧 / 批 2 → 12 帧 **6.98 帧/秒**，整卡峰值 **7527 / 8151MB**，0 OOM |
| 空闲 3.04GB（老口径，同一刻对照） | 1134MB | 1.93GB | **拒绝**（差 0.53GB）—— 用户看到的就是这一档 |
| 空闲 0.63GB | 256MB | 0.38GB | 仍拒绝；`DFL_VRAM_RESERVE_MODE=legacy20` 可复原老口径做对照/回滚 |

### 2.3 三种「容易把可跑的活判死」的口径

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

    {"ok": false, "error": "快速导出：显存不够 —— 需要 1.44 GB · 空闲 3.04 GB 里预留 1.11 GB …",
     "detail": "同上", "kind": "vram", "label": "快速导出",
     "need_mb": 2522.3, "avail_mb": 1976.0, "free_mb": 3110, "total_mb": 8150,
     "reserve_mb": 1134.0, "reserve_min_mb": 256, "reserve_pct_mb": 155.5, "safety": 0.2,
     "own_mb": 8.0, "own_inproc_mb": 8.0, "own_preview_mb": 0.0, "other_mb": 5040.0,
     "summary": "需要 2.46 GB · 空闲 3.04 GB 里预留 1.11 GB 给驱动/碎片（规则 …）"
                " = 实际可用 1.93 GB · 本程序 0.01 GB / 其它程序 4.92 GB / 空闲 3.04 GB",
     "can_force": true, "forced": false,
     "procs": 1, "frames": 8, "batch": 1,
     "suggest": ["关掉占 GPU 的程序后重试", "点「仍然尝试（可能爆显存）」强制放行一次", …],
     "vram": { …完整 plan… }}

' + BT + 'summary' + BT + ' 是**一行话**（需要 / 空闲 − 预留 = 可用 / 本程序·其它程序·空闲），
前端 toast 和被拒面板都直接显示它 —— 老版本只给 need_mb/avail_mb，界面上根本看不到
「预留」和「谁占的」，用户当然算不明白。

前端只弹**一条** toast（「显存不足：需要 2.44 GB，可用 0.60 GB」+ 一句建议）、进度页收起六步
只留一行错误 + 「返回导出设置」；同一条拒绝 4 秒内不重复弹（连点不叠 toast）。

两边都支持 dry_run=1：只回计划（vram_plan + need_mb/avail_mb/free_mb/suggest），不建 job ——
前端在起 job 之前先拿它做一次预检。

**重试不需要重启后端**：拒绝发生在加载之前，_DFM_SESS / 预览 worker 一个字都没动；
用户关掉占显存的程序后回设置页再点一次即可。验收里这条用
<WORKSPACE>/_vram_free_mb.txt（一行整数 MB）复现：它比 DFL_VRAM_FREE_MB 多一个能力 ——
**同一个后端进程里改得动**（env 换档必须重启，那等于把「缓存有没有坏」这件事测没了）。

### 3.3 「仍然尝试（可能 OOM）」：用户能推翻闸门，但推翻不了物理

拒绝面板多一颗次要按钮（`mgpForceTry`）。点了之后前端把 `force=1` 写进 body，
`_vram_guard(..., force=True)` 放行**最小档**（帧 8 / 批 1），plan 上留 `forced=true`
与「差多少 MB」，然后正常加载、正常跑：

- 跑成了 → 正常出片（status 里 `vram_plan.forced=true`，日志有一行「⚠ 强制尝试」）；
- 真跑不动 → worker 抓异常 → `_vram_forced_rollback()`：
  ① 放掉本进程的 dfm/XSeg 会话 + `gc` + `torch.cuda.empty_cache()`；
  ② **重新探一次空闲**，按当下的数重算一份拒绝计划（不拿 OOM 之前那份旧账）；
  ③ 把 `{kind:"vram_oom", summary, need_mb, avail_mb, free_mb, reserve_mb, own_mb, other_mb}`
     挂到 `/export/status` 与 `/stage/status` 的 `vram_oom` 键上，错误文本写明**已回滚**；
  ④ 别人的东西一概不动（预览 worker 不重载、进程不杀），所以可以再点一次。

前端拿到 `vram_oom` 就把界面从「进度页」搬回被拒态：标题改成「确实不够（已回滚）」，
正文是 `vram_oom.summary`（当下的数字），按钮还是那两颗 —— **不是**一句「导出失败」。

判据（`_vram_forced_rollback`）：报错文本像显存问题，**或者**回滚之后重探仍然装不下。
只看文本会漏判（ORT 有时只给一句 CUDA failure）；反过来「重探之后其实装得下」就不改写成
「显存不够」—— 那是真 bug，该老实报错。

**为什么验收要注入 OOM**：WDDM 会把别的进程的显存**换页**出去，本机把空闲压到 600MB
再 force=1 竟然还是跑完了 12 帧（慢，但不失败）。所以「强制尝试撞 OOM」这条路只能用
`DFL_ALLOW_TEST_HOOK=1 + DFL_VRAM_OOM_INJECT=1` 在 `_dfm_session` 里注入一次
（生产环境两个开关都没人设，一个字节都不走）。实测注入口径下：OOM 被捕获 →
`phase=failed` + `vram_oom` 有数 → 临时帧目录 0 新增、无 `.part`、`own_inproc` 42MB
（会话已放掉）→ 再 force 一次仍被受理 → 放开压力后预览照常出图。

### 3.4 显存归属：那一行必须能看懂「为什么被拒」

   需要 2.46 GB · 空闲 3.04 GB 里预留 1.11 GB 给驱动/碎片（规则 …） = 实际可用 1.93 GB
    · 本程序 0.01 GB / 其它程序 4.92 GB / 空闲 3.04 GB

「本程序」= 本进程实测增量（`_vram_own_*`）+ 预览 worker 实测（`_used_mb`）；
「其它程序」= 总显存 − 空闲 − 本程序（只在真探到时才算，伪造 free 时给 `?`）。
三个数都进 `summary`，也都在 status / 日志里。

另外加了一条**兜底回收**：`_vram_reclaim()` 卸掉预览模型之后如果只腾出不到一半，
就把 worker **进程**收掉（下次预览懒加载：boot 1.6s + 冷加载 3.5~4.2s）。为什么需要：
WDDM 上「释放」不等于「还给驱动」—— 本机实测预览 worker 卸掉一份 384 模型之后，进程
仍握着 **2347MB** 专用显存（Windows「GPU Process Memory · Dedicated」，实测），而
nvidia-smi 的 free 只涨了 135MB。这种残留正是「我什么都没跑却被堵死」的另一半。
开关：`DFL_VRAM_RECYCLE_WORKER=0` 可关掉（默认开）。

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
| `DFL_VRAM_RESERVE_MODE=legacy20` | 复原老预留口径（free×20%+512MB），只给对照/回滚用 |
| `DFL_VRAM_RECYCLE_WORKER=0` | 关掉「卸模型没腾出显存就把 worker 进程收掉」那条兜底（默认开） |
| `DFL_ALLOW_TEST_HOOK=1` + `DFL_VRAM_OOM_INJECT=1` | 验收专用：在 `_dfm_session` 里注入一次 OOM（生产不设就一个字节都不走） |
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
runtime\server\Scripts\python.exe app\tools\vram_plan_accept.py      # 三档伪造显存（51 项）
runtime\server\Scripts\python.exe app\tools\vram_reclaim_accept.py   # 预览常驻 → 导出（26 项）
runtime\server\Scripts\python.exe app\tools\vram_force_accept.py     # 预留 / 归属 / 强制尝试（45 项）
```

`vram_force_accept.py` 四段：W 归属（本程序/其它程序/空闲 + 按进程实测）、
A 空闲 3.4GB（新口径放行 vs 老口径拒绝，并真跑 12 帧报帧率/峰值）、
B 空闲 600MB（默认拒绝 → force=1 → 注入 OOM → 回滚 → 再试 → 压力释放后预览仍能出图）、
F 前端接线（按钮/force=1/字典/样式，静态断言）。

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
