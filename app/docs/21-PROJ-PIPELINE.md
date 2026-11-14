# 21 · 项目导出改成流水线（不检测 · 一份会话 · dfm 攒批 · ROI 合成多线程）

> 这一页回答一个问题：**项目导出比视频导出慢一半（10.7 vs 22.3 帧/秒），凭什么？**
> 三个结构性问题的核实结果、改了什么、改完多少帧/秒、像素差多少，全在下面。
>
> 机器：RTX 5070 Laptop **8GB** · R9 8940HX（16 线程）· 32GB · Windows
> 素材：**同一份 120 帧真项目**（000001~000120.png · 1080×1920 · 90 张对齐脸，
> 即 75% 的帧有脸）· 参数 seamless + mt + XSegLite（mask_mode=8 / erode 0 / blur 4）
> · quality 92 · `force=true`（每档都真做）。

---

## 一、三个问题的核实（都是真的，数字在右边）

### ① 重复检测 —— **属实**

老路（`_proj_run._prep` → `_preview_prepare` → `_swap_desc`）对每一帧做：

    读帧 → _detect_boxes(检测器) → _pick_face → _frame_landmarks(5点粗对齐 + 2D106 回归 68 点)

而 `aligned/*.jpg` 是 **② 切脸**自己写出来的 DFLJPG，APP15 里就带着
`source_landmarks`（原帧 68 点）/ `image_to_face_mat` / `source_rect` / `face_index`
/ `side`。也就是说：**同一批 68 点被算了两遍**。

实测（本机，tinymog，1080p）：

| 项 | 数字 |
|---|---|
| 检测（`_detect_boxes`） | **61.0 ms/帧**（含首次 CUDA Graph 捕获摊销） |
| 2D106 回归 68 点（`_frame_landmarks`） | **6.1 ms/帧** |
| 合计白跑的 | **67.1 ms/帧** |
| DFLJPG 里 68 点的自洽性 | `max‖mat⁻¹·face_lm − source_lm‖ = 1.5e-4 px` |

> 注：docs/17 里写的「检测 21ms/帧」是单帧稳定态；这里是整段平均（含冷启动），
> 但量级结论不变 —— 每帧白扔 20~70ms。

### ② 合成单线程 —— **属实**

老路的合成发生在 worker 子进程里，由 `MergeStudio.core.merger.MergeMaskedFace`
一帧一次跑完：整帧 float32 归一化 → 全帧 hull 遮罩 → warp → dfm → 遮罩组合 →
erode/blur → **整帧** warp 回来 → `cv2.seamlessClone` → 再按遮罩混合。

同一帧（1080×1920、416 dfm、mask_mode 8 + seamless + mt）单线程实测：

| 做法 | 单帧中位 |
|---|---|
| `MergeMaskedFace`（老路） | **340 ms** |
| `_merge_compose.compose_frame`（ROI 版，视频 fused 那条路用的） | **141 ms** |

再叠上「一帧一请求」的串行，就是 17-PROJ-BATCH 里那个 330ms/帧的墙。

### ③ 没有 dfm 批、没有深队列 —— **属实**

worker 的 `_swap_item` 是「一帧 → `MergeMaskedFace` → 里面一次 dfm 前向（batch=1）」。
视频 fused 那条路是「攒批 8 张 → `_dfm_run_batch` 一次前向」。416 的阶梯实测：
B1 = 29.1 ms/张 · B4 = 28.7 · B8 = 31.0 —— 攒批 4 张每张省 ~0.4ms，而且**把 GPU 的
空转填掉了**（batch=1 时 GPU 等 CPU 组帧，CPU 等 GPU 出 pred）。

---

## 二、改成了什么（`app/server/_proj_pipe.py`）

四段 + 一条兜底，段间都是有界队列（depth 24），乱序完成后**按帧序提交**：

    ① 读帧/对齐 (4 线程)      cv2 读帧 → 取 DFLJPG 的 68 点 → warp 到 S → dfm 输入
                              （**不检测、不回归**；只有连 landmarks 都没有的对齐脸
                                才回退检测，并在日志里计数）
         ↓ q_gpu
    ② dfm 攒批 + XSeg (1 线程) 攒 4 张 → _dfm_run_batch → pred/prd/dst
                              → XSeg mask_batch(pred, dst)（只在这条线程碰 GPU）
         ↓ q_cmp
    ③ 合成 (4 线程)           _merge_compose.compose_frame（ROI 版，纯 CPU）
         ↓ q_enc
    ④ 编码 (6 线程)           .png 帧编 PNG(level3) / .jpg 帧编 q92 JPEG
         ↓
    主线程                     按帧序 .part → os.replace（断点续跑语义不变）

关键点：

* **一份 dfm 会话**（384 实测 2.3GB）。老路是「N 个 worker 进程各一份」，8GB 卡上那等于
  把换页余量交出去（20-PROJ-ADAPTIVE 的 D 档：2×2 = 3.49 帧/秒、整卡 95.2%）。
* **队列深**：q_gpu / q_cmp / q_enc 各 24 帧，谁忙都不空转；`swap_ms` 只算
  「第一张脸进 GPU → 合成完」这一段，排队等位不算（否则 31 帧/秒时能积到 1~2 秒，
  看起来像单帧变慢）。
* **取消/断点续跑**：取消后 merged/ 里留下的一定是**前缀**（提交严格按帧序）；
  续跑按「文件存在且非空」跳过 —— 和老路一个判据。
* 开关：`DFL_PROJ_PIPELINE=legacy` 回到老路（worker 子进程 + MergeMaskedFace），
  A/B 与排障用；`DFL_PROJ_READ / _ENC / _DFM_BATCH / _PNG_LEVEL` 调线程与攒批。
* **画布**：`DFL_PROJ_COMPOSE=aligned|facelib`（默认 aligned，见 [23](23-ALIGNED-COMPOSE.md)）；
  合成线程数改成 `DFL_PROJ_COMPOSE_THREADS`（默认 4）；老写法 `DFL_PROJ_COMPOSE=8`
  仍按线程数解释（全数字不当模式用）。
* 收尾（对账 + ffmpeg 编码）抽成 `_proj_finish()`，两条路共用一份实现。

---

## 三、两条「能不能更省」的质疑，核实结果

### 3.1 「贴回只依赖 image_to_face_mat，不需要 68 点」—— **成立，但要整条链都在 aligned 画布上**（见 [23-ALIGNED-COMPOSE](23-ALIGNED-COMPOSE.md)）

> **2026-11 更正**：这一节原来写的是「不成立」，那个结论是**混用两张画布**得出的 ——
> 它拿 facelib 画布的矩阵 A 去贴 aligned 画布的像素（或反之），当然错几十像素。
> 正确做法是**整条链只用一张画布**：dfm 输入、遮罩、ct、反贴全在 aligned 画布上走，
> 67 行就变成：`warpAffine(pred, image_to_face_mat, (W,H), WARP_INVERSE_MAP)`。
> 已经实现成 `_merge_compose.compose_aligned_frame()`，项目模式默认走它
> （`DFL_PROJ_COMPOSE=facelib` 可回退）。实测：同一张画布内闭环误差 **0.078 px**，
> 而混用两张画布是 **13 px**（画布四角 19~38 px）—— 全部数字在 23 页。

原来的核实数据仍然有效，但只说明一件事：**两张矩阵不是同一张**。

`compose_frame` 贴回用的矩阵是 **facelib 的 WHOLE_FACE**：`get_transform_mat(lm68, S,
WHOLE_FACE)`。而 `aligned/` 的画布是 app 自己的 68 点模板（`_align.align_mat_from_68`，
padding 0.40 + 中心下移 7%）。两张矩阵**不是同一张** —— 这正是 `_stage_dst_face_mat`
存在的原因（`routes_merge.py:7977` 的注释：68 点残差均值 6.1~7.2px、缩放差 5.5%）。

量了「能不能用一张常数补偿矩阵把 image_to_face_mat 抬到 facelib 画布」：

| 量 | 结果 |
|---|---|
| 逐脸 C = A·B⁻¹（aligned→facelib 画布）把四角+中心映射到画布的坐标 | 逐点**标准差 2.8~21.2 px**、极差最大 **74 px** |
| 用第一张的 C₀ 当常数代替逐脸 C，画布四角+中心映回原帧的误差 | **mean 73.1 px / p95 173.6 / max 249.6 px** |
| 原图同 68 点上两个模板的缩放比（C 的线性部分，40 张真实对齐脸） | **1.13 ~ 1.15**（facelib 画布的脸大 13%） |

结论（更正后）：**在同一张画布内**，image_to_face_mat 的逆就是贴回矩阵，68 点一个都
不需要；**跨画布**则没有任何一张矩阵能救（补偿矩阵 C 是逐脸的，不是常数）——所以不是
「补一张矩阵」，而是「别换画布」。

> 产品取舍仍在，只是换了位置：aligned 画布比 facelib 画布**宽 13%**（模板形状差异，
> 不是代码 bug），所以 dfm 看到的构图会和「按 facelib 裁法训的模型」差 13% 缩放。
> 谁的画布和模型训练时用的一致，那个模式就占优 —— 这条只能对着训练数据确认，
> 23 页把两条路的像素差与帧率给全了，不替产品做选择。

### 3.2 「色彩迁移必须在 aligned 小图上算，不能在整帧上算」—— **本来就是（视频/项目都是 S×S 域）**

`compose_frame` 里 ct 的位置（`_merge_compose.py:487-494`）：

    dw = cv2.warpAffine(roi_f, fwd, (sz, sz), INTER_CUBIC)   # 原帧 ROI → 脸画布 S×S
    fo = _color_transfer(fo, dw, wrk, ct, timings)          # 两张都是 S×S

统计量（均值/方差/直方）只在 **S×S 的 pred** 与 **S×S 的 dw** 之间取，而且
`_match_tone_roi` / `_color_transfer` 又把范围压到**遮罩包围盒**里算（数学等价，
末位噪声级）。整帧只在两处被碰：反贴（`warpAffine` 回原帧，9.9ms）和融合
（`seamlessClone`，只在遮罩 bbox 里，37.5ms）。

所以「ct 在整帧上做」这个假设在本仓库不成立 —— 它一直是 aligned 域。

---

## 四、帧速（A/B，同机同素材同参数，force=true）

| 模型 | 老路（worker + MergeMaskedFace） | 流水线 | 倍数 | 目标 | 达标 |
|---|---|---|---|---|---|
| **416** 阿証…-416（0.51GB · 134.8M 参数 · 输入 416） | **12.69**（2×2，它的默认档） | **31.3**（31.34 / 31.29 两次） | **2.46×** | ≥ 20 | ✅ |
| **384** 张元英（1.53GB · 410.8M 参数 · 输入 384） | **10.35**（1×4，它的默认档） | **22.2 / 24.1** | **2.1~2.3×** | ≥ 18 | ✅ |

> 384 那份没有 TRT 引擎（日志：没有 dfm 的 TRT 引擎 → ort-cuda），dfm 44.4ms/张；
> 有引擎还能再快一档。老路同一份也是 ort-cuda，A/B 公平。

### 4.1 每帧各段（ms，按 90 张有脸帧摊；括号里是线程数）

| 段 | 416 | 384 |
|---|---|---|
| 读帧 + warp（4 线程） | 38.0 | 36.4 |
| dfm 前向（攒批 4，GPU 1 线程） | **30.6** | **44.4** |
| XSeg 两张遮罩（GPU 同一条线程） | 6.0 | 5.7 |
| 合成合计（4 线程） | 85.6 | 83.3 |
| ├ 遮罩准备（erode/blur/组合） | 3.2 | 2.4 |
| ├ **色彩迁移（S×S 域）** | **27.5** | 28.2 |
| ├ 反贴（一次 warp 回原帧） | 9.9 | 10.3 |
| └ 融合（seamlessClone，遮罩 bbox 内） | 37.5 | 35.0 |
| 编码（6 线程，PNG level3） | 154.3 | 147.9 |
| 落盘（原子替换） | 6.1 | 5.3 |
| **墙钟/有脸帧** | **42 ms**（= 3.8s / 90） | **55 ms**（= 5.0s / 90） |

读法：每帧的**串行**部分只有 dfm + xseg（416：36.6ms / 384：50.1ms）—— 那是这块卡的
天花板；其余各段都被线程摊掉了（合成 85.6/4 = 21ms、编码 154.3/6 = 26ms）。
编码仍是第二大项：`DFL_PROJ_PNG_LEVEL=1` 可再抬到 **31.9**（416），但那会改 .png 的
字节（像素不变），默认保持 level 3 与老路同档。

---

## 五、像素：与改造前项目导出的差

同一份项目、同一份模型、force=true 各跑一遍，`merged/` 逐像素对拍：

| 模型 | mean | max | 差异像素 | 逐字节相同 |
|---|---|---|---|---|
| 416 | **0.1026 LSB** | 61 | 1.24% | 30/120（那 30 帧是无脸帧，原样搬） |
| 384 | **0.0734 LSB** | 115 | 1.23% | 30/120 |

门槛 0.5 LSB，**都过**。拆开看（单帧对照实验，416）：

| 变的东西 | mean |
|---|---|
| 只换合成器（MergeMaskedFace → compose_frame，**同一套 68 点**） | 0.1924 LSB |
| 只换 68 点来源（重检测的 68 点 → DFLJPG 里的 68 点） | 0.0211 LSB |
| 两者一起（= 老路 vs 流水线） | 0.1994 LSB |

也就是说：像素差几乎全来自**换合成器**（整帧版 vs ROI 版在 seamless 边缘的取舍），
「不再检测」只贡献 0.02 LSB。两条 68 点来源本身差 mean 2.8 px / max 6.8 px
（2D106 的输入构图不同：② 用 192 粗裁，老路用 `align_face` 512），但落到合成上很小。

---

## 六、验收

### 6.1 线上（HTTP 接口）跑的回归

| 验收 | 结果 |
|---|---|
| `proj_pool_accept.py --cases a,b,c,d`（120 帧真项目） | **16/16 全过**（协议形状、逐像素、取消+续跑） |
| `proj_default_accept.py --cases a,b,c,d,e`（60 帧 · 100% 有脸夹具） | **28/28 全过** |
| `stage6_accept.py`（视频六阶段，含 ④⑤ 的 compose 对拍） | **64/64 全过**（这条路的代码本次一行没改，只被项目模式复用） |

proj_pool 里几条硬的：

    A status 回显一份会话 workers=1 / dfm 攒批 ≥4          workers=1 batch=4 compose=4
    A 默认档日志写着「一份 dfm 会话 · 攒批 N 张 · 合成 N 线程」
    A 走的是流水线（status.pipeline=fused）                 pipeline='fused'
    C 逐像素 1×4 vs 1×8                                    mean 0.0000 LSB · 逐字节相同 120/120
    D 取消落地                                             phase=cancelled done=55 files=55
    D 取消后留下的是前缀                                   000001…000055（55 帧，没有洞）
    D 续跑补齐到满                                         120 帧（跳过 65）
    D 续跑产物 vs 一次跑完                                  mean 0.000000 LSB · 逐字节相同 120/120

proj_default 里几条硬的：

| 量 | 大模型 384 | 小模型 416 |
|---|---|---|
| 帧/秒（100% 有脸夹具） | 16.8 | 22.4 |
| 每张有脸帧 | 60.0 ms | 45.0 ms |
| swap p50（换脸+遮罩+合成） | 78.8 ms | — |
| 本程序显存峰值 | 3308 MB = 40.6%（整卡 60.7%） | 0 MB（上一单的基线已经含它） |
| 逐像素（A 默认档 vs C 降级档） | mean 0.009 LSB | |

### 6.2 本次给的三条验收

| 项 | 结果 |
|---|---|
| 帧速 384 ≥ 18 | **22.2 / 24.1**（75% 有脸的真项目）· 16.8（100% 有脸夹具）✅ |
| 帧速 416 ≥ 20 | **31.3**（75% 有脸）· 22.4（100% 有脸夹具）✅ |
| 逐像素 vs 改造前 ≤ 0.5 LSB | **0.1026 / 0.0734** ✅ |
| 断点续跑语义不变 | ✅（判据、前缀、续跑产物逐字节一致） |

> 两个数别混：**75% 有脸**那份是「用户真项目」（30/120 帧无脸，原样搬，便宜）；
> **100% 有脸**那份是 proj_default 的夹具（每帧都要真换脸）。同一台机器上
> 前者 22~31 帧/秒、后者 16.8~22.4 帧/秒，差的就是那 25% 的搬运帧。

---

## 七、还没做的（下一轮）

1. **dfm 的 TRT 引擎**：384 那份只有 ort-cuda（44.4ms/张）。建 bf16 引擎（`_trt_build.py`）
   后每张应该能压到 ~31ms，384 的墙钟会从 55ms 掉到 ~42ms（≈ 28 帧/秒）。
2. **PNG 编码**：`level=3` 是 154ms/帧，level=1 是 ~70ms。像素不变、字节会变 ——
   要不要换是产品决定（老路是 level 3）。
3. **多模型**：流水线支持（每张脸按 `_model_for_face` 分派到各自的 dfm，各自攒批），
   但一次导出里同时压 2~3 份会话会顶显存，需要显存预算那一侧再给一档。
4. **自适应标定**（20-PROJ-ADAPTIVE 的 bench 表）：流水线的档位只有「攒批」一维，
   标定那套 workers×batch 的扫描已经没意义，`_projplan` 的缓存表在这一版里不再参与。
