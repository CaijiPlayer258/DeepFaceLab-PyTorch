# 23 · 项目模式改成 aligned 原生合成（只有一张画布）

> 这一页回答一个问题：**为什么项目导出要「用 68 点现算 facelib 矩阵」才能贴回去？**
> 答案：不是因为 image_to_face_mat 不够用，而是因为整条链上**同时存在两张画布**。
> 把链子收到一张画布上以后，贴回就是 `warpAffine(pred, image_to_face_mat, (W,H),
> WARP_INVERSE_MAP)` 一句话，68 点一个都不需要。
>
> 机器：RTX 5070 Laptop **8GB** · R9 8940HX（16 线程）· 32GB · Windows
> 素材：`proj_pool_accept` 建的**同一份 120 帧真项目**（1080×1920 · 90 张对齐脸 = 75%）
> 参数 seamless + mt + XSegLite（mask_mode=8 / erode 0 / blur 4）· quality 92 · force=true

---

## 一、速览

| 项 | 结果 |
|---|---|
| 画布 | 整条链（dfm 输入 / 遮罩 / ct / 反贴）只用 **aligned 画布** = DFLJPG 那张对齐脸 |
| 贴回矩阵 | `image_to_face_mat` 本身（`WARP_INVERSE_MAP`），**68 点不参与合成** |
| 同一画布内闭环误差 | **0.078 px**（画布坐标 → 原帧 → 画布） |
| 混用两张画布的误差 | **13.06 px**（画布四角 19~38 px）—— 这就是原来「几十像素」的来源 |
| 与 facelib 路逐像素差 | mean **0.0196 / 0.0268 LSB** · max 103 / 131 · 差异像素 **0.40%**（416/384） |
| 差异是不是整体错位 | 不是：相位相关位移 **0.01 px**，>16 LSB 的差异 **97~99% 落在原帧强边缘上** |
| 帧速 416（120 帧项目 · 3 次中位） | facelib **33.68** → aligned **34.22** 帧/秒（+1.6%）；**dfm 换回 ONNX（现行默认）后 25.32 → 25.19**，见 §六 |
| 帧速 384 | facelib **26.87** → aligned **26.92** 帧/秒（+0.2%，噪声内；另一轮 +2.5%） |
| 开关 | `DFL_PROJ_COMPOSE=aligned`（默认）/ `facelib`（老行为，A/B 用） |
| 回归 | proj_pool 16/16 · proj_default 28/28 · stage6 64/64 · fused 64/64 |

---

## 二、为什么会有两张画布（历史原因）

### 2.1 facelib 的画布 = 33 点 + padding 0.40 + 中心下移 7%

`facelib.LandmarksProcessor.get_transform_mat(lm68, S, WHOLE_FACE)`：拿 68 点里的
**33 个**（17..48 + 54）去拟合 DFL 自己的模板 `landmarks_2D_new`，得到「原图 → 单位方形」
的相似变换；再按 `padding=0.40`、WHOLE_FACE 的 `0.07` 下移，把单位方形映射到 S×S。
4.0 的切脸（`Extractor`）和贴回（`MergeMaskedFace`）都用这一张 —— 自洽。

### 2.2 4.1 的切脸自己造了一张（`app/server/_align.py`）——**已在 2026-12 修掉**

> 这一节记的是**历史**（2026-12 之前）：当时的 ② 切脸是一张和 facelib 不同的画布，
> 所以第三节以后的「两张画布」才成立。现在的 `align_mat_from_68` 只拟合 33 点、
> 与 `get_transform_mat` 逐位等价（见 2.2.1 与 [24-ALIGN-WHOLEFACE](24-ALIGN-WHOLEFACE.md)），
> 两张画布**已经是同一张**；本节保留是为了解释清楚「当初为什么会错」。

4.1 的 ② 切脸走 `_align.align_mat_from_68`：用**全部 68 点**去拟合一个经验模板
`TEMPLATE_68`（4.0 一万多张对齐脸的均值），其中 33 个 DFL 点被强制等于
`landmarks_2D_new`（`_build_unit_68`），几何（padding / 0.07 / 三角点求仿射）逐行照抄
`get_transform_mat`。**在一张「完美贴合模板」的假脸上，两者逐项相同**（实测 1e-13）。

但真实人脸不是模板：多出来的 35 个点（下颌/下巴，单位空间里 x∈[-0.12, 1.05]、
y∈[0.02, 1.16]，**跑到单位方形外面**）在 68 点最小二乘里权重不小，把相似变换的
缩放和旋转一起拉偏。

### 2.2.1 更正：13% 是分布的尾部，中位只有 **+0.4%**（1203 张实测）

这一页原来写的是「缩放 1.13~1.15、旋转 0.7°~1.2°、四角 mean 19.25 px，这是两张模板
形状不同」。**形状不同这句对，「13%」这句错** —— 那是分布的尾部（p95），不是典型值。
拿 `workspace/data_src/aligned` 的 **1203 组真实 source_landmarks** 逐张量
`C = facelib⁻¹ · image_to_face_mat` 的线性部分（`align_wf_accept.py --cases a`）：

| 量 | 中位 | p05 / p95 | min / max |
|---|---|---|---|
| 缩放（facelib / aligned） | **+0.36%** | −6.3% / **+13.3%** | −27.7% / +22.5% |
| 旋转 | **−2.53°** | −4.88° / +4.75° | −8.3° / +13.4° |
| 画布（四角+中心）位移 | **33.7 px** | 11.4 / 66.2 px | 5.2 / 149.6 px |

读法：**中位只有 +0.4%，主因是 3° 的旋转加上与姿态相关的缩放；根因是拟合点数**
（68 点 vs 33 点）。13% 那个数是 p95 —— 只有侧脸/极端姿态才到得了，拿它当「常数」
去补，正是老代码在 ③⑤ 里搬数据要付的代价。

修法（2026-12）：拟合点从 68 降到 **33**（17..48 + 54，与 facelib 同一组），模板直接
从 `facelib.LandmarksProcessor.landmarks_2D_new` 取（不再手抄）；多出来的 35 个点只做
**外点检查**。改完实测 zoom 1.00000 / rot 0.0000° / 四角 0.000 px，两张画布合成一张。

### 2.3 于是同一条链上并存两张画布（历史；2026-12 之后已合并成一张）

`21-PROJ-PIPELINE.md` 那一版把项目导出改成流水线时，合成器直接复用了视频 fused 那条
路的 `_merge_compose.compose_frame` —— 它是给 **facelib 画布**写的（内部
`get_transform_mat(lm68, S, WHOLE_FACE)`）。于是出现了这样的分工：

    ② 切脸        aligned/ 里的图      = B 画布（68 点模板）
    ③⑤ 换脸/合成  dfm 输入 + 贴回矩阵  = A 画布（facelib WHOLE_FACE）

两张画布之间要过数据（dst 遮罩、`_stage_dst_face_mat` 的几何对齐），就得搬一张补偿
矩阵 `C = A·B⁻¹`。而 **C 是逐脸的**（表 2.2：标准差 2.8~21.2 px、极差最大 74 px），
拿第一张当常数去补，画布四角错 73/174/250 px（mean/p95/max）—— 这就是「几十像素错位」
的来源，也是上一版得出「贴回必须用 68 点现算矩阵」的原因。

**用户的判断是对的**：`image_to_face_mat` 本来就是「从原帧裁出这张脸」的矩阵，
它的逆就是贴回矩阵。之所以「只用它会错」，是因为 pred 是在**另一张**画布上算出来的。

实测三条（同一张对齐脸，462×462 画布，与真帧 1080×1920）：

| 做法 | 与原帧的差 |
|---|---|
| `warpAffine(frame, image_to_face_mat, (S,S))` vs 落盘的 aligned 图 | **mean 0.34 LSB / max 4** |
| `warpAffine(frame, facelib WHOLE_FACE, (S,S))` vs 同一张 aligned 图 | mean 17.9~20.1 LSB / max 220 / **47% 像素 >8 LSB** |
| 反贴：`warpAffine(aligned, mat, (W,H), WARP_INVERSE_MAP)` vs 原帧脸区 | mean 0.52 LSB（JPEG 边缘 max 109） |
| 反贴写错成 `inv(mat)` + `WARP_INVERSE_MAP`（两次求逆） | mean 82.4 LSB（脸贴到别处） |

最后一行是这一版**最容易踩的坑**：`WARP_INVERSE_MAP` 的语义是「M 是 dst→src」，而
`image_to_face_mat` 正是 原帧→画布（贴回时的 dst→src），所以**不能**再套一层
`invertAffineTransform`；两次求逆等于恒等，脸会贴到原点附近。

---

## 三、新实现：`compose_aligned_frame()`

`app/server/_merge_compose.py`，和 `compose_frame()` 并列的第二个入口（视频模式仍走
原来的 `compose_frame`，一行没动）。

    def compose_aligned_frame(frame_bgr, faces, cfg, timings=None, roi=True):
        faces = [{"mat": 原帧→画布 2×3（已把画布尺寸折进 pred 的 S）,
                  "pred": (S,S,3) 0..1,
                  "dw": (S,S,3) 可选 —— 画布上的原帧脸（ct 参考）,
                  "landmarks": 可选（只有 mask_mode==1 的 dst hull 用）,
                  "prd_dfm"/"dst_dfm"/"xseg_prd"/"xseg_dst": (S,S) 或 (256,256)}]

流程全在 S×S 上：

    ① 遮罩   _prepare_mask(f, frame_shape, cfg, S, fmat=mat)   ← 复用同一套（含 erode/blur）
    ② ct     dw = f["dw"]（就是画布上的那张脸，不再从原帧 warp 过来）
             _color_transfer(pred, dw, wrk, ct) —— 和原来同一份实现，范围仍是遮罩包围盒
    ③ 反贴   warpAffine(wrk / pred, fwd, (rw,rh), WARP_INVERSE_MAP|CUBIC)
             fwd = mat · T(ROI 原点)  ← 和 compose_frame 同一套 ROI 折算法
    ④ 融合   _blend_layer()（overlay / hist-match / seamless 都在 ROI 里）

要点：

* **没有 68 点，没有 `get_transform_mat`**：矩阵由调用方给（`image_to_face_mat`）。
* **ct 的参考脸不再重采样**：`compose_frame` 里 `dw = warpAffine(原帧 ROI, fwd)`，
  aligned 这条路直接用画布上的那张脸（就是 dfm 的输入本身）。
* **遮罩来源跟着画布走**：XSeg 的 dst 遮罩从画布缩到 256（`_resize_canvas`：缩小
  `INTER_AREA`、放大 `INTER_CUBIC`），不再从原帧 warp —— 否则又是两张画布。
* 画布尺寸和模型输入不一定相等（本项目 aligned 是 462~480，模型是 384/416）：
  `mat` 乘 `S/S_aligned`（整张画布按比例缩放，仿射矩阵线性缩放即等价），
  像素用 `_resize_canvas` 一次；几何仍然精确（见第五节 0.078 px）。

### 3.1 流水线怎么选画布（`app/server/_proj_pipe.py`）

    DFL_PROJ_COMPOSE=aligned   默认。读帧段：读原帧 + 读对齐脸 → _resize_canvas(S)，
                               mat = image_to_face_mat × S/S_al；合成段：compose_aligned_frame
    DFL_PROJ_COMPOSE=facelib   老行为。读帧段：68 点 → get_transform_mat → warp 原帧到 S；
                               合成段：compose_frame

* 合成线程数搬到 `DFL_PROJ_COMPOSE_THREADS`（默认 4）；老写法 `DFL_PROJ_COMPOSE=8`
  仍按线程数解释（**全数字不当模式用**，免得旧脚本把模式写成 8）。
* 个别对齐脸**缺 `image_to_face_mat`**（5 点模式切的、元数据被洗过的）会自动退回
  facelib 画布，并在日志里计数：`合成画布统计：aligned N 张 · facelib 兜底 M 张`。
  同一帧里两种画布都有时，按画布分组各贴一次（正常项目里不会发生）。
* 日志第一行就写着画布：`… 合成 4 线程（aligned 画布）…`，status 里是
  `compose_mode=aligned`。

### 3.2 每张脸每帧的重采样次数

| 环节 | facelib 画布 | aligned 原生 |
|---|---|---|
| dfm 输入 | 原帧 → S（warp，每帧一次） | 画布 → S（resize；画布本身是切脸时采的那一次） |
| ct 参考脸 | 原帧 ROI → S（warp） | **0**（就是画布） |
| dst 遮罩 256 | 原帧 → 256（warp） | 画布 → 256（resize，只碰遮罩） |
| 反贴 | S → 原帧（warp） | S → 原帧（warp，输出 ROI 大 13%——重切前；重切后 <1%） |
| 合计（脸像素） | **3 次**（+1 次遮罩） | **2 次**（+1 次遮罩） |

---

## 四、像素差异（与 facelib 路逐像素对拍）

同一份项目、同一份模型、`force=true` 各跑一遍，merged/ 逐像素比（60 帧，全有脸）：

| 模型 | mean | max | 差异像素 | 逐字节相同 | 相位相关位移 | 强边缘上 mean / 平坦处 mean |
|---|---|---|---|---|---|---|
| 416 阿証 | **0.0196 LSB** | 103 | **0.40%** | 30/120 | 0.0097 px | 0.102 / 0.026 LSB |
| 384 张元英 | **0.0268 LSB** | 131 | **0.40%** | 30/120 | 0.0062 px | 0.134 / 0.034 LSB |

（逐字节相同的 30 帧是**没脸的原样搬帧** —— 90 张对齐脸分在 120 帧里。60 帧全有脸的
夹具上同一对参数是 mean 0.0275 / 0.0388 LSB、逐字节相同 0/60。）

差异的形状（这也是「不是整体错位」的证据）：

* **相位相关位移 0.01 px**：两张成片之间没有任何整体平移 —— 错位会立刻暴露成一个
  `(dx,dy)` 峰值。
* **>16 LSB 的差异像素里 97~99% 落在原帧的强边缘上**（灰度 Sobel > 24）：重采样点的
  差异只会出现在梯度大的地方（轮廓、睫毛、发丝、下巴边）。全帧 16405 / 30898 个这样的
  像素，只占 0.40% 的面积的一部分。
* 平坦区域（皮肤）mean 只有 **0.026~0.034 LSB** —— 差不到 1/30 个色阶。
* 差异图（×8 放大，左= facelib，中= aligned，右= 差异）：
  `%TEMP%\dfl_compose_ab\out\diff_416_crop.png` / `diff_384_crop.png`，全幅 `diff_416.png`。

> 差异为什么不是「只有边缘」？因为画布换了以后 **dfm 看到的构图也变了**：aligned 画布
> 比 facelib 宽 13%（**重切前**；重切后两张画布重合，这项差异趋零），模型输入里人脸占的
> 比例不同，pred 本身就有亚色阶级的差别；它摊在
> 整张脸上是 mean 0.03 LSB，只有落在轮廓上的那部分才被放大到几十 LSB。
> 换句话说：**这一版不是「同一张图少采一次」，而是「换到了另一张（本来就该用的）画布上」。**

---

## 五、位置一致性（验收 2）

做法：在**画布**上画 4 个已知坐标的品红斑（远离中心，越靠边两张画布的差越大），
让它走一次合成、贴回原帧，再用**同一张 `image_to_face_mat`** 取回画布，量重心的偏移。
三个对照：

| 画布位置 | ① aligned 原生（mat） | ② 同一张画布 + facelib 矩阵（混用） | ③ 自检（facelib 画布 + facelib 矩阵） |
|---|---|---|---|
| (145.6, 133.1) | **0.025 px** | 11.760 px | 0.084 px |
| (257.9, 133.1) | **0.028 px** | 4.221 px | 0.042 px |
| (145.6, 257.9) | **0.051 px** | 13.062 px | 0.053 px |
| (257.9, 257.9) | **0.078 px** | 7.072 px | 0.081 px |

* ① 就是新路的闭环误差：**< 0.1 px**，远好于 1 px 的门槛。
* ③ 证明仪器本身是对的（facelib 画布 + facelib 矩阵同样闭合）。
* ② 就是**混用两张画布**的代价：4~13 px（斑点在画布中部；画布四角是 19~38 px，
  见表 2.2）—— 原来的「几十像素错位」不是 image_to_face_mat 的错，是跨画布的错。

---

## 六、帧速（验收 3）

同一份 120 帧项目（75% 有脸）、各 3 次取中位，`force=true`：

> **后端口径（现行）**：按用户要求 **dfm 只用 onnxruntime**（`DFL_DFM_TRT=0` 是默认，
> 见 [13-MERGE-SEMANTICS](13-MERGE-SEMANTICS.md)）。所以这里把「老一轮（416 吃的是 dfm 的
> bf16 TRT 引擎）」和「ONNX 复测」并排放 —— 同一台机器、同一份 120 帧夹具、同样 3 次取中位。

| 模型 / dfm 后端 | facelib | aligned | 变化 |
|---|---|---|---|
| **416** 阿証（**TRT bf16**，老一轮） | 33.41 / 33.68 / 33.74 → **33.68** | 33.37 / 34.22 / 34.30 → **34.22** | **+1.6%**（另一轮 +1.7% / +2.1%） |
| **416** 阿証（**ONNX / ort-cuda**，现行默认） | 22.37 / 25.32 / 25.42 → **25.32** | 25.05 / 25.19 / 25.34 → **25.19** | **−0.5%**（噪声内） |
| **384** 张元英（ort-cuda，老一轮） | 26.60 / 26.87 / 27.59 → **26.87** | 26.03 / 26.92 / 26.97 → **26.92** | **+0.2%**（另一轮 +2.5% / -1.1%） |
| **384** 张元英（ort-cuda，ONNX 复测） | 25.83 / 27.07 / 27.21 → **27.07** | 26.83 / 26.93 / 27.42 → **26.93** | **−0.5%**（噪声内） |

> **416 换回 ONNX 的价格**：33.68 → **25.32**（facelib）／34.22 → **25.19**（aligned）= **−24.8% / −26.4%**。
> 落点全在 dfm 前向上：**29.8 → 42.3 ms/张**（+12.5ms/张 × 90 张脸 ≈ +1.13 s / 120 帧，
> 与墙钟 3.56s → 4.74s 差 1.18s 对得上）。这是**用户口径要求的代价**（dfm 只走 onnxruntime），
> 不是回归。
> **对照组**：384 两次复测 26.87 → 27.07（+0.7%）· 26.92 → 26.93（0.0%）—— 机器没变，
> 差异全来自 416 的后端。
> **显存**（416 · ONNX）：一份 worker 会话 **1.05GB**，整卡峰值 **4288 MB**（开跑前 866）；
> 384 那一趟两个 session 同时在卡上，峰值 8511/8151 MB（贴墙换页）。
> 384 那档的「变化」落在噪声里（同档多次跑自身就有 ±3% 抖动：26.60~27.59）。
> 能确定的是：**aligned 从没有慢过一次**（416 三次都比 facelib 快，384 两轮合计持平）。

分段耗时（每帧 ms，416 / 384）：

| 段 | 416 facelib | 416 aligned | 384 facelib | 384 aligned |
|---|---|---|---|---|
| 读帧（含解码 + 画布） | 34.8 | **37.5** | 35.4 | **36.0** |
| dfm 前向 | 29.8 | 30.2 | 39.2 | 40.4 |
| XSeg | 4.3 | 4.6 | 4.3 | 4.4 |
| 合成合计 | 66.6 | **65.9** | 64.1 | **63.0** |
| ├ 遮罩准备 | 3.0 | 3.2 | 2.8 | 3.2 |
| ├ **色彩迁移** | 15.0 | **9.1** | 14.2 | **9.1** |
| ├ 反贴 | 9.7 | 11.5 | 9.7 | 12.0 |
| └ 融合 | 31.7 | 35.8 | 29.8 | 32.9 |
| 编码（6 线程） | 127.1 | 128.8 | 128.4 | 128.9 |

> ⚠️ 416 那两列的 **dfm 前向 29.8 / 30.2 ms** 是「dfm 后端 = 原生 TRT bf16」时量的。
> 换成**现行默认（ONNX / ort-cuda）**后同一份夹具是 **42.3 / 42.8 ms**；其余各段不变（±1ms），
> 也就是上表除 dfm 之外**与后端无关**。

读法（这也是「为什么只快 2%，而不是翻倍」）：

* **省掉的**：ct 里「原帧 ROI → 画布」那一趟 warp（15.0 → 9.1 ms，-6 ms）与读帧段
  的 `get_transform_mat` + 原帧 warp（~2 ms）。
* **新付的**：读帧段要解一张对齐 JPEG（462²，~2.5 ms）再缩到 416；反贴/融合的 ROI
  大 13%（画布更宽，覆盖的原帧面积更大）→ +1.8 / +4.2 ms。**重切后画布与 facelib 重合，
  这一项应当归零**（本表是重切前量的，没重跑）。
* 两边都没碰到的是大头：**dfm 前向（29~40 ms，串行）** 与 **PNG 编码（127 ms，6 线程）**。
  想让帧速再上一个台阶，该动的是编码档位（`DFL_PROJ_PNG_LEVEL`）或 dfm 本身，
  不是合成画布。

> 结论：**aligned 原生不是提速手段，是「少一处补偿、少一次重采样、几何只有一份真相」**；
> 帧速上是持平偏快（+1.7% / +2.5%，三次里每次都不低于 facelib）。

---

## 七、回归

| 验收 | 结果 | 关键条目 |
|---|---|---|
| `proj_pool_accept.py`（协议形状 / 攒批扫描 / 逐像素 / 取消续跑） | **16 PASS / 0 FAIL** | 1×4 vs 1×8 逐字节相同 120/120；取消后留下 53 帧前缀；续跑产物 vs 一次跑完逐字节相同 120/120 |
| `proj_default_accept.py`（默认档按模型体量收缩） | **28 PASS / 0 FAIL** | 大模型 1×4 / 小模型 1×2；显存峰值 ≤85%；A vs C 逐像素 mean 0.0019 LSB |
| `stage6_accept.py`（视频六阶段，含 ⑤ compose 对拍 4.0） | **64 PASS / 0 FAIL** | 与 4.0 逐帧 mean **0.001** / max 2；⑤ 纯 CPU、无推理调用；取消/续跑全过 |
| `fused_export_accept.py`（视频 fused 全链路） | **64 PASS / 0 FAIL** | fused **17.49 帧/秒**（dfm 走 ONNX；TRT 时代 20.24）；fused vs 分段 **0.30~0.70 LSB**（门槛 1.0，dfm 换 ONNX 后不再可能逐字节相同，见 §六）；降档/400/帧范围/项目导出回归全过 |

前两套走的是**项目模式**，也就是**默认的 aligned 画布** —— 它们同时在当这一版的功能
回归（画布换了、像素对拍与断点续跑语义没变）。后两套是**视频模式**，代码这次一行没改，
`compose_frame` 仍是它的唯一入口，跑它们是为了证明「真的没碰」。

---

## 八、已知取舍 / 边界

1. ~~**13% 缩放差是模板形状差异，不是 bug**~~ → **2026-12 更正**：那句把尾部当成了
   典型值。真实分布是 **中位 +0.4%、p95 +13%**，主因是 **3° 旋转**加姿态相关缩放，
   根因是**拟合点数**（68 vs 33）。`align_mat_from_68` 改成只拟合 33 点之后，aligned
   画布与 facelib 画布**逐位等价**（zoom 1.00000 / rot 0.0000° / 四角 0.000 px），
   「两种模式看到的构图差 13%」这件事已经不存在 —— 两条路现在是同一张画布，
   差别只剩「pred 在哪算」（aligned 直接吃落盘芯片，facelib 从原帧重采一次）。
   详见 [24-ALIGN-WHOLEFACE](24-ALIGN-WHOLEFACE.md)。
2. **aligned 模式依赖 DFLJPG 元数据完整性**：没有 `image_to_face_mat` 的对齐脸（5 点
   模式切的）会退回 facelib 画布并计数；连 68 点都没有的那种，只有 facelib 那条路
   还能靠检测兜底。
3. **视频模式没动**：fused 的 dst 脸是现场检测 + 对齐出来的，画布本来就是 facelib
   （`compose_frame`），要换画布等于换检测/对齐的输出契约，另议。
4. **收尾与对账不受影响**：断点续跑判据（文件存在且非空）、按帧序提交、ffmpeg 编码
   都在画布之外。

---

## 九、复现

    rem 位置 + 像素 + 帧速（一条命令，退出码 0 = 全过）
    runtime\server\Scripts\python.exe app\tools\proj_compose_ab.py

    rem 只跑某几段 / 换帧数 / 换重复次数
    runtime\server\Scripts\python.exe app\tools\proj_compose_ab.py --cases pos --frames 60
    runtime\server\Scripts\python.exe app\tools\proj_compose_ab.py --cases fps --frames 120 --reps 3

    rem 手工 A/B 一次项目导出（二选一）
    set DFL_PROJ_COMPOSE=facelib
    set DFL_PROJ_COMPOSE=aligned

    rem 合成线程数（默认 4；老写法 DFL_PROJ_COMPOSE=8 也认）
    set DFL_PROJ_COMPOSE_THREADS=4

产物：`%TEMP%\dfl_compose_ab\out\`（`diff_416.png` / `diff_416_crop.png` / `pixel.json` /
`fps.json`）与 `%TEMP%\dfl_compose_ab\result.json`（逐条 PASS/FAIL）。
