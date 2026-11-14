# 24 · 切脸与 iperov 的 WHOLE_FACE 完全等价（33 点拟合，68 点只当证人）

> 一句话：`_align.align_mat_from_68` 以前用**全部 68 点**拟合一张经验模板，和
> `facelib.get_transform_mat` 不是同一张画布（中位差 0.4% 缩放 + 2.5° 旋转，尾部
> 到 13%）。现在只拟合 facelib 认的 **33 点**、模板直接 import，
> 同一组 68 点算出来的矩阵与 iperov **逐位等价**（zoom 1.00000 / rot 0.0000° /
> 四角 0.000 px）；多出来的 35 个点保留，只做**外点检查**。
>
> 机器：RTX 5070 Laptop 8GB · R9 8940HX · 32GB · Windows · cv2 5.0.0 · numpy 2.5.4

---

## 一、速览

| 项 | 改前 | 改后 |
|---|---|---|
| 拟合点数 | 68（33 个 DFL 点 + 35 个经验点） | **33**（17..48 + 54，= facelib） |
| 模板来源 | 手抄的 float64 字面量 | **`facelib.LandmarksProcessor.landmarks_2D_new`** |
| vs facelib（1203 组真实点） | zoom 偏差 ≤ 0.384 · rot ≤ 13.4° · 四角 ≤ **149.6 px** | zoom 偏差 **4.2e−07** · rot **1.7e−05°** · 四角 **4.1e−04 px** |
| 4.0（iperov）芯片不动点 | 四角 ≤ 120.7 px | **≤ 0.0005 px**（3205 张） |
| 68 点 | 参与拟合（把几何拽偏） | **只做外点检查**（`align_resid_px` / `align_resid_rel`） |
| 外点阈值 | 无 | `DFL_ALIGN_OUTLIER_PX`（默认 0.55×画布）+ `DFL_ALIGN_OUTLIER_REL`（默认 12） |
| 数据 | 1203 + 1221 张旧芯片 | 已按新矩阵重切（旧芯片在 `aligned.bak_20261114/`，见第六节） |

改的只有一件事：**用哪几个点拟合**。padding 0.40、整脸下移 7%、三角形求仿射、
umeyama、`align_mat_from_5`（粗裁 192 只喂 2D106）**一行没动**。

---

## 二、根因：拟合点数，不是模板形状

`_align` 以前的做法是：把 68 个点整批拟合到 `UNIT_68`（其中 33 个被钉成 facelib 的
`landmarks_2D_new`，另外 35 个是「4.0 一万张对齐脸的均值」）。多出来的下颌/下巴点
在单位空间里落在 **x∈[−0.12, 1.05]、y∈[0.02, 1.16]** —— 跑到单位方形外面去了，
最小二乘为了照顾它们，把缩放和旋转一起拽偏。

1203 组真实 `source_landmarks` 实测（`facelib.get_transform_mat` 当分母）：

| 量 | 中位 | p05 / p95 | min / max |
|---|---|---|---|
| 缩放 | **+0.36%** | −6.3% / **+13.3%** | −27.7% / +22.5% |
| 旋转 | **−2.53°** | −4.88° / +4.75° | −8.3° / +13.4° |
| 画布（四角+中心）位移 | **33.7 px** | 11.4 / 66.2 px | 5.2 / 149.6 px |

**13% 是 p95，不是典型值**（`23-ALIGNED-COMPOSE.md` `2.2 原来那句是错的，已更正）。
典型情况是「转 2.5°、缩 0.4%」—— 但这已经足够让 ③⑤ 的贴回错几十像素，因为
aligned 画布与 facelib 画布不是同一个东西。

---

## 三、改法（`app/server/_align.py`）

    _TPL_2D_NEW = facelib.LandmarksProcessor.landmarks_2D_new     # 不再手抄
    _REAL_IDX   = list(range(17, 49)) + [54]                     # 33 个下标

    def align_fit(kps68, output_size):
        T    = umeyama(src[_REAL_IDX], _TPL_2D_NEW, True)         # ← 只这 33 对点
        mat  = _affine_from_unit_square(T[0:2], output_size, PADDING, WHOLE_FACE_SHIFT)
        resid = || T·src − UNIT_68 || × (芯片像素/单位)            # 68 个，做外点检查
        return mat, T, resid, px_per_unit

* **模板只从 facelib 取**：手抄的 float64 副本和 facelib 的 float32 差 2.9e−08，
  够让芯片挪 0.01 px 且不报错。现在 `_align` 在需要时才 import（engine 不在
  `sys.path` 上就自己 append 一次，append 不 insert）。
* **`align_mat_from_68` 名字不变**：调用方（extract / add / 合成 / 工具）一行都不用改，
  语义从「68 点拟合」变成「33 点拟合 + 68 点外点检查」。
* **`align_mat_from_5` / `coarse_align_from_5` 原样**：粗裁只给 2D106 准备输入，
  不进几何。
* 元数据多两个读数：`align_resid_px`（芯片像素）/ `align_resid_rel`（max / 中位）。

---

## 四、回归门（`app/tools/align_wf_accept.py`，16 项全 PASS）

    汇总：PASS 16 · FAIL 0 · 用时 21.4s

| 门 | 判据 | 实测 |
|---|---|---|
| A1 zoom | 1.00000 ± 1e−5 | **max 偏差 4.2e−07** |
| A2 rot | 0.000° ± 1e−4 | **max 1.7e−05°** |
| A3 四角位移 | < 0.05 px | **max 4.1e−04 px** |
| B2 4.0 芯片不动点 | < 0.05 px | **max 5.0e−04 px**（n=3205） |
| B3 另一族的差是常数 | zoom 极差 < 1e−4 | **2.0e−06**、rot ≤ 2.3e−05° |
| C1 真实脸不被误拒 | 0 张 | **0 张** |
| C2 人工坏点必须被拒 | 两道门都过阈 | 465.0 px > 206.8 px · **21.4 > 12** 倍 |
| C3 近邻帧的点不误拒 | 两道门都不过阈 | 45.6 px ≤ 206.8 px · 1.9 ≤ 12 倍 |
| D1 与 facelib 同 warp | ≤ 1e−3 LSB | **2.9e−04 LSB**（浮点舍入） |
| D2 落盘 q95 4:4:4 | ≤ 1 LSB | **0.396 LSB** |
| E1 mat∘src = 芯片坐标 | < 0.05 px | **1e−04 px**（n=2424） |
| E2 新芯片上 get_transform_mat 对恒等 | < 0.05 px | **4e−04 px**（n=2424） |

矩阵元素的绝对差 max **7.5e−04** —— 平移项本身就是几百 px，等价于相对误差 1e−6；
这点差来自 facelib 中途把角点转成 float32（`tb_diag_vec.astype(np.float32)`），
落到像素上是 4e−04 px，比门槛小两个数量级。

### 4.1 关于 4.0 芯片的那个「24.02 px」

4.0 的 `data_src/aligned` 有 **4561** 张，按名字分成两族：

| 族 | 张数 | 新矩阵 vs 存储矩阵 |
|---|---|---|
| `NNNNN.jpg`（4.0 自己切的） | **3205** | 四角 **≤ 0.0005 px**（= 恒等） |
| `NNNNNNN_N.jpg`（别处导进来的） | 1356 | 恒定 24.022 px：zoom **0.93365**、rot 0、平移 (16.99, 16.99) |

第二族的差是**同一个常数相似变换**（zoom 极差 2e−06、rot ≤ 2.3e−05°）—— 说明它们的
`image_to_face_mat` 是「facelib 在 478 上算的矩阵放到 512 的画布上」这一类历史产物，
本来就不可能对成恒等。**逐脸漂移为 0** 才是「我们的矩阵与它是同一个函数」的证据，
这一条由 B3 在守。

---

## 五、68 点：只做外点检查

### 5.1 残差怎么算

拟出 `T`（原图→单位）以后，把**全部 68 个点**映射进单位空间，与 `UNIT_68` 比，
再乘「芯片像素 / 单位」（`mat ∘ T⁻¹` 的缩放）→ `resid_px`。用芯片像素是为了把
**画布尺寸**归一化掉（同一个脸在 376 和 946 的画布上不该是两套阈值）；
**姿态没有归一化**，见 5.2。

### 5.2 分布（2424 张真实脸，`--cases c`）

| 集合 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| `data_src/aligned`（1203） | 0.141 | 0.248 | 0.341 | **0.370** |
| `merge/…/aligned`（1221） | 0.102 | 0.184 | 0.227 | 0.239 |

（表里是 `resid_max / S`，乘回画布尺寸就是像素。）

**但这个分布的主变量是姿态，不是检测质量**：

| 绝对 yaw | 张数 | resid/S p50 | p95 |
|---|---|---|---|
| 0–15° | 742 | 0.084 | 0.133 |
| 15–30° | 1160 | 0.122 | 0.193 |
| 30–45° | 238 | 0.182 | 0.273 |
| 45–60° | 284 | 0.223 | 0.339 |

同一个模板贴侧脸当然贴不上。所以**按 p99 定阈值（0.34×画布）等于把侧脸全扔掉** ——
这正是这一轮实测推翻的直觉。另外 2424 张里没有一张是「某几个点单独跑掉」
（max/中位 最大 10.6，p50 4.8~5.5），也就是说**现有数据里根本没有外点**，
这两道门是给将来的抽取准备的。

### 5.3 两道门

| 门 | 环境变量 | 默认 | 管什么 |
|---|---|---|---|
| 绝对 | `DFL_ALIGN_OUTLIER_PX` | `0.55 × 画布边长`（`off` / 数字都认） | 检测整体跑飞：点落到背景/别的脸上 |
| 相对 | `DFL_ALIGN_OUTLIER_REL` | `12.0`（max/中位，`off` 也认） | 某几个点单独崩坏（姿态会被约掉） |

被拒的脸 **不写文件**：`align_face_68` 返回 `meta.rejected=True` + `reject_reason` +
`align_resid_px/rel`；抽取侧计 `stats["rejected"]` 并写「rejected：…」日志，
`/api/faces` 那条路直接把原因当 400 的文案。

自检（C2/C3）证明它**两向都对**：3 个下颌点推出 2 个单位 → 465 px / 21.4 倍，被拒；
近邻帧同一个人的点 → 45.6 px / 1.9 倍，放过。

---

## 六、重切（`app/tools/align_recut.py`）

* 输入：**已经存在 DFLJPG 里的 68 点**（`source_landmarks`）+ 原帧（`source_filename`
  指到的那一帧）。**不重新检测、不重新跑 2D106。**
* 尺寸 S 保持原样；插值默认 CUBIC（与 4.1 合成链路一致，`--interp lanczos4` 可换）。
* 元数据：`landmarks`（芯片坐标）= 新矩阵 ∘ `source_landmarks`、`image_to_face_mat` =
  新矩阵、`align_kernel="facelib33"`；**`xseg_mask` 用 `delta = 新 ∘ 旧⁻¹` 一起搬**
  （3° 在外圈能挪 10 px，不搬就会和脸错位）；pitch/yaw/roll、`source_rect`、
  `source_filename` 原样保留（`source_rect` 是**检测框**，重切不重新检测）。
* 旧芯片先复制到 `aligned.bak_YYYYMMDD/`（不覆盖、不删除，验收后再由人决定）。

| 集合 | 张数 | 耗时 | 拒收 | 缺帧 | 新 vs iperov | 新 vs 旧芯片 |
|---|---|---|---|---|---|---|
| `data_src/aligned` | 1203 | 51.1 s（42.5 ms/张） | 0 | 0 | mean **0.0004** / max 1 LSB | mean 31.4 LSB |
| `merge/9月29日 (5)/aligned` | 1221 | 48.6 s（39.8 ms/张） | 0 | 0 | mean **0.0005** / max 1 LSB | mean 35.0 LSB |

「新 vs iperov」= 同一帧、同一组 68 点，用 `get_transform_mat` + **同一套 warp 参数**
算出来的参考芯片；差的只有浮点舍入（max 1 LSB 只出现在个别像素上）。
「新 vs 旧芯片」的 31/35 LSB 就是这次修掉的几何差本身。

对照图（每行：**旧芯片 | 新芯片 | iperov**，每批 40 张、8 行一版）：

    %TEMP%\dfl_align_ab\recut\plates_src_1.png  … plates_src_5.png     （data_src 1203）
    %TEMP%\dfl_align_ab\recut\plates_proj_1.png … plates_proj_5.png    （项目 1221）
    %TEMP%\dfl_align_ab\recut\report.json                              （逐项数字）

备份：`workspace/data_src/aligned.bak_20261114/`（1203）、
`workspace/merge/9月29日 (5)/aligned.bak_20261114/`（1221）。

### 6.1 全链回归（重切之后，六套一起跑）

| 套件 | 结果 | 与这一版相关的读数 |
|---|---|---|
| `align_wf_accept.py` | **16 PASS / 0 FAIL** | 见第四节 |
| `proj_pool_accept.py`（项目批量协议，会**重切夹具**） | **16 PASS / 0 FAIL** | 1×4 → 24.67 帧/秒 · 峰值 4225MB |
| `proj_compose_ab.py --cases pos,pixel` | **14 PASS / 0 FAIL** | 位置闭环 **0.081 px** · 混用 facelib 矩阵也 **0.081 px**（改前 13.06）· 真实脸两张画布 **mean 0.00 / max 0.00 px**（改前 19.25/37.68）· aligned vs facelib 像素 **0.0117 / 0.0134 LSB**（改前 0.0196/0.0268） |
| `proj_default_accept.py` | **25 PASS / 2 FAIL（与切脸无关，见下）** | 显存预算/帧速全过 |
| `stage6_accept.py`（视频六阶段） | **64 PASS / 0 FAIL** | 与 4.0 对拍照旧 |
| `fused_export_accept.py`（视频 fused） | **64 PASS / 0 FAIL** | fused vs 分段照旧 |
| `vram_plan_accept.py` | **51 PASS / 0 FAIL** | 三档伪造显存全过 |

> `proj_default` 那 2 条 FAIL 是**机器状态**不是代码：`[project] … 空闲 5.50GB · 计划 1 worker(s) × 2
> 线程（线程 4→2，仍不够）`—— 桌面占了 ~2.6GB，1×4 要的那份会话装不下，于是计划被降成 1×2，
> 而断言写死了「× 4 线程」；连带没跑出大模型的会话峰值，`_vram_sess_peak.json` 里只剩小模型那条。
> **用改动前的 `_align.py` / `app.py` / `routes_*.py` 复跑，同样 25 PASS / 2 FAIL**（记录在
> `_baseline_proj_default.log`），所以这两条与切脸无关。

---

## 七、还剩什么差异（诚实段）

1. **关键点模型的形状约定不同**。几何等价只保证「同一组 68 点 → 同一张矩阵」。
   2D106 的 68 点和 DFL 的 FAN 68 点不是同一套标注：拟合到 facelib 模板后，
   33 点残差（单位空间）中位 **0.10~0.12**，而 iperov 自己的芯片是 **0.075**；
   68 点全残差中位 0.21~0.29 vs 0.10。也就是说 **4.1 的芯片仍比真 iperov 芯片
   「更不贴模板」**，这只能靠换关键点模型解决，不是矩阵的事。
2. **插值**：4.1 用 CUBIC（和合成链路一致），iperov 的 `Extractor` 用
   `INTER_LANCZOS4`（`mainscripts/Extractor.py:211`）。同一张矩阵下两者 mean 差
   **0.21 / 0.20 LSB**（1203 / 1221 张）。要逐字节复刻 iperov 就 `--interp lanczos4`。
3. **`source_rect` 没动**：它是检测框，重切不重新检测，所以框比新构图略紧/略松
   （排序、预览框用它，不影响几何）。
4. `data_src` 那批原来落的是 **4:2:0**（早期版本），重切后统一 **4:4:4 / q95**，
   色度比原来好一档 —— 这是顺带的，不是验收项。

---

## 八、复现

    rem 1) 几何门 + 不动点 + 外点分布 + 像素（15 项，退出码 0 = 全过）
    runtime\server\Scripts\python.exe app\tools\align_wf_accept.py

    rem 2) 只看几何（1203 组，最快）
    runtime\server\Scripts\python.exe app\tools\align_wf_accept.py --cases a

    rem 3) 重切（先 --dry-run 看数字，再真跑；--limit 30 试跑）
    runtime\server\Scripts\python.exe app\tools\align_recut.py --dry-run
    runtime\server\Scripts\python.exe app\tools\align_recut.py

    rem 4) 外点阈值（默认 auto = 0.55×画布 / 12 倍）
    set DFL_ALIGN_OUTLIER_PX=off
    set DFL_ALIGN_OUTLIER_PX=300
    set DFL_ALIGN_OUTLIER_REL=8

产物：`%TEMP%\dfl_align_ab\result.json`（回归门）、
`%TEMP%\dfl_align_ab\recut\report.json` + `plates_*.png`（重切）。

## 已接受、不再处理的差异（用户决定，勿再当成欠账去修）

以下两条是**有意保留**的：

1. **关键点标注约定（2D106 vs DFL/FAN）**：拟合残差中位 0.10~0.12（单位空间），iperov 自己芯片 0.075。
   要消除只能换关键点模型 —— **不做**。几何已在 1e-7 量级等价，残留只是标注风格差异。
2. **插值核（CUBIC vs iperov 的 LANCZOS4）**：差 0.21 LSB。
   **不做**：我们按原始分辨率切脸（缩放接近 1:1），插值核的差别在此量级下没有实际意义。
   `align_recut.py --interp lanczos4` 保留作逐字节复刻的逃生门，默认不启用。
