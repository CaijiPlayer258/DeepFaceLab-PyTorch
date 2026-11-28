# 41 · 引擎精度矩阵（每个模型该编什么精度）

> 口径来源：`app/docs/09-PERF.md`（实测）、`19-SPEED.md`（dfm 不编 TRT）、
> `36-FACE-SR.md`（GPEN）、`10-OPTIMIZATION-ROUND.md`（ArcFace）。
> 代码落点：`app/server/backend_models/engines.py`（MODEL_SPECS / PROFILES / try_engine）、
> `app/server/backend_engines/routes.py`（编译入口）。

---

## 一、结论表（这就是"每个模型编什么精度"）

| 模型 | onnx | 期望精度 | 运行时查找顺序 | 依据 |
|---|---|---|---|---|
| `xseglite` 遮罩 | `workspace/model/XSegLite/xseglite.onnx` | **int8**（没有就 bf16） | `("int8", "bf16")` | 算力瓶颈（29 Conv + 4 上采样）：2.04 → **0.751 ms（2.7×）**；bf16 只快 10%；CUDA Graph 反而更慢 15% |
| `xseg` 全量遮罩 | `workspace/model/XSeg/XSeg.onnx` | **int8**（没有就 bf16） | `("int8", "bf16")` | 同 XSegLite（同一套网络，只是权重不同） |
| `tinymog` 快检测 | `weights/tinymog.onnx` | **bf16 + CUDA Graph** | `("int8", "bf16")` | launch 瓶颈：Graph **2.0×**（3.91 → 2.98 ms）；INT8 只 1.22× 且**框质量没验证过 → 未采用** |
| `retinaface_10g` 高精度检测 | `weights/retinaface_10g.onnx` | **bf16** | 默认 `"bf16"` | 高精度档，不为速度降精度；CUDA Graph 和 torch 抢同一线程会炸（`captures_underway.empty()`），必须 `use_graph=False` |
| `insightface_2d106` 特征点 | `weights/insightface_2d106.onnx` | **bf16 + CUDA Graph** | 默认 `"bf16"` | launch 瓶颈：Graph **3.6×**（0.711 ms/脸）；INT8 **1.05× 且关键点漂移 2.26 px（最大 9.33）→ 否决** |
| `arcface_mbf` 人脸特征 | `weights/arcface_mbf.onnx` | **bf16** | 无（走 onnxruntime） | 13 MB，比 r50 快 5 倍；精度够聚类用。引擎编了但目前没人加载（见 §四） |
| `gpen` 人脸超分 256 / 512 | `weights/GPEN-BFR-256.onnx` / `-512` | **bf16** | 直接 `TrtSession` | 256 = 9.5 ms、512 = 57 ms；**INT8 暂不做**（要准备校准集，`trt_build.py` 只有 `--bf16`；256 的 9.5 ms 已经够） |
| `dfm` 换脸模型 | 用户 `.dfm` | **不编 TRT** | 硬规则 `DFL_DFM_TRT=0` | dfm 只走 onnxruntime CUDA EP / torch（bf16w）。磁盘上有 `.engine` 也不加载，只会在日志里喊一句 |

### 一句话记忆

> **浅网络（launch 密集）→ CUDA Graph + bf16；深网络（算力密集）→ INT8；要坐标/要画质的 → 一律 bf16。**

---

## 二、为什么不能"全都上 INT8"

| 模型 | INT8 质量代价 | 结论 |
|---|---|---|
| 遮罩 | IoU 0.970 / 二值一致 99.1%，差异只在轮廓 1~2 px，且**单向内收** | 可用（软标签，边缘抖动会被训练平滑） |
| 特征点 | 关键点平均漂移 **2.26 px**（最大 9.33） | **否决**（硬坐标，会传给对齐矩阵） |
| 检测 | 框质量**未验证** | 不采用 |
| GPEN | 未验证 | 不做 |

> 判据不是"快多少"，是**误差会不会被下游放大**。

---

## 三、引擎命名与编译命令

命名（`engines.py:101 engine_path()`）：

```
<模型名>_<onnx的md5前12>_sm<SM>_trt<TRT版本>.<精度>.engine
例：xseglite_2ca27ae913e9_sm1200_trt101339.bf16
```

* SM = `主版本*100 + 次版本*10`：30 系 860 · 40 系 890 · 50 系 1200
* 换显卡 / 换 onnx / 升 TRT 大版本 → 必须重编（三个都在文件名里）
* 两个根：`workspace/_engines`（MODEL_SPECS 里的模型）、`weights/_engine`（GPEN）

命令（界面「引擎」页背后跑的就是这条）：

```bat
python app\server\backend_models\trt_build.py ^
  --onnx weights\tinymog.onnx ^
  --out  workspace\_engines\tinymog_<md5>_sm1200_trt101339.bf16.engine ^
  --name tinymog --bf16 ^
  --profile "1,3,640,640;8,3,640,640;8,3,640,640"
```

优化档位（`PROFILES`，动态维模型必须给，否则 TRT 拒绝构建）：

| 模型 | min / opt / max |
|---|---|
| `xseglite` · `xseg` | `1,3,256,256` ×3（永远单张） |
| `tinymog` | `1,3,640,640` / `8,3,640,640` / `8,3,640,640` |
| `retinaface_10g` | `1,3,640,640` ×3 |
| `insightface_2d106` | `1,3,192,192` / `8,3,192,192` / `8,3,192,192` |
| `arcface_mbf` | `1,3,112,112` / `32,3,112,112` / `32,3,112,112` |

---

## 四、现状与缺口（核对过的）

| 项 | 状态 |
|---|---|
| 已编 bf16（规范命名） | `xseglite` · `tinymog` · `retinaface_10g` · `insightface_2d106` · `arcface_mbf`（sm1200 / trt101339） |
| 已编 bf16（GPEN） | `gpen_bfr_256.1200.bf16.engine` · `gpen_bfr_512.1200.bf16.engine` |
| **缺** | `xseg`（全量）只有 onnx，没有引擎 |
| **死文件** | `xseglite_int8` · `tinymog_int8` · `insightface_2d106_int8` · `xseglite_bf16_lvl5` · `xseglite_mixed_bf16` —— 旧命名，`try_engine` 永远命中不到 |
| **编译端限制** | `trt_build.py` **只有 `--bf16`**，没有 `--int8`，也没有校准集通道 → 界面上编不出 INT8 引擎。所以 `("int8","bf16")` 里的 int8 目前是**空头支票**（除非手工用别的工具编出规范命名的 int8 引擎） |
| `arcface_mbf` | 引擎编了，但 `face_embedder.py` 走 onnxruntime，没有加载点 |

### 要做 INT8 还缺什么

1. `trt_build.py` 加 `--int8` + `IInt8EntropyCalibrator2`，校准集用**真实对齐脸**（200 张，合成数据会让 scale 偏）；
2. 校准集缓存文件写进引擎名（否则换校准集不重编，scale 是旧的）；
3. 只有遮罩值得编 —— 检测/特征点按 §二 的结论不编。
