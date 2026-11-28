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
| `xseg` 全量遮罩 | `workspace/model/XSeg/XSeg.onnx` | **bf16** | `("int8", "bf16")` | 同一套网络但更深：INT8 实测 IoU **0.9277** / 二值一致 98.13%（对比 XSegLite INT8 的 0.9861 / 99.64%）。它存在的理由就是比 Lite 准，为了速度丢掉 7% 准头等于抹平这个档 → **只编 bf16** |
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

| 模型 | INT8 质量代价（对同一条预处理、同一批 60 张真实对齐脸） | 结论 |
|---|---|---|
| XSegLite | IoU **0.9861** / 二值一致 99.64% / MAE 0.0036 | 可用（软标签，边缘抖动会被训练平滑） |
| XSeg（全量） | IoU **0.9277** / 二值一致 98.13% / MAE 0.0204 | **否决**（它本来就是"准"的那一档） |
| 特征点 | 关键点平均漂移 **2.26 px**（最大 9.33） | **否决**（硬坐标，会传给对齐矩阵） |
| 检测 | 框质量**未验证**；INT8 只 1.22×，而 Graph 有 2.0× | 不采用 |
| GPEN | 未验证；256 的 9.5 ms 已经够 | 不做 |

> 老 int8 引擎（手工编的那份）在同批数据上是 IoU 0.9681 / 99.15% / MAE 0.0084 ——
> 用 `trt_build.py --int8` 重新校准后 **0.9861 / 99.64% / MAE 0.0036**，质量翻倍接近，
> 代价是速度从 1.250 → 1.367 ms（−8%）。取质量那版。

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

命令（界面「引擎」页背后跑的就是这条。**精度由 `engines.precision_for(name)` 决定**，
不是全局开关 —— 以前界面写死 `--bf16`，所以 INT8 那张"空头支票"一直兑现不了）：

```bat
:: bf16（检测 / 特征点 / 特征向量）
python app\server\backend_models\trt_build.py ^
  --onnx weights\tinymog.onnx ^
  --out  workspace\_engines\tinymog_<md5>_sm1200_trt101339.bf16.engine ^
  --name tinymog --bf16 ^
  --profile "1,3,640,640;8,3,640,640;8,3,640,640"

:: int8（只有 XSegLite 走这条；约 40 s 校准 + 45 s 构建）
python app\server\backend_models\trt_build.py ^
  --onnx workspace\model\XSegLite\xseglite.onnx ^
  --out  workspace\_engines\xseglite_<md5>_sm1200_trt101339.int8.engine ^
  --name xseglite --int8 ^
  --calib-dir workspace\data_dst\aligned --calib-n 200 ^
  --profile "1,3,256,256;1,3,256,256;1,3,256,256"
```

INT8 的校准集必须**和推理同一条预处理**，否则 scale 是按另一套输入分布估的：

```python
cv2.dnn.blobFromImage(img, 1/255, (256,256), swapRB=True, crop=False)   # _vision/xseg.py mask()
```

编完会在引擎旁留一份 `<engine>.calib.json`（样本数 / 目录 / 种子 / 构建时间），
排查「这份引擎到底用什么校准的」时看它。

优化档位（`PROFILES`，动态维模型必须给，否则 TRT 拒绝构建）：

| 模型 | min / opt / max |
|---|---|
| `xseglite` · `xseg` | `1,3,256,256` ×3（永远单张） |
| `tinymog` | `1,3,640,640` / `8,3,640,640` / `8,3,640,640` |
| `retinaface_10g` | `1,3,640,640` ×3 |
| `insightface_2d106` | `1,3,192,192` / `8,3,192,192` / `8,3,192,192` |
| `arcface_mbf` | `1,3,112,112` / `32,3,112,112` / `32,3,112,112` |

---

## 四、现状

| 项 | 状态 |
|---|---|
| **INT8 通道** | ✅ 已实现：`trt_build.py --int8`（`IInt8EntropyCalibrator2` + 真实对齐脸校准）。界面「引擎」页点编译就按 `precision_for(name)` 自动选精度 |
| `xseglite` | **int8** · 7.6 MB（bf16 是 33.2 MB）· 2.458 → **1.367 ms/张** · IoU 0.9861 |
| `xseg`（全量） | **bf16**（按 §二 的结论不用 int8） |
| 已编 bf16 | `tinymog` · `retinaface_10g` · `insightface_2d106` · `arcface_mbf`（sm1200 / trt101339） |
| 已编 bf16（GPEN） | `gpen_bfr_256.1200.bf16.engine` · `gpen_bfr_512.1200.bf16.engine` |
| **死文件** | `xseglite_int8.engine` · `tinymog_int8.engine` · `insightface_2d106_int8.engine` · `xseglite_bf16_lvl5.engine` · `xseglite_mixed_bf16.engine` —— 旧命名，`try_engine` 永远命中不到，可以删 |
| `arcface_mbf` | 引擎编了，但 `face_embedder.py` 走 onnxruntime，没有加载点 |

### 已知限制

1. **TRT 10 的 implicit 校准表不导出**：`write_calibration_cache` 拿到了空内容，
   `*.calib.cache` 是 0 字节 —— 所以每次重编都要重跑那 200 张校准（约 40 s）。
   要省这 40 s 得换 explicit quantization（QDQ 图），那是另一套 ONNX 改造。
2. **int8 引擎不能跨显卡 / 跨 TRT 大版本**（SM 与 trt 都在文件名里，换卡自动重编）。
3. 校准集变了要重编：`<engine>.calib.json` 里记着样本数与目录，对不上就重编，
   否则 scale 是按旧分布估的（引擎名不带校准指纹，这是当前命名方案的取舍）。
