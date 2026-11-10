# 对齐脸写盘改成真 4:4:4（色度不抽稀）

一句话：`write_dfljpg(..., subsampling=0)` 在 cv2 5.0 上一直是**假的** —— 参数被拒、
静默落成 4:2:0，而且每次编码刷一句 warning 仍照旧写盘。现在改成「探测一次 → 按文件头
核对 → 缓存」，写不写得出来只影响一条日志，不再影响磁盘上的字节。

## 出了什么问题

cv2 5.0 把 `IMWRITE_JPEG_SAMPLING_FACTOR` 的值改成了
`ImwriteJPEGSamplingFactorParams` 枚举（`IMWRITE_JPEG_SAMPLING_FACTOR_444` 之类）。
老写法 `[..., IMWRITE_JPEG_SAMPLING_FACTOR, 0]` 会被拒绝：

```
[ WARN:0@0.053] global grfmt_jpeg.cpp:809 cv::JpegEncoder::write The value(0) for
IMWRITE_JPEG_SAMPLING_FACTOR must be one of ImwriteJPEGSamplingFactorParams.
It is fallbacked to IMWRITE_JPEG_SAMPLING_FACTOR_420.
```

关键在于 **imencode 照样返回 True、照样落盘**：

| 编码参数 | 字节数 | 文件头采样因子 |
|---|---|---|
| `[QUALITY,100]`（不传采样参数） | 78070 | (2,2)(1,1)(1,1) = 4:2:0 |
| `[QUALITY,100, SAMPLING_FACTOR, 0]`（老「4:4:4」写法） | 78070 | (2,2)(1,1)(1,1) = 4:2:0 |
| `[QUALITY,100, SAMPLING_FACTOR, SAMPLING_FACTOR_444]`（现在） | 177049 | (1,1)(1,1)(1,1) = 4:4:4 |

两张字节数**完全相同**、文件头完全相同 —— 老代码里那句 `except AttributeError` 兜不住
它（常量在，是**值**错了）。后果：

- 4.1 提取出来的 1203 张 `data_src/aligned` 实测 **1203/1203 全是 (2,2)**，没有一张是 4:4:4；
- 分段导出的切脸阶段写了 `quality=100, subsampling=0`，同样一直是 4:2:0；
- 每张脸还会往 stderr 刷一句 fallback warning（几千张就是几千行）。

## 现在的写法（`app/server/_align.py`）

1. `jpeg444_probe()`：**探测一次**，按 cv2 有没有那两个常量组候选参数：
   - `enum444`：`IMWRITE_JPEG_SAMPLING_FACTOR` + `IMWRITE_JPEG_SAMPLING_FACTOR_444`（cv2 ≥ 4.5.3 / 5.x）；
   - `int0`：老 cv2 认的裸整数 `0`；
   - 不硬编码枚举数字，全靠 `getattr` 探测。
2. 候选不是「看参数对不对」，而是**编码一张 16×16 样板图、解析落盘字节的 SOF 头**，
   三个分量都 (1,1) 才算数（`jpeg_sof_sampling()`）。参数写对了不等于文件是 4:4:4。
3. 结果缓存在模块里（`threading.Lock` 保护）：**之后不再编码探测、不再打日志**。
4. 三条路径的日志：
   - 成功 → 首次一条 INFO：`[align] JPEG 4:4:4 探测：cv2 5.0.0 → enum444（…SOF 采样因子 [(1,1),(1,1),(1,1)]）`
   - 两条候选都不行 → 首次一条 WARN，并写清后果（**色度被抽稀，嘴唇/眼白边缘和肤色过渡
     的细节永久丢失，训练再补不回来**），之后再写盘一律安静；
   - 中途任何一张编不出来都抛 `RuntimeError`，不吞。
5. `write_dfljpg()` 默认 `subsampling=0`（4:4:4）。`subsampling=None` = 一个采样参数都不带、
   库默认 4:2:0 —— **只给对照/复现老产物用**。
6. `routes_merge._stage_imencode`（分段导出帧图的 jpg 分支）删掉自己那份
   `try/except AttributeError`，改成共用 `jpeg_encode_params(100, 0)`：一次探测、全进程一个策略。

回退路径也验过（验收 `a2`）：把 `IMWRITE_JPEG_SAMPLING_FACTOR_444` 从 cv2 上删掉，模拟
「没有枚举常量的老版本」→ 探测明确判 `path=none / ok=False`，**不谎报成功**；连写 3 张图
只出现 **1 条** 告警和 **1 条** cv2 自己的 fallback warning（都只在前 3 张里的第一次），
之后全部安静地按 4:2:0 落盘。

## 实测数字（cv2 5.0.0；OpenCV 5.0.0 的枚举，4.13 同源）

**体积**（q95，真实对齐脸）：

| 输入 | 旧 4:2:0 | 新 4:4:4 | 增幅 |
|---|---|---|---|
| 现有 aligned（432²~448²，**本身已是 4:2:0 解码结果**） | 29.2~29.9 KB | 35.7~36.5 KB | **+21.5%** |
| 从无损 PNG 帧按 `image_to_face_mat` 重新 warp 的「新鲜」裁剪 | 30.4~30.9 KB | 39.3~40.1 KB | **+29.0%** |

**PSNR（写盘后解码回来 vs 写盘前的输入）**：

| 场景 | 整体 PSNR 旧→新 | 色度 PSNR 旧→新 | 第二代色度 旧→新 |
|---|---|---|---|
| 新鲜裁剪（提取器真正的输入） | 47.4~47.8 → 49.1~49.4（**+1.66 dB**） | 51.1~51.3 → 53.3~53.6（**+2.24 dB**） | 50.8~51.1 → 53.3~53.5（**+2.50 dB**） |
| 现有 aligned（已是 4:2:0） | 58.9 → 54.2（−4.7 dB） | 63.8 → 57.9 | 61.8 → 57.8 |

### 为什么第二行是「4:4:4 更差」—— 别被它骗了

把**已经 4:2:0 解码出来**的图再走一遍 4:2:0，是同一套滤波链的不动点：先 box 下采样、
再 fancy 上采样，几乎正好还原输入，所以 PSNR 高得离谱（58.9 dB），**它量的不是保真度，
是「这套滤波对自己不动点有多忠实」**。要量编码器的保真度，输入必须是没被抽稀过的图 ——
也就是提取器从源帧 warp 出来的新鲜裁剪（上面第一行）。所以：

- **不许**拿老 aligned 当基准去比较两个编码器；
- 老数据的 4:2:0 损失已经发生，重编码成 4:4:4 不会把信息变回来（见下节）。

## 读取端：老 4:2:0 与新 4:4:4 并存

没有问题，也做了断言：

- `DFLJPG.load` 只用 SOF0 段里的 `precision/height/width` 取 shape，**根本不看采样因子**；
  APP15（pickle）的位置、内容与采样因子无关，换采样因子不改元数据（表1 里 APP15 两边集合一致）。
- 引擎 `DFLJPY.load / get_landmarks`：混合目录 33 张（30 新 + 3 老）→ load 33/33、
  `get_landmarks` 全是 (68,2)、`cv2.imread` 33/33。
- 服务端 `read_dfljpg_meta`：33/33 都能解出 dict。
- 采样因子只影响**解码后的像素**，不影响文件名、元数据、数据集扫描、遮罩读写
  （`rewrite_dfljpg_meta` 只重写 APP15，不重编码 JPEG）。

## 要不要重提老数据

**结论：按「源视频还在不在」分。**

| 情况 | 建议 | 理由 |
|---|---|---|
| 源视频/帧还在，这批脸要拿来训练 | **值得重提** | 重提才会真的多出 +1.7 dB 整体 / +2.2 dB 色度；老文件里那些色度已经没了 |
| 源没了 / 只是留档 | **不用动** | 新旧混用读取端完全一致（见上）；只是同一数据集里两种保真度并存，训练会按解码结果一视同仁 |
| 想「把老 4:2:0 批量另存成 4:4:4」 | **不要做** | 体积 +21.5%、信息量 +0 —— 抽掉的高频回不来，只是把文件变大 |

也就是说：**不是正确性问题，是素材质量问题**。不重提不会让任何东西读不出来或训崩。

## 影响面清单

现在会写 4:4:4 的路径（都走 `write_dfljpg` / `jpeg_encode_params`）：

- GUI 提取：`app.py` `_align_stream` → `write_dfljpg`（默认 4:4:4）；
- 分段导出切脸：`routes_merge.py` `write_dfljpg(..., quality=100, subsampling=0)`；
- 帧图 jpg 分支：`routes_merge._stage_imencode`（默认还是 png，jpg 只在 `image_format=jpg` 时）；
- `app/tools/faces_profile.py` 统计的就是同一个函数。

仍然按库默认写 4:2:0 的（本仓库 GUI 不走，列出来免得误会）：

- 4.0 引擎自带的 `app/engine/mainscripts/Extractor.py`、`xlib/face/Faceset.py`
  （`IMWRITE_JPEG_QUALITY` 一个参数，不碰采样因子）；
- XSeg 遮罩：走 PNG（`encode_xseg_mask`），与 JPEG 采样因子无关；
- `rewrite_dfljpg_meta` / `write_app15_head`：只搬 APP15 段，不重编码。

## 复核

```powershell
runtime\server\Scripts\python.exe app\tools\aligned444_accept.py
```

38 项断言，当前全过（`PASS 38 · FAIL 0`）：

- 旧写法字节 == 不传参数（静默降级证据）；新写法文件头 (1,1)(1,1)(1,1)；
- 模拟「没有 _444 常量」：判定 `path=none`，告警 1 条，落盘 4:2:0 且不刷屏；
- 单张新旧 size / 整体 PSNR / 色度 PSNR / 第二代色度 PSNR（表1 + 表2）；
- 30 帧真提取：`30 帧 → 30 张脸`，文件头 **30/30 = (1,1)(1,1)(1,1)**，单张平均 47.6 KB；
  **外部解码器独立核对**：PIL `Image.layer` 全是 (1,1)、ffmpeg 报 `yuvj444p`（老 aligned 对照是 `yuvj420p`）；
  引擎 `DFLJPG.load` 30/30、`get_landmarks` 30/30 = (68,2)、服务端 `read_dfljpg_meta` 30/30；
- 后端日志：「探测」INFO **1 条**、cv2 fallback warning **0 次**、告警 **0 次**；
- 老新混放 33 张（30 新 + 3 老）：load 33/33、get_landmarks 33/33、imread 33/33。

退出码 0 = 全过。产物留在 `%TEMP%\dfl_aligned444_accept`。
