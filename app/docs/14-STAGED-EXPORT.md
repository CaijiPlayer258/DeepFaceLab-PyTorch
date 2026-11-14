# 分段式视频导出（<视频名>_cache + 四个阶段）

> 这一页写的是**视频模式导出的结构**：一个阶段一件事、各自落盘、能单跑能续跑。
> 合成语义（谁换谁、不匹配不换）仍然看 [13-MERGE-SEMANTICS.md](13-MERGE-SEMANTICS.md)。

## 目录

源视频 `D:/x/9月29日 (5).mp4` 旁边：

    D:/x/9月29日 (5)_cache/
      frames/   000001.png …   切帧（1 起、6 位补零）；合成阶段**同名就地覆盖**成合成结果
      aligned/  000001_00.jpg  切脸：DFLJPG（original_name + 68 点 + face_mat）
      meta.json                源信息 / 阶段标记 / settings 快照 / 续跑游标 / last_error

没有 `merged/`：合成结果直接覆盖 `frames/` 里的同名帧（不凭空多一倍空间）。
`meta.frame_swapped=true` 就是「frames 里已经是合成结果」的标记。

## 六个端点

    POST /api/merge/stage/extract-frames   视频      -> frames/        （别名 frames / 切帧）
    POST /api/merge/stage/extract-faces    frames/   -> aligned/       （别名 faces / 切脸）
    POST /api/merge/stage/merge            frames+aligned+dfm -> 就地覆盖 frames/ （别名 swap / 合成）
    POST /api/merge/stage/encode           frames/ + 源音轨 -> <视频名>_merged.mp4 （别名 export / 导出）
    GET  /api/merge/stage/status           当前阶段 / 进度 / 每阶段标记 / 计数 / 日志
    POST /api/merge/stage/cancel           取消（已完成的阶段和产物都留着，下次续跑）

请求体（四个阶段同一份，用不到的字段忽略）：

    {"video": "D:/x/9月29日 (5).mp4",   // 必填；也可以只给 cache_dir，按同目录同名反推
     "cache_dir": "",                  // 默认 <视频父目录>/<视频名>_cache
     "model": "阿証...-416",            // 或 "dfm": "D:/.../x.dfm"；merge 必须有
     "models": [{"name": "...", "faces": []}],   // 多模型时用它，给了就不看 model
     "faces_project": "",              // 人脸库目录；空 = <工作区>/merge
     "detector": "", "quality": "high", "thresh": 0.5, "align_thresh": 0.5,
     "settings": {...}, "output_dir": "", "force": false}

- `detector`：**切脸/对齐真正用的检测器**（界面「检测器」下拉发的就是它）。
  给了就以它为准；空着才看 `quality`。这条优先级是 2026-11 修的 bug：以前
  `cfg["det_align"]` 只从 `quality` 推，而界面根本不发 `quality`（`payload.align_quality || "high"`），
  于是「检测器」下拉无论选什么都落成 retinaface_10g —— 界面写 TinyMog、日志里 45~49ms/帧。
  现在 fused 与切脸两处日志都会写「检测器 X（来源）」，见下面「检测器从哪来」。
- `quality`：切脸档的**兜底**，`high` = RetinaFace(det_10g)，`fast` = TinyMog。
  只在请求体没给 `detector` 时生效（老脚本 / 验收工具 `quality=fast` 走的就是这条）。
- `force`：重做这一阶段。切帧比较特殊：`frames` 已被合成结果覆盖时必须 `force=true`
  才会重切，否则 409 —— 那把原帧盖回来、却留下「已合成」的标记。
- 幂等：已完成的阶段再点一次 -> `{"ok":true,"started":false,"skipped":true,"reason":"already-done"}`。

## cache 的一生

| 时机 | 行为 |
|---|---|
| 第一次 POST 任一阶段 | 建 `<视频名>_cache/`（源视频旁） |
| 阶段成功 | 写 meta.json 的阶段标记（原子替换：.tmp -> os.replace） |
| 合成中（每 10 帧） | 写 `stages.merge.done_frames` 游标 —— 断电后从这里接着换 |
| **导出成功（encode 完成）** | **整个 `<视频名>_cache/` 删掉**（它只是缓存） |
| 失败 / 取消 | 保留，meta 记 last_error / last_stage / 游标，下次从这里续跑 |

`DFL_STAGE_KEEP=1`（环境变量，只给验收用）可以保留 cache 用于数文件，不是界面选项。

## 帧图质量

帧图默认 **PNG 无损**：`ffmpeg -compression_level 1` 抽帧，合成结果由 `_img_bytes(..., "png")`
写 `IMWRITE_PNG_COMPRESSION=3`。jpg 兜底分支写 `IMWRITE_JPEG_QUALITY=100`（不用默认 95/85/75）。

**JPEG 质量固定 100，界面没有这个旋钮**（`routes_merge._STAGE_JPEG_QUALITY`）：aligned 的
DFLJPG 与 fused 的 dst 裁图都走 q100 + 4:4:4。以前导出设置页那颗「质量 95」既不可编辑、
后端也从来没读过它 —— 现在控件删掉，参数口径留下：
`body.jpeg_quality` 仍然认（1..100，默认 100，只给老客户端/验收脚本），
响应（`_stage_brief_meta`）与日志都回显真实值（`jpeg_quality=100` / `JPEG quality=100`），
meta.json 的 `stages.frames.jpeg_quality` 也是 100。**PNG 路径不受这个数影响**。

## 检测器从哪来（一个复发性 bug 的形状）

界面只有一个「检测器」下拉（`mgDetector`，默认 **TinyMog**），请求体里叫 `detector`。
但 4.1 之前 `det_align`（切脸/对齐那一段真正用的检测器）**只**从 `quality` 推：

    cfg["det_align"] = _stage_quality_detector(body.get("quality") or body.get("align_quality"))

而前端发的是 `quality: payload.align_quality || "high"` —— 页面上根本没有 align_quality 这个控件。
结果：**界面选 TinyMog，导出实际跑 retinaface_10g**。和 `mask_source` 那次是同一个形状
（界面选了、没传）。修法是一处明确的优先级，落在 `_align_detector(body)`：

| 请求体 | 切脸/对齐用的检测器 | 日志 |
|---|---|---|
| `detector="tinymog"` | tinymog | `检测器 tinymog（界面/请求体 detector=tinymog）` |
| `detector=""`、`quality="fast"` | tinymog | `…（请求体没给 detector → 按 quality=fast 推）` |
| 都没给 | retinaface_10g（老行为） | `… 按 quality=high 推` |

fused 与分段两条链共用这一个函数：`cfg["detector"]`（预览 / 人脸库 embedding 用的那个）
与 `cfg["det_align"]` 从此是同一个值 —— 以前 boxes 来自 retinaface、embedding 那条路却按
tinymog 走，属于两套口径。

**日志对账**：`_fused_run` 在起流水线之前就打一条（分段链在切脸开头打同样一条）：

    快速导出：检测器 tinymog（界面/请求体 detector=tinymog · TrtSession）· 1080p 单帧 2.48 ms
              = letterbox 0.82 + 前向 1.83 + NMS/解码 0.0（零图 3 次中位）

这条日志的意义是**名字和毫秒在一行**（零图空跑一张预热后再量，第一张的 CUDA Graph 捕获
不吃进中位数）。tinymog 正常 2~6ms（bf16 的 TRT 引擎）、retinaface_10g 12~16ms；单帧超过
20ms 会额外打一行 `★` 提示（查 TRT 引擎 / 是否退回 onnxruntime / 是否在 CPU EP 上）。
注意这是**空闲**口径；流水线里 detect 段还要排队等 GPU（见下）。

## 检测输入（letterbox）在解码线程做

检测器的第一步是 `1080p → 640 短边 + 填边`（cv2.INTER_LINEAR；RetinaFace 填 0、TinyMog 填 128）。
这一步被拆成 `prep()` + `detect_ctx()`（`app/server/_detect.py`），fused 里由**解码线程**先算，
检测线程只剩「前向 + NMS」，profile 多一列 `det_prep`。

- 为什么拆：解码线程只忙 ~15ms/帧（30~40%），检测线程是硬底（retinaface 89% 占用）。
- 为什么是“搬家”不是“省掉”：那 0.7~0.9ms 的 resize 必须有人做；搬到有富余的线程才是净赚。
- **逐位一致**：同一个 `_prep_ctx()`、同一个插值、同一张 canvas，`detect()` 与
  `prep()+detect_ctx()` 的框/关键点/分数误差 **0.000**（`app/tools/det_prep_parity.py`，退出码 2 = 有误差）。
- `DFL_FUSED_DET_PREP=0` 关掉（回老口径），A/B 用。

**为什么不做「ffmpeg 直接解 640」**（`-vf scale=640:-2`）：swscale 的 bilinear 和 cv2 的
INTER_LINEAR 不是同一个核，逐像素 mean 1.49 LSB / max 57 → 框偏 3.6~7.1px、关键点偏 3.4~8.2px，
tinymog 上还**丢过一张脸**（12 张里少 1 张）。要「检测结果一致」就只能复用同一个 resize，
所以最终选的是这条。数据：`det_prep_parity.py --scale-route`。

**收益（本机 1080x1920 / 120 帧 / 384 dfm / 合成 8 线程）**：

| 配置 | 帧/秒 | detect 段 ms/帧 | det_prep ms/帧 |
|---|---|---|---|
| retinaface_10g，letterbox 在检测线程（老口径） | 22.35 | 38.69 | — |
| retinaface_10g，letterbox 在解码线程 | 23.26 | 37.68 | 1.53 |
| tinymog，letterbox 在检测线程 | 28.85 | 18.37 | — |
| tinymog，letterbox 在解码线程（默认） | 28.30 | 16.23 | 1.86 |

老实说结论：detect 段本身降了 2.1ms（−11%），但**tinymog 下端到端是 0**（±2% 抖动内）——
那时瓶颈已经移到 GPU：dfm 59% + detect 46% + xseg 9% ≈ **113% 的墙钟都压在同一块卡上**，
每一段量到的毫秒里都含"排队等 GPU"。只有 retinaface 那一档（detect 单独就 88~89%）才看得见。
把这条链从 20 提到 28 帧/秒的是**上面那个检测器 bug**（+24%），不是这次搬家。
下一步能榨出两位数的地方只剩 GPU 那侧：给 dfm 编 TRT 引擎（现在日志明确回退
`onnxruntime(CUDA)`，dfm 20.7ms/帧是最大的一块）。

> ⚠️ **按用户要求，dfm 默认不使用 TRT**（`DFL_DFM_TRT=0`，ONNX only）：磁盘上就算有
> `_engines/dfm_*.engine` 也一律忽略。上面这条结论留着当「这块卡上 dfm 还剩多少余量」的参照，
> **不是待办**；真要对照就显式 `DFL_DFM_TRT=1`，那组数字不属于本页口径。

**像素一致性怎么验**：`fused_export_accept.py --cases a,b`（同一段 120 帧，fused vs 分段）
本期实测 **mean=0.0000 LSB / max=0 / 120 帧全等**，PASS 23 / FAIL 0。注意这条链本身有
**跑一次一个样**的噪声底：同一配置连跑两次，实测 mean 0.28~0.46 LSB、max 31~60（dfm 攒批的
组成随队列时序、ORT/cuDNN 选核），所以像素级比较只对「同一批组成」有意义；本改动的验收锚点
是框层面 0.000。

## 不变的东西

- 视频模式：没人脸库 -> 左栏那个 dfm 换全部脸；有人脸库 -> embedding 归组，不匹配不换；
- 无人脸的帧：不跑 dfm，原帧就是结果（连重编码都不做）；
- 组均值 / auto_plan 缓存：serve 启动清一次，一轮导出结束（encode）再清一次。
