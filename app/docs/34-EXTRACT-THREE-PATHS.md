# 34 · 三条「提取」链路：别看名字，看谁落盘

> 起因：打开 `backend_extract/` 会以为"提取器"只有一套，但**合成时也在切脸**，
> 而且是另一套调度代码。以前它们只差在名字上，找代码时找错。

## 一句话分辨

| 你想要的 | 代码在哪 | 会不会落盘 | 谁触发 |
|---|---|---|---|
| **前端提取页**：视频 → 人脸图 | `backend_extract/align_stream.py`（`_align_stream`） | **会**：`aligned/*.jpg`（DFLJPG） | `POST /api/extract/start` |
| **快速导出**（融合流水线） | `backend_merge/pipeline_fused.py` 的 det#0 / dfm#0 车道 | 不会，只在内存里过 | `POST /api/merge/export`（fused） |
| **项目导出** | 4.0 的 `engine/MergeStudio/core/export_pipeline.py`（`_stage2_detect_frames_mp`） | 不会，写 sqlite | `POST /api/merge/export`（project） |

三者**共用同一套几何**：`_vision/detect.py`（检测）+ `_vision/align.py`
（5 点粗对齐 → 2D106 → `landmark106to68` → `align_face_68`）。
差别只在：落不落盘、有没有缓存、谁调度。

**纪律**：不许在任何一条链路里自己写 `round()` / 选插值 —— 今天那些 0.1px 量化、
INTER_LINEAR 混用的坑，全是"某处自己写了一遍"造成的。要改几何改 `_vision/align.py`。

## backend_extract/ 现在的分工（2026-11 拆过）

```
align_stream.py   ~680 行  边解码边对齐（提取页主路径，三级流水：reader/detector/writer）
resume.py          ~88 行  续跑指纹（换了源素材要能发现，否则旧帧被当"已完成"跳过）
frames.py         ~158 行  只抽帧（可选对齐）+ 老名字转发
media.py          ~248 行  ffmpeg 解码 / 抽帧 / 视频探测 / 项目扫描
pipeline.py       ~269 行  通用三级流水（decode → detect_batch → emit），由调用方注入三段
dataset_fast.py   ~264 行  数据集侧：清晰度、dhash 去重
routes.py          ~19 行  只有 GET /api/extract/status（**前端不用**，已标 deprecated）
```

## 接口层的一处已知错位（未动）

```
POST /api/extract/start     在 backend_models/trt.py:318     ← 不在 backend_extract 里
POST /api/extract/cancel    在 backend_models/trt.py:361
GET  /api/extract/progress  在 backend_models/trt.py:367     ← 前端轮询的是这条（SSE）
GET  /api/extract/status    在 backend_extract/routes.py:16  ← 只给验收脚本
```

功能性没问题，位置错。搬它属于纯搬家 + 重跑提取验收，**等要动结构时一起做**。

## 合成用的取帧为什么不在提取器里

`backend_merge/frames_prep.py`（`_merge_prepare_worker` / `_merge_frame_image`）：
视频↔项目互转、合成器按当前打开的视频/项目取一帧。调用方全在 `backend_merge`，
之前寄居在 `backend_extract/frames.py` 里，是"提取器有两个职责"这个错觉的来源。
