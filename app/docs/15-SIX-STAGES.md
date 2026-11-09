# 15 · 六阶段视频导出（切帧 / 切脸 / 换脸 / 遮罩 / 合成 / 编码）

> 前一版是四阶段（切帧 → 切脸 → 合成 → 导出），合成阶段里一口气做「检测 + dfm + 遮罩
> + 融合」，实测 1080×1920 只有 2.0 帧/秒。这一版按用户口径拆成六步，**推理全部前移**，
> 合成阶段只剩纯 CPU 的融合，并把语义基准钉在 4.0 的视频导出（MergeStudio）上。

## 目录结构（沿用 <视频>_cache/）

    <视频>_cache/
      frames/    000001.png …     ① 切帧（ffmpeg -compression_level 1，无损）
                                  ⑤ 合成**同名就地覆盖**成合成结果（不另开 merged/）
      aligned/   000001_00.jpg …  ② 切脸：DFLJPG（original_name / 68 点 / face_mat / source_rect）
       pred/     000001_00.png     ③ 换脸：dfm 输出的正脸（无损 PNG）
                000001_00_prdm.png     dfm 自带 prd 遮罩
                000001_00_dstdm.png    dfm 自带 dst 遮罩
       mask/     000001_00_xseg_prd.png  ④ 遮罩推理（XSeg(pred)，256×256 灰度）
                000001_00_xseg_dst.png      XSeg(原帧按 68 点裁到 256)
                000001_00_dfm_prd.png      来源=DFM 时把 ③ 的遮罩也落在这里留档
      meta.json                    阶段标记 + 每阶段游标 + settings 快照

## 六步与 4.0 的对应（语义基准）

| 我们 | API | 4.0 里的位置 | 说明 |
|---|---|---|---|
| ① 切帧 | POST /api/merge/stage/frames | `MergeStudio/core/export_pipeline.py::_stage1_extract_frames`（1291） | ffmpeg 直接写 PNG 序列 |
| ② 切脸 | …/faces | `_stage2_detect_frames_mp`（645）+ `_stage2_read_dfl_aligned`（1590） | 4.0 只写 sqlite；我们额外落 DFLJPG（能直接喂训练/遮罩绘制） |
| ③ 换脸 | …/pred | `_phase_swap_worker`（693-738）里 dfm 那一半 | 一次 N 张（dfm batch 维是动态的） |
| ④ 遮罩 | …/mask | `_phase_mask_worker`（758-799） | XSeg(pred) + XSeg(dst)，两张 256 灰图 |
| ⑤ 合成 | …/merge | `_phase_merge_worker`（808-976） | **纯 CPU**，逐段见下 |
| ⑥ 编码 | …/encode | `_stage5_encode_video`（1705） | ffmpeg 直接读 PNG 序列（不经过 cv2） |

⑤ 合成的逐段映射（`app/server/_merge_compose.py` 顶部也写着同一张表）：

| 段 | 4.0 行号 | 做什么 |
|---|---|---|
| 遮罩准备 | 840-893 | mask_mode 0-9 组合（1=dst hull、2/4/5=prd、3=dst、6/8=xseg_prd、7=xseg_dst、9=全乘）+ pad32 腐蚀/膨胀 + 高斯模糊 + clip |
| 色彩迁移 | 896-935 | none/rct/lct/mkl/mkl-m/idt/idt-m/sot-m/mix/mt/lut（mt=match_tone_np，遮罩软加权、不转 Lab） |
| 融合算法 | 931-966 | overlay（按 mask alpha 叠）/ hist-match（color_hist_match 后再叠）/ seamless（cv2.seamlessClone NORMAL_CLONE 再按 mw 混）/ seamless-hist-match |
| 反贴回原帧 | 937-943 | fmat = get_transform_mat(lm, sz, WHOLE_FACE)，WARP_INVERSE_MAP 把脸和遮罩一起变回原帧 |
| 写盘 | 969-975 | PNG 无损；就地覆盖 frames/ |

与 iperov 的差异（以 4.0 为准）：4.0 视频模式的合成**没有** MergeMaskedFace 里那段
pp（seamless 之后再把结果 warp 回脸空间做二次色彩迁移）；`mt`（match_tone_np）也是
4.0 自己加的，iperov 没有。单帧预览那条路（MergeMaskedFace）保留，但现在预览也切到
compose 上了（见下）。

## 性能

1080×1920、单 dfm、60 帧、8 线程、PNG level 1：

- ⑤ 合成 **32.3 帧/秒**（4 线程时 20.9，单线程 7.4；8 线程 31.8-32.3）
- 每帧分解（ms）：read 25.9 / prep 4.0 / mask 3.7 / **ct 52.8** / paste 9.1 / blend 14.0 / **encode 80.5** / write 3.9
- ③ 换脸 batch=8，约 10 张/秒；④ 遮罩约 24 张/秒；⑥ 编码 60 帧 0.8s（约 70 帧/秒）
- 与 4.0 逐像素对拍（真跑 4.0 的 `_phase_merge_worker`，engine python）：mean 0.001 / max 1-2

优化点（按收益）：整帧 float32（25MB）的重活全部压进脸的包围盒 ROI；遮罩的统计与
写回限制在遮罩包围盒；帧读取换 np.fromfile+imdecode（_imread_any 要 40ms，这里 21ms）；
PNG 写回 level 1；多线程跑帧（cv2 释放 GIL）。

## 验收

`python app/tools/stage6_accept.py`（真后端 + 1080×1920 样片 + 现有 dfm，全在 %TEMP%）：
六步顺序跑通、每步只产出自己的目录、遮罩产物读回断言、409 闸门、来源切换失效、
③④ 取消续跑、纯 CPU 断言（推理调用 0 / session 不增 / 显存不涨）、fps ≥ 20、
成片 vs 单帧预览 mean ≤ 8、与 4.0 逐像素对拍 mean ≤ 2 / max ≤ 8、预览 vs 导出帧 PNG = 0。
