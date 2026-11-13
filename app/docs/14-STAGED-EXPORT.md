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

- `quality`：切脸档，`high` = RetinaFace(det_10g)，`fast` = TinyMog。
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

## 不变的东西

- 视频模式：没人脸库 -> 左栏那个 dfm 换全部脸；有人脸库 -> embedding 归组，不匹配不换；
- 无人脸的帧：不跑 dfm，原帧就是结果（连重编码都不做）；
- 组均值 / auto_plan 缓存：serve 启动清一次，一轮导出结束（encode）再清一次。
