# 合成器（合并工作台）

把训练好的 `.dfm` 贴回一段视频。整条链路的每一步都在本仓库里，不依赖外部脚本。

## 一、用户视角的流程

```
导入视频 ──▶ 自动抽帧 + 对齐 ──▶ 预览/调参 ──▶ 导出设置 ──▶ 合成中 ──▶ .mp4
            （后台任务）        （逐帧预览）    （编码/路径）   （进度页）
```

1. **导入视频** —— 左栏「＋ 导入视频」选一个文件（桌面版直接弹文件对话框）。
   点了之后后端开始抽帧 + 对齐，左栏素材列表里出现一条 `名字 · N 帧`。
2. **预览** —— 中间是原帧画面，时间轴拖动逐帧看；「检测」标签会画人脸框。
3. **调参** —— 右栏：检测器、人脸类型、合成模式、遮罩来源/模式、色域迁移、腐蚀/羽化。
4. **导出** —— 右下「导出」→ 导出设置（格式 / 编码器 / 解码加速 / 并行进程 / 输出路径）
   →「开始导出」→ 进度页显示真实阶段（载入模型 → 逐帧合成 → 编码 → 完成）。

## 二、后端接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/api/merge/prepare` | 视频 → 项目（抽帧 + 对齐），后台跑 |
| GET | `/api/merge/prepare/status` | 准备进度：`stage/pct/msg/frames/aligned` |
| GET | `/api/merge/videos` | 列素材（已排除 `*_merged.mp4` 与缓存目录） |
| POST | `/api/merge/open` | 打开视频或项目目录（项目模式优先用**原帧**做预览） |
| GET | `/api/merge/frame/{i}` | 取第 i 帧（视频解码 / 项目直接读图） |
| GET | `/api/merge/detect/{i}` | 对当前帧跑检测，返回框 |
| GET | `/api/merge/analyze` | 按步长扫全片，给人脸轨统计 |
| POST | `/api/merge/export` | 启动合成：`project` + `model`（或 `dfm`）+ `settings` + `encoder/hwaccel/workers` |
| GET | `/api/merge/export/status` | 进度：`running/ok/pct/stage/msg/out/log` |
| POST | `/api/merge/export/cancel` | 终止合成子进程 |
| POST | `/api/merge/preview-frame` | 静态帧文件 → 检测/换脸/合成 → 图（项目模式预览用） |
| POST | `/api/merge/export-project` | **DFL 项目模式**：逐帧合成到 `<项目>/merged/` 再编码；支持 `force` 重做 |
| GET | `/api/merge/export-project/status` | 项目导出进度：`done/total/frame_fps/skipped/failed/warnings/missing/encode` |
| POST | `/api/merge/export-project/cancel` | 取消项目导出（已写好的帧保留） |
| POST | `/api/merge/reveal` | 在资源管理器里定位文件/目录（「打开所在目录」） |

## 三、为什么要「抽帧 + 对齐」两步

4.1 的提取器**只落对齐脸、不落原帧**（省磁盘），但合成必须要有原帧才能把脸贴回去。
`/api/merge/prepare` 就是把这两件事补齐：

```
<项目>/000000.jpg        原帧（ffmpeg 逐帧，6 位序号，从 0 起）
<项目>/aligned/000000_0.jpg   对齐脸（DFLJPG，带 landmarks / source_filename）
<项目>/merged/000000.jpg      项目导出的产物：换脸后的整帧
```

序号必须对得上：`_align_stream` 按解码顺序写 `source_filename = "%06d" % idx`（**idx 从 0 起**），
`_extract_frames` 也必须从 0 起（`-start_number 0`）—— 以前这儿写的是 1，两边整整差一位，
于是 `stage_project` 认不出任何一张对齐脸（"没有任何对齐脸能对上帧"）。
`_proj_scan_aligned` 认帧时先按原名，再按去扩展名，最后才试整体平移 ±1（老项目兼容），
三条都对不上就进 `missing_source` 并明确报出来，不瞎配。

## 四、合成本体

`app/server/_merge_runner.py` 是桥：把 4.1 的目录布局搭成 DFL 项目（硬链接零拷贝），
再调 4.0 `MergeStudio` 的 `run_export_pipeline()`：

```
staging（硬链接）→ 载入 DFM → 逐帧合成 → ffmpeg 编码 → 清理 staging
```

几个必须注意的点：

- **CUDA DLL 要先热**：不调用 `_setup_dll_paths()`，`CUDAExecutionProvider` 会静默退回 CPU，
  日志里只有一行 warning（实测差 12 倍）。runner 里在建 session 前先调它，并把实际
  provider 打进日志。
- **staging 目录名必须是 ASCII**：MergeStudio 内部用 `cv2.imwrite` 写合并帧，OpenCV 在
  Windows 上走 ANSI 路径，遇到中文目录会**静默失败** —— merged/ 变空，最后编码阶段报
  「没有可编码的帧」，而整体退出码还是 0。所以 `work` 用输出名 stem 的 sha1 前 8 位
  （`.merge_1a2b3c4d`），最终 mp4 仍然可以叫中文名。
- **自己编码**：`_make_encoder_no_probe` 替掉原来的 `_stage5_encode_video`，不查 ffprobe、
  不混音，按已知 fps 直接编码；硬件编码器不可用时自动退回 `libx264`。
- **帧率**：合成的是「帧序列」，fps 由源视频真实帧率决定（`prepare` 时记下来）。
- **调试**：`DFL_MERGE_KEEP=1` 跑 `_merge_runner.py` 会保留 staging 目录，方便看 `proj/merged`。

## 五、项目模式导出（/api/merge/export-project）

「导出」是 4.0 MergeStudio 子进程一口气出 mp4；「导出项目」是另一条路，就在本进程里跑：

```
<项目>/<帧名>.jpg  ──检测→68 点→DFM 换脸→遮罩→合成──▶  <项目>/merged/<帧名>.jpg
                                                        │
              ffmpeg -framerate <原fps> -i merged/%06d.jpg -i <原视频> -map 0:v -map 1:a?
                     -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a copy -movflags +faststart <out>
```

- **逐帧合成不另写一套**：走的就是单帧预览的 `_preview_job`（`routes_merge.py`），
  只把「从视频里按时刻取帧」换成「直接读 `<项目>/<帧名>`」。所以同一帧走预览
  （`/api/merge/preview-frame`）和走导出，输出是同一份字节。
- **帧序**：以源帧文件名的字典序为准（= ffmpeg image2 的读取顺序）；`aligned/*.jpg` 只用来
  决定「这一帧有没有脸」和点名缺帧，认不出来的进 `missing_source`，不猜。
- **断点续跑**：`merged/<帧名>` 存在且非空就跳过（`force: true` 才重做）。落盘是
  `.part` → `os.replace`，被杀不会留下半张图被当成已完成。
- **没脸的帧**：原帧**原样 copy** 进 merged（不重编码，不掉画质），所以成片长度和原片一致。
- **缺帧**：编码前对账，`merged` 比源帧少就明确报「缺哪几张」（能按命名规律推算时就点名），
  并把「成片会比原片短 N 秒」写进 warnings。编号有洞时按顺序硬链接重排成 `%06d` 再编 ——
  image2 遇到第一个洞就会停，否则成片会被悄悄截短。
- **fps / 分辨率 / 音轨**：一律来自原视频（`_video_probe`）；读不到就报错，不猜。
  默认输出名 `<项目目录>.mp4`，如果它正好等于原视频文件则改用 `<项目目录>_merged.mp4`，
  绝不覆盖素材。
- **不要 `-shortest`**：音轨比画面长时它会把成片截短（4.0 那边踩过）。
  编码前把 `scale=out_range=tv` 挂上，否则 JPEG 帧（full range）会让成片标成 `yuvj420p`。

## 六、怎么验收

```
python app/tools/merge_accept.py                 # 默认样片，跑到「导出启动 + 取消」为止，约 30 秒
python app/tools/merge_accept.py 视频.mp4 --keep  # 连导出一起跑完（千帧约 2 分钟）
```

它起后端、用 CDP 驱动无头 Edge **真实点击**，逐项断言：导入素材 / 选模型 / 检测出框 /
时间轴换帧 / 合成视图 / 参数采集 / 导出启动 / 进度 / 取消。全部 PASS 才算过。

> 注意：脚本里 `RL = Path(__file__).resolve().parents[1]` 假定自己在 `<仓库>/tools/`，
> 而它现在在 `app/tools/`，直接跑会拿到 `<仓库>/app/app` 这个不存在的目录
> （`NotADirectoryError`）。临时验证时把文件复制到 `<仓库>/tools/` 再跑即可。

项目模式导出的验收（不在 merge_accept 里）：

```
python app/tools/test_merge_api.py        # 接口层
# 端到端：prepare 一个 3 秒 90 帧的项目 → POST /api/merge/export-project
#        → merged/ 90 张、尺寸与源帧一致、ffmpeg 报 90/30=3.0s + aac 音轨
#        → 删一半 merged 再跑（只补缺的）、删源帧再跑（点名缺哪几张）
```

## 七、已知限制

- 「人脸库」这一步（分组 / 去重 / 计算特征）还没做，导出流程**跳过**它 ——
  合成本身不需要人脸库，MergeStudio 靠 DFLJPG 元数据定位人脸。
- 速度约 4~5 帧/秒（1080p，含解码 / 遮罩 / 编码），和 4.0 的原管线一致；
  瓶颈在逐帧 PNG 落盘与遮罩计算，不在 DFM 推理。
  项目模式实测 1080x1920 / 416 的 DFM：逐帧 3.9~4.4 帧/秒，90 帧总计 23~25 秒
  （其中编码 1.1 秒）。
- 项目模式的 merged 帧和源帧同名同格式（jpg→jpg / png→png）。jpg 会按 quality（默认 92，
  与单帧预览一致）重编码一次；没脸的帧走原文件 copy，不重编码。
- CUDA 推理的浮点累加顺序不保证逐次一致：同一帧连跑两次，像素可能差 0~8/255（JPEG 量化会
  放大这点差异）。逐字节一致是常态（实测 10 帧里 5 帧完全相同，其余最大差 ≤8）。
- 质量（JPEG 百分比）与帧缓存格式目前不影响最终视频，编码走 ffmpeg 的默认质量。
