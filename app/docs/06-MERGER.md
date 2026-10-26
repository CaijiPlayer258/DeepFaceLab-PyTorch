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

## 三、为什么要「抽帧 + 对齐」两步

4.1 的提取器**只落对齐脸、不落原帧**（省磁盘），但合成必须要有原帧才能把脸贴回去。
`/api/merge/prepare` 就是把这两件事补齐：

```
<项目>/000001.jpg        原帧（ffmpeg 逐帧，6 位序号）
<项目>/aligned/000001_0.jpg   对齐脸（DFLJPG，带 landmarks / source_filename）
```

序号必须对得上：`_align_stream` 写 `source_filename = "%06d" % idx`，
抽帧也用 6 位、从 1 开始，于是 `stage_project` 里能一一对应上。

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

## 五、怎么验收

```
python tools/merge_accept.py                 # 默认样片，跑到「导出启动 + 取消」为止，约 30 秒
python tools/merge_accept.py 视频.mp4 --keep  # 连导出一起跑完（千帧约 2 分钟）
```

它起后端、用 CDP 驱动无头 Edge **真实点击**，逐项断言：导入素材 / 选模型 / 检测出框 /
时间轴换帧 / 合成视图 / 参数采集 / 导出启动 / 进度 / 取消。全部 PASS 才算过。

## 六、已知限制

- 「人脸库」这一步（分组 / 去重 / 计算特征）还没做，导出流程**跳过**它 ——
  合成本身不需要人脸库，MergeStudio 靠 DFLJPG 元数据定位人脸。
- 速度约 4~5 帧/秒（1080p，含解码 / 遮罩 / 编码），和 4.0 的原管线一致；
  瓶颈在逐帧 PNG 落盘与遮罩计算，不在 DFM 推理。
- 质量（JPEG 百分比）与帧缓存格式目前不影响最终视频，编码走 ffmpeg 的默认质量。
