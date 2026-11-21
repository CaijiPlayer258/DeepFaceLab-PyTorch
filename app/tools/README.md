# app/tools —— 脚本索引

分四类，按用途进目录。全部是**开发/验收工具**，产品运行不依赖它们。

## accept/ 验收（回归门，改代码后跑这些）

| 脚本 | 验什么 | 典型耗时 |
|---|---|---|
| `accept/align_wf_accept.py` | 切脸 =原版 WHOLE_FACE（zoom/rot/四角位移 + 4.0 不动点） | 21 s |
| `accept/fused_export_accept.py` | 视频快速导出（fused）全流程 + 与分段逐像素对拍 | 108 s |
| `accept/stage6_accept.py` | 六阶段导出 + 与 4.0 对拍 | 64 s |
| `accept/fused_decode_parity.py` | 解码 rawvideo vs frames PNG（8 类素材逐位相同） | 40 s |
| `accept/det_prep_parity.py` | 检测预处理搬家后框/点一致（0.000 px） | 20 s |
| `accept/proj_pool_accept.py` | 项目导出批量协议 + 断点续跑 | 200 s |
| `accept/proj_default_accept.py` | 项目导出默认档（按权重选 worker/线程） | 80 s |
| `accept/proj_compose_ab.py` | aligned 原生合成：位置闭环 + 像素 + 帧率 | 157 s |
| `accept/roi_parity_accept.py` | 合成 ROI 化与整帧版逐位一致 | 40 s |
| `accept/roi_export_accept.py` | ROI 合成的端到端与续跑 | 120 s |
| `accept/vram_plan_accept.py` | 显存预算三档（拒绝/降档/正常） | 88 s |
| `accept/vram_force_accept.py` | 「仍然尝试」force=1 成功/失败两条路 | 60 s |
| `accept/vram_reclaim_accept.py` | 预览显存回收与归属（含「可回收进不进公式」「先回收再算预算」的顺序钉子） | 82 s |
| `accept/eff_accept.py` | `[eff]` 生效配置 + 三类告警 + 前端横幅 | 40 s |
| `accept/merge_accept.py` | 合成器界面（时间轴/预览/导出按钮） | 76 s |
| `accept/merge_ui_accept.py` | 合成器设置页布局与行数 | 30 s |
| `accept/progress_lit_accept.py` | 导出进度页「流水线实况」：status.stages 六键 / lanes == 实际线程数 / 两段贝塞尔 + 光点 / 点亮 1.15s / 卡住变琥珀 / 抽屉 14 字段（47 条 + 3 张截图） | 150 s |
| `accept/test_faces_api.py` | 人脸库 API | 30 s |
| `accept/fdb_accept.py` · `face_group_accept.py` | 人脸库/分组语义 | 60 s |

较老的、仅作历史参照：`aligned444_accept` `frame_range_accept` `mask_pred_accept`
`merge_chip_accept` `merge_face_debug_accept` `preview_fast_accept` `stage_export_accept`
`test_detector` `test_merge_api` `trainer_accept`。它们仍可跑，但已不是必经门。

## bench/ 性能与剖析

| 脚本 | 用途 |
|---|---|
| `bench/speed_bench.py` | 总速度台架（视频/项目/分段/预览，3 次中位） |
| `bench/export_mem_probe.py` | 导出前后 RSS/提交/USS 时间线 |
| `bench/fused_pipeline_bench.py` | fused 流水线各段耗时 |
| `bench/proj_pipe_bench.py` | 项目流水线 A/B 台架 |
| `bench/stage_perf_profile.py` | 六阶段逐段剖析 |
| `bench/faces_profile.py` | 切脸/检测剖析 |
| `bench/sess_bench.py` · `sess_sweep.py` · `sess_plan_probe.py` | session 池标定与策略扫描 |
| `bench/arm_bench.py` | 四臂 A/B/C/D（合成在 CPU 还是 GPU × x264/NVENC）：12 次导出、NVML+psutil 外挂采样、四项 wait（结论见 `docs/notes/experiments/arm_ab.md`） |
| `bench/fused_lane_ab.py` | GPU 车道数 A/B（`DFL_FUSED_GPU_LANES` 1/2/3）：一臂一子进程 + 轮次交替，含 SM%/显存采样与两份会话的逐位对拍（结论见 `docs/notes/experiments/gpu_lanes.md`） |
| `bench/vram_probe.py` | 显存探测口径对照 |
| `bench/vram_export_ab.py` | 帧预算降档 A/B：预览常驻 → 30 帧/批 8 vs 8 帧/批 1（带帧率）78s |

## migrate/ 数据迁移与一次性工具

`convert_dfl_tf_to_torch` `export_dfm` `export_onnx_to_trt` `fix_dfljpg_app15`
`import_legacy_dat` `migrate_by_id` `unpack_pak` `tonemap_aligned_faces`
`clear_Iter_str` `verify_converted_dfl_model` `tf_strict_infer` `tf_verify_infer`
`train_errface` `build_errface_dataset`（含 `TF_TO_TORCH_CONVERSION.md`）

## audit/ 文案审计

`audit/i18n_audit.py` —— 未翻译文案清点。

## 根目录

`align_recut.py` —— **重切芯片**（用 DFLJPG 里已存的 68 点，不重新检测；`--interp lanczos4` 可逐字节复刻原版）。

## 约定

- 所有脚本从 `parents[N]` 找仓库根：**在子目录里是 `parents[3]`**，根目录是 `parents[2]`。移动脚本时要一起改。
- 验收脚本一律「PASS/FAIL + 退出码」，`0` 才算过；数字要能从日志复现。
- 新脚本进对应目录；不要放在 `app/tools/` 根下（除 `align_recut.py` 这类手工工具）。
