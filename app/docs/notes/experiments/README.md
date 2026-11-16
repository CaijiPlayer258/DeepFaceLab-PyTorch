# 实验记录（从仓库根的 `experiments/` 摘出来）

这里是各轮性能/正确性实验的**文字结论**，一份实验一个文件：

| 文件 | 原目录 | 讲什么 |
|---|---|---|
| `dec_lane.md` | experiments/dec_lane/ | 解码腿：readinto 环形缓冲、-threads 池、101→119 帧/秒 |
| `det_lane.md` | experiments/det_lane/ | 检测腿：GIL、探针、lane 划分 |
| `dfm_graph.md` | experiments/dfm_graph/ | DFM 图执行/算子数/压力 |
| `seamless.md` | experiments/seamless/ | cv2.seamlessClone 的 torch 复刻（逐位 97.14%、其余差 1 LSB） |
| `skip_bad.md` | experiments/skip_bad/ | 坏帧降级（默认关逐字节相同） |
| `enc_unlock.md` | experiments/enc_unlock/ | 编码器解锁矩阵 |
| `gpu_tl.md` / `gpu_tl.tables.md` / `gpu_tl.prose.md` | experiments/gpu_tl/ | GPU 时间线/并发实测（表格与正文分开存） |
| `conc_bench.md` | experiments/conc_bench/ | 并发台架 |

**没有进仓的东西**：这些实验的 `*.json` / `*.png` / `*.mp4` / `_debug/` / 一次性 `*.py` 探针
（共约 70 MB）都在清理时删掉了 —— 它们只是生成上面这些结论的中间产物，结论与复现口径都在
正文里。唯一例外是 `experiments/seamless/torch_seamless.py`：它是 `compose_gpu` 的**运行期
依赖**，已经移到 `app/server/backend_merge/torch_seamless.py`（见 app/docs/26-COMPOSE-GPU.md）。

4K 编码器那一份实测报告不在这个目录，在 `app/tools/bench/enc4k/`（配套 6 个测量脚本）。
