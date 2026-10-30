# 自测报告（主智能体）

生成时间：2026-10-21 04:05 ｜ 测试环境：Electron 1500×1000 / 1280×800，后端 FastAPI 本地端口

## A 合成工作室（p-merge）

| 编号 | 项目 | 结果 | 实测值 |
|---|---|---|---|
| A1 | 三个页签可切换 | PASS | `.mgview` 3 个，点击后面板 `data-view` 依次为 main → facedb → settings |
| A2 | 时间轴 4 条轨道 | PASS | `document.querySelectorAll('.mtl-trk').length === 4`；刻度尺/播放头/#mtlFrame/#mtlTotal/缩放档 4 项齐备 |
| A3 | 时间轴高度 ≤150px 且不撑破页面 | PASS | `.mtl` 高 139px；页面高 873px（视口 937），`scrollHeight == innerHeight` |
| A4 | 点时间轴跳帧 | PASS | 子智能体 A 的 DOM 桩测试：点 x=200px → 帧 100；`setFrame(150)` → 输入 150、时间码 00:05.00 |
| A5 | 页面本身不滚动 | PASS | `.main` 加 `no-scroll`，`mainScroll: false` |
| A6 | 右栏参数左边缘对齐 | PASS | 17 个 `.mgrow` 的第二子元素 `left` 去重后只有 1 个值（1275） |
| A7 | 无编造参数 | PASS | 右栏实为 14 个检测器 / insightface-2d106det / 5 种 face_type / 8 种模式 / DFM·XSeg·XSegLite / prd×dst·prd·dst / 9 种色域；「羽化·直切·整脸·仅面部·边缘留白·106点·68点」全部不存在 |
| A8 | 窄窗 1280×800 | PASS | 页面 673px、时间轴 139px、右栏无横向溢出（`rightScrollX: false`） |
| A9 | 人脸库页 | PASS | 24 个 84×84 占位格 + 头部「已建库 24 张人脸」+「计算人脸特征」主按钮 |
| A10 | 导出设置页 | PASS | 6 行参数 + 3 组（输出/编码/性能）+ 底部「开始导出」+ 状态行 |

## B 遮罩绘制（p-tool-mask）

| 编号 | 项目 | 结果 | 实测值 |
|---|---|---|---|
| B1 | 4.0 缺失的 5 个控件已补 | PASS | 模型文件下拉 / 分辨率 256 / 反转遮罩（否·是）/ 应用遮罩 / 导出 XSeg·XSegLite → ONNX |
| B2 | 数据集默认路径 | PASS | `workspace\data_src\aligned` |
| B3 | 默认遮罩模型 | PASS | `.seg` 的 `.on` 落在 XSegLite 上 |
| B4 | 启动按钮状态切换 | PASS | 点击后 `mdState` 从「未启动」变为「运行中 · :8000」 |

## C 参数设置（p-train-cfg）

| 编号 | 项目 | 结果 | 实测值 |
|---|---|---|---|
| C1 | 「参数」按钮可跳转并渲染 | PASS | 修复前该按钮无任何绑定；修复后渲染 8 张卡 / 56 行 |
| C2 | 标签右边缘对齐 | PASS | `.cfgrow .cl` 的 `right` 去重后 1 个值 |
| C3 | 键名修正 | PASS | `d_mask_dims`（原 mask_dims）、`lr_total_steps`（原 lr_anneal）、`edge_enhance_power`（原 edge_enhance）、`pia_cycle_power`（原 pia_cycle） |
| C4 | 超 4 项改下拉 | PASS | `selects: 3` —— face_type(5 项) / ct_mode(11 项) / random_color_alg(5 项) 均为 select |
| C5 | 模型卡四按钮均有反应 | PASS | 继续训练→参数页；参数→参数页；导出→toast；删除→确认框（三个原本都是死按钮） |
| C6 | 补齐 4.0 缺失参数 | PASS | `TRAIN_EXTRA_GROUPS` 5 组 19 行已 splice 进 CFG_GROUPS（masked_training / blur_out_mask / backup_interval / freeze_* 等） |

## D 全局

| 编号 | 项目 | 结果 | 实测值 |
|---|---|---|---|
| D1 | 12 页逐页切换无错误 | PASS | home/extract/dataset/train/merge/settings/about/tool-mask/tool-bili/tool-color/tool-angle/debug 全部渲染，`TOTAL ERRORS: 0` |
| D2 | 输入框拼写检查 | PASS | 32 个输入框，`spellcheck=on` 数量为 0 |
| D3 | 路径框可编辑 | PASS | 8 个 `.pathval` 均为 `<input>`，`readOnly: false`；点图标才唤起系统目录选择器 |

## 未完成 / 已知限制

1. 合成器的预览渲染、批量导出、人脸库 embedding 计算均只有 UI，未接后端（需要 `MergeStudio/api/routes_preview.py` 49KB 与 `core/merger.py` 26KB 移植到 FastAPI）。
2. 遮罩绘制的「启动 MaskProcessor」「应用遮罩」「导出 ONNX」只切换状态与 toast，未真正拉起子进程。
3. 训练页「导出」「删除」按钮接了 toast / 确认框，删除接口未实现。
4. 数据处理页、B 站页、调试页高度超过视口，可正常滚动（只有合成器页锁定了滚动）。