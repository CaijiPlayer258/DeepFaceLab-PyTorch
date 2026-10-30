# 4.1 独立复核报告（合成工作室 / 遮罩绘制 / 参数设置 / 全局）

复核对象：`D:\MyProject\DeepFaceLab-PyTorch-4.1`
参考实现：`D:\MyProject\DeepFaceLab-PyTorch`（4.0，只读，未做任何写入）
复核方：独立审查（不采信任何交付方说法，全部数值为当次实测）

## 0 环境与方法

| 项 | 实测值 |
| --- | --- |
| 后端 | `app\serve.py` + `..\_python\Scripts\pythonw.exe`，打印 `PORT=8761`，`GET /` = 200（51509 B） |
| 前端 | Electron **33.4.11**（`desktop\node_modules\electron`），`win.loadURL('http://127.0.0.1:8761/')` |
| 窗口 | 1500×1000（innerHeight 937）；窄窗回归 1440/1280/1100/1024 |
| 工作区 | 被 `workspace.txt` 指向旧工程 `D:\MyProject\DeepFaceLab-PyTorch\workspace`（只读使用） |
| 截图 | `_shots\review_merge_main.png`、`review_merge_1280.png`、`review_tool_mask.png`、`review_train_cfg.png`、`review_confirm.png`、`review_merge_after.png` |

**方法学修正（重要）**：Electron 33 的 `console-message` 签名是 `(event, level:number, message, line, sourceId)`，`level` 是**数字** 0–3（实测 `console.error` → level=3，`console.warn` → 2，`console.log` → 1，且 `event.level === undefined`）。
用题面给的 `e.level === 'error'` 判定**永远收不到任何错误**，会得到假的 "0 error"。本轮改为三重采集：`level===3` + 页面侧 `window.onerror` + `unhandledrejection` + `console.error` 包装，三路结果一致（均为空），D1/D3 才成立。

## 1 逐项结果

| 编号 | 项目 | 结果 | 实测值或失败原因 |
| --- | --- | --- | --- |
| A1 | 三个页签切换 | 通过 | 依次点 main/facedb/settings，`.mgview.in` 的 data-view 依次为 `main`→`facedb`→`settings`；`.mgview` 共 3 个，`.in` 恒为 1 个 |
| A2 | 多轨时间轴要素 | 通过 | `.mtl-trk`=**4**（mtlVideo/mtlFaces/mtlCut/mtlAngle）；`#mtlRuler` 存在、刻度 `.mtl-tk`=2；`#mtlHead` 存在（left=0px）；`#mtlFrame`=0、`#mtlTotal`=300；`#mtlZoomSeg b`=**4**（1/0.5/0.25/0.125） |
| A3 | 时间轴高度 / 不撑滚动条 | 通过 | `.mtl` 高 **139.33px**（max-height 150px）；document.scrollHeight=**937** = innerHeight=**937**，溢出 **0px**；1440/1280/1100/1024 四档复测均 139.33px、溢出 0、`body.scrollWidth-innerWidth`=0 |
| A4 | 点时间轴播放头移动 | 通过 | 点 #mtlVideo 62%：`#mtlFrame` 0→**299**，`#mtlHead` left 0px→**598px**，时间码 00:00.00→00:09.96；另测 20%→112 帧(224px)、40%→225 帧(450px)，2px/帧线性 |
| A5 | 页面本身不滚动 | 通过 | `.main` scrollHeight=**897** / clientHeight=**897**，差 **0px**（阈值 2px）；overflow-y=hidden；1024–1500 宽全部差 0 |
| A6 | 右栏参数左边缘对齐 | 通过 | 17 个可见 `.mgrow` 的第二子元素 `left` 去重后 **1 个值：1274.7px**；展开「高级」details 后仍 17 行、仍 1 个值 |
| A7 | 无编造项 / 真实值齐全 | 通过 | 17 个标签对黑名单**精确命中 0**；唯一子串命中「遮罩羽化」= 4.0 `blur_mask_modifier`（`MergeStudio/ui/js/params.js:158`，默认 4），**不是编造**。真值实测：检测器 14 项（RetinaFace_10g/500m/MogFace/TinyMog/S3FD/MTCNN/BlazeFace/CenterFace/DamoFD/LightweightFD/ULFD/YOLOv8/YoloV5Face/YoloV11nFace）；特征点 insightface-2d106det；人脸类型 5；模式 8（overlay…debug）；遮罩来源 DFM*/XSeg/XSegLite；遮罩模式 prd×dst*/prd/dst；色域迁移 9 项与 4.0 `params.js:162` 完全一致 |
| B1 | 遮罩页六个控件 | 通过 | `#mdModelFile` 下拉（3 项）、`#mdRes`=**256**、`#mdInvert`=否*/是、`#mdApply`「应用遮罩」、`#mdExportXSeg`「导出 XSeg → ONNX」、`#mdExportLite`「导出 XSegLite → ONNX」 |
| B2 | 数据集默认值 | 通过 | `value="workspace\data_src\aligned"`（attribute 与 property 一致） |
| B3 | 遮罩模型默认 XSegLite | 通过 | 该 `.seg` 内 `b.on` 唯一命中 **XSegLite**（XSeg 未选中） |
| B4 | 启动遮罩编辑器 | 通过 | `#mdState`：**未启动 → 运行中 · :8000** |
| C1 | 参数页跳转 / 卡片 / 行数 | 通过 | 点「参数」后 `.page.on`=**p-train-cfg**；`#cfgBox .card`=**8**（模型架构/数据集/训练/优化器/GAN/补丁/数据增强/冻结层）；`.cfgrow`=**56** |
| C2 | 标签右边缘对齐 | 通过 | 54 个 `.cfgrow .cl` 的 `right` 全为 **410.7px**（去重 1 个值，min=max=410.7，无 0 值） |
| C3 | 键名正确 | 通过 | 源码 69 个 `k:"…"` 键：`lr_total_steps` ✓ / `lr_anneal` ✗；`edge_enhance_power` ✓ / `edge_enhance` ✗；`pia_cycle_power` ✓ / `pia_cycle` ✗；`d_mask_dims` 以只读行 `["d_mask_dims","d_mask_dims"]` 存在，`mask_dims` **不存在** |
| C4 | >4 项必须 select | 通过 | 人脸类型 = **SELECT 5 项**；颜色迁移 = **SELECT 11 项**；偏色算法 = **SELECT 5 项**；对照：训练精度 seg 2 项、优化器 seg 3 项（≤4 用 seg 正确） |
| C5 | 四个按钮都有反应 | 通过 | 继续训练→p-train-cfg；参数→p-train-cfg（8 卡）；导出→toast「导出 DFM / 384wf · 导出流程还没接后端」；删除→`#confirmBox` display=**grid**、标题「删除模型」、点取消后 display=**none** |
| D1 | 控制台无 error | 通过 | **0**（载入 + 12 页切换 + 合成/遮罩/参数全部交互之后）；三重采集均为空 |
| D2 | spellcheck | 通过 | input 总数 **57**，`spellcheck===false` **57**，违规 **0** |
| D3 | 12 页逐页切换 | 通过 | 每页停 300ms，逐页新增 error 全为 **0**，累计 **0**；每页 `.page.on` 均落在预期 id（p-home…p-tool-angle） |

结论：**19/19 清单项通过，无阻断项**。

## 2 附加发现（不在清单内，但会影响交付质量）

| 编号 | 项目 | 结果 | 实测值 |
| --- | --- | --- | --- |
| X1 | 训练参数「补齐片段」是死代码 | 缺陷 | `app.js:1963` 的 `if (window.TRAIN_EXTRA_GROUPS) CFG_GROUPS.push(...)` 先执行，而 `window.TRAIN_EXTRA_GROUPS = EXTRA_GROUPS` 在 `app.js:3509` 才赋值 → 永远拿不到。源码 69 个 `k:` 键，实际渲染 47 个参数键（+7 只读 +2 路径 = 56 行），差额 **17 个键 / 7 组**（masked_training、blur_out_mask、prioritize_mouth_eyes、backup_interval、vgg_perceptual_power、mouth_scale、random_src_flip、random_dst_flip、uniform_yaw_distribution、freeze_inter…、pretrain_dataset_path、device、preview_grade_alg）**永不出现**。`train-params.css` 已并入 `app.css:2696+`（`[data-part="train-extra"]`），但没有任何 DOM 用到 |
| X2 | 时间轴是写死的 300 帧演示数据 | 缺陷 | `MergeTimeline.init()` 除 `DOMContentLoaded → init(300)` 外无任何调用（`app.js:2863` 注释承诺「宿主随后调用会覆盖」，实际没有）。`#mtlTotal` 恒 300；内容宽 600px 而 `.mtl-stack`/轨道实际宽 **1124.7px**（`style.width=600px` 被 `min-width:100%` 拉伸），右侧 **524.7px（46.6%）空白**；刻度尺只有 2 个数字（0、250）；点 60% 与 80% 都夹到 299 帧 |
| X3 | 模型文件下拉误报「未检测到模型文件」 | 缺陷 | 默认选项「（未检测到模型文件）」，但工作区里 `workspace\model\XSegLite\xseglite.onnx` **存在（19,025,269 B）**；选项是写死的 3 条，切到 XSeg 也不会换列表 |
| X4 | 右栏 3 项 seg 折行 | 缺陷 | 262px 右栏内 `遮罩来源`、`锐化` 的 `.seg` 高 **58.1px**（正常 31px），折成两行；`prd×dst/prd/dst`、`总是重检/沿用 DFL` 仍为单行 31px |
| X5 | 「继续训练」与「参数」同义 | 缺陷 | 两个按钮都绑 `openCfg(m)` → 都跳参数页；`openRun(m)`（训练页）没有任何按钮调用 |
| X6 | `pia_cycle_power` 归组错误 | 瑕疵 | 它被放在「优化器参数」组末尾（`app.js:1928`），语义上属 PIA 分节 |

## 3 必须修的清单（按严重程度）

**P1 阻断：无。** 清单 19 项全部通过，没有会导致不可用或数据错误的问题。

**P2 交付前应修（4 项）**

1. **X1 训练参数补齐死代码**——最该先修：`TRAIN_EXTRA_GROUPS` 的合并发生在赋值之前，17 个参数 / 7 组永远不渲染，等于这段交付物完全没生效。
   修法：把合并语句移到该段 IIFE 之后，或把补丁段整体前移到 `CFG_GROUPS` 定义之前；也可在 `openCfg()` 内先 `if (window.TRAIN_EXTRA_GROUPS && !CFG_GROUPS.__x)` 合并。
2. **X2 时间轴死数据**——`#mtlTotal` 恒 300、47% 空白、刻度尺只有 2 个数字、点右半区一律到末帧。
   修法：导入视频后调用 `MergeTimeline.init(state.totalFrames, state.fps)`；同时把 `.mtl-stack` 宽度设为 `max(内容宽, 容器宽)`，让刻度尺铺满可视宽度。
3. **X3 模型文件下拉误报**——真实存在 `xseglite.onnx` 却显示「未检测到模型文件」。
   修法：进页面时扫 `workspace\model\XSeg` `\XSegLite` 下的 `.onnx`/`.engine`，按当前模型重建选项并默认选中第一个。
4. **X4 右栏 seg 折行**——`遮罩来源`/`锐化` 两行高 58.1px 明显错位。
   修法：三选项改用短名（DFM / XSeg / Lite、关 / box / gauss），或给 `.mgrow .seg` 加 `flex-wrap:nowrap` + 压缩内边距。

**P3 建议修（2 项）**

5. **X5 继续训练应进训练页**：`run` 按钮改绑 `openRun(m)`，与「参数」区分开。
6. **X6 `pia_cycle_power` 移到 PIA 分组**，别留在优化器组末尾。

## 4 备注（非缺陷）

- 遮罩页「应用遮罩 / 导出 ONNX」只弹 toast、不产生真实命令，与 `mask-draw-notes.md` 的「纯前端演示」声明一致，清单只要求控件存在，判通过。
- 时间轴演示数据本身能撑住 A2–A5 的所有判定（4 轨、播放头、帧号变化、不撑高页面），问题只在「数据来源」这一层。
- 复核用 Electron 渲染器在本机 GPU 下会 10 秒左右崩溃（`render-process-gone: crashed`），加 `--disable-gpu --disable-gpu-compositing --no-sandbox` 后稳定；这只影响复核环境，不影响应用本身。
