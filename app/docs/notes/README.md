# app/docs/notes —— 调研笔记与实测记录

编号文档（`app/docs/01…32`）写「现在是什么、为什么这么设计」；这个目录放**过程材料**：
审计清单、性能实测、复核记录、历史实验。都不参与构建，删了不影响运行，但里面是唯一的原始数据。

## 索引

| 文件 | 内容 |
|---|---|
| `audit.md` | 全站对齐审计的 28 条清单（哪条真、哪条假、依据在哪） |
| `summary-and-cfg-audit.md` | 参数表 ↔ 配置的逐项核对 |
| `cfg-roundtrip-audit.md` | 配置读写往返（写死一个值再读回来还等不等） |
| `DELIVERY.md` | 交付记录：每个模块改了什么、实测数字、还差什么 |
| `review-report.md` / `self-review.md` | 两轮复核报告（含推翻自己的结论） |
| `mask-draw-notes.md` | 遮罩绘制页与 4.0 MaskProcessor 的逐项对照 |
| `aligned-jpeg-444.md` | 对齐图存 4:4:4 JPEG 的取舍与验收 |
| `sess_pool.md` + `sess_sweep_*.json` | session 池标定：扫描台、原始数据、调度设计 |
| `export_memory.md` | 导出/合成的显存峰值拆解 |
| `experiments/` | 各轮性能实验的文字结论（解码腿 / 检测腿 / DFM 图 / seamless / 坏帧降级 / 编码器 / 并发 / GPU 时间线），见其中的 README |

## 读之前先知道：文中引用的 4.0 源码在哪

这些笔记写的时候是照着 4.0 的源码逐行核对的，正文里出现的
`MaskProcessor/`、`DataAugmenter/`、`MergeStudio/api/`、`ui/components/`、`legacy/qt-ui/`
这类路径，**源码本身已经归档**，现在都在 `app/docs/legacy-reference/` 下（对应关系见那个目录的
README）。路径没逐条改写，因为它们是「当时的证据」，改了反而看不出核对的到底是哪一版。

## 附：前端拆分草稿的实测状态

这些 `.js` / `.css` / `.html` 草稿现在在 `app/docs/legacy-reference/parts/`（**构建不读、运行时也不读** ——
`app.py` 只挂 `web/` 为静态目录，`index.html` 也只引 `js/app.js` + `css/app.css`）。
留着是因为有几份材料在别处没有副本。状态是用 200 字窗口逐字比对出来的，不是靠文件名猜：

| 文件 | 结论 |
|---|---|
| `merge-tracks.js` | 已逐字内联进 `js/app.js`（3/3 段命中） |
| `train-params.js` | 已逐字内联进 `js/app.js`（3/3） |
| `merge-pages.js` | 基本内联进 `js/app.js`（2/3，剩下一段是旧文件头注释） |
| `mask-draw.css` / `merge-pages.css` / `merge-tracks.css` / `train-params.css` | 都已内联进 `css/app.css`（各 3/3） |
| `merge-pages.html` / `merge-tracks.html` | 部分内容进了 `index.html`（1/3），**不完整** |
| `mask-draw.html` / `mask-draw.js` | 在 `app.js`/`app.css`/`index.html`/`mask-editor/index.html` 里**都找不到**逐字副本 |

所以：8 个是过期重复件（3 个 js + 5 个 css），其余 3 个性状不明或另有所属，删之前要先确认。
文件头里那句「勿改 app.js」是当年的施工说明，现在反过来才对 —— 真正生效的是 `js/app.js`。

**清理计划**：见 `app/docs/archive/4.0/05-RESTRUCTURE-PLAN.md` 的 S6 —— 前端按这几个边界真正
拆开之后，那批草稿才有参考价值，再整目录删掉。
