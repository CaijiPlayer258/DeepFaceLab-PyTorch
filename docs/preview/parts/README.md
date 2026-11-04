# _parts/ 是什么

早期把单页前端拆成小文件时的草稿区。**构建不读这里，运行时也不读这里** ——
`app.py` 只挂 `web/` 为静态目录，`index.html` 也只引 `js/app.js` + `css/app.css`。
留着是因为里面有几份材料在别处没有副本。

## 实测状态（用 200 字窗口逐字比对，不是靠文件名猜）

| 文件 | 结论 |
|---|---|
| `merge-tracks.js` | 已逐字内联进 `js/app.js`（3/3 段命中） |
| `train-params.js` | 已逐字内联进 `js/app.js`（3/3） |
| `merge-pages.js` | 基本内联进 `js/app.js`（2/3，剩下一段是旧文件头注释） |
| `mask-draw.css` / `merge-pages.css` / `merge-tracks.css` / `train-params.css` | 都已内联进 `css/app.css`（各 3/3） |
| `merge-pages.html` / `merge-tracks.html` | 部分内容进了 `index.html`（1/3），**不完整** |
| `mask-draw.html` / `mask-draw.js` | 在 `app.js`/`app.css`/`index.html`/`mask-editor/index.html` 里**都找不到**逐字副本 |

所以：**8 个是过期重复件**（3 个 js + 5 个 css），其余 3 个性状不明或另有所属，
删之前要先确认。文件头里那句"勿改 app.js"是当年的施工说明，现在反过来才对 ——
真正生效的是 `js/app.js`，这里改了不会有任何效果。

## 三份 .md 有独有内容

- `DELIVERY.md` —— 交付记录：每个模块改了什么、实测数字、还差什么
- `audit.md` —— 全站对齐审计的 28 条清单
- `review-report.md` / `self-review.md` / `mask-draw-notes.md` —— 复核与遮罩绘制笔记

## 清理计划

见 `docs/05-RESTRUCTURE-PLAN.md` 的 S6：前端拆模块时，先把 `app.js` 按这几个
边界真正拆开（那时这里的草稿才有参考价值），再整目录删掉。
