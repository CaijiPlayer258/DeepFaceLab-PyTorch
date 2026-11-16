# 生效配置（[eff]）：把「我设的」和「它在跑的」摆在一起

> 起因：检测器那个 bug —— 界面显示 TinyMog、后端跑 retinaface_10g 跑了几十次，日志里
> 每一单都写着 `检测器 retinaface_10g thresh=0.50`。信息在，但没人能从一串流水账里
> 看出「这跟我设的不一样」。所以日志里必须有**一份可核对的生效配置**，界面也得能对照。

## 一行 [eff]

每次导出/预览**真正启动**时打一行（stdout，进度页 status 里同一份）：

```
[eff] {"route": "/api/merge/export", "kind": "fused", "at": 1794627567.405,
       "detector": "tinymog", "detector_src": "ui",
       "mask_source": "XSegLite", "mask_source_src": "inferred",
       "ct_mode": "mt", "ct_mode_src": "ui",
       "blend": "seamless", "blend_src": "ui",
       "erode": 0, "erode_src": "ui", "blur": 4, "blur_src": "ui",
       "models": ["阿証証証証-416"], "models_src": "ui",
       "faces_project": "faces_project", "faces_project_src": "ui",
       "library_enabled": true, "library_enabled_src": "ui",
       "fast_export": true, "fast_export_src": "ui",
       "frame_range": "全部", "frame_range_src": "default",
       "workers": 8, "workers_src": "default",
       "batch": 8, "batch_src": "default",
       "jpeg_quality": 100, "jpeg_quality_src": "default",
       "src_note": {"detector": "quality=high"}, "req_unused": []}
```

- **字段**：`detector / mask_source / ct_mode / blend / erode / blur / models /
  faces_project / library_enabled / fast_export / frame_range / workers / batch /
  jpeg_quality`。每个字段紧跟一个 `<字段>_src`。
- **来源**：`ui` = 请求体里明确给了（界面发的）；`default` = 请求体没有，后端默认；
  `inferred` = 后端从别的字段推的（`src_note` 里写推的人是谁，例如 `quality=high`）。
- **值来自 cfg**（真正进 worker 的那份），不是把请求体再抄一遍。
- `route` + `kind`：`fused` / `staged` / `stage:frames|faces|pred|mask|merge|encode` /
  `project`（DFL 项目导出）/ `project-modes`（4.0 MergeStudio 老项目模式）/ `preview`。
- 预览是高频入口（播放/拖动时每帧一次）：**配置没变就不重复打**（指纹不含 `at`）；
  预览的 `[req]` 也不把 `t/frame/face/idx` 算进「内容变没变」。
- 视频导出的 `[eff]` / `[eff-warn]` 还会贴进产物日志 `<成片>.log`（用户报 bug 时
  最常附的那份文件）。

## 请求体回显 [req]

```
[req] {"video": "clip12.mp4", "fused": 1, "detector": "tinymog", ...}
[req-unused] foo
```

- 路径类键（`video / project_dir / dfm / output_dir / faces_project …`）**只留文件名**；
  值本身是绝对路径的也按文件名。
- base64 / data URL 换成 `<base64 N chars>`；超长普通字符串截断到 160。
- 每次 HTTP 请求一行（`dry_run` 预检也打）；预览按内容去重。
- 没有任何入口消费的键（不在 `_eff.KNOWN_KEYS` 里）单独一行 `[req-unused] <key>` ——
  「界面发了、后端不认识」和检测器那个 bug 是同一类对不上账。

## 三类 [eff-warn]（同时进 status.warnings[]）

| 类型 | 触发条件 | 样例 |
|---|---|---|
| missing | 请求体缺字段，后端默认值又和界面默认值不一致 | `detector 未在请求体中出现，按 default=retinaface_10g 执行（界面可能显示 TinyMog）` |
| inferred | 值从别的字段推出来，且与界面默认值不一致 | `detector 由 quality=high 推断为 retinaface_10g，请求体未指定 detector` |
| rewritten | 请求体给了值，后端改写（显存预算降档等） | `workers 由 2 改为 1（显存预算降档），请求体写的是 2` |

- 界面默认值在 `server/_eff.py: UI_DEFAULTS`（和 index.html 控件初始值对齐）。
- 规则上就不该报警的字段用 `quiet`（例：`mask_source` ← `mask_mode` 是契约；
  单个阶段没有「快速导出」这个旋钮）。
- 每条 warning 结构：`{"code": "eff-missing|eff-inferred|eff-rewritten", "field": "detector",
  "level": "warn", "text": "…"}`。

## status

`/api/merge/export/status`、`/api/merge/stage/status`、`/api/merge/export-project/status`、
`/api/merge/preview/status` 都多三个键：

| 键 | 内容 |
|---|---|
| `eff` | 同一份生效配置（含 `warnings` / `src_note` / `req_unused`） |
| `eff_warnings` | 上面三类警告的结构化列表 |
| `warnings` | 纯文本列表（项目导出的老字段也在这里追加） |

## 前端

进度页（`#p-merge .mgview[data-view=progress]`）：

- `#mgpEffWarn`：可关闭横幅（复用 `.alert.warn.loadbanner`），文案 = `eff_warnings[].text`；
  换一单在 `mgpReset()` 里复位（关掉不该延续到下一单）。**这个必须留着** ——
  它是「界面设了、后端没按它跑」的唯一提示。
- `#mgpEffOpen`：右下角的文字入口「查看生效配置」（没有 `eff` 回显时是 `disabled`）。
- `#mgpEffDrawer`：右侧抽屉（`.mgeff-drawer` / `.mgeff-sheet`），里面是那 14 个字段 +
  来源标签（界面 / 推断 / 默认），`src_note` 放 `title`。**点开才渲染**（`mgpEffPaint(true)`）——
  以前那块常驻面板（旧的 `#mgpEff`）已删掉：太丑、也占版面，字段展示以后归「控制台输出」界面。
  三个出口：`#mgpEffClose` / 点遮罩 / Esc。
- 渲染入口：`window.mgpEffRender(eff, effWarnings)`，由 `mgpFusedPaint` /
  `mgpStatusPaint` 在每次轮询时调用；抽屉关着时它只记数据、不动 DOM。
- 六阶段是**横排**（`#mgpSteps > .mgps-n`）：fused 下六段并行，六个节点可以同时是 on，
  只有真的 done 才点亮（`data-lit` 时间戳 + `.lit`，去重见 `mgpLight()`）。

## 怎么加字段

1. `_eff.FIELDS` 加名字；需要认请求体的加到 `BODY_KEYS`；界面默认值加到 `UI_DEFAULTS` /
   `UI_LABELS`；新入口键加到 `KNOWN_KEYS`（否则会被 `[req-unused]` 点名）。
2. `routes_merge._eff_values_of` 从 cfg 里取**真正生效**的值（不要重新解析请求体）。
3. 前端 `MGP_EFF_LABEL` 加一行。

## 验收

```
runtime\server\Scripts\python.exe app\tools\eff_accept.py
```

覆盖：正常路径（detector=tinymog → src=ui、无 warn）、老 bug 复现（不带 detector +
quality=high → [eff-warn] + status.warnings + 前端横幅截图）、`[req-unused]`、脱敏
（路径/base64）、六阶段链的 [eff]。截图写到 `%TEMP%\dfl_eff_accept\banner.png`。

进度页 / 抽屉 / 点亮动效另有一份 UI 验收（最快的一个，先跑它）：

```
runtime\server\Scripts\python.exe app\tools\accept\progress_lit_accept.py
```
