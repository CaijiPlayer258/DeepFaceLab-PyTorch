# 35 · 大文件审查（>1000 行）

口径：`app/` 下（不含 runtime / workspace / weights / node_modules），py/js/css/html，按行数降序。
审查日期 2026-11，共 **39 个文件 / 89802 行**。

## 一句话结论

**这些大文件里几乎没有死代码 —— 问题不是"没用"，是"一个大函数干了一整条流水线"。**
11 个自研后端文件里，未被引用的顶层函数只有 **2 个、16 行**；其余全是真活。
真正的债在**形状**：单个函数 1697 行（内部 25 个闭包）、两套并行实现、594 个函数挤在一个 app.js 里。

## 分类（39 个）

| 类别 | 数量 | 处置 |
|---|---|---|
| 自研后端（server/） | 11 | **本页重点** |
| 自研前端（web/） | 4 | app.js 是主要问题 |
| 归 4.0 引擎（vendored，只读） | 11 | 不动；但要知道它在哪 |
| 验收/台架脚本（tools/） | 4 | 不动（测试代码长是常态） |
| 归档区（legacy-reference） | 5 | 不动 |
| 生成物（noto.css / i18n 数据表） | 2 | 不动 |

## 自研后端：11 个文件

| 文件 | 行 | 顶层 def | 最大函数 | 真实问题 |
|---|---|---|---|---|
| backend_merge/pipeline_stages.py | 4190 | 89 | `_stage_do_mask` **530**（12 闭包） | 四个 `_stage_do_*` 共 1627 行，各是一整条流水线 |
| backend_merge/pipeline_fused.py | 3178 | 25 | `_fused_run` **1697**（25 闭包） | 一个函数装下整条融合流水线 |
| backend_merge/vram.py | 2890 | 88 | `_vram_plan` 354 | 相对健康（88 个函数分摊），不优先 |
| backend_merge/pipeline_project.py | 2296 | 28 | `run` 748 / `_proj_run` 725 | **两套并行实现**，开关决定走哪套 |
| backend_merge/preview.py | 2284 | 37 | `_PreviewWorker` 405 | 可接受（类） |
| backend_merge/routes.py | 2044 | 35 | `merge_export_project` 295 | 路由多而长，可拆文件 |
| backend_merge/compose.py | 1934 | 53 | `_frame_payload` 287 | **2 个未引用函数（16 行）** |
| _merge_runner.py | 1830 | 49 | `main` 315 | 子进程入口，可接受 |
| backend_mask/painter.py | 1397 | 31 | `_MaskInfer` 168 | 可接受 |
| _vision/align.py | 1286 | 49 | `align_face_68` 101 | 健康（几何都在这里，是好事） |
| backend_faces/store.py | 1147 | 61 | `normalize_index` 106 | 可接受 |

**死代码（唯一的两处）**：`compose.py:_paste_layer`（13 行）、`compose.py:compose_frame_frame_scope`（3 行）。

## 自研前端

| 文件 | 行 | 结构 |
|---|---|---|
| web/js/app.js | **15938** | **594 个函数**，一个文件装五个页面的交互 |
| web/css/app.css | 6206 | 1617 个规则块 / 489 段注释 —— 有分区注释，可接受 |
| web/index.html | 1684 | 单页多视图，视图分区清楚 |
| web/js/face_db.js | 1261 | 75 个函数，最大 136 —— 健康 |
| web/js/i18n.js | 1849 | `apply` 1188 行是**翻译数据表**，不是逻辑 —— 不算债 |

## 建议顺序（按"省下的认知成本 / 风险"排）

1. **`pipeline_project` 的两套并存**：先确认新流水（`run` → `_proj_pipe`）在真机上稳定，再删老路 `_proj_run`（725 行）。
   风险：低（删的是分支，不是能力）；收益：文件减三分之一，排障不用再问"这单走哪套"。
2. **`_fused_run` 1697 行按内部 25 个闭包的自然边界拆**：decode / detect / dfm / xseg / compose / encode / report 七段。
   风险：中（碰性能关键路径，必须改完重跑 `fused_export_accept` 与帧号/md5 基线）；收益：最大。
3. **`pipeline_stages` 的四个 `_stage_do_*` 拆成一阶段一模块**（1627 行）。
   风险：中；收益：改某一阶段时不用在 4190 行里找。
4. **app.js 按页面切**（至少把 merge 那一坨提成 `js/merge/*.js`）。
   风险：中（前端没有模块系统，得靠 `<script>` 顺序）；收益：15938 行不再是"改动必冲突"的单文件。
5. **删那 16 行死代码**（随时可做）。

## 明确不动的

| 文件 | 理由 |
|---|---|
| `engine/models/*/Model_pytorch.py`（2819/1521/1263/1117） | 4.0 vendored，训练图就是那样 |
| `engine/FacesetProcessor/{Filter,Analyzer,Sorter}.py`（2309/1916/1577） | vendored，且被训练页用着 |
| `engine/WebUI/page_trainer.py`（1758） | vendored，内联 HTML 是它的形态 |
| `engine/MergeStudio/core/export_pipeline.py`（1431） | vendored，项目导出的真身 |
| `engine/xlib/face/FLandmarks2D.py`（1355） | vendored |
| `tools/bench/fused_pipeline_bench.py` 等 4 个 | 台架/验收，长是正常的 |
| `docs/legacy-reference/**` 5 个 | 已归档，不进运行路径 |
| `css/noto.css`、`js/i18n.js` 的数据段 | 生成/数据 |

## 审查方法的局限（诚实说明）

- 死代码判定 = 「该标识符在全仓（排除归档区）出现的次数」。
  对 `run` / `main` 这种通用名几乎无效 —— `pipeline_project.run` 因为名字太常见，
  被这条规则判成"活着"，实际它确实是活的，但**判据本身不可靠**。所以"0 个死代码"应读作"没有明显死代码"。
- 函数长度按 AST 的 `end_lineno` 算（py）；JS 用缩进启发式，`mgStageEmpty(12371)`
  这种数字是启发式失效（该函数之后没有同级函数），不代表它真有 1.2 万行。
