# 遮罩绘制（#p-tool-mask）增量补全说明

对照基准：`D:\MyProject\DeepFaceLab-PyTorch\ui\components\page_maskprocessor\page_maskprocessor.py`（368 行，已全文读完）
4.1 现状：`4.1\app\web\index.html` 第 198–219 行，`#p-tool-mask` **已有可用骨架**，不是空壳。

结论先说：**现有结构方向正确，不推倒；缺 5 个控件 + 1 处默认值不对，其余无需补充。**

---

## 一、4.1 现有元素（index.html:198–219）

| 元素 | 4.1 现有实现 | 结论 |
|---|---|---|
| 标题 / 副标题 | `.title` 遮罩绘制 + `.sub` 说明 | 4.0 用 `setTitle("遮罩绘制")`，对齐 ✅ |
| 数据集路径 | `.inp.mono.pathinp` + `.pathval`，值 `workspace\data_src\` | 行在，**值要改**（见二-⑥） |
| 遮罩模型 | `.seg`：`XSeg`(on) / `XSegLite` | 2 项 → 分段控件合规 ✅；默认项与 4.0 相反（见二-⑦） |
| 服务端口 | `<span class="kbd">8000</span>` 只读 | 与 `MaskProcessor/__main__.py:25` 的 `port = 8000` 一致 ✅ |
| 启动遮罩编辑器 | `.btn.pri` | ✅（4.0 叫「启动 MaskProcessor」，语义相同） |
| 在浏览器打开 | `.mini` | 4.1 自己加的，4.0 没有，但有用，保留 ✅ |

---

## 二、4.0 有、4.1 缺的元素

| # | 元素 | 4.0 出处 | 4.0 的真实取值 / 行为 |
|---|---|---|---|
| ① | 模型文件（下拉） | `page_maskprocessor.py:171-177`、`_refresh_xseg_file_list()` ,270-286 | 扫 `workspace/model/XSegLite/` 下的 `.onnx` / `.engine`；默认选**第一个 .engine**，没有再选第一个 .onnx；空目录显示 `（未检测到模型文件）`；tooltip：`.engine = TRT BF16 (最快)` / `.onnx = ONNX` |
| ② | 分辨率 | `:179-186` | 自由文本，`placeholder="默认 256"`，默认值 `"256"` |
| ③ | 反转遮罩 | `:196-203` | `addItems(["否", "是"])`，默认「否」；「是」→ 命令追加 `--invert` |
| ④ | 应用遮罩（按钮） | `:205-207` → `_on_apply_xseg()` | `python -m DataAugmenter -i "<人脸集>" -m <模型> -r <分辨率> [--invert] [--model-file <文件>] [--trt]` |
| ⑤ | 导出 XSeg → ONNX / 导出 XSegLite → ONNX | `:124-130` → `_on_export_onnx()` | `models/Model_XSeg/export_XSeg.py` / `models/Model_XSegLite/export_XSegLite.py` |
| ⑥ | 人脸集路径默认值应为 `…/data_src/aligned` | `:148` | `setText(project_root/"workspace"/"data_src"/"aligned")`；DataAugmenter 只处理已对齐 DFLJPG（`XSegAugmenter._save_mask_fast` 需要 APP15 chunk），指到 `data_src\` 会扫到未对齐原图 |
| ⑦ | 模型类型默认项 | `:163-164` | `addItems(["XSegLite", "XSeg"])` + `setCurrentText("XSegLite")`，**默认 XSegLite**；4.1 现在默认 XSeg |

另：4.1 这页在 `app.js` 里**没有任何绑定**（grep `tool-mask` 只有导航映射），所以现有 `.seg` 点了不切换、「启动遮罩编辑器」点了没反应 —— 这也是要补 JS 的原因。

---

## 三、建议怎么补（对应三个片段文件）

1. `mask-draw.html` ① → 插进第一张 `.card`，位置在「遮罩模型」行之后、「服务端口」行之前：**模型文件 / 分辨率 / 反转遮罩** 三行，沿用页面既有 `.dbgrow` 行（不另造 grid，保持与同页一致）。
2. `mask-draw.html` ② → 整块替换「操作」卡：两个原按钮加 `id`，`.cardhd` 的 `.hint` 改成状态位 `未启动`，并在第二行补 **应用遮罩** 主按钮（4.0 里它在「反转遮罩」旁边，这里挪进操作卡，一行只留一个主按钮）。
3. `mask-draw.html` ③ → 页面末尾新增一张 **导出 ONNX** 卡（两个 `.mini` 按钮，`title` 里带真实脚本路径）。
4. `mask-draw.html` ④ → 只改数据集路径那一处 `value`：`workspace\data_src\` → `workspace\data_src\aligned`。
5. `mask-draw.css` → 仅 2 条：`.dbgrow > .mgsel` / `.dbgrow > .mgtiny` 拉到本页 33px 控件高度（`.mgsel` 基础样式只有 26px）。其余 `.card/.cardhd/.dbgrow/.seg/.mini/.dbgacts/.pathinp` 全部复用，**不新增也不覆盖任何既有类**。
6. `mask-draw.js` → `window.MaskDraw = { init, buildApplyCmd, params }`：seg 点击切 `on`（幂等，`dataset.mdReady` 防重复绑）、启动按钮翻状态文字、应用遮罩按 4.0 的拼接顺序生成命令并 toast 出来（对应 `show_command_notification`）、导出按钮给脚本路径。纯前端演示，不拉起进程。
7. ⑦ 默认模型类型：若要与 4.0 完全一致，把现有那行 `<b class="on">XSeg</b><b>XSegLite</b>` 改成 `<b>XSeg</b><b class="on">XSegLite</b>`。属一个 class 的位置调整，未放进片段，由主智能体定。

---

## 四、刻意**没有**加的东西（源码里没有）

- **画布 / 笔刷大小 / 硬度 / 撤销栈 / 结果统计 / 进度条**：`page_maskprocessor.py` 里完全没有这些 —— 4.0 这页只负责**拉起 MaskProcessor 的本地 Web 服务**，真正绘图在打开的浏览器页面里（`MaskProcessor/api/routes_mask.py`、`routes_xseg.py` 提供 REST）。不造。
- **「端口被占用自动顺延」**：`MaskProcessor/__main__.py:25` 是 `port = 8000` 写死，`uvicorn.run(...)` 直接绑定，没有顺延逻辑，也没有 try/except 重试。不造。
- **可编辑端口输入框**：4.0 有 `SiLabeledLineEdit` 并存进 `webui_settings.json` 的 `maskprocessor_port` 键，但 `_on_launch()` 拼的 cmd 只有 `{sys.executable} -m MaskProcessor`，**端口并没有传给子进程**（`:344-357`）。所以 4.1 现在的只读 `8000` 反而比 4.0 更接近真实行为 —— 不回填输入框，仅在说明里点明。
- **分辨率 seg(256\|512)**：`-r` 是 `type=int, default=256` 的自由整数（`DataAugmenter/__main__.py:58-63`），页面侧也是文本输入（`:181-183`），没有枚举。用 `.mgtiny` 数字框，不硬凑分段控件。（256 / 512 只出现在 CLI epilog 示例里。）
- **workers(-w) / --trt-batch**：CLI 有，但 4.0 的页面不暴露，不越界加。

---

## 五、核实到的参数名清单（全部来自源码，无编造）

- 启动服务：`python -m MaskProcessor`，`host = "127.0.0.1"`，`port = 8000`，`webbrowser.open("http://127.0.0.1:8000")`（`MaskProcessor/__main__.py:24-28`）
- 端口设置键：`maskprocessor_port`（`page_maskprocessor.py:91-98`）
- 应用遮罩：`-i` / `--input`、`-m` / `--model-type`（choices `XSeg` \| `XSegLite`）、`-r` / `--resolution`（int，默认 256）、`--invert`、`--model-file`、`--trt`（`DataAugmenter/__main__.py:45-93`）
- 页面侧命令拼接顺序：`DataAugmenter -i "<path>" -m <model> -r <res> [--invert] [--model-file <f>] [--trt]`（`page_maskprocessor.py:315-320`），`--trt` 仅在 `model_file.endswith(".engine")` 时追加（`:311-313`）
- 模型目录：`workspace/model/XSegLite/`，后缀 `.onnx` / `.engine`（`:273-277`）
- 模型文件 XSegLite 期望名：`xseglite.onnx`（`DataAugmenter/XSegAugmenter.py:229`）；XSeg：`XSeg.onnx`（`:228`）
- 导出脚本：`models/Model_XSeg/export_XSeg.py`、`models/Model_XSegLite/export_XSegLite.py`（`:222-223`）
- 人脸集默认路径：`<project_root>/workspace/data_src/aligned`（`:148`）
- 反转遮罩取值：`否` / `是`；模型类型取值：`XSegLite` / `XSeg`（`:163`、`:200`）
