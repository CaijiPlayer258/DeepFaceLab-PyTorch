# 4.1 前端参数行 · 对齐审计报告

审计范围：`app/web/index.html`（760 行）、`app/web/css/app.css`（2139 行）；参数表单另由 `app/web/js/app.js` 的 `CFG_GROUPS`（1860–1928 行）渲染。
基准：设计 token 与「行布局 = grid 两列 `68px minmax(0,1fr)`、gap 10px、align-items center」「≤4 项 seg，>4 项 select」。

---

## 一、布局类问题（左边缘对不齐 / 控件位置飘 / 标签折行）

| 文件 | 选择器或行号 | 问题 | 建议改法 |
|---|---|---|---|
| app.css | `.cfgrow` L1061 | 用 `display:flex`，不是 grid。全站唯一合规的行是 `.mgrow`（L2055 grid 68px+1fr），训练参数页因此与合成页左边缘对不齐 | 改成 `display:grid;grid-template-columns:68px minmax(0,1fr);gap:10px;align-items:center`，`padding:7px 0` 保留 |
| app.css | `.cfgrow .cl` L1063 | `width:130px` 固定标签列，比合成页的 68px 宽一倍；且没有 `white-space:nowrap` | 去掉固定 width；加 `white-space:nowrap;overflow:hidden;text-overflow:ellipsis` |
| app.css | `.cfgrow .cd` L1064 | `flex:1` 让「说明」列吃掉所有剩余宽度，控件 `.cc` 被推到行尾，右边缘随说明文字长度浮动（同一张卡里控件左右不齐） | 说明列与控件同占第 2 列：`.cd{grid-area:1/2;min-width:0}`、`.cc{grid-area:1/2;justify-self:end;background:var(--panel);padding-left:12px}` |
| app.css | `.cfgrow .cl` L1063 | 长标签会折行：`lr_policy` 的「余弦退火·热重启（谨慎开启）」、`log_code_stats` 的「Inter code 分布日志」在 920px 容器里必折 | 同上一行加 `nowrap + ellipsis`；`title` 属性放全称 |
| app.css | `.dbgrow` L813 / L1109 / L1120 | 同一条规则被覆盖三次：L1109 `justify-content:space-between` → L1120 `flex-start`；控件位置随文件修改历史反复变动，宽屏下还额外被 L1111 `margin-left:auto` 拉到行尾、L1121 又改回 12px | 只留一条：`display:grid;grid-template-columns:68px minmax(0,1fr);gap:10px;align-items:center`，删掉 1109/1111/1116/1120/1121 |
| app.css | `.dbgrow .f` L815 `width:96px` vs L1110 `min-width:88px` | 两处冲突：`width:96px` 被 `width:auto` 覆盖，只剩 `min-width:88px`，于是「UP 主 UID」「持续（毫秒）」「服务端口」三个标签宽度各不相同，控件起点跟着漂 | 统一进 grid 标签列（68px），标签不再参与宽度计算 |
| app.css | `.dbgrow .f` L822 | 被塞进 `.cardhd b, .brand .btext b …` 的 `font-weight:700` 组里，与 `.cardhd b` 同权重 —— 参数标签被当成标题加粗，视觉层级错乱 | 从该组移除 `.dbgrow .f`，单独给 `font-weight:500` |
| app.css | `.cfgrow .cc .inp` L1066 | 数字框固定 `width:150px`，而合成页数字框 `.mgtiny` 是 52–58px（L2087、L2125、L2138）。同一个「数字」控件三个宽度 | 统一 104px（够显示 6 位小数）或直接复用 `.mgtiny` 的尺寸 |
| app.css | L1116 `.dbgrow{max-width:760px}` / L1117 `.cfgrow{max-width:920px}` | L1120/L1122 已把两者改回 `none`，这两条是死规则，留着会误导后来者 | 删除 L1116–1117 |
| app.css | `.prow` L258 / `.prow.compact` L554 | 提取器页用「3 列均分网格」（L258），又被 `.prow.compact{display:flex;flex-wrap:wrap}` 覆盖成 flex（L554）。同一页面两套排法，且都不是 68px+1fr | 短参数行也走 `.cfgrow` 网格；确实要横排时用 `.prow.compact` 且明确不参与行对齐 |
| index.html | L114 `<div class="hd">` | 提取器「参数」卡头用的是 `.hd`，其余所有页都是 `.cardhd`（L201/213/227/238/250/266/287/322/355/377/426/448…）。标题字号/间距不一致 | 改成 `<div class="cardhd"><b>参数</b></div>` |
| index.html | L136 | 「帧跳跃」行内联 `style="display:flex;align-items:center;gap:8px"`，硬塞进 `.pc`（该容器本身是 `flex-direction:column`，L259） | 拆成独立 `.cfgrow`，控件列放 `.num` |
| index.html | L632 / L647 | 设置页「隐私密码」「B 站账号」是裸 `<div style="display:flex;align-items:center;gap:10px">`，完全脱离行体系，标签列宽由内容决定 | 改成 `.cfgrow` + `.cl` + `.cc` |
| index.html | L656 | 「GitHub 代理」把值写成静态文本 `<div class="inp mono">http://127.0.0.1:26561</div>`，看着是输入框但不可编辑 | 换成 `<input class="pathval">`，整行走 `.cfgpathrow` |
| app.css | `.cfgrow` L1061 vs `#dsSteps .cfgrow` L1771 vs `#p-merge .mgrow` L2050 | padding 三套值：`7px 0` / `8px 0` / `4px 0`；字号三套：`.cl` 12.5px、`.f` 11.5px、`.mgrow .f` 11.5px | 抽 `--row-py`、`--row-fs` 两个变量，三类行统一引用 |

## 二、控件类型违规（≤4 项 seg，>4 项 select）

| 文件 | 选择器或行号 | 问题 | 建议改法 |
|---|---|---|---|
| app.js | `CFG_GROUPS` L1917 `random_color_alg` | `t:"seg"` 配 5 个选项（none/direct/rct/mt/lct），>4 项却用了分段控件，窄容器里必换行 | 改 `t:"sel"`，并在 `openCfg()` 里加 `sel` 渲染分支（`<select class="mgsel">`） |
| app.js | `CFG_GROUPS` L1914 `ct_mode` | seg 只给 3 项（none/rct/lct）；4.0 的真实取值是 8 项 `none/rct/mt/lct/mkl/idt/sot/lut`（training_config_page.py:939） | 改 `sel` 并补齐 8 个选项 |
| app.js | `CFG_GROUPS` L1868 `face_type` | seg 只给 3 项（wf/f/head）；4.0 是 5 项 `wf/f/head/hf/ff`（training_config_page.py:286） | 改 `sel`，补 `hf`（midfull_face）、`ff`（full_face） |
| index.html | L128 / L132 | 提取器「质量档」2 项、`检测角度` 4 项 —— 其中 4 项正好卡在边界，可保留，但与「检测角度」同卡片里的「帧跳跃」是数字框，三种形态混排 | 保留 seg，把三行的行高/字号拉齐 |
| index.html | L502 `#mgFaceType` | 5 项用 `select` ✅ 正确 | — |
| index.html | L493 `#mgDetector` | 14 项用 `select` ✅ 正确 | — |
| app.js | `CFG_GROUPS` L1892 `lr_plateau_factor` | 4 项 seg ✅ 边界合规，但选项文本「1.0（关闭）」带全角括号，长度是其它项两倍，分段宽度不均 | 文案压成「关 / 0.5 / 0.7 / 0.3」 |
| app.js | `CFG_GROUPS` L1874 `use_bf16` | `t:"seg"` 的 `o` 是 `["bf16","fp32"]`，但 `v` 是 `"bf16"` —— `openCfg()` 一律把第 0 项设为 `.on`（L1983），一旦默认值改成 fp32 就会显示错 | 渲染时按 `v` 匹配 `.on`，而不是硬编码 `i===0` |

## 三、参数名不一致（4.1 键名 vs 4.0 真实 options）

| 文件 | 选择器或行号 | 问题 | 建议改法 |
|---|---|---|---|
| app.js | L1863 `["d_mask_dims","mask_dims"]` | 键名写成 `mask_dims`，4.0 是 `d_mask_dims`（Model_SAEHD/Model.py:44） | 改 `d_mask_dims` |
| app.js | L1873 `autobackup_hour` | 4.0 只有 `backup_interval`（按迭代数，ModelBase.py:337）；按小时自动保存不是 4.0 的参数 | 保留现值但改键为 `backup_interval`，单位「次迭代」 |
| app.js | L1891 `lr_anneal` | 4.0 的 `config_data` 键是 `lr_total_steps`（training_config_page.py:1996），模型 option 同名 | 统一为 `lr_total_steps` |
| app.js | L1919 `edge_enhance` | 4.0 是 `edge_enhance_power`（ModelBase.py:849） | 改名 |
| app.js | L1896 `pia_cycle` | 4.0 是 `pia_cycle_power`（training_config_page.py:2022） | 改名 |

> 以上第三节属于「键名对不上」，会导致回写 `_data.dat` 时静默丢参数；建议随本次合并一起改，但**不在我的片段里覆盖**，避免与并行改动冲突。

---

## 四、4.0 有、4.1 `CFG_GROUPS` 缺的训练参数

| # | 4.0 键名 | 中文标签 | 4.0 出处 | 已产出 |
|---|---|---|---|---|
| 1 | `masked_training` | 遮罩训练 | models/ModelBase.py:835；training_config_page.py:465 | train-params.js |
| 2 | `blur_out_mask` | 遮罩羽化 | models/ModelBase.py:836；training_config_page.py:475 | train-params.js |
| 3 | `prioritize_mouth_eyes`（模型 option `eyes_mouth_prio`） | 嘴眼优先 | models/ModelBase.py:837；training_config_page.py:455 | train-params.js |
| 4 | `backup_interval` | 备份间隔（迭代） | models/ModelBase.py:336–338、867；mainscripts/Trainer.py:505 | train-params.js |
| 5 | `device` | 设备 | training_config_page.py:305–377、1508–1512 | train-params.js（需 select） |
| 6 | `vgg_perceptual_power` | VGG 感知损失 | models/ModelBase.py:848；training_config_page.py:868 | train-params.js |
| 7 | `mouth_scale` | 嘴巴范围系数 | models/ModelBase.py:840；training_config_page.py:1055 | train-params.js |
| 8 | `random_src_flip` | SRC 翻转 | models/ModelBase.py:833；training_config_page.py:1087 | train-params.js |
| 9 | `random_dst_flip` | DST 翻转 | models/ModelBase.py:834；training_config_page.py:1097 | train-params.js |
| 10 | `uniform_yaw_distribution`（option `uniform_yaw`） | 均匀偏航 | models/ModelBase.py:841；training_config_page.py:1107 | train-params.js |
| 11 | `preview_grade_alg` | 合并预览调色 | models/ModelBase.py:850；training_config_page.py:1069 | train-params.js（需 select） |
| 12 | `freeze_inter` | 冻结中间层 | Model_SAEHD/Model_pytorch.py:140；training_config_page.py:1145 | train-params.js |
| 13 | `freeze_decoder_dst` | 冻结解码器 Dst | Model_SAEHD/Model_pytorch.py:144；training_config_page.py:1205 | train-params.js |
| 14 | `freeze_inter_src` | 冻结源中间层 | Model_AMP/Model_pytorch.py:56；training_config_page.py:1175 | train-params.js |
| 15 | `freeze_inter_dst` | 冻结目标中间层 | Model_AMP/Model_pytorch.py:57；training_config_page.py:1185 | train-params.js |
| 16 | `freeze_bottleneck` | 冻结瓶颈层 | Model_DeepFakeLarge/Model_pytorch.py:404；training_config_page.py:1217 | train-params.js |
| 17 | `freeze_decoderA_mask` | 冻结 A 遮罩 | Model_DeepFakeLarge/Model_pytorch.py:397；training_config_page.py:1227 | train-params.js |
| 18 | `freeze_decoderB_mask` | 冻结 B 遮罩 | Model_DeepFakeLarge/Model_pytorch.py:398；training_config_page.py:1237 | train-params.js |
| 19 | `freeze_bottleneck_AB` | 冻结共享瓶颈 | Model_LIAELarge/Model_pytorch.py:376；training_config_page.py:1249 | train-params.js |
| 20 | `freeze_bottleneck_B` | 冻结目标瓶颈 | Model_LIAELarge/Model_pytorch.py:377；training_config_page.py:1259 | train-params.js |
| 21 | `pretrain_dataset_path` | 预训练集路径（PIA） | training_config_page.py:261–274、1500 | train-params.js |

4.1 已有、无需补的：`name/resolution/archi/e_dims/ae_dims/d_dims/d_mask_dims`、`src/dst/face_type`、`batch_size/target_iter/use_bf16/gradient_checkpointing/clipgrad/models_opt_on_gpu/use_fast_generator/write_preview_history/pretrain/pretrain_single_decoder/crash_threshold/log_code_stats/max_backups`、`optimizer/lr/lr_policy/lr_cos/lr_anneal/lr_plateau_*/lr_dropout/pia_cycle`、`gan_power/gan_patch_size/gan_dims`、`true_face_power/face_style_power/bg_style_power`、`rotation_range/scale_range/t_range/random_warp/random_hsv_power/ct_mode/random_occlusion/random_noise/random_color_alg/random_color_power/edge_enhance/eyes_scale`、`freeze_encoder/freeze_decoder_mask/freeze_inter_AB/freeze_inter_B`。

---

## 五、修复优先级

1. **P0** — `.cfgrow` 改 grid 68px+1fr（影响整页齐平）；`.cfgrow .cl` 加 nowrap。
2. **P0** — `random_color_alg` / `ct_mode` / `face_type` 改 select 并补全取值。
3. **P1** — 删掉 `.dbgrow` 的 5 条互相覆盖的规则（L1109/1111/1116/1120/1121）。
4. **P1** — 提取器 `.hd` → `.cardhd`；设置页两条裸 flex 行改 `.cfgrow`。
5. **P2** — 第三节五个键名对齐；行 padding/字号抽变量。
