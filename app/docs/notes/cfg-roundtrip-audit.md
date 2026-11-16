# 训练参数往返审计（改了记不住）

对象：训练器「参数设置」页（`#cfgBox`，由 `app/web/js/app.js` 的 `CFG_GROUPS` +
`TrainExtra` 补齐块渲染）。往返链路：

```
读 dat → 渲染/回填 → 用户改 → collectCfgPatch() → POST /api/models/{name}/config
      → 写 *_data.dat → 重新 GET /config → 重新渲染/回填 → 比对
```

## 怎么测的

两层都是**真跑**的，不是静态推断：

1. 后端：`python app/serve.py`（venv），`DFL_WORKSPACE` 指向 `%TEMP%/cfgaudit/ws`，
   里面放两份现有模型 `_data.dat` 的副本（`auditA` / `auditB`）。仓库 `workspace/` 只读。
2. 前端：`%TEMP%/cfgaudit/audit.mjs` —— 用客户端包了一个最小 DOM，然后把 app.js 里
   **真实的** `openCfg` / `collectCfgPatch` / `syncCfgBaseline` / `CFG_VALS` /
   `filterFreezeRows` / TrainExtra 整个 IIFE 切出来执行（按括号配对切，不照抄一遍）。
   一次「页面加载」= 新建 document + 重新执行这些函数 = 真·刷新。

每个参数：重置 dat → 打开页面读初始值 → 改成另一个合法值 → 触发 dirty →
`collectCfgPatch()` → POST → GET → 重新打开页面 → 读回显 → 比对。

## 结果

- 改前：78 项 → {"STRUCTURAL": 7, "FAIL_NO_PATCH": 26, "PASS": 43, "FAIL_HTTP": 1, "NO_CONTROL": 1}
- 改后：77 项 → {"STRUCTURAL": 7, "PASS": 69, "STRUCT_RO": 1}
- 失败项：**0** —— 可编辑参数 69 项全部往返一致
- 结构性字段 8 项单列（下表「结构」行）；`autobackup_hour` 这个假参数已删除（78 行 → 77 行）
- 随机抽 10 项（seed 20240929）：src, max_backups, lr_plateau_min_ratio, pia_cycle_power, true_face_power, scale_range, random_warp, eyes_mouth_prio, freeze_decoder_dst, freeze_decoderA_mask → {"STRUCTURAL": 7, "PASS": 10}

## 改前 / 改后对照（78 项）

| # | 参数 | 分组 | 控件 | 改前 | 改后 | 修复后回显 |
|---|---|---|---|---|---|---|
| 1 | `name` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 2 | `resolution` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 3 | `archi` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 4 | `e_dims` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 5 | `ae_dims` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 6 | `d_dims` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 7 | `d_mask_dims` | 模型架构参数 | readonly-card | 结构(只读卡) | 结构(只读卡) | - |
| 8 | `src` | 数据集参数 | input | FAIL 进不了 patch | PASS | workspace\data_src\ |
| 9 | `dst` | 数据集参数 | input | FAIL 进不了 patch | PASS | workspace\data_dst\ |
| 10 | `face_type` | 数据集参数 | select | FAIL 进不了 patch | 结构(只读控件) | full_face |
| 11 | `batch_size` | 训练参数 | input | PASS | PASS | 8 |
| 12 | `target_batch` | 训练参数 | input | PASS | PASS | 0 |
| 13 | `target_iter` | 训练参数 | input | PASS | PASS | 0 |
| 14 | `autobackup_hour` | 训练参数 | input | PASS | - | - |
| 15 | `use_bf16` | 训练参数 | seg | PASS | PASS | bf16 |
| 16 | `gradient_checkpointing` | 训练参数 | sw | PASS | PASS | 0 |
| 17 | `clipgrad` | 训练参数 | sw | PASS | PASS | 0 |
| 18 | `models_opt_on_gpu` | 训练参数 | sw | FAIL 进不了 patch | PASS | 1 |
| 19 | `use_fast_generator` | 训练参数 | sw | PASS | PASS | 0 |
| 20 | `write_preview_history` | 训练参数 | sw | PASS | PASS | 0 |
| 21 | `pretrain` | 训练参数 | sw | PASS | PASS | 0 |
| 22 | `pretrain_single_decoder` | 训练参数 | sw | PASS | PASS | 0 |
| 23 | `crash_threshold` | 训练参数 | input | PASS | PASS | 4 |
| 24 | `log_code_stats` | 训练参数 | input | PASS | PASS | 0 |
| 25 | `max_backups` | 训练参数 | input | PASS | PASS | 24 |
| 26 | `optimizer` | 优化器参数 | seg | PASS | PASS | adabelief |
| 27 | `lr` | 优化器参数 | input | PASS | PASS | 0.00005 |
| 28 | `lr_policy` | 优化器参数 | seg | PASS | PASS | None |
| 29 | `lr_cos` | 优化器参数 | input | PASS | PASS | 0 |
| 30 | `lr_total_steps` | 优化器参数 | input | PASS | PASS | 0 |
| 31 | `lr_plateau_factor` | 优化器参数 | seg | FAIL 进不了 patch | PASS | 1.0 |
| 32 | `lr_plateau_patience` | 优化器参数 | input | PASS | PASS | 0 |
| 33 | `lr_plateau_min_ratio` | 优化器参数 | input | PASS | PASS | 0.1 |
| 34 | `lr_dropout` | 优化器参数 | seg | PASS | PASS | n |
| 35 | `pia_cycle_power` | 优化器参数 | input | PASS | PASS | 1.0 |
| 36 | `gan_power` | GAN 训练参数 | input | FAIL 进不了 patch | PASS | 0 |
| 37 | `gan_patch_size` | GAN 训练参数 | input | FAIL 后端 500 | PASS | 32 |
| 38 | `gan_dims` | GAN 训练参数 | input | PASS | PASS | 16 |
| 39 | `true_face_power` | 补丁参数 | input | FAIL 进不了 patch | PASS | 0 |
| 40 | `face_style_power` | 补丁参数 | input | FAIL 进不了 patch | PASS | 0 |
| 41 | `bg_style_power` | 补丁参数 | input | PASS | PASS | 0 |
| 42 | `rotation_range` | 数据增强参数 | input | PASS | PASS | 10 |
| 43 | `scale_range` | 数据增强参数 | input | PASS | PASS | 0.15 |
| 44 | `t_range` | 数据增强参数 | input | PASS | PASS | 0.05 |
| 45 | `random_warp` | 数据增强参数 | sw | PASS | PASS | 1 |
| 46 | `random_hsv_power` | 数据增强参数 | input | PASS | PASS | 0 |
| 47 | `ct_mode` | 数据增强参数 | select | PASS | PASS | none |
| 48 | `random_occlusion` | 数据增强参数 | sw | PASS | PASS | 0 |
| 49 | `random_noise` | 数据增强参数 | sw | PASS | PASS | 0 |
| 50 | `random_color_alg` | 数据增强参数 | select | PASS | PASS | none |
| 51 | `random_color_power` | 数据增强参数 | input | PASS | PASS | 0 |
| 52 | `edge_enhance_power` | 数据增强参数 | input | PASS | PASS | 0 |
| 53 | `eyes_scale` | 数据增强参数 | input | PASS | PASS | 1 |
| 54 | `freeze_encoder` | 冻结层参数 | sw | PASS | PASS | 0 |
| 55 | `freeze_decoder_mask` | 冻结层参数 | sw | PASS | PASS | 0 |
| 56 | `freeze_inter_AB` | 冻结层参数 | sw | PASS | PASS | 0 |
| 57 | `freeze_inter_B` | 冻结层参数 | sw | PASS | PASS | 0 |
| 58 | `masked_training` | 训练参数 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 1 |
| 59 | `blur_out_mask` | 训练参数 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 60 | `eyes_mouth_prio` | 训练参数 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 61 | `backup_interval` | 训练参数 · 补充(补齐) | input | FAIL 进不了 patch | PASS | 500 |
| 62 | `vgg_perceptual_power` | 补丁参数 · 补充(补齐) | input | FAIL 进不了 patch | PASS | 0 |
| 63 | `mouth_scale` | 数据增强 · 补充(补齐) | input | FAIL 进不了 patch | PASS | 1 |
| 64 | `random_src_flip` | 数据增强 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 65 | `random_dst_flip` | 数据增强 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 1 |
| 66 | `uniform_yaw` | 数据增强 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 67 | `freeze_inter` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 68 | `freeze_decoder_dst` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 69 | `freeze_inter_src` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 70 | `freeze_inter_dst` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 71 | `freeze_bottleneck` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 72 | `freeze_decoderA_mask` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 73 | `freeze_decoderB_mask` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 74 | `freeze_bottleneck_AB` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 75 | `freeze_bottleneck_B` | 冻结层 · 补充(补齐) | sw | FAIL 进不了 patch | PASS | 0 |
| 76 | `pretrain_dataset_path` | 数据集参数 · 补充(补齐) | input | PASS | PASS |  |
| 77 | `device` | 设备(补齐) | select | PASS | PASS | 0 |
| 78 | `preview_grade_alg` | 数据增强 · 补充(补齐) | select | 无控件 | PASS | same |

## 修了什么（6 处，全在现有链路上，没加存储层）

### 1. 补齐块的开关/数字/分段控件根本没有 `data-cfg`（18 个参数）

`TrainExtra.buildRow()` 只有 `path` 和 `sel` 分支写过 `dataset.cfg`，`num` / `seg` / `sw`
三个分支都没写。后果是双向的：`openCfg` 回填扫不到它（**显示的是硬编码默认值**），
`collectCfgPatch` 也不报它（**改了发不出去**）。

证据：`backup_interval` 页面显示 1000（行定义里的 `v`），dat 里是 500。

修：三个分支各补一行 `if (r.k) x.dataset.cfg = r.k;`，seg 的 token 补 `dataset.v`
（避免 i18n 翻译后按文本比不中）。`app/tools/parts/train-params.js` 同源副本同步。

覆盖：masked_training / blur_out_mask / eyes_mouth_prio / backup_interval /
vgg_perceptual_power / mouth_scale / random_src_flip / random_dst_flip / uniform_yaw /
freeze_inter / freeze_decoder_dst / freeze_inter_src / freeze_inter_dst /
freeze_bottleneck / freeze_decoderA_mask / freeze_decoderB_mask /
freeze_bottleneck_AB / freeze_bottleneck_B

### 2. `collectCfgPatch` 的 STRUCT 白名单多剔了 6 个可改参数

`src` / `dst` / `models_opt_on_gpu` / `gan_power` / `true_face_power` / `face_style_power`
被当成结构参数在客户端直接丢掉。

- `src` / `dst`：引擎 `_train_worker` 确实读 `options.src/dst` 决定数据集目录，是可改的；
- 后四个：每次训练的损失权重 / 优化器放置开关，不是「创建后不可改」。而且
  `face_style_power` 本来就在 `_HOT_OPTIONS` 里（设计上要能热改），
  却同时被列进两个「结构」名单 —— 自相矛盾。

修：前端 `STRUCT` 收缩到 7 个（resolution archi e_dims ae_dims d_dims d_mask_dims face_type），
后端 `_CFG_STRUCTURAL` 收缩到同一组 7 个。三处（前端 STRUCT / `_CFG_STRUCTURAL` /
`_STRUCTURAL_KEYS`）现在一致。

### 3. `gan_patch_size` 保存报 500（返回体里的 numpy 标量）

dat 里 `gan_patch_size` 是 `numpy.int64`，`changed[k] = {"from": opts.get(k), ...}` 把它塞进
返回体，FastAPI 序列化抛 `ValueError: 'numpy.int64' object is not iterable` → 500。
值其实已经写进 dat 了，但用户看到「保存失败」；更糟的是**同一次保存里的其它参数**也一起被判失败。

修：`routes_models.py` 加 `_jsonable()`，只在返回体上把 numpy 标量转成 Python 原生类型。

### 4. `face_type` 是可点的下拉框，但永远存不进去（假可编辑）

结构性参数：客户端 STRUCT 剔除、后端 `_clean_options` 静默 `continue`。
用户改完保存、刷新回原值，还以为是自己没点对。

修：渲染成 `disabled` 下拉框 + `data-ro="1"`（`collectCfgPatch` 见到就跳过），
`title` 说明原因。值是**从 dat 回显**的，不是空白。

顺带修了回显表：`_ALIAS` 里写的是 `wf/ff/hf`，而 4.0 引擎的短码是 `h/mf/f/wf/head`
（`Model_SAEHD:195`）。于是 `face_type="f"` 的模型在框里显示成第一项 `whole_face`。
只读行显示错的值比空白更误导，所以一并修（`f→full_face`、`h→half_face`，旧码留着容忍）。

### 5. `autobackup_hour` 是纯 UI 占位

行定义自己的说明就写着「4.0 无此项，仅为 UI 占位」。它能存进 dat（`_clean_options`
不认识的键也照写），但引擎从来不读 —— 往返「通过」也只是写了个没人看的键。

修：删掉这一行（真参数是补齐块里的 `backup_interval`）。**dat 里已存在的这个键不动**，
只是不再让人编辑，老 dat 不会被无意改写。

### 6. `preview_grade_alg` 页面上根本不存在

`TrainExtra.mount()` 按 base 名去重：「数据增强 · 补充」在 `EXTRA_GROUPS` 和
`EXTRA_SELECT_GROUPS` 里各出现一次，第二组被 `if (seen[base]) return;` 整组跳过，
于是 `preview_grade_alg` 那个 select 从来没有被画出来。

修：先按 base 归并 rows 再并卡（顺序不变，先出现的组在前；`pin` / `noheader` 跟着走），
同一张卡仍然只合并一次。`preview_grade_alg` 现在落在「数据增强参数」卡的最后一行。

### 附带：`lr_plateau_factor` 看着像「没选中」

dat 存 `1.0`（float），seg 的 token 是字符串 "1.0"，回填时 `String(b.dataset.v) === String(v)`
永远不等，四项一个都不亮。修：回填时数字按数值比。

## 结构性字段（8 项，预期不持久，已确认 UI 上不是假象）

| 参数 | 现状 | 为什么不能持久 |
|---|---|---|
| `name` | 只读卡（`CFG_GROUPS` 里 `ro:true`，`openCfg` 直接跳过整组） | 模型名就是文件名 |
| `resolution` | 同上 | 决定网络展平尺寸，改了权重全废 |
| `archi` | 同上 | 网络拓扑 |
| `e_dims` | 同上 | encoder 宽度 |
| `ae_dims` | 同上 | inter 维度 |
| `d_dims` | 同上 | decoder 宽度 |
| `d_mask_dims` | 同上 | mask 解码器宽度 |
| `face_type` | **改成 disabled 下拉框**（原来可点，是假象） | 决定对齐方式与裁剪范围 |

前 7 项在页面上只出现在顶部「模型结构」只读卡里：`openCfg` 里 `if (g.ro) return;`
连参数行都不渲染 —— 不存在可编辑假象。第 8 项以前是**可点的下拉框**，已改只读。

要改这 8 项只有一条路：新建模型（`POST /api/models/create`）。

## 验证

```
A. 以前被误判为结构的 5 项：POST 200 + GET 一致 + dat 文件里确实有
   src / gan_power / true_face_power / face_style_power / models_opt_on_gpu  -> 全 PASS
B. face_type 仍然 400（后端守住结构参数），dat 里的 face_type 没被改写
C. gan_patch_size（np.int64）改完 200，changed.from/to 都是 JSON 原生类型
D. 两模型隔离（SHA256）：
   auditA 全量往返 77 项后，auditB 的 dat 与 pristine 逐字节相同
   auditB 全量往返期间，auditA 的 dat 一字节没变
E. 随机抽 10 项（seed 20240929）刷新后断言：10/10 一致
F. node --check app/web/js/app.js / app/tools/parts/train-params.js  -> OK
   py_compile app/server/app.py app/server/routes_models.py        -> OK
```

## 已知遗留（不在这次范围，但审计时看清了）

- `device`（「设备」卡）：会写进 dat 的 `options.device`，但 `_train_worker` / 引擎
  没有任何地方读它。它属于**训练启动参数**，不是模型参数 —— 现在是「存了但没用」。
  要接的话在 `_train_worker` 起进程处按 `device` 设 `CUDA_VISIBLE_DEVICES`（或 `--cpu-only`）。
- `target_batch`：界面会换算成 `accum_steps` 写进 dat，但引擎**没有实现梯度累计**
  （`app.py` 起训时会打印提示）。往返是通的，生效不是。
- `optimizer`：能存能回显，但换优化器后老的优化器状态（`src_dst_opt.pth`）对不上，
  实际只对新建模型有意义。
- 「恢复默认」按钮的文案是错的：它只调 `openCfg()` 重读 dat，不会恢复任何默认值。
