# 训练摘要（summary.txt）与训练参数保存 —— 独立复核

对象：`D:/MyProject/DeepFaceLab-PyTorch-4.1`（未 commit / 未 push）。
测试全部在 `%TEMP%/dfls/` 里跑：3 份 `_data.dat` 样本、一个只指向临时工作区的后端
（`DFL_WORKSPACE=%TEMP%/dfls/ws`）、以及用 **jsdom 加载的真实页面**
（`app/web/index.html` + 真实 `app.js`，不是照着源码复刻一套逻辑）。
两个问题共用一套方法：**改前先量化差集，改后用同一个断言证明差集为 0。**

---

# A) summary 与真实训练参数不一致

## A.1 summary 在哪生成

| 位置 | 作用 |
|---|---|
| `app/engine/models/ModelBase.py:1468` `get_summary_text()` | 摘要正文（唯一生成处；改前是 1291–1620 行） |
| `app/engine/models/ModelBase.py:1291` `get_summary_path()` | 文件名 `<模型名>_summary.txt` |
| `app/engine/models/ModelBase.py:962` `save()` | `Path(...).write_text(self.get_summary_text())` —— 唯一落盘处 |
| `app/engine/models/ModelBase.py:354` `on_initialize()` | `io.log_info(self.get_summary_text())`，开训时也往控制台打一份 |
| `app/engine/models/ModelBase.py:1032` `autobackup()` | 自动备份把 summary.txt 一起复制 |

服务器端**没有任何接口读写 summary**（`grep summary app/server` 为空），所以
"把模型交给别人"时对方看到的那份 txt 就是这里生成的。
另外 `write_text()` 原来没给 `encoding`：Windows 上按 locale（cp936）写，摘要里只要有一个
cp936 编不出的字符就抛 UnicodeEncodeError，**整个 save()（权重 + dat）一起失败**，
换台机器读还是乱码。已固定 `encoding='utf-8'`。

## A.2 改前：它到底写了哪些字段

改前是 4 张**手写白名单**（`arch_opts` / `aug_opts` / `patch_opts` / `train_opts`），
共 51 个条目，其中 3 个是 dat 里永远不存在的死键：

- `use_gradient_checkpointing` —— 真键是 `gradient_checkpointing`（拼错，永远打印不出来）
- `lr_scheduler` / `lr_cos_period` —— 只有 AMP 有，SAEHD 的 dat 里没有
- `pia_cycle_power` —— 只在 PIA 下显示，其它架构被 `_is_pia` 过滤掉

## A.3 差集（dat 有 / summary 没有）

| 模型 | dat 键数 | 改前能打印 | dat 有 / summary 无 | summary 有 / dat 无 |
|---|---|---|---|---|
| mA（真实 `_smoketrain`，liae-ud） | 58 | 50 | **12**：`eager_mode` `freeze_decoder_dst` `freeze_decoder_mask` `freeze_encoder` `freeze_inter` `freeze_inter_AB` `freeze_inter_B` `gradient_checkpointing` `lr_cos` `lr_plateau_factor` `lr_plateau_min_ratio` `lr_plateau_patience` | **4**：`lr_cos_period` `lr_scheduler` `pia_cycle_power` `use_gradient_checkpointing` |
| mB（SAEHD df-ud，dat 里塞满参数页能写的键） | 72 | 50 | **26**：上列 12 个 + `src` `dst` `pretrain_dataset_path` `device` `target_batch` `accum_steps` `log_code_stats` `lr_warmup_steps` `nan_guard` `random_flip` `rotation_range` `scale_range` `t_range` `write_train_log` | 同上 4 个 |
| mC（XSeg） | 19 | 50 | **8**：`gradient_checkpointing` `lr_cos` `lr_plateau_factor` `lr_plateau_min_ratio` `lr_plateau_patience` `lr_warmup_steps` `random_flip` `write_train_log` | **39**：SAEHD 专属键（archi / 各 dims / gan_* / *_style_power / random_* …）全在白名单里，XSeg 的 dat 一个都没有 |

结论：**freeze_* 全系列、lr_cos / lr_plateau_* / lr_warmup_steps、gradient_checkpointing、
src / dst / device / target_batch、nan_guard / write_train_log / log_code_stats、
rotation_range / scale_range / t_range** 这些真实参数在摘要里**完全看不到**；
而 XSeg 这种非 SAEHD 模型的摘要里，50 行有 39 行是"这个模型根本没有的键"。

## A.4 改了什么

重写 `get_summary_text()`（330 行 → 419 行）：

1. **不再有白名单**：打印集合 = `self.options` 的键集合本身。认识的键进对应分组，
   不认识的按前缀归组（`freeze_*` / `random_*` / `gan_*` / `lr*` / `xseg*`），
   实在猜不出来进「其他」——**无论哪条路都一定会被打印**。
   `SUMMARY_LABELS` / `SUMMARY_GROUP_OF` 只决定排版（中文名、排哪个小标题下），
   漏一条最多少个中文名，绝不会少一行参数。
2. **结构参数与训练参数分区**：`resolution / face_type / archi / e_dims / ae_dims /
   d_dims / d_mask_dims` 单独一区并标注「创建后不可改」；其余按 数据集 / 训练设置 /
   优化器 / 精度与显存 / GAN / 损失与补丁 / 数据增强 / 冻结层 / 预训练 / 备份与日志 /
   遮罩-XSeg / 设备 / 其他 分组，组内按键名排序（不依赖手写顺序，也不依赖 dat 键序）。
3. **数值精度与 dat 一致**：`repr()` 输出（`5e-05` 还是 `5e-05`）；numpy 标量先收敛
   （`np.int64(16)` → `16`）；bool 打 `True/False`；**空串打 `''`**（否则
   「`pretrain_dataset_path: `」和「这一项没打印」长得一模一样）。
4. `options_show_override`（预训练时引擎按"实际生效值"覆盖几项）不再混进全量列表，
   单独一块「实际生效值（引擎覆盖，非 dat 条目）」，保证主列表键集合严格等于 dat。
5. 顺手修两处"摘要说假话"：`compute_mode` 原来读 `getattr(self,'use_bf16')`
   （模型对象没这个属性，永远打印 FP32）→ 读 `options['use_bf16']`；预估显存读
   `use_gradient_checkpointing`（不存在的键）→ `gradient_checkpointing`。

## A.5 断言（改后，用真实 `get_summary_text()` 跑 3 个模型）

**A-1 键集合对比（白名单 = 空，0 项排除）**

| 模型 | dat options 键数 | summary 里解析出的键 | dat 有/summary 无 | summary 有/dat 无 | 白名单 |
|---|---|---|---|---|---|
| mA | 58 | 58 | 无 | 无 | 0 |
| mB | 72 | 72 | 无 | 无 | 0 |
| mC | 19 | 19 | 无 | 无 | 0 |

> 白名单是空的，不需要逐项理由：`self.options` 里的每一项都是训练参数，没有一项
> 该被藏起来。模型名 / 当前迭代 / 模块参数量 / 真显卡型号在独立区块里，不属于
> options，解析时按区块排除（它们也没有对应的 dat 键）。

**A-2 10 个参数值逐字比对**

| 参数 | dat 里的值（pickle repr） | mA 摘要 | mB 摘要 | mC 摘要 |
|---|---|---|---|---|
| `lr` | `5e-05` | `5e-05` | `5e-05` | `0.0001` |
| `gan_patch_size` | `np.int64(16)` / `np.int64(48)` | `16` | `48` | — |
| `resolution` | `np.int64(128)` / `np.int64(256)` | `128` | `256` | — |
| `src` | `'workspace\\data_src\\'` | — | `workspace\data_src\` | — |
| `pretrain_dataset_path` | `''` | — | `''` | — |
| `crash_threshold` | `4.0` / `3.5` / `0.0` | `4.0` | `3.5` | `0.0` |
| `random_warp` | `True` | `True` | `True` | — |
| `optimizer` | `'adabelief'` / `'lion'` | `adabelief` | `lion` | — |
| `device` | `'0'` | — | `0` | — |
| `lr_plateau_factor` | `1.0` / `0.5` | `1.0` | `0.5` | `1.0` |

（— = 该模型的 dat 里没有这个键，摘要本来就不该有。）

**A-3 未知键也必须打印**（证明"没有白名单"这件事本身）：

```
options = {resolution:256, archi:"df-ud", batch_size:8,
           foo_bar_baz:3.5, weird_key:True, mask_thing:"x", lr:5e-05}
==-- · 遮罩/XSeg ---------------------------------------------------------==
==  mask_thing: x                    |                                    ==
==-- · 其他 --------------------------------------------------------------==
==  foo_bar_baz: 3.5                 | weird_key: True                    ==
```

---

# B) 「参数不保存」复核（不信旧报告，重新独立验证）

## B.1 方法

1. **环境**：`runtime/server` 起 `app/serve.py`，`DFL_WORKSPACE` 指到
   `%TEMP%/dfls/ws`（3 份 dat 副本）。仓库 `workspace/` 一个字节没动。
2. **页面**：jsdom 打开真实 `http://127.0.0.1:8760/`，`runScripts:"dangerously"` +
   `resources:"usable"`，跑的是**没改过的 index.html + app.js**。
   只补了浏览器环境缺口：`matchMedia` / `ResizeObserver` / `Element.animate` /
   `scrollIntoView` / canvas（`@napi-rs/canvas`）/ `window.fetch` 桥到 Node fetch。
3. **一次「刷新页面」** = 重新调 `window.openCfg(model)`：清空 `#cfgBox`、按
   `CFG_GROUPS` 重画、`TrainExtra.mount()`、再 `GET /config` 回填；参数页没有别处缓存值。
   另外每轮结束还做了**两次整页重载**（新建 jsdom）抽样复核。
4. **每个参数**：还原 dat → 打开页面读回显 → 改成另一个合法值 → `collectCfgPatch()`
   → `POST /config` → python 桥直接解 `_data.dat`（不信接口返回）→ 再刷新 → 读回 UI。
5. **判据**：改前回显 == dat、patch 里有这个键、dat 里真的变了、刷新后 UI == dat、
   同一次保存没有顺带改别的键（`extraChanged` 全表为 0）。

## B.2 改前：12 项失败

| 参数 | 模型 | 现象 |
|---|---|---|
| `device` | mA | 下拉框显示 `0`，dat 是 `（无此键）`；改完保存后刷新又变回 `0` |
| `eager_mode` | mA | 参数页没有控件（dat 里有 `False`） |
| `gan_patch_size` | mA | 界面显示 `32`，dat 是 `16`（np.int64 被 /config 过滤掉） |
| `use_bf16` | mA | 判据问题，不算 bug：分段 token `fp32` ↔ dat `False`，实际写入/回读都对 |
| `device` | mB | 下拉框显示 `0`，dat 是 `0`；改完保存后刷新又变回 `0` |
| `eager_mode` | mB | 参数页没有控件（dat 里有 `False`） |
| `gan_patch_size` | mB | 界面显示 `32`，dat 是 `48`（np.int64 被 /config 过滤掉） |
| `lr_warmup_steps` | mB | 参数页没有控件（dat 里有 `100`） |
| `nan_guard` | mB | 参数页没有控件（dat 里有 `True`） |
| `random_flip` | mB | 参数页没有控件（dat 里有 `True`） |
| `use_bf16` | mB | 判据问题，不算 bug：分段 token `fp32` ↔ dat `False`，实际写入/回读都对 |
| `write_train_log` | mB | 参数页没有控件（dat 里有 `True`） |

其中 `use_bf16` 两条是**判据写错**（分段 token `fp32` ↔ dat `False`），代码本身对；
其余是 3 个真 bug。

### Bug B-1：`/api/models/{name}/config` 把 numpy 标量整个过滤掉

返回值过滤条件是 `isinstance(v, (int,float,str,bool))`，而 `np.int64` **不是** `int`
（`np.float64` 是 `float` 子类，所以只有整数漏了）。老 dat 里 `resolution` 和
`gan_patch_size` 就是 `np.int64` → 这两个键**根本不在 /config 的返回里** → 参数页
拿不到值就退回写死的默认值：**GAN 块大小 dat 里是 16，页面上显示 32**。
改完保存能写进去，一刷新又看到 32。

修：新增 `_scalar_options()`，先 `_jsonable()` 收敛 numpy 标量再过滤；`/config` 与
`/api/models/{name}` 两个口都走它。顺带给 `_clean_options` 加了 np.bool_ /
np.integer / np.floating 的类型收敛（原来 `cur` 是 np.int64 时 isinstance 分支全落空，
界面上填的数字会**当字符串**存进 dat）。

### Bug B-2：「设备」下拉框的回显永远是写死的默认值

`TrainExtra.mount()` 里的「设备」卡是 `openCfg()` 之后 **80ms** 由 `setTimeout` 挂上来的
（要先找宿主卡片，找不到才自建），而 `/config` 的回填在那之前就跑完了 —— 这一行
**从来没有被回填过**：dat 里 `device="auto"`，页面显示 `0`；改成 `auto` 保存成功，
一刷新又显示 `0`。

修：回填循环抽成 `fillCfgRows(opts, onlyNew)` 挂到 `window.__fillCfgRows`；
`openCfg` 拿到 `/config` 后记一份 `window.__cfgOpts`；`TrainExtra` 的 80ms 延时块
插完卡片后调一次 `fillCfgRows(opts, true)`（只填还没有基线的行）再 `syncCfgBaseline()`。
另外把这一行的默认值从 `"0"` 改成 `"auto"` —— dat 里没有 `device` 时引擎就是自动挑卡，
显示 `0` 是骗人的。

### Bug B-3：4 个真实参数在参数页上根本没有控件

`lr_warmup_steps`（优化器真的在读）、`nan_guard`、`write_train_log`、`eager_mode`
都是引擎读、也写在 dat 里的参数，参数页一行都没有 → 只能手改 dat。

修：`CFG_GROUPS` 加 4 行 —— `lr_warmup_steps` 进「优化器参数」，
`nan_guard` / `write_train_log` / `eager_mode` 进「训练参数」。

## B.3 改后：完整参数表

### mA（真实 `_smoketrain` dat，liae-ud；73 行 = 58 个 dat 键 ∪ 65 个控件）

| # | 参数 | 控件 | 能写进 dat | 能读回 UI | 是否进 summary |
|---|---|---|---|---|---|
| 1 | `src` | input | ✓ `workspace\data_src\x` | ✓ `workspace\data_src\x` | —（原 dat 无此键） |
| 2 | `dst` | input | ✓ `workspace\data_dst\x` | ✓ `workspace\data_dst\x` | —（原 dat 无此键） |
| 3 | `pretrain_dataset_path` | input | ✓ `x` | ✓ `x` | —（原 dat 无此键） |
| 4 | `face_type` | select | — | — | ✓ |
| 5 | `device` | select | ✓ `cpu` | ✓ `cpu` | —（原 dat 无此键） |
| 6 | `batch_size` | input | ✓ `9` | ✓ `9` | ✓ |
| 7 | `target_batch` | input | ✓ `1` | ✓ `1` | —（原 dat 无此键） |
| 8 | `target_iter` | input | ✓ `1` | ✓ `1` | ✓ |
| 9 | `use_bf16` | 分段 | ✓ `False` | ✓ `fp32` | ✓ |
| 10 | `gradient_checkpointing` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 11 | `clipgrad` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 12 | `models_opt_on_gpu` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 13 | `use_fast_generator` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 14 | `write_preview_history` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 15 | `pretrain` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 16 | `pretrain_single_decoder` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 17 | `crash_threshold` | input | ✓ `5.0` | ✓ `5`（dat `5.0`，同一数值） | ✓ |
| 18 | `log_code_stats` | input | ✓ `1` | ✓ `1` | —（原 dat 无此键） |
| 19 | `max_backups` | input | ✓ `25` | ✓ `25` | ✓ |
| 20 | `nan_guard` | 开关 | ✓ `True` | ✓ 开 | —（原 dat 无此键） |
| 21 | `write_train_log` | 开关 | ✓ `True` | ✓ 开 | —（原 dat 无此键） |
| 22 | `eager_mode` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 23 | `masked_training` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 24 | `blur_out_mask` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 25 | `eyes_mouth_prio` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 26 | `backup_interval` | input | ✓ `501` | ✓ `501` | ✓ |
| 27 | `optimizer` | 分段 | ✓ `adam` | ✓ `adam` | ✓ |
| 28 | `lr` | input | ✓ `0.0001` | ✓ `0.0001` | ✓ |
| 29 | `lr_policy` | 分段 | ✓ `CosineAnnealingLR` | ✓ `CosineAnnealingLR` | —（原 dat 无此键） |
| 30 | `lr_cos` | input | ✓ `1` | ✓ `1` | ✓ |
| 31 | `lr_total_steps` | input | ✓ `1` | ✓ `1` | —（原 dat 无此键） |
| 32 | `lr_plateau_factor` | 分段 | ✓ `0.5` | ✓ `0.5` | ✓ |
| 33 | `lr_plateau_patience` | input | ✓ `1` | ✓ `1` | ✓ |
| 34 | `lr_plateau_min_ratio` | input | ✓ `0.2` | ✓ `0.2` | ✓ |
| 35 | `lr_warmup_steps` | input | ✓ `1` | ✓ `1` | —（原 dat 无此键） |
| 36 | `lr_dropout` | 分段 | ✓ `y` | ✓ `y` | ✓ |
| 37 | `pia_cycle_power` | input | ✓ `2` | ✓ `2` | —（原 dat 无此键） |
| 38 | `gan_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 39 | `gan_patch_size` | input | ✓ `17` | ✓ `17` | ✓ |
| 40 | `gan_dims` | input | ✓ `17` | ✓ `17` | ✓ |
| 41 | `true_face_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 42 | `face_style_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 43 | `bg_style_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 44 | `vgg_perceptual_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 45 | `rotation_range` | input | ✓ `11` | ✓ `11` | —（原 dat 无此键） |
| 46 | `scale_range` | input | ✓ `0.3` | ✓ `0.3` | —（原 dat 无此键） |
| 47 | `t_range` | input | ✓ `0.1` | ✓ `0.1` | —（原 dat 无此键） |
| 48 | `random_warp` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 49 | `random_hsv_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 50 | `ct_mode` | select | ✓ `rct` | ✓ `rct` | ✓ |
| 51 | `random_occlusion` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 52 | `random_noise` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 53 | `random_color_alg` | select | ✓ `direct` | ✓ `direct` | ✓ |
| 54 | `random_color_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 55 | `edge_enhance_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 56 | `eyes_scale` | input | ✓ `2.0` | ✓ `2`（dat `2.0`，同一数值） | ✓ |
| 57 | `mouth_scale` | input | ✓ `2.0` | ✓ `2`（dat `2.0`，同一数值） | ✓ |
| 58 | `random_src_flip` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 59 | `random_dst_flip` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 60 | `uniform_yaw` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 61 | `preview_grade_alg` | select | ✓ `none` | ✓ `none` | ✓ |
| 62 | `freeze_encoder` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 63 | `freeze_decoder_mask` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 64 | `freeze_inter_AB` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 65 | `freeze_inter_B` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 66 | `ae_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 67 | `archi` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 68 | `d_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 69 | `d_mask_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 70 | `e_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 71 | `freeze_decoder_dst` | 无（该架构没有这一层） | — | — | ✓ |
| 72 | `freeze_inter` | 无（该架构没有这一层） | — | — | ✓ |
| 73 | `resolution` | 无（结构参数，只在只读卡） | — | — | ✓ |

### mB（SAEHD df-ud；75 行 = 72 个 dat 键 ∪ 65 个控件）

| # | 参数 | 控件 | 能写进 dat | 能读回 UI | 是否进 summary |
|---|---|---|---|---|---|
| 1 | `src` | input | ✓ `workspace\data_src\x` | ✓ `workspace\data_src\x` | ✓ |
| 2 | `dst` | input | ✓ `workspace\data_dst\x` | ✓ `workspace\data_dst\x` | ✓ |
| 3 | `pretrain_dataset_path` | input | ✓ `x` | ✓ `x` | ✓ |
| 4 | `face_type` | select | — | — | ✓ |
| 5 | `device` | select | ✓ `auto` | ✓ `auto` | ✓ |
| 6 | `batch_size` | input | ✓ `13` | ✓ `13` | ✓ |
| 7 | `target_batch` | input | ✓ `25` | ✓ `25` | ✓ |
| 8 | `target_iter` | input | ✓ `500001` | ✓ `500001` | ✓ |
| 9 | `use_bf16` | 分段 | ✓ `False` | ✓ `fp32` | ✓ |
| 10 | `gradient_checkpointing` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 11 | `clipgrad` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 12 | `models_opt_on_gpu` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 13 | `use_fast_generator` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 14 | `write_preview_history` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 15 | `pretrain` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 16 | `pretrain_single_decoder` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 17 | `crash_threshold` | input | ✓ `7.0` | ✓ `7`（dat `7.0`，同一数值） | ✓ |
| 18 | `log_code_stats` | input | ✓ `501` | ✓ `501` | ✓ |
| 19 | `max_backups` | input | ✓ `8` | ✓ `8` | ✓ |
| 20 | `nan_guard` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 21 | `write_train_log` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 22 | `eager_mode` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 23 | `masked_training` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 24 | `blur_out_mask` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 25 | `eyes_mouth_prio` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 26 | `backup_interval` | input | ✓ `2501` | ✓ `2501` | ✓ |
| 27 | `optimizer` | 分段 | ✓ `adam` | ✓ `adam` | ✓ |
| 28 | `lr` | input | ✓ `0.0001` | ✓ `0.0001` | ✓ |
| 29 | `lr_policy` | 分段 | ✓ `CosineAnnealingLR` | ✓ `CosineAnnealingLR` | —（原 dat 无此键） |
| 30 | `lr_cos` | input | ✓ `250001` | ✓ `250001` | ✓ |
| 31 | `lr_total_steps` | input | ✓ `1` | ✓ `1` | —（原 dat 无此键） |
| 32 | `lr_plateau_factor` | 分段 | ✓ `1.0` | ✓ `1.0` | ✓ |
| 33 | `lr_plateau_patience` | input | ✓ `4` | ✓ `4` | ✓ |
| 34 | `lr_plateau_min_ratio` | input | ✓ `0.2` | ✓ `0.2` | ✓ |
| 35 | `lr_warmup_steps` | input | ✓ `101` | ✓ `101` | ✓ |
| 36 | `lr_dropout` | 分段 | ✓ `n` | ✓ `n` | ✓ |
| 37 | `pia_cycle_power` | input | ✓ `2` | ✓ `2` | —（原 dat 无此键） |
| 38 | `gan_power` | input | ✓ `0.2` | ✓ `0.2` | ✓ |
| 39 | `gan_patch_size` | input | ✓ `49` | ✓ `49` | ✓ |
| 40 | `gan_dims` | input | ✓ `33` | ✓ `33` | ✓ |
| 41 | `true_face_power` | input | ✓ `0.04` | ✓ `0.04` | ✓ |
| 42 | `face_style_power` | input | ✓ `0.002` | ✓ `0.002` | ✓ |
| 43 | `bg_style_power` | input | ✓ `1.0` | ✓ `1`（dat `1.0`，同一数值） | ✓ |
| 44 | `vgg_perceptual_power` | input | ✓ `4.0` | ✓ `4`（dat `4.0`，同一数值） | ✓ |
| 45 | `rotation_range` | input | ✓ `11.0` | ✓ `11`（dat `11.0`，同一数值） | ✓ |
| 46 | `scale_range` | input | ✓ `0.3` | ✓ `0.3` | ✓ |
| 47 | `t_range` | input | ✓ `0.1` | ✓ `0.1` | ✓ |
| 48 | `random_warp` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 49 | `random_hsv_power` | input | ✓ `0.2` | ✓ `0.2` | ✓ |
| 50 | `ct_mode` | select | ✓ `none` | ✓ `none` | ✓ |
| 51 | `random_occlusion` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 52 | `random_noise` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 53 | `random_color_alg` | select | ✓ `none` | ✓ `none` | ✓ |
| 54 | `random_color_power` | input | ✓ `0.7` | ✓ `0.7` | ✓ |
| 55 | `edge_enhance_power` | input | ✓ `3.0` | ✓ `3`（dat `3.0`，同一数值） | ✓ |
| 56 | `eyes_scale` | input | ✓ `2.5` | ✓ `2.5` | ✓ |
| 57 | `mouth_scale` | input | ✓ `1.8` | ✓ `1.8` | ✓ |
| 58 | `random_src_flip` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 59 | `random_dst_flip` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 60 | `uniform_yaw` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 61 | `preview_grade_alg` | select | ✓ `same` | ✓ `same` | ✓ |
| 62 | `freeze_encoder` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 63 | `freeze_decoder_mask` | 开关 | ✓ `True` | ✓ 开 | ✓ |
| 64 | `freeze_inter` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 65 | `freeze_decoder_dst` | 开关 | ✓ `False` | ✓ 关 | ✓ |
| 66 | `accum_steps` | 无（由 target_batch×batch_size 换算） | — | — | ✓ |
| 67 | `ae_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 68 | `archi` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 69 | `d_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 70 | `d_mask_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 71 | `e_dims` | 无（结构参数，只在只读卡） | — | — | ✓ |
| 72 | `freeze_inter_AB` | 无（该架构没有这一层） | — | — | ✓ |
| 73 | `freeze_inter_B` | 无（该架构没有这一层） | — | — | ✓ |
| 74 | `random_flip` | 无（遗留键，引擎不读） | — | — | ✓ |
| 75 | `resolution` | 无（结构参数，只在只读卡） | — | — | ✓ |

**统计**：改前 `{PASS:114, FAIL:6, NO_CONTROL:6, STRUCT_RO:14, ARCH_NA:4, DERIVED:1}`（145 行）
→ 改后 `{PASS:128, STRUCT_RO:14, ARCH_NA:4, DERIVED:1, LEGACY_UNUSED:1}`（148 行），
**失败 0**。

**表格注**：

1. 「无（结构参数，只在只读卡）」：`resolution / archi / e_dims / ae_dims / d_dims /
   d_mask_dims` 创建后不可改，参数页只把它们画在顶部只读卡里；`face_type` 是 disabled
   下拉框（值**从 dat 回显**，只是不许改）。要改这 7 项只有一条路：新建模型。
2. 标 `≈` 的是同一数值的不同写法（浏览器把 `5.0` 印成 `5`、`gan_power` 存 `1.0` 显示 `1`）；
   分段控件 `use_bf16` 的 token 是 `bf16/fp32` 而 dat 里是 `True/False`，判据按
   「第 0 项 = 开」换算，不是字符串对比。
3. 「无（该架构没有这一层）」：`freeze_inter` / `freeze_decoder_dst` 只在 df 出现，
   `freeze_inter_AB` / `freeze_inter_B` 只在 liae/amp 出现（`FREEZE_ARCH`）。
   dat 里留着的键**不被改写**，只是不给一个点了没用的开关。
4. 「无（由 target_batch × batch_size 换算）」：`accum_steps` 是派生值，改
   `batch_size` 或 `target_batch` 时由 `collectCfgPatch()` 一起写进 dat（已验证）。
5. 「无（遗留键，引擎不读）」：`random_flip`。`ModelBase.py:321` 只是把值存进
   `self.random_flip`，SAEHD 的两条支路用的是 `random_src_flip`/`random_dst_flip`
   （`Model_SAEHD:1459/1467`）；`Trainer.py:294` 那处只是把生成器的
   `sample_process_options` 读出来推给 WebUI 缓存。给它加开关就是造假旋钮
   （`autobackup_hour` 的老问题），所以**故意不加**，但摘要里照样打印它的真实值。

## B.4 交叉验证（不串值）

| 跑完 | mA 的 dat | mB 的 dat |
|---|---|---|
| mA 全量往返（73 次保存） | 每次由 pristine 还原 | **未变**（前后 SHA256 相同） |
| mB 全量往返（75 次保存） | **未变**（前后 SHA256 相同） | 每次由 pristine 还原 |

每次保存的 `changed` 里**只有目标键**（+ 必要的 `accum_steps`），`extraChanged`
全表为 0 —— "保存一次把整页默认值都推过去"的旧行为没有回来。

整页重载抽样（新建 jsdom，完整加载一遍）：

| 模型 | batch_size | random_warp | backup_interval |
|---|---|---|---|
| mA | 9 = 9 ✓ | false = False ✓ | 501 = 501 ✓ |
| mB | 13 = 13 ✓ | false = False ✓ | 2501 = 2501 ✓ |

---

# 改了哪些文件

| 文件 | 改动 |
|---|---|
| `app/engine/models/ModelBase.py` | 重写 `get_summary_text()`（全量输出 + 分组 + 分区 + 精度）；新增 `SUMMARY_LABELS` / `SUMMARY_GROUP_OF` / `SUMMARY_STRUCT_KEYS` / `SUMMARY_GROUP_ORDER` / `_summary_group_of` / `_summary_fmt_value`；`save()` 显式 `encoding='utf-8'` |
| `app/server/routes_models.py` | 新增 `_scalar_options()`（numpy 标量不再被 /config 丢掉）；`_clean_options` 增加 numpy 类型收敛 |
| `app/web/js/app.js` | `openCfg` 的回填抽成 `fillCfgRows(opts, onlyNew)` + `window.__cfgOpts`；`TrainExtra` 80ms 补挂后补填一次；device 行默认值 `"0"`→`"auto"`；新增 `lr_warmup_steps` / `nan_guard` / `write_train_log` / `eager_mode` 四行 |

# 没动的 / 已知遗留

- **`target_batch` 仍然只是"存得住"**：界面会换算成 `accum_steps` 写进 dat，但引擎里
  没有任何地方实现梯度累积（`grep accum_steps app/engine` 0 命中）。往返通，生效不是。
- **`device` 现在真的会生效**（`routes_train.py` 起训前按 dat 里的 device 校验并映射
  `CUDA_VISIBLE_DEVICES`），但只在"开始训练"那一刻读一次。
- `app/tools/parts/train-params.js` 是早期补齐片段的独立副本（index.html 不加载），
  键名还是旧的（`prioritize_mouth_eyes`），和 app.js 不同步 —— 本次没动。
