# 4.1 前后端参数协议

> 这份文档只讲**参数怎么从前端走到磁盘**。接口实现见 `app/server/routes_models.py`
> 与 `routes_train.py`。

## 原则（先定规矩，再改代码）

1. **前端只负责收集值**：把页面上的输入读成一个 JSON，发出去。
   前端不判断合法性、不补默认值、不"猜"哪些字段该发。
2. **后端是唯一的裁判**：类型收敛、范围夹取、结构参数拒绝，全在后端做。
3. **后端返回"实际生效的 options"**：界面按返回渲染，就不会出现"界面显示 1024、
   磁盘里是 640"这种事。
4. **结构性参数只在创建时给**：分辨率 / 架构 / 各维度 / face_type 一旦建好就不能改
   （权重形状由它们决定）。它们走 `spec`，走进 `options` 会被拒绝。

## 端点

| 方法 | 路径 | 请求 | 返回 |
|---|---|---|---|
| GET  | /api/model_spec | — | 结构可选项（archi / 子分支 / 算子 / 脸型 / 维度上限）+ 显卡推荐值 |
| POST | /api/models/create | `{name, spec, options}` | `{ok, name, dat, options(生效), clamped, preset, note}` |
| POST | /api/models/{name}/config | `{options}` | `{ok, changed, hot, need_restart, hot_applied, hot_error}` |
| POST | /api/train/start | `{name, cpu_only?}` | `{ok, name, running}` |
| POST | /api/train/stop · /pause · /resume · /save | `{}` | `{ok, ...}` |
| GET  | /api/models · /api/models/{n}/config · /stats · /api/train/status · /preview | — | 读取用 |

### 创建

```json
POST /api/models/create
{
  "name": "OMG2",
  "spec": {
    "archi_display": "DF",        // DF / LIAE / DF Lite / LIAE Lite
    "subarchi": "-udt",           // -u / -ud / -ut / -udt / -d / -dt / -t
    "kernel_label": "dw5",        // 算子（Lite 架构才有意义）
    "face_type": "wf",            // h / mf / f / wf / head
    "resolution": 640,
    "ae_dims": 512, "e_dims": 128, "d_dims": 128, "d_mask_dims": 32
  },
  "options": {                    // 训练参数，和 /config 完全同构
    "batch_size": 1,
    "pretrain": true,
    "pretrain_dataset_path": "workspace/data_dst/aligned",
    "use_bf16": "bf16",
    "gradient_checkpointing": true,
    "max_backups": 7
  }
}
```

- `spec` / `options` 都可省略（省略即用预设默认值）。
- 老的平铺写法（`archi_display` 直接放顶层）仍然兼容。
- 返回的 `options` 是**实际写进 dat 的全部键值**；`clamped` 列出被范围修正的项
  （例如 `{"resolution":{"requested":4096,"used":1536,...}}`）。

### 保存

```json
POST /api/models/{name}/config
{ "options": { "batch_size": 2, "lr": 0.0001 } }
```

- 只传要改的键；空串 == 不改这一项。
- 结构性键出现在 `options` 里 → 400。
- 训练中改的项若能热推，会转发给 4.0 的 `/update-model-options`（`hot`），
  否则下次启动生效（`need_restart`）。

## 值的约定

| 前端送来 | 后端怎么处理 |
|---|---|
| `"1"` / `"0"` 给 bool 项 | 认 `1/true/yes/on/y/开/bf16` 为真，其余为假 |
| `"0.00005"` 给 float 项 | `float()`，失败则该键报 400 |
| `"640"` 给 int 项 | `int(round(float()))` |
| `""` | 忽略（不改） |
| 结构性键写进 options | 丢弃并 400 |

实现集中在 `_clean_options(raw, base)`：create 与 config 共用同一份。


## 训练控制（同一套规矩）

| 方法 | 路径 | 请求 | 说明 |
|---|---|---|---|
| POST | /api/train/start | `{name, options?, cpu_only?}` | `options` 可选：给了就先写进 dat 再开训，一次请求完成「保存 + 启动」 |
| POST | /api/train/stop · /pause · /resume · /save | `{name}` | `name` 是**当前在训的模型名**；和实际在训的不一致 → 409，避免界面切到别的模型后误停 |
| GET  | /api/train/status | — | 进度 / loss / 备份 / ETA 的唯一来源 |
| GET  | /api/train/preview?wait=N | — | 阻塞最多 N 秒等新预览，返回 4 行 × col_spec 列 |

### 热更新（训练中改参数）

4.1 把 `/api/models/{name}/config` 里 `hot` 那几项转发给 4.0 的
`POST /update-model-options`（带密码），4.0 存到 `_model_pending`，
Trainer 每个迭代取用一次。**值已经是 `_clean_options` 收敛过的原生类型**
（bool/int/float），4.0 那边直接赋给 `model.options` 即可。

约束：`crash_threshold` / `max_backups` / `backup_interval` 只有在模型原本就有该键时才热推
（否则界面默认值会悄悄改掉训练行为）。这三个键现在由 create 统一写入，所以新模型都能热改。

## 还没做（技术债）

- 参数行目前是**前端手写**的 `CFG_GROUPS`（50 行：键、标签、类型、默认值、说明）。
  理想形态是后端 `/api/model_spec` 直接返回 schema，前端按 schema 渲染，
  新增参数只改后端一处。现在两边要各改一次，容易漏。
- 前端仍保留 `collectCfgPatch(all)` 的两套模式：
    - `all=false`（保存/开训）：只发"用户真的改过"的行，避免把界面默认值写进模型；
    - `all=true`（新建）：全量发，因为此时模型还不存在，页面上的值就是用户要的。
  这套"基线/dirty"判断是历史包袱，等 schema 化之后可以整体删掉。