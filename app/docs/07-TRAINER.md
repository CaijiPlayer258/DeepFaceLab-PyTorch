# 训练器（Trainer）

4.1 不自己实现训练，它把 4.0 的训练**当成一个真进程**拉起来，然后代理它的进度、预览和
控制指令。这一页写清楚这条链路，以及三个必须踩对的地方。

## 一、链路

```
模型列表(/api/models)
   └─ 点「训练」→ POST /api/train/start
        └─ 4.0 python + app/server/_dfl_train.py train …
             ├─ stdout → 解析进度行 → /api/train/status
             └─ WebUI(127.0.0.1:6790) → /request-preview + /preview → /api/train/preview
   └─ 参数页 → GET/POST /api/models/{name}/config （写进 <name>_SAEHD_data.dat 的 options）
   └─ 训练页 → 暂停 / 继续 / 立即保存 / 停止
```

训练进程用的解释器是 **4.0 的 python**（有 torch + onnx），不是 4.1 的。启动参数：

```
_dfl_train.py train
  --training-data-src-dir <options.src 或 workspace/data_src/aligned>
  --training-data-dst-dir <options.dst 或 workspace/data_dst/aligned>
  --model-dir <workspace>/model --model SAEHD --force-model-name <名字>
  --no-preview --silent-start
  --webui-port 6790 --webui-password dfl41-webui
```

`_dfl_train.py` 只做两件事：禁掉 4.0 自己弹浏览器；保留它的 WebUI（4.1 的预览要靠它）。

## 二、三个坑

### 1. 进度行不能用「数字 + it」抓

4.0 每轮输出一行：

```
[08:42:26][#002012][0061ms][lr:5.00e-05][1.0574][1.8756]
 时间       迭代     每轮耗时   学习率        src_loss dst_loss
```

而加载阶段的 tqdm 进度条长这样：

```
Initializing models: 100%|##########| 5/5 [00:00<00:00, 73.25it/s]
```

以前用 `(\d+)\s*(?:it|iter)` 抓，抓到的是 `25`、`73` 这种 tqdm 数字 —— 界面上
迭代数在乱跳、速度永远不准。现在整行匹配：

```python
ITER_RE = re.compile(r"^\[[\d:]+\]\[#(\d+)\]\[(\d+)ms\]\[lr:([0-9.eE+-]+)\]"
                     r"\[([0-9.]+)\]\[([0-9.]+)\]")
```

速度直接由 `ms` 反算（`1000/ms`），再指数平滑；loss 和 lr 顺手取出来给曲线用。

### 2. 「停止」不是杀进程，是请它自己保存后退出

4.0 的 WebUI 有两个控制接口，都要密码：

| 接口 | 作用 |
|---|---|
| `POST /request-save` | 下一个迭代后落盘，训练继续 |
| `POST /request-quit` | 保存 + 退出（Trainer 主循环里 `is_close_requested()` 那支） |

4.1 的 `/api/train/stop` 先调 `/request-quit`（失败重试 6 次，暂停刚恢复时
WebUI 线程可能还没醒），等进程自己退出；最多 90 秒，还没退才按进程树强杀。
密码是启动时用 `--webui-password` 指定的 `dfl41-webui`，不是 4.0 的默认 `caiji`。

手动停止后进程退出码通常非 0（强杀时），所以 worker 里要看 `stopping` 标记：
用户按的停止不该显示成「训练失败」。

### 3. 暂停：4.0 没有这个语义，用挂起实现

`POST /api/train/pause` 用 `psutil.Process.suspend()` 把训练进程（含子进程）挂起：
迭代停在原地、显存不释放；`resume` 反向恢复。挂起期间 WebUI 线程也停着，
所以**挂起状态下发控制指令会超时** —— stop 里会先 resume 再发 quit。

### 4. 停止前必须先恢复挂起

挂起是整进程冻结，WebUI 线程也冻着。先发 `/request-quit` 只会一路超时，白等
90 秒最后还是要强杀（实测 122 秒）。正确顺序：**先 resume，再 quit** —— 这样
「暂停中点停止」13 秒就能保存退出。

### 4. 别留下孤儿训练进程

4.1 一退出（关窗口、Ctrl+C、崩了），训练进程如果没被带走，就会一直在后台占着显存，
用户下次开 4.1 还会看到「已经在训练」。所以：

- `shutdown` 钩子：还在训练就先 `/request-quit`，10 秒不退就杀树；
- 每个 `/api/train/stop` 最后都确认进程真的没了；
- **启动清扫**：上一次 4.1 被强杀（任务管理器 / 崩溃）时训练进程会活下来，占着
  6790 端口 —— 下一次训练就永远拿不到 WebUI（保存、停止、预览全部失灵）。开服时
  扫一遍命令行带 `_dfl_train.py` 的进程，只杀**父进程已经不在**的那些，避免误杀
  另一个 4.1 实例正在用的训练。

## 二点五、训练页怎么读

**一条状态，不刷日志。** 4.0 的 stdout 里九成是 loss —— 而 loss 已经有曲线和数字了。
所以页面上只保留一行「现在在干什么」：

| 状态 | 什么时候 | 副标题 |
|---|---|---|
| 加载模型… | 进程起来了、还没出第一个 iter | 已等 12s |
| 训练中 | 正常跑 | 已跑 3m 20s |
| 正在保存… | 点了「立即保存」，等落盘 | 下一个迭代后落盘 |
| 已暂停 | 挂起中 | 停在 12,345 iter |
| 正在停止… | 等它保存完退出 | 等它保存完再退出 |
| 已停止 | 正常结束或被停 | 停在 12,345 iter |
| 训练失败 | 进程非 0 退出 | 错误原因（截断 60 字） |

圆点颜色：灰=未运行 · 绿=训练中 · 黄=暂停 · 红=失败 · 呼吸动画=正在忙（加载/保存/停止）。

**没在训练时显示什么。** 运行条读模型里保存的迭代数（不是 0），速度显示「—」，
loss 显示最后一次的值；状态条写「已停止 · 停在 12,345 iter」。预览区放模型里存的那张
样本图（`<模型>_preview.jpg`，通常是一行五张：src / src-pred / dst / dst-pred / 换脸），
标注「上次保存的样本 · 12,345 iter」—— 点「刷新」才去问 4.0 要新的。

### 状态圆点

左上角那颗点就是当前状态，不用看文字：

| 圆点 | 徽章文案 | 什么时候 |
|---|---|---|
| 🔵 蓝（呼吸） | 启动中 / 连接中 / 加载中 | 点下训练到进程起来（启动中）；进页面还没拿到状态（连接中）；进程起了但还没吐第一行迭代（加载中） |
| 🟢 绿（呼吸） | 训练中 | 正常训练（iter 在涨） |
| 🟡 黄 | 已暂停 | 挂起中 |
| 🔴 红（快闪） | 正在停止 / 训练失败 | 保存并退出中；或进程非 0 退出 |
| ⚪ 灰 | 已停止 | 没在训练 |

圆点、徽章、状态条由同一个 `trainState()` 决定，三处不会各说各话。

**状态条只在徽章说不清的时候出现**：加载模型 / 正在保存 / 正在停止 / 训练失败。
训练中、已暂停、已停止都不显示 —— 徽章已经写着状态，迭代数和「已跑多久」上面那行都有，
再写一遍就是同一件事说两遍。（以前这里会同时出现「训练中」「已跑 46s」，和徽章、进度行重复。）

### loss 曲线

两条线：**src 用主题色**（品牌蓝 `#5d6ccf`，暗色下提亮到 `#7f8cf0`），
**dst 用它的对位暖色**（`#d98b2b` / 暗色 `#e8b04a`）—— 一眼能分清谁是谁，
又不会和「暂停的黄」「失败的红」撞车。

每条线都画两遍：0.8px、35% 透明的原始值 + 1.8px 的 8 点滑动平均（看得清趋势）。
训练中每 2 秒按 `/api/train/status` 的 `history` 重画一次（最多 400 点），
没在训练时才退回模型里存的历史。标题会写清是「实时」还是「模型里存的」。

### 备份与「下次自动」

4.0 的自动备份落在 `<模型>_autobackups/01、02、03 …` —— **每个编号是一个目录**，
里面才是权重文件（以前 4.1 只数目录里的散文件，所以永远显示 `0 / 3 份`）。

- 计数 = 编号目录个数（没有再退回数文件），上限取 `max_backups`
- 「下次自动」按**真实的上次备份迭代数**算：`backup_interval - (iter - last_backup_iter)`。
  4.0 是在 `iter - last_backup_iter >= interval` 时备份，而 `last_backup_iter` 每次启动会
  重置成当时的 iter —— 所以不能用 `iter % interval` 猜（重启后锚点会偏）。
- `last_backup_iter` 从 4.0 的 `正在创建备份...` 日志行里抓；抓不到（还没备份过）时
  才退回取模，并在数字后面标「（约）」。
- 子进程的 stdout 现在强制 `PYTHONIOENCODING=utf-8`：之前按 UTF-8 解 GBK，
  中文全是乱码，这行永远匹配不到、日志尾巴也没法看。

### 右侧资源

| 行 | 来源 |
|---|---|
| GPU 显存 + 利用率 | `nvidia-smi`（每次请求现查） |
| CPU | `psutil.cpu_percent(None)`，即上次轮询到现在的平均占用 |
| 内存 | `psutil.virtual_memory()` 的 used / total 与百分比 |
| 磁盘 / 学习率 / 上次保存 / 下次自动 / 备份 | `/api/models/{name}/stats` |

**预览一次只放一张。** 4.0 会发四张整图（原图预览 / 遮罩下 / 原始输入 / 合并预览，
每张都是 `min(4, batch_size)` 行 × 5 列的网格）。页面上是「原图 · 遮罩 · 输入 · 合并」
四个按钮 + 一张大图，切换只换本地缓存，不联网。

### 训练中改参数（热更新）

参数分两类，参数页在训练中会给能热改的行挂一个「热」标记：

| 能热改（改完下一个迭代就生效） | 必须重启训练 |
|---|---|
| 数据增强：随机扭曲 / 随机遮挡 / 随机噪点 / 随机偏色 / 随机 HSV / 刚性形变（旋转·缩放·偏移范围） | 分辨率、架构、各维度、face_type |
| 眼睛缩放 / 嘴巴范围 / 边缘增强 / 遮罩羽化（blur_out_mask） | 冻结分支开关 |
| 颜色迁移模式 ct_mode、翻转、uniform_yaw | batch_size、优化器、预训练 |
| 损失权重：face_style / bg_style、学习率 lr / 余弦周期 / lr_dropout | GAN 强度（要重建判别器）、真脸强度（同理） |
| 崩溃阈值、备份间隔、最大备份数、masked_training、blur_out_mask、eyes_mouth_prio | 冻结分支开关 |

实现分两头：
- **4.0**（`mainscripts/Trainer.py`）：`_HOT_KEYS` 扩展 + `_apply_model_options` 里按字段改
  生成器的 `output_sample_types[0]`（每路生成器的第一项就是喂进网络的输入图，
  目标项不动；ct_mode 例外，输入和目标一起改），顺手清掉 loader 缓存。
- **4.1**：参数页「保存」（不重启训练）→ `POST /api/models/{name}/config` →
  后端把可热改的部分 POST 到 4.0 的 `/update-model-options`，返回 `hot / need_restart / hot_applied`，
  UI 据此说明「哪几项即时生效、哪几项要重启」。

实测（训练中改 `random_warp` 与 `random_noise`）4.0 日志：

```
[WebUI] Model option random_warp: True -> False
[WebUI] random_warp -> False（已更新 2 个样本项）
[WebUI] Model option random_noise: False -> True
[WebUI] random_noise -> True（已更新 2 个样本项）
```

**自动拉第一次。** 训练跑出第一行迭代（`iter_seen` 变真）时，页面会自己拉一次预览，
把「上次保存的样本」换掉 —— 4.0 要跑出第一张才给图，一次可能落空，所以最多重试 3 次、每次隔 12 秒。
之后仍然只有点「刷新」才更新。

**刷新是手动的。** 自动刷新会在你正盯着遮罩边缘时把图换掉，所以只在
「进入页面」和「点刷新」时各拉一次。标题右侧显示这张图的来源：
`原图预览 · 12,345 iter · 8s 前`，避免把旧图当成最新的。

## 三、接口表

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/models` | 模型列表（含 `state`: 训练中 / 已暂停 / 已停止） |
| POST | `/api/models/create` | 建模型（写一份 iter=1 的 `_data.dat`） |
| GET/POST | `/api/models/{name}/config` | 读/写训练参数（结构性字段只读） |
| POST | `/api/models/{name}/delete` | 删模型全部文件（训练中拒绝） |
| POST | `/api/models/{name}/export` | 导出 .dfm（后台） |
| GET | `/api/models/export/status` | 导出进度 |
| POST | `/api/train/start` | 拉起 4.0 训练进程 |
| GET | `/api/train/status` | iter / it_s / lr / loss / history / paused / saving … |
| GET | `/api/train/preview` | 代理 4.0 WebUI 的实时预览（pop 语义，留 last-good） |
| POST | `/api/train/save` | 立即保存 |
| POST | `/api/train/pause` / `resume` | 挂起 / 恢复 |
| POST | `/api/train/stop` | 保存后退出，超时强杀 |

参数写回是**原子**的：先写 `*_data.dat.tmp` 再 `os.replace`，中途崩只丢临时文件；
写之前会把原文件复制成 `.dat.bak`。

## 四、模型列表上的「导出 / 删除」

- 导出：`POST /api/models/{name}/export` → 轮询 `/api/models/export/status`，按钮上显示进度；
- 删除：`POST /api/models/{name}/delete`，删 `<name>_*` 全部文件（权重、data.dat、备份目录），
  训练中的模型拒绝删。

### 启动耗时（实测）

model_004 这个量级（171M 参数 / 1.9GB 权重）：

| 阶段 | 之前 | 现在 |
|---|---|---|
| 进程 + import（torch 单独 3.4s） | ~4.5–6.4s | 同（冷启动固定成本） |
| 权重加载 + CUDA 初始化 | ~2.7s | 同 |
| 样本扫描 | ~0.9s | 同（实测 iterdir 4 万个只要 0.03s，这段是别的活） |
| 损失历史 / 摘要 | ~1.9s | 后台延后，不再挡路 |
| **到第一行迭代** | **~17.5s** | **~14.5s（-3s）** |

做过的：把 117MB `data.dat` 的 loss_history 解析**延后 8 秒**再跑 —— `pickle.loads` 虽在 C 里
执行但一直握着 GIL，原来正好和权重加载 / 首个迭代抢同一把锁。真到保存时如果它还没跑完，
`_ensure_loss_history_complete()` 会就地同步补完，不会截断历史。

试过但**撤掉**的：对齐目录列表缓存（按目录 mtime 缓存 4 万个文件名）—— 实测 iterdir 只要
0.03s，那 0.9s 的等待是 SampleLoaderV4 起预取进程/开 sqlite，不是扫描，缓存没意义。

没做的（留给以后）：权重 `mmap` 需要把「先搬 GPU 再 load」改成「先 load 再搬」，收益不确定；
常驻预热进程能把 4.5s 的 import 摊掉，但 Windows 没有 fork，架构改动大。

## 五、已知限制

- **参数分两类**：`resolution / archi / ae_dims / e_dims / d_dims / d_mask_dims / face_type`
  建完就固定（4.0 的权重形状绑死），参数页里是只读的；改 batch_size、lr、各种 power
  这类可以，**但只在下一次启动训练时生效**（4.0 在启动时读 options）。
- **人脸库页**（分组去重）还没做，训练流程里跳过。
- 训练页的「隐藏」只是返回模型列表，不停训练。

## 六、怎么验收

```
python tools/trainer_accept.py          # 建临时小模型，跑完自动删，约 3 分钟
python tools/trainer_accept.py --keep   # 留着临时模型看现场
python tools/trainer_accept.py --model 384wf   # 拿真模型跑（会真的训练它）
```

断言的是「行为」不是「字段」：迭代在涨、速度接近日志里的 ms、loss/lr 有值、
暂停时迭代不动、继续后恢复、保存后 `_data.dat` 时间戳变化、停止后 `ok=True`
且没有残留进程、再次启动能从保存处续上。
