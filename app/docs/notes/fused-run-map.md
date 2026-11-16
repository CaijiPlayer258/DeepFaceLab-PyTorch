# `_fused_run` 地图（1697 行 / 20 个闭包）

> 起因：这个函数是整条融合流水线（解码→检测→对齐/攒批→dfm+XSeg→合成→PNG/编码），
> 改任何一段都要先在 1700 行里定位。**它没有拆分**，因为闭包共享同一批局部状态，
> 拆=改写成 ctx 对象（重写）。这份地图先解决"找不到"，拆分等有基线后再做（基线见 `fused-baseline.md`）。

## 六段（按函数内横幅）

| 行 | 段 |
|---|---|
| 883 | 合成放在哪儿算：CPU（默认）还是 GPU（DFL_COMPOSE_GPU=1） |
| 980 | 模型 / 显存 / 帧预算（起流水线之前，出问题要在这里就喊出来） |
| 1070 | GPU 车道：预检（DFL_FUSED_GPU_LANES，默认 1） |
| 1216 | 坏帧降级（DFL_FUSED_SKIP_BAD，**默认关**） |
| 1340 | 显存保护：dfm 批大小按 session 实测阶梯选（每个 dfm 一次） |
| 1361 | ① 解码：ffmpeg 的 rawvideo 帧流 → 帧缓冲（信号量卡死上限，满了解码就停，天然背压） |
| 1454 | ② 检测：默认 1 路（检测器是共享对象，本来就不能并发）；异构池时 N 路 |
| 1511 | ③ 对齐/预处理 + 归组：68 点、dfm 输入、xseg 的 dst 画布、这一帧用哪个 dfm |
| 1610 | ③ 换脸那一刀的输入：同一条 get_transform_mat(lm, S, WHOLE_FACE) |
| 1651 | ④ dfm + ⑤ XSeg：按 dfm 攒批，一次前向；回填到各自的帧 |
| 1756 | ① 剩下的窗口里最多再来 rate*left 张，凑不满 → 现在送 |
| 1759 | ② GPU 段本来就空着：攒批换不来吞吐，只是在给每张脸加延迟 |
| 1917 | ⑤ 合成（CPU：compose_frame；GPU：compose_frame_gpu，DFL_COMPOSE_GPU |
| 2020 | ⑥ PNG 编码 + 落盘（无损，level 同 ⑤ 合成阶段） |
| 2094 | 异构 session 池：探测 → 微基准 → 配比（见 server/_sess.py） |
| 2125 | 流水线实况：lane 数 = **实际**线程数（不是写死的） |
| 2144 | 成片出口：raw 管道要先起 ffmpeg，再起唯一写者 |
| 2336 | 坏帧降级收口：日志里写清「本单跳过 N 帧」 |
| 2356 | ① 1x1 / 1x2（多挂一路 CPU 检测 session）值不值； |
| 2357 | ② 0x1（检测整段走 CPU EP）值不值 —— 这是 DFL_DET_EP=auto 的门②。 |
| 2408 | ⑦ 编码 |
| 2480 | 断言：全程没碰 <视频>_cache/（这是 fused 和分段的关键区别） |

## 闭包（行号 = 定义行）

| 闭包 | 行 | 长度 | 干什么 |
|---|---|---|---|
| `_lane_ent` | 1106 | 23 | 这条车道用哪份 dfm 会话（预检没覆盖到的模型在这里补建）。 |
| `_xseg_masks` | 1131 | 6 | 车道共用的遮罩会话（锁只在多条车道时存在，见 xseg_lock）。 |
| `_fail` | 1230 | 3 | errors.append("%s: %r" % (stage, exc)) |
| `_put` | 1234 | 8 | while not stop.is_set(): |
| `_get` | 1243 | 7 | while not stop.is_set(): |
| `_poll` | 1251 | 10 | 非阻塞取一个：这一拍没活就返回 None，停止时返回 END。 |
| `_report` | 1262 | 36 | now = time.time() |
| `_frame_done` | 1299 | 21 | 一帧「出去了」：老路 = PNG 已落盘，raw 路 = 已交给 ffmpeg。 |
| `_skip_note` | 1321 | 18 | 坏帧的可见性：status 的计数 + 帧号列表 + 进度页那一行（日志由调用处 |
| `_batch_for` | 1346 | 14 | with batch_lock: |
| `_decode` | 1390 | 63 | proc = None |
| `_detect` | 1461 | 49 | try: |
| `_prep` | 1512 | 138 | try: |
| `_ladder_per_face` | 1654 | 10 | 阶梯实测的 {batch: 每张 ms}（_pred_pick_batch 留下 |
| `_target_for` | 1665 | 30 | 这一路 dfm 一次前向喂几张：攒批目标（默认 8），但**实测说不行就让开** |
| `_gpu` | 1702 | 214 | 一条 GPU 车道：从 q_gpu 抢脸 → 攒批 → dfm 前向 → u8  |
| `_compose_args_gpu` | 1918 | 6 | GPU 入口直接吃 uint8 —— 省掉 CPU 上的 5 次 astype( |
| `_compose_args_cpu` | 1925 | 20 | args = [] |
| `_compose` | 1946 | 73 | try: |
| `_encode` | 2026 | 48 | try: |

## 要改什么去哪一行

```
解码（rawvideo / png）         _decode           1390
检测（含 detect 批、TRT/ORT）   _detect           1461
对齐 + 攒批 + 每脸目标           _prep / _target_for  1512 / 1665
dfm + XSeg + u8 + 成片缓存       _gpu              1702
合成（CPU/GPU 两套入参）         _compose_args_* / _compose  1918-2018
PNG 落盘 + 编码 + 写管道         _encode           2026
帧预算 / 跳过坏帧 / 报告         _batch_for 1346 · _skip_note 1321 · _report 1262
```

## 改动前后的硬判据

- `runtime\server\Scripts\python.exe app\tools\accept\fused_export_accept.py --cases a` → **PASS 20 · FAIL 0**；
- 三轮中位 fps ≥ 基线的 85%（基线 21.31，见 `fused-baseline.md`）。
