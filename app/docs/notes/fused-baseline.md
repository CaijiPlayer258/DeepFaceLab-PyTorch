# fused 基线与回归口径（拆分 _fused_run 之前必须先钉的东西）

目的：`pipeline_fused._fused_run` 是 1697 行 / 25 个闭包的一个函数，拆它等于改写成 ctx 对象。
改之前得有一组**可重复、能判红**的数，否则"改完好像没变慢"只是感觉。

## 怎么跑（固定命令，别加参数）

```bat
runtime\server\Scripts\python.exe app\tools\accept\fused_export_accept.py --cases a
```

`--cases a` 只跑第一条：真后端 + 真 150 帧 1080×1920 素材 + 现成 dfm/xseg，
断言 20 项（出片 / 帧数 / 时长 / 音轨 / **无 `<视频>_cache/`** / 帧号范围 / 各段耗时与占用）。

## 基线（2026-11，三次背靠背，同一台机）

| 轮 | fps | 断言 |
|---|---|---|
| 1 | 18.99 | PASS 20 · FAIL 0 |
| 2 | **21.31** | PASS 20 · FAIL 0 |
| 3 | 21.66 | PASS 20 · FAIL 0 |

中位 **21.31 帧/秒**；三次离散 19.0~21.7（±7%）。整轮套件（a~g）里那次记到 22.26 ——
**同一份代码在 19.0~22.3 之间跳**，所以判定阈值不能定成 ±2%。

## 判红的标准（拆分后按这个比）

1. `PASS 20 · FAIL 0` 必须成立（硬门槛：这是**功能**，不是性能）。
2. 三次中位 fps **不低于 21.31 的 85%**（≈18.1）。低于它 → 认为真的慢了，回滚。
3. 任何一次出现 `FAIL` → 不看 fps，直接回滚。

## 为什么不用 md5 当基线

`cudnn.benchmark=True` + `deterministic=False`（`app/engine/core/leras/nn.py:73-75`）+
dfm 攒批时序不同 → **同一份代码跑两遍成片 md5 就会变**（今天实测 0/40 帧逐位相同）。
要比像素得先切确定性模式（`app/docs/28-PIXEL-PARITY.md`）。

## 帧号序列的口径

`_FusedRawWriter` 的硬约束是**单写者 + 帧号严格递增**（乱序帧号会被 ffmpeg 当成丢帧）。
这条由 `fused_export_accept` 的帧数/时长断言间接覆盖（帧数不对就会 FAIL），
另外 `<成片>.log` 里有 `raw 管道` 那一行的首末帧号，拆分前后应一致。
