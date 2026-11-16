# detect 车道吞吐：天花板、被谁按住、要改哪里

> 台架在 experiments/det_lane/（lane.py 车道 · mech.py 机制 · gil.py 争用探针 · parity.py 逐帧对拍 ·
> e2e_ab.py 端到端 A/B · matrix.py 一键复跑）。
> 机器：RTX 5070 Laptop 8GB · 32 逻辑核（16 物理）· 素材 1080x1920 30fps · tinymog TRT bf16 + CUDA Graph
> （1 张图，shape=(1,3,640,640)）· dfm 416 bf16w torch。除注明外同进程交替、多轮取中位。
> 标 PROV 的格子是机器被别人占着时量的，等干净机器复量。

## 1. 天花板（只 decode + detect，150 帧）

| 臂 | 车道 帧/秒 | detect ms | det_wait ms | det_prep ms | decode ms | u_det | GPU util |
|---|---|---|---|---|---|---|---|
| pre：帧先在内存，无解码 | **542 / 稳态 553** | 1.78 | 0.01 | 1.12 | - | 82% | 1% |
| live：ffmpeg rawvideo 真解码 + letterbox 在解码线程 | **91.4 / 稳态 92.4** | 2.31 | 8.58 | 1.46 | 12.85 | 15% | 12% |

检测自身天花板 = 1.8ms/帧（约 550 帧/秒），不可能被 detect 自身卡在 30。带真解码时车道 91 帧/秒，
已被上游喂给按住：det_wait 8.58ms = 等 decode+prep 交下一帧（12.85+1.46=14.3ms/帧，约 70 帧/秒）。

## 2. 争用（默认流上挂 dfm 21.4ms/次，150 帧）

| 臂 | 车道 帧/秒 | detect ms | det_wait | GPU util |
|---|---|---|---|---|
| base：device 同步 + 默认流（老口径） | 91.1 | 2.34 | 8.59 | 10% |
| event：只等自己这条流，仍在默认流 | 88.4 | 2.37 | 8.89 | 14% |
| fix：event + 检测独占流 | 86.5 | 2.51 | 9.02 | 9% |

dfm b1 不饱和 GPU（util 9~14%），device 级同步没有可等的东西，三档一模一样。所以「A) detect 排在
一批 dfm 后面」在 b1 档不是主因（b8 待复量 PROV）。同进程五臂（24 张 canvas x 3 轮交替，dfm b1 负载在跑）
一帧 detect 的中位：base 8.40 / event 7.90 / own 9.80 / 关图 10.31 / CPU 27.36 ms —— 同样没有分离。

## 3. 那一笔 19~46ms 到底是谁（gil.py，同进程）

| 环境 | detect 中位 | p95 |
|---|---|---|
| 安静 | **1.62** | 1.86 |
| +8 条纯 Python 忙线程 | **4403** | 6028 |
| +8 条 numpy 忙线程（放 GIL、占核） | **16.36** | 26.9 |
| +8 条 cv2.resize（放 GIL、占核） | **17.27** | 26.9 |
| +1 条 dfm 规模 GPU 耗子 | 8.37 | 10.7 |

detect 那一笔的膨胀在 host 侧（GIL/CPU 调度 + 等 GPU 空位），不是「排队等 dfm 算完」。流水线里量的
19~46ms 与「8 条 numpy/cv2 重活线程」这一档（16~17ms）同量级。

## 4. 端到端 A/B（真 _fused_run，150 帧，两档配置）

light = overlay 无遮罩（下游让开，看车道）；heavy = seamless + mask 8（界面默认档）

| 配置 | 臂 | fps | detect | det_wait | 车道 帧/秒 | dfm | xseg | compose | u_det |
|---|---|---|---|---|---|---|---|---|---|
| light | base | 18.73 | 19.74 | 28.29 | 21.8 | 36.6 | - | 182 | 42% |
| light | fix | 21.53 | 8.85 | 28.29 | 27.0 | 34.4 | - | 112 | 21% |
| heavy | base | 19.34 | 8.16 | 32.23 | 24.8 | 30.8 | 9.2 | 101 | 17% |
| heavy | fix | 20.23 | 7.71 | 32.33 | 25.0 | 29.8 | 8.8 | 98 | 16% |

- 独占流 + 事件同步把 detect 那笔 19.7 -> 8.9ms（-55%），端到端 18.7 -> 21.5 帧/秒（+15%）。
- 车道仍只有 22~27 帧/秒，卡在 det_wait 28~32ms：上游 decode 在这台机器上一帧要 21~25ms（同一份
  代码真空跑 12.85ms —— 差的 10ms 被 8 条合成 + 4 条对齐 + 3 条编码的 Python 线程抢走）。

## 5. 对拍

- 机制五臂（同进程、同 24 张 canvas）：base vs event/own/关图 24/24 帧逐值相同，max 位移 0.000；
  CPU 臂 3/24 相同、max 0.1（不同 EP 的浮点差，本来就不是同一条数学路径）。
- 150 帧对拍（parity.py）待跑 PROV。

## 6. 结论

- 车道上限 = min(1/上游, 1/(detect+等GPU))；detect 侧 1.8ms -> 550 帧/秒，上游侧 14.3ms -> 91 帧/秒。
- 到 50 帧/秒要同时满足两条，缺一不可：
  1) detect 那一笔 <=3ms：独占流 + 事件同步（本次改动，已量到 19.7 -> 8.9 这一段）；
  2) 上游喂给 <=15ms/帧：decode+letterbox 在这台机器上 21~25ms（GIL 争用），候选机制是合成走 GPU
     （少 8 条 Python 线程）/ decode 预分配缓冲 readinto（省一次 6.2MB 拷贝）/ letterbox 进独立小池，
     每条都要单独量 PROV。
