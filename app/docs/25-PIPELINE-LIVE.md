# 25 · 流水线实况（导出进度页的 SVG 图）

> 一句话：进度页原来那排「六个圆点」换成 **6 列 lane + 腰点收束的贝塞尔连线 + 光点**，
> 数据只有一处来源 —— `GET /api/merge/export/status` 的 `stages`。
> 验收：`app/tools/accept/progress_lit_accept.py`（47 条断言 + 3 张截图）。

## 1. 后端：`_stage_tick` + `_StageMeter`（唯一数据源）

`app/server/routes_merge.py`：

- `_StageMeter`（模块级单例 `_STAGE_METER`，一个进程同时只跑一单导出）：
  `ticks`（每段过了多少帧/张脸）、`lanes`（**真实线程数**）、`queues`（下游队列）、
  `ms` / `batch`（每帧 ms 与 dfm 攒批）。tick 只做一次 dict 自增（一把 Lock），
  不拼字符串、不进 JSON、不碰队列；只有 status 那一拍（500ms）才 `snapshot()`。
- `_stage_tick(key, n=1)`：在**入队/交给下一阶段之前**调用。六个 key：
  `frames / det / dfm / xseg / compose / encode`。

| 链路 | frames | det | dfm | xseg | compose | encode |
|---|---|---|---|---|---|---|
| 快速导出（fused，默认） | 解码→q_dec | 检测→q_det | 对齐→q_gpu（每张脸） | dfm/xseg→q_cmp | 合成→q_enc | PNG→写盘 |
| 分段链（串行六段） | frames/*.png 落盘 | aligned/*.jpg | pred/*.png | mask/*.png | merged/*.png | ffmpeg 进度 |
| 项目导出（`_proj_pipe`） | — | — | 读帧→q_gpu | — | →q_enc | 编码→写盘 |

### `status.stages`（顶层，和 `eff`/`vid` 平级）

```json
"stages": {
  "frames":  {"ticks": 66,  "inflight": 1,  "lanes": ["解码"],            "ms": 2.4},
  "det":     {"ticks": 64,  "inflight": 6,  "lanes": ["det#0"],           "ms": 123.52},
  "dfm":     {"ticks": 220, "inflight": 64, "lanes": ["dfm#0"],           "ms": 158.98, "batch": 8},
  "xseg":    {"ticks": 36,  "inflight": 0,  "lanes": ["xse#0"],           "ms": 34.84},
  "compose": {"ticks": 36,  "inflight": 0,  "lanes": ["t0","t1", "…t7"],  "ms": 118.18},
  "encode":  {"ticks": 36,  "inflight": 0,  "lanes": ["e0","e1", "…e5"],  "ms": 3.59}
}
```

- `lanes` **不是写死的**：合成池 8 线程就是 `t0..t7`、编码 6 线程就是 `e0..e5`、
  dfm/xseg 各一个 session 就是 1 条、det 有几路异构 session 就有几条；
  这一单不用 XSeg（遮罩来自 dfm）时 `xseg.lanes = []` —— 界面画成「不用」，不假装有线程。
- `inflight` = 该段**下游那个队列**的 `qsize()`（frames→q_dec、det→q_det、dfm→q_gpu、
  xseg→q_cmp、compose→q_enc、encode→0）。
- `ms` = 每帧 ms（和 `<成片>.log` 里 `profile（每帧 ms）` 同一口径）。
- 分段链的 `/api/merge/stage/status` 里那个 `stages` 是**每段元数据**（done/msg/…），
  所以实况那一份叫 `stages_live`，别混。

## 2. 前端：`MG_PIPE`（app.js）

- 结构：6 列 × N 条 lane（`rect.lane`，74×≤24、rx 7、`fill:var(--panel-2)`、
  `stroke:var(--line-strong)` 1px、文字 9.5px 等宽）；一次建好后**只改文字/坐标**，
  同一份 lanes 不重建节点（验收 C20 盯着元素身份）。
- 连线：相邻两列**共用一个腰点**（`x` = 两列中线，`y` = 两列所有 lane 的 y 的中点）。
  源侧每条 lane → 腰点一条（`edgesIn`），腰点 → 目标侧每条 lane 一条（`edgesOut`）：
  合成 8 × 导出 6 原来是 48 条交叉，现在是 8 + 6 = 14 条，进出共用同一个点 ——
  视觉上是「先收束到一点、再散开」。单段是三次贝塞尔（`arc()`），两个控制点分别与
  两端同 y，所以**两端切线都水平**，两段在腰点处方向连续，拼起来是一整条平滑的流。
  `stroke:var(--line-strong)` 1px、opacity .75；空列跳过。
- 光点：每次 tick 的**增量**放一个 2.4px `var(--ok)` 圆点；路径是「源 lane→腰点 +
  腰点→目标 lane」两段 `d` 拼接的**临时 path**（第二段以 `M 腰点` 开头，位置正好接上），
  `getPointAtLength` 走 470–750ms，走完连 path 一起删（路径长度按「源#腰#目标」缓存，
  不重复 `getTotalLength()` —— 它每次都逼一次 SVG 布局）。圆点走 circle 池复用，
  回收时连 `cx/cy` 一起清掉（池里不留上一次飞行的残坐标）。
  多 lane 阶段拿不到「是哪一条 lane 干的」（status 不按帧带这个），按 tick 序号**轮转**，
  保证光点始终落在真实存在的 lane 上。

  **密度口径**（用户反馈「闪烁频率不够、把上限去掉」之后定的）：
  - 轮询 `500ms → 250ms`（`POLL_MS`，`mgFusedPoll` 同值）；一拍的光点**均匀铺在这
    250ms 窗口里**（`sched()` 给每一发算一个「应该发生」的时刻；`t0` 在未来的先藏着，
    `kk<0` 时既不画也不调 `getPointAtLength` —— 负长度会抛错、把整帧 rAF 打死）；
  - 点亮同样铺在窗口里（`flashes[].at`），同一帧里同一条 lane 只闪一次（16ms 去抖，
    5 帧叠在 1/60 秒看不出差别，但会白跑 4 次强制重排）；
  - 两个**物理上限**取代原来的 `TICK_MAX=10`：`PULSE_PER_S=240`（按窗口折算 = 每拍最多
    60 个，超了按比例稀释，各段比例不变）+ `MAX_PULSE=200`（同屏在飞）。
    常见档（20~30 帧/秒、每帧 1~4 张脸）**一条都不丢**：20 帧/秒时每段 5 发/拍 ÷ 250ms
    = 50ms 一发，肉眼是连续流，而不是「每半秒跳一波」。
- 点亮：`classList.remove('lit') + 强制重排 + add('lit')`重启，关键帧
  `0% panel-2 → 6% rgba(61,220,132,.16)+var(--ok) → 40% 半衰 → 100% 回常态`，
  1.15s（`forwards`），1.25s 后摘类；叠加天然成立（衰减中再来一次会重新点亮）。
  `prefers-reduced-motion:reduce` → `animation:none`。
- 卡住：某 key >600ms 没 tick 且这一单还在跑、这段还没完成 → 该列所有 lane 加 `.stall`
  （`stroke:var(--warn)`、`fill: rgba(240,180,41,.10)`，浅色主题另有低透明取值）。
- 列间**没有**「在飞 N」那行字（用户要求去掉：腰点本身就是「这一列有多少在途」的
  视觉表达）。`status.stages[*].inflight` 后端照给，界面不再显示。
- 性能：轮询 250ms（一单一次请求，不按帧发；分段链那条仍是 600ms —— 它每次要 listdir
  四个产物目录）；`requestAnimationFrame` 里只写 `cx/cy/opacity`（不读布局）；
  点亮合到一帧里做，强制重排只在真点亮的那几条 lane 上发生，同一帧同一条 lane 最多一次。
- 视觉 token：只读 `app.css` 现成变量（panel / panel-2 / line / line-strong / ok / warn /
  shadow），层级靠「面 + 1px 描边」，不发光、不缩放、不描彩色边。

## 3. 常驻 eff 面板（已拆）

`#mgpEff`（14 字段常驻面板）与相关 CSS 一起删掉；**告警横幅 `#mgpEffWarn` 保留**
（它是「界面设了、后端没按它跑」的唯一提示）。14 个字段挪进右下角「查看生效配置」→
抽屉（`#mgpEffDrawer`），点开才渲染。详见 `22-EFF-CONFIG.md`。

## 4. 验收

```
runtime\server\Scripts\python.exe app\tools\accept\progress_lit_accept.py
```

- 真跑一单 72 帧快速导出（素材 = 训练预览图裁块，有脸 → 六段都动），采 90+ 拍 status；
- 断言：六键齐全 / lanes == 实际线程数（并和后端日志「线程 …合成8/编码6」对账）/
  ticks 单调不减 / inflight 整数 / DOM 没有 #mgpEff 有 #mgpEffWarn / 抽屉 14 字段 /
  六列 / 腰点收束（条数 = 源+目标、进出同一坐标集合、单段一个 C）/ 光点在曲线上 /
  上限 60 / 点亮 1.15s 且 1.25s 后摘 / 卡住变琥珀再复原 / 图上没有「在飞」/ 节点复用 / 减动效；
- 截图 3 张进 `%TEMP%\dfl_progress_lit_accept\`：`prog-live.png` / `prog-stall.png` /
  `prog-done.png`；
- `--shots-only`：复用上次的 status/eff（不跑导出）只重拍截图，改样式时用这个；
- `--no-browser`：只跑后端那半段（快，用来改后端时的冒烟）。

> 拍「进行中」那张时会把 **CSS 动画暂停在 6% 峰值**、把光点时钟钉在出生后约 45%
> （`window.mgpPipeFreeze`，默认 0 = 正常跑）：headless 一次截图往返常常比 1.15s 还长，
> 不按住时钟只能拍到「已经暗回去」的稳态。动画本身是真的，断言 C6/C8/C10–C14 验的就是真跑。
