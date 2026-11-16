# 坏帧跳过降级（DFL_FUSED_SKIP_BAD）

**结论**：raw 管道多了一个**默认关**的降级档。关着时行为与改动前逐字节相同（同一单成片
md5 都是 `ae157fdda56d47b35eaa7addd68b262c`，改前跑两次、改后跑两次）。打开后某帧在合成
阶段失败只顶替这一格，成片照常出、帧数/时长/音轨不变，跳过哪几帧写进日志与 status；
连着坏/累计坏超阈值仍然整单失败。`fused_export_accept` **67 PASS / 0 FAIL**。

## 1. 做了什么

| 文件 | 改动 |
| --- | --- |
| `app/server/backend_merge/pipeline_fused.py` | 开关 + 账本 `_FusedSkipBad` + 阈值；合成腿 `try/except`；写者腿 `_slot()` 顶替；`_frame_done`/`_report` 上报；收尾一行「本单跳过 N 帧」 |
| `app/server/backend_merge/routes.py` | `/api/merge/export/status` 与 `/api/merge/stage/status` 加 `skip_bad / skipped_count / skipped_frames / skip_hint` |
| `app/server/backend_merge/pipeline_video.py` | 起单时把这几个键清干净（否则新一单会显示上一单的跳过数） |
| `app/web/js/app.js` | 进度页副标题 + 结果行各一行提示（只在真跳过时改，默认路径一个字符不动） |

开关：

```
DFL_FUSED_SKIP_BAD=1          打开降级（默认 0）
DFL_FUSED_SKIP_MAX=30         连续失败上限的「顶」（默认 30）
DFL_FUSED_SKIP_BAD_FAULT=10,20,30   验收注入：点名这些帧在合成阶段强制失败
```

## 2. 为什么「重复上一帧」，不是「黑帧」

先说前置结论：**两者都不能「少一帧」**。raw 管道是定长流（`-s WxH -pix_fmt bgr24`），
少写一帧后面全错位；帧范围导出还带 `-frames:v N`，少写就永远凑不满 N（`_fused_pipe_cmd`
里那条「43 → 44 个音频包」就是这件事的实测）。所以「跳过」只可能是**顶替一格像素**。

在这个前提下选「重复上一帧」：

1. **原帧是好的**，坏的只是合成那一步。刷黑 = 为了一步失败把整帧真实画面（背景、别人、
   没换的脸）一起扔掉。
2. **30fps 下一次重复 = 33ms 停顿**，眼睛几乎看不见；黑帧是整屏黑闪一下，还和「片子本来
   就有黑场」混在一起，事后分不清哪一帧是降级产物。
3. **可见性不该靠烧进画面来买**：帧号 + 原因进 `<成片>.log`、进 `status.skipped_frames[]`、
   进度页一行提示、收尾一行「本单跳过 N 帧」。日志是给人查的，画面是给人看的。
4. **顶替只借一帧**：帧号/时长/音轨一个字节都不动（定长帧 + `-framerate`）——「时长不变」
   是构造上的，不靠测。
5. 实现上只有单写者知道「上一帧是谁」（合成是 8 条并发线程，谁先完成没有次序），所以顶替
   放在 `_FusedRawWriter._slot()`：它保着上一份**真的写进管道**的字节。

两个例外：**首帧**就坏（没有上一帧）→ 黑帧；**PNG 落盘那条排障路**（`DFL_FUSED_RAWPIPE=0`）
没有单写者、3 条编码线程乱序写 `%06d.png`，「上一帧」没有定义 → 一律黑帧（同样是顶替一帧）。

代价（诚实记账）：镜头**硬切**处如果正好坏一帧，会把上一个镜头的最后一帧多留 33ms。

## 3. 阈值

```
连续上限 = max(3, min(DFL_FUSED_SKIP_MAX, round(5% × 帧数)))
累计上限 = max(3, round(5% × 帧数))
越线 → 整单失败（.part 删除，错误文案写明「连续 N 帧失败（上限 M）」）
```

| 分段帧数 | 12 | 40 | 120 | 400 | 1500 | 3000 |
| --- | --- | --- | --- | --- | --- | --- |
| 连续上限 | 3 | 3 | 6 | 20 | 30 | 30 |
| 累计上限 | 3 | 3 | 6 | 20 | 75 | 150 |

- **为什么必须有**：一帧坏是意外，连着坏是病。继续顶替出来的是一段**静止画面** —— 那才是
  真正的「静默产出坏片」，用户要几小时后才发现第 900 帧往后全冻住。
- **为什么 30 帧**：30fps 下 <1 秒不可能是系统性坏；又长到不会被一次磁盘抖动/OOM 重试误杀。
- **为什么还要一条累计 5%**：治「隔一帧坏一帧」—— 连续计数永远是 1，但已经坏了一半画面。
- 短片用 5% 会只剩 1~2 帧，所以两头都压 3 帧下限，免得小单被两帧抖动判死。

## 4. 证据（`experiments/skip_bad/` 里跑出来的，脚本都在同目录）

### 4.1 默认关：成片 md5 与现状相同

同一份素材（`workspace/9月29日 (5).mp4`，1080x1920@30）、同一段（1..40）、默认参数：

| 什么时候 | md5 | 字节 | 帧 | 时长 |
| --- | --- | --- | --- | --- |
| 改代码**前** 第 1 次 | `ae157fdda56d47b35eaa7addd68b262c` | 1232860 | 40 | 1.3333s |
| 改代码**前** 第 2 次 | `ae157fdda56d47b35eaa7addd68b262c` | 1232860 | 40 | 1.3333s |
| 改代码**后** 第 1 次 | `ae157fdda56d47b35eaa7addd68b262c` | 1232860 | 40 | 1.3333s |
| 改代码**后** 第 2 次 | `ae157fdda56d47b35eaa7addd68b262c` | 1232860 | 40 | 1.3333s |

（`lab.py baseline`；两次同码也相同 = 这台机器这份素材本身可复现，md5 才有判别力。）
第三次出现在 `lab.py det` 的 `det_base`（`DFL_FUSED_BATCH=1`）：md5 仍是
`ae157fdda56d47b35eaa7addd68b262c` —— 攒批口径也不影响像素，这条 md5 有判别力。

### 4.2 打开 + 注入 3 帧失败 → 成片照常出

`DFL_FUSED_SKIP_BAD=1 DFL_FUSED_SKIP_BAD_FAULT=10,20,30`（`lab.py skip3`）：

- `phase=done` · 40 帧 · 1.3333s（与基准**同一时长**）· 1204161 字节
- `status.vid.skipped_count=3` · `skipped_frames=[10, 20, 30]`
- 日志（`<成片>.log`）两处都写：合成腿「第 10 帧合成失败 → 跳过（RuntimeError: 注入的坏帧…）」，
  写者腿「第 10 帧坏 → 跳过（…）；这一格用上一帧顶替（本单已跳过 1 帧）」
- 收尾一行：`本单跳过 3 帧（共 40 帧 · 帧号 10、20、30）—— 每一帧都照常写进管道…`
- 进度页：副标题「跳过 3 帧（坏帧降级 · 详见成片日志）」、进度文案带「已跳过 3 帧」、
  `warnings` 一条；运行中轮询也读得到 `skipped_count`（lab.py 记到 `live_skipped_seen=3`）

**像素层面**（`verify_dump.py` / `verify_frames.py`，确定性口径 `DFL_FUSED_BATCH=1`）：
- 逐帧 `*.dump/<帧>_composed.png` 对拍：**37 帧逐字节相同、0 帧不同**；被注入的 10/20/30
  两边都没有 dump（合成没跑完 → 不产出像素），符合预期。
- 解码后看：每个坏帧最像的都是基准的 **k-1** 帧（mean 0.76 / 0.93 / 0.94、max 14/28/26），
  而 vs 本该有的 k 帧是 1.01 / 1.18 / **13.09**（第 30 帧正好是切镜）—— 顶替的是上一帧，
  不是原内容。同号帧的差异是 x264 码率控制的 QP 抖动（流里少了一帧真实内容），不是像素改了。

### 4.3 连续超阈值 → 整单失败

`FAULT=10..40`（40 帧单的连续上限是 3）→ 第 4 个坏帧就判死（`lab.py burst`）：

```
phase=failed
error=坏帧降级判定不可恢复（成片作废）：连续 4 帧失败（上限 3 帧）—— 零星坏帧能顶替，
      连着坏说明这条线病了（模型/素材/编码器），判定不可恢复，整单作废
skipped_frames=[10, 11, 12, 13] · 无成片 · 无 .part
```

### 4.4 取消不算坏帧

开关开着 + `FAULT=2..7`（6 个连续坏帧；这一单是 400 帧，连续上限 20，所以它们**都在允许范围内**，
验的是「取消不会被降级逻辑改写成失败」）+ 第 18 帧时中途取消（`lab.py cancel`）：

```
phase=cancelled · 无成片 · 无 .part · skipped_count=6（照实记，但没升级成失败）
```

而且 `_fused_run` 的收尾顺序是「先判取消、再判 errors」，加上 `skip.stop_skip()`（stop 一置位
就不再算坏帧），取消永远赢。

### 4.5 PNG 落盘那条排障路（`DFL_FUSED_RAWPIPE=0`）

`DFL_FUSED_SKIP_BAD=1 DFL_FUSED_SKIP_BAD_FAULT=10,20,30 DFL_FUSED_RAWPIPE=0`（`lab.py rawpng`）：
`phase=done` · 40 帧 · `skipped_count=3`（这条路没有单写者，坏帧用**黑帧**顶替，见第 2 节）。

### 4.6 回归

`python app/tools/accept/fused_export_accept.py` → **PASS 67 · FAIL 0**（最终代码 187.8s；
中途那版也是 67/0 · 219.4s）。覆盖：a 默认出片 / b fused-vs-分段像素 / d 取消 + 409 + 取消后
再跑一单 / e 帧范围 101 帧 / f 项目导出 / c 显存降档 + 400 拒绝 / g PNG 解码路径。
（帧/秒那两行别拿来对比：同期这台机器上还有别的基准在跑。）

## 5. 复现

```
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py baseline   # 默认关，两次 md5
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py skip3      # 注入 3 帧
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py burst      # 连续超阈值
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py cancel     # 取消
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py rawpng     # PNG 落盘路（黑帧顶替）
runtime\server\Scripts\python.exe experiments\skip_bad\lab.py det        # 确定性口径 + dump
runtime\server\Scripts\python.exe experiments\skip_bad\verify_dump.py det_base_0001-0040.dump det_skip_0001-0040.dump 10,20,30
runtime\server\Scripts\python.exe experiments\skip_bad\verify_frames.py det_base_0001-0040.mp4 det_skip_0001-0040.mp4
```

产物与原始输出在 `experiments/skip_bad/out/`、`*_run.txt`、`server_*.log`。没有 commit/push。
