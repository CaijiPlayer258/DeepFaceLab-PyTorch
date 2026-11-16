# 4.1 清理记录（2026-11-16）

用户批准「没用的都移除掉」。原则：**只删/只移审计确认零引用的东西**，拿不准的一律保留并写在这里；
跟踪文件用 `git mv` 进归档区（保历史），未跟踪的 scratch 直接删；每步一提交、每步验
`compileall` + `import server.app`，引擎有改动的那几步额外跑一次真 fused 导出。

范围约束（全程遵守）：没碰 `workspace/ weights/ runtime/ desktop/node_modules`；没删
`RunGUI.bat`、`UI设计规范.md`、`README*`、`app/engine/main.py`，以及 xlib / core / models /
MergeStudio/core / facelib / DFLIMG / localization / mainscripts / WebUI / samplelib / modelhub /
vendor 里审计标「必须留」的部分。

## 一、逐步账

| 步骤 | 提交 | 做了什么 | 回收 | 验证 |
|---|---|---|---|---|
| 1a | `62e8bd2` | `_enc4k_bench/` 里被 docs/29 引用的 6 个测量脚本 + `REPORT_4K_encoder.md`（+ 被报告引用的 DESIGN 文档）→ `app/tools/bench/enc4k/`；顺带把脚本里写死的 `D:/MyProject/...` 改成按 `__file__` 定位，外部素材走 `ENC4K_EASY_SRC`；docs/29 路径同步 | 0（搬迁） | py_compile 6 个脚本 exit 0 |
| 1b/1c | `aeb9ec2` | `experiments/` 10 份 md → `app/docs/notes/experiments/`（`<子目录>.md`，附 README）；删 `experiments/` 与 `_enc4k_bench/` | **4362.3 MB**（experiments 70.57 + enc4k 4291.71） | compileall 0 · import 0 · `seamless_mod()` 从新路径加载成功 |
| 2 | 无提交 | 删根下 `_refs.json` `_refs2.json` `_probe_export.py` `_electron_backend.log`、空目录 `_common/`、`app/engine/__pycache__`、`MergeStudio/workspace/preview_cache/`（`audio.m4a` 本来就不在） | 0.06 MB + 4 个空目录 | compileall 0 · import 0 · git status 干净（全是未跟踪，没有可提交的东西） |
| 3 | `146f71d` | 引擎死代码 `git mv` → `app/docs/legacy-reference/`：MaskProcessor（32）、MergeStudio/api + `__main__` + arcface + model_converter（11）、DataAugmenter（3）、core/qtex（6）、core/biliauth 5 个（留 cookie_store）、archi_spec、ExportDFM、serve_angle_view、modelhub/onnx 6 个转换脚本；删 `assets/screenshots`（7 张） | **4.51 MB** 净删（另有 0.40 MB 转成归档） | compileall 0 · import 0 · fused 导出 PASS 20 / FAIL 0（120 帧、音轨在、无 cache） |
| 4 | `060ffb6` | `legacy/qt-ui/` 整目录（58 跟踪文件）→ `app/docs/legacy-reference/qt-ui/`，删 12 个 `__pycache__` | 0（搬迁） | compileall 0 · import 0 · `git check-ignore` 逐条确认凭证文件仍被挡 |
| 5 | `ee1cb64` | 删 12 个零入边转发壳：`bili_{follow,http,placeholder,queue,user_info,video}.py`、`biliauth.py`、`_convert.py`、`_face_emb.py`、`_faceid_{extract,fast,trt}.py` | 约 0.015 MB | compileall 0 · import 0 · 6 个真身模块单独 import 全绿 · fused PASS 20 / FAIL 0 |
| 6 | `4d1a1f2` + `6dea48e` | 删 `docs/preview/`（3.99 MB）+ 两个 theme html；docs/ 的 md 收进 `app/docs/`（31-里程碑 / 32-ALIGN-GATE / notes/）；01/02/05 → archive/4.0/；改掉文档里与现实矛盾的句子 | **4.02 MB** | compileall 0 · import 0 |

合计：净回收 **约 4.37 GB**（其中 4.29 GB 是 4K 台架的 raw/mp4），文档与源码侧 ~8.6 MB；
归档区新增 5.4 MB（`app/docs/legacy-reference/`）。

## 二、审计推翻的四条前提（这几处**没删**）

| 东西 | 清单怎么说 | 实测 | 处置 |
|---|---|---|---|
| `app/engine/vendor/`（102 文件） | 「随 MaskProcessor 一起归档，唯一引用者是 bisenet_parser」 | **sam2 与 groundingdino 是 `backend_mask/painter.py` 的运行期依赖**（按路径挂成 `xlib.models.sam2` / `xlib.models.grounding_dino` 再 import，缺了 AI 遮罩直接不可用；painter 里还有一条用户可见的「建模代码缺失」报错在指它） | 已 `git mv` 归位。只有 `bisenet_net.py` / `download_sam.py` 属于 MaskProcessor 那套 |
| `app/engine/FacesetProcessor/`（8 跟踪文件） | 「7 个文件归档」 | `WebUI/page_trainer.py` 的 `/studio/face-id-cluster` 与 `/studio/merge-aligned` 两个接口 `from FacesetProcessor.Filter import FaceIDFilter`，而 WebUI 是必须留的目录 | 整包留下 |
| `app/engine/Extractor/strings.py` | 「归档」 | `Extractor/Extractor.py:71` 有裸的 `from strings import S`（Py3 里之所以能跑，是因为引擎把 `app/engine/Extractor` 也挂进了 sys.path）；`Extractor.Extractor` 又被 `FacesetProcessor/Analyzer.py` import | 留下 |
| `app/engine/ui/`（清单说是「空包」） | 「空包，删」 | 里面有 `ui/user/bilibili_cookie.txt`（`core/biliauth/cookie_store.py` 的落点，B 站登录态）。`git ls-files` 看着是空的，是因为整目录被 .gitignore 挡了 | 留下（删了就是把人踢下线） |

## 三、保留但没动（不确定 / 明确不该动）

1. `app/server` 还有 **21 个 `_*.py` PEP 562 转发壳**没动。本次只删了批准的 12 个；剩下这些里
   `_dataset_fast / _engines / _face_group / _hwaccel / _legacy / _load_events / _nowin /
   _numexpr_shim / _ortcuda / _pipeline / _train_device / _trt_session / _xseg / _errface` 等
   仍有真实入边（`from .._xxx import`），另有 `_align / _detect / _eff / _faces / _merge_compose /
   _projplan / _proj_pipe / _paths / _sess / _occlusion / _oneshot / _scan_runner / _trt_build /
   _faceid_cli` 被文档或注释当作「现行路径」引用（第 5 步的清单没点名，就没有扩大范围）。
   `_faceid_cli.py` 是零 import 的转发壳，按第 5 步的口径应该是下一批的候选。
2. `app/docs/legacy-reference/qt-ui/user/` 下两份 4.0 遗留凭证（`bilibili_cookie.txt` 与 `.bak`）：
   没删，随目录搬走，仍被 `**/` 规则挡住。
3. `app/docs/legacy-reference/qt-ui/img/` 里 4 张较大的图（avatar1.png 1.5 MB、
   music_covers/cover1.jpg 967 KB、homepage_background.png 829 KB、icons.dat 345 KB）：
   按「>2 MB 且零引用才删」的口径，都不够 2 MB，只列出不动。
4. `app/engine/xlib/qt/_unused/`（Qt 时代的废弃件）没审也没动：在 xlib 里，属于「必须留」的目录。
5. `app/server/_vision`、`backend_*` 的内部重复实现没碰（不在本次范围）。
6. 根目录新出现的 `_acc_*.log` / `_probe_cfg.py`：是**并发代理**这几轮跑验收留下的，不是我产生的，
   且都被 `/_*.log` `_probe_*.py` 挡住，没有替别人清。

## 四、验证结果

> 时间线说明：本轮清理与另一路并发工作（`626c962` 收敛 API 协议、动过 `app/web`）有重叠。
> `fused_export / frame_range / stage6 / merge_ui / eff` 是在 `6dea48e`（我这一串最后一个代码提交）
> 的树上跑的；`stage_export` 那次在 `626c962` 之后。两份树在 `app/server` 与 `app/engine` 上的差异
> 与本次清理无关，最终 `compileall / import / openapi` 复测在 `c5f2f9b`（= `626c962` + 三个文档）。

| 检查 | 结果 |
|---|---|
| `python -m compileall -q app/server` | exit 0 |
| `import server.app` | exit 0 |
| openapi paths | **124** |
| `fused_export_accept` | **exit 0**（PASS 20 · FAIL 0 · 120 帧 / 20.9 帧每秒，出片、音轨、无 cache 全对） |
| `frame_range_accept` | **exit 0** |
| `stage6_accept` | **exit 0**（PASS 64 · FAIL 0 · 63 s 档） |
| `eff_accept` | 41 / 42（唯一红项 F2 见下，**基线同样红**） |
| `merge_ui_accept` | PASS 31 · FAIL 1（D6，**基线同样红**） |
| `stage_export_accept` | PASS 64 · FAIL 6（6 项全在 f 案例，**基线同样红**，见下） |

### 三个红项都是「清理之前就红」，不是这次弄的

拿 `git worktree` 在 **基线 5e998ba** 上起了第二份代码（`runtime/ weights/ workspace` 用 junction
挂回同一份），同样的命令跑 `merge_ui_accept` 与 `eff_accept`：

```
BASE merge_ui_accept  → PASS 31 · FAIL 1   [FAIL] D6 刷新后还是 fp32 / gpu · [('dfm_dtype','fp32'),('compose_backend','auto')]
BASE eff_accept       → 41 / 42            [FAIL] F2 横幅文案里有 detector · 合成后端：选了 gpu、实际跑 cpu…
```

FAIL 文本与 HEAD 上**逐字相同**。

`stage_export_accept` 的 f 案例（六步动画）更直接：它找 `#mgpSteps .cvst`，而
`id="mgpSteps"` 在**基线 5e998ba 的 `app/web/index.html` 里就不存在**（`git show 5e998ba:app/web/index.html`
grep 计数 0）—— 进度页在 `7d5d014` 就换成了「流水线实况」（`#mgpStepsBox` / `#mgPipe`），旧的六步
stepper 没了，验收脚本没跟着改。附带后果：f 找不到节点时会**空转 900 秒**再判红。

这三项要修得单独立项（改验收口径 / 改 DOM 契约），不属于「清理」，本次没动。
