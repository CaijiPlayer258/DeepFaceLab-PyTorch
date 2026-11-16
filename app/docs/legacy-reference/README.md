# 归档区（legacy reference）

这里放**从 `app/engine/` 摘下来的死代码**：4.1 一行都不 import 它们，但删掉之前先把源码留在仓里
（历史 + 万一要照着重写）。目录结构照着原来的 `app/engine/` 摆，方便对照。

进来之前每个都做过「入边审计」：`git grep` 模块名 / 文件名 / 字符串 / `git ls-files`，确认零引用。
审计里发现**其实还有引用**的几个没进来，留在原地（清单见 `app/docs/notes/cleanup-2026-02.md`）。

## 目录

| 归档路径 | 原路径 | 为什么它死了 |
|---|---|---|
| `MaskProcessor/` | `app/engine/MaskProcessor/` | 独立 FastAPI 遮罩标注服务（32 文件）。4.1 的遮罩编辑器是 `app/server/backend_dataset/maskops.py` + `app/web/mask-editor/` 重写的，零 import。也是 `backend_mask` 的母本，留着当参考实现。 |
| `MergeStudio/api/` + `MergeStudio/__main__.py` | 同左 | MergeStudio 的独立 HTTP 服务。4.1 走**进程内** `run_export_pipeline`（`app/server/_merge_runner.py`），没人再起这个服务；唯一的内向引用是 `core/export_pipeline.py` 里 `_get_cache_dir`，已经就地算掉。 |
| `MergeStudio/core/arcface.py` | 同左 | 零 import（embedding 走 `MergeStudio/core/face_embedder.py`）。 |
| `MergeStudio/core/model_converter.py` | 同左 | 零 import。 |
| `DataAugmenter/` | `app/engine/DataAugmenter/` | XSeg 批量刷遮罩 CLI，只被自己的 `__main__` 引用。 |
| `core/qtex/` | `app/engine/core/qtex/` | Qt 外壳（QSubprocessor / QXMainWindow），4.1 没有 Qt 界面。`main.py qt selftest` 那个入口在归档前就已经指向不存在的 `qt_selftest.py`。 |
| `core/biliauth/*.py` | `app/engine/core/biliauth/` | 4.0 的 B 站登录/关注/投稿/播放地址。4.1 已移植到 `app/server/backend_bili/`；只留 `cookie_store.py`（它读 `app/engine/ui/user/` 那份真 cookie，两边共用）。 |
| `core/leras/archis/archi_spec.py` | 同左 | 选项表的转发层，真的那份在 `app/engine/core/archi_names.py`（4.1 就是 import 它）。 |
| `mainscripts/ExportDFM.py` | 同左 | 零 import（4.1 的导出走 fused 流水线）。 |
| `samplelib/serve_angle_view.py` | 同左 | 独立小服务，零 import。 |
| `modelhub/onnx/<模型>/convert_to_onnx.py` | `app/engine/modelhub/onnx/<模型>/` | 6 个一次性转换脚本。`app/modelhub/onnx/` 整个目录在 .gitignore 里，**它们之前根本没进过仓**——归档这一步顺便把它们收进历史。 |
| `qt-ui/` | `legacy/qt-ui/` | 4.0 的 PyQt 界面（整目录），4.1 是网页版。 |

## 别混淆

- `app/engine/vendor/` **不在**这里：其中的 `sam2` / `groundingdino` 是
  `app/server/backend_mask/painter.py` 的运行期依赖（AI 遮罩），审计结论是必须留。
- `app/engine/FacesetProcessor/` **不在**这里：`WebUI/page_trainer.py` 的
  `/studio/face-id-cluster` 与 `/studio/merge-aligned` 两个接口还 import 它。
