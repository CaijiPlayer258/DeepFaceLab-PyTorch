# 33 · 前后端分层与 API 契约审计

> 只读静态审计。**没有启动任何服务、没有跑导出、没有接模型**；唯一一次执行是
> `import server.app` + `app.openapi()`（用 `runtime/server/Scripts/python.exe`）。
> 基线：`server.app` 的运行时 OpenAPI —— **132 个 operation / 124 个路径模板**。
>
> 相关：`app/docs/11-API-PROTOCOL.md`（模型/训练**参数**怎么走）、`08-SERVER-MAP.md`（路由在哪）。
> 本文管的是**形状**：端点清单、请求/响应字段、错误口径、以及不一致之处。

---

## 0. 一句话结论

**分层是干净的，协议是分裂的。**

- **分层**：`app/web` 是唯一前端；`app/server/**/*.py` 里 **0 个文件**吐 markup（只有 vendored 的
  `engine/WebUI/page_*.py` 和 `engine/samplelib/page_sample_viz.py` 带内联 HTML，属 4.0 移植件，HTTP 隔离）。
  Electron 只有主进程 + preload，不碰渲染层。**但前端产物有 7 处散落在 `app/web` 之外**（§7），另有 1 个在 `app/web` 内但零引用。
- **协议**：132 个端点里只有 **39 个**（`backend_merge` 29 + `routes_faces` 10）走
  `{ok,error,detail}` 统一信封；**另外 90 个**抛 `HTTPException` → FastAPI 的 `{detail}`。
  前端因此同时写了 `d.error || d.detail` 两套兜底。这是"协议不统一"的**总根因**，不是个例。
- **对账**：前端调了但后端没有 **4 条**（全在 mask-editor，且都是死代码，不是活 404）；
  method 错配 **0 条**；后端有但前端从不调 **20 条**（6 条只给验收脚本，14 条真死）。

---

## 1. 审计口径与复现

| 项 | 做法 |
|---|---|
| 后端口径 | `import server.app` → `app.openapi()`，得 132 operation / 124 path template；再用 `ast` 扫一遍源码的 `@router.*` **交叉验证 132 = 132**（无动态注册漏项） |
| 前端口径 | 扫 `app/web/js/*.js`、`app/web/index.html`、`app/web/mask-editor/**`：**120 处 `fetch()` 调用点**、1 处 `EventSource`、0 处 `XMLHttpRequest`、0 处 `WebSocket` |
| URL 解析 | 对 `fetch("…" + expr + "…")` 做顶层 `+` 切分：字符串段保留、表达式段记 `{x}`；模板串 `${…}` 同样记 `{x}`。**模板化后再比**，`/api/merge/frame/12` 归到 `/api/merge/frame/{idx}` |
| 间接调用 | 4 处 `fetch(变量)` 手工回溯：`mvList()`、`trainCtl(path)`、`face_db.js::call/post/del`、`mask-editor/api.js::request()`；都由字面量喂进来，全部还原 |
| 资源加载 | `<img src>` / `<video src>` 也计入"被前端使用"：`/api/dataset/thumb`、`/api/merge/video`、`/api/merge/faces/thumb`（`face_db.js:162`） |
| 匹配规则 | 段数必须相等；前端 `{x}` **只能**匹配后端参数段，不能吞字面量段（否则 `stage/{x}` 会误吞 `stage/status`）；字面量必须逐字相等 |
| 输出 | 前端去重后 **116 个 (method, path)** —— 对 132 个 operation 做差集 |

---

## 2. 端点清单（按模块分组）

列含义：**前端** = `app/web` 里是否有调用点（✅/❌）；**请求体关键字段** = 从 `body.get("…")` 抽的
实际读取键（不是文档抄的）；**错误口径** = 该 handler 抛错后浏览器实际收到的 JSON 形状。

> 响应形状里的 `—` = handler 用变量/构造函数返回（如 `return cur`、`return sorted(...)`），
> 静态推不出字面量，不是"没有响应"。


### extract · 4 个端点（前端用到 3）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `POST` `/api/extract/cancel` | — | `{ok}` | ✅ | — | `backend_models/trt.py:362` |
| `GET` `/api/extract/progress` | — | `StreamingResponse()` | ✅ | — | `backend_models/trt.py:368` |
| `POST` `/api/extract/start` | `align`, `dst`, `fps`, `quality`, `rotate`, `size`, `skip`, `src`, `strict_align` | `{ok, src, dst, frames, fps, skip, rotate, align, quality, size, strict_align}` | ✅ | `{detail}` | `backend_models/trt.py:319` |
| `GET` `/api/extract/status` | — | `_job_status()` | ❌ | — | `backend_extract/routes.py:17` |

### dataset（含 oneshot 体检） · 15 个端点（前端用到 14）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `POST` `/api/dataset/apply` | `dir`, `mode`, `names` | `{ok, moved, items, failed, trash, mode}` | ✅ | — | `backend_oneshot/routes.py:910` |
| `POST` `/api/dataset/count` | `dir` | `{count, dir, n_files, ok, recursive_would_be, redirected_to_aligned, skipped_dirs, target}` | ✅ | — | `backend_oneshot/routes.py:80` |
| `POST` `/api/dataset/faceid/prewarm` | *(任意 JSON)* | `{ok, started, ready, warming, ms, alive}` | ✅ | — | `backend_oneshot/routes.py:409` |
| `GET` `/api/dataset/faceid/progress` | — | `{ok}` | ✅ | — | `backend_oneshot/routes.py:180` |
| `GET` `/api/dataset/faceid/warm` | — | `{ok, ready, warming, alive, boot_ms, engine_ms, warm_ms, provider, requests, error}` | ❌ | — | `backend_oneshot/routes.py:421` |
| `POST` `/api/dataset/mask` | `dir`, `model`, `res`, `workers` | `{ok, dir, workers, model, res}` | ✅ | `{detail}` | `backend_models/trt.py:267` |
| `GET` `/api/dataset/mask/progress` | — | `{running, pct, msg, done, total, speed, eta, stage}` | ✅ | — | `backend_models/trt.py:258` |
| `POST` `/api/dataset/merge-subdirs` | `dir` | `{ok, merged, conflicts, renamed, errors, removed, kept_empty, dirs}` | ✅ | — | `backend_oneshot/routes.py:434` |
| `POST` `/api/dataset/occlusion` | `action`, `dir`, `target`, `threshold` | `{ok, action, dir, target, threshold, scanned, count, moved, items, failed, trash, manifest, ms}` | ✅ | `{detail}` | `backend_oneshot/routes.py:790` |
| `POST` `/api/dataset/oneshot` | `bucket_floor`, `dir`, `level`, `score_pct` | `{ok, total}` | ✅ | `{detail}` | `backend_oneshot/routes.py:93` |
| `GET` `/api/dataset/oneshot/status` | — | — | ✅ | — | `backend_oneshot/routes.py:169` |
| `POST` `/api/dataset/restore` | `dir`, `items` | `{ok, restored, failed}` | ✅ | — | `backend_oneshot/routes.py:930` |
| `POST` `/api/dataset/scan` | `dir`, `eps`, `estimate`, `merge`, `min_samples`, `mode`, `percent`, `similarity`, `split_by_identity`, `threshold` | `{total, hits, count, unreadable, info, dir, n_files, redirected_to_aligned, skipped_dirs}` | ✅ | `{detail}` | `backend_oneshot/routes.py:560` |
| `POST` `/api/dataset/sort` | `by`, `dir` | `{ok, count, by}` | ✅ | — | `backend_oneshot/routes.py:712` |
| `GET` `/api/dataset/thumb` | — | `Response()` | ✅ | `{detail}` | `backend_oneshot/routes.py:833` |

### mask / image / progress / project · 14 个端点（前端用到 12）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/image/{index}` | — | `{index, name, total, image, mime, has_mask}` | ✅ | `{detail}` | `backend_mask/routes.py:68` |
| `GET` `/api/image/{index}/mask` | — | `{mask, w, h, source}` | ✅ | `{detail}` | `backend_mask/routes.py:112` |
| `POST` `/api/mask/bisenet` | *(任意 JSON)* | — | ✅ | — | `backend_mask/routes.py:102` |
| `POST` `/api/mask/gdino` | `box_threshold`, `image`, `text`, `text_threshold` | `{ok, boxes, scores, phrases, text, w, h, device, infer_ms, ms}` | ✅ | — | `backend_mask/routes.py:325` |
| `GET` `/api/mask/models` | — | `{kind, dir, files}` | ❌ | — | `backend_mask/routes.py:229` |
| `POST` `/api/mask/predict` | *(任意 JSON)* | — | ✅ | — | `backend_mask/routes.py:90` |
| `POST` `/api/mask/sam/load` | `image`, `variant` | `{ok, session, w, h, cached, variant, device, image, embed_ms, cached_count, ms}` | ✅ | `MaskErr` | `backend_mask/routes.py:247` |
| `POST` `/api/mask/sam/predict` | `box`, `multimask`, `points`, `session` | `{ok, masks, scores, w, h, multimask, infer_ms, ms}` | ✅ | `MaskErr` | `backend_mask/routes.py:275` |
| `GET` `/api/mask/sam/status` | — | `{ok, code, detail, alive, error}` | ❌ | — | `backend_mask/routes.py:343` |
| `POST` `/api/mask/save` | `image_index`, `index`, `mask` | `{ok, path, image_index, w, h, mode, bytes, embedded, xseg_mask, note}` | ✅ | `MaskErr` | `backend_mask/routes.py:165` |
| `POST` `/api/mask/text` | *(任意 JSON)* | — | ✅ | — | `backend_mask/routes.py:96` |
| `GET` `/api/progress/load` | — | `{index, dir}` | ✅ | — | `backend_mask/routes.py:148` |
| `POST` `/api/progress/save` | `index` | `{ok, index}` | ✅ | — | `backend_mask/routes.py:133` |
| `POST` `/api/project/open` | *(任意 JSON)* | `{path, count, resume, files}` | ✅ | `{detail}` | `backend_mask/routes.py:36` |

### merge · 43 个端点（前端用到 34）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/merge/analyze` | — | `{ok:true, total, step, scanned, model, detector, thresh, marks, truncated, error}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:733` |
| `GET` `/api/merge/detect/{idx}` | — | `{ok:true, idx, total, width, height, model, detector, thresh, boxes}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:707` |
| `POST` `/api/merge/export` | `aligned_dir`, `detector`, `dfm`, `encoder`, `fps`, `hwaccel`, `input_dir`, `model`, `output`, `output_dir` | `{ok:true, input_dir, aligned_dir, output, output_dir, dfm, model, settings, fps, workers, encoder, hwaccel, mask_onnx, detector, detector_engine}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:299` |
| `POST` `/api/merge/export-project` | `aligned_dir`, `batch`, `detector`, `force`, `fps`, `frames_dir`, `merged_dir`, `out`, `out_video`, `project` | `{ok:true, …}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1340` |
| `POST` `/api/merge/export-project/cancel` | — | `{ok:true, cancelled, running, already, reason}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1675` |
| `GET` `/api/merge/export-project/status` | — | — | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1639` |
| `POST` `/api/merge/export/cancel` | — | `{ok:true, cancelled, running, already, killed, reason}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:673` |
| `GET` `/api/merge/export/status` | — | — | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:487` |
| `GET` `/api/merge/face-debug` | *(任意 JSON)* | `_face_debug_payload()` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:265` |
| `POST` `/api/merge/face-debug` | *(任意 JSON)* | `_face_debug_payload()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:265` |
| `DELETE` `/api/merge/faces` | `ids`, `project` | `{ok, project, deleted, count, missing_files}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:839` |
| `GET` `/api/merge/faces` | — | `{ok, project, deduped, embedded, emb_have, emb_total, emb_failed, emb_pending}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:438` |
| `POST` `/api/merge/faces/add` | `box`, `detector`, `dfm`, `face_index`, `face_threshold`, `frame`, `group`, `idx`, `kps`, `kps68` | `{ok, project, created, dfm, group, group_name, group_dfm, matched, similarity, threshold, auto, source, size, align, landmarks_from, detector, mode, box, idx, replaced, id, frame, count, face, faces}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:643` |
| `GET` `/api/merge/faces/cache` | — | — | ❌ | `{ok:false,error,detail}` | `routes_faces.py:885` |
| `POST` `/api/merge/faces/extract` | `detector`, `dfm`, `frame`, `project` | `{ok, project, created, frame, size, align, detector, dfm, group, count, replaced, detected, skipped, faces}` | ❌ | `{ok:false,error,detail}` | `routes_faces.py:477` |
| `POST` `/api/merge/faces/groups` | `dfms`, `project` | `{ok, project}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:849` |
| `POST` `/api/merge/faces/move` | `group`, `ids`, `project` | `{ok, project, group, updated, faces}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:803` |
| `POST` `/api/merge/faces/restore` | `ids`, `project` | `{ok, project, updated, faces, missing_files}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:829` |
| `GET` `/api/merge/faces/source` | — | `{ok, available, mode, frame, why, project, default_project, material, material_mode, frames}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:777` |
| `GET` `/api/merge/faces/thumb` | — | `Response | _err()` | ✅ | — | `routes_faces.py:860` |
| `POST` `/api/merge/faces/trash` | `ids`, `project` | `{ok, project, updated, faces, missing_files}` | ✅ | `{ok:false,error,detail}` | `routes_faces.py:819` |
| `POST` `/api/merge/frame` | *(任意 JSON)* | `_frame_payload()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:244` |
| `GET` `/api/merge/frame/{idx}` | *(任意 JSON)* | `_frame_payload()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:217` |
| `POST` `/api/merge/frame/{idx}` | *(任意 JSON)* | `_frame_payload()` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:217` |
| `GET` `/api/merge/info` | — | `{ok:true, …}` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:800` |
| `POST` `/api/merge/merged` | `w` | `_merged_from_export()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:644` |
| `GET` `/api/merge/merged/{idx}` | `w` | `_merged_from_export()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:605` |
| `POST` `/api/merge/merged/{idx}` | `w` | `_merged_from_export()` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:605` |
| `POST` `/api/merge/open` | `fps`, `path` | `{ok:true, path, name, mode, frames, fps, width, height}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:164` |
| `POST` `/api/merge/prepare` | `dir`, `video` | `{ok:true, project, video}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:85` |
| `GET` `/api/merge/prepare/status` | — | `{…, running, success}`（`success` 是 `ok` 的别名） | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:154` |
| `POST` `/api/merge/preview` | `blur_mask_modifier`, `color_transfer_mode`, `detector`, `erode_mask_modifier`, `faces_project`, `mask_mode`, `mode`, `no_swap_faces`, `path`, `settings` | `{ok:true, png, ms, box, has_face, no_faces, swapped, t, duration, model, mode, color_transfer, erode_mask, blur_mask, …}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1004` |
| `POST` `/api/merge/preview-frame` | `detector`, `fmt`, `frame_file`, `image`, `no_swap_faces`, `path`, `quality`, `settings`, `thresh` | `{ok:true, png, …}` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:1217` |
| `GET` `/api/merge/preview/status` | — | `{ok:true, …}` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:1199` |
| `POST` `/api/merge/preview/warm` | `detector` | `{ok:true, warming, ready, ms, cold, model, dfm, detector, worker_pid, models, models_used, max_models, started, reason}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1290` |
| `GET` `/api/merge/preview/warm/status` | — | `{ok:true, …}` | ❌ | `{ok:false,error,detail}` | `backend_merge/routes.py:1333` |
| `POST` `/api/merge/project` | `path` | `{ok:true, path, kind, total, has_aligned, aligned_dir, frames_dir, with_landmarks, sampled, sample}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:770` |
| `POST` `/api/merge/reveal` | `path` | `{ok:true, path, dir}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1944` |
| `POST` `/api/merge/stage/{name}` | `dry_run`, `force`, `jpeg_quality`, `keep_cache`, `meta`, `png_level`, `range`, `range_changed`, `range_key`, `range_total` | `{ok:true, started, skipped, …}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1842` |
| `POST` `/api/merge/stage/cancel` | *(任意 JSON)* | `{ok:true, running, cancelled, stage, note}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1826` |
| `GET` `/api/merge/stage/status` | — | `{…, stage, stage_index, stage_pct, pct, frames_total, frames_done, frame_fps, eta_s}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:1691` |
| `GET` `/api/merge/video` | — | `StreamingResponse | JSONResponse()` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:922` |
| `GET` `/api/merge/videos` | — | `{ok:true, dir, videos, count, truncated}` | ✅ | `{ok:false,error,detail}` | `backend_merge/routes.py:45` |

### train / errface · 9 个端点（前端用到 8）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/errface/info` | — | `{**}` | ✅ | — | `backend_train/errface_routes.py:23` |
| `POST` `/api/errface/scan` | `dir`, `limit`, `percent`, `threshold` | `{ok, unavailable, capability, reason, install, status}` | ❌ | `{detail}` | `backend_train/errface_routes.py:33` |
| `POST` `/api/train/pause` | *(任意 JSON)* | `{ok, paused}` | ✅ | `{detail}` | `backend_train/routes.py:128` |
| `GET` `/api/train/preview` | — | `_pack | Response | read()` | ✅ | — | `backend_train/routes.py:235` |
| `POST` `/api/train/resume` | *(任意 JSON)* | `{ok, paused}` | ✅ | `{detail}` | `backend_train/routes.py:156` |
| `POST` `/api/train/save` | *(任意 JSON)* | `{ok, msg}` | ✅ | `{detail}` | `backend_train/routes.py:110` |
| `POST` `/api/train/start` | `cpu_only`, `name`, `options` | `{ok, name, running, device, device_banner}` | ✅ | `{detail}` | `backend_train/routes.py:51` |
| `GET` `/api/train/status` | — | `{ok, running, name, iter, error, log, it_s, elapsed_s, target_iter, eta_s, iter_seen, paused, saving, stopping, last_backup_iter, lr, loss_src, loss_dst, history, warnings, has_preview, preview_iter, device}` | ✅ | — | `backend_train/routes.py:312` |
| `POST` `/api/train/stop` | *(任意 JSON)* | `{ok, graceful, msg}` | ✅ | — | `backend_train/routes.py:183` |

### models / model_spec / model · 12 个端点（前端用到 10）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/model_spec` | — | — | ✅ | — | `backend_models/routes.py:214` |
| `POST` `/api/model/preload` | *(任意 JSON)* | `{ok, note}` | ✅ | — | `backend_mask/routes.py:107` |
| `GET` `/api/models` | — | `{models, training_pid}` | ✅ | — | `backend_models/routes.py:681` |
| `GET` `/api/models/{name}` | — | `{name, dat, iter, options, readonly, legacy, model}` | ❌ | `{detail}` | `backend_models/routes.py:51` |
| `GET` `/api/models/{name}/config` | — | `{name, iter, options, readonly, legacy}` | ✅ | `{detail}` | `backend_models/routes.py:25` |
| `POST` `/api/models/{name}/config` | *(任意 JSON)* | `{ok, name, changed, count, hot, need_restart, hot_applied, hot_error, legacy}` | ✅ | `{detail}` | `backend_models/routes.py:78` |
| `POST` `/api/models/{name}/delete` | — | `{ok, name, removed, count}` | ✅ | `{detail}` | `backend_models/routes.py:162` |
| `POST` `/api/models/{name}/export` | — | `{ok, name, arch, running}` | ✅ | `{detail}` | `backend_models/routes.py:187` |
| `GET` `/api/models/{name}/preview.png` | — | `FileResponse()` | ❌ | `{detail}` | `backend_models/routes.py:741` |
| `GET` `/api/models/{name}/stats` | — | `{name, gpu, disk_free_gb, model_gb, cpu_percent, ram_used_gb, ram_total_gb, ram_percent, backup_count, has_backup_dir, backup_interval, max_backups}` | ✅ | `{detail}` | `backend_models/routes.py:500` |
| `POST` `/api/models/create` | `crash_threshold`, `gradient_checkpointing`, `lr`, `name`, `optimizer`, `options`, `preset`, `pretrain_single_decoder`, `spec`, `use_bf16` | `{ok, name, preset, clamped, dat, options, note}` | ✅ | `{detail}` | `backend_models/routes.py:355` |
| `GET` `/api/models/export/status` | — | `{ok, running, name, error, log}` | ✅ | — | `backend_models/routes.py:206` |

### bili / auth · 15 个端点（前端用到 13）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/auth` | — | `{logged_in, uid, name, avatar, since, onboarded}` | ✅ | — | `backend_bili/account.py:49` |
| `POST` `/api/auth/login` | `avatar`, `name`, `uid` | `{ok}` | ✅ | — | `backend_bili/account.py:62` |
| `POST` `/api/auth/logout` | — | `{ok}` | ✅ | — | `backend_bili/account.py:86` |
| `POST` `/api/auth/skip` | — | `{ok}` | ✅ | — | `backend_bili/account.py:77` |
| `POST` `/api/bili/batch` | `bvids`, `quality`, `workers` | — | ✅ | `{detail}` | `backend_bili/routes.py:258` |
| `POST` `/api/bili/batch/cancel` | — | `{cancelled, done, elapsed, error, eta, face_rate, faces, faces_new, finished, fps, …}` | ✅ | — | `backend_bili/routes.py:299` |
| `POST` `/api/bili/batch/pause` | *(任意 JSON)* | `{cancelled, done, elapsed, error, eta, face_rate, faces, faces_new, finished, fps, …}` | ✅ | — | `backend_bili/routes.py:292` |
| `GET` `/api/bili/batch/status` | — | `{cancelled, done, elapsed, error, eta, face_rate, faces, faces_new, finished, fps, …}` | ✅ | — | `backend_bili/routes.py:286` |
| `POST` `/api/bili/download` | `bvid`, `quality` | `{ok, path, size}` | ❌ | `{detail}` | `backend_bili/routes.py:136` |
| `POST` `/api/bili/logout` | — | `{ok}` | ✅ | — | `backend_bili/routes.py:204` |
| `GET` `/api/bili/me` | — | `{logged_in, name, uid, avatar, level, code, message}` | ✅ | `_err` | `backend_bili/routes.py:179` |
| `GET` `/api/bili/me/detail` | — | `{ok, user}` | ❌ | `{detail}` | `backend_bili/routes.py:305` |
| `POST` `/api/bili/poll` | *(任意 JSON)* | `{status, name, uid, avatar}` | ✅ | `{detail}` | `backend_bili/routes.py:108` |
| `POST` `/api/bili/qr` | — | `{ok, key, qr}` | ✅ | `{detail}` | `backend_bili/routes.py:95` |
| `POST` `/api/bili/up/videos` | `mid`, `page`, `ps` | — | ✅ | `{detail}` | `backend_bili/routes.py:214` |

### trt — devices / build / engines / perf / weights · 8 个端点（前端用到 7）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/build` | — | `{ts, iso, files}` | ✅ | — | `backend_models/trt.py:76` |
| `GET` `/api/devices` | — | `{gpus, cpu}` | ✅ | — | `backend_models/trt.py:47` |
| `GET` `/api/engines` | — | — | ✅ | — | `backend_models/trt.py:98` |
| `POST` `/api/engines/build` | `name` | `{ok, name}` | ✅ | `{detail}` | `backend_models/trt.py:166` |
| `POST` `/api/perf/bench` | *(任意 JSON)* | `{ok, rows}` | ❌ | — | `backend_models/trt.py:186` |
| `POST` `/api/weights/fetch` | `group`, `url_base` | `{ok, job, group, files}` | ✅ | `{detail}` | `backend_models/weights.py:373` |
| `GET` `/api/weights/fetch/status` | — | — | ✅ | — | `backend_models/weights.py:417` |
| `GET` `/api/weights/status` | — | `{ok, dir, manifest, verify, packages, total_mb, fetch}` | ✅ | — | `backend_models/weights.py:232` |

### state / remote / settings · 4 个端点（前端用到 3）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/api/remote` | — | `{enabled, port, ip, url}` | ✅ | — | `app.py:207` |
| `POST` `/api/remote` | `enabled`, `port` | `{enabled, ip, port, url}` | ✅ | `{detail}` | `app.py:217` |
| `POST` `/api/settings` | *(任意 JSON)* | — | ❌ | — | `app.py:140` |
| `GET` `/api/state` | — | `{version, runtime, ports_status, datasets, models, ports, capabilities}` | ✅ | — | `backend_dataset/routes.py:17` |

### 其它顶层（root / face / convert / gpu） · 8 个端点（前端用到 8）

| 端点 | 请求体关键字段 | 响应形状 | 前端 | 错误口径 | 源码 |
|---|---|---|---|---|---|
| `GET` `/` | — | `Response()` | ✅ | — | `app.py:91` |
| `GET` `/api/convert/list` | — | `{ok, items}` | ✅ | — | `backend_oneshot/routes_convert.py:13` |
| `POST` `/api/convert/start` | `name` | `{count, error, modules, name, ok}` | ✅ | `{detail}` | `backend_oneshot/routes_convert.py:18` |
| `GET` `/api/convert/status` | — | `{boot_ms, cpu_note, dir, idle_s, max_models, models, on_cpu, pid, providers, running, …}` | ✅ | — | `backend_oneshot/routes_convert.py:32` |
| `DELETE` `/api/face` | — | `{ok}` | ✅ | — | `routes_face.py:43` |
| `GET` `/api/face` | — | `{points}` | ✅ | — | `routes_face.py:25` |
| `POST` `/api/face` | `points` | `{ok}` | ✅ | `{detail}` | `routes_face.py:33` |
| `GET` `/api/gpu` | — | `{gpus, best, recommend, presets}` | ✅ | — | `backend_models/routes.py:226` |

---

## 3. 三张对账清单

### A · 前端调了、后端没有（4 条 —— 全是 mask-editor 的死方法）

| 前端调用 | 位置 | 后端现状 | 判定 |
|---|---|---|---|
| `POST /api/mask/commit` | `app/web/mask-editor/js/api.js:161-163` | 不存在（只有 `POST /api/mask/save`） | **死代码** |
| `POST /api/mask/undo` | `app/web/mask-editor/js/api.js:168-170` | 不存在 | **死代码** |
| `POST /api/xseg/predict` | `app/web/mask-editor/js/api.js:148-150` | 不存在（4.1 把 XSeg 并进 `mask/predict`） | **死代码** |
| `POST /api/xseglite/predict` | `app/web/mask-editor/js/api.js:151-153` | 不存在 | **死代码** |

**关键补充（别误判成 bug）**：这 4 个方法在 mask-editor 里**没有任何调用点** ——
grep `commitMask|undoMask|xsegPredict|xseglitePredict` 在 `mask-editor/js/**` 只命中 `api.js` 自身。
所以它们**不会**在界面上触发 404。它们是从 `app/docs/legacy-reference/MaskProcessor/ui/js/api.js`
搬过来时**多搬了一层**：`backend_mask/routes.py` 只移植了 15 条（见 `backend_mask/MAPPING.md`），
客户端却留着 17 个方法 —— 实际只有 9 个被自家 UI 用到。

> 一句话：**api.js 是后端能力集的超集**。要么删这 4 个方法，要么补 UI；放着不管，
> 下一个人照着 `api.js` 写按钮就会 404。

### B · 后端有、前端从没调（20 条）

**B1 · 只给 `app/tools` 验收脚本用（6 条，不能删）**

| 端点 | 谁在用 |
|---|---|
| `GET /api/extract/status` | `app/tools/aligned444_accept.py` |
| `GET /api/merge/info` | `app/tools/accept/test_merge_api.py` |
| `GET /api/merge/preview/status` | `vram_force_accept.py`、`vram_plan_accept.py`、`vram_reclaim_accept.py`、`vram_export_ab.py` |
| `GET /api/merge/preview/warm/status` | `preview_fast_accept.py` |
| `POST /api/merge/preview-frame` | `preview_fast_accept.py`、`stage6_accept.py`、`stage_export_accept.py` |
| `POST /api/merge/faces/extract` | `test_faces_api.py` |

**B2 · 真死接口（14 条）**

| 端点 | 死因 | 建议 |
|---|---|---|
| `GET /api/bili/me/detail` | 前端只读 `/api/bili/me`（字段已够） | 删或标 deprecated |
| `POST /api/bili/download` | 单 BV 下载；界面走 `/api/bili/batch` | 保留给 CLI，标注"非 UI 接口" |
| `GET /api/dataset/faceid/warm` | 前端只轮询 `/api/dataset/faceid/progress` | 删（与 progress 重复） |
| `GET /api/mask/models?kind=` | mask-editor 的模型列表写死在前端 | 保留（选模型时会用） |
| `GET /api/mask/sam/status` | 前端用 404/501 探测 SAM 是否可用（`api.js:55 isUnsupported`） | 保留，但这个设计值得改 —— 见 §5 |
| `GET /api/merge/face-debug` | 前端只用 POST 版 | 删 GET 装饰器 |
| `GET /api/merge/faces/cache?fresh=` | 无人调用 | 删 |
| `GET /api/models/{name}` | 前端用 `/api/models` + `/config` + `/stats` | 保留 |
| `GET /api/models/{name}/preview.png` | 预览图走 data URL，无人取 raw PNG | 保留（外链用） |
| `POST /api/errface/scan` | 前端只读 `/api/errface/info` | 保留（扫描是 info 的懒触发） |
| `POST /api/merge/frame/{idx}` | 前端只 GET `{idx}`、只 POST 无参版 | **删**（三个入口里最少人用的） |
| `POST /api/merge/merged/{idx}` | 同上 | **删** |
| `POST /api/perf/bench` | 无 UI | 保留给排障 |
| `POST /api/settings` | 无人调用（端口设置在别处） | **删或接到设置页** |

### C · 路径匹配、method 不同（0 条）

**没有一条真错配**。三个"看起来像"的，实际是**故意成对注册**（同一个函数挂两个装饰器）：

- `GET` + `POST` `/api/merge/frame/{idx}` → `merge_frame`（`backend_merge/routes.py:210-217`）
- `GET` + `POST` `/api/merge/merged/{idx}` → `merge_merged_frame`
- `GET` + `POST` `/api/merge/face-debug` → `merge_face_debug`

前端在这三处**只用了一个方向**（frame 用 GET、merged 用 GET、face-debug 用 POST），另一个方向就落进 B 清单。
**这是"多入口"设计留下的垃圾，不是 bug。**

---

## 4. 同类接口的不同协议（逐字段实证）

### 4.1 三个"导出"，三套字段名

| 语义 | `POST /api/merge/export` | `POST /api/merge/export-project` | `POST /api/merge/stage/{frames…encode}` |
|---|---|---|---|
| 输入 | `input_dir` / `project` | `project_dir` / `project` | `video`（源视频路径） |
| 帧目录 | 同 `input_dir` | `frames_dir`（默认 = project） | 隐式 `cache_dir` |
| 对齐集 | `aligned_dir` | `aligned_dir` | 隐式（cache 内） |
| 输出 | `output` / `output_dir` | `out_video` | `out`（只有 encode 段） |
| 模型 | `dfm` / `model` | `dfm` / `model` / `models[]` | `dfm` / `model` / `models[]` |
| 全量重做 | ✗ | `force` | `force` |
| 帧范围 | ✗ | ✗ | `range` / `start_frame`+`end_frame` |
| 质量 | ✗ | `quality` | `jpeg_quality` / `png_level` |
| 并发 | `workers` | `workers` + `batch` | `workers` + `batch` |
| 状态 | `GET /export/status` | `GET /export-project/status` | `GET /stage/status` |
| 取消 | `POST /export/cancel` | `POST /export-project/cancel` | `POST /stage/cancel` |

**同一件事（"把帧合成为 mp4"）三条路、四套字段名。** 前端的写法也跟着分叉：
`app.js:8095/8289/8511` 发 `body.output_dir`，`app.js:6485` 发 `body.out_video`。

**最刺眼的一处**（`backend_merge/routes.py:478-479`）：

```python
return _ok(input_dir=str(ip), aligned_dir=str(aligned), output=str(out_mp4),
           output_dir=str(out_mp4), ...)
```

`output` 和 `output_dir` **是同一个值**，而且都是**文件路径**（不是目录）。
`output_dir` 这个键名在这一层是**假的**。前端 `app.js:8528` 读的是 `d.output || d.input_dir`，
所以 `output_dir` 至今没人读 —— 删它零风险。

### 4.2 `ok` 有**四种**语义

| 语义 | 出现处 | 前端怎么用 |
|---|---|---|
| ① 请求成功 | `_ok()` 包装的 39 个端点 | `if (d.ok) …` |
| ② **后台任务**成功 | `GET /api/merge/export/status`、`/prepare/status`（`routes.py:487-488` 注释、`routes_merge.py:24-27` 文档） | `app.js` 拿 `d.ok` 决定弹"合成完成"还是"合成失败" |
| ③ **训练循环**状态 | `GET /api/train/status` 响应里的 `ok` 键 | 同名字段、另一含义 |
| ④ 请求被接受 | `POST /api/dataset/oneshot` 直接返回 `{ok, total}`（没走 `_ok`） | 同上，但形状是裸 dict |

②③ 是**改不动的**：把 `export/status` 的 `ok` 改成恒 `true`，
每一次失败的导出都会在界面上弹"合成完成"（`routes_merge.py:25-27` 已写明这个坑）。

### 4.3 错误口径：39 vs 90

| 家族 | handler 数 | 抛错后浏览器收到 | 代表代码 |
|---|---|---|---|
| `@_api` 包装（`_common/http.py:62`） | **39** | `{"ok":false,"error":"…","detail":"…"}` + 原状态码 | `backend_merge/routes.py` 全部 29、`routes_faces.py` 10 |
| 未包装（直接用 `HTTPException`） | **90** | `{"detail":"…"}`（FastAPI 默认） | 其余 14 个文件 |

未包装的 90 个分布 —— 这就是"改哪几行"的清单：

```
app.py                            4    backend_models/routes.py          12
backend_bili/account.py           4    backend_models/trt.py             10
backend_bili/routes.py           11    backend_models/weights.py          3
backend_dataset/routes.py         1    backend_oneshot/routes.py         13
backend_extract/routes.py         1    backend_oneshot/routes_convert.py  3
backend_mask/routes.py           15    backend_train/errface_routes.py    2
backend_train/routes.py           7    routes_face.py                     3
routes_faces.py                   1   （face_thumb 返回 Response，故意不包）
```

前端已经**同时**兜两套（这是"协议不统一"的前端账单）：

- `app/web/js/app.js`：**48 处** `.detail`
- `app/web/js/face_db.js:99`：`throw new Error(d.error || d.detail || ("HTTP " + r.status))`
- `app/web/mask-editor/js/api.js:35`：`data.detail || data.error || data.message`

### 4.4 删除有四种写法

| 写法 | 端点 |
|---|---|
| 动词进路径 | `POST /api/models/{name}/delete` |
| REST + body | `DELETE /api/merge/faces`（body 带 `project` + `ids`） |
| REST 无 body | `DELETE /api/face` |
| 软删 + 恢复 | `POST /api/merge/faces/trash` / `…/restore` |

### 4.5 两个"face"资源，毫不相干

| 端点 | 是什么 | 响应 |
|---|---|---|
| `GET/POST/DELETE /api/face`（`routes_face.py`） | **68 个关键点**，存 `workspace/face68.json` | `{points: [...]}` / `{ok:true}` / `{ok:true}` |
| `GET/POST/DELETE /api/merge/faces`（7+ 端点） | **人脸库**（缩略图、分组、回收站） | `{ok, project, deduped, embedded, …}` |

`GET /api/face` **没有 `ok` 键**（失败时返回 `{points: null}`，HTTP 200）——
这是全站唯一一个"用 200 + 空数据表达失败"的接口，前端 `app.js:8990` 得单独判 `d.points == null`。

### 4.6 帧号：0 起 vs 1 起

| 处 | 基准 | 证据 |
|---|---|---|
| `GET /api/merge/frame/{idx}` | **0 起** | `backend_merge/compose.py:761`：`"帧号越界：%d，这里只有 %d 帧（0~%d）"` |
| `POST /api/merge/stage/{name}` 的 `start_frame`/`end_frame` | **1 起、闭区间** | `pipeline_stages.py:3404` 注释；`:3421` `start_frame 从 1 数起` |

同一个合成页面上，**时间轴取帧从 0 数，帧范围输入框从 1 数**。验收脚本 `test_merge_api.py`
用 `/api/merge/frame/0`，`frame_range_accept.py` 用 `start_frame: 1` —— 两边都对，因为后端两套都在。

### 4.7 `settings` 一个对象，三代键名并存

`_common/merge_run.py:53-76` 一次认三套：

| 4.1 现用 | 别名 1 | 别名 2（4.0 原味） |
|---|---|---|
| `blend` | `mode` | — |
| `ct_mode` | `color_transfer_mode` | `color_transfer` |
| `mask_erode` | `erode_mask` | `erode_mask_modifier` |
| `mask_blur` | `mask_feather` | `blur_mask_modifier` |

注释里已写明为什么不能删别名："只转发别名的话，别名给的数会被那个 0 盖掉，等于没传。"
**这是"协议不统一"里最贵的一处**：三套名字 × 4 个旋钮 = 12 种写法，任何一种写错都**静默生效为默认值**。

### 4.8 分页 / 范围参数

全站 **0 处分页**（没有 `limit` / `offset` / `page` 参数）。范围参数有四种互不相干的口径：

| 用途 | 参数 | 位置 |
|---|---|---|
| 帧区间 | `start_frame` / `end_frame`（1 起闭区间） | body |
| 图片宽度 | `w`（200–1600，夹取） | query |
| 缩略图边长 | `s`（默认 96） | query |
| 日志尾巴 | `tail`（行数） | query |

唯一一处真分页是 `POST /api/bili/up/videos` 的 `page`/`ps` —— 那是**B 站 API 原样透传**，不算我们的口径。

### 4.9 同一资源的多个入口

| 资源 | 入口数 | 列表 |
|---|---|---|
| 合成一帧 | **3** | `GET frame/{idx}` · `POST frame/{idx}` · `POST frame`（idx 在 body） |
| 成片一帧 | **3** | `GET merged/{idx}` · `POST merged/{idx}` · `POST merged` |
| 预览 | **3** | `POST preview`（按时刻）· `POST preview-frame`（按帧文件）· `POST stage/merge` |
| 导出 | **3** | `export` · `export-project` · `stage/encode` |
| 人脸库 | **8** | `/api/merge/faces`（GET/DELETE）+ add/extract/groups/move/trash/restore/thumb/cache |
| 训练预览 | **2** | `GET /api/train/preview`（代理 4.0）· 4.0 自己的 `/preview`（前端不直接碰） |

### 4.10 路径参数占位名三种写法

`{idx}`（`/api/merge/frame/{idx}`、`/api/merge/detect/{idx}`）、
`{name}`（`/api/merge/stage/{name}`、`/api/models/{name}`）、
`{index}`（`/api/image/{index}`）。同一个概念三个名字 —— 对生成客户端代码不友好，对人无所谓。

### 4.11 响应类型随 query 变（不是 bug，但很容易误判）

`GET /api/merge/frame/{idx}` 的 `fmt=png` + `raw=1` **仍然返回 JSON**
（`{png: "data:image/png;base64,…", idx, …}`），不是 image/png 字节流。
`app.js:2925` 用 `r.json()` 读它**是对的**。
真返回字节的是 `GET /api/merge/video`（206 + Range）、`GET /api/merge/faces/thumb`、`GET /api/dataset/thumb`。

### 4.12 `png` 这个键里装的是 JPEG

`POST /api/merge/preview`（`routes.py:1164`）与 `POST /api/merge/preview-frame`（`routes.py:1272`）：

```python
resp = _ok(png="data:image/jpeg;base64," + _b64.b64encode(jpg).decode("ascii"), ms=…)
```

键叫 `png`，值是 **JPEG** 的 data URL（MIME 自己写着 `image/jpeg`）。前端不校验 MIME，
直接塞进 `<img src>` 所以没炸；但任何按 `.endswith(".png")` 或按 MIME 落盘的消费方都会存出错扩展名。

同一份响应里还把**两代字段名并排返回**（`:1174-1176`）：

| 4.0 原味键 | 4.1 现用键 |
|---|---|
| `mode` | （同义：`blend` 的旧名） |
| `color_transfer` | （同义：`ct_mode` 的旧名） |
| — | `erode_mask` |
| — | `blur_mask` |

所以一个响应体里同时出现 `mode` + `color_transfer`（旧）和 `erode_mask` + `blur_mask`（新）。
**四个旋钮，两套命名，一半新一半旧** —— 这正是"协议不统一"最难改的一种：不是错，是两种约定在同一行代码里共存。

---

## 5. 统一建议

### 5.1 能低成本统一（按性价比排序）

| # | 动作 | 改动量 | 风险 | 收益 |
|---|---|---|---|---|
| 1 | 删 mask-editor 的 4 个死方法（`mask-editor/js/api.js:148-153`、`161-170`） | 4 个函数 | 0（无调用点） | A 清单归零 |
| 2 | 删 `backend_merge/routes.py:479` 的 `output_dir=` | 1 行 | 0（前端只读 `output`） | 去掉"键名撒谎" |
| 3 | 给 `/api/face` 的 GET 补 `ok:true`（`routes_face.py:25-31`），失败改 404 而非 `{points:null}` | 3 行 | 低（`app.js:8990` 需同步判 `d.ok`） | 去掉全站唯一的 200+null |
| 4 | 给 `backend_extract/routes.py:17` 的 `/api/extract/status` 标 deprecated，注释指向 `/api/extract/progress` | 1 行注释 | 0 | B 清单少 1 条 |
| 5 | `POST /api/settings` 接进设置页，或删掉 | 二选一 | 0 | 去掉"有口子没人用" |
| 6 | 把 `_api` 铺到剩余 14 个文件 | 每个 handler 1 行装饰器，共 90 行 | **中** —— 见 5.2 | 错误口径从 2 套收敛成 1 套 |

### 5.2 铺 `_api` 的正确做法（有前提）

`@_api` 会把 `HTTPException` 转成 `{"ok":false,"error","detail"}`，
状态码保留、`detail` 保留 —— 所以**现有前端不会坏**（它们读 `d.detail`）。要小心的只有两处：

1. `GET /api/merge/export/status`、`/prepare/status`：`ok` 是**任务**语义，
   **不要**顺手改成恒 `true`。它们现在已在 `@_api` 里，靠 handler 自己 `_ok(…任务状态… )` 传值 —— 保持。
2. `GET /api/train/status`：响应体自带 `ok`（训练循环），套 `@_api` 后**顶层 ok 不会出现**
   （`@_api` 只在出错时替换整个 body），所以安全；但要写进注释，否则下一个人会以为冲突。

### 5.3 不能动（前端已依赖）

| 不能动 | 为什么 |
|---|---|
| `ok` 的双/三语义 | `app.js` **101 处** `.ok` 判定；`export/status` 的 `ok` 决定弹窗文案 |
| `settings` 的三代别名 | 旧前端、验收脚本、已存的 `meta.json` 都在发旧名；删了会静默退回默认值 |
| `frame/{idx}` 0 起 | 时间轴、`test_merge_api.py::/frame/0` 都按 0 |
| 90 个 `{detail}` 接口 | `app.js` **48 处** `.detail` 兜底；先铺 `_api`（保留 detail）而不是直接换字段 |
| `_ok()` 里同时给 `error` 和 `detail` | `_common/http.py:39` 就是为兼容旧前端留的 |

---

## 6. 差集恒等式

```
后端 operation : 132
前端 distinct  : 116   （120 个调用点去重后）
A 前端有后端无 : 4
C method 错配  : 0
B 后端有前端无 : 20

132 - 116 + 4 = 20    ✓
```

---

## 7. 分层巡检：`app/web` 之外散落的前端

### 7.1 结论表

| 位置 | 文件数 / 体积 | 是什么 | 判断 |
|---|---|---|---|
| `app/engine/MergeStudio/ui/` | 10 文件 / 147KB | 4.0 MergeStudio 的**独立 Web UI**（英文、v1.0） | **该删/该归档**：它的 API 服务已归档到 `app/docs/legacy-reference/MergeStudio/api`（`export_pipeline.py:794` 注释），4.1 全仓 **0 处引用** |
| `app/tools/parts/` | 11 文件 / 80KB | 早期"片段"稿：`merge-tracks.js`、`merge-pages.js`、`train-params.js`、`mask-draw.*` 及配套 css/html | **该移**：`merge-*.js` / `train-params.js` 已**逐字内联**进 `app/web/js/app.js`（`app/docs/notes/README.md` 记录 3/3 命中），css 已内联进 `app.css`；但 `mask-draw.html` / `mask-draw.js` **找不到逐字副本**，不能直接删 |
| `app/engine/WebUI/page_*.py` | 4 文件内联 HTML | 4.0 训练进程的页面（trainer / settings / sample_viz / flit_game） | **vendored，不动**。由 `engine/mainscripts/Trainer.py:128` 在**训练子进程**里起 HTTP 服务，4.1 只经 `/api/train/preview` 代理 |
| `app/engine/samplelib/page_sample_viz.py` | 1 文件内联 HTML | 样本可视化页（4.0 遗留） | **vendored，不动** |
| `app/docs/legacy-reference/MaskProcessor/ui/` | 10 文件 / 125KB | mask-editor 的**上一代** | **保留归档**（§3-A 那 4 个死方法的来源，对照用） |
| `app/docs/00-ui-mockup.html` | 1 文件 / 51KB | 设计稿（`UI设计规范.md` 的点阵原型） | **保留**（文档资产） |
| `desktop/main.js` + `preload.js` | 2 文件 / 6KB | Electron **主进程 + preload**，不是渲染层 | **不动**：它 `loadURL("http://127.0.0.1:" + port + "/")`，渲染层完全由 `app/web` 提供 —— 分层是对的 |
| `app/web/preview-nodes.html` | 1 文件 / 1.2KB | "变体 B 神经网络星图"设计稿，`import '/static/js/welcomenodes.js'` | **在前端目录里但没人引用**（全仓 0 命中）→ 挪到 `app/docs/` 或删 |

### 7.2 顺带发现的三处"代码放错模块"

1. **extract 家族被劈成两半**：`POST /api/extract/start`、`/cancel`、`GET /progress` 在
   `backend_models/trt.py:319-368`，而 `GET /api/extract/status` 在 `backend_extract/routes.py:17`。
   同一前缀跨两个包，且**活在 trt 里的那三条才是前端在用的那三条**。
2. **`POST /api/dataset/mask`、`/api/dataset/mask/progress` 也在 `backend_models/trt.py`**。
3. `app/server/routes_merge.py`（16KB）已退化成纯转发壳（`routes_merge.py:65-70` 自述），
   真正实现在 `backend_merge/*`；`MAPPING.md` 里有旧行号对照。这是**做对的一次拆分**，当模板用。

### 7.3 一句提醒

本次审计开始时仓库根还有 `docs/` 与 `legacy/` 两个目录，过程中已不存在（应为并行进程移动/删除，非本次操作）。
本文落在 **`app/docs/33-API-CONTRACT.md`** —— 全仓编号文档（00–32）都在 `app/docs/`。

---

## 8. 收敛记录（本文之后落地的改动）

这一节只记**已经落了什么**和**还剩什么**。口径与本文 §1 一致，但现在可以一键重跑：

```
runtime\server\Scripts\python.exe app\tools\audit\api_contract_audit.py
```

脚本扫 `app/web` 的 `fetch()` / `new EventSource()` / `.src =`，与 `app.openapi()`
比对（并用 ast 数装饰器交叉验证 132 = 132），输出 A / B / C 三张清单 + D（URL 静态
解不出的 fetch）。退出码：A 或 C 非空 = 1，D 非空 = 2，后端自检失败 = 3。

### 8.1 改前 → 改后

| 项 | 改前 | 改后 |
|---|---|---|
| 后端 path / operation | 124 / 132 | 124 / 132（零增删） |
| 前端 distinct (method, path) | 114 | 110 |
| **A 前端调了、后端没有** | **4**（全是 mask-editor） | **0** |
| **C method 错配** | **0** | **0** |
| **D URL 静态解不出** | **0** | **0** |
| B 后端有、前端没调 | 22 | 22（见 8.3） |

### 8.2 落地的五件

1. **mask-editor 的幻影端点归零**（§3-A 那 4 条）
   - `commitMask` / `undoMask`：**真死代码**（全仓 0 调用点）→ 删。
   - `xsegPredict` / `xseglitePredict`：**不是死代码** —— `mask-editor/index.html:185-186`
     两个按钮在调它们（§3-A 的 grep 只扫了 `mask-editor/js/**`，漏了 index.html）。
     后端从没移植过 `/api/xseg/predict`、`/api/xseglite/predict`，而 4.1 已经把它并进
     `POST /api/mask/predict`（`backend_mask/routes.py:88`，501 占位）。
     → 合成 1 个 `xsegPredict(index, variant, res)` 打 `/mask/predict`：两个按钮照旧在，
     报错从「404 幻觉」变成后端那句「要加载 AI 模型（SAM / BiSeNet / XSeg）」。

2. **`POST /api/merge/export` 回包里删掉 `output_dir`**（§4.1 那个「键名撒谎」）。
   `output` 保留（`app.js:8528` 读的就是它）。实测老项目模式响应：`output` 在、
   `output_dir` 不在；同一张表里的视频模式（`pipeline_video.py:488/511`）**没动**。

3. **`GET /api/face` 收进统一口径**（§4.5 那个「唯一 200 + null」）
   - 成功：`{ok: true, points: [...]}`（`points` 原样）。
   - 失败：**404 + `{detail}`**，不再是 `200 + {points: null}`。
   - 前端 `app.js:8990` 改成先看 `r.ok`；404 时 toast「这一帧没有关键点」+ 后端 detail。

4. **`GET /api/extract/status` 标 deprecated**（只加注释 + 响应里一个 `deprecated` 字段，
   行为零改动）。前端用的一直是 `GET /api/extract/progress`。

5. **三处错位文件归位**（git mv，历史保留）
   - `app/engine/MergeStudio/ui/`（10 文件）→ `app/docs/legacy-reference/MergeStudio-ui/`
   - `app/tools/parts/`（11 文件，含 `mask-draw.*` 那两份**唯一副本**）→
     `app/docs/legacy-reference/parts/`
   - `app/web/preview-nodes.html` → `app/docs/legacy-reference/preview-nodes.html`

### 8.3 还剩的协议不一致（**明确不动**，留给人拍板）

| 项 | 为什么不动 |
|---|---|
| 90 个端点还是 `{detail}`，39 个是 `{ok,error,detail}` | 前端 **48 处** `.detail` 兜底；铺 `@_api` 是 90 行改动 + 全局出错口径变化，风险不对等 |
| `ok` 的四种语义（§4.2） | `export/status` 的 `ok` 决定弹「合成完成/失败」，改成恒 true 会把每次失败都说成成功 |
| `settings` 三代别名（§4.7） | 旧前端、验收脚本、已存 meta.json 都在发旧名，删了静默退回默认值 |
| 帧号 0 起 / 1 起两套（§4.6） | 时间轴与验收脚本各按一套，改一边就崩一边 |
| 三个「导出」四套字段名（§4.1） | 要动的是三条链路的入参，不是改个键名 |
| `png` 键里装 JPEG（§4.12） | 消费方都不校验 MIME，改键名反而会打断它们 |
| B 清单 22 条「后端有、前端没调」 | 其中 6 条验收脚本在用（不能删），其余是排障口子（`perf/bench`、`settings`、`mask/models`…） |

> B 清单 22 与 §3-B 的 20 的差：§3 把「资源加载」（`<img src>`）也算进"前端在用"，
> 脚本算得更细 —— `/api/merge/faces/thumb`、`GET /api/merge/face-debug` 这两条
> 在脚本里仍留在 B，并标了「资源加载」。

