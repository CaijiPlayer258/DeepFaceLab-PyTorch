# -*- coding: utf-8 -*-
"""合成器路由（从 app.py 按路径前缀抽出）。

这一版把合成工作室（Merger / 合成工作室）里"参数进来了却没生效、坏了还不报"的地方
修掉，全部改动都在这一个文件里：

  1) 入参校验：fps / workers / encoder / hwaccel / input_dir / dfm / model / settings /
     thresh / step / xseg_onnx 一律**先校验再动后台线程**。非法值给 400 + 中文说明，
     不再掉进 500，也不再"静默退回默认值"（以前选中的模型没导出 .dfm 会被悄悄换成
     列表里第一个 .dfm，成片用的是别的脸，谁都不知道）。
  2) 缓存与并发：_merge_cache 的读写统一走 _cache_lock，每个请求先取一份快照再算，
     避免 mode / names / srcdir 分别取自两次 update 之间的混合状态；frame / merged /
     detect / analyze 对越界、缺文件、缓存过期分别给 404 / 410 / 422，不再 500。
  3) 导出状态：_merge_job["log"] 本来就是子进程 stdout+stderr 合流，失败时把尾巴挑出来
     放进 error / error_detail / log_tail；终态时 stage 归 -1、pct 归 0、msg 写"合成失败"，
     不卡在中间值。
  4) 取消幂等：重复取消照样 200；取消旗子先立（子进程还没起来也能取消），看门狗负责
     补刀和兜底收尾，终态落成 cancelled，不会永远 running。
  5) 路径统一：相对路径按仓库根解析，反斜杠/正斜杠/引号/结尾分隔符全部归一，Windows
     大小写用 os.path.normcase 比较（见 _norm_path / _same_path）。
     app/server/_paths.py 只做引擎根解析、没有通用的路径规范化函数（本次该文件只读），
     所以这里复用它的 ROOT 当相对路径基准，规范化在本地实现。
  6) 响应体统一：成功 {"ok": true, ...}；失败 {"ok": false, "error": "...", "detail": "..."}
     （detail 只是兼容旧前端 d.detail || "…" 的写法）。
     例外：/prepare/status 与 /export/status 的 ok 是"任务成功"标志 —— 前端拿 d.ok
     决定弹"合成完成"还是"合成失败"，改成恒 true 会把每次失败都报成成功；这两个接口
     多给一个 success 字段（语义同 ok），其余字段一律照旧。

  7) 视频播放与单帧实时预览（这两个是新增能力，重活都在这一个文件里）：
     · GET /api/merge/video —— FileResponse 不支持 Range（<video> 拖时间轴就废），
       所以自己解析 Range 头、自己按片段读文件：206 + Content-Range/Accept-Ranges，
       没有 Range 时 200 全量流式返回。文件不存在 404，路径越界 400。
     · POST /api/merge/preview —— 取 t 时刻一帧 → 检测 → 2D106 的 68 点 → 换脸
       → 遮罩 → 合成 → JPEG。换脸只能在 4.0 引擎的 python 里跑（MergeStudio /
       facelib 都在那边），所以这里养一个常驻 worker 子进程：JSON lines 走
       stdin/stdout，DFM 与 XSeg 只加载一次，空闲 10 分钟自动退出，崩了下次请求
       自动重启；单帧干活上限 15s，超时就杀掉 worker（ONNX 推理没法从外部打断）
       并回 504。

  8) 多模型预览：/api/merge/preview 的 body 多认一个 models: [{name, dfm, faces}]
     （faces = 该模型负责的人脸序号，按检测顺序从 0 起；单模型时等价于 faces=[]
     或缺省；老字段 model 照样能用）。worker 里最多常驻 3 个 DFM，超了先卸最久
     没用过的（LRU）再加载新的，卸载时 gc + 清 CUDA 缓存；一帧里检测到几张脸就
     按分配一张一张贴，没分配的用第 0 个。返回体多给 models_used / faces / swaps。
     /api/merge/preview/warm 同样认 models 数组，一次热一组（上限仍是 3）。

  9) 标准 DFL 项目模式（iperov 布局，本文件末尾那一大段）：
     POST /api/merge/export-project —— <项目>/<帧名> + aligned/ → 逐帧合成 →
     <项目>/merged/<帧名>（整帧）→ 按原 fps / 音轨编码成 <项目>.mp4。
     · 逐帧合成不另写一套：复用 _preview_job（检测 → 68 点 → 换脸/遮罩/合成），
       只把「从视频按时刻取帧」换成「直接读帧文件」；POST /api/merge/preview-frame
       是同一份逻辑的静态帧入口。
     · 帧序 = 源帧文件名字典序（= ffmpeg image2 的读取顺序）；aligned 的
       source_filename 只用来决定「这一帧有没有脸」和点名缺帧（认不出就报，不猜）。
     · 断点续跑：merged 里存在且非空的帧跳过（force=true 才重做）；落盘是
       .part → os.replace，不会把半张图当成已完成。
     · 编码前对账：merged 比源帧少就点名缺哪几张，并写明成片会短多少秒；
       编号有洞时先硬链接重排成 %06d 再编（image2 遇洞会停住，成片被悄悄截短）。
     · fps / 分辨率 / 音轨只认原视频（读不到就报错）；默认输出 <项目目录>.mp4，
       和原视频重名时改用 <项目目录>_merged.mp4，绝不覆盖素材。
     · 编码命令刻意不带 -shortest（音轨比画面长会把成片截短），并挂
       scale=out_range=tv，否则 JPEG 帧会让成片标成 yuvj420p。
     · POST /api/merge/reveal —— 资源管理器里定位产物（「打开所在目录」）。

拆分后（结构重构，未改行为）：真正的实现搬到 app/server/backend_merge/*（外加 _common/http、
_media/*），本文件只剩「把 router 交出去」和「把名字原样再导出」——旧的
    from .routes_merge import _frame_landmarks
    rm._stage_read_image(...)
照旧能用，调用点不用跟着改。旧行号 → 新文件 的对照表见
app/server/backend_merge/MAPPING.md。
"""
from __future__ import annotations

from ._common.http import (
    _BadInput, _MISSING, _api, _as_dict, _bool_val, _fail, _float_val, _int_val, _ok,
)

from ._common.env import (
    _env_int, _env_secs,
)

from ._common.proc import (
    _kill_proc_tree, _proc_mem_mb,
)

from .backend_merge._shared import (
    DEFAULT_BLEND, DEFAULT_CT_MODE, _COLOR_TRANSFER, _CT_MODES, _DEFAULT_DETECTOR, _DETECTORS,
    _DET_ALIAS_API, _IMG_FORMATS, _MASK_LABELS, _MERGE_MODES, _MERGE_VIEW_MODES, _PRED_BATCH_CAP,
    _REMOVED_DETECTORS, _fmt_mb, _mask_mode_val, _norm_path, _same_path, _settings_val, _trim_ws,
)

from .backend_merge.vram import (
    _CPU_EP, _CPU_EP_LOCK, _DET_EP, _DFM_INFO, _DFM_INFO_LOCK, _ORT_DLL_COOKIES, _ORT_DLL_DIRS,
    _PB_DT_SIZE, _PRED_VRAM_SAFETY, _SESS_BENCH, _SESS_BUSY, _SESS_LOCK, _SESS_PLAN_LAST,
    _SMI_MEM, _VRAM_ARENA_MB, _VRAM_AUX_MB, _VRAM_BASE_PX, _VRAM_BATCH_PTS, _VRAM_BATCH_PTS_384,
    _VRAM_BATCH_PTS_416, _VRAM_BATCH_TABLES, _VRAM_CPU_SLOW, _VRAM_CTX_MB, _VRAM_ENV_CPU,
    _VRAM_ENV_FREE, _VRAM_LAST, _VRAM_LOCK, _VRAM_LOGGED, _VRAM_MODEL_CACHE, _VRAM_OWN,
    _VRAM_OWN_LOCK, _VRAM_OWN_PENDING, _VRAM_PROBE, _VRAM_PROBE_TTL, _VRAM_RECLAIM_MARK,
    _VRAM_RESERVE_MAX_MB, _VRAM_RESERVE_MB, _VRAM_SAFETY, _VRAM_SESS_LOCK, _VRAM_SESS_PAGE_PCT,
    _VRAM_SESS_PEAK, _VRAM_SESS_RUNTIME_MB, _VRAM_SESS_VER, _VRAM_SIZE_CACHE, _det_ep_apply,
    _det_ep_line, _det_ep_resolve, _det_prep, _det_quick_profile, _detect_bench,
    _detect_boxes_ctx, _detect_for, _detect_on, _detect_on_ctx, _dfm_info, _dfm_size_hint,
    _dv_threads, _ensure_ort_dll_dirs, _free_preview_sessions, _gpu_mem_used_mb, _is_oom,
    _mem_delta, _ort_dll_candidates, _pb_fields, _pb_input_px, _pb_tensor, _pb_varint,
    _pred_pick_batch, _preview_guard, _proj_record_sess_peak, _sess_batch_cap, _sess_bench_get,
    _sess_bench_load, _sess_bench_path, _sess_bench_put, _sess_ensure, _vram_apply,
    _vram_batch_mb, _vram_batch_table, _vram_cands, _vram_cpu_fallback_env, _vram_cpu_on,
    _vram_forced_rollback, _vram_frame_mb, _vram_free_mb, _vram_free_smi, _vram_free_torch,
    _vram_guard, _vram_last, _vram_log_lines, _vram_measure_begin, _vram_measure_end,
    _vram_model_mb, _vram_models_mb, _vram_need_mb, _vram_oom_inject_on, _vram_oom_is_text,
    _vram_oom_rollback, _vram_own_detail, _vram_own_drop, _vram_own_inproc_mb, _vram_own_note,
    _vram_owners, _vram_plan, _vram_preview_info, _vram_probe, _vram_reclaim, _vram_reclaim_once,
    _vram_refuse_extra, _vram_refuse_msg, _vram_release_inproc, _vram_reserve_mb,
    _vram_reserve_rule_text, _vram_sess_get, _vram_sess_key, _vram_sess_load, _vram_sess_path,
    _vram_sess_peak_mb, _vram_sess_put, _vram_set_cpu, _vram_src_real, _vram_summary_line,
)

from .backend_merge.eff import (
    _PREVIEW_EFF, _PREVIEW_EFF_LOCK, _eff_basename, _eff_inferred_of, _eff_model_names,
    _eff_publish, _eff_requested_of, _eff_status_patch, _eff_values_of,
)

from .backend_merge.meter import (
    _FusedTicks, _STAGE_KEYS, _STAGE_METER, _STAGE_MS_SRC, _StageMeter, _StageTicks, _stage_tick,
)

from .backend_merge.compose import (
    _AUTO_EMB_LIMIT, _AUTO_PLAN_CACHE, _AUTO_PLAN_MAX, _COMPOSE_CACHE, _COMPOSE_CACHE_LOCK,
    _COMPOSE_CACHE_MAX, _DEBUG_BOX_BGR, _DEBUG_EDGE_BGR, _DEBUG_HUD_BGR, _DEBUG_HUD_TXT,
    _DEBUG_NOSWAP_BGR, _DEBUG_TINT_BGR, _FACE_DBG_CACHE, _FACE_DBG_LOCK, _FACE_DBG_MAX, _LM106,
    _auto_cache_get, _auto_cache_put, _auto_face_plan, _auto_faces_merged, _auto_library,
    _body_idx, _cache_lock, _cache_mode, _cache_set, _cache_snap, _check_cache_ready, _check_idx,
    _compose_cached, _compose_sig, _compose_store, _debug_face_idx, _debug_frame_idx,
    _debug_view_image, _det_load_lock, _det_run_lock, _detect_boxes, _detector, _detector_val,
    _dfm_by_name, _encode_frame, _face_debug_payload, _face_vector_from_frame, _frame_body,
    _frame_image, _frame_landmarks, _frame_payload, _has_dfl_meta, _landmarks_model,
    _manual_faces, _merge_query_into, _merged_from_export, _scan_first, _source_frame,
    _source_ident, _with_kps68,
)

from .backend_merge.pipeline_video import (
    _ERR_HINT, _VID_LOG_FH, _VID_LOG_LOCK, _VID_MAX_LOG, _VID_MAX_RECORDS, _VID_PCT_SWAP,
    _brief_failure, _cancel_guard, _cancel_watch, _cancel_watching, _ensure_cancel_watch,
    _finalize, _log_tail, _run_export, _start_video_export, _vid_cancelled, _vid_close_log,
    _vid_encode, _vid_log, _vid_models_for_frame, _vid_open_log, _vid_record, _vid_stats,
)

from .backend_merge.pipeline_stages import (
    _DFM_ENGINE_ENV, _DFM_SESS, _DFM_TRT_ENV, _DFM_TRT_ON, _FACES_BATCH_ENV, _FACES_END,
    _FACES_QUEUE_ENV, _FACES_READ_ENV, _INFER_CALLS, _MASK_DST_INPUT, _MERGE_CV2_THREADS_DEFAULT,
    _MERGE_PNG_LEVEL, _MERGE_THREADS_DEFAULT, _MERGE_WORKERS_DEFAULT, _PRED_BATCH_DEFAULT,
    _PRED_BATCH_ENV, _PRED_LOCK, _RANGE_END_KEYS, _RANGE_START_KEYS, _STAGE_ALIAS, _STAGE_FN,
    _STAGE_IMG_EXTS, _STAGE_JOB, _STAGE_JPEG_QUALITY, _STAGE_KEEP, _STAGE_LABEL, _STAGE_LOCK,
    _STAGE_LOG_MAX, _STAGE_META_NAME, _STAGE_NUM, _STAGE_ORDER, _STAGE_PCT, _StageCancel,
    _XSEG_SESS, _XSEG_SESS_LOCK, _align_detector, _dfm_engine_name, _dfm_engines_on_disk,
    _dfm_mb, _dfm_run_batch, _dfm_session, _dfm_trt_enabled, _dfm_trt_why, _engine_workspace,
    _mask2d, _mask_signature, _mask_src, _merge_cv2_threads, _merge_worker_count, _range_int,
    _stage_aligned_dir, _stage_brief, _stage_brief_meta, _stage_cancelled, _stage_cfg,
    _stage_check_cancel, _stage_clear_dir, _stage_counts, _stage_dfm_pair, _stage_do_encode,
    _stage_do_faces, _stage_do_frames, _stage_do_mask, _stage_do_merge, _stage_do_pred,
    _stage_done_list, _stage_drop_cache, _stage_dst_face_mat, _stage_extract_frames_ffmpeg,
    _stage_face_name, _stage_face_rows, _stage_faces_batch, _stage_faces_batchable,
    _stage_faces_canvas, _stage_faces_detect, _stage_faces_int, _stage_faces_queue,
    _stage_faces_read_threads, _stage_files, _stage_filter_range, _stage_frame_index,
    _stage_frame_name, _stage_frame_path, _stage_frames_dir, _stage_frames_inproc,
    _stage_gpu_paths, _stage_imencode, _stage_imencode_png, _stage_in_range, _stage_log,
    _stage_mask_dir, _stage_merge_face_inputs, _stage_meta_blank, _stage_meta_load,
    _stage_meta_path, _stage_meta_ready, _stage_meta_save, _stage_mirror, _stage_model_resident,
    _stage_name, _stage_next, _stage_out_path, _stage_precheck, _stage_pred_assign,
    _stage_pred_dir, _stage_profile_blank, _stage_progress, _stage_quality_detector,
    _stage_range, _stage_range_norm, _stage_range_raw, _stage_read_dst_face, _stage_read_frame,
    _stage_read_image, _stage_set, _stage_stages_meta, _stage_sync_range, _stage_vram_gate,
    _stage_worker, _stage_write_atomic, _stage_xseg, _xseg_resolve, _xseg_sess_alive,
)

from .backend_merge.preview import (
    _PREVIEW_BOOT_TIMEOUT, _PREVIEW_HIT, _PREVIEW_HIT_LOCK, _PREVIEW_IDLE_EXIT,
    _PREVIEW_LOAD_TIMEOUT, _PREVIEW_MAX_BATCH, _PREVIEW_MAX_MODELS, _PREVIEW_MAX_SWAPS,
    _PREVIEW_QUALITY, _PREVIEW_SEND_QUALITY, _PREVIEW_WARM_TIMEOUT, _PREVIEW_WORKER_SRC,
    _PREVIEW_WORK_TIMEOUT, _PROJ_BIG_MB, _PROJ_SMALL_MB, _PreviewWorker, _PreviewWorkerPool,
    _WARM_LOCK, _WARM_PENDING, _WARM_STATE, _WARM_THREAD, _WORKER, _WORKER_MAX_MODELS,
    _det_cached, _dfm_of, _face_ids_val, _faces_val, _img_bytes, _jpeg_bytes, _model_for_face,
    _models_of, _pick_face, _plain_frame, _preview_finish, _preview_job, _preview_prepare,
    _proj_batch_default, _proj_out_raw, _proj_plan_note, _proj_send_raw, _proj_workers_default,
    _send_frame_bytes, _shutdown_preview_worker, _swap_blob, _swap_desc, _used_names, _warm_do,
    _warm_fail, _warm_hot, _warm_kick, _warm_loop, _warm_note_hot, _warm_set, _warm_status,
    _worker_models, _worker_pid, _worker_swap, _xseg_of,
)

from .backend_merge.pipeline_project import (
    _PROJ_FRAME_EXT, _PROJ_JOB, _PROJ_LOCK, _PROJ_MAX_LIST, _PROJ_PCT_SWAP, _ProjCancel,
    _proj_cancelled, _proj_encode, _proj_finish, _proj_frame_files, _proj_head_meta, _proj_log,
    _proj_match_frame, _proj_pattern, _proj_put, _proj_run, _proj_scan_aligned, _proj_seq_info,
    _proj_set, _proj_snap, _proj_source_video, _proj_warn, _proj_worker,
)

from .backend_merge.pipeline_fused import (
    _FUSED_BATCH_TARGET, _FUSED_BATCH_WAIT_MS, _FUSED_BUDGETS, _FUSED_CMP_DEFAULT,
    _FUSED_ENV_BUDGET, _FUSED_ENV_DECODE, _FUSED_ENV_DET_PREP, _FUSED_ENV_DUMP, _FUSED_ENV_VRAM,
    _FUSED_PIX_TOKEN, _FUSED_PREP_DEFAULT, _FUSED_QDEPTH, _FUSED_RAW_MODES, _FUSED_RESERVE_MB,
    _FUSED_VRAM_SAFETY, _FUSED_WIDE_TOKEN, _FusedFrame, _PNG_SIG, _fused_budget,
    _fused_cancelled, _fused_cfg, _fused_det_plan, _fused_dump, _fused_frame_mb, _fused_on,
    _fused_open_frames, _fused_run, _fused_src_bits, _fused_worker, _src_of,
)

from ._media.frames import (
    _DUR_CACHE, _DUR_LOCK, _GRAB_FFMPEG_FIRST, _GRAB_FFMPEG_TIMEOUT, _GRAB_FWD_WINDOW,
    _GRAB_LOCK, _GRAB_STATE, _IMAGE_EXT, _VIDEO_EXT, _grab_close_locked, _grab_ensure_locked,
    _grab_frame_at, _grab_index, _grab_key, _grab_via_ffmpeg, _video_duration,
)

from ._media.encoder import (
    _ENCODERS, _HWACCEL, _encoder_val, _hwaccel_val, _vid_cmd, _vid_probe,
)

from .backend_merge.routes import (
    _VIDEO_CHUNK, _VIDEO_CTYPE, _file_chunks, _guard_prepare_target, _inside_repo, _parse_range,
    _raw_is_absolute, _video_ctype, merge_analyze, merge_detect, merge_export,
    merge_export_cancel, merge_export_project, merge_export_project_cancel,
    merge_export_project_status, merge_export_status, merge_face_debug, merge_frame,
    merge_frame_post, merge_info, merge_merged_frame, merge_merged_post, merge_open,
    merge_prepare, merge_prepare_status, merge_preview, merge_preview_frame,
    merge_preview_status, merge_preview_warm, merge_preview_warm_status, merge_project,
    merge_reveal, merge_stage, merge_stage_cancel, merge_stage_status, merge_video, merge_videos,
    router,
)

