# routes_merge.py 拆分对照表（旧行号 → 新文件）

原文件 `app/server/routes_merge.py` **16741 行**（HEAD 状态），按关注点切成上面这些文件。
函数体、注释、日志文案逐字节照抄；只有 import 结构与一处 `__file__` 相对路径下标变了。

## 文件对照（行数：从 routes_merge 搬来的 / 现在实际）

| 新文件 | 搬来的行数 | 现在行数 | 顶层条目 |
|---|---|---|---|
| `_common/http.py` | 135 | 149 | 9 |
| `_common/env.py` | 39 | 50 | 2 |
| `_common/proc.py` | 92 | 103 | 2 |
| `backend_merge/_shared.py` | 201 | 233 | 19 |
| `backend_merge/vram.py` | 2538 | 2597 | 123 |
| `backend_merge/eff.py` | 165 | 600 | 9 |
| `backend_merge/meter.py` | 190 | 201 | 7 |
| `backend_merge/compose.py` | 1252 | 1932 | 57 |
| `backend_merge/pipeline_video.py` | 489 | 551 | 23 |
| `backend_merge/pipeline_stages.py` | 3955 | 4037 | 124 |
| `backend_merge/preview.py` | 2210 | 2276 | 59 |
| `backend_merge/pipeline_project.py` | 1231 | 2264 | 23 |
| `backend_merge/pipeline_fused.py` | 1762 | 1856 | 30 |
| `_media/frames.py` | 251 | 279 | 16 |
| `_media/encoder.py` | 90 | 118 | 6 |
| `backend_merge/routes.py` | 1900 | 2007 | 38 |

## 逐个条目

### `_common/http.py`

- L195–204 `_ok`
- L205–224 `_fail`
- L225–239 `_BadInput`
- L240–261 `_api`
- L262–269 `_as_dict`
- L302–305 `_MISSING`
- L306–323 `_float_val`
- L324–346 `_int_val`
- L986–1000 `_bool_val`

### `_common/env.py`

- L2781–2812 `_env_secs`
- L13229–13235 `_env_int`

### `_common/proc.py`

- L665–706 `_kill_proc_tree`
- L9777–9826 `_proc_mem_mb`

### `backend_merge/_shared.py`

- L101–106 `Try`
- L270–291 `_norm_path`
- L292–301 `_same_path`
- L359–366 `_MERGE_MODES`
- L367–369 `_MERGE_VIEW_MODES`
- L370–373 `_COLOR_TRANSFER`
- L374–375 `_CT_MODES`
- L376–378 `DEFAULT_CT_MODE`
- L379–379 `DEFAULT_BLEND`
- L380–380 `_IMG_FORMATS`
- L381–386 `_DETECTORS`
- L387–391 `_DET_ALIAS_API`
- L392–393 `_REMOVED_DETECTORS`
- L394–394 `_DEFAULT_DETECTOR`
- L414–428 `_mask_mode_val`
- L429–504 `_settings_val`
- L9772–9776 `_fmt_mb`
- L9847–9876 `_trim_ws`
- L11385–11385 `_PRED_BATCH_CAP`

### `backend_merge/vram.py`

- L109–121 `_ORT_DLL_DIRS`
- L122–122 `_ORT_DLL_COOKIES`
- L123–158 `_ort_dll_candidates`
- L159–188 `_ensure_ort_dll_dirs`
- L189–194 `Try`
- L8965–8981 `_sess_batch_cap`
- L9312–9324 `_vram_free_torch`
- L9325–9327 `_SMI_MEM`
- L9328–9354 `_vram_free_smi`
- L9355–9377 `_vram_free_mb`
- L9378–9405 `_VRAM_ENV_FREE`
- L9406–9406 `_VRAM_ENV_CPU`
- L9407–9409 `_VRAM_SAFETY`
- L9410–9410 `_VRAM_RESERVE_MB`
- L9411–9411 `_VRAM_RESERVE_MAX_MB`
- L9412–9412 `_VRAM_CTX_MB`
- L9413–9413 `_VRAM_ARENA_MB`
- L9414–9414 `_VRAM_AUX_MB`
- L9415–9415 `_VRAM_CPU_SLOW`
- L9416–9416 `_VRAM_BASE_PX`
- L9417–9428 `_VRAM_BATCH_PTS_416`
- L9429–9429 `_VRAM_BATCH_PTS_384`
- L9430–9432 `_VRAM_BATCH_TABLES`
- L9433–9438 `_VRAM_BATCH_PTS`
- L9439–9450 `_VRAM_SESS_RUNTIME_MB`
- L9451–9451 `_VRAM_PROBE_TTL`
- L9452–9452 `_VRAM_PROBE`
- L9453–9453 `_VRAM_MODEL_CACHE`
- L9454–9454 `_VRAM_SIZE_CACHE`
- L9455–9455 `_VRAM_LAST`
- L9456–9456 `_VRAM_LOGGED`
- L9457–9457 `_VRAM_LOCK`
- L9458–9458 `_CPU_EP`
- L9459–9459 `_CPU_EP_LOCK`
- L9460–9476 `_VRAM_OWN`
- L9477–9477 `_VRAM_OWN_LOCK`
- L9478–9478 `_VRAM_OWN_PENDING`
- L9479–9488 `_vram_own_note`
- L9489–9494 `_vram_own_drop`
- L9495–9499 `_vram_own_detail`
- L9500–9505 `_vram_own_inproc_mb`
- L9506–9511 `_vram_measure_begin`
- L9512–9523 `_vram_measure_end`
- L9524–9552 `_vram_preview_info`
- L9553–9592 `_vram_owners`
- L9593–9728 `_vram_reclaim`
- L9729–9735 `_vram_src_real`
- L9736–9738 `_VRAM_RECLAIM_MARK`
- L9739–9771 `_vram_reclaim_once`
- L9827–9846 `_mem_delta`
- L9877–9903 `_free_preview_sessions`
- L9904–9950 `_vram_release_inproc`
- L9951–10003 `_vram_probe`
- L10004–10018 `_DFM_INFO`
- L10019–10019 `_DFM_INFO_LOCK`
- L10020–10021 `_PB_DT_SIZE`
- L10022–10032 `_pb_varint`
- L10033–10054 `_pb_fields`
- L10055–10077 `_pb_tensor`
- L10078–10100 `_pb_input_px`
- L10101–10165 `_dfm_info`
- L10166–10188 `_vram_model_mb`
- L10189–10201 `_vram_models_mb`
- L10202–10216 `_VRAM_SESS_PEAK`
- L10217–10217 `_VRAM_SESS_LOCK`
- L10218–10218 `_VRAM_SESS_PAGE_PCT`
- L10219–10221 `_VRAM_SESS_VER`
- L10222–10233 `_vram_sess_key`
- L10234–10237 `_vram_sess_path`
- L10238–10249 `_vram_sess_load`
- L10250–10263 `_vram_sess_put`
- L10264–10277 `_vram_sess_get`
- L10278–10314 `_vram_sess_peak_mb`
- L10315–10369 `_proj_record_sess_peak`
- L10370–10404 `_dfm_size_hint`
- L10405–10413 `_vram_frame_mb`
- L10414–10430 `_vram_batch_table`
- L10431–10454 `_vram_batch_mb`
- L10455–10492 `_vram_reserve_mb`
- L10493–10495 `_vram_cpu_fallback_env`
- L10496–10505 `_vram_cpu_on`
- L10506–10517 `_vram_set_cpu`
- L10518–10554 `_vram_cands`
- L10555–10571 `_vram_need_mb`
- L10572–10910 `_vram_plan`
- L10911–10959 `_vram_refuse_extra`
- L10960–11011 `_vram_summary_line`
- L11012–11057 `_vram_refuse_msg`
- L11058–11170 `_vram_log_lines`
- L11171–11197 `_vram_apply`
- L11198–11204 `_vram_last`
- L11205–11232 `_preview_guard`
- L11233–11270 `_vram_guard`
- L11271–11289 `_vram_forced_rollback`
- L11290–11302 `_vram_oom_inject_on`
- L11303–11310 `_vram_reserve_rule_text`
- L11311–11319 `_vram_oom_is_text`
- L11320–11382 `_vram_oom_rollback`
- L11383–11384 `_PRED_VRAM_SAFETY`
- L11386–11498 `_pred_pick_batch`
- L11499–11503 `_is_oom`
- L12639–12651 `_gpu_mem_used_mb`
- L14333–14334 `_SESS_LOCK`
- L14335–14335 `_SESS_BUSY`
- L14336–14336 `_SESS_BENCH`
- L14337–14337 `_SESS_PLAN_LAST`
- L14338–14338 `_DET_EP`
- L14339–14342 `_sess_bench_path`
- L14343–14354 `_sess_bench_load`
- L14355–14365 `_sess_bench_put`
- L14366–14374 `_sess_bench_get`
- L14385–14393 `_sess_ensure`
- L14394–14453 `_detect_for`
- L14454–14459 `_detect_on`
- L14460–14471 `_det_prep`
- L14472–14477 `_detect_on_ctx`
- L14478–14488 `_detect_boxes_ctx`
- L14489–14539 `_det_quick_profile`
- L14540–14586 `_detect_bench`
- L14587–14596 `_dv_threads`
- L14597–14613 `_det_ep_resolve`
- L14614–14625 `_det_ep_line`
- L14626–14658 `_det_ep_apply`

### `backend_merge/eff.py`

- L2825–2827 `_PREVIEW_EFF`
- L2828–2828 `_PREVIEW_EFF_LOCK`
- L7673–7694 `_eff_model_names`
- L7695–7702 `_eff_basename`
- L7703–7745 `_eff_values_of`
- L7746–7759 `_eff_inferred_of`
- L7760–7793 `_eff_requested_of`
- L7794–7826 `_eff_publish`
- L7827–7833 `_eff_status_patch`

### `backend_merge/meter.py`

- L8439–8459 `_StageTicks`
- L15022–15037 `_STAGE_KEYS`
- L15038–15040 `_STAGE_MS_SRC`
- L15041–15159 `_StageMeter`
- L15160–15162 `_STAGE_METER`
- L15163–15170 `_stage_tick`
- L15171–15190 `_FusedTicks`

### `backend_merge/compose.py`

- L505–510 `_cache_lock`
- L511–511 `_det_load_lock`
- L512–512 `_det_run_lock`
- L513–531 `_cache_snap`
- L532–536 `_cache_set`
- L537–547 `_cache_mode`
- L548–565 `_scan_first`
- L566–580 `_has_dfl_meta`
- L931–949 `_check_cache_ready`
- L950–962 `_check_idx`
- L963–983 `_COMPOSE_CACHE`
- L984–984 `_COMPOSE_CACHE_LOCK`
- L985–985 `_COMPOSE_CACHE_MAX`
- L1001–1037 `_source_frame`
- L1038–1052 `_encode_frame`
- L1053–1059 `_source_ident`
- L1060–1082 `_compose_sig`
- L1083–1092 `_compose_cached`
- L1093–1102 `_compose_store`
- L1103–1114 `_merge_query_into`
- L1115–1128 `_AUTO_PLAN_CACHE`
- L1129–1129 `_AUTO_PLAN_MAX`
- L1130–1136 `Try`
- L1137–1137 `_AUTO_EMB_LIMIT`
- L1138–1145 `_auto_cache_get`
- L1146–1155 `_auto_cache_put`
- L1156–1171 `_auto_library`
- L1172–1182 `_dfm_by_name`
- L1183–1193 `_detect_boxes`
- L1194–1211 `_face_vector_from_frame`
- L1212–1303 `_auto_face_plan`
- L1304–1322 `_auto_faces_merged`
- L1323–1350 `_manual_faces`
- L1351–1362 `_DEBUG_TINT_BGR`
- L1363–1363 `_DEBUG_EDGE_BGR`
- L1364–1364 `_DEBUG_BOX_BGR`
- L1365–1365 `_DEBUG_NOSWAP_BGR`
- L1366–1366 `_DEBUG_HUD_BGR`
- L1367–1367 `_DEBUG_HUD_TXT`
- L1368–1455 `_debug_view_image`
- L1456–1744 `_frame_payload`
- L1745–1752 `_body_idx`
- L1753–1759 `_frame_body`
- L1808–1823 `_FACE_DBG_CACHE`
- L1824–1824 `_FACE_DBG_MAX`
- L1825–1825 `_FACE_DBG_LOCK`
- L1826–1837 `_debug_face_idx`
- L1838–1844 `_debug_frame_idx`
- L1845–1967 `_face_debug_payload`
- L2282–2325 `_merged_from_export`
- L2429–2450 `_frame_image`
- L2451–2471 `_detector_val`
- L2472–2487 `_detector`
- L2488–2515 `_with_kps68`
- L4474–4476 `_LM106`
- L4477–4507 `_landmarks_model`
- L4508–4544 `_frame_landmarks`

### `backend_merge/pipeline_video.py`

- L581–588 `_ERR_HINT`
- L589–593 `_log_tail`
- L594–607 `_brief_failure`
- L608–645 `_finalize`
- L646–664 `_run_export`
- L707–709 `_cancel_guard`
- L710–710 `_cancel_watching`
- L711–719 `_ensure_cancel_watch`
- L720–756 `_cancel_watch`
- L7292–7313 `_VID_PCT_SWAP`
- L7314–7314 `_VID_MAX_LOG`
- L7315–7315 `_VID_MAX_RECORDS`
- L7325–7329 `_VID_LOG_LOCK`
- L7330–7330 `_VID_LOG_FH`
- L7331–7342 `_vid_open_log`
- L7343–7356 `_vid_close_log`
- L7357–7376 `_vid_log`
- L7377–7383 `_vid_stats`
- L7384–7394 `_vid_record`
- L7395–7399 `_vid_cancelled`
- L7425–7441 `_vid_models_for_frame`
- L13030–13082 `_vid_encode`
- L16527–16712 `_start_video_export`

### `backend_merge/pipeline_stages.py`

- L7533–7565 `_STAGE_ORDER`
- L7566–7567 `_STAGE_LABEL`
- L7568–7570 `_STAGE_NUM`
- L7571–7572 `_STAGE_PCT`
- L7573–7589 `_STAGE_ALIAS`
- L7590–7590 `_STAGE_META_NAME`
- L7591–7591 `_STAGE_IMG_EXTS`
- L7592–7592 `_STAGE_LOG_MAX`
- L7593–7593 `_STAGE_LOCK`
- L7594–7604 `_STAGE_JOB`
- L7605–7609 `_stage_name`
- L7610–7672 `_mask_src`
- L7834–7852 `_stage_files`
- L7853–7856 `_stage_frames_dir`
- L7857–7860 `_stage_aligned_dir`
- L7861–7865 `_stage_frame_name`
- L7866–7869 `_stage_face_name`
- L7870–7873 `_stage_meta_path`
- L7874–7886 `_stage_meta_load`
- L7887–7905 `_stage_meta_save`
- L7906–7919 `_stage_meta_blank`
- L7920–7943 `_stage_meta_ready`
- L7944–7967 `_stage_imencode`
- L7968–7989 `_stage_imencode_png`
- L7990–7998 `_stage_write_atomic`
- L7999–8002 `_stage_read_image`
- L8003–8020 `_stage_read_frame`
- L8021–8065 `_stage_dst_face_mat`
- L8066–8099 `_stage_read_dst_face`
- L8100–8113 `_stage_clear_dir`
- L8114–8133 `_stage_log`
- L8134–8138 `_stage_set`
- L8139–8156 `_stage_progress`
- L8157–8178 `_stage_mirror`
- L8179–8188 `_stage_cancelled`
- L8189–8192 `_StageCancel`
- L8193–8197 `_stage_check_cancel`
- L8198–8204 `_stage_quality_detector`
- L8205–8224 `_align_detector`
- L8225–8233 `_stage_brief`
- L8234–8300 `_stage_extract_frames_ffmpeg`
- L8301–8383 `_stage_do_frames`
- L8384–8425 `_stage_frames_inproc`
- L8426–8435 `_FACES_END`
- L8436–8436 `_FACES_READ_ENV`
- L8437–8437 `_FACES_QUEUE_ENV`
- L8438–8438 `_FACES_BATCH_ENV`
- L8460–8468 `_stage_faces_int`
- L8469–8478 `_stage_faces_read_threads`
- L8479–8483 `_stage_faces_queue`
- L8484–8488 `_stage_faces_batchable`
- L8489–8523 `_stage_faces_batch`
- L8524–8542 `_stage_faces_canvas`
- L8543–8561 `_stage_faces_detect`
- L8562–8925 `_stage_do_faces`
- L8926–8934 `_PRED_LOCK`
- L8935–8935 `_DFM_SESS`
- L8936–8936 `_INFER_CALLS`
- L8937–8940 `_stage_pred_dir`
- L8941–8944 `_stage_mask_dir`
- L8945–8964 `_dfm_engine_name`
- L8982–8993 `_DFM_TRT_ENV`
- L8994–8994 `_DFM_ENGINE_ENV`
- L8995–8995 `_DFM_TRT_ON`
- L8996–9006 `_dfm_trt_enabled`
- L9007–9015 `_dfm_trt_why`
- L9016–9022 `_dfm_mb`
- L9023–9039 `_dfm_engines_on_disk`
- L9040–9169 `_dfm_session`
- L9170–9182 `_mask2d`
- L9183–9237 `_dfm_run_batch`
- L9238–9249 `_stage_face_rows`
- L9250–9258 `_stage_frame_path`
- L9259–9311 `_stage_pred_assign`
- L11504–11968 `_stage_do_pred`
- L11969–11972 `_XSEG_SESS`
- L11973–11973 `_XSEG_SESS_LOCK`
- L11974–11983 `_engine_workspace`
- L11984–11989 `_xseg_resolve`
- L11990–12008 `_mask_signature`
- L12009–12047 `_xseg_sess_alive`
- L12048–12090 `_stage_xseg`
- L12091–12096 `_MASK_DST_INPUT`
- L12097–12628 `_stage_do_mask`
- L12629–12638 `_stage_profile_blank`
- L12652–12678 `_stage_merge_face_inputs`
- L12679–12691 `_merge_worker_count`
- L12692–12720 `_merge_cv2_threads`
- L12721–12989 `_stage_do_merge`
- L13083–13197 `_stage_do_encode`
- L13198–13209 `_stage_drop_cache`
- L13210–13219 `_STAGE_FN`
- L13220–13223 `_STAGE_KEEP`
- L13224–13227 `_PRED_BATCH_ENV`
- L13228–13228 `_PRED_BATCH_DEFAULT`
- L13236–13253 `_MERGE_WORKERS_DEFAULT`
- L13254–13254 `_MERGE_WORKERS_DEFAULT`
- L13255–13256 `_MERGE_THREADS_DEFAULT`
- L13257–13257 `_MERGE_CV2_THREADS_DEFAULT`
- L13258–13258 `_MERGE_PNG_LEVEL`
- L13259–13264 `_STAGE_JPEG_QUALITY`
- L13265–13272 `_RANGE_START_KEYS`
- L13273–13273 `_RANGE_END_KEYS`
- L13274–13293 `_range_int`
- L13294–13314 `_stage_range_raw`
- L13315–13347 `_stage_range_norm`
- L13348–13357 `_stage_range`
- L13358–13373 `_stage_sync_range`
- L13374–13383 `_stage_frame_index`
- L13384–13391 `_stage_in_range`
- L13392–13401 `_stage_filter_range`
- L13402–13411 `_stage_dfm_pair`
- L13412–13436 `_stage_out_path`
- L13437–13570 `_stage_cfg`
- L13571–13706 `_stage_precheck`
- L13707–13717 `_stage_counts`
- L13718–13724 `_stage_stages_meta`
- L13725–13729 `_stage_done_list`
- L13730–13737 `_stage_next`
- L13738–13776 `_stage_brief_meta`
- L13777–13905 `_stage_worker`
- L14052–14079 `_stage_gpu_paths`
- L14080–14104 `_stage_model_resident`
- L14105–14137 `_stage_vram_gate`

### `backend_merge/preview.py`

- L2813–2817 `_PREVIEW_WORK_TIMEOUT`
- L2818–2818 `_PREVIEW_LOAD_TIMEOUT`
- L2819–2819 `_PREVIEW_BOOT_TIMEOUT`
- L2820–2820 `_PREVIEW_IDLE_EXIT`
- L2821–2821 `_PREVIEW_SEND_QUALITY`
- L2822–2822 `_PREVIEW_QUALITY`
- L2823–2823 `_PREVIEW_HIT`
- L2824–2824 `_PREVIEW_HIT_LOCK`
- L2829–2829 `_WORKER_MAX_MODELS`
- L2830–2830 `_PREVIEW_MAX_MODELS`
- L2831–2831 `_PREVIEW_MAX_SWAPS`
- L2832–2832 `_PREVIEW_MAX_BATCH`
- L2833–3519 `_PREVIEW_WORKER_SRC`
- L3520–3926 `_PreviewWorker`
- L3927–3929 `_WORKER`
- L3930–3943 `_PROJ_BIG_MB`
- L3944–3944 `_PROJ_SMALL_MB`
- L3945–3976 `_proj_workers_default`
- L3977–4013 `_proj_batch_default`
- L4014–4070 `_proj_plan_note`
- L4071–4083 `_proj_send_raw`
- L4084–4099 `_proj_out_raw`
- L4100–4204 `_PreviewWorkerPool`
- L4205–4211 `_shutdown_preview_worker`
- L4212–4214 `Expr`
- L4460–4473 `_pick_face`
- L4545–4572 `_dfm_of`
- L4573–4592 `_faces_val`
- L4593–4623 `_face_ids_val`
- L4624–4663 `_models_of`
- L4664–4681 `_model_for_face`
- L4682–4700 `_used_names`
- L4701–4715 `_xseg_of`
- L4716–4723 `_jpeg_bytes`
- L4724–4735 `_img_bytes`
- L4736–4761 `_send_frame_bytes`
- L4762–4784 `_plain_frame`
- L4785–4892 `_swap_desc`
- L4893–4903 `_swap_blob`
- L4904–4929 `_worker_swap`
- L4930–4971 `_preview_finish`
- L4972–5037 `_preview_prepare`
- L5038–5062 `_preview_job`
- L5381–5404 `_PREVIEW_WARM_TIMEOUT`
- L5405–5405 `_WARM_LOCK`
- L5406–5406 `_WARM_THREAD`
- L5407–5407 `_WARM_PENDING`
- L5408–5428 `_WARM_STATE`
- L5429–5441 `_det_cached`
- L5442–5452 `_worker_pid`
- L5453–5459 `_worker_models`
- L5460–5478 `_warm_hot`
- L5479–5500 `_warm_note_hot`
- L5501–5505 `_warm_set`
- L5506–5517 `_warm_fail`
- L5518–5597 `_warm_do`
- L5598–5619 `_warm_loop`
- L5620–5640 `_warm_kick`
- L5641–5660 `_warm_status`

### `backend_merge/pipeline_project.py`

- L5711–5733 `_PROJ_LOCK`
- L5734–5734 `_PROJ_PCT_SWAP`
- L5735–5735 `_PROJ_MAX_LIST`
- L5736–5736 `_PROJ_FRAME_EXT`
- L5737–5749 `_PROJ_JOB`
- L5750–5753 `_ProjCancel`
- L5754–5764 `_proj_snap`
- L5765–5769 `_proj_set`
- L5770–5781 `_proj_log`
- L5782–5796 `_proj_warn`
- L5797–5801 `_proj_cancelled`
- L5802–5817 `_proj_frame_files`
- L5818–5857 `_proj_head_meta`
- L5858–5877 `_proj_match_frame`
- L5878–5928 `_proj_scan_aligned`
- L5929–5955 `_proj_seq_info`
- L5956–5964 `_proj_pattern`
- L5965–5986 `_proj_put`
- L5987–6056 `_proj_encode`
- L6057–6095 `_proj_source_video`
- L6096–6179 `_proj_finish`
- L6180–6905 `_proj_run`
- L6906–6941 `_proj_worker`

### `backend_merge/pipeline_fused.py`

- L14240–14277 `_FUSED_BUDGETS`
- L14278–14278 `_FUSED_QDEPTH`
- L14279–14279 `_FUSED_RESERVE_MB`
- L14280–14280 `_FUSED_VRAM_SAFETY`
- L14281–14281 `_FUSED_ENV_VRAM`
- L14282–14282 `_FUSED_ENV_BUDGET`
- L14283–14283 `_FUSED_ENV_DUMP`
- L14284–14297 `_FUSED_BATCH_TARGET`
- L14298–14309 `_FUSED_BATCH_WAIT_MS`
- L14310–14312 `_FUSED_PREP_DEFAULT`
- L14313–14313 `_FUSED_CMP_DEFAULT`
- L14659–14682 `_fused_det_plan`
- L14683–14689 `_fused_on`
- L14701–14708 `_fused_frame_mb`
- L14709–14739 `_fused_budget`
- L14740–14846 `_fused_cfg`
- L14847–14849 `_PNG_SIG`
- L14850–14853 `_FUSED_ENV_DECODE`
- L14854–14855 `_FUSED_ENV_DET_PREP`
- L14856–14856 `_FUSED_RAW_MODES`
- L14857–14858 `_FUSED_PIX_TOKEN`
- L14859–14859 `_FUSED_WIDE_TOKEN`
- L14860–14889 `_fused_src_bits`
- L14890–15001 `_fused_open_frames`
- L15002–15021 `_FusedFrame`
- L15191–15198 `_fused_cancelled`
- L15199–15233 `_fused_dump`
- L15234–15254 `_src_of`
- L15255–16352 `_fused_run`
- L16353–16526 `_fused_worker`

### `_media/frames.py`

- L347–351 `_VIDEO_EXT`
- L352–352 `_IMAGE_EXT`
- L4215–4218 `_DUR_CACHE`
- L4219–4219 `_DUR_LOCK`
- L4220–4257 `_video_duration`
- L4258–4271 `_GRAB_LOCK`
- L4272–4272 `_GRAB_FWD_WINDOW`
- L4273–4273 `_GRAB_FFMPEG_FIRST`
- L4274–4274 `_GRAB_FFMPEG_TIMEOUT`
- L4275–4276 `_GRAB_STATE`
- L4277–4286 `_grab_key`
- L4287–4298 `_grab_close_locked`
- L4299–4308 `_grab_index`
- L4309–4338 `_grab_ensure_locked`
- L4339–4369 `_grab_via_ffmpeg`
- L4370–4459 `_grab_frame_at`

### `_media/encoder.py`

- L353–355 `_ENCODERS`
- L356–358 `_HWACCEL`
- L395–404 `_encoder_val`
- L405–413 `_hwaccel_val`
- L7400–7424 `_vid_probe`
- L12990–13029 `_vid_cmd`

### `backend_merge/routes.py`

- L107–108 `router`
- L757–799 `merge_videos`
- L800–836 `merge_prepare`
- L837–868 `_guard_prepare_target`
- L869–878 `merge_prepare_status`
- L879–930 `merge_open`
- L1760–1787 `merge_frame`
- L1788–1807 `merge_frame_post`
- L1968–2002 `merge_face_debug`
- L2003–2174 `merge_export`
- L2175–2281 `merge_export_status`
- L2326–2365 `merge_merged_frame`
- L2366–2394 `merge_merged_post`
- L2395–2428 `merge_export_cancel`
- L2516–2541 `merge_detect`
- L2542–2578 `merge_analyze`
- L2579–2608 `merge_project`
- L2609–2628 `merge_info`
- L2629–2643 `_VIDEO_CTYPE`
- L2644–2644 `_VIDEO_CHUNK`
- L2645–2648 `_video_ctype`
- L2649–2658 `_inside_repo`
- L2659–2676 `_raw_is_absolute`
- L2677–2711 `_parse_range`
- L2712–2730 `_file_chunks`
- L2731–2780 `merge_video`
- L5063–5289 `merge_preview`
- L5290–5308 `merge_preview_status`
- L5309–5380 `merge_preview_frame`
- L5661–5703 `merge_preview_warm`
- L5704–5710 `merge_preview_warm_status`
- L6942–7240 `merge_export_project`
- L7241–7276 `merge_export_project_status`
- L7277–7291 `merge_export_project_cancel`
- L13906–14035 `merge_stage_status`
- L14036–14051 `merge_stage_cancel`
- L14138–14239 `merge_stage`
- L16713–16740 `merge_reveal`

## 按证据删掉的死代码

全仓（.py/.js/.html/.md，排除兼容层与本次生成的临时脚本）匹配次数 = 1，也就是只剩 def 自己：

| 原行号 | 名字 | 备注 |
|---|---|---|
| L7318 | `_VidCancel` | 无任何调用点 |
| L7322 | `_vid_set` | 无任何调用点 |
| L7454 | `_vid_face_rows` | 调用者 `_vid_one` 也一起删了（级联） |
| L7492 | `_vid_one` | 无任何调用点 |
| L14377 | `_sess_model_key` | 无任何调用点 |
| L14692 | `_fused_vram_free` | 无任何调用点 |
