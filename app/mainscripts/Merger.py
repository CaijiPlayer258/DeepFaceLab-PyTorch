import math
import multiprocessing
import sys
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, InteractiveMergerSubprocessor, MergeMasked, MergerConfig


# Module-level worker for multiprocessing Pool (must be picklable on Windows)
def _process_one(args):
    (frame, predictor_input_shape, cfg, output_path, output_mask_path,
     dfm_onnx_path, xseg_onnx_path, xseg_onnx_res, run_on_cpu) = args

    import onnxruntime as ort, numpy as np, cv2, time
    from pathlib import Path as _P
    from core.cv2ex import cv2_imread, cv2_imwrite
    from core import imagelib
    from merger.MergeMasked import MergeMasked as _MergeMasked

    if dfm_onnx_path:
        dfm_sess = ort.InferenceSession(dfm_onnx_path,
                                        providers=['CUDAExecutionProvider'] if not run_on_cpu
                                        else ['CPUExecutionProvider'])
        dfm_in = dfm_sess.get_inputs()[0].name
        def _pred(fb):
            out = dfm_sess.run(None, {dfm_in: fb[None, ...].astype(np.float32)})
            return out[1][0], out[0][0], out[2][0]
    else:
        _pred = None

    if xseg_onnx_path:
        _xres = max(128, xseg_onnx_res)
        xseg_sess = ort.InferenceSession(xseg_onnx_path,
                                         providers=['CUDAExecutionProvider'] if not run_on_cpu
                                         else ['CPUExecutionProvider'])
        xseg_in = xseg_sess.get_inputs()[0].name
        def _xseg(x):
            if x.ndim==3: x=x[None,...]
            h,w=x.shape[1:3]; inp=cv2.resize(x[0],(_xres,_xres))[None,...].astype(np.float32)
            o=xseg_sess.run(None,{xseg_in:inp})[0]; o=cv2.resize(o[0],(w,h)); o=np.clip(o,0,1); o[o<0.1]=0.0
            if o.ndim==2: o=o[...,None]; return o
    else:
        _xseg = None

    fp = frame.frame_info.filepath
    out_f = _P(output_path) / (fp.stem + '.png')
    mask_f = _P(output_mask_path) / (fp.stem + '.png')
    if out_f.exists() and mask_f.exists():
        return None

    t0 = time.perf_counter()
    if len(frame.frame_info.landmarks_list) == 0:
        img = cv2_imread(str(fp)); imagelib.normalize_channels(img,3)
        h,w=img.shape[:2]; msk=np.zeros((h,w,1),dtype=img.dtype); t_face=0
    else:
        try:
            fin = _MergeMasked(_pred, predictor_input_shape, face_enhancer_func=None,
                              xseg_256_extract_func=_xseg, cfg=cfg, frame_info=frame.frame_info)
            img=fin[...,:3]; msk=fin[...,3:4]; t_face=time.perf_counter()-t0
        except: return None
    t0=time.perf_counter()
    cv2_imwrite(str(out_f),img); cv2_imwrite(str(mask_f),msk)
    t_write=time.perf_counter()-t0
    return (t_face, t_write)


def _run_pipeline(predictor_func, predictor_input_shape, xseg_extract_func,
                  face_enhancer_func, cfg, frames, output_path, output_mask_path,
                  dfm_onnx_path=None, xseg_onnx_path=None, xseg_onnx_res=256,
                  run_on_cpu=False):
    """Thread-pool pipeline: shared ONNX sessions + lock, parallel MergeMaskedFace on threads."""
    import time, collections, threading
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    n_total = len(frames)
    n_workers = max(2, min(6, multiprocessing.cpu_count() - 1))
    timings = collections.deque(maxlen=50)
    t_start = time.perf_counter()

    pbar = tqdm(total=n_total, desc='Merging', unit='it', ascii=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')

    # Rebuild ONNX predictor/xseg directly — shared across threads with lock
    gpu_lock = threading.Lock()

    if dfm_onnx_path is not None:
        import onnxruntime as _ort
        _dfm_sess = _ort.InferenceSession(str(dfm_onnx_path),
                                          providers=['CUDAExecutionProvider'] if not run_on_cpu
                                          else ['CPUExecutionProvider'])
        _dfm_in = _dfm_sess.get_inputs()[0].name

        def raw_predictor(face_bgr):
            with gpu_lock:
                out = _dfm_sess.run(None, {_dfm_in: face_bgr[None, ...].astype(np.float32)})
            return out[1][0], out[0][0], out[2][0]
    else:
        raw_predictor = predictor_func.f if hasattr(predictor_func, 'f') else predictor_func

    if xseg_onnx_path is not None:
        import onnxruntime as _ort2
        _xseg_res = max(128, xseg_onnx_res)
        _xseg_sess = _ort2.InferenceSession(str(xseg_onnx_path),
                                            providers=['CUDAExecutionProvider'] if not run_on_cpu
                                            else ['CPUExecutionProvider'])
        _xseg_in = _xseg_sess.get_inputs()[0].name

        def raw_xseg(input_image):
            if input_image.ndim == 3: input_image = input_image[None, ...]
            h, w = input_image.shape[1:3]
            inp = cv2.resize(input_image[0], (_xseg_res, _xseg_res))[None, ...].astype(np.float32)
            with gpu_lock:
                out = _xseg_sess.run(None, {_xseg_in: inp})[0]
            out = cv2.resize(out[0], (w, h))
            out = np.clip(out, 0, 1); out[out < 0.1] = 0.0
            if out.ndim == 2: out = out[..., None]
            return out
    else:
        raw_xseg = xseg_extract_func.f if hasattr(xseg_extract_func, 'f') else xseg_extract_func

    def _process_frame(frame):
        fp = frame.frame_info.filepath
        out_f = output_path / (fp.stem + '.png')
        mask_f = output_mask_path / (fp.stem + '.png')
        if out_f.exists() and mask_f.exists(): return None
        t0 = time.perf_counter()
        if len(frame.frame_info.landmarks_list) == 0:
            img = cv2_imread(str(fp)); imagelib.normalize_channels(img, 3)
            h, w = img.shape[:2]; m = np.zeros((h, w, 1), dtype=img.dtype); tf = 0
        else:
            try:
                fin = MergeMasked(raw_predictor, predictor_input_shape, face_enhancer_func=None,
                                  xseg_256_extract_func=raw_xseg, cfg=cfg, frame_info=frame.frame_info)
                img, m, tf = fin[...,:3], fin[...,3:4], time.perf_counter()-t0
            except Exception as e: return None
        t0 = time.perf_counter()
        cv2_imwrite(str(out_f), img); cv2_imwrite(str(mask_f), m)
        return (tf, time.perf_counter() - t0)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_frame, f): f for f in frames}
        for fut in futures:
            result = fut.result()
            pbar.update(1)
            if result:
                timings.append(result)
                if len(timings) >= 10:
                    af = sum(t[0] for t in timings) / len(timings) * 1000
                    aw = sum(t[1] for t in timings) / len(timings) * 1000
                    rate = pbar.n / max(time.perf_counter() - t_start, 0.001)
                    pbar.set_postfix_str(f'face {af:.0f}ms write {aw:.0f}ms {rate:.1f}it/s')

    pbar.close()
    if timings:
        avg_f = sum(t[0] for t in timings) / len(timings) * 1000
        avg_w = sum(t[1] for t in timings) / len(timings) * 1000
        io.log_info(f'Pipeline done.  face {avg_f:.0f}ms  write {avg_w:.0f}ms')


def _apply_merge_settings(cfg, settings):
    """Apply CLI merge settings to a MergerConfigMasked object, skipping interactive prompts."""
    from merger.MergerConfig import ctm_str_dict, mode_str_dict, mask_mode_dict, FaceType

    mode_map = {'overlay':'overlay','hist-match':'hist-match','seamless':'seamless',
                'seamless-hist-match':'seamless-hist-match','raw-rgb':'raw-rgb','raw-predict':'raw-predict'}
    ft_map = {'half_face':FaceType.HALF, 'mid_full':FaceType.MID_FULL, 'full_face':FaceType.FULL,
              'whole_face':FaceType.WHOLE_FACE, 'head':FaceType.HEAD}

    if settings.get('mode'): cfg.mode = mode_map.get(settings['mode'], cfg.mode)
    if settings.get('face_type') and settings['face_type'] in ft_map:
        cfg.face_type = ft_map[settings['face_type']]
    if settings.get('masked_hist_match') is not None: cfg.masked_hist_match = bool(settings['masked_hist_match'])
    if settings.get('hist_match_threshold') is not None: cfg.hist_match_threshold = int(settings['hist_match_threshold'])
    if settings.get('mask_mode') is not None: cfg.mask_mode = int(settings['mask_mode'])
    if settings.get('erode_mask') is not None: cfg.erode_mask_modifier = int(settings['erode_mask'])
    if settings.get('blur_mask') is not None: cfg.blur_mask_modifier = int(settings['blur_mask'])
    if settings.get('motion_blur') is not None: cfg.motion_blur_power = int(settings['motion_blur'])
    if settings.get('output_face_scale') is not None: cfg.output_face_scale = int(settings['output_face_scale'])
    if settings.get('color_transfer') and settings['color_transfer'] in ctm_str_dict:
        cfg.color_transfer_mode = ctm_str_dict[settings['color_transfer']]
    if settings.get('super_resolution') is not None: cfg.super_resolution_power = int(settings['super_resolution'])
    if settings.get('image_denoise') is not None: cfg.image_denoise_power = int(settings['image_denoise'])
    if settings.get('bicubic_degrade') is not None: cfg.bicubic_degrade_power = int(settings['bicubic_degrade'])
    if settings.get('color_degrade') is not None: cfg.color_degrade_power = int(settings['color_degrade'])
    if settings.get('sharpen_mode') is not None: cfg.sharpen_mode = int(settings['sharpen_mode'])
    if settings.get('sharpen_amount') is not None: cfg.blursharpen_amount = int(settings['sharpen_amount'])

    io.log_info(f"合成配置已应用: mode={cfg.mode} mask_mode={cfg.mask_mode} "
                f"color_transfer={cfg.color_transfer_mode} sharpen={cfg.sharpen_mode}")


def main (model_class_name=None,
          saved_models_path=None,
          training_data_src_path=None,
          force_model_name=None,
          input_path=None,
          output_path=None,
          output_mask_path=None,
          aligned_path=None,
          force_gpu_idxs=None,
          cpu_only=None,
          dfm_onnx_path=None,
          xseg_onnx_path=None,
          xseg_onnx_res=256,
          pipeline=False,
          merge_settings=None):
    io.log_info ("正在运行合成器（Merger）。\r\n")

    try:
        if not input_path.exists():
            io.log_err(f'未找到输入目录。请确认路径存在：{str(input_path)}')
            return

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not output_mask_path.exists():
            output_mask_path.mkdir(parents=True, exist_ok=True)

        if not saved_models_path.exists():
            io.log_err('未找到模型目录。请确认路径存在。')
            return

        # 初始化模型 (PyTorch or ONNX)
        import models
        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0

        if dfm_onnx_path is not None:
            from merger.MergerConfig import MergerConfigMasked
            io.log_info(f'使用 ONNX DFM: {dfm_onnx_path}')

            import onnxruntime as ort
            dfm_sess = ort.InferenceSession(str(dfm_onnx_path),
                                            providers=['CUDAExecutionProvider'] if not run_on_cpu
                                            else ['CPUExecutionProvider'])
            dfm_in_name = dfm_sess.get_inputs()[0].name
            dfm_res = dfm_sess.get_inputs()[0].shape[1]  # NHWC: (N,H,W,C)
            io.log_info(f'  DFM ONNX provider: {dfm_sess.get_providers()[0]}  res={dfm_res}x{dfm_res}')

            def onnx_predictor_func(face_bgr):
                """ONNX DFM predictor: face HWC float32 → (bgr, mask_src, mask_dst)."""
                inp = face_bgr[None, ...].astype(np.float32)
                outputs = dfm_sess.run(None, {dfm_in_name: inp})
                # outputs: [out_face_mask, out_celeb_face, out_celeb_face_mask]
                prd_face_bgr = outputs[1][0]
                prd_face_mask_a = outputs[0][0]
                prd_face_dst_mask_a = outputs[2][0]
                return prd_face_bgr, prd_face_mask_a, prd_face_dst_mask_a

            predictor_func = MPFunc(onnx_predictor_func)
            predictor_input_shape = (dfm_res, dfm_res, 3)
            cfg = MergerConfigMasked()

            # Fake model-like object for session and iter storage
            class _FakeModel:
                def get_iter(self): return 0
                def get_strpath_storage_for_file(self, fname):
                    return str(Path(saved_models_path) / fname)
                def finalize(self): pass
            model = _FakeModel()

        else:
            model = models.import_model(model_class_name)(is_training=False,
                                                          saved_models_path=saved_models_path,
                                                          force_gpu_idxs=force_gpu_idxs,
                                                          force_model_name=force_model_name,
                                                          cpu_only=cpu_only)
            predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()
            predictor_func = MPFunc(predictor_func)

        # XSeg mask extractor
        if xseg_onnx_path is not None:
            io.log_info(f'使用 ONNX XSeg: {xseg_onnx_path}  (分辨率: {xseg_onnx_res})')
            import onnxruntime as ort

            # One session per worker avoids pickle issues with MPFunc
            def _make_xseg_session():
                return ort.InferenceSession(str(xseg_onnx_path),
                                            providers=['CUDAExecutionProvider'] if not run_on_cpu
                                            else ['CPUExecutionProvider'])
            xseg_res = max(128, xseg_onnx_res)

            _xseg_prov_logged = [False]
            _xseg_cached_sess = [None]

            def _xseg_onnx_extract(input_image):
                sess = _xseg_cached_sess[0]
                if sess is None:
                    sess = _make_xseg_session()
                    _xseg_cached_sess[0] = sess
                    io.log_info(f'  XSeg ONNX provider: {sess.get_providers()[0]}  res={xseg_res}x{xseg_res}')
                in_name = sess.get_inputs()[0].name
                if input_image.ndim == 3:
                    input_image = input_image[None, ...]
                h, w = input_image.shape[1:3]
                inp = cv2.resize(input_image[0], (xseg_res, xseg_res))
                inp = inp[None, ...].astype(np.float32)
                out = sess.run(None, {in_name: inp})[0]
                out = cv2.resize(out[0], (w, h))
                out = np.clip(out, 0, 1)
                out[out < 0.1] = 0.0
                if out.ndim == 2:
                    out = out[..., None]
                return out

            xseg_256_extract_func = MPFunc(_xseg_onnx_extract)
        else:
            xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',
                                                        name='XSeg',
                                                        resolution=256,
                                                        weights_file_root=saved_models_path,
                                                        place_model_on_cpu=True,
                                                        run_on_cpu=run_on_cpu)

        face_enhancer_func = MPClassFuncOnDemand(FaceEnhancer, 'enhance',
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        if merge_settings and len(merge_settings) > 0:
            io.log_info("流水线模式：使用命令行指定的合成参数。")
            _apply_merge_settings(cfg, merge_settings)
        elif pipeline:
            is_interactive = False
            io.log_info("流水线模式：跳过所有交互，使用默认设置。")
        else:
            is_interactive = io.input_bool ("是否使用交互式合成？", True) if not io.is_colab() else False

        if not is_interactive and not (merge_settings and len(merge_settings) > 0):
            cfg.ask_settings()

        if pipeline:
            subprocess_count = max(4, multiprocessing.cpu_count() - 2)
        else:
            subprocess_count = io.input_int(
                "工作进程数量？",
                max(8, multiprocessing.cpu_count()),
                valid_range=[1, multiprocessing.cpu_count()],
                help_message="指定用于处理的线程/进程数量。数值过低可能影响性能；数值过高可能导致内存不足。该值不得超过 CPU 核心数。",
            )

        input_path_image_paths = pathex.get_image_paths(input_path)

        if cfg.type == MergerConfig.TYPE_MASKED:
            if not aligned_path.exists():
                io.log_err(f'未找到 aligned 目录。请确认路径存在：{str(aligned_path)}')
                return

            packed_samples = None
            try:
                packed_samples = samplelib.PackedFaceset.load(aligned_path)
            except:
                io.log_err(f"加载 samplelib.PackedFaceset.load 失败：{str(aligned_path)}，{traceback.format_exc()}")


            if packed_samples is not None:
                io.log_info ("检测到 PackedFaceset，将使用打包 faceset。")
                def _load_packed(fp, smp):
                    return Path(fp), DFLIMG.load(Path(fp), loader_func=lambda x: smp.read_raw_file())
                items = [(s.filename, s) for s in packed_samples]
                load_func = lambda a: _load_packed(*a)
            else:
                def _load_normal(fp):
                    return Path(fp), DFLIMG.load(Path(fp))
                items = [Path(p) for p in pathex.get_image_paths(aligned_path)]
                load_func = _load_normal

            # Parallel loading with real progress
            from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed
            results = []
            io.progress_bar("正在收集对齐信息", len(items))
            with _TPE(max_workers=8) as pool:
                futures = {pool.submit(load_func, item): item for item in items}
                for f in as_completed(futures):
                    results.append(f.result())
                    io.progress_bar_inc(1)
            io.progress_bar_close()

            def generator():
                yield from results

            alignments = {}
            multiple_faces_detected = False

            for filepath, dflimg in generator():
                if dflimg is None or not dflimg.has_data():
                    io.log_err (f"{filepath.name} 不是 DFL 图像文件")
                    continue

                source_filename = dflimg.get_source_filename()
                if source_filename is None:
                    # source_filename missing — derive from aligned filename stem
                    # e.g. "00003_0.jpg" → "00003"
                    aligned_stem = filepath.stem
                    if '_' in aligned_stem:
                        aligned_stem = aligned_stem.rsplit('_', 1)[0]
                    source_filename = aligned_stem + filepath.suffix
                    source_filepath = Path(source_filename)
                else:
                    source_filepath = Path(source_filename)
                source_filename_stem = source_filepath.stem

                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments_ar = alignments[ source_filename_stem ]
                alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath ) )

                if len(alignments_ar) > 1:
                    multiple_faces_detected = True

            if multiple_faces_detected:
                io.log_info ("")
                io.log_info ("警告：检测到多张人脸。一个源文件通常应只对应一个 alignment 文件。")
                io.log_info ("")

            for a_key in list(alignments.keys()):
                a_ar = alignments[a_key]
                if len(a_ar) > 1:
                    for _, filepath, source_filepath in a_ar:
                        io.log_info (f"alignment {filepath.name} 指向源文件 {source_filepath.name}")
                    io.log_info ("")

                alignments[a_key] = [ a[0] for a in a_ar]

            if multiple_faces_detected:
                io.log_info ("强烈建议将不同人脸分别处理（拆分 faceset）。")
                io.log_info ("可使用“恢复 aligned 原始文件名”来确定具体的重复项。")
                io.log_info ("")

            frames = [ InteractiveMergerSubprocessor.Frame( frame_info=FrameInfo(filepath=Path(p),
                                                                     landmarks_list=alignments.get(Path(p).stem, None)
                                                                    )
                                              )
                       for p in input_path_image_paths ]

            if multiple_faces_detected:
                io.log_info ("警告：检测到多张人脸，将不会使用运动模糊参数。")
                io.log_info ("")
            else:
                s = 256
                local_pts = [ (s//2-1, s//2-1), (s//2-1,0) ] #center+up
                frames_len = len(frames)
                for i in io.progress_bar_generator(range(len(frames)), "正在计算运动向量"):
                    fi_prev = frames[max(0, i-1)].frame_info
                    fi      = frames[i].frame_info
                    fi_next = frames[min(i+1, frames_len-1)].frame_info
                    if len(fi_prev.landmarks_list) == 0 or \
                       len(fi.landmarks_list) == 0 or \
                       len(fi_next.landmarks_list) == 0:
                            continue

                    mat_prev = LandmarksProcessor.get_transform_mat ( fi_prev.landmarks_list[0], s, face_type=FaceType.FULL)
                    mat      = LandmarksProcessor.get_transform_mat ( fi.landmarks_list[0]     , s, face_type=FaceType.FULL)
                    mat_next = LandmarksProcessor.get_transform_mat ( fi_next.landmarks_list[0], s, face_type=FaceType.FULL)

                    pts_prev = LandmarksProcessor.transform_points (local_pts, mat_prev, True)
                    pts      = LandmarksProcessor.transform_points (local_pts, mat, True)
                    pts_next = LandmarksProcessor.transform_points (local_pts, mat_next, True)

                    prev_vector = pts[0]-pts_prev[0]
                    next_vector = pts_next[0]-pts[0]

                    motion_vector = pts_next[0] - pts_prev[0]
                    fi.motion_power = npla.norm(motion_vector)

                    motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0,0],dtype=np.float32)

                    fi.motion_deg = -math.atan2(motion_vector[1],motion_vector[0])*180 / math.pi


        if len(frames) == 0:
            io.log_info ("输入目录中没有可合成的帧。")
        else:
            if pipeline:
                _run_pipeline(predictor_func, predictor_input_shape,
                              xseg_256_extract_func, face_enhancer_func,
                              cfg, frames, output_path, output_mask_path,
                              dfm_onnx_path=dfm_onnx_path,
                              xseg_onnx_path=xseg_onnx_path,
                              xseg_onnx_res=xseg_onnx_res,
                              run_on_cpu=run_on_cpu)
            else:
                InteractiveMergerSubprocessor (
                            is_interactive         = is_interactive,
                            merger_session_filepath = model.get_strpath_storage_for_file('merger_session.dat'),
                            predictor_func         = predictor_func,
                            predictor_input_shape  = predictor_input_shape,
                            face_enhancer_func     = face_enhancer_func,
                            xseg_256_extract_func = xseg_256_extract_func,
                            merger_config          = cfg,
                            frames                 = frames,
                            frames_root_path       = input_path,
                            output_path            = output_path,
                            output_mask_path       = output_mask_path,
                            model_iter             = model.get_iter(),
                            subprocess_count       = subprocess_count,
                        ).run()

        model.finalize()

    except Exception as e:
        print ( traceback.format_exc() )


"""
elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
filesdata = []
for filepath in io.progress_bar_generator(input_path_image_paths, "收集信息"):
    filepath = Path(filepath)

    dflimg = DFLIMG.x(filepath)
    if dflimg is None:
        io.log_err ("%s 不是 DFL 图像文件" % (filepath.name) )
        continue
    filesdata += [ ( FrameInfo(filepath=filepath, landmarks_list=[dflimg.get_landmarks()] ), dflimg.get_source_filename() ) ]

filesdata = sorted(filesdata, key=operator.itemgetter(1)) #按 source_filename 排序
frames = []
filesdata_len = len(filesdata)
for i in range(len(filesdata)):
    frame_info = filesdata[i][0]

    prev_temporal_frame_infos = []
    next_temporal_frame_infos = []

    for t in range (cfg.temporal_face_count):
        prev_frame_info = filesdata[ max(i -t, 0) ][0]
        next_frame_info = filesdata[ min(i +t, filesdata_len-1 )][0]

        prev_temporal_frame_infos.insert (0, prev_frame_info )
        next_temporal_frame_infos.append (   next_frame_info )

    frames.append ( InteractiveMergerSubprocessor.Frame(prev_temporal_frame_infos=prev_temporal_frame_infos,
                                                frame_info=frame_info,
                                                next_temporal_frame_infos=next_temporal_frame_infos) )
"""

#插值 landmarks（关键点）
#from facelib import LandmarksProcessor
#from facelib import FaceType
#a = sorted(alignments.keys())
#a_len = len(a)
#
#box_pts = 3
#box = np.ones(box_pts)/box_pts
#for i in range( a_len ):
#    if i >= box_pts and i <= a_len-box_pts-1:
#        af0 = alignments[ a[i] ][0] ##first face
#        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)
#
#        points = []
#
#        for j in range(-box_pts, box_pts+1):
#            af = alignments[ a[i+j] ][0] ##first face
#            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)
#            p = LandmarksProcessor.transform_points (af, m)
#            points.append (p)
#
#        points = np.array(points)
#        points_len = len(points)
#        t_points = np.transpose(points, [1,0,2])
#
#        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
#        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
#
#        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
#
#        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
