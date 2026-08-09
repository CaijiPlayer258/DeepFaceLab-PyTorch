"""
5-stage export pipeline.
Stages: FFmpeg extract -> Face detect -> Embed match -> Swap -> FFmpeg encode
Uses SQLite for inter-stage data, supporting concurrent access for multiprocessing.
"""
import os, sys, subprocess, json, shutil, sqlite3, traceback, time
from pathlib import Path
import cv2
import numpy as np
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

_FFMPEG_DIR = Path(__file__).parent.parent.parent / "ffmpeg"
_FFMPEG = str(_FFMPEG_DIR / "ffmpeg.exe")
_FFPROBE = str(_FFMPEG_DIR / "ffprobe.exe")

class StopRequested(Exception):
    """Raised when the user cancels the export."""
    pass


STAGE_NAMES = [
    "Extracting frames",
    "Detecting faces",
    "Matching identities",
    "Swapping faces",
    "Mask inference",
    "Merging frames",
    "Encoding video",
]


def _check_ffmpeg():
    try:
        subprocess.run([_FFMPEG, "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("FFmpeg not found.")


def _find_frame(dir_path, frame_idx):
    """Find frame file by index with flexible padding (08d → 05d → raw)."""
    from pathlib import Path
    p = Path(dir_path)
    for pad in [f"{frame_idx:08d}", f"{frame_idx:07d}", f"{frame_idx:05d}", f"{frame_idx:04d}", str(frame_idx)]:
        m = sorted(p.glob(f"{pad}.*"))
        if m: return m[0]
    return None


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def _init_db(db_path: Path):
    """Create and return a SQLite connection with WAL mode."""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_data (
            frame_idx INTEGER NOT NULL,
            face_idx  INTEGER NOT NULL,
            face_rect TEXT DEFAULT '[]',
            landmarks TEXT DEFAULT '[]',
            transform_mat TEXT DEFAULT '[]',
            out_size INTEGER DEFAULT 256,
            model TEXT DEFAULT '',
            face_id TEXT DEFAULT '',
            PRIMARY KEY (frame_idx, face_idx)
        )
    """)
    conn.execute("DELETE FROM face_data")
    conn.commit()
    return conn


def _db_insert_face(conn, frame_idx, face_idx, face_rect, landmarks, mat, out_size):
    conn.execute(
        "INSERT OR REPLACE INTO face_data VALUES (?,?,?,?,?,?,?,?)",
        (frame_idx, face_idx,
         json.dumps(face_rect),
         json.dumps(landmarks),
         json.dumps(mat),
         out_size, "", "")
    )


def _db_update_model(conn, frame_idx, face_idx, model, face_id):
    for _retry in range(5):
        try:
            conn.execute(
                "UPDATE face_data SET model=?, face_id=? WHERE frame_idx=? AND face_idx=?",
                (model, face_id, frame_idx, face_idx)
            )
            return
        except sqlite3.OperationalError as _e:
            if "locked" in str(_e) and _retry < 4:
                import time; time.sleep(0.2 * (_retry + 1))
                continue
            raise


def _db_get_faces_for_frame(conn, frame_idx):
    cur = conn.execute(
        "SELECT face_idx, face_rect, landmarks, transform_mat, out_size, model, face_id "
        "FROM face_data WHERE frame_idx=? ORDER BY face_idx", (frame_idx,)
    )
    rows = []
    for row in cur.fetchall():
        rows.append({
            "face_idx": row[0],
            "face_rect": json.loads(row[1]),
            "landmarks": json.loads(row[2]),
            "transform_mat": json.loads(row[3]),
            "out_size": row[4],
            "model": row[5],
            "face_id": row[6],
        })
    return rows


# ---------------------------------------------------------------------------
# Multiprocessing workers (module-level for pickling)
# ---------------------------------------------------------------------------

_mp_predictors = {}
_mp_xseg = None
_mp_cfg = None
_mp_db_path = None

# Hold DLL cookies so they aren't garbage collected
_dll_cookies = []

def _setup_dll_paths():
    """Add CUDA/cuDNN DLL directories to the search path for this process."""
    global _dll_cookies
    if _dll_cookies:
        return  # already set up
    import os, sys
    _paths = [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
               os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
               os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin")]
    for _p in _paths:
        if os.path.isdir(_p):
            try:
                _dll_cookies.append(os.add_dll_directory(_p))
            except:
                pass

def _worker_mp_init(all_model_paths, model_size, is_nchw, xseg_path, xseg_is_nchw, config_dict, db_path):
    """Per-process init: load all DFM models + XSeg + build config."""
    global _mp_predictors, _mp_xseg, _mp_cfg, _mp_db_path
    _setup_dll_paths()
    import torch as _torch
    _torch.cuda.is_available()
    import onnxruntime
    from pathlib import Path
    from MergeStudio.core.config import MergerConfigMasked
    import cv2, numpy as np

    _mp_predictors = {}
    for mp in all_model_paths:
        try:
            # Worker processes: CPU only (main process handles CUDA preview)
            sess = onnxruntime.InferenceSession(mp, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            inp_name = sess.get_inputs()[0].name
            model_key = Path(mp).stem.replace('_model', '').split('_')[0]
            _mp_predictors[model_key] = (sess, inp_name, model_size, is_nchw)
        except Exception as e:
            print(f"[Export] Worker DFM {Path(mp).name} FAILED: {e}")

    # XSeg (CPU)
    _mp_xseg = None
    if xseg_path and Path(xseg_path).exists():
        try:
            xs = onnxruntime.InferenceSession(xseg_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            xi = xs.get_inputs()[0].name
            def _xseg_extract(face_img_in):
                oh, ow = face_img_in.shape[:2]
                fr = face_img_in if (oh, ow) == (256, 256) else cv2.resize(face_img_in, (256, 256), interpolation=cv2.INTER_CUBIC)
                fc = np.clip(fr, 0, 1) if fr.max() < 2.0 else fr
                rgb = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB).astype(np.float32)
                if rgb.max() > 1.0: rgb /= 255.0
                inp = np.transpose(rgb, (2, 0, 1))[None, :, :, :] if xseg_is_nchw else rgb[None, :, :, :]
                out = np.squeeze(xs.run(None, {xi: inp})[0])
                return cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC) if out.shape[:2] != (oh, ow) else np.clip(out, 0, 1).astype(np.float32)
            _mp_xseg = _xseg_extract
        except Exception as e:
            print(f"[Export] Worker XSeg FAILED: {e}")

    _valid_keys = {'face_type','default_mode','mode','masked_hist_match','hist_match_threshold',
                   'mask_mode','seg_mode','erode_mask_modifier','blur_mask_modifier',
                   'motion_blur_power','output_face_scale','super_resolution_power',
                   'color_transfer_mode','image_denoise_power','bicubic_degrade_power',
                   'color_degrade_power','show_debug'}
    _mp_cfg = MergerConfigMasked(**{k: v for k, v in config_dict.items() if k in _valid_keys})
    _mp_db_path = db_path
    print(f"[Export] Worker ready: {len(_mp_predictors)} models ({list(_mp_predictors.keys())})" +
          (f" + XSeg" if _mp_xseg else ""))


def _worker_mp_fn(args):
    """Process a frame chunk. args tuple: (chunk_strs,). Returns frame count."""
    chunk_strs = args[0]
    import sqlite3, json, cv2, numpy as np
    from pathlib import Path
    from MergeStudio.core.merger import MergeMaskedFace

    conn = sqlite3.connect(str(_mp_db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")

    for fp_str in chunk_strs:
        fp = Path(fp_str)
        frame = cv2.imread(str(fp))
        if frame is None: continue
        idx = int(fp.stem)
        cur = conn.execute("SELECT face_idx, landmarks, model FROM face_data WHERE frame_idx=? ORDER BY face_idx", (idx,))
        rows = cur.fetchall()
        if not rows:
            continue

        swapped = frame.astype(np.float32)
        for row in rows:
            fi, lm_json, model_name = row
            lm = json.loads(lm_json)
            if len(lm) < 4: continue
            sess_info = None
            if model_name and model_name in _mp_predictors:
                sess_info = _mp_predictors[model_name]
            elif model_name:
                for k, v in _mp_predictors.items():
                    if k.startswith(model_name) or model_name.startswith(k):
                        sess_info = v; break
            if not sess_info and _mp_predictors:
                sess_info = next(iter(_mp_predictors.values()))
            if not sess_info: continue
            sess, inp_name, sz, nchw = sess_info

            def _pred(face_img):
                hh, ww = face_img.shape[:2]
                if hh != sz or ww != sz:
                    face_img = cv2.resize(face_img, (sz, sz), interpolation=cv2.INTER_LANCZOS4)
                img = face_img.astype(np.float32)
                inp = np.transpose(img, (2,0,1))[None,:,:,:] if nchw else img[None,:,:,:]
                outputs = sess.run(None, {inp_name: inp})
                fo = None
                for o in outputs:
                    if o.ndim == 4 and o.shape[-1] == 3: fo = o; break
                if fo is None: fo = outputs[0]
                if fo.shape[-1] == 1: fo = np.repeat(fo, 3, axis=-1)
                pm, dm = None, None
                if len(outputs) >= 3:
                    m0 = np.squeeze(outputs[0])
                    if m0.ndim == 3: m0 = m0[:,:,0]
                    pm = cv2.resize(m0.astype(np.float32), (sz,sz))
                    m2 = np.squeeze(outputs[2])
                    if m2.ndim == 3: m2 = m2[:,:,0]
                    dm = cv2.resize(m2.astype(np.float32), (sz,sz))
                while fo.ndim >= 5: fo = fo[0]
                if fo.ndim == 4: fo = fo[0] if not nchw else np.transpose(fo[0], (1,2,0))
                if fo.ndim == 2: fo = np.stack([fo]*3, axis=-1)
                if fo.shape[-1] > 3: fo = fo[:,:,:3]
                elif fo.shape[-1] == 1: fo = np.repeat(fo,3,axis=-1)
                if fo.shape[:2] != (sz,sz): fo = cv2.resize(fo,(sz,sz),interpolation=cv2.INTER_LANCZOS4)
                return (fo, pm, dm)

            try:
                lmk = np.array(lm, dtype=np.float32)
                fo, fo_mask = MergeMaskedFace(frame, lmk, _mp_cfg, _pred, xseg_256_extract_func=_mp_xseg)
                if fo.shape[:2] == swapped.shape[:2]:
                    if fo_mask is not None:
                        m = fo_mask.astype(np.float32) / 255.0
                        if m.ndim == 2: m = m[..., None]
                        swapped = swapped * (1 - m) + fo.astype(np.float32) * m
                    else:
                        swapped = fo.astype(np.float32)
                        if np.array_equal(fo, frame):
                            print(f"[Worker] MergeMaskedFace returned ORIGINAL", flush=True)
            except Exception:
                pass
        cv2.imwrite(str(fp), np.clip(swapped, 0, 255).astype(np.uint8))

    conn.close()
    return len(chunk_strs)

def _worker_swap_chunk(args):
    """Worker for Stage 4: swap faces in a chunk of frames using local predictor."""
    chunk, db_path, config_dict = args
    from MergeStudio.core.merger import MergeMaskedFace
    from MergeStudio.core.config import MergerConfigMasked

    _valid_keys = {'face_type','default_mode','mode','masked_hist_match','hist_match_threshold',
                   'mask_mode','seg_mode','erode_mask_modifier','blur_mask_modifier',
                   'motion_blur_power','output_face_scale','super_resolution_power',
                   'color_transfer_mode','image_denoise_power','bicubic_degrade_power',
                   'color_degrade_power','show_debug'}
    cfg = MergerConfigMasked(**{k: v for k, v in config_dict.items() if k in _valid_keys})

    # Build local predictor closure using the pre-loaded session
    def _predictor(face_img):
        hh, ww = face_img.shape[:2]
        if hh != _worker_size or ww != _worker_size:
            face_img = cv2.resize(face_img, (_worker_size, _worker_size), interpolation=cv2.INTER_LANCZOS4)
        img = face_img.astype(np.float32)
        if _worker_nchw:
            inp = np.transpose(img, (2, 0, 1))[None, :, :, :]
        else:
            inp = img[None, :, :, :]
        outputs = _worker_session.run(None, {_worker_input_name: inp})
        face_out = None
        for o in outputs:
            if o.ndim == 4 and o.shape[-1] == 3:
                face_out = o; break
        if face_out is None:
            face_out = outputs[0]
            if face_out.shape[-1] == 1:
                face_out = np.repeat(face_out, 3, axis=-1)
        pm, dm = None, None
        if len(outputs) >= 3:
            m0 = np.squeeze(outputs[0])
            if m0.ndim == 3: m0 = m0[:,:,0]
            pm = cv2.resize(m0.astype(np.float32), (_worker_size, _worker_size)) if m0.shape[:2]!=(_worker_size,_worker_size) else m0.astype(np.float32)
            m2 = np.squeeze(outputs[2])
            if m2.ndim == 3: m2 = m2[:,:,0]
            dm = cv2.resize(m2.astype(np.float32), (_worker_size, _worker_size)) if m2.shape[:2]!=(_worker_size,_worker_size) else m2.astype(np.float32)
        while face_out.ndim >= 5: face_out = face_out[0]
        if face_out.ndim == 4:
            face_out = face_out[0] if not _worker_nchw else np.transpose(face_out[0], (1,2,0))
        if face_out.ndim == 2:
            face_out = np.stack([face_out]*3, axis=-1)
        if face_out.shape[-1] > 3: face_out = face_out[:,:,:3]
        elif face_out.shape[-1] == 1: face_out = np.repeat(face_out, 3, axis=-1)
        if face_out.shape[:2] != (_worker_size, _worker_size):
            face_out = cv2.resize(face_out, (_worker_size, _worker_size), interpolation=cv2.INTER_LANCZOS4)
        return (face_out, pm, dm)

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    done = 0
    for fp_str in chunk:
        fp = Path(fp_str)
        frame = cv2.imread(str(fp))
        if frame is None:
            done += 1; continue
        idx = int(fp.stem)
        cur = conn.execute(
            "SELECT face_idx, landmarks, model FROM face_data WHERE frame_idx=? ORDER BY face_idx",
            (idx,)
        )
        rows = cur.fetchall()
        if not rows:
            # No faces — keep original frame (no copy needed, stays in place)
            done += 1; continue
        swapped = frame.astype(np.float32)
        for row in rows:
            _, lm_json, model_name = row
            lm = json.loads(lm_json)
            if len(lm) < 4: continue
            lmk = np.array(lm, dtype=np.float32)
            try:
                fo, _ = MergeMaskedFace(frame, lmk, cfg, _predictor,
                                        xseg_256_extract_func=_worker_xseg_func)
                if fo.shape[:2] == swapped.shape[:2]:
                    swapped = fo.astype(np.float32)
            except Exception:
                pass
        out = np.clip(swapped, 0, 255).astype(np.uint8)
        # Overwrite the original frame in-place
        cv2.imwrite(str(fp), out)
        done += 1
    conn.close()
    return done


# ---------------------------------------------------------------------------
# Multiprocessing coordinators
# ---------------------------------------------------------------------------

def _run_pool_shared(pool, tasks, worker_fn, progress, total_frames, stop_event, stage):
    """Submit tasks, workers push frame completions onto a Queue for smooth progress."""
    import multiprocessing
    progress_queue = multiprocessing.Manager().Queue()
    started_flag = multiprocessing.Manager().Value('b', False)

    wrapped_tasks = [(t, progress_queue, started_flag) for t in tasks]
    async_results = [pool.apply_async(worker_fn, (wt,)) for wt in wrapped_tasks]

    t0 = time.time()
    done = 0
    errors = 0
    abort_timeout = 120
    last_done = 0
    no_progress = 0

    while True:
        if stop_event and stop_event.is_set():
            pool.terminate(); pool.join()
            raise StopRequested("Export cancelled")

        # Drain queue
        new_jobs = 0
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                if msg == 'start':
                    pass
                elif msg == 'done':
                    done += 1
                elif isinstance(msg, tuple) and msg[0] == 'err':
                    errors += msg[1]
            except:
                break

        started = started_flag.value
        elapsed = time.time() - t0
        all_complete = all(r.ready() for r in async_results)

        if elapsed > abort_timeout and not started:
            print(f"[Export] Workers failed to start after {abort_timeout}s — aborting")
            pool.terminate(); pool.join()
            progress(stage, 1.0, "workers failed")
            return

        if all_complete:
            # Drain remaining
            while not progress_queue.empty():
                try:
                    msg = progress_queue.get_nowait()
                    if msg == 'done': done += 1
                except:
                    break
            print(f"[Export] Stage {stage} done: {done}/{total_frames}f {errors}err")
            break

        if done > last_done:
            no_progress = 0
            last_done = done
        else:
            no_progress += 1
        if no_progress > 60:
            print(f"[Export] Stall {done}/{total_frames}f {errors}err {elapsed:.0f}s")
            no_progress = 0

        fps = done / elapsed if elapsed > 0 else 0
        progress(stage, min(1.0, done / max(1, total_frames)),
                 f"{fps:.1f} it/s · {done}/{total_frames}" if stage == 3 else f"{done}/{total_frames}")
        time.sleep(0.5)

    pool.close(); pool.join()


def _worker_detect_chunk_wrapped(args):
    """Wrapper that unpacks progress queue and calls the real worker."""
    task, progress_queue, started_flag = args
    started_flag.value = True
    chunk, db_path, detector_str, landmarker_str, _, face_margin, _angle_segments = task  # res_scale no longer used
    import sqlite3, json
    from MergeStudio.core.detector.factory import DetectorFactory, LandmarkFactory, get_device_info as _gdi
    from MergeStudio.core.detector.pipeline import detect_and_align as _daa
    from facelib import FaceType
    _DETECT_H = 720  # scale so height=720px, width proportional
    device = _gdi()
    det = DetectorFactory.create(detector_str, device)
    lm = LandmarkFactory.create(landmarker_str, device)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for fp in chunk:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        idx = int(fp.stem)
        h, w = frame.shape[:2]
        scale = _DETECT_H / h  # e.g. 720/1080 = 0.667
        det_frame = frame if scale >= 1.0 else cv2.resize(frame, (int(w*scale), int(h*scale)))
        # Determine detection angles for this frame from angle_segments
        _da_for_frame = [0]
        for _seg in (_angle_segments or []):
            _s = _seg.get('start'); _e = _seg.get('end')
            if _s is not None and _e is not None and _s <= idx <= _e:
                try:
                    _da_for_frame = [int(x) for x in _seg.get('angles', '0').split(',') if x.strip()]
                except Exception as _e2:
                    print(f"[Export] angle parse error for frame {idx}: {_e2}", flush=True)
                break
        try:
            faces = _daa(det, lm, det_frame, FaceType.WHOLE_FACE, margin=face_margin,
                         detection_angles=_da_for_frame)
        except Exception as _de:
            import traceback as _tb
            print(f"[Export] detect_and_align FAILED for frame {idx}: {_de}", flush=True)
            _tb.print_exc()
            faces = []
        s = 1.0 / scale  # restore to original coordinates
        for fidx, face in enumerate(faces):
            mat = face.get('transform_mat')
            if mat is None: continue
            ms = mat.copy(); ms[0,2]*=s; ms[1,2]*=s
            r = face.get('face_rect')
            rs = tuple(int(v*s) for v in r) if r else (0,0,0,0)
            lmk = face.get('landmarks')
            ll = []
            if lmk is not None:
                lmk_scaled = lmk * s
                ll = lmk_scaled.tolist() if isinstance(lmk_scaled, np.ndarray) else (np.array(lmk, dtype=np.float32) * s).tolist()
            _db_insert_face(conn, idx, fidx, list(rs), ll, ms.tolist(), face.get('out_size', 256))
        conn.commit()
        try:
            progress_queue.put('done', block=False)
        except:
            pass
    conn.close()
    return len(chunk)


def _worker_swap_chunk_wrapped(args):
    """Wrapper that unpacks progress queue and calls the real worker."""
    task, progress_queue, started_flag = args
    started_flag.value = True
    chunk, db_path, config_dict = task

    from MergeStudio.core.merger import MergeMaskedFace
    from MergeStudio.core.config import MergerConfigMasked
    _valid_keys = {'face_type','default_mode','mode','masked_hist_match','hist_match_threshold',
                   'mask_mode','seg_mode','erode_mask_modifier','blur_mask_modifier',
                   'motion_blur_power','output_face_scale','super_resolution_power',
                   'color_transfer_mode','image_denoise_power','bicubic_degrade_power',
                   'color_degrade_power','show_debug'}
    cfg = MergerConfigMasked(**{k: v for k, v in config_dict.items() if k in _valid_keys})

    def _make_predictor(session, inp_name):
        """Create a predictor closure bound to a specific ONNX session."""
        def _pred(face_img):
            hh, ww = face_img.shape[:2]
            if hh != _worker_size or ww != _worker_size:
                face_img = cv2.resize(face_img, (_worker_size, _worker_size), interpolation=cv2.INTER_LANCZOS4)
            img = face_img.astype(np.float32)
            if _worker_nchw:
                inp = np.transpose(img, (2, 0, 1))[None, :, :, :]
            else:
                inp = img[None, :, :, :]
            outputs = session.run(None, {inp_name: inp})
            face_out = None
            for o in outputs:
                if o.ndim == 4 and o.shape[-1] == 3:
                    face_out = o; break
            if face_out is None:
                face_out = outputs[0]
                if face_out.shape[-1] == 1: face_out = np.repeat(face_out, 3, axis=-1)
            pm, dm = None, None
            if len(outputs) >= 3:
                m0 = np.squeeze(outputs[0])
                if m0.ndim == 3: m0 = m0[:,:,0]
                pm = cv2.resize(m0.astype(np.float32), (_worker_size, _worker_size)) if m0.shape[:2]!=(_worker_size,_worker_size) else m0.astype(np.float32)
                m2 = np.squeeze(outputs[2])
                if m2.ndim == 3: m2 = m2[:,:,0]
                dm = cv2.resize(m2.astype(np.float32), (_worker_size, _worker_size)) if m2.shape[:2]!=(_worker_size,_worker_size) else m2.astype(np.float32)
            while face_out.ndim >= 5: face_out = face_out[0]
            if face_out.ndim == 4:
                face_out = face_out[0] if not _worker_nchw else np.transpose(face_out[0], (1,2,0))
            if face_out.ndim == 2: face_out = np.stack([face_out]*3, axis=-1)
            if face_out.shape[-1] > 3: face_out = face_out[:,:,:3]
            elif face_out.shape[-1] == 1: face_out = np.repeat(face_out, 3, axis=-1)
            if face_out.shape[:2] != (_worker_size, _worker_size):
                face_out = cv2.resize(face_out, (_worker_size, _worker_size), interpolation=cv2.INTER_LANCZOS4)
            return (face_out, pm, dm)
        return _pred

    def _get_pred_for_model(model_name):
        """Find the right session + predictor for a model name (with prefix fallback)."""
        if not model_name:
            return _make_predictor(*next(iter(_worker_predictors.values()))) if _worker_predictors else None
        if model_name in _worker_predictors:
            return _make_predictor(*_worker_predictors[model_name])
        for key, (sess, inp) in _worker_predictors.items():
            if key.startswith(model_name) or model_name.startswith(key):
                return _make_predictor(sess, inp)
        # Fallback to first model
        return _make_predictor(*next(iter(_worker_predictors.values()))) if _worker_predictors else None

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    # Pre-build predictors for known model keys
    _pred_cache = {mk: _make_predictor(*vp) for mk, vp in _worker_predictors.items()}
    def _resolve_pred(mn):
        if not mn and _pred_cache:
            return next(iter(_pred_cache.values()))
        if mn in _pred_cache:
            return _pred_cache[mn]
        for k, p in _pred_cache.items():
            if k.startswith(mn) or mn.startswith(k):
                return p
        return next(iter(_pred_cache.values())) if _pred_cache else None

    for fp_str in chunk:
        fp = Path(fp_str)
        frame = cv2.imread(str(fp))
        if frame is None:
            try: progress_queue.put('done', block=False)
            except: pass
            continue
        idx = int(fp.stem)
        cur = conn.execute("SELECT face_idx, landmarks, model FROM face_data WHERE frame_idx=? ORDER BY face_idx", (idx,))
        rows = cur.fetchall()
        if rows:
            swapped = frame.astype(np.float32)
            for row in rows:
                _, lm_json, model_name = row
                lm = json.loads(lm_json)
                if len(lm) < 4: continue
                lmk = np.array(lm, dtype=np.float32)
                pred = _resolve_pred(model_name)
                if pred is None: continue
                try:
                    fo, _ = MergeMaskedFace(frame, lmk, cfg, pred, xseg_256_extract_func=_worker_xseg_func)
                    if fo.shape[:2] == swapped.shape[:2]:
                        swapped = fo.astype(np.float32)
                except Exception:
                    try: progress_queue.put(('err', 1), block=False)
                    except: pass
            out = np.clip(swapped, 0, 255).astype(np.uint8)
            cv2.imwrite(str(fp), out)
        try: progress_queue.put('done', block=False)
        except: pass
    conn.close()
    return len(chunk)


def _stage2_detect_frames_mp(frames_dir, db_path, detector_str, landmarker_str, res_scale, face_margin, progress, stop_event, num_workers, angle_segments=None):
    """Stage 2 using multiprocessing Pool with shared progress."""
    frames = sorted(frames_dir.glob("*.[jp][pn]g"))
    total = len(frames)
    if total == 0: return
    chunks = [frames[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]
    print(f"[Export] Stage2 MP: {total} frames, {len(chunks)} workers, margin={face_margin}")
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        tasks = [(chunk, str(db_path), detector_str, landmarker_str, res_scale, face_margin, angle_segments) for chunk in chunks]
        _run_pool_shared(pool, tasks, _worker_detect_chunk_wrapped, progress, total, stop_event, 1)



# ============================================================
# Phase workers for Stage 4 (module-level, picklable)
# ============================================================
_phase_swap_dir = None
_phase_mask_dir = None
_phase_frames_dir = None
_phase_db_path = None
_phase_all_faces = []  # [(idx, fi, lmj), ...]
_phase_model_paths = []
_phase_xseg_path = None
_phase_model_size = 416
_phase_is_nchw = False
_phase_merge_cfg = None  # MergerConfigMasked for merge workers
_phase_merge_out_dir = None  # output dir for DFL merged frames

def _phase_init_all(mp, sz, nchw, xp, sd, md, fd, dp, af):
    print(f"[SwapInit] loading {mp}", flush=True)
    global _phase_swap_sess, _phase_swap_inp, _phase_swap_sz, _phase_swap_nchw
    global _phase_xseg_path, _phase_swap_dir, _phase_mask_dir, _phase_frames_dir, _phase_db_path, _phase_all_faces
    _setup_dll_paths()
    # Import torch first so it loads its own cuDNN before onnxruntime loads cuDNN
    import torch as _torch
    _torch.cuda.is_available()
    print(f"[SwapInit] torch={_torch.__version__} cuda={_torch.cuda.is_available()}", flush=True)
    import onnxruntime as _ort2
    _phase_swap_sess = _ort2.InferenceSession(str(mp), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f'[SwapInit] providers: {_phase_swap_sess.get_providers()}', flush=True)
    _phase_swap_inp = _phase_swap_sess.get_inputs()[0].name
    _phase_swap_sz = sz; _phase_swap_nchw = nchw
    _phase_xseg_path = xp; _phase_swap_dir = sd; _phase_mask_dir = md
    _phase_frames_dir = fd; _phase_db_path = dp; _phase_all_faces = af

def _phase_swap_worker(tasks):
    import cv2, numpy as np, json
    from pathlib import Path
    from facelib.LandmarksProcessor import get_transform_mat as _gtm
    from facelib import FaceType
    if not tasks: return 0
    res = 0
    for idx, fi, lmj in tasks:
        fp = _find_frame(_phase_frames_dir, idx)
        if fp is None: continue
        frame = cv2.imread(str(fp))
        if frame is None: continue
        lm = np.array(json.loads(lmj), dtype=np.float32)
        if len(lm) < 4: continue
        face_mat = _gtm(lm, _phase_swap_sz, FaceType.WHOLE_FACE)
        crop = cv2.warpAffine(frame, face_mat, (_phase_swap_sz, _phase_swap_sz), flags=cv2.INTER_CUBIC)
        fn = crop.astype(np.float32) / 255.0
        inp = np.transpose(fn, (2,0,1))[None,:,:,:] if _phase_swap_nchw else fn[None,:,:,:]
        out = _phase_swap_sess.run(None, {_phase_swap_inp: inp})
        fo = None
        for o in out:
            if o.ndim == 4 and o.shape[-1] == 3: fo = o; break
        if fo is None: fo = out[0]
        if fo.shape[-1] == 1: fo = np.repeat(fo,3,axis=-1)
        while fo.ndim >= 5: fo = fo[0]
        if fo.ndim == 4: fo = fo[0] if not _phase_swap_nchw else np.transpose(fo[0],(1,2,0))
        if fo.ndim == 2: fo = np.stack([fo]*3, axis=-1)
        if fo.shape[-1] > 3: fo = fo[:,:,:3]
        elif fo.shape[-1] == 1: fo = np.repeat(fo,3,axis=-1)
        if fo.shape[:2] != (_phase_swap_sz,_phase_swap_sz): fo = cv2.resize(fo,(_phase_swap_sz,_phase_swap_sz),interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(Path(_phase_swap_dir) / f"{idx:08d}_{fi}.jpg"), np.clip(fo * 255, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Save DFM predicted masks for later mask mode selection
        if len(out) >= 3:
            pm = np.squeeze(out[0])
            if pm.ndim == 3: pm = pm[:,:,0]
            pm = cv2.resize(pm, (_phase_swap_sz, _phase_swap_sz), interpolation=cv2.INTER_CUBIC) if pm.shape[:2]!=(_phase_swap_sz,_phase_swap_sz) else pm
            pm = np.clip(pm * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(Path(_phase_swap_dir) / f"{idx:08d}_{fi}_prdm.png"), pm)
            dm = np.squeeze(out[2])
            if dm.ndim == 3: dm = dm[:,:,0]
            dm = cv2.resize(dm, (_phase_swap_sz, _phase_swap_sz), interpolation=cv2.INTER_CUBIC) if dm.shape[:2]!=(_phase_swap_sz,_phase_swap_sz) else dm
            dm = np.clip(dm * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(Path(_phase_swap_dir) / f"{idx:08d}_{fi}_dstdm.png"), dm)
        res += 1
    return res

def _phase_mask_init(xp, sd, md, fd, dp, af):
    global _phase_xseg_path, _phase_swap_dir, _phase_mask_dir, _phase_frames_dir, _phase_db_path, _phase_all_faces
    global _phase_mask_sess, _phase_mask_inp
    _phase_xseg_path = xp; _phase_swap_dir = sd; _phase_mask_dir = md
    _phase_frames_dir = fd; _phase_db_path = dp; _phase_all_faces = af
    _setup_dll_paths()
    import torch as _torch
    _torch.cuda.is_available()
    import onnxruntime as _ort3
    if _phase_xseg_path:
        _phase_mask_sess = _ort3.InferenceSession(_phase_xseg_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        _phase_mask_inp = _phase_mask_sess.get_inputs()[0].name

def _phase_mask_worker(tasks):
    if not _phase_xseg_path: return len(tasks)
    import cv2, numpy as np, json
    from pathlib import Path
    from facelib.LandmarksProcessor import get_transform_mat
    from facelib import FaceType
    res = 0
    for idx, fi, lmj, _model in tasks:
        lm = np.array(json.loads(lmj), dtype=np.float32)
        if len(lm) < 4: continue
        # ---- XSeg-prd: run XSeg on the swapped face ----
        swap_path = Path(_phase_swap_dir) / f"{idx:08d}_{fi}.jpg"
        if swap_path.exists():
            fo = cv2.imread(str(swap_path)).astype(np.float32) / 255.0
        else:
            fp = _find_frame(_phase_frames_dir, idx)
            if fp is None: continue
            frame = cv2.imread(str(fp))
            if frame is None: continue
            mat = get_transform_mat(lm, 256, FaceType.WHOLE_FACE)
            fo = cv2.warpAffine(frame, mat, (256,256), flags=cv2.INTER_CUBIC).astype(np.float32)/255.0
        def _run_xseg(face_img):
            xin = face_img if face_img.shape[:2] == (256,256) else cv2.resize(face_img, (256,256), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(xin, cv2.COLOR_BGR2RGB).astype(np.float32)
            inp = np.transpose(rgb, (2,0,1))[None,:,:,:]
            out = np.squeeze(_phase_mask_sess.run(None, {_phase_mask_inp: inp})[0])
            if out.shape[:2] != face_img.shape[:2]:
                out = cv2.resize(out, (face_img.shape[1], face_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            return np.clip(out, 0, 1).astype(np.float32)
        xseg_prd = _run_xseg(fo)
        cv2.imwrite(str(Path(_phase_mask_dir) / f"{idx:08d}_{fi}_xseg_prd.png"), np.clip(xseg_prd * 255, 0, 255).astype(np.uint8))
        # ---- XSeg-dst: run XSeg on the aligned dst face from original frame ----
        fp = _find_frame(_phase_frames_dir, idx)
        if fp:
            frame = cv2.imread(str(fp))
            if frame is not None:
                dst_mat = get_transform_mat(lm, 256, FaceType.WHOLE_FACE)
                dst_face = cv2.warpAffine(frame, dst_mat, (256,256), flags=cv2.INTER_CUBIC).astype(np.float32) / 255.0
                xseg_dst = _run_xseg(dst_face)
                cv2.imwrite(str(Path(_phase_mask_dir) / f"{idx:08d}_{fi}_xseg_dst.png"), np.clip(xseg_dst * 255, 0, 255).astype(np.uint8))
        res += 1
    return res

def _phase_merge_init(sd, md, fd, dp, af, cfg, od, sz=None):
    print(f"[MergeW] init", flush=True)
    global _phase_swap_dir, _phase_mask_dir, _phase_frames_dir, _phase_db_path, _phase_all_faces, _phase_merge_cfg, _phase_merge_out_dir, _phase_swap_sz
    _phase_swap_dir = sd; _phase_mask_dir = md; _phase_frames_dir = fd; _phase_db_path = dp; _phase_all_faces = af
    _phase_merge_cfg = cfg; _phase_merge_out_dir = od; _phase_swap_sz = sz
    _setup_dll_paths()

def _phase_merge_worker(fp_str):
    import cv2, numpy as np, json, sqlite3
    from pathlib import Path
    from facelib.LandmarksProcessor import get_transform_mat, get_image_hull_mask
    from facelib import FaceType
    fp = Path(fp_str)
    frame = cv2.imread(str(fp))
    if frame is None: return 0
    idx = int(fp.stem)
    fc = frame.astype(np.float32) / 255.0  # original frame in float
    cfg = _phase_merge_cfg
    mask_mode = cfg.get('mask_mode', 6) if cfg else 6
    mode = cfg.get('mode', 'overlay') if cfg else 'overlay'
    ct_mode = cfg.get('color_transfer_mode', 'none') if cfg else 'none'
    erode = cfg.get('erode_mask_modifier', 0) if cfg else 0
    blur = cfg.get('blur_mask_modifier', 0) if cfg else 0

    layers = []
    for (fidx, ffi, _, _) in _phase_all_faces:
        if fidx != idx: continue
        swap_path = Path(_phase_swap_dir) / f"{idx:08d}_{ffi}.jpg"
        if not swap_path.exists(): continue
        fo = cv2.imread(str(swap_path)).astype(np.float32) / 255.0

        conn = sqlite3.connect(_phase_db_path, timeout=60)
        cur = conn.execute("SELECT landmarks FROM face_data WHERE frame_idx=? AND face_idx=?", (idx, ffi))
        row = cur.fetchone()
        conn.close()
        if not row: continue
        lm = np.array(json.loads(row[0]), dtype=np.float32)

        # Determine mask
        wrk_mask = None
        if mask_mode == 1:
            frame_hull = get_image_hull_mask(frame.shape, lm)
            fmat_inv = get_transform_mat(lm, _phase_swap_sz, FaceType.WHOLE_FACE, scale=1.0)
            wrk_mask = cv2.warpAffine(frame_hull, fmat_inv, (fo.shape[1], fo.shape[0]), flags=cv2.INTER_CUBIC)
        elif mask_mode in (2, 4, 5):
            p = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_prdm.png"), cv2.IMREAD_GRAYSCALE)
            if p is not None:
                wrk_mask = p.astype(np.float32) / 255.0
                if mask_mode == 4:
                    d = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_dstdm.png"), cv2.IMREAD_GRAYSCALE)
                    if d is not None: wrk_mask = wrk_mask * (d.astype(np.float32) / 255.0)
                elif mask_mode == 5:
                    d = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_dstdm.png"), cv2.IMREAD_GRAYSCALE)
                    if d is not None: wrk_mask = np.clip(wrk_mask + d.astype(np.float32) / 255.0, 0, 1)
        elif mask_mode == 3:
            d = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_dstdm.png"), cv2.IMREAD_GRAYSCALE)
            if d is not None: wrk_mask = d.astype(np.float32) / 255.0
        elif mask_mode in (6, 8):
            p = cv2.imread(str(Path(_phase_mask_dir) / f"{idx:08d}_{ffi}_xseg_prd.png"), cv2.IMREAD_GRAYSCALE)
            if p is not None:
                wrk_mask = p.astype(np.float32) / 255.0
                if mask_mode == 8:
                    d = cv2.imread(str(Path(_phase_mask_dir) / f"{idx:08d}_{ffi}_xseg_dst.png"), cv2.IMREAD_GRAYSCALE)
                    if d is not None: wrk_mask = wrk_mask * (d.astype(np.float32) / 255.0)
        elif mask_mode == 7:
            d = cv2.imread(str(Path(_phase_mask_dir) / f"{idx:08d}_{ffi}_xseg_dst.png"), cv2.IMREAD_GRAYSCALE)
            if d is not None: wrk_mask = d.astype(np.float32) / 255.0
        elif mask_mode == 9:
            p1 = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_prdm.png"), cv2.IMREAD_GRAYSCALE)
            d1 = cv2.imread(str(Path(_phase_swap_dir) / f"{idx:08d}_{ffi}_dstdm.png"), cv2.IMREAD_GRAYSCALE)
            p2 = cv2.imread(str(Path(_phase_mask_dir) / f"{idx:08d}_{ffi}_xseg_prd.png"), cv2.IMREAD_GRAYSCALE)
            d2 = cv2.imread(str(Path(_phase_mask_dir) / f"{idx:08d}_{ffi}_xseg_dst.png"), cv2.IMREAD_GRAYSCALE)
            wrk_mask = np.ones((_phase_swap_sz, _phase_swap_sz), dtype=np.float32)
            if p1 is not None: wrk_mask *= p1.astype(np.float32) / 255.0
            if d1 is not None: wrk_mask *= d1.astype(np.float32) / 255.0
            if p2 is not None: wrk_mask *= p2.astype(np.float32) / 255.0
            if d2 is not None: wrk_mask *= d2.astype(np.float32) / 255.0
        if wrk_mask is None:
            wrk_mask = np.ones_like(fo[..., 0])
        if wrk_mask.shape[:2] != fo.shape[:2]:
            wrk_mask = cv2.resize(wrk_mask, (fo.shape[1], fo.shape[0]), interpolation=cv2.INTER_CUBIC)
        if erode != 0 or blur != 0:
            _pad = 32
            _msk = np.pad(wrk_mask, _pad)
            if erode > 0:
                _msk = cv2.erode(_msk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode)), iterations=1)
            elif erode < 0:
                _msk = cv2.dilate(_msk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-erode, -erode)), iterations=1)
            if blur > 0:
                _b = blur + (1 - blur % 2)
                _msk = cv2.GaussianBlur(_msk, (_b, _b), 0)
            wrk_mask = _msk[_pad:-_pad, _pad:-_pad]
        wrk_mask = np.clip(wrk_mask, 0, 1)

        sz = fo.shape[0]
        fmat = get_transform_mat(lm, sz, FaceType.WHOLE_FACE, scale=1.0)

        # Apply color transfer to the swapped face
        if ct_mode not in ('none', '0', 0, 'none'):
            _sct = str(ct_mode)
            dw = cv2.warpAffine(fc, fmat, (sz, sz), flags=cv2.INTER_CUBIC)
            from core.imagelib.color_transfer import (reinhard_color_transfer, linear_color_transfer,
                color_transfer_mkl, color_transfer_idt, color_transfer_sot, color_transfer_mix)
            _mask_a = wrk_mask[..., None] if wrk_mask.ndim == 2 else wrk_mask
            try:
                if _sct in ('rct', '1'):
                    fo = reinhard_color_transfer(fo, dw, target_mask=_mask_a, source_mask=_mask_a)
                elif _sct in ('lct', '2'):
                    fo = linear_color_transfer(fo, dw)
                elif _sct in ('mkl', '3'):
                    fo = color_transfer_mkl(fo, dw)
                elif _sct in ('mkl-m', '4'):
                    fo = color_transfer_mkl(fo * _mask_a, dw * _mask_a)
                elif _sct in ('idt', '5'):
                    fo = color_transfer_idt(fo, dw)
                elif _sct in ('idt-m', '6'):
                    fo = color_transfer_idt(fo * _mask_a, dw * _mask_a)
                elif _sct in ('sot-m', '7'):
                    fo = color_transfer_sot(fo * _mask_a, dw * _mask_a, steps=10, batch_size=30)
                elif _sct in ('mix-m', '8'):
                    fo = color_transfer_mix(fo * _mask_a, dw * _mask_a)
            except Exception as _ct_e:
                print(f"[MergeW]   face {ffi}: CT {_sct} FAILED: {_ct_e}", flush=True)
        if 'hist-match' in mode:
            dw = cv2.warpAffine(fc, fmat, (sz, sz), flags=cv2.INTER_CUBIC)
            _hm = int(cfg.get('hist_match_threshold', 255)) if cfg else 255
            from core.imagelib.color_transfer import color_hist_match as _chm
            fo = _chm(fo, dw, hist_match_threshold=_hm)

        h, w = frame.shape[:2]
        mw = cv2.warpAffine(wrk_mask, fmat, (w,h), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)
        mw = np.clip(mw, 0, 1)
        if mw.ndim == 2: mw = mw[..., None]
        fw = cv2.warpAffine(fo, fmat, (w,h), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC)
        fw = np.clip(fw, 0, 1)
        layers.append((fw, mw))

    # Composite all face layers onto original frame
    if layers:
        swapped = fc.copy()
        frame_u8 = frame.astype(np.uint8) if 'seamless' in mode else None
        for fw, mw in layers:
            if 'seamless' in mode:
                try:
                    mw_bin = np.clip(mw * 255, 0, 255).astype(np.uint8)
                    fw_u8 = np.clip(fw * 255, 0, 255).astype(np.uint8)
                    coords = cv2.findNonZero(mw_bin)
                    if coords is not None:
                        x, y, wm, hm = cv2.boundingRect(coords)
                        center = (x + wm//2, y + hm//2)
                        sc = cv2.seamlessClone(fw_u8, frame_u8, mw_bin, center, cv2.NORMAL_CLONE).astype(np.float32) / 255.0
                        swapped = swapped * (1 - mw) + sc * mw
                    else:
                        swapped = swapped * (1 - mw) + fw * mw
                except Exception:
                    swapped = swapped * (1 - mw) + fw * mw
            else:
                swapped = swapped * (1 - mw) + fw * mw
        swapped = np.clip(swapped * 255, 0, 255)
    else:
        swapped = frame.astype(np.float32)
    if _phase_merge_out_dir:
        # DFL mode: write to merged/ as PNG
        out_path = Path(_phase_merge_out_dir) / (fp.stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = fp
    cv2.imwrite(str(out_path), np.clip(swapped,0,255).astype(np.uint8))
    return 1

def _stage4_swap_faces_mp(frames_dir, db_path, work_dir, config, face_model_map, progress, stop_event, num_workers, cut_segments=None):
    """Phased Stage 4: Swap(one model at a time) -> Mask -> Merge."""
    import multiprocessing as _mp, time
    from pathlib import Path
    frames = sorted(frames_dir.glob("*.[jp][pn]g"))
    total = len(frames)
    if total == 0: return
    from MergeStudio.core.model_loader import model_loader
    all_loaded = list(model_loader._sessions.keys())
    if not all_loaded:
        progress(6, 1.0, "No model"); return

    used_names = set(v for v in face_model_map.values())
    all_model_paths = []
    for mp in all_loaded:
        mk = Path(mp).stem.replace('_model', '').split('_')[0]
        if mk in used_names or not used_names:
            all_model_paths.append(mp)
    if not all_model_paths:
        all_model_paths = all_loaded

    _setup_dll_paths()
    import onnxruntime
    probe_sess = onnxruntime.InferenceSession(str(all_model_paths[0]), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inp = probe_sess.get_inputs()[0]
    shape = inp.shape
    is_nchw = len(shape) >= 4 and shape[1] in (1, 3)
    model_size = shape[2] if is_nchw else shape[1]

    xseg_path = None
    _xc = Path(__file__).parent.parent.parent / "workspace" / "model" / "XSegLite" / "xseglite.onnx"
    if _xc.exists(): xseg_path = str(_xc)

    global _phase_swap_dir, _phase_mask_dir, _phase_frames_dir, _phase_db_path
    global _phase_all_faces, _phase_model_paths, _phase_xseg_path, _phase_model_size, _phase_is_nchw
    _phase_swap_dir = str(work_dir / "swap_pred")
    _phase_mask_dir = str(work_dir / "masks")
    _phase_frames_dir = str(frames_dir)
    _phase_db_path = str(db_path)
    _phase_model_paths = [str(p) for p in all_model_paths]
    _phase_xseg_path = xseg_path
    _phase_model_size = model_size
    _phase_is_nchw = is_nchw
    Path(_phase_swap_dir).mkdir(parents=True, exist_ok=True)
    Path(_phase_mask_dir).mkdir(parents=True, exist_ok=True)

    # Collect all faces from DB; filter to only assigned models
    conn = sqlite3.connect(str(db_path), timeout=30)
    cur = conn.execute("SELECT frame_idx, face_idx, landmarks, model FROM face_data ORDER BY frame_idx, face_idx")
    _phase_all_faces = cur.fetchall()
    conn.close()
    _phase_all_faces = [f for f in _phase_all_faces if f[3]]  # only faces with model assigned

    # If no faces assigned by ArcFace, default to all faces → first model
    if not _phase_all_faces and all_model_paths:
        _first_mk = Path(all_model_paths[0]).stem.replace('_model', '').split('_')[0]
        conn = sqlite3.connect(str(db_path), timeout=60)
        cur = conn.execute("SELECT frame_idx, face_idx, landmarks, ? FROM face_data", (_first_mk,))
        _phase_all_faces = cur.fetchall()
        conn.close()
        print(f"[Export] No assigned faces — defaulting all to {_first_mk} ({len(_phase_all_faces)} faces)", flush=True)

    _mp_ctx = _mp.get_context("spawn")
    _t0 = time.time()
    faces_by_model = {}
    for (idx, fi, lmj, model) in _phase_all_faces:
        faces_by_model.setdefault(model, []).append((idx, fi, lmj))

    # ---- Phase 1: SWAP (one model at a time) ----
    _total_swap_faces = sum(len(v) for v in faces_by_model.values() if v)
    _done_swap_total = 0
    for _mpath in all_model_paths:
        _mkey = Path(_mpath).stem.replace('_model', '').split('_')[0]
        _assignments = faces_by_model.get(_mkey, [])
        if not _assignments: continue
        total_faces = len(_assignments)
        print(f"[Export] Swap: {Path(_mpath).name} -> {total_faces} faces", flush=True)
        weight = total_faces / max(1, _total_swap_faces)
        with _mp_ctx.Pool(processes=min(num_workers, total_faces, 6),
                          initializer=_phase_init_all,
                          initargs=(str(_mpath), model_size, is_nchw, xseg_path,
                                    _phase_swap_dir, _phase_mask_dir, _phase_frames_dir,
                                    str(db_path), _phase_all_faces)) as pool:
            done_s = 0
            for r in pool.imap_unordered(_phase_swap_worker, [[a] for a in _assignments],
                                         chunksize=max(1, total_faces // 20)):
                done_s += r
                _done_swap_total += r
                pct = _done_swap_total / max(1, _total_swap_faces)
                progress(3, pct,
                         f"{_done_swap_total}/{_total_swap_faces} · {_done_swap_total/(time.time()-_t0):.1f}it/s")
                # Debug: log every 10%
                if int(pct * 10) > int((pct - r/max(1,_total_swap_faces)) * 10):
                    print(f"[Progress] Swap {pct*100:.0f}% tick={_done_swap_total}", flush=True)

    # ---- Phase 2: MASK ----
    total_m = len(_phase_all_faces)
    if total_m == 0:
        print("[Export] Mask: no faces to process, skipping", flush=True)
        progress(4, 1.0, "No faces")
    else:
        print(f"[Export] Mask: {total_m} faces", flush=True)
        progress(4, 0.0, f"starting {total_m} faces")
        with _mp_ctx.Pool(processes=min(num_workers, total_m, 4),
                          initializer=_phase_mask_init,
                      initargs=(xseg_path, _phase_swap_dir, _phase_mask_dir,
                                _phase_frames_dir, str(db_path), _phase_all_faces)) as pool:
            done_m = 0
            for r in pool.imap_unordered(_phase_mask_worker, [[a] for a in _phase_all_faces],
                                         chunksize=max(1, total_m // 20)):
                done_m += r
                progress(4, done_m / total_m,
                         f"{done_m}/{total_m} · {done_m/(time.time()-_t0):.1f}it/s")
            progress(4, 1.0, "Done")

    # ---- Phase 3: MERGE ----
    print(f"[Export] Merge: {total} frames", flush=True)
    progress(5, 0.0, f"starting {total} frames")
    mchunks = [str(fp) for fp in frames]
    _merge_config = config if isinstance(config, dict) else {}
    _merge_out = _phase_merge_out_dir  # None → overwrite in-place; string → write to dir
    with _mp_ctx.Pool(processes=min(num_workers, total, 8),
                      initializer=_phase_merge_init,
                      initargs=(_phase_swap_dir, _phase_mask_dir, _phase_frames_dir,
                                str(db_path), _phase_all_faces, _merge_config, _merge_out, model_size)) as pool:
        done_g = 0
        for r in pool.imap_unordered(_phase_merge_worker, mchunks,
                                     chunksize=max(1, total // 20)):
            done_g += r
            progress(5, done_g / total,
                     f"{done_g}/{total} · {done_g/(time.time()-_t0):.1f}it/s")
    progress(5, 1.0, "Done")
    print(f"[Export] Stage4 done in {time.time()-_t0:.1f}s", flush=True)
def run_export_pipeline(
    video_path: str, output_path: str,
    image_format: str = "jpg",
    encoder: str = "h264_nvenc",
    config: dict = None,
    face_db: dict = None,
    face_model_map: dict = None,
    cut_segments: list = None,
    angle_segments: list = None,
    detector: str = "YOLOv8",
    landmarker: str = "insightface-2d106det",
    res_scale: float = 0.5,
    hwaccel: str = '',
    progress_callback: Optional[Callable] = None,
    stop_event=None,
    num_workers: int = 4,
):
    """Run the full export pipeline. Uses multiprocessing for detect + swap stages."""
    _check_ffmpeg()

    # Detect DFL standard project mode:
    # project dir = video_path stem, containing aligned/ folder and frame files
    _video = Path(video_path)
    _proj_dir = _video.parent / _video.stem
    _aligned_dir = _proj_dir / "aligned"
    _is_dfl = False
    if _proj_dir.is_dir() and _aligned_dir.is_dir():
        _frames = sorted(_proj_dir.glob("[0-9]*.[jp][pn]g"))
        _aligned = sorted(_aligned_dir.glob("*.jpg"))
        if len(_frames) >= 10 and len(_aligned) >= 1:
            try:
                from DFLIMG import DFLIMG as _D
                _sample = _D.load(str(_aligned[0]))
                _is_dfl = _sample is not None and _sample.has_data()
            except:
                pass
    if _is_dfl:
        print(f"[Export] DFL project: {_proj_dir} ({len(_frames)}f {len(_aligned)}a)", flush=True)

    # Clear frame cache before starting
    try:
        from MergeStudio.api.routes_preview import _get_cache_dir
        cache_dir = _get_cache_dir()
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_file(): f.unlink()
                except Exception:
                    pass
        print("[Export] [OK] 预览缓存已清理")
    except Exception:
        pass

    import re
    _safe_stem = re.sub(r'[^a-zA-Z0-9_.-]', '_', Path(video_path).stem)
    work_dir = Path(output_path).parent / f".export_{_safe_stem}"
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "export.db"

    if _is_dfl:
        frames_dir = _proj_dir  # frames are already in the project dir
        merged_dir = _proj_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = work_dir / "frames"
        merged_dir = None

    def progress(stage, pct, msg=""):
        if stop_event and stop_event.is_set():
            raise StopRequested("Export cancelled")
        if progress_callback:
            progress_callback(stage, STAGE_NAMES[stage], pct, msg)

    try:
        # Stage 1: Frame extraction (skip in DFL mode — frames already in project dir)
        if _is_dfl:
            progress(0, 1.0, "DFL project — frames already extracted")
        else:
            progress(0, 0.0, "Extracting frames...")
            _hw = hwaccel if hwaccel else _hwaccel_for_encoder(encoder)
            _stage1_extract_frames(video_path, frames_dir, image_format, progress, stop_event=stop_event, hwaccel=_hw)
            progress(0, 1.0, "Frames extracted")

        # Keep only frames within cut segments (inclusion logic)
        _cut_count = 0
        _valid_segs = [s for s in (cut_segments or []) if s.get('start') is not None and s.get('end') is not None]
        print(f"[Export] raw cut_segments from frontend: {cut_segments}", flush=True)
        print(f"[Export] valid keep segments: {_valid_segs}", flush=True)
        if _valid_segs:
            _all_frames = sorted(frames_dir.glob("*.[jp][pn]g"))
            print(f"[Export] Total extracted frames before cut: {len(_all_frames)}", flush=True)
            for _fp in _all_frames:
                _idx = int(_fp.stem)  # 1-based file index
                _keep = any(_s + 1 <= _idx <= _e + 1 for _s, _e in
                            [(s['start'], s['end']) for s in _valid_segs])
                if not _keep:
                    _fp.unlink(); _cut_count += 1
            print(f"[Export] Removed {_cut_count} frames outside selected segments", flush=True)
            _remaining = sorted(frames_dir.glob("*.[jp][pn]g"))
            if _remaining:
                print(f"[Export] First kept frame file: {_remaining[0].name} (stem={_remaining[0].stem})", flush=True)
                print(f"[Export] Last kept frame file: {_remaining[-1].name} (stem={_remaining[-1].stem})", flush=True)
        else:
            print(f"[Export] No valid segments, keeping ALL frames", flush=True)

        # Stage 2: Face detection → SQLite (skip in DFL mode — use aligned DFLJPG data)
        progress(1, 0.0, "Loading faces..." if _is_dfl else "Detecting faces...")
        _init_db(db_path)
        if _is_dfl:
            _stage2_read_dfl_aligned(_proj_dir / "aligned", frames_dir, db_path, progress, stop_event)
        else:
            _face_margin = (config or {}).get('face_margin', 0.4)
            _stage2_detect_frames_mp(frames_dir, db_path, detector, landmarker, res_scale, _face_margin, progress, stop_event, num_workers, angle_segments=angle_segments)
        progress(1, 1.0, "Faces loaded" if _is_dfl else "Face detection complete")

        # Stage 3: Embedding matching (skip if no references selected)
        if face_model_map:
            progress(2, 0.0, "Matching identities...")
            _dfl_align = str(_proj_dir / "aligned") if _is_dfl else None
            _stage3_match_faces(frames_dir, db_path, face_db or {}, face_model_map or {}, progress, stop_event, aligned_dir=_dfl_align)
        else:
            progress(2, 1.0, "No references — all faces will use default model")
        progress(2, 1.0, "Identity matching complete")

        # Stage 4: Face swap + mask + merge (output to merged/ for DFL)
        progress(3, 0.0, "Swapping faces...")
        if _is_dfl:
            global _phase_merge_out_dir
            _phase_merge_out_dir = str(merged_dir)
        _stage4_swap_faces_mp(frames_dir, db_path, work_dir, config or {}, face_model_map or {}, progress, stop_event, num_workers, cut_segments=cut_segments)
        progress(3, 1.0, "Face swap complete")

        # Stage 5: FFmpeg encode
        if _is_dfl:
            progress(6, 0.0, "Encoding video from merged/...")
            _stage5_encode_video(merged_dir, output_path, video_path, encoder, "png", progress, stop_event=stop_event, cut_segments=cut_segments)
        else:
            progress(6, 0.0, "Encoding video...")
            _stage5_encode_video(frames_dir, output_path, video_path, encoder, image_format, progress, stop_event=stop_event, cut_segments=cut_segments)
        progress(6, 1.0, "Export complete")
        print(f"\n[Export] [OK] 导出完成: {output_path}", flush=True)
        # 等待文件句柄释放后删除临时目录
        import time as _time
        _time.sleep(1)
        for _attempt in range(3):
            if work_dir.exists():
                shutil.rmtree(str(work_dir), ignore_errors=True)
                _time.sleep(0.5)
        if not work_dir.exists():
            print(f"[Export] [OK] 缓存已清理: {work_dir.name}", flush=True)
        else:
            print(f"[Export] [WARN] 缓存未清理 (文件被占用): {work_dir.name}", flush=True)
    except StopRequested:
        if stop_event and stop_event.is_set():
            shutil.rmtree(str(work_dir), ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(str(work_dir), ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
def _hwaccel_for_encoder(encoder):
    """GPU 编码器 → 对应硬解码器（提取帧加速）。软编码器返回 None（软解码）。"""
    if not encoder:
        return None
    enc = encoder.lower()
    if 'nvenc' in enc or 'nv' in enc:
        return 'cuda'
    if 'amf' in enc:
        return 'd3d11va'
    if 'qsv' in enc:
        return 'qsv'
    if 'videotoolbox' in enc or 'vt' in enc:
        return 'videotoolbox'
    return None


# Stage 1: FFmpeg frame extraction
# ---------------------------------------------------------------------------
def _stage1_extract_frames(video_path, out_dir, fmt, progress, stop_event=None, hwaccel=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if fmt == "png" else "jpg"
    out_pattern = str(out_dir / f"%08d.{ext}")
    quality_args = ["-q:v", "2"] if ext == "jpg" else []

    def _run(_hw):
        cmd = [_FFMPEG, "-y"]
        if _hw:
            cmd += ["-hwaccel", _hw]
        cmd += ["-i", video_path, "-pix_fmt", "rgb24", *quality_args, out_pattern]
        print(f"[Export] Stage1: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last_done = 0
        while proc.poll() is None:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try: proc.wait(timeout=5)
                except: proc.kill(); proc.wait()
                raise StopRequested("Export cancelled")
            if out_dir.exists():
                done = len([f for f in out_dir.iterdir() if f.suffix in ('.jpg', '.png')])
                if done != last_done:
                    last_done = done
                    print(f"[Export] Extracting frames: {done}", flush=True)
            time.sleep(0.5)
        return proc.returncode, last_done

    rc, _last_done = _run(hwaccel)
    if rc != 0 and hwaccel:
        print(f"[Export] GPU 硬解码失败（exit={rc}），自动回退软解码", flush=True)
        rc, _last_done = _run(None)
    if rc != 0:
        raise RuntimeError(f"FFmpeg extract failed (exit={rc})")
    print(f"[Export] Frame extraction complete: {_last_done} frames")
    progress(0, 1.0, "Frames extracted")


# ---------------------------------------------------------------------------
# Stage 3: Embedding matching — for each face compute embedding, match
#           against face_db, and write model assignment to SQLite.
# ---------------------------------------------------------------------------
# Stage 3 multiprocessing worker
# ---------------------------------------------------------------------------
def _worker_stage3(args):
    """Process a chunk of face rows: compute embedding, match, update DB."""
    chunk, frames_dir_str, db_path_str, all_emb_serial, clusters_serial, face_model_map = args[:6]
    aligned_dir = args[6] if len(args) > 6 else None
    from MergeStudio.core.face_embedder import compute_embedding
    from MergeStudio.core.face_cluster import match_face_to_cluster
    import sqlite3, json, cv2, numpy as np
    from pathlib import Path

    frames_dir = Path(frames_dir_str)
    # Deserialize embeddings
    all_embeddings = {}
    for k, vlist in all_emb_serial.items():
        all_embeddings[k] = np.array(vlist, dtype=np.float32)
    # Deserialize clusters (list of member keys per cluster)
    clusters = {}
    for main_id, members in clusters_serial.items():
        clusters[main_id] = members

    conn = sqlite3.connect(db_path_str, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    done = 0
    _results = []

    for frame_idx, face_idx, face_rect_json in chunk:
        # For DFL mode: load aligned DFLIMG directly using its metadata
        if aligned_dir:
            from DFLIMG import DFLIMG
            ad = Path(aligned_dir)
            # Build cache: (source_frame_idx, face_idx) → filepath once per worker
            if not hasattr(_worker_stage3, "_dfl_map"):
                _worker_stage3._dfl_map = {}
                for _f in ad.glob("*.jpg"):
                    _dfl = DFLIMG.load(str(_f))
                    if _dfl is not None and _dfl.has_data():
                        _src = _dfl.get_source_filename()
                        if _src:
                            _sf = int(Path(_src).stem)
                            _fs = _f.stem
                            _fi = 0
                            if '_' in _fs:
                                try: _fi = int(_fs.rsplit('_', 1)[1])
                                except: pass
                            _worker_stage3._dfl_map[(_sf, _fi)] = _f
            _dfl_path = _worker_stage3._dfl_map.get((frame_idx, face_idx))
            if _dfl_path is None:
                done += 1; continue
            dfl = DFLIMG.load(str(_dfl_path))
            if dfl is not None and dfl.has_data():
                face_img = dfl.get_img()
                if face_img is not None:
                    face_crop = face_img
                else:
                    done += 1; continue
            else:
                done += 1; continue
        else:
            # Try multiple padding lengths
            mpads = [f"{frame_idx:08d}", f"{frame_idx:07d}", f"{frame_idx:05d}", f"{frame_idx:04d}", str(frame_idx)]
            matches = []
            for _p in mpads:
                matches = sorted(frames_dir.glob(f"{_p}.*"))
                if matches: break
            if not matches:
                done += 1; continue
            frame = cv2.imread(str(matches[0]))
            if frame is None:
                done += 1; continue
            face_rect = json.loads(face_rect_json)
            if len(face_rect) >= 4:
                x1, y1, x2, y2 = face_rect[:4]
                margin = int((x2 - x1) * 0.15)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)
                face_crop = frame[y1:y2, x1:x2]
            else:
                done += 1; continue
        if face_crop.size == 0:
            done += 1; continue

        model_name = ""
        matched_id = None

        if all_embeddings:
            try:
                emb = compute_embedding(face_crop)
            except Exception:
                emb = None
            if emb is not None:
                matched_id = match_face_to_cluster(emb, clusters, all_embeddings, threshold=0.5)
                if matched_id and matched_id in face_model_map:
                    model_name = face_model_map[matched_id]
                elif matched_id:
                    for fk, fm in face_model_map.items():
                        if fk == matched_id or fk.split('_')[-1] == matched_id.split('_')[-1]:
                            model_name = fm
                            break

        if not model_name:
            for off in (0, -1):
                ck = f"{frame_idx + off}_{face_idx}"
                if ck in face_model_map:
                    model_name = face_model_map[ck]
                    matched_id = ck
                    break
            if not model_name and str(face_idx) in face_model_map:
                model_name = face_model_map[str(face_idx)]
                matched_id = str(face_idx)

        _results.append((frame_idx, face_idx, model_name, matched_id or ""))
        done += 1

    conn.close()
    return _results


# ---------------------------------------------------------------------------
def _stage3_match_faces(frames_dir, db_path, face_db, face_model_map, progress, stop_event=None, aligned_dir=None):
    from MergeStudio.core.face_embedder import compute_embedding
    from MergeStudio.core.face_cluster import match_face_to_cluster
    import cv2

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")

    # DEBUG: log face_model_map status
    print(f"[Export] Stage3: face_model_map has {len(face_model_map)} entries", flush=True)
    if face_model_map:
        sample = list(face_model_map.items())[:5]
        print(f"[Export] Stage3: sample keys: {sample}", flush=True)

    # ---- Step 1: build reference embeddings from face_model_map ----
    # Only faces the user explicitly assigned to a model.
    all_embeddings = {}
    # Pre-computed embeddings from face_db
    for fid, info in face_db.items():
        emb = None
        if isinstance(info, dict):
            emb = info.get("embedding")
        elif isinstance(info, (list, np.ndarray)):
            emb = info
        if emb is not None and fid in face_model_map:
            all_embeddings[fid] = np.array(emb, dtype=np.float32)
    # Build a flexible frame index→file lookup (handles any padding length)
    _all_frames = sorted(frames_dir.glob("*.[jp][pn]g"))
    _frame_map = {}
    for _f in _all_frames:
        try:
            _frame_map[int(_f.stem)] = _f
        except ValueError:
            pass
    # Auto-compute embeddings for face_model_map entries that lack them
    auto_count = 0
    for fk in face_model_map:
        if fk in all_embeddings:
            continue
        parts = fk.split('_')
        if len(parts) < 2:
            continue
        ref_idx = int(parts[0])
        ref_face = int(parts[1])
        # Try 0-based (DFL native) first, then 1-based (FFmpeg export)
        ref_frame = ref_idx if ref_idx in _frame_map else ref_idx + 1
        ref_matches = [_frame_map.get(ref_frame)] if ref_frame in _frame_map else []
        if not ref_matches:
            continue
        ref_img = cv2.imread(str(ref_matches[0]))
        if ref_img is None:
            continue
        cur2 = conn.execute("SELECT face_rect FROM face_data WHERE frame_idx=? AND face_idx=?", (ref_frame, ref_face))
        row2 = cur2.fetchone()
        if not row2:
            continue
        ref_rect = json.loads(row2[0])
        if len(ref_rect) < 4:
            continue
        x1, y1, x2, y2 = ref_rect[:4]
        margin = int((x2 - x1) * 0.15)
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(ref_img.shape[1], x2 + margin); y2 = min(ref_img.shape[0], y2 + margin)
        crop = ref_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            emb = compute_embedding(crop)
            all_embeddings[fk] = emb
            auto_count += 1
        except Exception:
            continue

    # Diagnostic: show what's being used
    print(f"[Export] Stage3: face_db={len(face_db)} entries face_model_map={len(face_model_map)} entries", flush=True)
    for fk, mn in list(face_model_map.items())[:10]:
        has_emb = "emb" if fk in all_embeddings else "no-emb"
        print(f"  ref {fk} → {mn} ({has_emb})", flush=True)

    # DBSCAN clustering
    from MergeStudio.core.face_cluster import cluster_embeddings_dbscan
    clusters = cluster_embeddings_dbscan(all_embeddings, eps=0.3, min_samples=1)
    print(f"[Export] Stage3: {len(all_embeddings)} references → {len(clusters)} clusters", flush=True)

    total_faces = conn.execute("SELECT COUNT(*) FROM face_data").fetchone()[0]
    if total_faces == 0:
        conn.close()
        return

    cur = conn.execute("SELECT frame_idx, face_idx, face_rect FROM face_data ORDER BY frame_idx, face_idx")
    rows = cur.fetchall()
    conn.close()  # main connection no longer needed, workers write directly

    # Split into chunks for multiprocessing
    import multiprocessing as _mp
    num_workers = max(1, min(_mp.cpu_count() // 2, 8))
    chunks = [rows[i::num_workers] for i in range(num_workers)]
    chunks = [c for c in chunks if c]

    # Serialize embeddings and clusters for pickling
    all_emb_serial = {k: v.tolist() for k, v in all_embeddings.items()}
    clusters_serial = {k: v for k, v in clusters.items()}

    print(f"[Export] Stage3 MP: {total_faces} faces, {len(chunks)} workers", flush=True)
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        tasks = [(c, str(frames_dir), str(db_path), all_emb_serial, clusters_serial, face_model_map, aligned_dir) for c in chunks]
        t0 = time.time()
        _all_results = []
        done = 0
        for _chunk_result in pool.imap_unordered(_worker_stage3, tasks):
            if stop_event and stop_event.is_set():
                pool.terminate()
                raise StopRequested("Export cancelled")
            if _chunk_result:
                _all_results.extend(_chunk_result)
                done += len(_chunk_result)
            elapsed = time.time() - t0
            progress(2, min(1.0, done / total_faces), f"{done}/{total_faces} · {elapsed:.0f}s")
        pool.close()
        pool.join()
        if _all_results:
            _wconn = sqlite3.connect(str(db_path))
            for _fi, _fni, _mid, _md2 in _all_results:
                _wconn.execute("UPDATE face_data SET model=?, face_id=? WHERE frame_idx=? AND face_idx=?", (_mid, _md2, _fi, _fni))
            _wconn.commit()
            _wconn.close()
        elapsed = time.time() - t0
        print(f"[Export] Stage3 done: {done}/{total_faces} faces in {elapsed:.1f}s", flush=True)

    if total_faces > 0:
        progress(2, 1.0)


def _stage2_read_dfl_aligned(aligned_dir, frames_dir, db_path, progress, stop_event):
    """Read DFL aligned JPG files and populate face_data in SQLite.
    Skips face detection — landmarks/rect/mat come from embedded DFLJPG metadata.
    """
    import cv2, json, numpy as np
    from pathlib import Path
    from DFLIMG import DFLIMG

    aligned = Path(aligned_dir)
    if not aligned.is_dir():
        raise FileNotFoundError(f"DFL aligned dir not found: {aligned_dir}")

    jpgs = sorted(aligned.glob("*.jpg"))
    if not jpgs:
        print(f"[Export] No aligned JPGs found in {aligned_dir}", flush=True)
        return

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    _init_db(db_path)

    done = 0
    total = len(jpgs)
    print(f"[Export] DFL aligned: {total} files", flush=True)
    for jpg in jpgs:
        if stop_event and stop_event.is_set():
            raise StopRequested("Export cancelled")
        dfl = DFLIMG.load(str(jpg))
        if dfl is None or not dfl.has_data():
            continue
        src_name = dfl.get_source_filename()
        if not src_name:
            # Fallback: derive from aligned filename "00000_0.jpg" → "00000.jpg"
            src_name = jpg.name
            stem = Path(src_name).stem
            if '_' in stem:
                stem = stem.rsplit('_', 1)[0]
            src_name = stem + jpg.suffix
        src_stem = Path(src_name).stem
        try:
            frame_idx = int(src_stem)
        except ValueError:
            continue
        # Determine face index: filename like "00003_0.jpg" or "00003.jpg"
        fname_stem = jpg.stem
        face_idx = 0
        if '_' in fname_stem:
            try:
                face_idx = int(fname_stem.rsplit('_', 1)[1])
            except ValueError:
                face_idx = 0
        landmarks = dfl.get_source_landmarks()
        if landmarks is None or len(landmarks) < 4:
            landmarks = dfl.get_landmarks()
            if landmarks is None or len(landmarks) < 4:
                continue
        face_rect = dfl.get_source_rect()
        if face_rect is None:
            # Compute from landmarks
            x1, y1 = landmarks.min(axis=0).tolist()
            x2, y2 = landmarks.max(axis=0).tolist()
            face_rect = (int(x1), int(y1), int(x2), int(y2))
        mat = dfl.get_image_to_face_mat()
        if mat is None:
            # Compute from landmarks using LandmarksProcessor
            from facelib.LandmarksProcessor import get_transform_mat
            from facelib import FaceType
            mat = get_transform_mat(landmarks, 256, FaceType.WHOLE_FACE)
        # Check if the source frame exists
        frame_path = _find_frame(frames_dir, frame_idx)
        if frame_path is None:
            continue
        _db_insert_face(conn, frame_idx, face_idx,
                        list(face_rect) if isinstance(face_rect, (list, tuple)) else face_rect.tolist(),
                        landmarks.tolist(), mat.tolist(), 256)
        done += 1
        if done % 100 == 0:
            conn.commit()
            progress(1, done / total, f"{done}/{total}")
    conn.commit()
    conn.close()
    print(f"[Export] DFL aligned done: {done}/{total} faces", flush=True)
    progress(1, 1.0, f"{done} faces loaded from DFL aligned")


def _stage2_assign_models_dfl(db_path, face_model_map, progress):
    """Direct model assignment for DFL projects — no ArcFace matching needed."""
    import sqlite3
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.execute("SELECT frame_idx, face_idx FROM face_data ORDER BY frame_idx, face_idx")
    rows = cur.fetchall()
    assigned = 0
    for frame_idx, face_idx in rows:
        model_name = None
        for off in (0, -1, 1):
            ck = f"{frame_idx + off}_{face_idx}"
            if ck in face_model_map:
                model_name = face_model_map[ck]
                break
        if model_name:
            conn.execute("UPDATE face_data SET model=? WHERE frame_idx=? AND face_idx=?",
                         (model_name, frame_idx, face_idx))
            assigned += 1
    conn.commit()
    conn.close()
    print(f"[Export] DFL model assign: {assigned}/{len(rows)} faces", flush=True)
    progress(2, 1.0, f"{assigned} faces assigned")



# ---------------------------------------------------------------------------
# Stage 5: FFmpeg encode
# ---------------------------------------------------------------------------
def _stage5_encode_video(frames_dir, output_path, source_video, encoder, fmt, progress, stop_event=None, cut_segments=None):
    ext = "png" if fmt == "png" else "jpg"

    # Detect frame padding width from actual files
    _pad = 8
    _files = sorted(frames_dir.glob(f"*.{ext}"))
    if _files:
        _stem = _files[0].stem
        if _stem.isdigit():
            _pad = len(_stem)

    probe = subprocess.run(
        [_FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,bit_rate", "-of", "csv=p=0", source_video],
        capture_output=True, text=True
    )
    parts = probe.stdout.strip().split(',')
    fps_str = parts[0].split('/')
    fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 else 30.0
    src_bitrate = parts[1].strip() if len(parts) > 1 and parts[1].strip() else ""

    # Re-validate segments (filter out None start/end)
    _valid_segs = [s for s in (cut_segments or []) if s.get('start') is not None and s.get('end') is not None]

    # Renumber remaining frames sequentially for gap-free encoding
    _frames = sorted(frames_dir.glob(f"*.{ext}"))
    if not _frames:
        print(f"[Export] Stage5: no frames found in {frames_dir}", flush=True)
        return
    if _valid_segs:
        for _i, _f in enumerate(_frames):
            _new = frames_dir / f"{_i + 1:08d}{_f.suffix}"
            if _new != _f:
                _f.rename(_new)
        _pad = 8
        print(f"[Export] Renumbered {len(_frames)} frames for encoding", flush=True)

    input_pattern = str(frames_dir / f"%0{_pad}d.{ext}")
    # Build audio trim filter for cut segments
    if _valid_segs:
        _aselect_parts = []
        for _seg in _valid_segs:
            _s = _seg.get('start', 0)
            _e = _seg.get('end', 0)
            _t0 = _s / fps
            _t1 = (_e + 1) / fps
            _aselect_parts.append(f"between(t,{_t0},{_t1})")
        _af = "aselect='" + "+".join(_aselect_parts) + "',asetpts=N/SR/TB"
        cmd = [_FFMPEG, "-y", "-r", str(fps), "-i", input_pattern,
               "-i", source_video,
               "-filter_complex", f"[1:a]{_af}[a]",
               "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd = [_FFMPEG, "-y", "-r", str(fps), "-i", input_pattern,
               "-i", source_video, "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", encoder, "-pix_fmt", "yuv420p", "-shortest"]
    print(f"[Export] FFmpeg encode cmd: {' '.join(cmd[:8])} ...", flush=True)
    if src_bitrate:
        cmd += ["-b:v", src_bitrate]
        print(f"[Export] Stage5: using source bitrate {src_bitrate}")
    cmd.append(output_path)
    print(f"[Export] Stage5: {' '.join(cmd)}", flush=True)

    def _run_ffmpeg(cmd, label="encode"):
        total_f = len(_frames) if _frames else None
        proc = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while proc.poll() is None:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try: proc.wait(timeout=5)
                except: proc.kill(); proc.wait()
                raise StopRequested("Export cancelled")
            # Read a line from progress pipe
            line = proc.stdout.readline()
            if line:
                try:
                    line_s = line.decode(errors='replace').strip()
                    if line_s.startswith('frame='):
                        parts = line_s.split()
                        for p in parts:
                            if p.startswith('frame='):
                                f = int(p.split('=')[1])
                                if total_f and total_f > 0:
                                    progress(6, min(1.0, f / total_f),
                                             f"编码 {f}/{total_f} {' '.join(parts[:3])}")
                except Exception:
                    pass
            else:
                time.sleep(0.2)
        sout, serr = proc.communicate()
        return proc.returncode, sout, serr

    # Try with audio first
    rc, sout, serr = _run_ffmpeg(cmd)
    if rc != 0:
        err_text = serr.decode(errors='replace') if isinstance(serr, bytes) else serr
        # Retry without audio (first audio attempt may fail due to codec)
        print(f"[Export] Stage5 FFmpeg audio failed, retrying without audio", flush=True)
        cmd2 = [_FFMPEG, "-y", "-r", str(fps), "-i", input_pattern,
                "-c:v", encoder, "-pix_fmt", "yuv420p", output_path]
        rc2, _, serr2 = _run_ffmpeg(cmd2, label="retry")
        if rc2 != 0:
            err2 = serr2.decode(errors='replace') if isinstance(serr2, bytes) else serr2
            print(f"[Export] Stage5 retry failed: {err2[:500]}", flush=True)
            raise RuntimeError(f"FFmpeg encode failed: {err2[:200]}")
        print(f"[Export] Stage5 done (no audio): {output_path}", flush=True)
    else:
        print(f"[Export] Stage5 done: {output_path}", flush=True)
    progress(6, 1.0)
