import math
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from collections import OrderedDict
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from MergeStudio.api.schemas import AnalyzeRequest, ReCompositeRequest
from MergeStudio.core.merger import MergeMaskedFace
from MergeStudio.core.model_loader import model_loader
from MergeStudio.core.config import MergerConfigMasked
from MergeStudio.core.detector.pipeline import detect_and_align
from MergeStudio.core.detector.factory import DetectorFactory, LandmarkFactory, get_device_info as _get_device_info_det

router = APIRouter()

_onnx_cache = {}
_current_video = None
_current_model_path = None
_current_predictor = None
_predictors = {}  # model_name -> predictor function (for multi-model)
_predictor_sizes = {}  # model_name -> input_size
_cache_dir = None

# ===== Frame cache for 4K playback =====
# Three-tier: memory LRU → disk cache → live decode + sequential predecode
_frame_cache = OrderedDict()
FRAME_CACHE_MAX = 300
FRAME_CACHE_QUALITY = 85
_total_frames_for_cache = 0
_cache_dir_frames = None
_predecode_task = None  # track if predecode is already running
_last_detection = {"frame_idx": -1, "face_list": [], "faces_json": [], "detection_img": None}  # cache for remerge
_prediction_cache = {}  # frame_idx -> {orig_fi: (prd_face, prd_mask, dst_mask)} per face
_merge_gen = 0  # incremented on remerge to invalidate stale background workers

# Cached detector/landmarker instances (persist across frame switches)
_cached_detector = None
_cached_detector_name = None
_cached_landmarker = None
_cached_landmarker_name = None


def _set_total_frames(n: int):
    global _total_frames_for_cache
    _total_frames_for_cache = n


def _get_frames_cache_dir():
    """Get or create the persistent disk cache directory for frame JPEGs."""
    global _cache_dir_frames
    if _cache_dir_frames is None:
        _cache_dir_frames = _get_cache_dir() / "frames"
        _cache_dir_frames.mkdir(parents=True, exist_ok=True)
    return _cache_dir_frames


def _disk_cache_path(idx: int) -> Path:
    return _get_frames_cache_dir() / f"{idx}.jpg"


def _predecode_worker(start: int, count: int = 80):
    """Background sequential predecode: fills both disk cache and memory LRU.
    Runs in a thread executor so it doesn't block the API."""
    global _predecode_task
    try:
        if _current_video is None:
            return
        end = min(start + count, _total_frames_for_cache)
        if end <= start:
            return

        disk_dir = _get_frames_cache_dir()
        cap = cv2.VideoCapture(_current_video)
        if not cap.isOpened():
            return

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end):
            # Check if already cached on disk (skip if so)
            disk_path = disk_dir / f"{i}.jpg"
            if disk_path.exists():
                # Skip to next frame without decoding
                continue

            ret, frame = cap.read()
            if not ret:
                break

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CACHE_QUALITY])
            jpeg_bytes = jpeg.tobytes()

            # Write to disk cache
            try:
                disk_path.write_bytes(jpeg_bytes)
            except Exception:
                pass

            # Fill memory LRU
            _frame_cache[i] = jpeg_bytes
            while len(_frame_cache) > FRAME_CACHE_MAX:
                _frame_cache.popitem(last=False)

        cap.release()
    finally:
        _predecode_task = None


def _trigger_predecode(start: int):
    """Fire off a background thread to sequentially predecode frames."""
    global _predecode_task
    if _predecode_task is not None:
        return  # already running
    import threading
    t = threading.Thread(target=_predecode_worker, args=(start, 80), daemon=True)
    _predecode_task = t
    t.start()


def _get_cached_frame(idx: int) -> bytes:
    """Three-tier frame access: memory LRU → disk cache → live decode + predecode."""
    # Tier 1: memory LRU
    if idx in _frame_cache:
        _frame_cache.move_to_end(idx)
        # Trigger background predecode from this position (non-blocking)
        _trigger_predecode(idx + 1)
        return _frame_cache[idx]

    if _current_video is None:
        raise HTTPException(400, "No video loaded")

    # Tier 2: disk cache
    disk_path = _disk_cache_path(idx)
    if disk_path.exists():
        jpeg_bytes = disk_path.read_bytes()
        _frame_cache[idx] = jpeg_bytes
        while len(_frame_cache) > FRAME_CACHE_MAX:
            _frame_cache.popitem(last=False)
        _trigger_predecode(idx + 1)
        return jpeg_bytes

    # Tier 3: live decode
    cap = cv2.VideoCapture(_current_video)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise HTTPException(404, "Frame " + str(idx) + " not found")

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CACHE_QUALITY])
    jpeg_bytes = jpeg.tobytes()

    # Cache to memory
    _frame_cache[idx] = jpeg_bytes
    while len(_frame_cache) > FRAME_CACHE_MAX:
        _frame_cache.popitem(last=False)

    # Cache to disk
    try:
        disk_path.write_bytes(jpeg_bytes)
    except Exception:
        pass

    # Keep cap open for sequential predecode of next 30 frames
    try:
        for n in range(1, 31):
            nidx = idx + n
            if nidx >= _total_frames_for_cache:
                break
            ndisk = _disk_cache_path(nidx)
            if ndisk.exists() or nidx in _frame_cache:
                continue
            ret_n, frame_n = cap.read()
            if not ret_n:
                break
            _, jpeg_n = cv2.imencode('.jpg', frame_n, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CACHE_QUALITY])
            jb = jpeg_n.tobytes()
            _frame_cache[nidx] = jb
            while len(_frame_cache) > FRAME_CACHE_MAX:
                _frame_cache.popitem(last=False)
            try:
                ndisk.write_bytes(jb)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        cap.release()

    # Fire background predecode for further frames
    _trigger_predecode(idx + 31)

    return jpeg_bytes


@router.get("/preview/cache-status")
async def cache_status():
    """Return cache fill status (for progress display)."""
    total = _total_frames_for_cache
    if total <= 0:
        return {"cached": 0, "total": 0, "pct": 0}
    disk_dir = _get_frames_cache_dir()
    cached = len(list(disk_dir.glob("*.jpg"))) if disk_dir.exists() else 0
    return {"cached": cached, "total": total, "pct": round(cached / total * 100, 1)}


class SelectVideoRequest(BaseModel):
    path: str


def _get_cache_dir():
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = Path(__file__).parent.parent / "workspace" / "preview_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
        # Clear old cache contents on startup without deleting the dir itself
        for old_file in _cache_dir.iterdir():
            try:
                if old_file.is_file():
                    old_file.unlink()
            except Exception:
                pass
    return _cache_dir


def _dfl_aligned_path(aligned_dir, idx: int, face_idx: int = 0) -> str:
    """Find aligned face file for given frame, trying all zero-pad lengths."""
    d = Path(aligned_dir)
    if not d.exists():
        return ""
    for pad in range(1, 9):
        fname = f"{idx:0{pad}d}_{face_idx}.jpg"
        fp = d / fname
        if fp.exists():
            return str(fp)
    # Fallback: scan directory for any file starting with idx (any padding)
    for fp in d.glob("*.jpg"):
        stem = fp.stem
        parts = stem.split("_")
        if parts and (parts[0].lstrip('0') == str(idx) or (parts[0].lstrip('0') == '' and idx == 0)) and (len(parts) < 2 or parts[1] == str(face_idx)):
            return str(fp)
    return ""


def _check_dfl(video_path: str) -> dict:
    """Check if video has a standard DFL project structure (video + same-name dir + aligned/)."""
    p = Path(video_path)
    video_stem = p.stem
    aligned_dir = p.parent / video_stem / "aligned"
    count = 0
    if aligned_dir.exists():
        count = len(list(aligned_dir.glob("*.jpg")))
    return {"is_dfl": count > 0, "aligned_dir": str(aligned_dir) if aligned_dir.exists() else "", "aligned_count": count}


def _resolve_video_path(path_param: str = None) -> str:
    """Use explicit path param if provided, else fall back to _current_video."""
    if path_param:
        return path_param
    if _current_video is None:
        raise HTTPException(400, "No video loaded")
    return _current_video


def _read_frame(video_path: str, idx: int, quality: int = 85):
    """Open video, read frame, encode JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video: " + video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(404, "Frame " + str(idx) + " not found")
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


@router.get("/preview/frame/{idx}")
async def get_frame(idx: int, q: int = Query(85, alias="q", ge=10, le=100),
                    path: str = Query(None, alias="path")):
    video_path = _resolve_video_path(path)

    # Quick path: cached at default quality (85) — only for _current_video path
    if q >= FRAME_CACHE_QUALITY - 5 and not path:
        try:
            return Response(content=_get_cached_frame(idx), media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        except HTTPException:
            raise

    jpeg_bytes = _read_frame(video_path, idx, q)
    return Response(content=jpeg_bytes, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(idx) + " not found")

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@router.get("/preview/video-stream")
async def video_stream():
    """Serve the source video file directly for native browser playback.
    Browser's <video> element uses GPU decoding (DXVA/VAAPI/VideoToolbox)
    through the platform's native media framework."""
    if _current_video is None:
        raise HTTPException(400, "No video loaded")
    video_path = Path(_current_video)
    if not video_path.exists():
        raise HTTPException(404, "Video file not found")
    return FileResponse(str(video_path), media_type="video/mp4")


@router.post("/preview/analyze")
async def analyze_frame(req: AnalyzeRequest):
    if _current_video is None:
        raise HTTPException(400, "No video loaded")

    cap = cv2.VideoCapture(_current_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, req.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(req.frame_idx) + " not found")

    cfg = _config_from_dict(req.config)
    cache_dir = _get_cache_dir()
    idx = req.frame_idx

    # Resolution scaling for detection
    res_scale = max(0.125, min(1.0, getattr(req, 'res_scale', 0.5)))
    det_frame = frame
    h_orig, w_orig = frame.shape[:2]
    if res_scale < 1.0:
        det_w = int(w_orig * res_scale)
        det_h = int(h_orig * res_scale)
        det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)

    # Detect faces (skip for DFL standard projects if detect_mode=skip_dfl)
    face_list = []
    detect_mode = req.config.get('detect_mode', 'always')
    if detect_mode == 'skip_dfl' and _current_video:
        dfl_info = _check_dfl(_current_video)
        if dfl_info["is_dfl"]:
            aligned_dir = Path(dfl_info["aligned_dir"])
            for fi in range(10):
                aligned_path = _dfl_aligned_path(aligned_dir, idx, fi)
                if not aligned_path:
                    break
                aligned_img = cv2.imread(aligned_path)
                if aligned_img is None:
                    break
                # Read DFL metadata from APP15 marker in the JPEG
                import pickle, struct
                try:
                    with open(aligned_path, 'rb') as _fh:
                        _raw = _fh.read()
                    _app15 = _raw.find(b'\xff\xef')
                    if _app15 < 0:
                        break
                    _sz = struct.unpack('>H', _raw[_app15+2:_app15+4])[0]
                    _meta = pickle.loads(_raw[_app15+4:_app15+4+_sz-2])
                    _src_rect = _meta.get('source_rect')
                    _src_lm = _meta.get('source_landmarks')
                    _face_mat = _meta.get('image_to_face_mat')
                    _lm = _meta.get('landmarks')
                except Exception as _e:
                    break
                if _src_rect is None or _src_lm is None:
                    break
                _x, _y, _r, _b = _src_rect[:4]
                # Aligned face dimensions from the JPEG itself
                _aligned_h, _aligned_w = aligned_img.shape[:2]
                face_list.append({
                    "face_rect": (int(_x), int(_y), int(_r), int(_b)),
                    "landmarks": _src_lm.astype(np.float32) if hasattr(_src_lm, 'astype') else np.array(_src_lm, dtype=np.float32),
                    "transform_mat": _face_mat.astype(np.float64) if hasattr(_face_mat, 'astype') else np.array(_face_mat, dtype=np.float64),
                    "out_size": max(_aligned_w, _aligned_h),
                    "_dfl": True,
                })
    if not face_list:
        try:
            global _cached_detector, _cached_detector_name, _cached_landmarker, _cached_landmarker_name
            device = _get_device_info_det()
            if (_cached_detector is None or _cached_detector_name != req.detector):
                _cached_detector = DetectorFactory.create(req.detector, device)
                _cached_detector_name = req.detector
            if (_cached_landmarker is None or _cached_landmarker_name != req.landmarker):
                _cached_landmarker = LandmarkFactory.create(req.landmarker, device)
                _cached_landmarker_name = req.landmarker
            face_margin = req.config.get('face_margin', req.face_margin)
            # Determine detection angles for current frame from angle_segments
            _da = [0]  # default
            if req.angle_segments:
                print(f"[Preview] frame {idx} angle_segments={req.angle_segments}", flush=True)
                for _seg in req.angle_segments:
                    _s = _seg.get('start')
                    _e = _seg.get('end')
                    if _s is not None and _e is not None and _s <= idx <= _e:
                        try:
                            _da = [int(x) for x in _seg.get('angles', '0').split(',') if x.strip()]
                            print(f"[Preview] frame {idx} matched seg start={_s} end={_e} angles={_da}", flush=True)
                        except Exception as _ae:
                            print(f"[Preview] angle parse error: {_ae}", flush=True)
                        break
            face_list = detect_and_align(_cached_detector, _cached_landmarker, det_frame,
                                         cfg.face_type, margin=face_margin,
                                         detection_angles=_da)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Detection error:", e)

    # Scale face coords back to original resolution (skip for DFL — already in frame coords)
    if res_scale < 1.0 and face_list:
        s = 1.0 / res_scale
        for face in face_list:
            if face.get('_dfl'):
                continue
            face['face_rect'] = tuple(int(v * s) for v in face['face_rect'])
            if 'crop_rect' in face:
                face['crop_rect'] = tuple(int(v * s) for v in face['crop_rect'])
            if 'landmarks' in face and face['landmarks'] is not None:
                face['landmarks'] = face['landmarks'] * s

    # Draw detection overlay with landmarks and alignment rect
    detection_img = frame.copy()
    faces_json = []
    for face in face_list:
        x, y, r, b = face['face_rect']
        landmarks = face.get('landmarks', [])
        lm_list = landmarks.tolist() if isinstance(landmarks, np.ndarray) else landmarks

        # Expanded square crop rect (landmarker input region)
        crop = face.get('crop_rect')
        if crop and not face.get('_dfl'):
            cx, cy, cr, cb = crop
            cv2.rectangle(detection_img, (cx, cy), (cr, cb), (91, 91, 214), 2)
            # Face direction arrow (top-left corner, points in face-up direction)
            if landmarks is not None and len(landmarks) >= 68:
                eye_c = np.mean(landmarks[36:48], axis=0)
                mouth_c = np.mean(landmarks[48:68], axis=0)
                up_v = eye_c - mouth_c
                up_angle = math.degrees(math.atan2(up_v[0], -up_v[1]))
                arr_len = min(cr - cx, cb - cy) // 6
                ax = int(cx + 6 + arr_len * math.sin(math.radians(up_angle)))
                ay = int(cy + 6 - arr_len * math.cos(math.radians(up_angle)))
                cv2.arrowedLine(detection_img, (cx + 6, cy + 6), (ax, ay), (91, 91, 214), 2, tipLength=0.35)
        elif not face.get('_dfl'):
            cv2.rectangle(detection_img, (x, y), (r, b), (91, 91, 214), 2)

        # Draw landmarks (68 points)
        for lx, ly in lm_list:
            cv2.circle(detection_img, (int(lx), int(ly)), 4, (75, 45, 175), -1)

        # Draw alignment rectangle (warp target area)
        mat = face.get('transform_mat')
        if mat is not None:
            out_size = face.get('out_size', 256)
            corners = np.float32([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]])
            inv_mat = cv2.invertAffineTransform(mat)
            orig_corners = cv2.transform(corners.reshape(1, -1, 2), inv_mat).reshape(-1, 2)
            # Scale corners back to original resolution if detection was scaled (skip DFL)
            if res_scale < 1.0 and not face.get('_dfl'):
                orig_corners = orig_corners * (1.0 / res_scale)
            pts = orig_corners.astype(np.int32)
            cv2.polylines(detection_img, [pts], True, (160, 80, 220), 2)  # purple
            # Alignment up arrow (at top-left corner, shows aligned face up direction)
            up_dir = (-inv_mat[0, 1], -inv_mat[1, 1])
            up_len = math.hypot(up_dir[0], up_dir[1])
            if up_len > 1e-6:
                arr_len = min(
                    math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]),
                    math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
                ) // 4
                ux = up_dir[0] / up_len
                uy = up_dir[1] / up_len
                tip_x = int(pts[0][0] + ux * arr_len)
                tip_y = int(pts[0][1] + uy * arr_len)
                cv2.arrowedLine(detection_img, (int(pts[0][0]), int(pts[0][1])), (tip_x, tip_y), (160, 80, 220), 2, tipLength=0.35)

        # Save face thumbnail
        if all(v > 0 for v in (x, y, r - x, b - y)):
            thumb_path = cache_dir / ("face_" + str(idx) + "_" + str(len(faces_json)) + ".jpg")
            if face.get('_dfl') and detect_mode == 'skip_dfl' and _current_video:
                # DFL: load aligned image directly
                dfl_info = _check_dfl(_current_video)
                if dfl_info["is_dfl"]:
                    aligned_path = _dfl_aligned_path(dfl_info["aligned_dir"], idx, len(faces_json))
                    if aligned_path and Path(aligned_path).exists():
                        aligned = cv2.imread(aligned_path)
                        if aligned is not None:
                            aligned = cv2.resize(aligned, (128, 128), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(str(thumb_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # Fallback: affine-transform from frame
            if not Path(thumb_path).exists():
                landmarks = face.get('landmarks')
                if landmarks is not None:
                    from facelib import FaceType, LandmarksProcessor as _lp
                    face_mat = _lp.get_transform_mat(landmarks, 128, face_type=FaceType.WHOLE_FACE)
                    aligned = cv2.warpAffine(frame, face_mat, (128, 128), flags=cv2.INTER_CUBIC)
                    cv2.imwrite(str(thumb_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 80])
                else:
                    face_crop = frame[max(0, y):min(frame.shape[0], b), max(0, x):min(frame.shape[1], r)]
                    if face_crop.size > 0:
                        face_crop = cv2.resize(face_crop, (128, 128), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(str(thumb_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])

        faces_json.append({
            'x': x, 'y': y, 'w': r - x, 'h': b - y,
            'landmarks': lm_list,
            'thumb_url': "/api/preview/face-thumb/" + str(idx) + "/" + str(len(faces_json)),
        })

    # Save detection image IMMEDIATELY after detection (before merge)
    _, det_jpeg = cv2.imencode('.jpg', detection_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (cache_dir / ("det_" + str(idx) + ".jpg")).write_bytes(det_jpeg.tobytes())

    # Delete stale swap cache so pollSwap doesn't find old image from previous merge
    stale_swap = cache_dir / ("swap_" + str(idx) + ".jpg")
    if stale_swap.exists():
        stale_swap.unlink()

    # Run merge in thread executor so event loop stays free for detection poll
    import asyncio, functools

    merge_is_debug = (req.config or {}).get('mode') == 'debug' or (req.config or {}).get('show_debug')

    def _merge_worker():
        global _merge_gen
        local_swapped = frame.copy().astype(np.float32) / 255.0
        local_accum = np.zeros(frame.shape[:2], dtype=np.float32)
        local_debug = []
        if _current_predictor is not None and face_list:
            l_seg = req.config.get('seg_mode', 'model')
            l_xseg = get_xseg_extractor(l_seg)
            for l_orig_fi, l_fd in enumerate(face_list):
                if req.selected_faces and l_orig_fi not in req.selected_faces:
                    continue
                l_landmarks = l_fd['landmarks']
                l_pred = _get_predictor_for_face(l_orig_fi, req.face_model_map or {})
                _set_model_input_size(l_pred)
                l_out, l_mask = MergeMaskedFace(frame.copy(), l_landmarks, cfg, l_pred,
                                                xseg_256_extract_func=l_xseg)
                if merge_is_debug:
                    local_debug.append(l_out.astype(np.float32) / 255.0)
                    continue
                l_face_f = l_out.astype(np.float32) / 255.0
                h_f, w_f = local_swapped.shape[:2]
                if l_face_f.shape[:2] != (h_f, w_f):
                    l_face_f = cv2.resize(l_face_f, (w_f, h_f), interpolation=cv2.INTER_CUBIC)
                l_mask_f = l_mask.astype(np.float32) / 255.0
                if l_mask_f.shape[:2] != (h_f, w_f):
                    l_mask_f = cv2.resize(l_mask_f, (w_f, h_f), interpolation=cv2.INTER_CUBIC)
                l_m3 = np.stack([l_mask_f, l_mask_f, l_mask_f], axis=-1)
                l_avail = 1.0 - local_accum[..., None]
                l_alpha = l_m3 * l_avail
                local_swapped = local_swapped * (1.0 - l_alpha) + l_face_f * l_alpha
                local_accum = np.maximum(local_accum, l_mask_f)
            local_swapped = np.clip(local_swapped * 255, 0, 255).astype(np.uint8)
        else:
            local_swapped = frame.copy()
        # Check if this worker is stale (remergeFrame may have been called)
        if _merge_gen != merge_gen:
            return None if not merge_is_debug else 0
        # Save results
        if merge_is_debug:
            _, local_swap_jpeg = cv2.imencode('.jpg', local_swapped, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (cache_dir / ("swap_" + str(idx) + ".jpg")).write_bytes(local_swap_jpeg.tobytes())
            for dfi, vis in enumerate(local_debug):
                vis_u8 = np.clip(vis * 255, 0, 255).astype(np.uint8)
                _, djpeg = cv2.imencode('.jpg', vis_u8, [cv2.IMWRITE_JPEG_QUALITY, 85])
                (cache_dir / ("debug_" + str(idx) + "_" + str(dfi) + ".jpg")).write_bytes(djpeg.tobytes())
            return len(local_debug)
        else:
            _, swap_jpeg = cv2.imencode('.jpg', local_swapped, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (cache_dir / ("swap_" + str(idx) + ".jpg")).write_bytes(swap_jpeg.tobytes())
            return None

    if _current_predictor is not None and face_list:
        global _merge_gen
        _merge_gen += 1
        merge_gen = _merge_gen
        if merge_is_debug:
            # Debug mode: await merge to return debug URLs in response
            loop = asyncio.get_event_loop()
            debug_count = await loop.run_in_executor(None, _merge_worker)
            debug_urls = [f"/api/preview/debug/{idx}/{fi}" for fi in range(debug_count or 0)]
        else:
            # Normal mode: fire-and-forget, detection response returns immediately
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _merge_worker)
            debug_urls = []
    else:
        # No faces or no predictor — save original frame as swap immediately
        _, swap_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (cache_dir / ("swap_" + str(idx) + ".jpg")).write_bytes(swap_jpeg.tobytes())
        debug_urls = []

    # Cache detection for remerge (skip detection on param changes)
    _last_detection.update({
        "frame_idx": idx,
        "face_list": face_list,
        "faces_json": faces_json,
        "detection_img": detection_img,
    })

    return {
        "face_count": len(face_list),
        "faces": faces_json,
        "detection_url": "/api/preview/detection/" + str(idx),
        "swapped_url": "/api/preview/swapped/" + str(idx) if (_current_predictor is not None and face_list) else None,
        "debug_urls": debug_urls,
    }


@router.post("/preview/remerge")
async def remerge(req: AnalyzeRequest):
    """Re-merge using cached detection + cached model predictions (skip both)."""
    global _last_detection, _prediction_cache, _merge_gen
    # Invalidate any in-flight background merge worker from analyze_frame
    _merge_gen += 1
    if _current_video is None:
        raise HTTPException(400, "No video loaded")
    if _current_predictor is None:
        raise HTTPException(400, "No model loaded")
    if _last_detection["frame_idx"] != req.frame_idx:
        return await analyze_frame(req)
    # Re-detect if angle_segments present (they may have changed since cached detection)
    if req.angle_segments:
        print(f"[Preview] remerge: angle_segments present, forcing re-detect for frame {req.frame_idx}", flush=True)
        return await analyze_frame(req)

    idx = req.frame_idx
    cfg = _config_from_dict(req.config)
    # Clear prediction cache when per-face model mapping active (different models produce different predictions)
    if req.face_model_map:
        _prediction_cache.pop(idx, None)
    cache_dir = _get_cache_dir()
    face_list = _last_detection["face_list"]
    from MergeStudio.core import merger as _merger_mod
    input_size = getattr(_merger_mod, '_model_input_size', 256)

    cap = cv2.VideoCapture(_current_video)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Cannot read video frame " + str(idx))
    # uint8 0-255 for MergeMaskedFace (it normalizes internally)
    frame_bgr_u8 = frame_bgr.copy()
    # float 0-1 for predictor cache extraction + compositing
    img_bgr_float = frame_bgr.astype(np.float32) / 255.0
    h, w = frame_bgr.shape[:2]
    swapped_img = img_bgr_float.copy()
    accum_mask = np.zeros((h, w), dtype=np.float32)
    seg_mode = req.config.get('seg_mode', 'model')
    xseg_func = get_xseg_extractor(seg_mode)

    # Cache for XSeg results (prd call=0, dst call=1 per face)
    _xseg_cache = {}

    def _cached_xseg_wrapper(orig_func, frame_idx, face_idx):
        call_count = [0]

        def wrapped(face_img):
            ci = call_count[0]
            call_count[0] += 1
            key = (frame_idx, face_idx, ci)
            if key in _xseg_cache:
                return _xseg_cache[key]
            result = orig_func(face_img)
            _xseg_cache[key] = result
            return result
        return wrapped

    debug_frames = []
    for orig_fi in range(len(face_list)):
        if req.selected_faces and orig_fi not in req.selected_faces:
            continue
        face_data = face_list[orig_fi]
        try:
            if idx in _prediction_cache and orig_fi in _prediction_cache[idx]:
                prd_face, prd_mask, dst_mask = _prediction_cache[idx][orig_fi]
            else:
                landmarks = face_data['landmarks']
                from facelib import FaceType, LandmarksProcessor
                out_size = input_size
                ft = cfg.face_type
                if isinstance(ft, str):
                    ft_map = {'half_face': FaceType.HALF, 'midfull_face': FaceType.MID_FULL,
                              'full_face': FaceType.FULL, 'whole_face': FaceType.WHOLE_FACE, 'head': FaceType.HEAD}
                    ft = ft_map.get(ft, FaceType.WHOLE_FACE)
                face_mat = LandmarksProcessor.get_transform_mat(landmarks, out_size, face_type=ft)
                dst_face = cv2.warpAffine(img_bgr_float, face_mat, (out_size, out_size), flags=cv2.INTER_CUBIC)
                dst_face = np.clip(dst_face, 0, 1)
                pred_fn = _get_predictor_for_face(orig_fi, req.face_model_map or {})
                pred_result = pred_fn(cv2.resize(dst_face, (input_size, input_size)))
                if isinstance(pred_result, tuple):
                    prd_face, prd_mask, dst_mask = pred_result
                else:
                    prd_face, prd_mask, dst_mask = pred_result, None, None
                if idx not in _prediction_cache:
                    _prediction_cache[idx] = {}
                _prediction_cache[idx][orig_fi] = (prd_face, prd_mask, dst_mask)

            def cached_pred(_x):
                return (prd_face, prd_mask, dst_mask)

            cached_xseg = _cached_xseg_wrapper(xseg_func, idx, orig_fi) if xseg_func else None
            # Switch model input size to match this face's assigned model
            pred_fn_for_size = _get_predictor_for_face(orig_fi, req.face_model_map or {})
            _set_model_input_size(pred_fn_for_size)

            face_out, face_mask = MergeMaskedFace(frame_bgr_u8, face_data['landmarks'], cfg, cached_pred,
                                                  xseg_256_extract_func=cached_xseg)
            if cfg.mode == 'debug' or getattr(cfg, 'show_debug', False):
                debug_frames.append(face_out.astype(np.float32) / 255.0)
                continue
            face_out_f = face_out.astype(np.float32) / 255.0
            if face_out_f.shape[:2] != (h, w):
                face_out_f = cv2.resize(face_out_f, (w, h), interpolation=cv2.INTER_CUBIC)
            mask_f = face_mask.astype(np.float32) / 255.0
            if mask_f.shape[:2] != (h, w):
                mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_CUBIC)
            mask_3 = np.stack([mask_f, mask_f, mask_f], axis=-1)
            avail = 1.0 - accum_mask[..., None]
            alpha = mask_3 * avail
            swapped_img = swapped_img * (1.0 - alpha) + face_out_f * alpha
            accum_mask = np.maximum(accum_mask, mask_f)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[remerge] face {orig_fi} failed: {e}")

    # Debug: save each face vis separately (browser handles layout)
    debug_urls = []
    if debug_frames:
        for dfi, vis in enumerate(debug_frames):
            vis_u8 = np.clip(vis * 255, 0, 255).astype(np.uint8)
            _, djpeg = cv2.imencode('.jpg', vis_u8, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (cache_dir / ("debug_" + str(idx) + "_" + str(dfi) + ".jpg")).write_bytes(djpeg.tobytes())
            debug_urls.append("/api/preview/debug/" + str(idx) + "/" + str(dfi))
    else:
        swapped_img = np.clip(swapped_img * 255, 0, 255).astype(np.uint8)
        _, swap_jpeg = cv2.imencode('.jpg', swapped_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (cache_dir / ("swap_" + str(idx) + ".jpg")).write_bytes(swap_jpeg.tobytes())

    return {
        "face_count": len(face_list),
        "faces": _last_detection["faces_json"],
        "detection_url": "/api/preview/detection/" + str(idx),
        "swapped_url": "/api/preview/swapped/" + str(idx),
        "debug_urls": debug_urls,
    }


@router.get("/preview/debug/{idx}/{fi}")
async def get_debug(idx: int, fi: int):
    cache_dir = _get_cache_dir()
    path = cache_dir / ("debug_" + str(idx) + "_" + str(fi) + ".jpg")
    if path.exists():
        return Response(content=path.read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    raise HTTPException(404, "Debug vis not found")


@router.get("/preview/detection/{idx}")
async def get_detection(idx: int):
    cache_dir = _get_cache_dir()
    path = cache_dir / ("det_" + str(idx) + ".jpg")
    if path.exists():
        return Response(content=path.read_bytes(), media_type="image/jpeg")
    raise HTTPException(404, "No cached detection for frame " + str(idx))


@router.get("/preview/swapped/{idx}")
async def get_swapped(idx: int, scale: float = 1.0):
    cache_dir = _get_cache_dir()
    path = cache_dir / ("swap_" + str(idx) + ".jpg")
    if not path.exists():
        raise HTTPException(404, "No cached swapped for frame " + str(idx))
    data = path.read_bytes()
    if scale < 1.0 and scale > 0:
        import cv2, numpy as np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = int(img.shape[0] * scale), int(img.shape[1] * scale)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ret: data = buf.tobytes()
    return Response(content=data, media_type="image/jpeg")


@router.get("/preview/face-thumb/{idx}/{face_idx}")
async def get_face_thumb(idx: int, face_idx: int):
    cache_dir = _get_cache_dir()
    path = cache_dir / ("face_" + str(idx) + "_" + str(face_idx) + ".jpg")
    if path.exists():
        return Response(content=path.read_bytes(), media_type="image/jpeg")
    raise HTTPException(404, "Face thumb not found")


@router.post("/preview/recomposite")
async def recomposite(req: ReCompositeRequest):
    if _current_video is None:
        raise HTTPException(400, "No video loaded")
    if _current_predictor is None:
        raise HTTPException(400, "No model loaded")

    cap = cv2.VideoCapture(_current_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, req.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(req.frame_idx) + " not found")

    cfg = _config_from_dict(req.config)

    try:
        device = _get_device_info_det()
        detector = DetectorFactory.create('TinyMog', device)
        landmarker = LandmarkFactory.create('insightface-2d106det', device)
        face_list = detect_and_align(detector, landmarker, frame, cfg.face_type)
    except Exception:
        face_list = []

    swapped_img = frame.copy()
    for face_data in face_list:
        landmarks = face_data['landmarks']
        face_out, _ = MergeMaskedFace(frame, landmarks, cfg, _current_predictor)
        swapped_img = face_out

    cache_dir = _get_cache_dir()
    _, swap_jpeg = cv2.imencode('.jpg', swapped_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (cache_dir / ("swap_" + str(req.frame_idx) + ".jpg")).write_bytes(swap_jpeg.tobytes())

    return {"status": "ok", "swapped_url": "/api/preview/swapped/" + str(req.frame_idx)}


def _config_from_dict(d: dict) -> MergerConfigMasked:
    valid_keys = [k for k in MergerConfigMasked.__init__.__code__.co_varnames
                  if k != 'self' and k != 'kwargs']
    kwargs = {k: v for k, v in d.items() if k in valid_keys}
    # Frontend sends color_transfer_mode as string ("rct", "lct", etc.) - convert to int
    if 'color_transfer_mode' in kwargs:
        if isinstance(kwargs['color_transfer_mode'], str):
            from MergeStudio.core.config import ctm_str_dict
            _raw = kwargs['color_transfer_mode']
            # 支持大小写不敏感匹配（前端 "none" → 后端 "None"）
            kwargs['color_transfer_mode'] = ctm_str_dict.get(_raw) or \
                                            ctm_str_dict.get(_raw.lower()) or \
                                            ctm_str_dict.get(_raw.capitalize(), 1)
        else:
            kwargs['color_transfer_mode'] = int(kwargs['color_transfer_mode'])
    # Ensure mask_mode is int
    if 'mask_mode' in kwargs:
        kwargs['mask_mode'] = int(kwargs['mask_mode'])
    return MergerConfigMasked(**kwargs)


@router.post("/select-video")
async def select_video(req: SelectVideoRequest):
    path = req.path
    if not cv2.VideoCapture(path).isOpened():
        raise HTTPException(400, "Cannot open video: " + path)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    set_current_video(path, total)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    dfl_info = _check_dfl(path)
    return {"status": "ok", "total_frames": total, "fps": fps, "is_dfl": dfl_info["is_dfl"], "aligned_count": dfl_info["aligned_count"]}


def set_current_video(video_path: str, total_frames: int = 0):
    global _current_video, _frame_cache, _prediction_cache, _last_detection
    _current_video = video_path
    _frame_cache.clear()
    _prediction_cache.clear()
    _last_detection["frame_idx"] = -1  # invalidate detection cache
    if total_frames > 0:
        _set_total_frames(total_frames)


# Load XSeg models for mask extraction
try:
    _xseg_session = onnxruntime.InferenceSession(
        str(Path(__file__).parent.parent.parent / "workspace" / "model" / "XSeg" / "XSeg.onnx"),
        providers=['CPUExecutionProvider'])
    _xseg_input = _xseg_session.get_inputs()[0].name
    print("[MergeStudio] XSeg model loaded")
except Exception as e:
    _xseg_session = None
    print(f"[MergeStudio] XSeg not available: {e}")

try:
    _xseglite_session = onnxruntime.InferenceSession(
        str(Path(__file__).parent.parent.parent / "workspace" / "model" / "XSegLite" / "xseglite.onnx"),
        providers=['CPUExecutionProvider'])
    _xseglite_input = _xseglite_session.get_inputs()[0].name
    print("[MergeStudio] XSegLite model loaded")
except Exception as e:
    _xseglite_session = None
    print(f"[MergeStudio] XSegLite not available: {e}")


def get_xseg_extractor(seg_mode: str):
    """Create XSeg mask extraction function based on seg_mode."""
    if seg_mode == 'xseg' and _xseg_session is not None:
        sess = _xseg_session
        inp_name = _xseg_input
        _shape = sess.get_inputs()[0].shape
        is_nchw = len(_shape) == 4 and _shape[1] in (1, 3)  # auto-detect NCHW
        size = 256
    elif seg_mode == 'xseglite' and _xseglite_session is not None:
        sess = _xseglite_session
        inp_name = _xseglite_input
        _shape = sess.get_inputs()[0].shape
        is_nchw = len(_shape) == 4 and _shape[1] in (1, 3)
        size = 256
    else:
        return None

    def extractor(face_img_in):
        """Extract XSeg mask from face image, resize to model size, then resize back."""
        orig_h, orig_w = face_img_in.shape[:2]
        # Resize to XSeg model input size
        if orig_h != size or orig_w != size:
            face_resized = cv2.resize(face_img_in, (size, size),
                                      interpolation=cv2.INTER_CUBIC)
        else:
            face_resized = face_img_in
        # Always clip to 0-1 for float input (warp can produce out-of-range values)
        if face_resized.max() < 2.0:
            face_clip = np.clip(face_resized, 0, 1)
        else:
            face_clip = face_resized  # uint8 0-255, don't clip
        # Convert BGR→RGB (XSeg models trained on RGB)
        face_rgb = cv2.cvtColor(face_clip, cv2.COLOR_BGR2RGB)
        # Normalize (input is 0-1 float or 0-255 uint8)
        inp = face_rgb.astype(np.float32)
        if inp.max() > 1.0:
            inp /= 255.0
        if is_nchw:
            inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        else:
            inp = inp[None, :, :, :]
        # Run XSeg
        out = sess.run(None, {inp_name: inp})[0]
        # Squeeze to 2D mask
        out = np.squeeze(out)
        if out.ndim == 3:
            out = out[:, :, 0]
        # Clip to valid 0-1 range
        out = np.clip(out, 0, 1)
        # Resize back to original face size
        if out.shape[0] != orig_h or out.shape[1] != orig_w:
            out = cv2.resize(out, (orig_w, orig_h),
                             interpolation=cv2.INTER_CUBIC)
        return out.astype(np.float32)

    return extractor


def set_current_model(model_path: str):
    global _current_model_path, _current_predictor, _prediction_cache, _predictors
    _current_model_path = model_path
    _prediction_cache.clear()  # new model = new predictions
    try:
        session = model_loader.load_model(model_path)
        inp = session.get_inputs()[0]
        input_name = inp.name
        input_shape = inp.shape  # e.g. [1, 3, 256, 256] (NCHW) or [1, 416, 416, 3] (NHWC)
        print(f"[MergeStudio] Model input shape: {input_shape}")

        # Detect format: NCHW if channel dim (1) is 1/3, NHWC if last dim is 1/3
        is_nchw = len(input_shape) == 4 and input_shape[1] in (1, 3)
        if not is_nchw and len(input_shape) == 4:
            # Check last dim for NHWC (e.g. DFM has ['batch', 416, 416, 3])
            if isinstance(input_shape[3], (int, float)) and input_shape[3] in (1, 3):
                is_nchw = False  # NHWC
            elif all(not isinstance(d, (int, float)) for d in input_shape):
                is_nchw = True  # all strings, default NCHW
        # Get required input size
        if is_nchw:
            req_size = input_shape[2]  # NCHW: [N, C, H, W]
        else:
            req_size = input_shape[1]  # NHWC: [N, H, W, C]

        # Store model info for MergeMaskedFace
        from MergeStudio.core import merger as _merger_mod2
        _merger_mod2._model_input_size = req_size

        def predictor(face_img):
            # Resize to model's required size
            h, w = face_img.shape[:2]
            if h != req_size or w != req_size:
                face_img = cv2.resize(face_img, (req_size, req_size),
                                      interpolation=cv2.INTER_LANCZOS4)
            img = face_img.astype(np.float32)
            if is_nchw:
                input_tensor = np.transpose(img, (2, 0, 1))[None, :, :, :]
            else:
                input_tensor = img[None, :, :, :]  # NHWC
            outputs = session.run(None, {input_name: input_tensor})
            # Model outputs: [0]=face_mask(1ch), [1]=celeb_face(3ch), [2]=celeb_mask(1ch)
            # Find the 3-channel output (the actual swapped face)
            face_output = None
            for out in outputs:
                if out.ndim == 4 and out.shape[-1] == 3:
                    face_output = out
                    break
            if face_output is None:
                face_output = outputs[0]
                if face_output.shape[-1] == 1:
                    face_output = np.repeat(face_output, 3, axis=-1)

            # Extract masks from model outputs (0-1 range)
            pm, dm = None, None
            if len(outputs) >= 3:
                print(f"[MergeStudio] Model masks: pred range=[{outputs[0].min():.3f},{outputs[0].max():.3f}] mean={outputs[0].mean():.3f}, "
                      f"dst range=[{outputs[2].min():.3f},{outputs[2].max():.3f}] mean={outputs[2].mean():.3f}")
                m0 = np.squeeze(outputs[0])
                if m0.ndim == 3: m0 = m0[:, :, 0]
                pm = cv2.resize(m0.astype(np.float32), (req_size, req_size)) if m0.shape[:2] != (req_size, req_size) else m0.astype(np.float32)
                m2 = np.squeeze(outputs[2])
                if m2.ndim == 3: m2 = m2[:, :, 0]
                dm = cv2.resize(m2.astype(np.float32), (req_size, req_size)) if m2.shape[:2] != (req_size, req_size) else m2.astype(np.float32)

            # Keep in 0-1 range (merger works in 0-1)
            while face_output.ndim >= 5: face_output = face_output[0]
            if face_output.ndim == 4:
                face_output = face_output[0] if not is_nchw else np.transpose(face_output[0], (1, 2, 0))
            if face_output.ndim == 2:
                face_output = np.stack([face_output]*3, axis=-1)
            if face_output.shape[-1] > 3: face_output = face_output[:, :, :3]
            elif face_output.shape[-1] == 1: face_output = np.repeat(face_output, 3, axis=-1)
            if face_output.shape[:2] != (req_size, req_size):
                face_output = cv2.resize(face_output, (req_size, req_size), interpolation=cv2.INTER_LANCZOS4)

            # Return (face, pred_mask, dst_mask) tuple
            return (face_output, pm, dm)

        _current_predictor = predictor
        # Also store in multi-model dict keyed by model prefix
        model_key = Path(model_path).stem.replace('_model', '').split('_')[0]
        _predictors[model_key] = predictor
        model_key = Path(model_path).stem.replace('_model', '').split('_')[0]
        _predictor_sizes[model_key] = req_size
        print(f"[MergeStudio] Predictor ready: {req_size}x{req_size}, {'NCHW' if is_nchw else 'NHWC'} (_predictors keys: {list(_predictors.keys())})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Failed to load model:", e)
        _current_predictor = None


def _set_model_input_size(pred_fn):
    """Set global _model_input_size to match the predictor's model."""
    from MergeStudio.core import merger as _merger_mod
    for name, p in _predictors.items():
        if p == pred_fn:
            sz = _predictor_sizes.get(name, _merger_mod._model_input_size)
            _merger_mod._model_input_size = sz
            return


def _get_predictor_for_face(face_idx: int, face_model_map: dict):
    """Resolve which predictor to use for a given face. Falls back to default."""
    if face_model_map:
        model_name = face_model_map.get(face_idx) or face_model_map.get(str(face_idx))
        if model_name:
            # Direct match
            if model_name in _predictors:
                return _predictors[model_name]
            # Fallback: try matching by prefix (model may have been loaded with full stem)
            for pname in _predictors:
                if pname.startswith(model_name) or model_name.startswith(pname):
                    return _predictors[pname]
            print(f"[MergeStudio] WARNING: model '{model_name}' not loaded for face {face_idx}, using default")
    return _current_predictor


# ===== Persistent config =====
_CONFIG_PATH = None

def _get_config_path():
    global _CONFIG_PATH
    if _CONFIG_PATH is None:
        _CONFIG_PATH = Path(__file__).parent.parent / "workspace" / "last_config.json"
    return _CONFIG_PATH

def _save_user_config(config_dict: dict):
    """Persist config so it survives server restarts."""
    import json
    try:
        _get_config_path().write_text(json.dumps(config_dict, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[MergeStudio] Failed to save config: {e}")

@router.get("/preview/log")
async def preview_log(msg: str = ""):
    print(f"[Frontend] {msg}", flush=True)
    return {"status": "ok"}

@router.post("/preview/save-config")
async def save_config(data: dict):
    _save_user_config(data)
    return {"status": "ok"}

@router.get("/preview/load-config")
async def load_config():
    path = _get_config_path()
    if path.exists():
        try:
            import json
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}
