import cv2
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException
from MergeStudio.api.schemas import CutSegment
from MergeStudio.core.detector.pipeline import detect_and_align
from MergeStudio.core.detector.factory import DetectorFactory, LandmarkFactory, get_device_info

router = APIRouter()

_cut_segments = []


@router.get("/timeline/faces")
async def get_faces():
    """Get face distribution data for timeline display.
    Returns data from aligned directory if available, otherwise empty."""
    from MergeStudio.api.routes_preview import _current_video
    if _current_video is None:
        return {"face_data": [], "source": "none"}

    video_path = Path(_current_video)
    aligned_dir = video_path.parent / "data_dst" / "aligned"
    if aligned_dir.exists():
        jpgs = sorted(aligned_dir.glob("*.jpg"))
        if jpgs:
            # Frame numbers from filenames
            face_data = []
            for jpg in jpgs:
                try:
                    # DFL aligned format: framename_faceidx.jpg
                    stem = jpg.stem
                    parts = stem.split('_')
                    frame_num = int(parts[0]) if parts[0].isdigit() else 0
                    face_data.append({"frame": frame_num, "face_count": 1})
                except Exception:
                    pass
            return {"face_data": face_data, "source": "aligned"}
    return {"face_data": [], "source": "none"}


@router.get("/timeline/scan-faces")
async def scan_faces():
    """Scan video and return face count per frame (sampled)."""
    from MergeStudio.api.routes_preview import _current_video
    if _current_video is None:
        raise HTTPException(400, "No video loaded")

    # First try aligned data (faster, more accurate)
    video_path = Path(_current_video)
    aligned_dir = video_path.parent / "data_dst" / "aligned"
    if aligned_dir.exists():
        jpgs = sorted(aligned_dir.glob("*.jpg"))
        if jpgs:
            face_data = set()
            for jpg in jpgs:
                try:
                    stem = jpg.stem
                    parts = stem.split('_')
                    frame_num = int(parts[0]) if parts[0].isdigit() else 0
                    face_data.add(frame_num)
                except Exception:
                    pass
            total_frames = 0
            cap = cv2.VideoCapture(str(_current_video))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            result = []
            for f in range(0, total_frames, max(1, total_frames // 200)):
                result.append({"frame": f, "face_count": 1 if f in face_data else 0})
            return {"face_data": result, "source": "aligned"}

    # Fallback: detect on sampled frames
    cap = cv2.VideoCapture(_current_video)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return {"face_data": [], "source": "detect"}

    try:
        device = get_device_info()
        detector = DetectorFactory.create('TinyMog', device)
        landmarker = LandmarkFactory.create('insightface-2d106det', device)
    except Exception:
        cap.release()
        return {"face_data": [], "source": "detect"}

    step = max(1, total // 200)
    face_data = []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            det_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            faces = detect_and_align(detector, landmarker, det_frame, 'whole_face')
            face_data.append({"frame": idx, "face_count": len(faces)})
        except Exception:
            face_data.append({"frame": idx, "face_count": 0})

    cap.release()
    return {"face_data": face_data, "source": "detect"}


@router.post("/timeline/cut")
async def update_cut(req: CutSegment):
    global _cut_segments

    if req.action == "add":
        _cut_segments.append({
            "start_frame": req.start_frame,
            "end_frame": req.end_frame,
        })
    elif req.action == "remove":
        _cut_segments = [
            s for s in _cut_segments
            if not (s['start_frame'] == req.start_frame and s['end_frame'] == req.end_frame)
        ]
    elif req.action == "update":
        for s in _cut_segments:
            if s['start_frame'] == req.start_frame:
                s['end_frame'] = req.end_frame

    return {"status": "ok", "segments": _cut_segments}
