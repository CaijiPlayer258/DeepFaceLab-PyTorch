"""
Export pipeline API endpoints.
"""
import threading
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

_export_stop = threading.Event()  # set to signal cancellation


class ExportStartRequest(BaseModel):
    video_path: str
    output_path: str = ""
    image_format: str = "jpg"
    encoder: str = "h264_nvenc"
    config: dict = {}
    face_db: dict = {}
    face_model_map: dict = {}
    cut_segments: list = []
    angle_segments: list = []
    detector: str = "YOLOv8"
    landmarker: str = "insightface-2d106det"
    res_scale: float = 0.5
    hwaccel: str = ''
    num_workers: int = 0


class ExportStatusResponse(BaseModel):
    running: bool = False
    stage: int = 0
    stage_name: str = ""
    progress: float = 0.0
    message: str = ""
    output_path: str = ""
    tick: int = 0


_export_job = {
    "running": False,
    "stage": 0,
    "stage_name": "",
    "progress": 0.0,
    "message": "",
    "output_path": "",
}


def _update_progress(stage: int, stage_name: str, progress: float, message: str):
    global _export_job
    if _export_stop.is_set():
        # Pipeline is checking in — cancellation requested
        raise StopRequested("Export cancelled by user")
    _export_job.update({
        "stage": stage, "stage_name": stage_name,
        "progress": progress, "message": message,
        "tick": _export_job.get("tick", 0) + 1,
    })
    if stage >= 6 and progress >= 1.0:
        _export_job["running"] = False
        _export_job["message"] = "Complete"


class StopRequested(Exception):
    """Raised inside the pipeline when the user cancels."""
    pass


@router.post("/export/start")
async def export_start(req: ExportStartRequest):
    global _export_job
    if _export_job["running"]:
        raise HTTPException(400, "Export already running")

    video = Path(req.video_path)
    if not video.exists():
        raise HTTPException(400, f"Video not found: {req.video_path}")

    output = req.output_path
    if not output:
        output = str(video.parent / f"{video.stem}_result.mp4")

    from MergeStudio.core.export_pipeline import run_export_pipeline

    _export_stop.clear()
    _export_job.update({
        "running": True, "stage": 0, "stage_name": "Initializing",
        "progress": 0.0, "message": "Starting export...", "output_path": output,
    })

    def _run():
        global _export_job
        import traceback
        try:
            import multiprocessing
            nw = req.num_workers if req.num_workers > 0 else max(1, multiprocessing.cpu_count() // 2)
            print(f"[Export] Starting pipeline with cut_segments={req.cut_segments}", flush=True)
            print(f"[Export] detector={req.detector} landmarker={req.landmarker}", flush=True)
            run_export_pipeline(
                video_path=req.video_path,
                output_path=output,
                image_format=req.image_format,
                encoder=req.encoder,
                config=req.config,
                face_db=req.face_db,
                face_model_map=req.face_model_map,
                cut_segments=req.cut_segments,
                angle_segments=req.angle_segments,
                detector=req.detector,
                landmarker=req.landmarker,
                res_scale=req.res_scale,
                hwaccel=req.hwaccel,
                progress_callback=_update_progress,
                stop_event=_export_stop,
                num_workers=nw,
            )
        except StopRequested:
            print("[Export] Cancelled by user", flush=True)
            _export_job.update({"running": False, "message": "Cancelled"})
        except Exception as e:
            print(f"[Export] FAILED: {traceback.format_exc()}", flush=True)
            _export_job.update({"running": False, "message": f"Failed: {e}"})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "output_path": output}


@router.get("/export/progress/{job_id}")
async def export_progress(job_id: str = "default"):
    global _export_job
    return ExportStatusResponse(**_export_job)


@router.post("/export/cancel/{job_id}")
async def export_cancel(job_id: str = "default"):
    global _export_job
    if not _export_job["running"]:
        return {"status": "not_running"}
    _export_stop.set()
    _export_job.update({"message": "Cancelling..."})
    return {"status": "cancelling"}


class ComputeEmbeddingsRequest(BaseModel):
    video_path: str = ""
    face_database: dict = {}  # {key: {thumb_url, label, ...}}


@router.post("/face/compute-embeddings")
async def compute_embeddings(req: ComputeEmbeddingsRequest):
    """Compute ArcFace embeddings for all faces in the database and cluster them."""
    from MergeStudio.core.face_embedder import compute_embedding
    from MergeStudio.core.face_cluster import cluster_embeddings_dbscan
    import cv2
    import numpy as np
    from pathlib import Path

    try:
        from MergeStudio.api.routes_preview import _get_cache_dir
        cache_dir = _get_cache_dir()
    except Exception:
        cache_dir = Path(__file__).parent.parent / "workspace" / "preview_cache"

    face_db = req.face_database or {}
    if not face_db:
        return {"embeddings": {}, "clusters": {}, "embedding_count": 0}

    embeddings = {}
    failed = 0

    for key, info in face_db.items():
        if isinstance(info, dict):
            parts = key.split('_')
            if len(parts) < 2:
                failed += 1
                continue
            fidx, fi = parts[0], parts[1]
            thumb_path = cache_dir / f"face_{fidx}_{fi}.jpg"
            if not thumb_path.exists():
                thumb_path = cache_dir / f"{key}.jpg"
            if thumb_path.exists():
                face_img = cv2.imread(str(thumb_path))
                if face_img is not None and face_img.size > 0:
                    try:
                        emb = compute_embedding(face_img)
                        embeddings[key] = emb.tolist()
                        continue
                    except Exception:
                        pass
        failed += 1

    if not embeddings:
        return {"embeddings": {}, "clusters": {}, "embedding_count": 0}

    emb_dict = {k: np.array(v, dtype=np.float32) for k, v in embeddings.items()}
    clusters = cluster_embeddings_dbscan(emb_dict, eps=0.3, min_samples=1)

    return {
        "embeddings": embeddings,
        "clusters": clusters,
        "embedding_count": len(embeddings),
    }
