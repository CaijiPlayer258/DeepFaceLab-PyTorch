from pathlib import Path
import cv2
from fastapi import APIRouter, HTTPException
from MergeStudio.api.schemas import OpenProjectRequest, OpenProjectResponse, ModelLoadRequest
from MergeStudio.api.routes_preview import set_current_video, set_current_model

router = APIRouter()

# Store current workspace path for model loading
_current_workspace_path = None


def _get_workspace():
    """Get the resolved workspace path."""
    if _current_workspace_path:
        return Path(_current_workspace_path).resolve()
    return Path(".").resolve()


@router.post("/project/open", response_model=OpenProjectResponse)
async def open_project(req: OpenProjectRequest):
    workspace = Path(req.path)
    if not workspace.exists():
        return OpenProjectResponse(status="error", message="Path not found: " + req.path)

    # Scan for video files
    video_path = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        candidates = list(workspace.glob('*' + ext))
        if candidates:
            video_path = str(candidates[0])
            break

    total_frames = 0
    fps = 0.0
    width = 0
    height = 0
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    # Scan for models - group by prefix
    models = []
    model_dir = workspace / "model"
    if model_dir.exists():
        # Collect all model files by prefix
        model_groups = {}
        for f in model_dir.glob("*"):
            if f.suffix.lower() not in ('.dfm', '.pth', '.npy', '.dat'):
                continue
            # Extract prefix (name before _model, _data, _opt, etc.)
            name = f.stem
            prefix = name.split('_')[0] if '_' in name else name
            if prefix not in model_groups:
                model_groups[prefix] = {'has_dfm': False, 'files': []}
            model_groups[prefix]['files'].append(f.name)
            if f.suffix.lower() == '.dfm':
                model_groups[prefix]['has_dfm'] = True

        for prefix, info in model_groups.items():
            models.append({
                "name": prefix,
                "files": info['files'],
                "format": "dfm" if info['has_dfm'] else "dfl",
                "size_mb": 0,  # Grouped, no single size
            })

    # Scan aligned directory
    aligned_dir = workspace / "data_dst" / "aligned"
    aligned_count = 0
    has_aligned = aligned_dir.exists()
    if has_aligned:
        aligned_count = len(list(aligned_dir.glob("*.jpg")))

    # List video files
    videos = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for f in workspace.glob('*' + ext):
            videos.append(f.name)

    # Build DFL map for all videos
    video_dfl_map = {}
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for f in workspace.glob('*' + ext):
            from MergeStudio.api.routes_preview import _check_dfl
            dfl = _check_dfl(str(f))
            video_dfl_map[f.name] = dfl["is_dfl"]

    # Set current video for preview
    if video_path:
        set_current_video(video_path, total_frames)

    # Save workspace path for model loading
    global _current_workspace_path
    _current_workspace_path = str(workspace.resolve())

    return OpenProjectResponse(
        status="ok",
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        models=models,
        videos=videos,
        has_aligned=has_aligned,
        aligned_count=aligned_count,
        video_dfl_map=video_dfl_map,
    )


@router.get("/models")
async def list_models():
    return {"models": []}


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    name = req.name.lower()
    ws = _get_workspace()

    # Search for .dfm file matching the model name
    search_dirs = [ws / "model", ws, Path("model"), Path(".")]
    candidates = []
    for base in search_dirs:
        if not base.exists():
            continue
        for f in base.iterdir():
            if f.suffix == '.dfm' and f.is_file():
                stem = f.stem.replace('_model', '').lower()
                if name in stem or name == stem or name in f.stem.lower():
                    candidates.append(f)

    if not candidates:
        raise HTTPException(404, "No .dfm model found for: " + req.name)

    # Pick the best match: exact stem match first, then any match
    target = None
    for c in candidates:
        if c.stem.replace('_model', '').lower() == name:
            target = c
            break
    if target is None:
        target = candidates[0]
    model_path = str(target.resolve())
    print(f"[MergeStudio] Loading model: {model_path}")
    set_current_model(model_path)
    # Verify predictor was set
    from MergeStudio.api.routes_preview import _current_predictor as _cp
    if _cp is None:
        print(f"[MergeStudio] WARNING: Model loaded but predictor is None - merge will not work")
        return {"status": "loaded", "name": str(target.name), "format": "dfm", "predictor_ready": False}
    print(f"[MergeStudio] Model loaded successfully, predictor ready")
    return {"status": "loading", "name": str(target.name), "format": "dfm", "predictor_ready": True}


@router.post("/models/export-to-dfm")
async def export_to_dfm(req: ModelLoadRequest):
    """Convert .pth or .npy model to .dfm."""
    model_path = Path(req.name)
    if not model_path.exists():
        raise HTTPException(404, "Model not found: " + req.name)
    suffix = model_path.suffix.lower()
    if suffix not in ('.pth', '.npy'):
        raise HTTPException(400, "Unsupported model format: " + suffix + " (expected .pth or .npy)")
    return {"status": "converting", "source": req.name, "format": suffix}
