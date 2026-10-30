"""Progress persistence — remember last viewed image index per workspace."""

import json
from pathlib import Path

from fastapi import APIRouter

from MaskProcessor.api.routes_project import get_workspace

router = APIRouter()

PROGRESS_FILENAME = ".maskprocessor_progress.json"


def _progress_path() -> Path:
    """Path of the progress file inside the active workspace directory."""
    ws = get_workspace()
    return Path(ws.root) / PROGRESS_FILENAME


@router.post("/progress/save")
async def save_progress(data: dict):
    """Save the last viewed image index for the current workspace."""
    index = data.get("index", 0)
    try:
        p = _progress_path()
        p.write_text(json.dumps({"index": index}), encoding="utf-8")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/progress/load")
async def load_progress():
    """Load the last viewed image index for the current workspace."""
    try:
        p = _progress_path()
        if p.exists():
            state = json.loads(p.read_text(encoding="utf-8"))
            return {"success": True, "index": state.get("index", 0)}
        return {"success": True, "index": 0}
    except Exception as e:
        return {"success": False, "index": 0, "error": str(e)}
