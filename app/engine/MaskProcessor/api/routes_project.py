"""Project (workspace) routes — open and manage aligned facesets."""

from fastapi import APIRouter, HTTPException

from MaskProcessor.api.schemas import ProjectOpenRequest, ProjectOpenResponse, ErrorResponse
from MaskProcessor.workspace import Workspace

router = APIRouter()

# Singleton workspace — one at a time
_workspace: Workspace | None = None


def get_workspace() -> Workspace:
    """Return the active workspace singleton or raise 409."""
    if _workspace is None:
        raise HTTPException(status_code=409, detail="No project open. Call /api/project/open first.")
    return _workspace


def _set_workspace(ws: Workspace | None) -> None:
    """Set the active workspace (used by mask routes for undo)."""
    global _workspace
    _workspace = ws


@router.post("/project/open", response_model=ProjectOpenResponse)
async def open_project(req: ProjectOpenRequest):
    """Open an aligned faceset directory and return the list of images."""
    try:
        ws = Workspace(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open workspace: {e}")

    _set_workspace(ws)

    files = [
        {
            "name": entry.name,
            "path": str(entry.path),
            "has_mask": entry.has_mask,
        }
        for entry in ws.files
    ]
    return ProjectOpenResponse(files=files, count=len(files))
