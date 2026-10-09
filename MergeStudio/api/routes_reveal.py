"""U4 后端：/api/export/reveal — 在资源管理器中打开导出文件所在目录。

安全约束：只允许打开 workspace 内的路径（explorer /select 防注入：路径作为
独立 argv 传递，不经过 shell 拼接）。
"""
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RevealRequest(BaseModel):
    path: str


def _workspace_root() -> Path:
    try:
        from MergeStudio.api.routes_preview import _get_cache_dir
        # cache_dir 在 MergeStudio/workspace 下，项目根是它的上两级
        return Path(_get_cache_dir()).parents[1]
    except Exception:
        return Path(__file__).resolve().parents[2] / "workspace"


@router.post("/export/reveal")
async def export_reveal(req: RevealRequest):
    target = Path(req.path)
    root = _workspace_root().resolve()
    try:
        target_resolved = target.resolve()
    except OSError:
        return {"status": "error", "message": "invalid path"}

    if root not in target_resolved.parents and target_resolved != root:
        return {"status": "error", "message": "path outside workspace"}

    if not target_resolved.exists():
        return {"status": "error", "message": "file not found"}

    import subprocess
    subprocess.Popen(["explorer", "/select,", str(target_resolved)])
    return {"status": "ok"}
