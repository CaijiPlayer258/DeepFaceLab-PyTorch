"""
FastAPI app factory with CORS, API routes, and SPA fallback.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


def create_app() -> FastAPI:
    app = FastAPI(title="MergeStudio", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes FIRST (takes precedence)
    from MergeStudio.api.routes_project import router as project_router
    from MergeStudio.api.routes_preview import router as preview_router
    from MergeStudio.api.routes_timeline import router as timeline_router
    from MergeStudio.api.routes_export import router as export_router

    app.include_router(project_router, prefix="/api")
    app.include_router(preview_router, prefix="/api")
    app.include_router(timeline_router, prefix="/api")
    app.include_router(export_router, prefix="/api")

    # Serve static UI files (catch-all for non-API paths)
    ui_dir = Path(__file__).parent.parent / "ui"
    _no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    @app.get("/{full_path:path}")
    async def serve_ui(full_path: str):
        file_path = ui_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path), headers=_no_cache_headers)
        # Fallback to index.html (SPA)
        index = ui_dir / "index.html"
        if index.exists():
            return FileResponse(str(index), headers=_no_cache_headers)
        return {"error": "not found"}


    @app.on_event("shutdown")
    async def cleanup_cache():
        try:
            from MergeStudio.api.routes_preview import _cache_dir
            if _cache_dir is not None and _cache_dir.exists():
                import shutil
                shutil.rmtree(str(_cache_dir), ignore_errors=True)
                print("[MergeStudio] Preview cache cleaned up")
        except Exception as e:
            print(f"[MergeStudio] Cache cleanup: {e}")

    return app
