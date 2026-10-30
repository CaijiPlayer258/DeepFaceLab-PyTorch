from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from MaskProcessor.api.routes_project import router as project_router
from MaskProcessor.api.routes_image import router as image_router
from MaskProcessor.api.routes_mask import router as mask_router
from MaskProcessor.api.routes_xseg import router as xseg_router
from MaskProcessor.api.routes_progress import router as progress_router

app = FastAPI(title="Mask Processor", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Middleware: disable caching for all static UI files
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api") or path.startswith("/ws"):
            return response
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheMiddleware)

app.include_router(project_router, prefix="/api")
app.include_router(image_router, prefix="/api")
app.include_router(mask_router, prefix="/api")
app.include_router(xseg_router, prefix="/api")
app.include_router(progress_router, prefix="/api")

# Mount static UI files if ui/ directory exists
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
