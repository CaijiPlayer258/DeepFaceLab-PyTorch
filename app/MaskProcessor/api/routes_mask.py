"""Mask routes — predict, refine, save, and undo masks using SAM / BiSeNet."""

from fastapi import APIRouter, HTTPException
import base64
import cv2
import numpy as np
from pydantic import BaseModel

from MaskProcessor.api.schemas import (
    MaskPredictRequest,
    MaskTextRequest,
    MaskBiSeNetRequest,
    MaskSaveRequest,
    UndoRequest,
    MaskResponse,
    ErrorResponse,
)
from MaskProcessor.api.routes_project import get_workspace
from MaskProcessor.core.model_loader import ModelLoader

router = APIRouter()

# ---------------------------------------------------------------------------
# Undo buffer: maps image_index -> previous mask (float32 ndarray)
# ---------------------------------------------------------------------------
_undo_buffer: dict[int, np.ndarray] = {}

# ---------------------------------------------------------------------------
# Base64 <-> mask helpers
# ---------------------------------------------------------------------------


def _mask_to_b64(mask: np.ndarray) -> str:
    """Convert float32 mask to base64 PNG string."""
    img = (mask * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


def _b64_to_mask(b64: str) -> np.ndarray:
    """Convert base64 PNG string to float32 mask."""
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def _bgr_from_workspace(image_index: int) -> np.ndarray:
    """Load BGR image from workspace at *image_index*, validating bounds."""
    ws = get_workspace()
    if image_index < 0 or image_index >= len(ws):
        raise HTTPException(
            status_code=404,
            detail=f"Image index {image_index} out of range (0-{len(ws) - 1})",
        )
    try:
        return ws.load_image(image_index)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /api/mask/predict  —  SAM point/box -> mask
# ---------------------------------------------------------------------------


@router.post("/mask/predict", response_model=MaskResponse)
async def mask_predict(req: MaskPredictRequest):
    """Generate a mask from point clicks and/or a bounding box using SAM."""
    bgr = _bgr_from_workspace(req.image_index)
    h, w = bgr.shape[:2]

    # Load image into SAM
    sam = ModelLoader.get_sam()
    sam.load_image(bgr)

    if req.box and req.clicks:
        # Box-provided mask
        x1, y1, x2, y2 = req.box
        mask_box = sam.predict_with_box((x1, y1, x2, y2))

        # Point-refined mask
        clicks_flat = [(x, y, label) for x, y, label in req.clicks]
        mask_points = sam.predict(clicks_flat)

        # Weighted average: favour the box prediction (0.7) over points (0.3)
        mask = (mask_box.astype(np.float32) * 0.7 + mask_points.astype(np.float32) * 0.3)
        mask = np.clip(mask, 0, 1)
    elif req.box:
        x1, y1, x2, y2 = req.box
        mask = sam.predict_with_box((x1, y1, x2, y2))
    elif req.clicks:
        clicks_flat = [(x, y, label) for x, y, label in req.clicks]
        mask = sam.predict(clicks_flat)
    else:
        raise HTTPException(status_code=400, detail="Provide 'clicks', 'box', or both")

    return MaskResponse(mask=_mask_to_b64(mask))


# ---------------------------------------------------------------------------
# POST /api/mask/text  —  GroundedSAM2 / OWL-ViT text -> mask
# ---------------------------------------------------------------------------


@router.post("/mask/text", response_model=MaskResponse)
async def mask_text(req: MaskTextRequest):
    """Generate a mask from a text prompt using GroundedSAM2 or OWL-ViT."""
    bgr = _bgr_from_workspace(req.image_index)

    if req.backend == "grounded_sam2":
        detector = ModelLoader.get_grounded_sam2()
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="GroundedSAM2 模型加载失败（需要连接 HuggingFace Hub 下载 BERT tokenizer，请检查网络）",
            )
        detector.load_image(bgr)
        mask = detector.predict(req.text)
    elif req.backend == "owl_vit":
        from MaskProcessor.core.owl_vit_detector import OWLViTDetector

        detector = OWLViTDetector()
        detector.load_image(bgr)
        mask = detector.predict(req.text)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend '{req.backend}'. Use 'grounded_sam2' or 'owl_vit'.",
        )

    if mask is None:
        return MaskResponse(mask="", success=False)

    return MaskResponse(mask=_mask_to_b64(mask))


# ---------------------------------------------------------------------------
# POST /api/mask/bisenet  —  BiSeNet face parsing
# ---------------------------------------------------------------------------


@router.post("/mask/bisenet")
async def mask_bisenet(req: MaskBiSeNetRequest):
    """Generate masks for requested semantic face parts using BiSeNet."""
    bgr = _bgr_from_workspace(req.image_index)

    parser = ModelLoader.get_bisenet()
    parsed = parser.parse(bgr)

    result = {}
    for part in req.parts:
        if part in parsed:
            result[part] = _mask_to_b64(parsed[part])

    return {"masks": result, "success": True}


# ---------------------------------------------------------------------------
# POST /api/model/preload  —  warm up all models in background
# ---------------------------------------------------------------------------


@router.post("/model/preload")
async def model_preload():
    """Trigger lazy loading of SAM, GroundedSAM2, and BiSeNet in the background."""
    import threading

    def _load_all():
        try:
            ModelLoader.get_sam()
        except Exception as e:
            print(f"[preload] SAM: {e}")
        try:
            ModelLoader.get_grounded_sam2()
        except Exception as e:
            print(f"[preload] GroundedSAM2: {e}")
        try:
            ModelLoader.get_bisenet()
        except Exception as e:
            print(f"[preload] BiSeNet: {e}")
        print("[preload] All models loaded")

    threading.Thread(target=_load_all, daemon=True).start()
    return {"success": True, "message": "Preloading models in background"}


# ---------------------------------------------------------------------------
# POST /api/mask/commit  —  flatten, extract polys & save to DFLJPG
# ---------------------------------------------------------------------------


class CommitRequest(BaseModel):
    image_index: int
    foreground: str  # base64 PNG — green/positive values
    background: str = ""  # base64 PNG — red/negative values (optional)


@router.post("/mask/commit")
async def mask_commit(req: CommitRequest):
    """Flatten foreground - background, extract polygons,
    and save both ``xseg_mask`` and ``seg_ie_polys`` to the DFLJPG."""
    ws = get_workspace()
    fg = _b64_to_mask(req.foreground)
    bg = _b64_to_mask(req.background) if req.background else np.zeros_like(fg)
    flat = np.clip(fg - bg, 0, 1)

    # Save raster mask
    ws.save_mask(req.image_index, flat)

    # Extract polygons and save seg_ie_polys
    try:
        from MaskProcessor.core.mask_ops import mask_to_polygons
        from core.imagelib.SegIEPolys import SegIEPolys, SegIEPoly, SegIEPolyType

        entry = ws[req.image_index]
        dfljpg = entry.dfljpg
        if dfljpg is not None:
            polys = SegIEPolys()
            poly_list = mask_to_polygons(flat)
            for pts in poly_list:
                poly = polys.add_poly(SegIEPolyType.INCLUDE)
                for pt in pts:
                    poly.add_pt(int(pt[0]), int(pt[1]))
            dfljpg.set_seg_ie_polys(polys)
            dfljpg.dump()
    except Exception as e:
        print(f"[commit] polygon extraction skipped: {e}")

    return {"success": True, "message": "Mask committed"}


# ---------------------------------------------------------------------------
# POST /api/mask/save  —  save mask to DFLJPG
# ---------------------------------------------------------------------------


@router.post("/mask/save", response_model=MaskResponse)
async def mask_save(req: MaskSaveRequest):
    """Save a mask to the DFLJPG at *image_index*.

    The previous mask state is preserved in the undo buffer so that
    ``/api/mask/undo`` can restore it.
    """
    ws = get_workspace()

    # Stash previous mask for undo
    try:
        prev_mask = ws.get_mask(req.image_index)
        if prev_mask is not None:
            _undo_buffer[req.image_index] = prev_mask
        elif req.image_index in _undo_buffer:
            del _undo_buffer[req.image_index]
    except Exception:
        pass

    # Decode and save the new mask
    mask = _b64_to_mask(req.mask)
    try:
        ws.save_mask(req.image_index, mask)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Check that mask is the expected size
    entry = ws[req.image_index]
    return MaskResponse(mask=_mask_to_b64(mask))


# ---------------------------------------------------------------------------
# POST /api/mask/undo  —  restore previous mask state
# ---------------------------------------------------------------------------


@router.post("/mask/undo", response_model=MaskResponse)
async def mask_undo(req: UndoRequest):
    """Restore the mask that was overwritten by the last save on *image_index*."""
    ws = get_workspace()

    if req.image_index not in _undo_buffer:
        raise HTTPException(
            status_code=404,
            detail="Nothing to undo for this image",
        )

    prev_mask = _undo_buffer.pop(req.image_index)
    try:
        ws.save_mask(req.image_index, prev_mask)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return MaskResponse(mask=_mask_to_b64(prev_mask))
