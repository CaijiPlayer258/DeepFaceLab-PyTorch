"""Image routes — load and serve workspace images as base64 JPEG."""

from fastapi import APIRouter, HTTPException
import cv2
import base64
import numpy as np

from MaskProcessor.api.schemas import ImageResponse, ErrorResponse
from MaskProcessor.api.routes_project import get_workspace

router = APIRouter()


@router.get("/image/{index}", response_model=ImageResponse)
async def get_image(index: int):
    """Load image at *index* from the workspace and return as base64 JPEG."""
    ws = get_workspace()

    if index < 0 or index >= len(ws):
        raise HTTPException(status_code=404, detail=f"Image index {index} out of range (0-{len(ws) - 1})")

    try:
        bgr = ws.load_image(index)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    height, width = bgr.shape[:2]

    # Encode as JPEG in memory
    success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image as JPEG")

    b64 = base64.b64encode(buf).decode("utf-8")

    entry = ws[index]
    return ImageResponse(image=b64, width=width, height=height, has_mask=entry.has_mask)


@router.get("/image/{index}/mask")
async def get_image_mask(index: int):
    """Return the existing XSeg mask from DFLJPG as base64 PNG."""
    ws = get_workspace()
    mask = ws.get_mask(index)
    if mask is None:
        return {"mask": "", "success": False}
    mask_img = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    if len(mask_img.shape) == 3 and mask_img.shape[2] == 1:
        mask_img = mask_img[:, :, 0]
    _, buf = cv2.imencode(".png", mask_img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return {"mask": b64, "success": True}
