"""XSeg mask generation routes — XSeg and XSegLite models."""

from fastapi import APIRouter, HTTPException
import base64
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel

from MaskProcessor.api.routes_project import get_workspace

router = APIRouter()


class ImageIndexRequest(BaseModel):
    image_index: int

PROJECT_ROOT = Path(__file__).parent.parent.parent
XSEG_DIR = PROJECT_ROOT / "workspace" / "model" / "XSeg"
XSEGLITE_DIR = PROJECT_ROOT / "workspace" / "model" / "XSegLite"


def _find_weight(weights_dir, names):
    """Find first existing weight file from a list of (filename, priority)."""
    for name in names:
        p = weights_dir / name
        if p.exists():
            return str(p)
    return None


def _mask_to_b64(mask: np.ndarray) -> str:
    img = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


@router.post("/xseg/predict")
async def xseg_predict(req: ImageIndexRequest):
    """Generate XSeg mask — try .npy first, fall back to .onnx."""
    ws = get_workspace()
    bgr = ws.load_image(req.image_index)
    h, w = bgr.shape[:2]

    # ONNX path
    onnx_path = _find_weight(XSEG_DIR, ["XSeg.onnx"])
    if onnx_path:
        try:
            import onnxruntime as ort
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))[None, ...]  # NCHW

            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_img})
            mask = np.squeeze(outputs[0])
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            return {"mask": _mask_to_b64(mask), "success": True, "backend": "onnx"}
        except Exception as e:
            print(f"[xseg] onnx failed: {e}")

    # Fallback to .npy
    npy_path = _find_weight(XSEG_DIR, ["XSeg_256.npy", "XSeg_256_opt.npy"])
    if npy_path:
        try:
            from facelib.XSegNet import XSegNet
            model = XSegNet(
                name="XSeg", resolution=256, load_weights=True,
                weights_file_root=str(XSEG_DIR),
                place_model_on_cpu=False, run_on_cpu=False,
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
            rgb = cv2.resize(rgb, (256, 256))
            mask = model.extract(rgb)
            mask = mask.squeeze()
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            return {"mask": _mask_to_b64(mask), "success": True, "backend": "npy"}
        except Exception as e:
            print(f"[xseg] npy failed: {e}")

    raise HTTPException(status_code=404, detail="No XSeg weights found")


@router.post("/xseglite/predict")
async def xseglite_predict(req: ImageIndexRequest):
    """Generate XSegLite mask via ONNX (re-exported with sigmoid baked in)."""
    ws = get_workspace()
    bgr = ws.load_image(req.image_index)
    h, w = bgr.shape[:2]

    onnx_path = _find_weight(XSEGLITE_DIR, ["xseglite.onnx"])
    if onnx_path:
        try:
            import onnxruntime as ort
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))[None, ...]  # NCHW

            session = ort.InferenceSession(onnx_path)
            outputs = session.run(None, {session.get_inputs()[0].name: input_img})
            mask = np.squeeze(outputs[0])
            mask = (mask > 0.5).astype(np.float32)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            return {"mask": _mask_to_b64(mask), "success": True, "backend": "onnx"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"XSegLite ONNX failed: {e}")

    raise HTTPException(status_code=404, detail="xseglite.onnx not found")
