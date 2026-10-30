"""GroundingDINO + SAM2 text-to-mask pipeline.

Chains GroundingDINO (text prompts -> bounding boxes) with SAM2
(bounding boxes -> segmentation masks) for open-vocabulary
text-driven segmentation.

Typical usage::

    predictor = GroundedSAM2()
    img = cv2.imread("photo.jpg")
    predictor.load_image(img)
    mask = predictor.predict("face.")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

# Local SAM2 model
from xlib.models.sam2.build_sam import build_sam2
from xlib.models.sam2.sam2_image_predictor import SAM2ImagePredictor

# Local GroundingDINO model
import xlib.models.grounding_dino.groundingdino.datasets.transforms as T
from xlib.models.grounding_dino.groundingdino.util.inference import load_model, predict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent

SAM2_CKPT_DIR = PROJECT_ROOT / "xlib" / "models" / "sam2" / "checkpoints"
GDINO_CKPT_DIR = PROJECT_ROOT / "xlib" / "models" / "grounding_dino" / "gdino_checkpoints"

# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

SAM2_MODELS: dict[str, tuple[str, str]] = {
    "sam2.1_hiera_tiny": ("sam2.1_hiera_tiny.pt", "sam2.1/sam2.1_hiera_t.yaml"),
    "sam2.1_hiera_small": ("sam2.1_hiera_small.pt", "sam2.1/sam2.1_hiera_s.yaml"),
    "sam2.1_hiera_base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2.1_hiera_large": ("sam2.1_hiera_large.pt", "sam2.1/sam2.1_hiera_l.yaml"),
}

GDINO_MODELS: dict[str, tuple[str, str]] = {
    "groundingdino_swint_ogc": (
        "groundingdino_swint_ogc.pth",
        "GroundingDINO_SwinT_OGC.py",
    ),
    "groundingdino_swinb_cogcoor": (
        "groundingdino_swinb_cogcoor.pth",
        "GroundingDINO_SwinB_cfg.py",
    ),
}

# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class GroundedSAM2:
    """GroundingDINO + SAM2 text-to-mask pipeline.

    Parameters
    ----------
    sam2_model:
        Key into :data:`SAM2_MODELS`.
    gdino_model:
        Key into :data:`GDINO_MODELS`.
    device:
        Torch device string. Auto-detected when *None*.
    box_threshold:
        Minimum box confidence for GroundingDINO.
    text_threshold:
        Minimum text-confidence for GroundingDINO phrase filtering.
    """

    def __init__(
        self,
        sam2_model: str = "sam2.1_hiera_small",
        gdino_model: str = "groundingdino_swint_ogc",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- SAM2 ---------------------------------------------------------
        sam2_ckpt_name, sam2_cfg_rel = SAM2_MODELS[sam2_model]
        sam2_cfg = str(PROJECT_ROOT / "xlib" / "models" / "sam2" / "configs" / sam2_cfg_rel)
        sam2_ckpt = str(SAM2_CKPT_DIR / sam2_ckpt_name)

        sam2_model_obj = build_sam2(sam2_cfg, sam2_ckpt, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model_obj)

        # ---- GroundingDINO -------------------------------------------------
        gdino_ckpt_name, gdino_cfg_name = GDINO_MODELS[gdino_model]
        gdino_cfg = str(
            PROJECT_ROOT
            / "xlib"
            / "models"
            / "grounding_dino"
            / "groundingdino"
            / "config"
            / gdino_cfg_name
        )
        gdino_ckpt = str(GDINO_CKPT_DIR / gdino_ckpt_name)

        self.gdino_model = load_model(gdino_cfg, gdino_ckpt, device=device)

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Cached image data (set by :meth:`load_image`)
        self._image_bgr: Optional[np.ndarray] = None
        self._image_rgb: Optional[np.ndarray] = None
        self._image_tensor: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image: np.ndarray):
        """Set image for mask generation.

        Preprocesses the image for both SAM2 (embedding computation) and
        GroundingDINO (normalized tensor).

        Args:
            image: BGR numpy array *(H, W, 3)* -- standard OpenCV format.
        """
        self._image_bgr = image
        self._image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SAM2 expects RGB image
        self.sam2_predictor.set_image(self._image_rgb)

        # GroundingDINO preprocessing (matches Model.preprocess_image)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(self._image_rgb)
        image_transformed, _ = transform(image_pillow, None)
        self._image_tensor = image_transformed

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text_prompt: str) -> Optional[np.ndarray]:
        """Generate a segmentation mask from a text prompt.

        Pipeline:

        1. **GroundingDINO** detects bounding boxes matching *text_prompt*.
        2. Each box is converted from *cxcywh* (normalized) to *xyxy* (pixel).
        3. **SAM2** generates a segmentation mask for every box with
           ``multimask_output=False``.
        4. All per-box masks are merged via element-wise maximum.

        Args:
            text_prompt:
                Lowercase text prompt, optionally ending with a period
                (e.g. ``"face."``, ``"hair"``, ``"glasses"``).

        Returns:
            Float32 mask in **[0, 1]** with shape *(H, W)*, or *None*
            when no objects are detected.
        """
        if self._image_bgr is None:
            raise RuntimeError("Call load_image() before predict().")

        # ---- Step 1: GroundingDINO box detection ---------------------------
        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=self._image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            return None

        # ---- Step 2: cxcywh (normalised) -> xyxy (pixel) ------------------
        h, w = self._image_bgr.shape[:2]
        boxes_xyxy = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = box_convert(boxes=boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy")
        boxes_xyxy_np = boxes_xyxy.cpu().numpy()

        # ---- Step 3: SAM2 mask generation per box -------------------------
        masks: list[np.ndarray] = []
        for box in boxes_xyxy_np:
            m, iou_predictions, low_res_masks = self.sam2_predictor.predict(
                box=box,
                multimask_output=False,
            )
            # m shape: CxHxW, where C = 1 when multimask_output=False
            mask = m[0].astype(np.float32)
            masks.append(mask)

        # ---- Step 4: Merge masks ------------------------------------------
        merged = np.maximum.reduce(masks).astype(np.float32)

        return merged
