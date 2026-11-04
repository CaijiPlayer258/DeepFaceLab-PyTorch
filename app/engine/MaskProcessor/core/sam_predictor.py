"""Point/box-based mask generation using SAM (segment_anything package)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

PROJECT_ROOT = Path(__file__).parent.parent.parent


class SAMPredictor:
    """Point/box-based mask generation using SAM (segment_anything package).

    Wraps ``sam_model_registry`` and ``SamPredictor`` from the official
    ``segment-anything`` pip package.

    Usage::

        predictor = SAMPredictor(model_type="vit_b")
        predictor.load_image(bgr_image)
        mask = predictor.predict([(x, y, 1)])
        mask = predictor.predict_with_box((x1, y1, x2, y2))
    """

    MODEL_FILES = {
        "vit_b": PROJECT_ROOT / "xlib" / "models" / "sam" / "sam_vit_b_01ec64.pth",
        "vit_l": PROJECT_ROOT / "xlib" / "models" / "sam" / "sam_vit_l_0b3195.pth",
        "vit_h": PROJECT_ROOT / "xlib" / "models" / "sam" / "sam_vit_h_4b8939.pth",
    }

    def __init__(self, model_type: str = "vit_b", device: Optional[str] = None):
        """Load SAM model.

        Args:
            model_type: One of ``"vit_b"``, ``"vit_l"``, ``"vit_h"``.
            device: Torch device string. Auto-detected if *None*.
        """
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = self.MODEL_FILES[model_type]
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self._device = device

    def load_image(self, image: np.ndarray):
        """Set image for mask generation.

        Args:
            image: BGR numpy array (H, W, 3) — the standard OpenCV format.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

    def predict(self, clicks: list[tuple]) -> np.ndarray:
        """Generate mask from point clicks.

        Args:
            clicks: List of ``(x, y, label)`` tuples where ``label=1``
                indicates a foreground point and ``label=0`` indicates a
                background point.

        Returns:
            float32 mask in [0, 1] with shape (H, W).
        """
        point_coords = np.array([(x, y) for x, y, _ in clicks], dtype=np.float32)
        point_labels = np.array([label for _, _, label in clicks], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick the mask with the highest confidence score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)

        # Morphology cleanup: CLOSE then OPEN with 3x3 elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def predict_with_box(self, box: tuple) -> np.ndarray:
        """Generate mask from a bounding box.

        Args:
            box: ``(x1, y1, x2, y2)`` bounding box coordinates.

        Returns:
            float32 mask in [0, 1] with shape (H, W).
        """
        box_np = np.array([box], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box_np,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.float32)

        # Morphology cleanup: CLOSE then OPEN with 3x3 elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask
