"""Lightweight text-to-mask using OWL-ViT (HuggingFace Transformers) + SAM.

A lighter alternative to GroundedSAM2 that requires no C++ CUDA compilation
and instead uses the standard ``transformers`` package for OWL-ViT zero-shot
object detection, then segments each detected box with SAM.

Typical usage::

    detector = OWLViTDetector()
    img = cv2.imread("photo.jpg")
    detector.load_image(img)
    mask = detector.predict("face")
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class OWLViTDetector:
    """Text-prompted mask generation using OWL-ViT detection + SAM segmentation.

    Parameters
    ----------
    detection_model:
        HuggingFace model ID for OWL-ViT.
    sam_model_type:
        SAM model variant passed to ``ModelLoader.get_sam()``
        (``"vit_b"``, ``"vit_l"``, or ``"vit_h"``).
    device:
        Torch device string. Auto-detected when *None*.
    box_threshold:
        Minimum box confidence for OWL-ViT.
    """

    def __init__(
        self,
        detection_model: str = "google/owlvit-base-patch32",
        sam_model_type: str = "vit_l",
        device: Optional[str] = None,
        box_threshold: float = 0.3,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.box_threshold = box_threshold

        # ---- OWL-ViT (HuggingFace) -----------------------------------------
        self.processor = AutoProcessor.from_pretrained(detection_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detection_model
        ).to(self.device)

        # ---- SAM predictor via ModelLoader ---------------------------------
        from MaskProcessor.core.model_loader import ModelLoader

        self.sam = ModelLoader.get_sam(sam_model_type)

        # Cached image data (set by :meth:`load_image`)
        self._image_bgr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image: np.ndarray):
        """Set image for mask generation.

        Loads the image into both OWL-ViT (via ``processor``) and SAM.

        Args:
            image: BGR numpy array *(H, W, 3)* -- standard OpenCV format.
        """
        self._image_bgr = image
        self.sam.load_image(image)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text_prompt: str) -> Optional[np.ndarray]:
        """Generate a segmentation mask from a text prompt.

        Pipeline:

        1. **OWL-ViT** detects bounding boxes matching *text_prompt*.
        2. Each box is passed to **SAM** for per-box segmentation.
        3. All per-box masks are merged via element-wise maximum.

        Args:
            text_prompt:
                Text description of the object to segment
                (e.g. ``"face"``, ``"hair"``, ``"glasses"``).

        Returns:
            Float32 mask in **[0, 1]** with shape *(H, W)*, or *None*
            when no objects are detected.
        """
        if self._image_bgr is None:
            raise RuntimeError("Call load_image() before predict().")

        # ---- Step 1: OWL-ViT detection -------------------------------------
        inputs = self.processor(
            text=[text_prompt], images=self._image_bgr, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.detector(**inputs)

        target_sizes = torch.tensor(
            [self._image_bgr.shape[:2][::-1]], device=self.device
        )
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.box_threshold, target_sizes=target_sizes
        )

        boxes = results[0]["boxes"].cpu().numpy()
        if len(boxes) == 0:
            return None

        # ---- Step 2: SAM mask generation per box ---------------------------
        masks: list[np.ndarray] = []
        for box in boxes:
            # box is (x1, y1, x2, y2) as expected by SAMPredictor.predict_with_box
            mask = self.sam.predict_with_box(tuple(box))
            if mask is not None:
                masks.append(mask)

        if not masks:
            return None

        # ---- Step 3: Merge masks -------------------------------------------
        from MaskProcessor.core.mask_ops import merge_masks

        return merge_masks(masks, "union")
