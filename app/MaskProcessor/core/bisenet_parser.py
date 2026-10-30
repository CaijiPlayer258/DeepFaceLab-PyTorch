"""Face parsing using BiSeNetV2. Returns masks for face, hair, eyes, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent


class BiSeNetParser:
    """Face parsing using BiSeNetV2. Returns masks for face, hair, eyes, etc.

    The underlying ``BiSeNetFaceParser`` (from ``xlib/models/bisenet/BiSeNet.py``)
    is lazy-loaded on the first call to :meth:`parse`.

    Usage::

        parser = BiSeNetParser()
        masks = parser.parse(bgr_image)  # {group_name: float32_mask}
    """

    PART_NAMES = {
        0: "background",
        1: "skin",
        2: "nose",
        3: "eye_glass",
        4: "left_eye",
        5: "right_eye",
        6: "left_brow",
        7: "right_brow",
        8: "left_ear",
        9: "right_ear",
        10: "mouth",
        11: "upper_lip",
        12: "lower_lip",
        13: "hair",
        14: "hat",
        15: "ear_ring",
        16: "necklace",
        17: "neck",
        18: "cloth",
    }

    # Indices correspond to BiSeNetFaceParser.CLASS_NAMES:
    #   0: background,   1: skin,    2: l_brow,   3: r_brow,
    #   4: l_eye,        5: r_eye,   6: eye_g,    7: l_ear,
    #   8: r_ear,        9: ear_r,  10: nose,    11: mouth,
    #  12: u_lip,       13: l_lip,  14: neck,    15: neck_l,
    #  16: cloth,       17: hair,   18: hat
    GROUPS = {
        "face": [1, 2, 3, 4, 5, 10, 11, 12, 13],
        "hair": [17],
        "skin": [1],
        "eyes": [4, 5],
        "mouth": [11, 12, 13],
        "nose": [10],
        "brows": [2, 3],
    }

    def __init__(self, device: Optional[str] = None):
        """Initialize the parser.

        Args:
            device: Torch device string (``"cuda"``, ``"cpu"``). Auto-detected
                when *None*.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the BiSeNet model from ``xlib/models/bisenet/``."""
        import importlib.util

        bisenet_path = PROJECT_ROOT / "xlib" / "models" / "bisenet" / "BiSeNet.py"
        model_path = PROJECT_ROOT / "xlib" / "models" / "bisenet" / "model_final.pth"

        spec = importlib.util.spec_from_file_location("BiSeNet", str(bisenet_path))
        bisenet_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bisenet_mod)
        self._model = bisenet_mod.BiSeNetFaceParser(
            str(model_path), device=self.device
        )

    def parse(self, image: np.ndarray) -> dict:
        """Parse a face image and return a mask for each semantic group.

        Args:
            image: BGR numpy array of shape ``(H, W, 3)``.

        Returns:
            Dictionary mapping group names (``"face"``, ``"hair"``, ``"skin"``,
            ``"eyes"``, ``"mouth"``, ``"nose"``, ``"brows"``) to float32 masks
            in ``[0, 1]`` with shape ``(H, W)``.
        """
        self._load_model()

        # Get the raw class-index map from the underlying model internals.
        # BiSeNetFaceParser.parse() returns a dict of per-class masks; here we
        # access the argmax output directly for the class-index map that the
        # GROUPS indices refer to.
        original_h, original_w = image.shape[:2]
        input_tensor = self._model._preprocess(image)
        with torch.no_grad():
            output = self._model.model(input_tensor)[0]
            parsing_map = output.argmax(dim=1).squeeze(0).cpu().numpy()

        if parsing_map.shape != (original_h, original_w):
            parsing_map = cv2.resize(
                parsing_map.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

        masks = {}
        for group_name, part_indices in self.GROUPS.items():
            mask = np.zeros(parsing_map.shape, dtype=np.float32)
            for idx in part_indices:
                mask = np.maximum(mask, (parsing_map == idx).astype(np.float32))
            masks[group_name] = mask

        return masks
