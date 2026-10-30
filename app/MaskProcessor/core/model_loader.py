"""Unified lazy model loading. Each model type is initialized once on first use."""

import threading
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent  # project root


# Sentinel: distinct from None to distinguish "not loaded" from "failed"
_FAILED = object()


class ModelLoader:
    """Thread-safe lazy model loader. Usage: ModelLoader.get_sam()"""

    _sam = None
    _sam_lock = threading.Lock()
    _grounded_sam2 = None
    _grounded_sam2_lock = threading.Lock()
    _bisenet = None
    _bisenet_lock = threading.Lock()
    _device: Optional[str] = None

    @classmethod
    def set_device(cls, device: str):
        cls._device = device

    @classmethod
    def _get_device(cls) -> str:
        if cls._device:
            return cls._device
        import torch

        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._device

    @classmethod
    def get_sam(cls, model_type: str = "vit_b") -> "SAMPredictor":
        if cls._sam is None:
            with cls._sam_lock:
                if cls._sam is None:
                    from MaskProcessor.core.sam_predictor import SAMPredictor

                    cls._sam = SAMPredictor(
                        model_type=model_type, device=cls._get_device()
                    )
        return cls._sam

    @classmethod
    def get_grounded_sam2(
        cls, model_name: str = "sam2.1_hiera_small"
    ) -> "GroundedSAM2":
        if cls._grounded_sam2 is None:
            with cls._grounded_sam2_lock:
                if cls._grounded_sam2 is None:
                    try:
                        from MaskProcessor.core.grounded_sam2 import GroundedSAM2

                        cls._grounded_sam2 = GroundedSAM2(
                            sam2_model=model_name, device=cls._get_device()
                        )
                    except Exception as ex:
                        cls._grounded_sam2 = _FAILED  # cache failure
                        raise  # still propagate so caller can log/handle
        if cls._grounded_sam2 is _FAILED:
            return None
        return cls._grounded_sam2

    @classmethod
    def get_bisenet(cls) -> "BiSeNetParser":
        if cls._bisenet is None:
            with cls._bisenet_lock:
                if cls._bisenet is None:
                    from MaskProcessor.core.bisenet_parser import BiSeNetParser

                    cls._bisenet = BiSeNetParser(device=cls._get_device())
        return cls._bisenet
