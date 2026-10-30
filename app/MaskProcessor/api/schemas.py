from pydantic import BaseModel
from typing import Optional


class ProjectOpenRequest(BaseModel):
    path: str


class ProjectOpenResponse(BaseModel):
    files: list[dict]
    count: int


class ImageResponse(BaseModel):
    image: str  # base64
    width: int
    height: int
    has_mask: bool


class MaskPredictRequest(BaseModel):
    image_index: int
    clicks: Optional[list[list[int]]] = None  # [[x, y, label], ...]
    box: Optional[list[int]] = None  # [x1, y1, x2, y2]


class MaskTextRequest(BaseModel):
    image_index: int
    text: str
    backend: str = "grounded_sam2"  # "grounded_sam2" or "owl_vit"


class MaskBiSeNetRequest(BaseModel):
    image_index: int
    parts: list[str]  # ["face", "hair", ...]


class MaskSaveRequest(BaseModel):
    image_index: int
    mask: str  # base64 encoded PNG


class MaskResponse(BaseModel):
    mask: str  # base64 encoded
    success: bool = True


class UndoRequest(BaseModel):
    image_index: int


class ErrorResponse(BaseModel):
    error: str
    success: bool = False
