from pydantic import BaseModel
from typing import Optional, List


class OpenProjectRequest(BaseModel):
    path: str


class OpenProjectResponse(BaseModel):
    status: str
    video_path: Optional[str] = None
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    models: List[dict] = []
    videos: List[str] = []
    has_aligned: bool = False
    aligned_count: int = 0
    message: Optional[str] = None
    video_dfl_map: dict = {}  # {video_name: is_dfl}


class AnalyzeRequest(BaseModel):
    frame_idx: int
    config: dict
    detector: str = "YOLOv8"
    landmarker: str = "insightface-2d106det"
    res_scale: float = 0.5
    selected_faces: list = []  # empty = merge all detected, otherwise indices to merge
    face_model_map: dict = {}  # {face_idx: model_name} for per-face model assignment
    face_margin: float = 0.4  # expansion ratio for landmarker crop rect (0.0-2.0)
    angle_segments: list = []  # [{start, end, angles}, ...] detection angle regions


class ReCompositeRequest(BaseModel):
    frame_idx: int
    config: dict


class CutSegment(BaseModel):
    action: str  # "add", "remove", "update"
    start_frame: int = 0
    end_frame: int = 0


class ExportStartRequest(BaseModel):
    config: dict
    format: str = "mp4"
    quality: str = "high"
    cut_segments: List[dict] = []


class ModelLoadRequest(BaseModel):
    name: str
