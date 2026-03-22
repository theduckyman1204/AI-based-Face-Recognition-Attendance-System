from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class RecognitionResult(BaseModel):
    name: str = Field(..., example="John Doe")
    confidence: Optional[float] = Field(None, example=0.91)
    appeared_at: datetime
    bbox: Optional[DetectionBox] = None


class ImageIn(BaseModel):
    source_name: Optional[str] = None


class VideoIn(BaseModel):
    source_name: Optional[str] = None


class DetectionResponse(BaseModel):
    results: List[RecognitionResult]


class AppearanceSearchResponse(BaseModel):
    names: List[str]
    events: List[RecognitionResult]
