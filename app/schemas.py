from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

class DetectResponse(BaseModel):
    boxes: List[Box]

class IdentityOut(BaseModel):
    id: int
    name: str
    score: float
    image_path: Optional[str]

class RecognizeResponse(BaseModel):
    matches: List[IdentityOut]
