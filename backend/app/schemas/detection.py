import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import Field
from app.schemas.auth import BaseSchema

class BoundingBoxDto(BaseSchema):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionDto(BaseSchema):
    id: uuid.UUID
    camera_id: uuid.UUID

    frame_id: int
    vehicle_type: str
    confidence: float
    bbox: BoundingBoxDto
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    detected_at: datetime

class DetectionCreate(BaseSchema):
    camera_id: str
    frame_id: int
    vehicle_type: str
    confidence: float
    bbox: BoundingBoxDto
    metadata: Optional[Dict[str, Any]] = None
