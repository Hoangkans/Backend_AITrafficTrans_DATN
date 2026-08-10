import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import Field
from app.schemas.auth import BaseSchema
from app.schemas.detection import DetectionDto

class ViolationDto(BaseSchema):
    id: uuid.UUID
    detection_id: Optional[uuid.UUID] = None
    camera_id: uuid.UUID

    violation_type: str
    vehicle_type: str
    license_plate: Optional[str] = None
    confidence: float
    evidence_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, validation_alias="metadata_")
    status: str = "pending"
    is_confirmed: bool
    confirmed_by: Optional[uuid.UUID] = None

    notes: Optional[str] = None
    created_at: datetime

    # Optional nested details
    detection: Optional[DetectionDto] = None

class ViolationCreate(BaseSchema):
    detection_id: Optional[str] = None
    camera_id: str
    violation_type: str
    vehicle_type: str
    license_plate: Optional[str] = None
    confidence: float
    evidence_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ViolationConfirmRequest(BaseSchema):
    notes: Optional[str] = None
    is_confirmed: Optional[bool] = None
    status: Optional[str] = Field(None, pattern="^(pending|verified|rejected)$")
