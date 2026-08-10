import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base

class Violation(Base):
    __tablename__ = "violations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id", ondelete="SET NULL"), nullable=True)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    violation_type = Column(String(50), nullable=False, index=True)  # 'red_light', 'speeding', 'wrong_lane', 'no_helmet'
    vehicle_type = Column(String(50), nullable=False)
    license_plate = Column(String(20), nullable=True, index=True)
    confidence = Column(Float, nullable=False)
    evidence_url = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)  # {"speed_kmh": 85, "speed_limit": 60}
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending, verified, rejected
    is_confirmed = Column(Boolean, nullable=False, default=False, index=True)
    confirmed_by = Column(UUID(as_uuid=True), ForeignKey("operators.id", ondelete="SET NULL"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    camera = relationship("Camera", backref="violations")
    detection = relationship("Detection", backref="violations")
    operator = relationship("Operator", backref="confirmed_violations")
