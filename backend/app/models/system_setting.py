from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String
from app.core.database import Base


class SystemSetting(Base):
    __tablename__ = "system_settings"

    id = Column(String(50), primary_key=True, default="default")
    app_name = Column(String(255), nullable=False)
    detection_threshold = Column(Float, nullable=False)
    max_cameras = Column(Integer, nullable=False)
    retention_days = Column(Integer, nullable=False)
    notification_sound_enabled = Column(Integer, nullable=False, default=1)
    email_alerts_enabled = Column(Integer, nullable=False, default=0)
    daily_report_time = Column(String(5), nullable=False, default="23:59")
    two_factor_enabled = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=True, onupdate=lambda: datetime.now(timezone.utc))
