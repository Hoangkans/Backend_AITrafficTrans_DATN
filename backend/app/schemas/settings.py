from typing import Optional
from pydantic import AliasChoices, Field

from app.schemas.auth import BaseSchema


class SettingsDto(BaseSchema):
    """System settings response — maps from ASP.NET SettingsDto."""
    app_name: str
    api_version: str
    detection_threshold: float
    max_cameras: int
    retention_days: int
    notification_sound_enabled: bool
    email_alerts_enabled: bool
    daily_report_time: str
    two_factor_enabled: bool


class UpdateSettingsRequest(BaseSchema):
    """Admin updates system settings — maps from ASP.NET UpdateSettingsRequest."""
    app_name: Optional[str] = None
    detection_threshold: Optional[float] = None
    max_cameras: Optional[int] = None
    retention_days: Optional[int] = None


class NotificationSettingsDto(BaseSchema):
    notification_sound_enabled: bool = Field(serialization_alias="soundEnabled")
    email_alerts_enabled: bool
    daily_report_time: str


class UpdateNotificationSettingsRequest(BaseSchema):
    notification_sound_enabled: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices("soundEnabled", "notificationSoundEnabled", "notification_sound_enabled"),
    )
    email_alerts_enabled: Optional[bool] = None
    daily_report_time: Optional[str] = None


class SecuritySettingsDto(BaseSchema):
    two_factor_enabled: bool


class UpdateSecuritySettingsRequest(BaseSchema):
    two_factor_enabled: Optional[bool] = None
