from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.dependencies.auth import get_current_operator, require_admin
from app.models.operator import Operator
from app.models.system_setting import SystemSetting
from app.schemas.settings import (
    NotificationSettingsDto,
    SecuritySettingsDto,
    SettingsDto,
    UpdateNotificationSettingsRequest,
    UpdateSecuritySettingsRequest,
    UpdateSettingsRequest,
)

router = APIRouter()

DEFAULT_SETTINGS_ID = "default"


async def get_or_create_settings(db: AsyncSession) -> SystemSetting:
    db_settings = await db.get(SystemSetting, DEFAULT_SETTINGS_ID)
    if db_settings:
        return db_settings

    db_settings = SystemSetting(
        id=DEFAULT_SETTINGS_ID,
        app_name=settings.APP_NAME,
        detection_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
        max_cameras=100,
        retention_days=90,
        notification_sound_enabled=1,
        email_alerts_enabled=0,
        daily_report_time="23:59",
        two_factor_enabled=0,
    )
    db.add(db_settings)
    await db.flush()
    return db_settings


def to_settings_dto(db_settings: SystemSetting) -> SettingsDto:
    return SettingsDto(
        app_name=db_settings.app_name,
        api_version=settings.APP_VERSION,
        detection_threshold=db_settings.detection_threshold,
        max_cameras=db_settings.max_cameras,
        retention_days=db_settings.retention_days,
        notification_sound_enabled=bool(db_settings.notification_sound_enabled),
        email_alerts_enabled=bool(db_settings.email_alerts_enabled),
        daily_report_time=db_settings.daily_report_time,
        two_factor_enabled=bool(db_settings.two_factor_enabled),
    )


def to_notification_settings_dto(db_settings: SystemSetting) -> NotificationSettingsDto:
    return NotificationSettingsDto(
        notification_sound_enabled=bool(db_settings.notification_sound_enabled),
        email_alerts_enabled=bool(db_settings.email_alerts_enabled),
        daily_report_time=db_settings.daily_report_time,
    )


def to_security_settings_dto(db_settings: SystemSetting) -> SecuritySettingsDto:
    return SecuritySettingsDto(
        two_factor_enabled=bool(db_settings.two_factor_enabled),
    )


@router.get("", response_model=SettingsDto)
async def get_settings(
    current_operator: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    await db.commit()
    return to_settings_dto(db_settings)


@router.put("", response_model=SettingsDto)
async def update_settings(
    request: UpdateSettingsRequest,
    current_operator: Operator = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    update_data = request.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_settings, field, value)

    db.add(db_settings)
    await db.commit()
    await db.refresh(db_settings)
    return to_settings_dto(db_settings)


@router.get("/notifications", response_model=NotificationSettingsDto)
async def get_notification_settings(
    current_operator: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    await db.commit()
    return to_notification_settings_dto(db_settings)


@router.put("/notifications", response_model=NotificationSettingsDto)
async def update_notification_settings(
    request: UpdateNotificationSettingsRequest,
    current_operator: Operator = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    update_data = request.model_dump(exclude_unset=True)
    if "notification_sound_enabled" in update_data:
        db_settings.notification_sound_enabled = int(update_data["notification_sound_enabled"])
    if "email_alerts_enabled" in update_data:
        db_settings.email_alerts_enabled = int(update_data["email_alerts_enabled"])
    if "daily_report_time" in update_data:
        db_settings.daily_report_time = update_data["daily_report_time"]

    db.add(db_settings)
    await db.commit()
    await db.refresh(db_settings)
    return to_notification_settings_dto(db_settings)


@router.get("/security", response_model=SecuritySettingsDto)
async def get_security_settings(
    current_operator: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    await db.commit()
    return to_security_settings_dto(db_settings)


@router.put("/security", response_model=SecuritySettingsDto)
async def update_security_settings(
    request: UpdateSecuritySettingsRequest,
    current_operator: Operator = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    db_settings = await get_or_create_settings(db)
    update_data = request.model_dump(exclude_unset=True)
    if "two_factor_enabled" in update_data:
        db_settings.two_factor_enabled = int(update_data["two_factor_enabled"])

    db.add(db_settings)
    await db.commit()
    await db.refresh(db_settings)
    return to_security_settings_dto(db_settings)
