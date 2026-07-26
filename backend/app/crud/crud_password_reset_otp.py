import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.password_reset_otp import PasswordResetOtp


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


async def create_password_reset_otp(
    db: AsyncSession,
    operator_id: uuid.UUID
) -> Tuple[PasswordResetOtp, str]:
    otp = f"{secrets.randbelow(1_000_000):06d}"
    otp_hash = hash_otp(otp)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_OTP_EXPIRE_MINUTES)

    db_otp = PasswordResetOtp(
        operator_id=operator_id,
        otp_hash=otp_hash,
        expires_at=expires_at
    )
    db.add(db_otp)
    await db.flush()
    return db_otp, otp


async def get_valid_password_reset_otp(
    db: AsyncSession,
    operator_id: uuid.UUID,
    otp: str
) -> Optional[PasswordResetOtp]:
    otp_hash = hash_otp(otp)
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(PasswordResetOtp)
        .filter(PasswordResetOtp.operator_id == operator_id)
        .filter(PasswordResetOtp.used_at.is_(None))
        .filter(PasswordResetOtp.expires_at > now)
        .order_by(PasswordResetOtp.created_at.desc())
    )

    for db_otp in result.scalars().all():
        if hmac.compare_digest(db_otp.otp_hash, otp_hash):
            return db_otp
    return None


async def revoke_active_password_reset_otps(
    db: AsyncSession,
    operator_id: uuid.UUID
) -> None:
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(PasswordResetOtp)
        .filter(PasswordResetOtp.operator_id == operator_id)
        .filter(PasswordResetOtp.used_at.is_(None))
    )
    for db_otp in result.scalars().all():
        db_otp.used_at = now
        db.add(db_otp)
    await db.flush()
