import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.email_verification_token import EmailVerificationToken


def hash_verification_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_email_verification_token(
    db: AsyncSession,
    operator_id: uuid.UUID
) -> Tuple[EmailVerificationToken, str]:
    otp = f"{secrets.randbelow(1_000_000):06d}"
    token_hash = hash_verification_token(otp)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS)

    db_token = EmailVerificationToken(
        operator_id=operator_id,
        token_hash=token_hash,
        expires_at=expires_at
    )
    db.add(db_token)
    await db.flush()
    return db_token, otp


async def get_valid_email_verification_token(
    db: AsyncSession,
    token: str,
    operator_id: Optional[uuid.UUID] = None
) -> Optional[EmailVerificationToken]:
    token_hash = hash_verification_token(token)
    now = datetime.now(timezone.utc)
    query = (
        select(EmailVerificationToken)
        .filter(EmailVerificationToken.token_hash == token_hash)
        .filter(EmailVerificationToken.used_at.is_(None))
        .filter(EmailVerificationToken.expires_at > now)
    )
    if operator_id:
        query = query.filter(EmailVerificationToken.operator_id == operator_id)
    result = await db.execute(query)
    return result.scalars().first()


async def revoke_active_email_verification_tokens(
    db: AsyncSession,
    operator_id: uuid.UUID
) -> None:
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(EmailVerificationToken)
        .filter(EmailVerificationToken.operator_id == operator_id)
        .filter(EmailVerificationToken.used_at.is_(None))
    )
    for token in result.scalars().all():
        token.used_at = now
        db.add(token)
    await db.flush()
