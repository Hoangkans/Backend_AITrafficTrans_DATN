import uuid
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.operator import Operator
from app.models.refresh_token import RefreshToken
from app.schemas.auth import RegisterRequest
from app.core.security import get_password_hash

async def get_operator(db: AsyncSession, operator_id: uuid.UUID) -> Optional[Operator]:
    result = await db.execute(select(Operator).filter(Operator.id == operator_id))
    return result.scalars().first()

async def get_operator_by_username(db: AsyncSession, username: str) -> Optional[Operator]:
    result = await db.execute(select(Operator).filter(Operator.username == username))
    return result.scalars().first()

async def get_operator_by_email(db: AsyncSession, email: str) -> Optional[Operator]:
    result = await db.execute(select(Operator).filter(Operator.email == email))
    return result.scalars().first()

async def get_operators(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Operator]:
    result = await db.execute(select(Operator).offset(skip).limit(limit))
    return list(result.scalars().all())

async def create_operator(
    db: AsyncSession,
    obj_in: RegisterRequest,
    role: str = "operator",
    is_email_verified: bool = False
) -> Operator:
    hashed_password = get_password_hash(obj_in.password)
    db_obj = Operator(
        username=obj_in.username,
        email=obj_in.email,
        hashed_password=hashed_password,
        full_name=obj_in.full_name,
        role=role,
        is_active=True,
        is_email_verified=is_email_verified
    )
    db.add(db_obj)
    await db.flush()  # Populates DB fields like ID without committing transaction
    return db_obj

async def count_operators(db: AsyncSession) -> int:
    from sqlalchemy import func
    result = await db.execute(select(func.count(Operator.id)))
    return result.scalar() or 0

async def update_operator_password(db: AsyncSession, db_obj: Operator, password: str) -> Operator:
    db_obj.hashed_password = get_password_hash(password)
    db.add(db_obj)
    await db.flush()
    return db_obj

async def update_operator(db: AsyncSession, db_obj: Operator, update_data: dict) -> Operator:
    """Update operator fields (email, full_name, role, is_active)."""
    for field, value in update_data.items():
        if value is not None and hasattr(db_obj, field):
            setattr(db_obj, field, value)
    db.add(db_obj)
    await db.flush()
    return db_obj

async def delete_operator(db: AsyncSession, operator_id: uuid.UUID) -> bool:
    """Delete an operator and their refresh tokens."""
    # Delete refresh tokens first
    result = await db.execute(select(RefreshToken).filter(RefreshToken.operator_id == operator_id))
    tokens = result.scalars().all()
    for token in tokens:
        await db.delete(token)

    result = await db.execute(select(Operator).filter(Operator.id == operator_id))
    db_obj = result.scalars().first()
    if not db_obj:
        return False
    await db.delete(db_obj)
    await db.flush()
    return True
