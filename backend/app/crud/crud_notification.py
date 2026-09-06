import uuid
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, delete, or_
from app.models.notification import Notification

async def get_notifications(
    db: AsyncSession,
    operator_id: Optional[uuid.UUID] = None,
    unread_only: bool = False,
    skip: int = 0,
    limit: int = 20
) -> List[Notification]:
    query = select(Notification)
    conditions = []
    
    if operator_id:
        conditions.append(or_(Notification.operator_id == operator_id, Notification.operator_id.is_(None)))
    if unread_only:
        conditions.append(Notification.is_read == False)
        
    if conditions:
        query = query.where(*conditions)
        
    query = query.order_by(Notification.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

async def count_notifications(
    db: AsyncSession,
    operator_id: Optional[uuid.UUID] = None,
    unread_only: bool = False
) -> int:
    query = select(func.count(Notification.id))
    conditions = []
    
    if operator_id:
        conditions.append(or_(Notification.operator_id == operator_id, Notification.operator_id.is_(None)))
    if unread_only:
        conditions.append(Notification.is_read == False)
        
    if conditions:
        query = query.where(*conditions)
        
    result = await db.execute(query)
    return result.scalar() or 0

async def count_unread_notifications(
    db: AsyncSession,
    operator_id: Optional[uuid.UUID] = None
) -> int:
    return await count_notifications(db, operator_id=operator_id, unread_only=True)

async def create_notification(
    db: AsyncSession,
    title: str,
    message: str,
    type: str = "system",
    operator_id: Optional[uuid.UUID] = None,
    reference_id: Optional[str] = None
) -> Notification:
    notification = Notification(
        title=title,
        message=message,
        type=type,
        operator_id=operator_id,
        reference_id=reference_id,
        is_read=False
    )
    db.add(notification)
    await db.commit()
    await db.refresh(notification)
    return notification

async def mark_as_read(
    db: AsyncSession,
    notification_id: uuid.UUID,
    operator_id: Optional[uuid.UUID] = None
) -> Optional[Notification]:
    query = select(Notification).where(Notification.id == notification_id)
    result = await db.execute(query)
    notification = result.scalar_one_or_none()
    if notification:
        notification.is_read = True
        await db.commit()
        await db.refresh(notification)
    return notification

async def mark_all_as_read(
    db: AsyncSession,
    operator_id: Optional[uuid.UUID] = None
) -> int:
    stmt = update(Notification).values(is_read=True)
    if operator_id:
        stmt = stmt.where(or_(Notification.operator_id == operator_id, Notification.operator_id.is_(None)))
    stmt = stmt.where(Notification.is_read == False)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount

async def delete_notification(
    db: AsyncSession,
    notification_id: uuid.UUID
) -> bool:
    stmt = delete(Notification).where(Notification.id == notification_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0
