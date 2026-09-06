import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.crud.crud_notification import (
    get_notifications,
    count_notifications,
    count_unread_notifications,
    mark_as_read,
    mark_all_as_read,
    delete_notification,
    create_notification
)

router = APIRouter()

def notification_to_dto(notification) -> Dict[str, Any]:
    return {
        "id": str(notification.id),
        "operatorId": str(notification.operator_id) if notification.operator_id else None,
        "title": notification.title,
        "message": notification.message,
        "type": notification.type,
        "referenceId": notification.reference_id,
        "isRead": notification.is_read,
        "createdAt": notification.created_at.isoformat() if notification.created_at else None
    }

@router.get("", response_model=Dict[str, Any])
async def read_notifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, alias="pageSize"),
    unread_only: bool = Query(False, alias="unreadOnly"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    skip = (page - 1) * page_size
    notifications = await get_notifications(
        db,
        operator_id=current_operator.id,
        unread_only=unread_only,
        skip=skip,
        limit=page_size
    )
    total = await count_notifications(db, operator_id=current_operator.id, unread_only=False)
    unread_count = await count_unread_notifications(db, operator_id=current_operator.id)
    
    return {
        "data": [notification_to_dto(n) for n in notifications],
        "unreadCount": unread_count,
        "total": total,
        "page": page,
        "pageSize": page_size
    }

@router.get("/unread-count")
async def get_unread_count(
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    unread_count = await count_unread_notifications(db, operator_id=current_operator.id)
    return {"unreadCount": unread_count}

@router.patch("/{notification_id}/read")
async def mark_notification_read(
    notification_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    n = await mark_as_read(db, notification_id=notification_id, operator_id=current_operator.id)
    if not n:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông báo."
        )
    return notification_to_dto(n)

@router.post("/mark-all-read")
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    updated_count = await mark_all_as_read(db, operator_id=current_operator.id)
    return {"message": "Đã đánh dấu tất cả là đã đọc", "count": updated_count}

@router.delete("/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_notification(
    notification_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    success = await delete_notification(db, notification_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông báo."
        )
    return None
