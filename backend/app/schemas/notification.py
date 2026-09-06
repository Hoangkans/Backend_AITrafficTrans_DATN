import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class NotificationDto(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    operatorId: Optional[uuid.UUID] = None
    title: str
    message: str
    type: str
    referenceId: Optional[str] = None
    isRead: bool
    createdAt: datetime

class NotificationCreateRequest(BaseModel):
    operator_id: Optional[uuid.UUID] = None
    title: str
    message: str
    type: str = "system"
    reference_id: Optional[str] = None

class NotificationListResponse(BaseModel):
    data: List[NotificationDto]
    unreadCount: int
    total: int
    page: int
    pageSize: int
