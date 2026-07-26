import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies.auth import get_current_operator, require_admin
from app.models.operator import Operator
from app.schemas.user import UserDto, CreateUserRequest, UpdateUserRequest
from app.crud.crud_operator import (
    get_operator,
    get_operator_by_username,
    get_operator_by_email,
    get_operators,
    count_operators,
    create_operator,
    update_operator,
    delete_operator
)
from app.services.email_service import send_admin_created_user_email

router = APIRouter()


def user_to_dto(user) -> Dict[str, Any]:
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "fullName": user.full_name,
        "role": user.role,
        "isActive": user.is_active,
        "isEmailVerified": user.is_email_verified,
        "emailVerifiedAt": user.email_verified_at.isoformat() if user.email_verified_at else None,
        "createdAt": user.created_at.isoformat() if user.created_at else None,
        "updatedAt": user.updated_at.isoformat() if user.updated_at else None
    }


@router.get("", response_model=Dict[str, Any])
async def read_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    """Lấy danh sách tất cả users — tương đương GET /api/users trong ASP.NET."""
    skip = (page - 1) * page_size
    users = await get_operators(db, skip=skip, limit=page_size)
    total = await count_operators(db)
    return {
        "data": [user_to_dto(user) for user in users],
        "total": total,
        "page": page,
        "pageSize": page_size
    }


@router.get("/{user_id}", response_model=UserDto)
async def read_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    """Lấy chi tiết user — tương đương GET /api/users/{id} trong ASP.NET."""
    user = await get_operator(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng."
        )
    return user_to_dto(user)


@router.post("", response_model=UserDto, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)
):
    """Admin tạo user mới — tương đương POST /api/users trong ASP.NET.
    Khác với /auth/register, endpoint này cho phép admin chỉ định role."""
    # Check if username exists
    existing = await get_operator_by_username(db, user_in.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên đăng nhập đã tồn tại."
        )

    # Check if email exists
    existing_email = await get_operator_by_email(db, user_in.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được sử dụng."
        )

    user = await create_operator(db, user_in, role=user_in.role or "operator", is_email_verified=True)
    user.email_verified_at = datetime.now(timezone.utc)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    send_admin_created_user_email(user, user_in.password)
    return user


@router.put("/{user_id}", response_model=UserDto)
async def update_user(
    user_id: uuid.UUID,
    user_in: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)
):
    """Admin cập nhật user — tương đương PUT /api/users/{id} trong ASP.NET.
    Cho phép thay đổi email, full_name, role, is_active."""
    db_obj = await get_operator(db, user_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng."
        )

    # If changing email, check for duplicates
    if user_in.email and user_in.email != db_obj.email:
        existing_email = await get_operator_by_email(db, user_in.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email đã được sử dụng."
            )

    update_data = user_in.model_dump(exclude_unset=True)
    user = await update_operator(db, db_obj, update_data)
    await db.commit()
    await db.refresh(user)
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)
):
    """Admin xóa user."""
    # Prevent admin from deleting themselves
    if user_id == current_operator.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể tự xóa tài khoản của mình."
        )

    deleted = await delete_operator(db, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng."
        )
    await db.commit()
    return None
