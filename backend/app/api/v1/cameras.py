import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.dependencies.auth import get_current_operator, require_admin
from app.models.operator import Operator
from app.schemas.camera import CameraDto, CreateCameraRequest, UpdateCameraRequest
from app.crud.crud_camera import (
    get_camera,
    get_cameras,
    count_cameras,
    create_camera,
    update_camera,
    delete_camera
)

router = APIRouter()


def camera_to_dto(camera) -> Dict[str, Any]:
    return {
        "id": str(camera.id),
        "name": camera.name,
        "rtspUrl": camera.rtsp_url,
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "address": camera.address,
        "intersection": camera.intersection,
        "direction": camera.direction,
        "status": camera.status,
        "config": camera.config or {},
        "vehicleTypes": camera.vehicle_types or [],
        "createdAt": camera.created_at.isoformat() if camera.created_at else None,
        "updatedAt": camera.updated_at.isoformat() if camera.updated_at else None
    }


@router.get("", response_model=Dict[str, Any])
async def read_cameras(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
    status: Optional[str] = None,
    intersection: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    skip = (page - 1) * page_size
    cameras = await get_cameras(db, status=status, intersection=intersection, skip=skip, limit=page_size)
    total = await count_cameras(db, status=status, intersection=intersection)
    return {
        "data": [camera_to_dto(camera) for camera in cameras],
        "total": total,
        "page": page,
        "pageSize": page_size
    }

@router.get("/{camera_id}", response_model=CameraDto)
async def read_camera(
    camera_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    camera = await get_camera(db, camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy camera."
        )
    return camera_to_dto(camera)

@router.post("", response_model=CameraDto, status_code=status.HTTP_201_CREATED)
async def add_camera(
    camera_in: CreateCameraRequest,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)  # Only admin can create
):
    camera = await create_camera(db, camera_in)
    await db.commit()
    await db.refresh(camera)
    return camera

@router.put("/{camera_id}", response_model=CameraDto)
async def modify_camera(
    camera_id: uuid.UUID,
    camera_in: UpdateCameraRequest,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)  # Only admin can edit
):
    db_obj = await get_camera(db, camera_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy camera."
        )
    camera = await update_camera(db, db_obj, camera_in)
    await db.commit()
    await db.refresh(camera)
    return camera

@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_camera(
    camera_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(require_admin)  # Only admin can delete
):
    deleted = await delete_camera(db, camera_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy camera."
        )
    await db.commit()
    return None
