import uuid
from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.camera import Camera
from app.schemas.camera import CreateCameraRequest, UpdateCameraRequest

async def get_camera(db: AsyncSession, camera_id: uuid.UUID) -> Optional[Camera]:
    result = await db.execute(select(Camera).filter(Camera.id == camera_id))
    return result.scalars().first()

async def get_cameras(
    db: AsyncSession, 
    status: Optional[str] = None, 
    intersection: Optional[str] = None, 
    skip: int = 0, 
    limit: int = 100
) -> List[Camera]:
    query = select(Camera)
    if status:
        query = query.filter(Camera.status == status)
    if intersection:
        query = query.filter(Camera.intersection.ilike(f"%{intersection}%"))
    
    # Order by name
    query = query.order_by(Camera.name).offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())

async def count_cameras(
    db: AsyncSession,
    status: Optional[str] = None,
    intersection: Optional[str] = None
) -> int:
    from sqlalchemy import func
    query = select(func.count(Camera.id))
    if status:
        query = query.filter(Camera.status == status)
    if intersection:
        query = query.filter(Camera.intersection.ilike(f"%{intersection}%"))
    result = await db.execute(query)
    return result.scalar() or 0

async def create_camera(db: AsyncSession, obj_in: CreateCameraRequest) -> Camera:
    db_obj = Camera(
        name=obj_in.name,
        rtsp_url=obj_in.rtsp_url,
        latitude=obj_in.latitude,
        longitude=obj_in.longitude,
        address=obj_in.address,
        intersection=obj_in.intersection,
        direction=obj_in.direction,
        status=obj_in.status or "active",
        config=obj_in.config or {},
        vehicle_types=obj_in.vehicle_types or []
    )
    db.add(db_obj)
    await db.flush()
    return db_obj

async def update_camera(db: AsyncSession, db_obj: Camera, obj_in: UpdateCameraRequest) -> Camera:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field in update_data:
        setattr(db_obj, field, update_data[field])
    db.add(db_obj)
    await db.flush()
    return db_obj

async def delete_camera(db: AsyncSession, camera_id: uuid.UUID) -> bool:
    db_obj = await get_camera(db, camera_id)
    if not db_obj:
        return False
    await db.delete(db_obj)
    await db.flush()
    return True
