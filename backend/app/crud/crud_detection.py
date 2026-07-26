import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.detection import Detection
from app.schemas.detection import DetectionCreate

async def get_detection(db: AsyncSession, detection_id: uuid.UUID) -> Optional[Detection]:
    result = await db.execute(select(Detection).filter(Detection.id == detection_id))
    return result.scalars().first()

async def get_detections(
    db: AsyncSession,
    camera_id: Optional[uuid.UUID] = None,
    vehicle_type: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    min_confidence: Optional[float] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Detection]:
    query = select(Detection)
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    if vehicle_type:
        query = query.filter(Detection.vehicle_type == vehicle_type)
    if from_date:
        query = query.filter(Detection.detected_at >= from_date)
    if to_date:
        query = query.filter(Detection.detected_at <= to_date)
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)
        
    query = query.order_by(Detection.detected_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())

async def count_detections(
    db: AsyncSession,
    camera_id: Optional[uuid.UUID] = None,
    vehicle_type: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    min_confidence: Optional[float] = None
) -> int:
    from sqlalchemy import func
    query = select(func.count(Detection.id))
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    if vehicle_type:
        query = query.filter(Detection.vehicle_type == vehicle_type)
    if from_date:
        query = query.filter(Detection.detected_at >= from_date)
    if to_date:
        query = query.filter(Detection.detected_at <= to_date)
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)
    result = await db.execute(query)
    return result.scalar() or 0

async def create_detection(db: AsyncSession, obj_in: DetectionCreate) -> Detection:
    db_obj = Detection(
        camera_id=uuid.UUID(obj_in.camera_id) if isinstance(obj_in.camera_id, str) else obj_in.camera_id,
        frame_id=obj_in.frame_id,
        vehicle_type=obj_in.vehicle_type,
        confidence=obj_in.confidence,
        bbox=obj_in.bbox.model_dump(),
        metadata_=obj_in.metadata or {}
    )
    db.add(db_obj)
    await db.flush()
    return db_obj
