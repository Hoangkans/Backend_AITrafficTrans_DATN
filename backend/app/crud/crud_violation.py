import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.violation import Violation
from app.schemas.violation import ViolationCreate

async def get_violation(db: AsyncSession, violation_id: uuid.UUID) -> Optional[Violation]:
    result = await db.execute(select(Violation).filter(Violation.id == violation_id))
    return result.scalars().first()

async def get_violations(
    db: AsyncSession,
    camera_id: Optional[uuid.UUID] = None,
    violation_type: Optional[str] = None,
    is_confirmed: Optional[bool] = None,
    status: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Violation]:
    query = select(Violation)
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if is_confirmed is not None:
        query = query.filter(Violation.is_confirmed == is_confirmed)
    if status:
        query = query.filter(Violation.status == status)
    if from_date:
        query = query.filter(Violation.created_at >= from_date)
    if to_date:
        query = query.filter(Violation.created_at <= to_date)
        
    query = query.order_by(Violation.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())

async def count_violations(
    db: AsyncSession,
    camera_id: Optional[uuid.UUID] = None,
    violation_type: Optional[str] = None,
    is_confirmed: Optional[bool] = None,
    status: Optional[str] = None
) -> int:
    from sqlalchemy import func
    query = select(func.count(Violation.id))
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if is_confirmed is not None:
        query = query.filter(Violation.is_confirmed == is_confirmed)
    if status:
        query = query.filter(Violation.status == status)
    result = await db.execute(query)
    return result.scalar() or 0

async def create_violation(db: AsyncSession, obj_in: ViolationCreate) -> Violation:
    db_obj = Violation(
        detection_id=uuid.UUID(obj_in.detection_id) if isinstance(obj_in.detection_id, str) else obj_in.detection_id,
        camera_id=uuid.UUID(obj_in.camera_id) if isinstance(obj_in.camera_id, str) else obj_in.camera_id,
        violation_type=obj_in.violation_type,
        vehicle_type=obj_in.vehicle_type,
        license_plate=obj_in.license_plate,
        confidence=obj_in.confidence,
        evidence_url=obj_in.evidence_url,
        metadata_=obj_in.metadata or {},
        status="pending",
        is_confirmed=False
    )
    db.add(db_obj)
    await db.flush()
    return db_obj

async def confirm_violation(
    db: AsyncSession, 
    db_obj: Violation, 
    operator_id: uuid.UUID, 
    notes: Optional[str] = None, 
    is_confirmed: Optional[bool] = None,
    review_status: Optional[str] = None
) -> Violation:
    if review_status:
        db_obj.status = review_status
        db_obj.is_confirmed = review_status == "verified"
    elif is_confirmed is not None:
        db_obj.is_confirmed = is_confirmed
        db_obj.status = "verified" if is_confirmed else "rejected"
    db_obj.confirmed_by = operator_id
    if notes is not None:
        db_obj.notes = notes
    db.add(db_obj)
    await db.flush()
    return db_obj

async def search_violations_by_license_plate(db: AsyncSession, q: str, skip: int = 0, limit: int = 100) -> List[Violation]:
    # Basic search on license plate or notes
    result = await db.execute(
        select(Violation)
        .filter(
            or_(
                Violation.license_plate.ilike(f"%{q}%"),
                Violation.notes.ilike(f"%{q}%")
            )
        )
        .order_by(Violation.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())
