import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.schemas.violation import ViolationDto, ViolationConfirmRequest
from app.crud.crud_violation import (
    get_violation,
    get_violations,
    count_violations,
    confirm_violation,
    search_violations_by_license_plate
)

router = APIRouter()


def violation_to_dto(violation) -> Dict[str, Any]:
    return {
        "id": str(violation.id),
        "detectionId": str(violation.detection_id) if violation.detection_id else None,
        "cameraId": str(violation.camera_id),
        "violationType": violation.violation_type,
        "vehicleType": violation.vehicle_type,
        "licensePlate": violation.license_plate,
        "confidence": violation.confidence,
        "evidenceUrl": violation.evidence_url,
        "metadata": violation.metadata_ or {},
        "status": violation.status,
        "isConfirmed": violation.is_confirmed,
        "confirmedBy": str(violation.confirmed_by) if violation.confirmed_by else None,
        "notes": violation.notes,
        "createdAt": violation.created_at.isoformat() if violation.created_at else None,
        "detection": None
    }


@router.get("", response_model=Dict[str, Any])
async def read_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
    camera_id: Optional[uuid.UUID] = Query(None, alias="cameraId"),
    violation_type: Optional[str] = Query(None, alias="violationType"),
    is_confirmed: Optional[bool] = Query(None, alias="isConfirmed"),
    review_status: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    skip = (page - 1) * page_size
    violations = await get_violations(
        db,
        camera_id=camera_id,
        violation_type=violation_type,
        is_confirmed=is_confirmed,
        status=review_status,
        skip=skip,
        limit=page_size
    )
    total = await count_violations(
        db,
        camera_id=camera_id,
        violation_type=violation_type,
        is_confirmed=is_confirmed,
        status=review_status
    )
    return {
        "data": [violation_to_dto(violation) for violation in violations],
        "total": total,
        "page": page,
        "pageSize": page_size
    }

@router.get("/search", response_model=List[ViolationDto])
async def search_violations(
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    violations = await search_violations_by_license_plate(db, q)
    return [violation_to_dto(violation) for violation in violations]

@router.get("/{violation_id}", response_model=ViolationDto)
async def read_violation(
    violation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    violation = await get_violation(db, violation_id)
    if not violation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bản ghi vi phạm."
        )
    return violation_to_dto(violation)

@router.patch("/{violation_id}", response_model=ViolationDto)
async def verify_violation(
    violation_id: uuid.UUID,
    confirm_data: ViolationConfirmRequest,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    db_obj = await get_violation(db, violation_id)
    if not db_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bản ghi vi phạm."
        )
    violation = await confirm_violation(
        db,
        db_obj=db_obj,
        operator_id=current_operator.id,
        notes=confirm_data.notes,
        is_confirmed=confirm_data.is_confirmed,
        review_status=confirm_data.status
    )
    await db.commit()
    await db.refresh(violation)
    return violation_to_dto(violation)
