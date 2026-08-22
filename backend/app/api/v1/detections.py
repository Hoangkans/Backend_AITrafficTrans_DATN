import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.crud.crud_camera import get_camera
from app.crud.crud_detection import count_detections, create_detection, get_detections
from app.crud.crud_violation import create_violation
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.schemas.detection import BoundingBoxDto, DetectionCreate, DetectionDto
from app.schemas.violation import ViolationCreate
from app.services.yolo_service import yolo_service

router = APIRouter()

UPLOAD_DIR = "static/evidence"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DETECTION_JOBS: Dict[str, Dict[str, Any]] = {}


def get_source_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return "video"
    return "image"


def detection_to_dto(detection) -> DetectionDto:
    bbox_data = detection.bbox or {}
    return DetectionDto(
        id=detection.id,
        camera_id=detection.camera_id,
        frame_id=detection.frame_id,
        vehicle_type=detection.vehicle_type,
        confidence=detection.confidence,
        bbox=BoundingBoxDto(
            x1=bbox_data.get("x1", 0),
            y1=bbox_data.get("y1", 0),
            x2=bbox_data.get("x2", 0),
            y2=bbox_data.get("y2", 0),
        ),
        metadata=detection.metadata_ or {},
        detected_at=detection.detected_at,
    )


def save_upload_file(file: UploadFile) -> tuple[str, str]:
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return filename, file_path


async def process_detection_file(
    db: AsyncSession,
    camera_id: str,
    file_path: str,
    evidence_filename: str,
) -> Dict[str, Any]:
    from app.services.violation_processor import ViolationProcessor
    processor = ViolationProcessor()
    return await processor.process_file(camera_id, file_path)



async def run_detection_job(
    job_id: str,
    camera_id: str,
    file_path: str,
    evidence_filename: str,
):
    DETECTION_JOBS[job_id].update(
        {
            "status": "running",
            "progress": 20,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
    )
    try:
        # Use ViolationProcessor for the full pipeline:
        # YOLO detect → crop xe → license_plate_model (if available) → OCR → normalize
        from app.services.violation_processor import ViolationProcessor
        processor = ViolationProcessor()
        result = await processor.process_file(camera_id, file_path)
        DETECTION_JOBS[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "result": result,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        DETECTION_JOBS[job_id].update(
            {
                "status": "failed",
                "progress": 100,
                "error": str(exc),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
        )


@router.get("", response_model=Dict[str, Any])
async def read_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, alias="pageSize"),
    camera_id: Optional[uuid.UUID] = Query(None, alias="cameraId"),
    vehicle_type: Optional[str] = Query(None, alias="vehicleType"),
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    min_confidence: Optional[float] = Query(None, alias="minConfidence"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator),
):
    skip = (page - 1) * page_size
    detections = await get_detections(
        db,
        camera_id=camera_id,
        vehicle_type=vehicle_type,
        from_date=from_date,
        to_date=to_date,
        min_confidence=min_confidence,
        skip=skip,
        limit=page_size,
    )
    total = await count_detections(
        db,
        camera_id=camera_id,
        vehicle_type=vehicle_type,
        from_date=from_date,
        to_date=to_date,
        min_confidence=min_confidence,
    )

    data = [detection_to_dto(detection).model_dump(mode="json", by_alias=True) for detection in detections]
    return {"data": data, "total": total, "page": page, "pageSize": page_size}


@router.post("/detect", status_code=status.HTTP_201_CREATED)
async def upload_and_detect(
    camera_id: str = Query(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator),
):
    filename, file_path = save_upload_file(file)
    # New synchronous processing using ViolationProcessor
    from app.services.violation_processor import ViolationProcessor
    processor = ViolationProcessor()
    result = await processor.process_file(camera_id, file_path)
    # Commit any DB changes performed inside the processor
    await db.commit()
    return result


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_detection_job(
    camera_id: str = Query(...),
    file: UploadFile = File(...),
    current_operator: Operator = Depends(get_current_operator),
):
    filename, file_path = save_upload_file(file)
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    DETECTION_JOBS[job_id] = {
        "id": job_id,
        "cameraId": camera_id,
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
        "createdAt": now,
        "updatedAt": now,
    }
    asyncio.create_task(run_detection_job(job_id, camera_id, file_path, filename))
    return DETECTION_JOBS[job_id]


@router.get("/jobs", response_model=Dict[str, Any])
async def read_detection_jobs(
    current_operator: Operator = Depends(get_current_operator),
):
    jobs = sorted(DETECTION_JOBS.values(), key=lambda job: job["createdAt"], reverse=True)
    return {"data": jobs, "total": len(jobs)}


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def read_detection_job(
    job_id: str,
    current_operator: Operator = Depends(get_current_operator),
):
    job = DETECTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Khong tim thay detection job.",
        )
    return job


@router.get("/{detection_id}", response_model=DetectionDto)
async def read_detection(
    detection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator),
):
    from app.crud.crud_detection import get_detection

    detection = await get_detection(db, detection_id)
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Khong tim thay ban ghi detection.",
        )

    return detection_to_dto(detection)
