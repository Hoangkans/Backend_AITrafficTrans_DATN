import uuid
import os
import shutil
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.schemas.detection import DetectionDto, BoundingBoxDto
from app.schemas.violation import ViolationCreate
from app.crud.crud_camera import get_camera
from app.crud.crud_detection import get_detections, count_detections, create_detection
from app.crud.crud_violation import create_violation
from app.services.yolo_service import yolo_service
from app.schemas.detection import DetectionCreate

router = APIRouter()

# Directory to save evidence/images
UPLOAD_DIR = "static/evidence"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
            y2=bbox_data.get("y2", 0)
        ),
        metadata=detection.metadata_ or {},
        detected_at=detection.detected_at
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
    current_operator: Operator = Depends(get_current_operator)
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
        limit=page_size
    )
    total = await count_detections(
        db,
        camera_id=camera_id,
        vehicle_type=vehicle_type,
        from_date=from_date,
        to_date=to_date,
        min_confidence=min_confidence
    )
    
    data = [detection_to_dto(d).model_dump(mode="json", by_alias=True) for d in detections]
        
    return {
        "data": data,
        "total": total,
        "page": page,
        "pageSize": page_size
    }

@router.get("/{detection_id}", response_model=DetectionDto)
async def read_detection(
    detection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    """Lấy chi tiết detection — tương đương GET /api/v1/detections/{id} trong ASP.NET."""
    from app.crud.crud_detection import get_detection
    detection = await get_detection(db, detection_id)
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy bản ghi detection."
        )

    return detection_to_dto(detection)

@router.post("/detect", status_code=status.HTTP_201_CREATED)
async def upload_and_detect(
    camera_id: str = Query(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    # Verify camera exists
    try:
        cam_uuid = uuid.UUID(camera_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID camera không hợp lệ."
        )
        
    camera = await get_camera(db, cam_uuid)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy camera."
        )
        
    # Save file locally
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run YOLO detection
    raw_detections = yolo_service.detect(file_path)
    
    saved_detections = []
    generated_violations = []
    
    # Process detections
    import random
    from app.api.v1.websocket import dashboard_manager
    
    for rd in raw_detections:
        # Create detection
        bbox_dto = BoundingBoxDto(**rd["bbox"])
        det_in = DetectionCreate(
            camera_id=camera_id,
            frame_id=random.randint(1000, 99999), # dummy frame_id
            vehicle_type=rd["class_name"],
            confidence=rd["confidence"],
            bbox=bbox_dto,
            metadata=rd.get("metadata", {})
        )
        
        db_det = await create_detection(db, det_in)
        saved_detections.append(db_det)
        
        # Check if speed exists in metadata and exceeds speed limit
        speed_limit = camera.config.get("speed_limit", 60)
        speed = rd.get("metadata", {}).get("speed_kmh", 0)
        
        is_violation = False
        violation_type = None
        
        if speed > speed_limit:
            is_violation = True
            violation_type = "speeding"
        # Or mock random violation (e.g. red light, wrong lane)
        elif random.random() < 0.15: # 15% chance of random violation
            is_violation = True
            violation_type = random.choice(["red_light", "wrong_lane", "no_helmet"])
            
        if is_violation:
            evidence_url = f"/evidence/{filename}"
            vio_in = ViolationCreate(
                detection_id=str(db_det.id),
                camera_id=camera_id,
                violation_type=violation_type,
                vehicle_type=rd["class_name"],
                license_plate=rd.get("metadata", {}).get("license_plate"),
                confidence=rd["confidence"],
                evidence_url=evidence_url,
                metadata={
                    "speed_kmh": speed,
                    "speed_limit": speed_limit,
                    "evidence_file": file_path
                }
            )
            db_vio = await create_violation(db, vio_in)
            generated_violations.append(db_vio)
            
            # Broadcast real-time WebSocket alert
            alert_message = {
                "type": "violation_alert",
                "priority": "high",
                "data": {
                    "id": str(db_vio.id),
                    "camera_id": camera_id,
                    "camera_name": camera.name,
                    "violation_type": violation_type,
                    "vehicle_type": rd["class_name"],
                    "license_plate": db_vio.license_plate,
                    "confidence": db_vio.confidence,
                    "evidence_url": db_vio.evidence_url,
                    "created_at": db_vio.created_at.isoformat()
                }
            }
            await dashboard_manager.broadcast(alert_message)
            
    await db.commit()
    
    # Return count and details
    return {
        "success": True,
        "count": len(saved_detections),
        "violations_count": len(generated_violations),
        "detections": [
            {
                "id": str(d.id),
                "vehicle_type": d.vehicle_type,
                "confidence": d.confidence,
                "bbox": d.bbox,
                "metadata": d.metadata_
            } for d in saved_detections
        ]
    }
