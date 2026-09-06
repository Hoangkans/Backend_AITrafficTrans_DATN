import csv
import io
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_db
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.models.violation import Violation
from app.models.camera import Camera
from app.models.traffic_stats import TrafficStat

router = APIRouter()

VIOLATION_TYPE_MAP = {
    "red_light": "Vượt đèn đỏ",
    "speeding": "Chạy quá tốc độ",
    "wrong_lane": "Đi sai làn đường",
    "no_helmet": "Không đội mũ bảo hiểm",
    "overcrowding": "Chở quá số người quy định",
    "illegal_parking": "Dừng đỗ trái phép"
}

STATUS_MAP = {
    "pending": "Chờ đối chiếu",
    "verified": "Đã phê duyệt",
    "rejected": "Bị từ chối"
}

@router.get("/export/violations")
async def export_violations(
    format: str = Query("excel", regex="^(excel|csv)$"),
    camera_id: Optional[uuid.UUID] = Query(None, alias="cameraId"),
    violation_type: Optional[str] = Query(None, alias="violationType"),
    status_filter: Optional[str] = Query(None, alias="status"),
    start_date: Optional[str] = Query(None, alias="startDate"),
    end_date: Optional[str] = Query(None, alias="endDate"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    query = select(Violation).join(Camera, Violation.camera_id == Camera.id, isouter=True)
    conditions = []

    if camera_id:
        conditions.append(Violation.camera_id == camera_id)
    if violation_type:
        conditions.append(Violation.violation_type == violation_type)
    if status_filter:
        conditions.append(Violation.status == status_filter)
    if start_date:
        try:
            dt_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            conditions.append(Violation.created_at >= dt_start)
        except Exception:
            pass
    if end_date:
        try:
            dt_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            conditions.append(Violation.created_at <= dt_end)
        except Exception:
            pass

    if conditions:
        query = query.where(and_(*conditions))

    query = query.order_by(Violation.created_at.desc())
    result = await db.execute(query)
    violations = result.scalars().all()

    # Load camera mapping for names & location addresses
    cam_ids = {v.camera_id for v in violations if v.camera_id}
    cameras_dict = {}
    if cam_ids:
        cam_res = await db.execute(select(Camera).where(Camera.id.in_(cam_ids)))
        for c in cam_res.scalars().all():
            cameras_dict[str(c.id)] = c

    # Format output rows
    headers = [
        "Mã vi phạm ID",
        "Biển kiểm soát",
        "Hành vi vi phạm",
        "Loại phương tiện",
        "Độ tin cậy AI (%)",
        "Địa điểm / Camera",
        "Tọa độ (Lat, Lng)",
        "Trạng thái xử lý",
        "Thời gian ghi nhận",
        "Ghi chú"
    ]

    rows = []
    for v in violations:
        cam = cameras_dict.get(str(v.camera_id))
        location_str = cam.address if cam and cam.address else (cam.name if cam else str(v.camera_id))
        lat_lng = f"{cam.latitude}, {cam.longitude}" if cam and cam.latitude and cam.longitude else "N/A"
        
        rows.append([
            str(v.id),
            v.license_plate or "Chưa nhận diện",
            VIOLATION_TYPE_MAP.get(v.violation_type, v.violation_type),
            v.vehicle_type,
            f"{round((v.confidence or 0) * 100, 1)}%",
            location_str,
            lat_lng,
            STATUS_MAP.get(v.status, v.status),
            v.created_at.strftime("%d/%m/%Y %H:%M:%S") if v.created_at else "",
            v.notes or ""
        ])

    # Generate CSV with UTF-8 BOM for Microsoft Excel compatibility
    output = io.StringIO()
    output.write('\ufeff')  # UTF-8 BOM header
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    writer.writerows(rows)

    csv_data = output.getvalue()
    output.close()

    filename = f"Bao_Cao_Vi_Pham_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data.encode('utf-8')),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )

@router.get("/export/stats")
async def export_traffic_stats(
    start_date: Optional[str] = Query(None, alias="startDate"),
    end_date: Optional[str] = Query(None, alias="endDate"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    query = select(TrafficStat).order_by(TrafficStat.recorded_at.desc())
    result = await db.execute(query)
    stats = result.scalars().all()

    headers = [
        "ID Thống kê",
        "Mã Camera ID",
        "Tổng lưu lượng",
        "Số ô tô",
        "Số xe máy",
        "Số xe tải/xe khách",
        "Số vi phạm",
        "Tốc độ trung bình (km/h)",
        "Thời gian ghi nhận"
    ]

    rows = []
    for s in stats:
        rows.append([
            str(s.id),
            str(s.camera_id) if s.camera_id else "N/A",
            s.total_count or 0,
            s.car_count or 0,
            s.motorcycle_count or 0,
            s.truck_count or 0,
            s.violation_count or 0,
            f"{s.avg_speed_kmh:.1f}" if s.avg_speed_kmh else "0",
            s.recorded_at.strftime("%d/%m/%Y %H:%M:%S") if s.recorded_at else ""
        ])

    output = io.StringIO()
    output.write('\ufeff')
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    writer.writerows(rows)

    csv_data = output.getvalue()
    output.close()

    filename = f"Bao_Cao_Thong_Ke_Giao_Thong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        io.BytesIO(csv_data.encode('utf-8')),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
