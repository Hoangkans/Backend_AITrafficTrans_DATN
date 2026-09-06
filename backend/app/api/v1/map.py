import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.models.camera import Camera
from app.models.violation import Violation

router = APIRouter()

@router.get("/cameras", response_model=List[Dict[str, Any]])
async def get_map_cameras(
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    """
    Trả về danh sách tất cả các camera kèm tọa độ latitude, longitude, địa điểm
    và thống kê vi phạm thực tế để hiển thị trên bản đồ map.
    """
    query = select(Camera)
    if status_filter:
        query = query.where(Camera.status == status_filter)
        
    result = await db.execute(query)
    cameras = result.scalars().all()
    
    # Query violation counts grouped by camera_id
    counts_query = (
        select(Violation.camera_id, func.count(Violation.id).label("total_violations"))
        .group_by(Violation.camera_id)
    )
    counts_res = await db.execute(counts_query)
    violation_counts = {str(row[0]): row[1] for row in counts_res.all()}
    
    map_markers = []
    for c in cameras:
        # Fallback coordinates for demo cameras if null
        default_lat = c.latitude if c.latitude is not None else 10.7769
        default_lng = c.longitude if c.longitude is not None else 106.7009
        
        map_markers.append({
            "id": str(c.id),
            "name": c.name,
            "rtspUrl": c.rtsp_url,
            "latitude": default_lat,
            "longitude": default_lng,
            "address": c.address or "Địa điểm mặc định, TP. Hồ Chí Minh",
            "intersection": c.intersection or "Nút giao thông",
            "direction": c.direction or "Tất cả hướng",
            "status": c.status,
            "violationCount": violation_counts.get(str(c.id), 0),
            "config": c.config or {}
        })
        
    return map_markers

@router.get("/heat", response_model=List[Dict[str, Any]])
async def get_map_violation_heat(
    db: AsyncSession = Depends(get_db),
    current_operator: Operator = Depends(get_current_operator)
):
    """
    Trả về dữ liệu bản đồ nhiệt (Heatmap) thể hiện mật độ vi phạm giao thông tại các nút tọa độ.
    """
    query = (
        select(
            Camera.latitude,
            Camera.longitude,
            Camera.name,
            Camera.address,
            func.count(Violation.id).label("intensity")
        )
        .join(Violation, Camera.id == Violation.camera_id)
        .group_by(Camera.id, Camera.latitude, Camera.longitude, Camera.name, Camera.address)
    )
    
    result = await db.execute(query)
    heat_points = []
    
    for row in result.all():
        lat, lng, name, address, intensity = row
        if lat is not None and lng is not None:
            heat_points.append({
                "latitude": lat,
                "longitude": lng,
                "name": name,
                "address": address,
                "intensity": intensity
            })
            
    return heat_points

@router.get("/geocode")
async def check_location_geocode(
    q: Optional[str] = Query(None, description="Địa chỉ hoặc nút giao cần tra cứu"),
    lat: Optional[float] = Query(None, description="Vĩ độ"),
    lng: Optional[float] = Query(None, description="Kinh độ"),
    current_operator: Operator = Depends(get_current_operator)
):
    """
    API kiểm tra và định vị tọa độ địa điểm cho camera.
    """
    if lat is not None and lng is not None:
        return {
            "latitude": lat,
            "longitude": lng,
            "address": q or f"Vị trí tọa độ ({lat:.5f}, {lng:.5f})",
            "city": "TP. Hồ Chí Minh",
            "country": "Việt Nam"
        }
        
    # Search dummy or matched coordinates for common intersection queries
    sample_locations = {
        "bến thành": {"latitude": 10.7721, "longitude": 106.6983, "address": "Chợ Bến Thành, Quận 1, TP. Hồ Chí Minh"},
        "hàng xanh": {"latitude": 10.8012, "longitude": 106.7115, "address": "Ngã tư Hàng Xanh, Bình Thạnh, TP. Hồ Chí Minh"},
        "thủ đức": {"latitude": 10.8505, "longitude": 106.7719, "address": "Ngã tư Thủ Đức, TP. Thủ Đức, TP. Hồ Chí Minh"},
        "lê duẩn": {"latitude": 10.7818, "longitude": 106.6995, "address": "Đại lộ Lê Duẩn, Quận 1, TP. Hồ Chí Minh"},
        "võ văn kiệt": {"latitude": 10.7533, "longitude": 106.6631, "address": "Đường Võ Văn Kiệt, Quận 5, TP. Hồ Chí Minh"}
    }
    
    query_str = (q or "").lower().strip()
    for key, loc in sample_locations.items():
        if key in query_str:
            return loc
            
    return {
        "latitude": 10.7769,
        "longitude": 106.7009,
        "address": q or "Trung tâm TP. Hồ Chí Minh",
        "city": "TP. Hồ Chí Minh",
        "country": "Việt Nam"
    }
