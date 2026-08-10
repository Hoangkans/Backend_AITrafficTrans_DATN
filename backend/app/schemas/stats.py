from datetime import datetime
from typing import Dict, Optional
from app.schemas.auth import BaseSchema

class HourlyStatDto(BaseSchema):
    hour: datetime
    total_vehicles: int
    vehicles_by_type: Dict[str, int]

class TimeSeriesStatDto(BaseSchema):
    timestamp: datetime
    total_vehicles: int
    car_count: int
    truck_count: int
    bus_count: int
    motorcycle_count: int
    bicycle_count: int
    violation_count: int

class ViolationStatDto(BaseSchema):
    violation_type: str
    count: int
    percentage: float

class VehicleTypeStatDto(BaseSchema):
    vehicle_type: str
    count: int
    percentage: float

class ViolationStatusStatDto(BaseSchema):
    status: str
    count: int
    percentage: float

class ViolationHotspotDto(BaseSchema):
    camera_id: str
    camera_name: str
    location: Optional[str] = None
    violation_count: int
    latest_violation_at: Optional[datetime] = None

class DashboardOverviewDto(BaseSchema):
    total_vehicles_daily: int
    total_detections: int
    total_violations: int
    pending_violations: int
    confirmed_violations: int
    motorcycle_count: int
    car_count: int
    truck_count: int
    bus_count: int
    online_cameras: int
    offline_cameras: int
    total_cameras: int

class CameraLiveDto(BaseSchema):
    camera_id: str
    camera_name: Optional[str] = None
    status: Optional[str] = None
    is_online: bool
    live_traffic_rate: float  # vehicles/min
    recent_detections_count: int
    recent_violations_count: int
    last_seen_at: Optional[datetime] = None
    latest_vehicle_type: Optional[str] = None
    latest_confidence: float = 0.0
