from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.cameras import router as cameras_router
from app.api.v1.detections import router as detections_router
from app.api.v1.violations import router as violations_router
from app.api.v1.stats import router as stats_router
from app.api.v1.users import router as users_router
from app.api.v1.settings import router as settings_router
from app.api.v1.websocket import router as websocket_router
from app.api.v1.admin_utils import router as admin_utils_router

api_router = APIRouter()

# Mount all module routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users_router, prefix="/users", tags=["Users"])
api_router.include_router(cameras_router, prefix="/cameras", tags=["Cameras"])
api_router.include_router(detections_router, prefix="/detections", tags=["Detections"])
api_router.include_router(violations_router, prefix="/violations", tags=["Violations"])
api_router.include_router(stats_router, prefix="/stats", tags=["Statistics"])
api_router.include_router(settings_router, prefix="/settings", tags=["Settings"])
api_router.include_router(websocket_router, tags=["WebSockets"])
api_router.include_router(admin_utils_router, tags=["Admin Utilities"])
