# Import declarative Base and all models here
# This is used by Alembic's env.py to auto-generate migrations
from app.core.database import Base
from app.models.operator import Operator
from app.models.camera import Camera
from app.models.detection import Detection
from app.models.violation import Violation
from app.models.traffic_stats import TrafficStat
from app.models.refresh_token import RefreshToken
from app.models.email_verification_token import EmailVerificationToken
from app.models.password_reset_otp import PasswordResetOtp
from app.models.system_setting import SystemSetting
from app.models.notification import Notification

