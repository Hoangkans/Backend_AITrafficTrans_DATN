import json
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    APP_NAME: str = "Traffic Monitoring System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # CORS Origins
    ALLOWED_ORIGINS: Union[str, List[str]] = ["*"]


    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

    # Database
    DATABASE_URL: str

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_database_url(cls, v: str) -> str:
        if isinstance(v, str):
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://") and not v.startswith("postgresql+"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
            
            # Clean up query parameters for asyncpg compatibility (e.g. Neon DB links)
            v = v.replace("sslmode=require", "ssl=require")
            v = v.replace("&channel_binding=require", "").replace("channel_binding=require&", "").replace("channel_binding=require", "")
        return v


    # JWT Security
    SECRET_KEY: str = "your-super-secret-key-at-least-32-characters-long"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = 24
    PASSWORD_RESET_OTP_EXPIRE_MINUTES: int = 10

    # Email delivery. When SMTP_HOST is not configured, verification links are logged for local demos.
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "no-reply@traffic.com"
    SMTP_USE_TLS: bool = True

    # YOLO & ByteTrack settings
    YOLO_MODEL_PATH: str = "Detection.pt"
    DETECTION_MODEL_PATH: str = "Detection.pt"
    LICENSE_PLATE_MODEL_PATH: str = "License_Plate.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.45
    DETECTION_FPS: float = 10.0
    DETECTION_INTERVAL: float = 0.10
    YOLO_FRAME_SKIP: int = 2

    # OCR settings
    OCR_LIBRARY: str = "pytesseract"
    TESSERACT_CMD: str = "tesseract"
    OCR_LANG: str = "eng"

    # URLs
    BACKEND_URL: str = "http://localhost:8000"
    FRONTEND_URL: str = "http://localhost:3000"

settings = Settings()
