import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import ResponseValidationError

from app.core.config import settings
from app.core.database import engine, Base, AsyncSessionLocal
from app.models import base
from app.api.v1.router import api_router
from app.core.security import get_password_hash
from app.models.operator import Operator
from sqlalchemy import text, select

import uuid
from app.models.camera import Camera
from sqlalchemy.exc import SQLAlchemyError

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DB and auto-create tables if they don't exist
    print("[*] Starting application lifespan...")
    async with engine.begin() as conn:
        # Create all tables if not exists
        await conn.run_sync(Base.metadata.create_all)
        # Schema migration: Add missing columns if they don't exist in existing tables
        if "sqlite" in str(engine.url):
            res = await conn.execute(text("PRAGMA table_info(violations);"))
            columns = [row[1] for row in res.fetchall()]
            if "status" not in columns:
                await conn.execute(text("ALTER TABLE violations ADD COLUMN status VARCHAR(20) DEFAULT 'pending';"))
            if "is_confirmed" not in columns:
                await conn.execute(text("ALTER TABLE violations ADD COLUMN is_confirmed BOOLEAN DEFAULT FALSE;"))
            if "confirmed_by" not in columns:
                await conn.execute(text("ALTER TABLE violations ADD COLUMN confirmed_by CHAR(36);"))
            if "notes" not in columns:
                await conn.execute(text("ALTER TABLE violations ADD COLUMN notes TEXT;"))
        else:
            await conn.execute(text("ALTER TABLE violations ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending';"))
            await conn.execute(text("ALTER TABLE violations ADD COLUMN IF NOT EXISTS is_confirmed BOOLEAN DEFAULT FALSE;"))
            await conn.execute(text("ALTER TABLE violations ADD COLUMN IF NOT EXISTS confirmed_by UUID;"))
            await conn.execute(text("ALTER TABLE violations ADD COLUMN IF NOT EXISTS notes TEXT;"))
        print("[+] Database tables initialized & migrated.")

    # Auto-sanitize legacy invalid/fake plates in DB on startup
    async with AsyncSessionLocal() as session:
        try:
            import re, json
            valid_re = re.compile(r'^\d{2}[A-Z]{1,2}-\d{4,5}$')
            
            # Clean violations
            v_res = await session.execute(text("SELECT id, license_plate FROM violations WHERE license_plate IS NOT NULL"))
            v_rows = v_res.fetchall()
            v_cleaned = 0
            for vid, plate in v_rows:
                if plate and not valid_re.match(plate.strip().upper()):
                    await session.execute(text("UPDATE violations SET license_plate = NULL WHERE id = :id"), {"id": vid})
                    v_cleaned += 1

            # Clean detections metadata
            d_res = await session.execute(text("SELECT id, metadata FROM detections WHERE metadata::text LIKE '%license_plate%'"))
            d_rows = d_res.fetchall()
            d_cleaned = 0
            for did, meta in d_rows:
                meta_dict = dict(meta or {})
                plate = meta_dict.get('license_plate')
                if plate and not valid_re.match(str(plate).strip().upper()):
                    meta_dict.pop('license_plate', None)
                    meta_dict['license_plate_source'] = 'unavailable'
                    await session.execute(
                        text("UPDATE detections SET metadata = :meta WHERE id = :id"),
                        {"meta": json.dumps(meta_dict), "id": did}
                    )
                    d_cleaned += 1

            if v_cleaned > 0 or d_cleaned > 0:
                await session.commit()
                print(f"[+] Startup DB Sanitization: Cleaned {v_cleaned} fake plates in violations and {d_cleaned} in detections.")
        except Exception as err:
            print(f"[!] Startup DB Sanitization skipped/failed: {err}")
            await session.rollback()

    # Seed admin user & default camera if they do not exist
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(select(Operator).filter(Operator.username == "admin"))
            admin_user = result.scalars().first()
            
            if not admin_user:
                print("[*] Seeding default admin user...")
                hashed_pw = get_password_hash("admin123")
                admin = Operator(
                    username="admin",
                    email="admin@traffic.com",
                    hashed_password=hashed_pw,
                    full_name="System Administrator",
                    role="admin",
                    is_active=True,
                    is_email_verified=True
                )
                session.add(admin)
                await session.commit()
                print("[+] Default admin user seeded successfully (username: 'admin', password: 'admin123').")
            else:
                updated = False
                if admin_user.email and (admin_user.email == "admin@traffic.local" or admin_user.email.endswith(".local")):
                    admin_user.email = admin_user.email.replace(".local", ".com")
                    updated = True
                if not admin_user.is_email_verified:
                    admin_user.is_email_verified = True
                    updated = True
                if updated:
                    session.add(admin_user)
                    await session.commit()
                    print(f"[+] Admin user updated in DB (email: {admin_user.email}).")
                else:
                    print("[+] Admin user already exists. Skipping seed.")

            # Seed default camera (UUID: 00000000-0000-0000-0000-000000000001)
            default_cam_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
            cam_result = await session.execute(select(Camera).filter(Camera.id == default_cam_uuid))
            default_cam = cam_result.scalars().first()
            if not default_cam:
                print("[*] Seeding default camera...")
                default_cam = Camera(
                    id=default_cam_uuid,
                    name="Camera Giám Sát Mặc Định - Ngã Tư Nguyễn Huệ",
                    rtsp_url="rtsp://localhost:8554/live/cam1",
                    address="Ngã tư Nguyễn Huệ - Lê Lợi, Quận 1, TP.HCM",
                    intersection="Nguyễn Huệ - Lê Lợi",
                    direction="north",
                    status="active",
                    config={"speed_limit": 60},
                    vehicle_types=["car", "motorcycle", "bus", "truck"]
                )
                session.add(default_cam)
                await session.commit()
                print("[+] Default camera seeded successfully.")
        except Exception as e:
            print(f"[-] Error seeding database: {e}")
            await session.rollback()

    yield
    # Shutdown
    await engine.dispose()
    print("[-] Application shutdown complete.")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI Backend for Smart Traffic Monitoring System",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    print(f"[-] ResponseValidationError on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Response validation error", "errors": exc.errors()},
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    print(f"[-] Database error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Lỗi cơ sở dữ liệu: {str(exc.orig) if hasattr(exc, 'orig') else str(exc)}"},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[-] Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal Server Error"},
    )

# Mount static folder for serving violation evidence images
evidence_dir = "static/evidence"
os.makedirs(evidence_dir, exist_ok=True)
app.mount("/evidence", StaticFiles(directory="static/evidence"), name="evidence")

# Include routers - mount under both /api and /api/v1 for maximum client compatibility
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "system": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }
