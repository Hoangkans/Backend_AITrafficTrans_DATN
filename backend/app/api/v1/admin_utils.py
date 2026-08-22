"""
Admin utility endpoints: dọn dẹp dữ liệu không hợp lệ.
CHỈ dùng cho admin. Đặt tại /api/v1/admin/...
"""
import re
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.dependencies.auth import require_admin

router = APIRouter(prefix="/admin", tags=["admin-utils"])

# Biển số VN hợp lệ: 51H-12345 hoặc 51HA-12345
VALID_PLATE_RE = re.compile(r'^\d{2}[A-Z]{1,2}-\d{4,5}$')


@router.get("/fake-plates/scan", response_model=Dict[str, Any])
async def scan_fake_plates(
    db: AsyncSession = Depends(get_db),
    _: Any = Depends(require_admin)
):
    """Quét và báo cáo các bản ghi có biển số không hợp lệ."""
    # Violations
    r = await db.execute(text("SELECT id, license_plate FROM violations"))
    violations = r.fetchall()
    invalid_violations = [
        {"id": str(v[0]), "plate": v[1]}
        for v in violations
        if v[1] and not VALID_PLATE_RE.match(v[1])
    ]

    # Detections với license_plate trong metadata
    r2 = await db.execute(text(
        "SELECT id, metadata FROM detections WHERE metadata::text LIKE '%license_plate%'"
    ))
    detections = r2.fetchall()
    invalid_detections = []
    for d in detections:
        meta = d[1] or {}
        plate = meta.get('license_plate', '')
        source = meta.get('license_plate_source', '')
        if plate and not VALID_PLATE_RE.match(plate):
            invalid_detections.append({
                "id": str(d[0]),
                "plate": plate,
                "source": source
            })

    return {
        "total_violations_scanned": len(violations),
        "invalid_violations": len(invalid_violations),
        "invalid_violations_list": invalid_violations[:50],
        "total_detections_scanned": len(detections),
        "invalid_detections": len(invalid_detections),
        "invalid_detections_list": invalid_detections[:50],
    }


@router.post("/fake-plates/cleanup", response_model=Dict[str, Any])
async def cleanup_fake_plates(
    db: AsyncSession = Depends(get_db),
    _: Any = Depends(require_admin)
):
    """
    Xóa biển số không hợp lệ khỏi violations (set NULL) và
    cập nhật metadata của detections (source=unavailable).
    """
    import json

    # 1. Violations: set license_plate = NULL nếu không hợp lệ
    r = await db.execute(text("SELECT id, license_plate FROM violations"))
    violations = r.fetchall()
    cleaned_violations = 0
    for v in violations:
        plate = v[1] or ''
        if plate and not VALID_PLATE_RE.match(plate):
            await db.execute(
                text("UPDATE violations SET license_plate = NULL WHERE id = :id"),
                {"id": v[0]}
            )
            cleaned_violations += 1

    # 2. Detections: xóa license_plate giả khỏi metadata
    r2 = await db.execute(text(
        "SELECT id, metadata FROM detections WHERE metadata::text LIKE '%license_plate%'"
    ))
    detections = r2.fetchall()
    cleaned_detections = 0
    for d in detections:
        meta = dict(d[1] or {})
        plate = meta.get('license_plate', '')
        if plate and not VALID_PLATE_RE.match(plate):
            meta.pop('license_plate', None)
            meta['license_plate_source'] = 'unavailable'
            await db.execute(
                text("UPDATE detections SET metadata = :meta::jsonb WHERE id = :id"),
                {"meta": json.dumps(meta), "id": d[0]}
            )
            cleaned_detections += 1

    await db.commit()

    return {
        "success": True,
        "cleaned_violations": cleaned_violations,
        "cleaned_detections": cleaned_detections,
        "message": f"Đã xóa {cleaned_violations} biển số giả khỏi violations và {cleaned_detections} detections."
    }
