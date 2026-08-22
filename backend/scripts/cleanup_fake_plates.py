"""
Script dọn dẹp biển số giả trong database.
Xóa các bản ghi violation có license_plate không hợp lệ (từ dữ liệu mock cũ),
và cập nhật metadata của detections để xóa license_plate không hợp lệ.
"""
import asyncio
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import AsyncSessionLocal
from sqlalchemy import text

# Regex biển số VN hợp lệ: VD 51H-12345 hoặc 51HA-12345
VALID_PLATE_RE = re.compile(r'^\d{2}[A-Z]{1,2}-\d{4,5}$')


async def main():
    async with AsyncSessionLocal() as db:
        # 1. Xem violations
        r = await db.execute(text("SELECT id, license_plate FROM violations ORDER BY created_at DESC"))
        violations = r.fetchall()
        print(f"\n=== Violations: {len(violations)} bản ghi ===")

        invalid_viol_ids = []
        for v in violations:
            plate = v[1] or ''
            is_valid = bool(VALID_PLATE_RE.match(plate)) if plate else False
            marker = "✓" if is_valid else "✗ GIẢ/TRỐNG"
            print(f"  {v[0]}: plate='{plate}' {marker}")
            if not is_valid:
                invalid_viol_ids.append(v[0])

        # 2. Xem detections với license_plate trong metadata
        r2 = await db.execute(text(
            "SELECT id, metadata FROM detections WHERE metadata::text LIKE '%license_plate%' ORDER BY detected_at DESC LIMIT 50"
        ))
        detections = r2.fetchall()
        print(f"\n=== Detections có license_plate: {len(detections)} bản ghi ===")

        import json
        invalid_det_ids = []
        for d in detections:
            meta = d[1] or {}
            plate = meta.get('license_plate', '')
            source = meta.get('license_plate_source', 'unknown')
            is_valid = bool(VALID_PLATE_RE.match(plate)) if plate else False
            marker = "✓" if is_valid else "✗ GIẢ/TRỐNG"
            print(f"  {d[0]}: plate='{plate}' source={source} {marker}")
            if not is_valid or source not in ('ocr',):
                invalid_det_ids.append((d[0], meta))

        # 3. Hỏi xác nhận
        print(f"\n>> Tìm thấy {len(invalid_viol_ids)} violations có plate giả/trống")
        print(f">> Tìm thấy {len(invalid_det_ids)} detections có plate giả/trống")
        confirm = input("\nXóa/cập nhật các bản ghi này? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("Hủy bỏ.")
            return

        # 4. Xóa violations giả (chỉ cập nhật license_plate thành NULL, không xóa bản ghi vi phạm)
        for vid in invalid_viol_ids:
            await db.execute(
                text("UPDATE violations SET license_plate = NULL WHERE id = :id"),
                {"id": vid}
            )
        print(f"  → Đã xóa license_plate khỏi {len(invalid_viol_ids)} violations")

        # 5. Cập nhật detections metadata - xóa plate giả, giữ source=unavailable
        for det_id, meta in invalid_det_ids:
            meta.pop('license_plate', None)
            meta['license_plate_source'] = 'unavailable'
            await db.execute(
                text("UPDATE detections SET metadata = :meta WHERE id = :id"),
                {"meta": json.dumps(meta), "id": det_id}
            )
        print(f"  → Đã cập nhật metadata của {len(invalid_det_ids)} detections")

        await db.commit()
        print("\n✅ Hoàn tất. Database đã được làm sạch.")


if __name__ == '__main__':
    asyncio.run(main())
