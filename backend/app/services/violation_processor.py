import os
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.core.config import settings
from app.schemas.detection import DetectionCreate
from app.schemas.violation import ViolationCreate
from app.services.ocr_service import OCRService
from app.crud.crud_violation import create_violation
from app.crud.crud_detection import create_detection
from app.services.yolo_service import yolo_service
import app.models.base
from app.core.database import AsyncSessionLocal

# Vehicle classes from detection.pt that are actual drivable vehicles (eligible for violation check)
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}

# Person/rider classes
PERSON_CLASSES = {"person", "rider"}

# Infrastructure classes (don't OCR, don't check speed violations on these)
INFRA_CLASSES = {"traffic_light", "traffic_sign"}


class ViolationProcessor:
    """Orchestrates vehicle detection, violation checking, license-plate detection, OCR and persistence."""

    def __init__(self):
        self.ocr = OCRService()
        self.yolo = yolo_service  # singleton instance

    def _crop(self, image: Any, bbox: dict) -> Any:
        """Crop a region from an image using a bbox dict (x1,y1,x2,y2)."""
        if hasattr(image, 'shape'):
            return image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            raise RuntimeError("Pillow is required for image cropping when cv2 is unavailable.")
        box = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        return image.crop(box) if hasattr(image, 'crop') else Image.fromarray(image).crop(box)

    def _source_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return "video"
        return "image"

    def _preview_url(self, file_path: str, frame: Any) -> str:
        if self._source_type(file_path) == "image":
            return f"/evidence/{os.path.basename(file_path)}"

        preview_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_preview.jpg"
        preview_path = os.path.join(os.path.dirname(file_path), preview_filename)

        try:
            if cv2 and frame is not None and hasattr(frame, "shape") and getattr(frame, "size", 0) > 0:
                if cv2.imwrite(preview_path, frame):
                    return f"/evidence/{preview_filename}"
            try:
                from PIL import Image
                if hasattr(frame, "shape"):
                    img = Image.fromarray(frame)
                    img.save(preview_path, "JPEG")
                    return f"/evidence/{preview_filename}"
            except Exception:
                pass
        except Exception as exc:
            print(f"[!] Failed to write video preview frame: {exc}")

        return f"/evidence/{os.path.basename(file_path)}"

    def _extract_plate_text(self, cap: Any, file_path: str, bbox: dict, t_sec: float = 0.0) -> tuple[str, str]:
        """
        Extract exact video frame at timestamp t_sec, crop vehicle region with 15% margin,
        run license_plate.pt to detect license plate box, crop plate and run OCR.
        Returns tuple of (normalized_license_plate_text, plate_snapshot_url).
        """
        if not cv2:
            return "", ""

        target_frame = None
        if cap and hasattr(cap, 'isOpened') and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
            ret, target_frame = cap.read()
            if not ret or target_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, target_frame = cap.read()
        elif os.path.exists(file_path) and self._source_type(file_path) == "image":
            target_frame = cv2.imread(file_path)

        if target_frame is None or not hasattr(target_frame, 'shape') or target_frame.size == 0:
            return "", ""

        fh, fw = target_frame.shape[:2]
        w_box = bbox.get("x2", fw) - bbox.get("x1", 0)
        h_box = bbox.get("y2", fh) - bbox.get("y1", 0)
        margin_x = int(w_box * 0.15)
        margin_y = int(h_box * 0.15)

        vx1 = max(0, int(bbox.get("x1", 0)) - margin_x)
        vy1 = max(0, int(bbox.get("y1", 0)) - margin_y)
        vx2 = min(fw, int(bbox.get("x2", fw)) + margin_x)
        vy2 = min(fh, int(bbox.get("y2", fh)) + margin_y)
        vehicle_crop = target_frame[vy1:vy2, vx1:vx2]

        if vehicle_crop.size == 0:
            return "", ""

        plate_text = ""
        plate_crop = None

        # Step 1: Run license_plate.pt model directly on vehicle crop
        if self.yolo.license_plate_model:
            try:
                plate_results = self.yolo.license_plate_model(vehicle_crop, conf=0.10)
                for pr in plate_results:
                    for pbox in pr.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                        pad = 4
                        plate_crop = vehicle_crop[
                            max(0, py1 - pad):min(vehicle_crop.shape[0], py2 + pad),
                            max(0, px1 - pad):min(vehicle_crop.shape[1], px2 + pad)
                        ]
                        if plate_crop.size > 0:
                            txt = self.ocr.extract_text(plate_crop)
                            norm = self.ocr.normalize_plate_text(txt)
                            if norm:
                                plate_text = norm
                                break
                    if plate_text:
                        break
            except Exception as e:
                print(f"[!] License plate model crop extraction error: {e}")

        # Step 2: Fallback OCR on bottom portion of vehicle crop if license_plate.pt did not yield text
        if not plate_text:
            try:
                h_c = vehicle_crop.shape[0]
                lower_crop = vehicle_crop[int(h_c * 0.5):, :]
                if lower_crop.size > 0:
                    txt = self.ocr.extract_text(lower_crop)
                    norm = self.ocr.normalize_plate_text(txt)
                    if norm:
                        plate_text = norm
                        plate_crop = lower_crop
            except Exception as exc:
                pass

        plate_url = ""
        if plate_crop is not None and plate_crop.size > 0:
            try:
                plate_filename = f"plate_{uuid.uuid4().hex[:8]}.jpg"
                plate_path = os.path.join(os.path.dirname(file_path), plate_filename)
                cv2.imwrite(plate_path, plate_crop)
                plate_url = f"/evidence/{plate_filename}"
            except Exception as save_err:
                print(f"[!] Failed to save plate crop image: {save_err}")

        return plate_text, plate_url

    async def process_file(self, camera_id: str, file_path: str) -> Dict[str, Any]:
        """Process an uploaded video/image synchronously and return detections & violations."""
        import asyncio
        cap = None
        if cv2 and self._source_type(file_path) == "video":
            cap = cv2.VideoCapture(file_path)

        preview_frame = None
        if cap and cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            ret, preview_frame = cap.read()

        raw_detections = await asyncio.to_thread(self.yolo.detect, file_path)

        saved_detections = []
        generated_violations = []

        default_cam_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        try:
            target_cam_uuid = uuid.UUID(camera_id)
        except (ValueError, TypeError):
            target_cam_uuid = default_cam_uuid

        vehicle_detections = [d for d in raw_detections if d.get("object_group") == "vehicle" or d.get("class_name") in VEHICLE_CLASSES]
        person_detections = [d for d in raw_detections if d.get("object_group") == "person" or d.get("class_name") in PERSON_CLASSES]
        infra_detections = [d for d in raw_detections if d.get("object_group") == "infrastructure" or d.get("class_name") in INFRA_CLASSES]
        traffic_light_boxes = [d["bbox"] for d in infra_detections if d.get("class_name") == "traffic_light"]

        async with AsyncSessionLocal() as db:
            from sqlalchemy.future import select
            from app.models.camera import Camera

            cam_result = await db.execute(select(Camera).filter(Camera.id == target_cam_uuid))
            db_cam = cam_result.scalars().first()

            if not db_cam:
                any_cam = (await db.execute(select(Camera))).scalars().first()
                if any_cam:
                    db_cam = any_cam
                    target_cam_uuid = db_cam.id
                else:
                    db_cam = Camera(
                        id=target_cam_uuid,
                        name="Camera Giám Sát Mặc Định - Ngã Tư Nguyễn Huệ",
                        rtsp_url="rtsp://localhost:8554/live/cam1",
                        address="Ngã tư Nguyễn Huệ - Lê Lợi, Quận 1, TP.HCM",
                        intersection="Nguyễn Huệ - Lê Lợi",
                        direction="north",
                        status="active",
                        config={"speed_limit": 60},
                        vehicle_types=["car", "motorcycle", "bus", "truck"]
                    )
                    db.add(db_cam)
                    await db.commit()

            valid_camera_id = str(target_cam_uuid)
            seen_violation_keys = set()
            speed_limit = (db_cam.config or {}).get("speed_limit", 60) if db_cam else 60

            for idx, raw in enumerate(raw_detections, start=1):
                metadata = dict(raw.get("metadata") or {})
                bbox = raw.get("bbox", {})
                obj_group = raw.get("object_group", metadata.get("object_group", "unknown"))
                class_name = raw.get("class_name", "unknown")
                t_sec = metadata.get("time", 0.0)

                if raw.get("bbox_pct"):
                    metadata["bbox_pct"] = raw["bbox_pct"]
                metadata["object_group"] = obj_group

                ocr_plate, plate_snapshot_url = "", ""
                if class_name in VEHICLE_CLASSES or obj_group == "vehicle":
                    ocr_plate, plate_snapshot_url = await asyncio.to_thread(
                        self._extract_plate_text, cap, file_path, bbox, t_sec
                    )

                if ocr_plate:
                    metadata["license_plate"] = ocr_plate
                    metadata["license_plate_source"] = "license_plate_pt_ocr"
                    if plate_snapshot_url:
                        metadata["plate_snapshot_url"] = plate_snapshot_url
                else:
                    metadata.pop("license_plate", None)
                    metadata["license_plate_source"] = "unavailable"

                detection_in = {
                    "camera_id": valid_camera_id,
                    "frame_id": idx,
                    "vehicle_type": class_name,
                    "confidence": raw["confidence"],
                    "bbox": bbox,
                    "metadata": metadata,
                }
                db_detection = await create_detection(db, DetectionCreate(**detection_in))  # type: ignore
                saved_detections.append(db_detection)

                if class_name not in VEHICLE_CLASSES and obj_group != "vehicle":
                    continue

                speed = metadata.get("speed_kmh", 0)
                is_speeding = speed > speed_limit

                veh_cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
                veh_cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2
                near_traffic_light = any(
                    abs(veh_cx - (tl.get("x1", 0) + tl.get("x2", 0)) / 2) < 400
                    and abs(veh_cy - (tl.get("y1", 0) + tl.get("y2", 0)) / 2) < 400
                    for tl in traffic_light_boxes
                )

                violation_type = None
                if is_speeding:
                    violation_type = "speeding"
                elif near_traffic_light and speed > 5:
                    violation_type = "red_light"

                if violation_type:
                    violation_key = (ocr_plate, class_name, violation_type) if ocr_plate else (class_name, violation_type, idx // 10)
                    if violation_key not in seen_violation_keys:
                        seen_violation_keys.add(violation_key)
                        viol_meta = {
                            "speed_kmh": speed,
                            "speed_limit": speed_limit,
                            "object_group": obj_group,
                        }
                        if plate_snapshot_url:
                            viol_meta["plate_snapshot_url"] = plate_snapshot_url
                        if near_traffic_light:
                            viol_meta["near_traffic_light"] = True
                        violation_in = {
                            "detection_id": str(db_detection.id),
                            "camera_id": valid_camera_id,
                            "violation_type": violation_type,
                            "vehicle_type": class_name,
                            "license_plate": ocr_plate or None,
                            "confidence": raw["confidence"],
                            "evidence_url": f"/evidence/{os.path.basename(file_path)}",
                            "metadata": viol_meta,
                        }
                        db_violation = await create_violation(db, ViolationCreate(**violation_in))  # type: ignore
                        generated_violations.append(db_violation)

        if cap and hasattr(cap, "release"):
            cap.release()

        return {
            "success": True,
            "source_url": f"/evidence/{os.path.basename(file_path)}",
            "source_type": self._source_type(file_path),
            "preview_url": self._preview_url(file_path, preview_frame),
            "count": len(saved_detections),
            "violations_count": len(generated_violations),
            "object_summary": {
                "vehicles": len(vehicle_detections),
                "persons": len(person_detections),
                "infrastructure": len(infra_detections),
                "traffic_lights": len(traffic_light_boxes),
            },
            "detections": [
                {
                    "id": str(d.id),
                    "camera_id": str(d.camera_id),
                    "frame_id": d.frame_id,
                    "vehicle_type": d.vehicle_type,
                    "object_group": (d.metadata_ or {}).get("object_group", "unknown"),
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "bbox_pct": (d.metadata_ or {}).get("bbox_pct") or {
                        "x": round(((d.bbox.get("x1", 100) if d.bbox else 100) / 12.8), 1),
                        "y": round(((d.bbox.get("y1", 200) if d.bbox else 200) / 7.2), 1),
                        "w": round((((d.bbox.get("x2", 400) if d.bbox else 400) - (d.bbox.get("x1", 100) if d.bbox else 100)) / 12.8), 1),
                        "h": round((((d.bbox.get("y2", 500) if d.bbox else 500) - (d.bbox.get("y1", 200) if d.bbox else 200)) / 7.2), 1)
                    },
                    "metadata": d.metadata_ or {},
                    "detected_at": d.detected_at.isoformat() if d.detected_at else None,
                }
                for d in saved_detections
            ],
            "violations": [
                {
                    "id": str(v.id),
                    "camera_id": str(v.camera_id),
                    "violation_type": v.violation_type,
                    "vehicle_type": v.vehicle_type,
                    "license_plate": v.license_plate,
                    "evidence_url": (v.metadata_ or {}).get("plate_snapshot_url") or v.evidence_url,
                    "confidence": v.confidence,
                    "metadata": v.metadata_ or {},
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in generated_violations
            ],
        }


