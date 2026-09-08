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
from typing import Any, Dict, List, Optional

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
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "motorbike", "motor", "bike", "bicycle", "rider", "train"}

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

    def _enhance_crop_for_ocr(self, crop: Any) -> Any:
        """Upscale and enhance low-resolution license plate crops for higher OCR accuracy."""
        if crop is None or not cv2 or not hasattr(crop, "shape") or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if h < 120 or w < 240:
            scale = max(120.0 / max(1, h), 240.0 / max(1, w))
            scale = min(scale, 3.5)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop

    def _check_wrong_lane(self, vehicle_type: str, bbox: dict, frame_w: float = 1280.0, frame_h: float = 720.0, config_dict: Optional[dict] = None) -> tuple[bool, str]:
        """
        Check if vehicle is driving in a wrong/prohibited lane.
        Returns tuple of (is_wrong_lane, lane_description).
        """
        v_type = (vehicle_type or "").lower()
        is_car_group = v_type in {"car", "truck", "bus", "automobile"}
        is_motor_group = v_type in {"motorcycle", "motorbike", "motor", "bike", "rider"}

        if not is_car_group and not is_motor_group:
            return False, "Bình thường"

        # Calculate bottom-center coordinate in percentage relative to frame
        x1 = float(bbox.get("x1", 0))
        x2 = float(bbox.get("x2", frame_w))
        bc_x = (x1 + x2) / 2.0
        bc_x_pct = (bc_x / max(1.0, frame_w)) * 100.0

        # Custom lane configuration check if present
        lanes = (config_dict or {}).get("lanes", [])
        if lanes and isinstance(lanes, list):
            for lane in lanes:
                x_min = lane.get("x_min_pct", 0)
                x_max = lane.get("x_max_pct", 100)
                allowed = [str(a).lower() for a in lane.get("allowed", [])]
                if x_min <= bc_x_pct <= x_max:
                    if allowed and "all" not in allowed:
                        # Normalize vehicle group matching
                        is_allowed = False
                        for a in allowed:
                            if a == v_type or a == "all":
                                is_allowed = True
                                break
                            if is_car_group and a in {"car", "truck", "bus", "vehicle", "automobile"}:
                                is_allowed = True
                                break
                            if is_motor_group and a in {"motorcycle", "motorbike", "motor", "bike", "rider"}:
                                is_allowed = True
                                break
                        if not is_allowed:
                            lane_name = lane.get("name", "Làn không được phép")
                            return True, f"Đi vào {lane_name}"
            return False, "Đúng làn quy định"

        # Default standard lane rules: conservative checks to avoid false positives
        if is_car_group and bc_x_pct > 96.0:
            return True, "Ô tô đi vào lề đường bên phải"
        if is_motor_group and bc_x_pct < 4.0:
            return True, "Xe máy đi vào giải phân cách bên trái"

        return False, "Đúng làn quy định"

    def _extract_plate_text(self, cap: Any, file_path: str, bbox: dict, t_sec: float = 0.0) -> tuple[str, str, str, Optional[dict]]:
        """
        Extract exact video frame at timestamp t_sec, crop vehicle region with 15% margin,
        run license_plate.pt to detect license plate box, crop plate, enhance image and run OCR.
        Returns tuple of (normalized_license_plate_text, vehicle_snapshot_url, plate_snapshot_url, plate_sub_bbox).
        """
        if not cv2:
            return "", "", "", None

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
            return "", "", "", None

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
            return "", "", "", None

        # Save cropped vehicle/target object snapshot image
        vehicle_snapshot_url = ""
        if vehicle_crop is not None and vehicle_crop.size > 0:
            try:
                vehicle_filename = f"crop_target_{uuid.uuid4().hex[:8]}.jpg"
                vehicle_path = os.path.join(os.path.dirname(file_path), vehicle_filename)
                cv2.imwrite(vehicle_path, vehicle_crop)
                vehicle_snapshot_url = f"/evidence/{vehicle_filename}"
            except Exception as e:
                print(f"[!] Failed to save vehicle crop snapshot: {e}")

        plate_text = ""
        plate_crop = None
        plate_sub_bbox = None
        plate_url = ""

        # BƯỚC 1: CAMERA / MÔ HÌNH BẮT ĐƯỢC HÌNH ẢNH BIỂN SỐ XE TRƯỚC (Detection Phase)
        # Sử dụng mô hình License_Plate.pt để tìm chính xác khung hình ảnh vùng biển số xe
        detected_plate = False
        if self.yolo.license_plate_model:
            try:
                plate_results = self.yolo.license_plate_model(vehicle_crop, conf=0.10)
                best_conf = 0.0
                best_p_crop = None
                best_sub_bbox = None

                for pr in plate_results:
                    for pbox in pr.boxes:
                        conf = float(pbox.conf[0])
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].tolist())
                        vw = max(1, vehicle_crop.shape[1])
                        vh = max(1, vehicle_crop.shape[0])

                        # Thêm lề (padding) 4px xung quanh vùng ảnh biển số đã bắt được
                        pad = 4
                        raw_p_crop = vehicle_crop[
                            max(0, py1 - pad):min(vh, py2 + pad),
                            max(0, px1 - pad):min(vw, px2 + pad)
                        ]

                        if raw_p_crop.size > 0 and conf > best_conf:
                            best_conf = conf
                            best_p_crop = raw_p_crop
                            best_sub_bbox = {
                                "x_pct": round((px1 / vw) * 100, 1),
                                "y_pct": round((py1 / vh) * 100, 1),
                                "w_pct": round(((px2 - px1) / vw) * 100, 1),
                                "h_pct": round(((py2 - py1) / vh) * 100, 1),
                                "conf": round(conf, 2),
                            }

                # BƯỚC 2: SAU KHI ĐÃ BẮT ĐƯỢC HÌNH ẢNH BIỂN SỐ -> MỚI TIẾN HÀNH XÁC ĐỊNH & ĐỌC OCR (Identification Phase)
                if best_p_crop is not None and best_p_crop.size > 0:
                    detected_plate = True
                    plate_crop = best_p_crop
                    plate_sub_bbox = best_sub_bbox

                    # Lưu ảnh snapshot vùng biển số đã bắt được để làm bằng chứng
                    try:
                        plate_filename = f"plate_{uuid.uuid4().hex[:8]}.jpg"
                        plate_path = os.path.join(os.path.dirname(file_path), plate_filename)
                        cv2.imwrite(plate_path, plate_crop)
                        plate_url = f"/evidence/{plate_filename}"
                    except Exception as save_err:
                        print(f"[!] Lỗi lưu ảnh chụp vùng biển số: {save_err}")

                    # Tiến hành tiền xử lý làm rõ ảnh & đọc chữ/số trên vùng ảnh biển số vừa bắt
                    enhanced_p = self._enhance_crop_for_ocr(plate_crop)
                    raw_txt = self.ocr.extract_text(enhanced_p)
                    norm_txt = self.ocr.normalize_plate_text(raw_txt)

                    if norm_txt:
                        plate_text = norm_txt
                    elif raw_txt and len(raw_txt.strip()) >= 3:
                        # Nếu đọc ra chuỗi nhưng chưa khớp regex chuẩn 100%, vẫn giữ chuỗi thô đã đọc
                        plate_text = raw_txt.strip().upper()

            except Exception as e:
                print(f"[!] Lỗi phát hiện/xác định biển số xe: {e}")

        # NẾU CAMERA KHÔNG BẮT ĐƯỢC HÌNH ẢNH BIỂN SỐ:
        # Trả về rỗng, KHÔNG thực hiện OCR mù ngẫu nhiên trên thân xe để tránh nhận diện sai
        if not detected_plate:
            return "", vehicle_snapshot_url, "", None

        return plate_text, vehicle_snapshot_url, plate_url, plate_sub_bbox

    def _check_no_helmet(self, cap: Any, file_path: str, bbox: dict, t_sec: float = 0.0) -> tuple[bool, Optional[dict]]:
        """Run Helmet.pt model on vehicle/rider crop to detect No-Helmet status and sub-bbox."""
        if not self.yolo.helmet_model or not cv2:
            return False, None
        target_frame = None
        if cap and hasattr(cap, 'isOpened') and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
            ret, target_frame = cap.read()
        elif os.path.exists(file_path) and self._source_type(file_path) == "image":
            target_frame = cv2.imread(file_path)

        if target_frame is None or not hasattr(target_frame, 'shape') or target_frame.size == 0:
            return False, None

        fh, fw = target_frame.shape[:2]
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", fw))
        y2 = int(bbox.get("y2", fh))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Pad top margin by 35% of bounding box height to ensure rider's head/helmet area is included
        pad_top = int(bh * 0.35)
        pad_side = int(bw * 0.15)

        vx1 = max(0, x1 - pad_side)
        vy1 = max(0, y1 - pad_top)
        vx2 = min(fw, x2 + pad_side)
        vy2 = min(fh, y2)

        crop = target_frame[vy1:vy2, vx1:vx2]

        if crop.size == 0 or crop.shape[0] < 35 or crop.shape[1] < 35:
            return False, None

        try:
            results = self.yolo.helmet_model(crop, conf=0.35)
            max_helmet_conf = 0.0
            max_no_helmet_conf = 0.0
            best_helmet_sub_box = None

            for r in results:
                boxes = getattr(r, "boxes", [])
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = str(self.yolo.helmet_model.names.get(cls_id, "")).lower()
                    hx1, hy1, hx2, hy2 = map(int, box.xyxy[0].tolist())
                    cw = max(1, crop.shape[1])
                    ch = max(1, crop.shape[0])

                    sub_b = {
                        "x_pct": round((hx1 / cw) * 100, 1),
                        "y_pct": round((hy1 / ch) * 100, 1),
                        "w_pct": round(((hx2 - hx1) / cw) * 100, 1),
                        "h_pct": round(((hy2 - hy1) / ch) * 100, 1),
                    }

                    if "no helmet" in cls_name or "no-helmet" in cls_name or cls_id == 1:
                        if conf > max_no_helmet_conf:
                            max_no_helmet_conf = conf
                            sub_b["status"] = "NO_HELMET"
                            best_helmet_sub_box = sub_b
                    else:
                        if conf > max_helmet_conf:
                            max_helmet_conf = conf
                            sub_b["status"] = "HELMET"
                            if not best_helmet_sub_box:
                                best_helmet_sub_box = sub_b

            # Nâng ngưỡng khắt khe: conf không đội mũ >= 0.55 và vượt trội hơn conf đội mũ ít nhất 0.15
            if max_no_helmet_conf >= 0.55 and max_no_helmet_conf > (max_helmet_conf + 0.15):
                return True, best_helmet_sub_box
            elif max_helmet_conf > 0.35:
                return False, best_helmet_sub_box

        except Exception as exc:
            print(f"[!] Helmet model check error: {exc}")
        return False, None

    def _get_traffic_light_status(self, frame: Any, traffic_light_boxes: list) -> str:
        """
        Dynamically inspect traffic light box in current frame to return 'RED', 'GREEN', or 'YELLOW'.
        Defaults to 'GREEN' if vehicles are actively flowing or green lamp HSV is detected.
        """
        if not traffic_light_boxes or not cv2 or frame is None or not hasattr(frame, "shape"):
            return "GREEN"

        fh, fw = frame.shape[:2]
        for t_box in traffic_light_boxes:
            tx1 = max(0, int(t_box.get("x1", 0)))
            ty1 = max(0, int(t_box.get("y1", 0)))
            tx2 = min(fw, int(t_box.get("x2", fw)))
            ty2 = min(fh, int(t_box.get("y2", fh)))
            t_crop = frame[ty1:ty2, tx1:tx2]

            if t_crop.size == 0:
                continue

            h_crop = t_crop.shape[0]
            top_crop = t_crop[:max(1, int(h_crop * 0.35)), :]
            bot_crop = t_crop[max(0, int(h_crop * 0.50)):, :]

            red_ratio = 0.0
            green_ratio = 0.0

            if top_crop.size > 0:
                hsv_top = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv_top, np.array([0, 100, 100]), np.array([10, 255, 255]))
                m2 = cv2.inRange(hsv_top, np.array([160, 100, 100]), np.array([180, 255, 255]))
                red_ratio = np.sum((m1 | m2) > 0) / float(max(1, top_crop.shape[0] * top_crop.shape[1]))

            if bot_crop.size > 0:
                hsv_bot = cv2.cvtColor(bot_crop, cv2.COLOR_BGR2HSV)
                m_g = cv2.inRange(hsv_bot, np.array([35, 60, 60]), np.array([95, 255, 255]))
                green_ratio = np.sum(m_g > 0) / float(max(1, bot_crop.shape[0] * bot_crop.shape[1]))

            if green_ratio > 0.03 or green_ratio >= red_ratio:
                return "GREEN"
            elif red_ratio >= 0.12 and red_ratio > (1.5 * green_ratio):
                return "RED"

        return "GREEN"

    def _is_red_light_active(self, frame: Any, traffic_light_boxes: list) -> bool:
        """
        Step 1: Verify traffic light box(es) were detected by YOLO in the frame.
        Step 2: Crop top 35% (Red lamp region) of vertical traffic light and perform HSV color inspection.
        Returns True ONLY if a traffic light is detected AND its top RED lamp is illuminated.
        """
        return self._get_traffic_light_status(frame, traffic_light_boxes) == "RED"

    async def process_file(self, camera_id: str, file_path: str, db: Optional[Any] = None) -> Dict[str, Any]:
        """Process an uploaded video/image synchronously and return detections & violations."""
        import asyncio
        cap = None
        if cv2 and self._source_type(file_path) == "video":
            cap = cv2.VideoCapture(file_path)

        preview_frame = None
        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        traffic_light_boxes = [d["bbox"] for d in infra_detections if d.get("class_name") in {"traffic_light", "traffic light"}]

        should_close_db = False
        if db is None:
            db = AsyncSessionLocal()
            should_close_db = True

        try:
            from sqlalchemy import select
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
                        config={"speed_limit": 60, "stop_line_y": 420},
                        vehicle_types=["car", "motorcycle", "bus", "truck"]
                    )
                    db.add(db_cam)
                    await db.commit()

            valid_camera_id = str(target_cam_uuid)
            seen_violation_keys = set()
            config_dict = db_cam.config or {} if db_cam else {}
            speed_limit = config_dict.get("speed_limit", 60)
            stop_line_y = config_dict.get("stop_line_y", 345)

            # Extract configured violation_zone (red zone past the blue stop-line into intersection/roundabout)
            v_zone = config_dict.get("violation_zone") or {}
            z_ymin = float(v_zone.get("y_min", 170))  # 24% of frame height (intersection/roundabout)
            z_ymax = float(v_zone.get("y_max", 345))  # 48% of frame height (stop line at traffic light)
            z_xmin = float(v_zone.get("x_min", 50))
            z_xmax = float(v_zone.get("x_max", 1230))
            violation_zone_dict = {"x_min": z_xmin, "x_max": z_xmax, "y_min": z_ymin, "y_max": z_ymax}

            from collections import defaultdict
            track_zone_history = defaultdict(list)

            for idx, raw in enumerate(raw_detections, start=1):
                metadata = dict(raw.get("metadata") or {})
                bbox = raw.get("bbox", {})
                obj_group = raw.get("object_group", metadata.get("object_group", "unknown"))
                class_name = raw.get("class_name", "unknown")
                t_sec = metadata.get("time", 0.0)
                track_id = metadata.get("track_id")

                if raw.get("bbox_pct"):
                    metadata["bbox_pct"] = raw["bbox_pct"]
                metadata["object_group"] = obj_group
                metadata["violation_zone"] = violation_zone_dict

                # Step 1 & 2: Detection & Violation check
                is_motor_vehicle = class_name.lower() in {"car", "truck", "bus", "motorcycle", "motorbike", "motor"}
                is_motorcycle = class_name.lower() in {"motorcycle", "motorbike", "motor", "bike", "rider"}

                violation_type = None
                ocr_plate, vehicle_snapshot_url, plate_snapshot_url = "", "", ""

                if is_motor_vehicle or is_motorcycle:
                    # 1. Check No-Helmet violation for motorcycles/riders
                    is_no_helmet = False
                    helmet_sub_box = None
                    if is_motorcycle:
                        is_no_helmet, helmet_sub_box = await asyncio.to_thread(self._check_no_helmet, cap, file_path, bbox, t_sec)
                        if helmet_sub_box:
                            metadata["helmet_sub_box"] = helmet_sub_box

                    # 2. Check Speeding violation (only if valid speed_kmh > speed_limit + 3.0 km/h)
                    speed_kmh = float(metadata.get("speed_kmh", 0.0) or 0.0)
                    is_speeding = (is_motor_vehicle and speed_kmh > 0 and speed_kmh > (speed_limit + 3.0))

                    # 3. Check Wrong Lane violation
                    is_wrong_lane, lane_desc = self._check_wrong_lane(class_name, bbox, 1280.0, 720.0, config_dict)
                    metadata["lane_info"] = {"description": lane_desc, "is_wrong_lane": is_wrong_lane}

                    # =========================================================================
                    # QUY TRÌNH PHÁT HIỆN VI PHẠM VƯỢT ĐÈN ĐỎ
                    # =========================================================================

                    # BƯỚC 1: PHÁT HIỆN ĐÈN GIAO THÔNG (Traffic Light Detection)
                    has_traffic_light = len(traffic_light_boxes) > 0

                    # BƯỚC 2: XÁC ĐỊNH TRẠNG THÁI ĐÈN (Determine Traffic Light Signal Status: RED)
                    frame_at_t = preview_frame
                    if cap and hasattr(cap, 'isOpened') and cap.isOpened() and t_sec > 0:
                        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
                        ret_t, frame_at_t = cap.read()
                        if not ret_t or frame_at_t is None:
                            frame_at_t = preview_frame

                    traffic_status = self._get_traffic_light_status(frame_at_t, traffic_light_boxes) if has_traffic_light else "GREEN"
                    is_red_signal = (traffic_status == "RED")
                    metadata["traffic_light_status"] = traffic_status

                    # BƯỚC 3 & 4: XÁC ĐỊNH VỊ TRÍ TÂM ĐÁY PHƯƠNG TIỆN SO VỚI VẠCH DỪNG ĐÈN GIAO THÔNG
                    bc_x = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2.0
                    bc_y = float(bbox.get("y2", 0))

                    # Phương tiện đã đi sâu qua Vạch dừng vào ngã tư trong lúc đèn Đỏ (stop_line_y = 345)
                    is_crossing_stop_line = (bc_y <= (stop_line_y - 25) and bc_y >= 100)

                    # BƯỚC 5: KIỂM TRA ĐIỀU KIỆN VI PHẠM ([ĐÈN: ĐỎ] AND [VỊ TRÍ: VƯỢT VẠCH DỪNG])
                    is_crossing_red_light = False
                    if is_motor_vehicle and has_traffic_light and is_red_signal and is_crossing_stop_line:
                        if track_id is not None:
                            track_zone_history[track_id].append((idx, t_sec, is_crossing_stop_line, is_red_signal))
                            recent_red_zone_frames = [s for s in track_zone_history[track_id] if s[2] and s[3]]
                            if len(recent_red_zone_frames) >= 2:
                                is_crossing_red_light = True
                        else:
                            is_crossing_red_light = True

                    # BƯỚC 6: BÁO CÁO VI PHẠM (CHỈ THỰC HIỆN KHI THỰC SỰ VI PHẠM)
                    if is_no_helmet:
                        violation_type = "NO_HELMET"
                    elif is_speeding:
                        violation_type = "SPEEDING"
                    elif is_crossing_red_light:
                        violation_type = "RED_LIGHT"
                        print(f"[XÁC NHẬN VƯỢT ĐÈN ĐỎ] Frame {idx:05d} | TrackID: {track_id} | Xe: {class_name} | Tâm đáy: ({bc_x:.1f},{bc_y:.1f})")
                    elif is_wrong_lane:
                        violation_type = "WRONG_LANE"
                        print(f"[XÁC NHẬN ĐI SAI LÀN ĐƯỜNG] Frame {idx:05d} | TrackID: {track_id} | Xe: {class_name} | Mô tả: {lane_desc}")

                    # CHỈ GỌI ALPR (BẮT BIỂN SỐ & ĐỌC OCR) KHI PHƯƠNG TIỆN THỰC SỰ VI PHẠM!
                    if violation_type is not None:
                        ocr_plate, vehicle_snapshot_url, plate_snapshot_url, plate_sub_bbox = await asyncio.to_thread(
                            self._extract_plate_text, cap, file_path, bbox, t_sec
                        )
                        if plate_sub_bbox:
                            metadata["plate_sub_box"] = plate_sub_bbox

                if vehicle_snapshot_url:
                    metadata["target_snapshot_url"] = vehicle_snapshot_url
                if plate_snapshot_url:
                    metadata["plate_snapshot_url"] = plate_snapshot_url

                if ocr_plate:
                    metadata["license_plate"] = ocr_plate
                    metadata["license_plate_source"] = "license_plate_pt_ocr"
                else:
                    metadata.pop("license_plate", None)
                    metadata["license_plate_source"] = "not_requested_no_violation" if (is_motor_vehicle or is_motorcycle) else "not_vehicle"

                if violation_type:
                    metadata["is_violation"] = True
                    metadata["violation_type"] = violation_type

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

                if (is_motor_vehicle or is_motorcycle) and violation_type:
                    v_key = (track_id, violation_type) if track_id is not None else (ocr_plate or class_name, violation_type, bbox.get("x1", 0) // 100)
                    
                    if v_key in seen_violation_keys:
                        print(f"[FRAME {idx:05d}] Track ID: {track_id} | Class: {class_name} | Violation: {violation_type} | Already Violated: YES -> SKIP INSERT")
                    else:
                        seen_violation_keys.add(v_key)
                        print(f"[FRAME {idx:05d}] Track ID: {track_id} | Class: {class_name} | Violation: {violation_type} | Conf: {raw['confidence']} | Already Violated: NO -> INSERT VIOLATION")
                        
                        viol_meta = {
                            "object_group": obj_group,
                            "track_id": track_id,
                        }
                        if vehicle_snapshot_url:
                            viol_meta["target_snapshot_url"] = vehicle_snapshot_url
                        if plate_snapshot_url:
                            viol_meta["plate_snapshot_url"] = plate_snapshot_url

                        violation_in = {
                            "detection_id": str(db_detection.id),
                            "camera_id": valid_camera_id,
                            "violation_type": violation_type,
                            "vehicle_type": class_name,
                            "license_plate": ocr_plate or None,
                            "confidence": raw["confidence"],
                            "evidence_url": vehicle_snapshot_url or plate_snapshot_url or f"/evidence/{os.path.basename(file_path)}",
                            "metadata": viol_meta,
                        }
                        db_violation = await create_violation(db, ViolationCreate(**violation_in))  # type: ignore
                        generated_violations.append(db_violation)

            # Ensure all created detections and violations are committed to DB
            await db.commit()
        finally:
            if should_close_db:
                await db.close()

        if cap and hasattr(cap, "release"):
            cap.release()

        # Return all detected and tracked ByteTrack objects (vehicles, persons, infrastructure) for full multi-object overlay
        display_detections = saved_detections

        # Sort so VIOLATING VEHICLES appear first at the top of the results list
        display_detections.sort(
            key=lambda d: 1 if (d.metadata_ or {}).get("is_violation") or (d.metadata_ or {}).get("violation_type") else 0,
            reverse=True
        )

        return {
            "success": True,
            "source_url": f"/evidence/{os.path.basename(file_path)}",
            "source_type": self._source_type(file_path),
            "preview_url": self._preview_url(file_path, preview_frame),
            "count": len(display_detections),
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
                    "class_name": d.vehicle_type,
                    "vehicle_type": d.vehicle_type,
                    "object_group": (d.metadata_ or {}).get("object_group", "vehicle"),
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "bbox_pct": (d.metadata_ or {}).get("bbox_pct") or {
                        "x": round(((d.bbox.get("x1", 100) if d.bbox else 100) / 12.8), 1),
                        "y": round(((d.bbox.get("y1", 200) if d.bbox else 200) / 7.2), 1),
                        "w": round((((d.bbox.get("x2", 400) if d.bbox else 400) - (d.bbox.get("x1", 100) if d.bbox else 100)) / 12.8), 1),
                        "h": round((((d.bbox.get("y2", 500) if d.bbox else 500) - (d.bbox.get("y1", 200) if d.bbox else 200)) / 7.2), 1)
                    },
                    "metadata": {
                        "track_id": (d.metadata_ or {}).get("track_id"),
                        "time": (d.metadata_ or {}).get("time", 0.0),
                        "speed_kmh": (d.metadata_ or {}).get("speed_kmh", 0.0),
                        **(d.metadata_ or {}),
                    },
                    "detected_at": d.detected_at.isoformat() if d.detected_at else None,
                }
                for d in display_detections
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


