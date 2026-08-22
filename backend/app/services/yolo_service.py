import os
from typing import List, Dict, Any, Optional
from app.core.config import settings

# Mapping of detection.pt class names → normalized internal names
DETECTION_CLASS_MAP: Dict[str, Dict[str, str]] = {
    # Vehicles
    "car":           {"type": "car",           "group": "vehicle"},
    "bus":           {"type": "bus",           "group": "vehicle"},
    "truck":         {"type": "truck",         "group": "vehicle"},
    "motorbike":     {"type": "motorcycle",    "group": "vehicle"},
    "motorcycle":    {"type": "motorcycle",    "group": "vehicle"},
    "motor":         {"type": "motorcycle",    "group": "vehicle"},
    "bike":          {"type": "bicycle",       "group": "vehicle"},
    "bicycle":       {"type": "bicycle",       "group": "vehicle"},
    "train":         {"type": "train",         "group": "vehicle"},
    # People
    "person":        {"type": "person",        "group": "person"},
    "rider":         {"type": "rider",         "group": "person"},
    # Infrastructure
    "traffic light": {"type": "traffic_light", "group": "infrastructure"},
    "traffic sign":  {"type": "traffic_sign",  "group": "infrastructure"},
}


class YOLOService:
    def __init__(
        self,
        detection_model_path: str = None,
        license_plate_model_path: str = None
    ):
        self.detection_model_path = detection_model_path or getattr(settings, "DETECTION_MODEL_PATH", settings.YOLO_MODEL_PATH)
        self.license_plate_model_path = license_plate_model_path or getattr(settings, "LICENSE_PLATE_MODEL_PATH", "weights/license_plate.pt")

        self.detection_model = None
        self.license_plate_model = None
        self.model_available = False

        # Load Detection Model (detection.pt — 10 classes)
        if os.path.exists(self.detection_model_path):
            try:
                from ultralytics import YOLO
                self.detection_model = YOLO(self.detection_model_path)
                self.model_available = True
                names = list(self.detection_model.names.values())
                print(f"[+] Detection Model loaded: {self.detection_model_path} | classes={names}")
            except Exception as e:
                print(f"[!] Failed to load Detection model: {e}. Fallback detection mode enabled.")
        else:
            print(f"[!] Detection Model file not found at '{self.detection_model_path}'. Fallback mode enabled.")

        # Load License Plate Model (license_plate.pt — 1 class: license_plate)
        if os.path.exists(self.license_plate_model_path):
            try:
                from ultralytics import YOLO
                self.license_plate_model = YOLO(self.license_plate_model_path)
                print(f"[+] License Plate Model loaded: {self.license_plate_model_path}")
            except Exception as e:
                print(f"[!] Failed to load License Plate model: {e}")
        else:
            print(f"[*] License Plate Model not found at '{self.license_plate_model_path}'.")

        # Backward compat alias
        self.model = self.detection_model

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_class(raw_label: str) -> Dict[str, str]:
        """Return normalized {type, group} for a raw YOLO class label."""
        return DETECTION_CLASS_MAP.get(raw_label.lower(), {"type": raw_label, "group": "unknown"})

    @staticmethod
    def _bbox_to_pct(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Dict[str, float]:
        return {
            "x": round((x1 / img_w) * 100, 1),
            "y": round((y1 / img_h) * 100, 1),
            "w": round(((x2 - x1) / img_w) * 100, 1),
            "h": round(((y2 - y1) / img_h) * 100, 1),
        }

    def _build_detection(
        self,
        box,
        model_names: Dict[int, str],
        img_w: int,
        img_h: int,
        track_id: Optional[int] = None,
        speed: float = 0.0,
        time_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Convert a YOLO box object into a unified detection dict."""
        cls_id = int(box.cls[0])
        raw_label = model_names[cls_id]
        norm = self._normalize_class(raw_label)

        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        bbox_pct = self._bbox_to_pct(x1, y1, x2, y2, img_w, img_h)

        meta: Dict[str, Any] = {
            "speed_kmh": speed,
            "object_group": norm["group"],
            "time": time_sec,
        }
        if track_id is not None:
            meta["track_id"] = track_id

        return {
            "class_id": cls_id,
            "class_name": norm["type"],
            "object_group": norm["group"],
            "confidence": round(conf, 2),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "bbox_pct": bbox_pct,
            "metadata": meta,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, image_data_or_path: Any, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Run detection on an image or video file.
        Returns unified detection dicts for ALL 10 classes (vehicles + persons + infrastructure).
        Video mode uses ByteTrack for consistent track_ids across frames.
        """
        threshold = conf_threshold or settings.YOLO_CONFIDENCE_THRESHOLD

        if self.model_available and self.detection_model:
            try:
                is_video = False
                if isinstance(image_data_or_path, str):
                    ext = os.path.splitext(image_data_or_path)[1].lower()
                    if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                        is_video = True

                detections: List[Dict[str, Any]] = []

                if is_video:
                    # Video mode: ByteTrack tracking across frames
                    frame_count = 0
                    skip_interval = max(5, getattr(settings, "YOLO_FRAME_SKIP", 5))
                    try:
                        results = self.detection_model.track(
                            source=image_data_or_path,
                            tracker="bytetrack.yaml",
                            stream=True,
                            conf=threshold,
                            imgsz=1280,
                            vid_stride=skip_interval,
                            persist=True,
                        )
                    except Exception as track_err:
                        print(f"[!] ByteTrack error, fallback to standard detect: {track_err}")
                        results = self.detection_model(
                            image_data_or_path, stream=True, conf=threshold, imgsz=1280, vid_stride=skip_interval
                        )

                    for r in results:
                        frame_count += skip_interval
                        if frame_count > 900:
                            break

                        time_sec = round(frame_count / 30.0, 2)
                        img_h, img_w = r.orig_shape if (hasattr(r, "orig_shape") and r.orig_shape) else (720, 1280)
                        boxes = getattr(r, "boxes", [])
                        if boxes is None:
                            continue

                        for box in boxes:
                            tid = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None
                            det = self._build_detection(
                                box, self.detection_model.names, img_w, img_h, track_id=tid, time_sec=time_sec
                            )
                            detections.append(det)

                    detections = self._deduplicate_detections(detections)

                else:
                    # Image mode
                    results = self.detection_model(image_data_or_path, conf=threshold)
                    for r in results:
                        img_h, img_w = r.orig_shape if (hasattr(r, "orig_shape") and r.orig_shape) else (720, 1280)
                        boxes = getattr(r, "boxes", [])
                        if boxes is None:
                            continue
                        for box in boxes:
                            det = self._build_detection(box, self.detection_model.names, img_w, img_h)
                            detections.append(det)
                    detections = self._deduplicate_detections(detections)

                # Always return genuine detection.pt model results when model is loaded
                return detections

            except Exception as e:
                print(f"[-] YOLO Inference failed: {e}.")
                return []

        # Fallback simulation (realistic positions, diverse classes with motion history)
        return [
            {
                "class_id": 2,
                "class_name": "car",
                "object_group": "vehicle",
                "confidence": 0.94,
                "bbox": {"x1": 150, "y1": 280, "x2": 450, "y2": 600},
                "bbox_pct": {"x": 10.0, "y": 38.0, "w": 25.0, "h": 44.0},
                "metadata": {
                    "speed_kmh": 85,
                    "object_group": "vehicle",
                    "history": [
                        {"time": 0.0, "box": {"x": 10.0, "y": 38.0, "w": 25.0, "h": 44.0}},
                        {"time": 2.5, "box": {"x": 18.0, "y": 42.0, "w": 23.0, "h": 41.0}},
                        {"time": 5.0, "box": {"x": 28.0, "y": 48.0, "w": 21.0, "h": 37.0}},
                        {"time": 7.5, "box": {"x": 40.0, "y": 56.0, "w": 19.0, "h": 33.0}},
                        {"time": 10.0, "box": {"x": 55.0, "y": 66.0, "w": 16.0, "h": 28.0}},
                    ]
                },
            },
            {
                "class_id": 6,
                "class_name": "motorcycle",
                "object_group": "vehicle",
                "confidence": 0.88,
                "bbox": {"x1": 800, "y1": 300, "x2": 1050, "y2": 650},
                "bbox_pct": {"x": 62.0, "y": 42.0, "w": 20.0, "h": 48.0},
                "metadata": {
                    "speed_kmh": 45,
                    "object_group": "vehicle",
                    "history": [
                        {"time": 0.0, "box": {"x": 62.0, "y": 42.0, "w": 20.0, "h": 48.0}},
                        {"time": 3.0, "box": {"x": 58.0, "y": 48.0, "w": 19.0, "h": 44.0}},
                        {"time": 6.0, "box": {"x": 52.0, "y": 56.0, "w": 18.0, "h": 40.0}},
                        {"time": 10.0, "box": {"x": 44.0, "y": 68.0, "w": 16.0, "h": 35.0}},
                    ]
                },
            },
        ]

    def detect_license_plate_on_crop(
        self, vehicle_crop: Any, conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Run license_plate.pt on a pre-cropped vehicle numpy array.
        Returns plate bboxes (coords relative to the crop) with confidence scores.
        Lower conf_threshold (0.25) used because crops are smaller than full frames.
        """
        if self.license_plate_model is None:
            return []
        try:
            results = self.license_plate_model(vehicle_crop, conf=conf_threshold)
            plates = []
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    plates.append({
                        "bbox": {
                            "x1": int(xyxy[0]),
                            "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]),
                            "y2": int(xyxy[3]),
                        },
                        "confidence": float(box.conf[0]),
                    })
            return plates
        except Exception as e:
            print(f"[!] License-plate crop detection error: {e}")
            return []

    def detect_license_plate(self, image_path: str) -> list[dict]:
        """Detect license-plate bounding boxes from a full image file path.
        Returns list of dicts with keys: bbox (x1,y1,x2,y2) and confidence.
        """
        if not os.path.exists(self.license_plate_model_path):
            print(f"[!] License-plate model not found at '{self.license_plate_model_path}'.")
            return []
        try:
            from ultralytics import YOLO
            plate_model = YOLO(self.license_plate_model_path)
            results = plate_model(image_path, conf=settings.YOLO_CONFIDENCE_THRESHOLD)
            plates = []
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    plates.append({
                        "bbox": {
                            "x1": int(xyxy[0]),
                            "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]),
                            "y2": int(xyxy[3]),
                        },
                        "confidence": float(box.conf[0]),
                    })
            return plates
        except Exception as e:
            print(f"[!] License-plate detection error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Deduplication helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_iou(self, boxA: dict, boxB: dict) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        if not boxA or not boxB:
            return 0.0
        xA = max(boxA.get("x1", 0), boxB.get("x1", 0))
        yA = max(boxA.get("y1", 0), boxB.get("y1", 0))
        xB = min(boxA.get("x2", 0), boxB.get("x2", 0))
        yB = min(boxA.get("y2", 0), boxB.get("y2", 0))
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA.get("x2", 0) - boxA.get("x1", 0)) * (boxA.get("y2", 0) - boxA.get("y1", 0)))
        boxBArea = max(1, (boxB.get("x2", 0) - boxB.get("x1", 0)) * (boxB.get("y2", 0) - boxB.get("y1", 0)))
        return interArea / float(boxAArea + boxBArea - interArea)

    def _deduplicate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. Group by ByteTrack track_id → keep best-confidence representative per track.
        2. Untracked detections → IoU-NMS suppression (IoU threshold 0.35).
        """
        if not detections:
            return []

        track_groups: Dict[int, List[Dict[str, Any]]] = {}
        untracked: List[Dict[str, Any]] = []

        for d in detections:
            t_id = d.get("metadata", {}).get("track_id")
            if t_id is not None:
                track_groups.setdefault(t_id, []).append(d)
            else:
                untracked.append(d)

        deduped = []
        for t_id, group in track_groups.items():
            best = dict(max(group, key=lambda item: item.get("confidence", 0)))
            sorted_group = sorted(group, key=lambda item: item.get("metadata", {}).get("time", 0))
            history = [
                {
                    "time": item.get("metadata", {}).get("time", 0.0),
                    "box": item.get("bbox_pct", {})
                }
                for item in sorted_group
                if item.get("bbox_pct")
            ]
            best["metadata"] = dict(best.get("metadata", {}))
            best["metadata"]["history"] = history
            deduped.append(best)

        # For untracked detections, perform IoU NMS suppression (threshold > 0.35)
        untracked_deduped = []
        sorted_untracked = sorted(untracked, key=lambda item: item.get("confidence", 0), reverse=True)
        for d in sorted_untracked:
            box = d.get("bbox", {})
            cls_name = d.get("class_name")
            overlap = False
            for existing in untracked_deduped + deduped:
                if existing.get("class_name") == cls_name:
                    iou = self._compute_iou(box, existing.get("bbox", {}))
                    if iou > 0.35:
                        overlap = True
                        break
            if not overlap:
                untracked_deduped.append(d)

        return deduped + untracked_deduped


yolo_service = YOLOService()

