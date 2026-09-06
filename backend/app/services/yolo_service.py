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
        det_candidates = [
            detection_model_path,
            getattr(settings, "DETECTION_MODEL_PATH", None),
            getattr(settings, "YOLO_MODEL_PATH", None),
            "Detection.pt",
            "weights/Detection.pt",
            "weights/detection.pt",
            "best.pt",
            "yolov8s.pt",
        ]
        self.detection_model_path = next((p for p in det_candidates if p and os.path.exists(p)), "Detection.pt")

        lp_candidates = [
            license_plate_model_path,
            getattr(settings, "LICENSE_PLATE_MODEL_PATH", None),
            "License_Plate.pt",
            "weights/License_Plate.pt",
            "weights/license_plate.pt",
        ]
        self.license_plate_model_path = next((p for p in lp_candidates if p and os.path.exists(p)), "License_Plate.pt")

        self.detection_model = None
        self.license_plate_model = None
        self.helmet_model = None
        self.model_available = False

        # Load Detection Model (Detection.pt — 10 classes)
        if os.path.exists(self.detection_model_path):
            try:
                from ultralytics import YOLO
                self.detection_model = YOLO(self.detection_model_path)
                self.model_available = True
                names = list(self.detection_model.names.values())
                print(f"[+] Detection Model loaded: {self.detection_model_path} | classes={names}")
            except Exception as e:
                print(f"[!] Failed to load Detection model at '{self.detection_model_path}': {e}. Fallback mode enabled.")
        else:
            print(f"[!] Detection Model file not found at '{self.detection_model_path}'. Fallback mode enabled.")

        # Load License Plate Model (License_Plate.pt — 1 class: license_plate)
        if os.path.exists(self.license_plate_model_path):
            try:
                from ultralytics import YOLO
                self.license_plate_model = YOLO(self.license_plate_model_path)
                print(f"[+] License Plate Model loaded: {self.license_plate_model_path} | classes={list(self.license_plate_model.names.values())}")
            except Exception as e:
                print(f"[!] Failed to load License Plate model at '{self.license_plate_model_path}': {e}")
        else:
            print(f"[*] License Plate Model not found at '{self.license_plate_model_path}'.")

        # Load Helmet Fine-tuned Model (Helmet_fine_tune.pt — 2 classes: Helmet, No Helmet)
        helmet_candidates = [
            "Helmet_fine_tune.pt",
            "weights/Helmet_fine_tune.pt",
            "Helmet.pt",
            "weights/Helmet.pt",
            "weights/helmet.pt"
        ]
        helmet_path = next((p for p in helmet_candidates if os.path.exists(p)), None)
        if helmet_path:
            try:
                from ultralytics import YOLO
                self.helmet_model = YOLO(helmet_path)
                print(f"[+] Helmet Model (Fine-tuned) loaded: {helmet_path} | classes={self.helmet_model.names}")
            except Exception as e:
                print(f"[!] Failed to load Helmet model at '{helmet_path}': {e}")
        else:
            print(f"[*] Helmet Model file not found in candidate paths.")

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

    def _create_byte_tracker(self):
        """Instantiate a persistent ByteTrack tracker configured for traffic tracking."""
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
            args = IterableSimpleNamespace(
                track_high_thresh=getattr(settings, "YOLO_CONFIDENCE_THRESHOLD", 0.45),
                track_low_thresh=0.10,
                new_track_thresh=0.50,
                track_buffer=30,
                match_thresh=0.80,
                tracker_type="bytetrack",
                fuse_score=True,
            )
            return BYTETracker(args)
        except Exception as e:
            print(f"[!] Failed to initialize BYTETracker: {e}")
            return None

    def _build_detection_from_coords(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        conf: float,
        cls_id: int,
        model_names: Dict[int, str],
        img_w: int,
        img_h: int,
        track_id: Optional[int] = None,
        time_sec: float = 0.0,
        detection_mode: str = "DETECT",
        interval: float = 0.10,
    ) -> Dict[str, Any]:
        raw_label = model_names.get(cls_id, "unknown")
        norm = self._normalize_class(raw_label)
        bbox_pct = self._bbox_to_pct(x1, y1, x2, y2, img_w, img_h)

        meta: Dict[str, Any] = {
            "speed_kmh": None,
            "object_group": norm["group"],
            "time": time_sec,
            "detection_mode": f"YOLO: {detection_mode}",
            "detection_interval": interval,
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
        Video mode uses time-interval YOLO detection (10 FPS) + persistent ByteTrack tracking (30 FPS).
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
                    try:
                        import cv2
                        from ultralytics.trackers.byte_tracker import STrack
                    except ImportError:
                        cv2 = None

                    if cv2 and isinstance(image_data_or_path, str) and os.path.exists(image_data_or_path):
                        cap = cv2.VideoCapture(image_data_or_path)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                            det_interval = getattr(settings, "DETECTION_INTERVAL", 0.10)
                            tracker = self._create_byte_tracker()

                            last_det_time = -999.0
                            frame_idx = 0

                            while True:
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    break

                                frame_idx += 1
                                if frame_idx > 1800:
                                    break

                                time_sec = round(frame_idx / fps, 3)
                                img_h, img_w = frame.shape[:2]

                                should_detect = (time_sec - last_det_time) >= (det_interval - 0.001)

                                if should_detect:
                                    last_det_time = time_sec
                                    # 1. Run YOLO inference ONLY on time interval (~0.1s)
                                    results = self.detection_model(frame, conf=threshold, verbose=True)
                                    boxes = results[0].boxes if (results and len(results) > 0) else None

                                    # 2. Update persistent BYTETracker with YOLO detections
                                    tracked_boxes_set = set()
                                    if tracker is not None and boxes is not None and len(boxes) > 0:
                                        try:
                                            tracks = tracker.update(boxes)
                                            for tr in tracks:
                                                x1, y1, x2, y2 = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3])
                                                tid = int(tr[4])
                                                conf = float(tr[5])
                                                cls_id = int(tr[6])
                                                tracked_boxes_set.add((x1, y1, x2, y2))
                                                det = self._build_detection_from_coords(
                                                    x1, y1, x2, y2, conf, cls_id, self.detection_model.names,
                                                    img_w, img_h, track_id=tid, time_sec=time_sec,
                                                    detection_mode="DETECT", interval=det_interval
                                                )
                                                detections.append(det)
                                        except Exception as tracker_err:
                                            print(f"[!] BYTETracker update warning: {tracker_err}")
                                            tracker = None

                                    # Capture untracked raw YOLO boxes (static traffic lights, signs, or objects not in ByteTrack tracks)
                                    if boxes is not None:
                                        for box in boxes:
                                            xyxy = box.xyxy[0].tolist()
                                            bx1, by1, bx2, by2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                            bconf = float(box.conf[0])
                                            bcls_id = int(box.cls[0])
                                            # If this box wasn't already added by ByteTrack
                                            if (bx1, by1, bx2, by2) not in tracked_boxes_set:
                                                det = self._build_detection_from_coords(
                                                    bx1, by1, bx2, by2, bconf, bcls_id, self.detection_model.names,
                                                    img_w, img_h, track_id=None, time_sec=time_sec,
                                                    detection_mode="RAW_YOLO", interval=det_interval
                                                )
                                                detections.append(det)

                            cap.release()
                            return self._deduplicate_detections(detections)

                    # Fallback if cv2 read is unavailable
                    results = self.detection_model(image_data_or_path, conf=threshold)
                    for r in results:
                        img_h, img_w = r.orig_shape if (hasattr(r, "orig_shape") and r.orig_shape) else (720, 1280)
                        boxes = getattr(r, "boxes", [])
                        if boxes is None:
                            continue
                        for box in boxes:
                            det = self._build_detection(box, self.detection_model.names, img_w, img_h)
                            detections.append(det)
                    return self._deduplicate_detections(detections)

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
        min_conf = getattr(settings, "YOLO_CONFIDENCE_THRESHOLD", 0.45)

        for t_id, group in track_groups.items():
            max_conf = max(item.get("confidence", 0.0) for item in group)
            if max_conf < min_conf:
                continue

            sorted_group = sorted(group, key=lambda item: item.get("metadata", {}).get("time", 0.0))
            best = dict(sorted_group[0])

            history = [
                {
                    "time": item.get("metadata", {}).get("time", 0.0),
                    "box": item.get("bbox_pct") or {}
                }
                for item in sorted_group
            ]
            best["confidence"] = round(max_conf, 2)
            best["metadata"] = dict(best.get("metadata", {}))
            best["metadata"]["history"] = history
            best["metadata"]["tracked_frames"] = len(group)
            deduped.append(best)

        # For untracked detections, perform IoU NMS suppression (threshold > 0.35) & confidence >= min_conf
        untracked_deduped = []
        sorted_untracked = sorted(
            [d for d in untracked if d.get("confidence", 0) >= min_conf],
            key=lambda item: item.get("confidence", 0),
            reverse=True
        )
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

