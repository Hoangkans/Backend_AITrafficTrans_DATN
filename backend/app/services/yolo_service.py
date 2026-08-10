import os
from typing import List, Dict, Any
from app.core.config import settings

class YOLOService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.model = None
        self.model_available = False
        
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model_available = True
                print(f"[+] YOLO Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"[!] Failed to load YOLO model: {e}. Detection will return no results.")
        else:
            print(f"[!] YOLO Model file not found at '{self.model_path}'. Detection will return no results.")

    def detect(self, image_data_or_path: Any, conf_threshold: float = None) -> List[Dict[str, Any]]:
        threshold = conf_threshold or settings.YOLO_CONFIDENCE_THRESHOLD
        
        if self.model_available and self.model:
            try:
                results = self.model(image_data_or_path, conf=threshold)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        # Map to supported vehicle types in database design
                        # DB accepts: 'car', 'truck', 'bus', 'motorcycle', 'bicycle'
                        vehicle_map = {
                            "car": "car",
                            "truck": "truck",
                            "bus": "bus",
                            "motorbike": "motorcycle",
                            "motorcycle": "motorcycle",
                            "bicycle": "bicycle",
                            "person": "bicycle" # fallback/dummy mapping if any
                        }
                        vehicle_type = vehicle_map.get(label, "car")
                        
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        detections.append({
                            "class_id": cls_id,
                            "class_name": vehicle_type,
                            "confidence": conf,
                            "bbox": {
                                "x1": int(xyxy[0]),
                                "y1": int(xyxy[1]),
                                "x2": int(xyxy[2]),
                                "y2": int(xyxy[3])
                            }
                        })
                return detections
            except Exception as e:
                print(f"[-] YOLO Inference failed: {e}.")

        return []

yolo_service = YOLOService()
