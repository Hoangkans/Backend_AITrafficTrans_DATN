import re
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

from app.core.config import settings


class OCRService:
    """License plate OCR engine using EasyOCR with pytesseract fallback."""

    def __init__(self):
        self.cmd = settings.TESSERACT_CMD
        self.lang = settings.OCR_LANG
        self._easyocr_reader = None
        if self.cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = self.cmd

    @property
    def reader(self):
        if self._easyocr_reader is None and easyocr:
            try:
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as exc:
                print(f"[!] Failed to initialize EasyOCR: {exc}")
        return self._easyocr_reader

    def normalize_plate_text(self, text: str) -> str:
        raw = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
        if len(raw) < 4:
            return ""

        digit_map = str.maketrans(
            {
                "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                "Z": "2", "S": "5", "B": "8", "G": "6",
            }
        )

        # Regex trích xuất định dạng chuẩn biển số Việt Nam (VD: 59X1-123.45 hoặc 29A-123.45)
        match = re.match(r"^(\d{2})([A-Z][A-Z0-9]?)(.+)$", raw)
        if match:
            prov = match.group(1)
            series = match.group(2)
            raw_digits = re.sub(r"[^0-9]", "", match.group(3).translate(digit_map))

            if len(raw_digits) == 5:
                return f"{prov}{series}-{raw_digits[:3]}.{raw_digits[3:]}"
            elif len(raw_digits) == 4:
                return f"{prov}{series}-{raw_digits}"
            elif len(raw_digits) > 5:
                digits_trimmed = raw_digits[:5]
                return f"{prov}{series}-{digits_trimmed[:3]}.{digits_trimmed[3:]}"

        char1 = raw[0] if raw[0].isdigit() else "2"
        char2 = raw[1] if (len(raw) > 1 and raw[1].isdigit()) else "9"
        prefix_raw = char1 + char2
        
        series = "A"
        if len(raw) > 2 and raw[2].isalpha():
            series = raw[2]

        number_raw = raw[3:] if len(raw) > 3 else raw[2:]
        suffix = re.sub(r"[^0-9]", "", number_raw.translate(digit_map))
        
        if len(suffix) >= 5:
            return f"{prefix_raw}{series}-{suffix[:3]}.{suffix[3:5]}"
        elif len(suffix) >= 2:
            return f"{prefix_raw}{series}-{suffix}"
        
        return raw[:9]

    def extract_text(self, image: Any) -> str:
        if image is None:
            return ""

        # Image preprocessing variants: Original & CLAHE contrast enhanced
        images_to_try = [image]
        if cv2 and hasattr(image, 'shape') and len(image.shape) == 3:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                images_to_try.append(enhanced)
            except Exception:
                pass

        # Step 1: EasyOCR extraction on image variants
        if self.reader is not None:
            for img_var in images_to_try:
                try:
                    ocr_results = self.reader.readtext(img_var, detail=0)
                    combined = "".join(ocr_results)
                    normalized = self.normalize_plate_text(combined)
                    if normalized:
                        return normalized
                except Exception as exc:
                    print(f"[!] EasyOCR extraction error: {exc}")

        # Step 2: pytesseract extraction fallback
        if pytesseract:
            for img_var in images_to_try:
                try:
                    if cv2 and len(img_var.shape) == 3:
                        gray_var = cv2.cvtColor(img_var, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_var = img_var
                    
                    resized = cv2.resize(gray_var, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC) if cv2 else gray_var
                    _, thresholded = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if cv2 else (None, resized)

                    for psm in (7, 8, 13, 6):
                        text = pytesseract.image_to_string(
                            thresholded,
                            lang=self.lang,
                            config="--psm " + str(psm) + " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
                        )
                        normalized = self.normalize_plate_text(text)
                        if normalized:
                            return normalized
                except Exception:
                    pass

        return ""


ocr_service = OCRService()
