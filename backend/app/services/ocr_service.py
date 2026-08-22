import re
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

if pytesseract is None:
    import warnings

    warnings.warn(
        "pytesseract is not installed. OCR functionality will be disabled. "
        "Run `pip install pytesseract` and ensure the Tesseract OCR engine is installed."
    )

from app.core.config import settings


class OCRService:
    """Thin wrapper around pytesseract OCR for license plate recognition."""

    def __init__(self):
        self.cmd = settings.TESSERACT_CMD
        self.lang = settings.OCR_LANG
        if self.cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = self.cmd

    def normalize_plate_text(self, text: str) -> str:
        raw = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
        if len(raw) < 6:
            return ""

        # First char MUST be a digit (1-9). This immediately filters out English words like "SPEED", "SIGNAL", "HIGHWAY".
        if not raw[0].isdigit() or raw[0] == "0":
            return ""

        digit_map = str.maketrans(
            {
                "O": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "L": "1",
                "Z": "2",
                "S": "5",
                "B": "8",
                "G": "6",
            }
        )

        char1 = raw[0]
        char2 = raw[1] if raw[1].isdigit() else raw[1].translate(digit_map)
        if not char2.isdigit():
            return ""

        prefix_raw = char1 + char2
        prefix_num = int(prefix_raw)
        if prefix_num < 11 or prefix_num > 99:
            return ""

        series = raw[2]
        if not series.isalpha():
            return ""

        number_raw = raw[3:]
        if len(number_raw) > 5 and len(raw) >= 8 and raw[3].isalpha():
            series += raw[3]
            number_raw = raw[4:]

        suffix = number_raw.translate(digit_map)

        if not suffix.isdigit() or len(suffix) < 4 or len(suffix) > 5:
            return ""

        return f"{prefix_raw}{series}-{suffix[:5]}"



    def extract_text(self, image: Any) -> str:
        if not pytesseract:
            return ""

        if cv2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 7, 75, 75)
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            processed = image

        try:
            for psm in (7, 8, 13):
                text = pytesseract.image_to_string(
                    processed,
                    lang=self.lang,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
                    ),
                )
                normalized = self.normalize_plate_text(text)
                if normalized:
                    return normalized
            return ""
        except Exception as exc:
            print(f"[!] OCR error: {exc}")
            return ""
