"""
Tesseract backend — pretrained, alternative to EasyOCR (see settings.OCR_ENGINE).

NOTE: pytesseract's simple `image_to_string` API does not return a
per-call confidence score, so a constant PLACEHOLDER confidence is used
whenever text is found. For real confidence values, `image_to_data` could
be used instead — left as a future improvement.
"""

from typing import List, Tuple

import numpy as np

from app.config.logging_config import get_logger
from app.models.ocr.base import OCRBackend

logger = get_logger(__name__)

# PLACEHOLDER — Tesseract's basic API gives no real confidence signal.
_PLACEHOLDER_CONFIDENCE = 0.5


class TesseractOCRBackend(OCRBackend):
    name = "tesseract"

    def __init__(self):
        self._available = False

    def load(self) -> None:
        try:
            import pytesseract  # noqa: F401

            self._available = True
            logger.info("Tesseract backend available.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Tesseract backend unavailable: %s", exc)
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def read_text(self, image: np.ndarray) -> List[Tuple[str, float, float]]:
        if not self.is_available():
            return []

        import pytesseract

        text = pytesseract.image_to_string(image, config="--psm 7").strip()
        if not text:
            return []
        # Single-line mode already returns one combined fragment, so there's
        # no ordering ambiguity — x_position is a placeholder (0.0).
        return [(text, _PLACEHOLDER_CONFIDENCE, 0.0)]
