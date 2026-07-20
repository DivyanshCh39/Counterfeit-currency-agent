"""
EasyOCR backend — pretrained, no custom training performed for this
prototype. Default OCR engine (see settings.OCR_ENGINE).
"""

from typing import List, Tuple

import numpy as np

from app.config.logging_config import get_logger
from app.models.ocr.base import OCRBackend

logger = get_logger(__name__)


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(self, languages: Tuple[str, ...] = ("en",)):
        self.languages = list(languages)
        self._reader = None

    def load(self) -> None:
        try:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=False)
            logger.info("EasyOCR backend loaded (languages=%s).", self.languages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load EasyOCR backend: %s", exc)
            self._reader = None

    def is_available(self) -> bool:
        return self._reader is not None

    def read_text(self, image: np.ndarray) -> List[Tuple[str, float, float]]:
        if not self.is_available():
            return []
        results = self._reader.readtext(image)
        # each result: (bbox_polygon, text, confidence)
        # bbox_polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (TL, TR, BR, BL).
        # Use the leftmost x so fragments can be sorted into reading order
        # by the caller — EasyOCR's own result order is not guaranteed to
        # match left-to-right reading order for split detections.
        return [
            (text, float(confidence), float(min(pt[0] for pt in bbox)))
            for (bbox, text, confidence) in results
        ]
