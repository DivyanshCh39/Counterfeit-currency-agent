"""
Serial number OCR service.

Delegates raw text extraction to a pluggable OCRBackend (EasyOCR by
default, Tesseract as an alternative — see app/models/ocr/), joins all
detected text fragments in left-to-right order, then hands the combined
text to SerialConsistencyService for format/consistency validation.

To use a custom-trained serial-recognition model later, add a new
OCRBackend implementation under app/models/ocr/ and register it in
_BACKEND_REGISTRY below — no other file needs to change.
"""

from typing import Optional

import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.exceptions import OCRProcessingError
from app.models.ocr.base import OCRBackend
from app.models.ocr.easyocr_backend import EasyOCRBackend
from app.models.ocr.tesseract_backend import TesseractOCRBackend
from app.services.serial_consistency_service import SerialConsistencyService

logger = get_logger(__name__)

_BACKEND_REGISTRY = {
    "easyocr": EasyOCRBackend,
    "tesseract": TesseractOCRBackend,
}


class OCRService:
    def __init__(self, engine: Optional[str] = None):
        engine = engine or settings.OCR_ENGINE
        backend_cls = _BACKEND_REGISTRY.get(engine)
        if backend_cls is None:
            raise OCRProcessingError(
                f"Unsupported OCR engine '{engine}'. "
                f"Available: {list(_BACKEND_REGISTRY.keys())}"
            )

        self.backend: OCRBackend = backend_cls()
        self.backend.load()
        self.consistency_service = SerialConsistencyService()

        if not self.backend.is_available():
            logger.warning(
                "OCR backend '%s' failed to initialize — serial number "
                "reads will return empty results until this is resolved.",
                self.backend.name,
            )

    def read_serial_number(self, roi_image: np.ndarray, denomination: str = "DEFAULT") -> dict:
        """
        Returns dict: {serial_number, normalized_text, confidence,
        quality_score, validation_flags, format_valid, suspicious_reasons}
        (see SerialConsistencyService.evaluate for field details).
        """
        if not self.backend.is_available():
            return self.consistency_service.evaluate(None, 0.0, denomination)

        try:
            fragments = self.backend.read_text(roi_image)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR backend '%s' failed: %s", self.backend.name, exc)
            raise OCRProcessingError(str(exc)) from exc

        if not fragments:
            return self.consistency_service.evaluate(None, 0.0, denomination)

        # Previously this took only the single highest-confidence fragment
        # (`max(fragments, key=lambda f: f[1])`), which silently dropped the
        # rest of the serial number whenever EasyOCR split it into multiple
        # fragments (e.g. a printed security tick mark breaking "6WS" from
        # "396618"). Sort left-to-right by x_position and join everything
        # instead, so the full serial number reaches validation.
        ordered_fragments = sorted(fragments, key=lambda f: f[2])
        combined_text = " ".join(text for text, _confidence, _x in ordered_fragments).strip()
        avg_confidence = sum(conf for _text, conf, _x in ordered_fragments) / len(ordered_fragments)

        return self.consistency_service.evaluate(combined_text, avg_confidence, denomination)
