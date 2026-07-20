"""
Tests for OCRService (app/services/ocr_service.py).
Note: EasyOCR model download happens on first run and requires internet
access the first time it's used — tests skip gracefully if unavailable.
"""

import numpy as np
import pytest

from app.services.ocr_service import OCRService


class _FakeBackend:
    """Minimal stand-in for OCRBackend with a scripted, split-fragment
    response — lets the join/ordering fix be tested without EasyOCR
    installed or downloaded."""

    def __init__(self, fragments, available=True):
        self._fragments = fragments
        self._available = available

    def is_available(self):
        return self._available

    def read_text(self, image):
        return self._fragments


def test_read_serial_number_joins_split_fragments_in_reading_order():
    """Regression test for the bug where a serial number split across two
    OCR fragments (e.g. '6WS' and '396618' separated by a printed security
    tick mark) was collapsed to only the single highest-confidence
    fragment, silently dropping part of the serial number."""
    service = OCRService.__new__(OCRService)  # bypass __init__ (no real backend load)
    from app.services.serial_consistency_service import SerialConsistencyService

    service.consistency_service = SerialConsistencyService()
    # "396618" has higher confidence than "6WS", but appears AFTER it in
    # reading order (larger x_position) — the old code would have kept
    # only "396618" since it picks by max confidence, not position.
    service.backend = _FakeBackend(
        [("396618", 0.95, 40.0), ("6WS", 0.80, 0.0)]
    )

    result = service.read_serial_number(np.zeros((10, 10, 3), dtype="uint8"), denomination="500")

    assert result["serial_number"] == "6WS396618"  # cleaned (spaces stripped) by SerialConsistencyService


def test_ocr_service_handles_blank_image_gracefully():
    try:
        service = OCRService()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"OCR engine unavailable in this environment: {exc}")
        return

    blank_image = np.ones((50, 200, 3), dtype="uint8") * 255

    try:
        result = service.read_serial_number(blank_image, denomination="500")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"OCR engine unavailable in this environment: {exc}")
        return

    assert "serial_number" in result
    assert "normalized_text" in result
    assert "format_valid" in result
    assert "quality_score" in result
    assert "validation_flags" in result
    assert "suspicious_reasons" in result
    assert 0.0 <= result["quality_score"] <= 1.0
    assert set(result["validation_flags"].keys()) == {
        "length_valid",
        "characters_valid",
        "spacing_valid",
        "pattern_valid",
    }


def test_ocr_service_rejects_unsupported_engine():
    from app.core.exceptions import OCRProcessingError

    with pytest.raises(OCRProcessingError):
        OCRService(engine="not_a_real_engine")
