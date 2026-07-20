"""
Tests for DenominationTextService (app/services/denomination_text_service.py).
Uses a fake OCR backend so these tests don't depend on EasyOCR/Tesseract
being installed or having internet access to download models.
"""

import numpy as np

from app.services.denomination_text_service import DenominationTextService


class _FakeOCRBackend:
    """Minimal stand-in for OCRBackend with a scripted response."""

    def __init__(self, fragments, available=True):
        self._fragments = fragments
        self._available = available

    def is_available(self):
        return self._available

    def read_text(self, image):
        return self._fragments


def _dummy_roi(width=100, height=50):
    return np.zeros((height, width, 3), dtype="uint8")


def test_numeral_matches_predicted_denomination():
    backend = _FakeOCRBackend([("500", 0.9, 0.0)])
    service = DenominationTextService(backend)

    result = service.read_denomination_numeral(_dummy_roi(), "500")

    assert result["extracted_text"] == "500"
    assert result["matches_predicted_denomination"] is True
    assert result["suspicious_reasons"] == []


def test_numeral_mismatch_is_flagged_suspicious():
    backend = _FakeOCRBackend([("500", 0.9, 0.0)])
    service = DenominationTextService(backend)

    result = service.read_denomination_numeral(_dummy_roi(), "100")

    assert result["extracted_text"] == "500"
    assert result["matches_predicted_denomination"] is False
    assert len(result["suspicious_reasons"]) == 1


def test_numeral_no_ocr_result_is_inconclusive_not_suspicious():
    backend = _FakeOCRBackend([])
    service = DenominationTextService(backend)

    result = service.read_denomination_numeral(_dummy_roi(), "500")

    assert result["extracted_text"] is None
    assert result["matches_predicted_denomination"] is None
    assert result["suspicious_reasons"] == []


def test_numeral_unavailable_backend_is_inconclusive():
    backend = _FakeOCRBackend([("500", 0.9, 0.0)], available=False)
    service = DenominationTextService(backend)

    result = service.read_denomination_numeral(_dummy_roi(), "500")

    assert result["matches_predicted_denomination"] is None


def test_numeral_joins_split_fragments_in_reading_order():
    """Regression test: a numeral OCR'd as separate fragments (e.g. the
    '₹' symbol detected apart from the digits) must be joined in
    left-to-right order, not collapsed to only the best-confidence piece."""
    backend = _FakeOCRBackend([("₹", 0.5, 40.0), ("500", 0.95, 0.0)])
    service = DenominationTextService(backend)

    result = service.read_denomination_numeral(_dummy_roi(), "500")

    assert result["extracted_text"] == "500"
    assert result["matches_predicted_denomination"] is True


def test_promise_clause_detects_expected_keywords():
    backend = _FakeOCRBackend(
        [("I PROMISE TO PAY THE BEARER", 0.8, 0.0), ("GOVERNOR", 0.7, 0.0)]
    )
    service = DenominationTextService(backend)

    result = service.verify_promise_clause(_dummy_roi())

    assert result["text_present"] is True
    assert set(result["matched_keywords"]) == {"PROMISE", "PAY", "BEARER", "GOVERNOR"}
    assert result["suspicious_reasons"] == []


def test_promise_clause_missing_keywords_is_flagged():
    backend = _FakeOCRBackend([("SOME UNRELATED TEXT", 0.5, 0.0)])
    service = DenominationTextService(backend)

    result = service.verify_promise_clause(_dummy_roi())

    assert result["text_present"] is False
    assert len(result["suspicious_reasons"]) == 1


def test_promise_clause_unavailable_backend_returns_safe_defaults():
    backend = _FakeOCRBackend([], available=False)
    service = DenominationTextService(backend)

    result = service.verify_promise_clause(_dummy_roi())

    assert result["text_present"] is False
    assert result["suspicious_reasons"] == []  # inconclusive, not penalized
