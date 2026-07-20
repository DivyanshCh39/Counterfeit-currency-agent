"""
Tests for SerialConsistencyService and serial_config.py.
These are pure-logic tests — no OCR engine required.
"""

from app.config.serial_config import get_serial_pattern, register_serial_pattern
from app.services.serial_consistency_service import SerialConsistencyService


def test_get_serial_pattern_falls_back_to_default():
    pattern, matched = get_serial_pattern("unknown")
    assert matched is False
    assert pattern  # non-empty regex string


def test_get_serial_pattern_returns_denomination_specific_pattern():
    pattern, matched = get_serial_pattern("500")
    assert matched is True


def test_register_serial_pattern_is_used_afterwards():
    register_serial_pattern("999", r"^ZZ[0-9]{4}$")
    pattern, matched = get_serial_pattern("999")
    assert matched is True
    assert pattern == r"^ZZ[0-9]{4}$"


def test_evaluate_valid_serial_passes_all_flags():
    service = SerialConsistencyService()
    # "500" pattern: 2 letters + 6 digits = 8 chars, matches length range (8, 8)
    result = service.evaluate("AB123456", ocr_confidence=0.9, denomination="500")

    assert result["serial_number"] == "AB123456"
    assert result["format_valid"] is True
    assert result["validation_flags"] == {
        "length_valid": True,
        "characters_valid": True,
        "spacing_valid": True,
        "pattern_valid": True,
    }
    assert result["suspicious_reasons"] == []
    assert result["quality_score"] > 0.8


def test_evaluate_confusable_characters_recovers_but_is_flagged_suspicious():
    service = SerialConsistencyService()
    # "I" and "O" will be misread in place of "1" and "0" -> normalization should fix it
    # (avoid letters A/E/G/... that are also in CHARACTER_CONFUSION_MAP, e.g. B/S/Z)
    result = service.evaluate("AXI2345O", ocr_confidence=0.7, denomination="500")

    assert result["normalized_text"] == "AX123450"
    assert result["format_valid"] is True
    # Even though validation_flags["pattern_valid"] ends up True (via normalization),
    # this case is still called out explicitly so a human reviewer can see the
    # match was not direct.
    assert any("OCR character misreads" in r for r in result["suspicious_reasons"])

    direct_result = service.evaluate("AX123450", ocr_confidence=0.7, denomination="500")
    assert result["quality_score"] < direct_result["quality_score"]


def test_evaluate_empty_text_returns_no_text_reason_and_zero_score():
    service = SerialConsistencyService()
    result = service.evaluate(None, ocr_confidence=0.0, denomination="500")

    assert result["format_valid"] is False
    assert result["serial_number"] is None
    assert result["quality_score"] == 0.0
    assert "No text detected in serial number region." in result["suspicious_reasons"]


def test_evaluate_wrong_length_flags_length_invalid():
    service = SerialConsistencyService()
    result = service.evaluate("AB123", ocr_confidence=0.8, denomination="500")  # too short

    assert result["validation_flags"]["length_valid"] is False
    assert any("length" in r.lower() for r in result["suspicious_reasons"])


def test_evaluate_irregular_spacing_flags_spacing_invalid():
    service = SerialConsistencyService()
    result = service.evaluate("AB  123456", ocr_confidence=0.8, denomination="500")

    assert result["validation_flags"]["spacing_valid"] is False
    assert any("spacing" in r.lower() for r in result["suspicious_reasons"])


def test_evaluate_completely_invalid_text_scores_low():
    service = SerialConsistencyService()
    result = service.evaluate("!!!", ocr_confidence=0.8, denomination="500")

    assert result["format_valid"] is False
    assert result["quality_score"] < 0.5
