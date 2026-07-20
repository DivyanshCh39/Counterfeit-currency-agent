"""
Tests for SecurityThreadService (app/services/security_thread_service.py).
"""

import cv2
import numpy as np

from app.services.security_thread_service import SecurityThreadService


def _strip_with_vertical_line(width=60, height=300) -> np.ndarray:
    """Synthetic strip with a strong continuous vertical line — should score well."""
    strip = np.full((height, width), 200, dtype="uint8")
    center = width // 2
    strip[:, center - 2 : center + 2] = 40  # dark continuous vertical band
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)


def _blank_strip(width=60, height=300) -> np.ndarray:
    """Synthetic blank/uniform strip — no thread — should score poorly."""
    strip = np.full((height, width), 200, dtype="uint8")
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)


def test_check_returns_all_expected_keys():
    service = SecurityThreadService()
    result = service.check(_strip_with_vertical_line())

    assert set(result.keys()) == {
        "thread_score", "continuity_score", "present", "sub_scores", "suspicious_flags",
    }
    assert set(result["sub_scores"].keys()) == {
        "region_contrast", "vertical_feature", "band_continuity",
    }
    assert 0.0 <= result["thread_score"] <= 1.0
    assert result["continuity_score"] == result["thread_score"]


def test_strip_with_vertical_line_scores_higher_than_blank_strip():
    service = SecurityThreadService()
    line_result = service.check(_strip_with_vertical_line())
    blank_result = service.check(_blank_strip())

    assert line_result["thread_score"] > blank_result["thread_score"]
    assert line_result["present"] is True


def test_blank_strip_produces_suspicious_flags_and_not_present():
    service = SecurityThreadService()
    result = service.check(_blank_strip())

    assert result["present"] is False
    assert len(result["suspicious_flags"]) > 0


def test_empty_roi_is_handled_gracefully():
    service = SecurityThreadService()
    result = service.check(np.zeros((0, 0, 3), dtype="uint8"))

    assert result["thread_score"] == 0.0
    assert result["present"] is False
    assert len(result["suspicious_flags"]) > 0
