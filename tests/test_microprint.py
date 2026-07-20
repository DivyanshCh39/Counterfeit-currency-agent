"""
Tests for MicroprintService (app/services/microprint_service.py).
"""

import cv2
import numpy as np

from app.services.microprint_service import MicroprintService


def _sharp_textured_patch(size=120) -> np.ndarray:
    """Synthetic high-frequency, high-contrast patch — should score well."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, (size, size), dtype="uint8")
    # checkerboard-like fine pattern on top of noise for extra high-frequency content
    checker = np.indices((size, size)).sum(axis=0) % 4 < 2
    pattern = np.where(checker, 255, 0).astype("uint8")
    combined = cv2.addWeighted(noise, 0.5, pattern, 0.5, 0)
    return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)


def _flat_blurred_patch(size=120) -> np.ndarray:
    """Synthetic flat, low-detail patch — should score poorly."""
    flat = np.full((size, size), 180, dtype="uint8")
    blurred = cv2.GaussianBlur(flat, (25, 25), 10)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def test_score_returns_all_expected_keys():
    service = MicroprintService()
    result = service.score(_sharp_textured_patch())

    assert set(result.keys()) == {"clarity_score", "passed", "sub_scores", "suspicious_flags"}
    assert set(result["sub_scores"].keys()) == {
        "sharpness", "edge_density", "frequency_detail", "patch_texture",
    }
    assert 0.0 <= result["clarity_score"] <= 1.0


def test_sharp_textured_patch_scores_higher_than_flat_blurred_patch():
    service = MicroprintService()
    sharp_result = service.score(_sharp_textured_patch())
    flat_result = service.score(_flat_blurred_patch())

    assert sharp_result["clarity_score"] > flat_result["clarity_score"]


def test_flat_blurred_patch_produces_suspicious_flags():
    service = MicroprintService()
    result = service.score(_flat_blurred_patch())

    assert result["passed"] is False
    assert len(result["suspicious_flags"]) > 0


def test_empty_roi_is_handled_gracefully():
    service = MicroprintService()
    result = service.score(np.zeros((0, 0, 3), dtype="uint8"))

    assert result["clarity_score"] == 0.0
    assert result["passed"] is False
    assert len(result["suspicious_flags"]) > 0
