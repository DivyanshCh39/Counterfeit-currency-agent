"""
Tests for PreprocessingService (app/services/preprocessing_service.py).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.services.preprocessing_service import PreprocessingService

SAMPLE_PATH = Path("data/sample_uploads/sample_note.jpg")


def _encode_dummy_jpeg(width=600, height=300, color=(200, 200, 200)) -> bytes:
    image = np.full((height, width, 3), color, dtype="uint8")
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_validate_format_rejects_unsupported_extension():
    service = PreprocessingService(debug_enabled=False)
    with pytest.raises(InvalidImageError):
        service.validate_format("note.bmp")


def test_validate_format_accepts_supported_extension():
    service = PreprocessingService(debug_enabled=False)
    service.validate_format("note.jpg")  # should not raise


def test_load_image_raises_on_garbage_bytes():
    service = PreprocessingService(debug_enabled=False)
    with pytest.raises(InvalidImageError):
        service.load_image(b"not a real image")


def test_assess_quality_flags_low_resolution_dummy_image():
    service = PreprocessingService(debug_enabled=False)
    tiny_image = np.full((50, 50, 3), 128, dtype="uint8")
    report = service.assess_quality(tiny_image)
    assert report.is_acceptable is False
    assert any("Resolution" in reason for reason in report.reasons)


def test_run_handles_low_quality_dummy_image_without_crashing():
    service = PreprocessingService(debug_enabled=False)
    image_bytes = _encode_dummy_jpeg(width=100, height=50)  # below MIN dims
    result = service.run(image_bytes, "note.jpg")

    assert result.quality.is_acceptable is False
    assert result.detected is False
    assert result.aligned_image is None


def test_run_on_real_sample_image_if_available():
    if not SAMPLE_PATH.exists():
        pytest.skip(f"No sample image found at {SAMPLE_PATH}")

    service = PreprocessingService(debug_enabled=True)
    image_bytes = SAMPLE_PATH.read_bytes()
    result = service.run(image_bytes, SAMPLE_PATH.name)

    assert result.original_image is not None
    assert result.resized_image is not None
    # quality/detection outcome depends on the actual sample image content
