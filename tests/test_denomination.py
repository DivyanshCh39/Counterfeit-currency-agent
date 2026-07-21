"""
Tests for DenominationService and its pluggable backends.
"""

import numpy as np

from app.core.constants import UNKNOWN_DENOMINATION
from app.services.denomination_service import DenominationService


def test_classify_falls_back_to_unknown_when_no_backend_confident():
    """
    With no ONNX weights and no reference images, both backends should be
    unavailable/inconclusive, so classify() must return the unknown
    fallback rather than raising or guessing.
    """
    service = DenominationService()
    dummy_image = np.random.randint(0, 255, (300, 500, 3), dtype="uint8")

    result = service.classify(dummy_image)

    assert result.predicted_value == UNKNOWN_DENOMINATION
    assert result.confidence == 0.0
    assert result.method == "fallback_unknown"


def test_onnx_backend_reports_unavailable_without_weights(tmp_path):
    from app.models.denomination_classifier.onnx_backend import OnnxDenominationBackend

    missing_path = tmp_path / "does_not_exist.onnx"
    backend = OnnxDenominationBackend(missing_path)
    backend.load()
    assert backend.is_available() is False