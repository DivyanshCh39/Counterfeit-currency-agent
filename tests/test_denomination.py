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


def test_onnx_backend_reports_unavailable_without_weights():
    from app.config.settings import settings
    from app.models.denomination_classifier.onnx_backend import OnnxDenominationBackend

    backend = OnnxDenominationBackend(settings.DENOMINATION_MODEL_PATH)
    backend.load()
    assert backend.is_available() is False
    assert backend.predict(np.zeros((10, 10, 3), dtype="uint8")) is None


def test_onnx_backend_preprocess_produces_nchw_layout():
    """
    Regression test: preprocessing must output NCHW (batch, channels,
    height, width) to match PyTorch/ONNX export convention used by
    training/train_denomination_classifier.py. This previously shipped as
    NHWC, which silently caused every real inference call to fail with an
    ONNXRuntime shape-mismatch error once a trained model was actually loaded.
    """
    from app.config.settings import settings
    from app.models.denomination_classifier.onnx_backend import OnnxDenominationBackend

    backend = OnnxDenominationBackend(settings.DENOMINATION_MODEL_PATH)
    dummy_image = np.random.randint(0, 255, (300, 500, 3), dtype="uint8")

    tensor = backend._preprocess(dummy_image)

    assert tensor.shape == (1, 3, *backend.INPUT_SIZE)
