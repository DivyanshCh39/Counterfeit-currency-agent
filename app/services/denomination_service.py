"""
Denomination classification service.

Design: an ordered list of interchangeable DenominationClassifierBackend
implementations (see app/models/denomination_classifier/base.py). The
service tries each backend in order and takes the first prediction whose
confidence clears settings.DENOMINATION_CONFIDENCE_THRESHOLD. If nothing
qualifies, it returns the "unknown" fallback rather than guessing.

Backend order today:
    1. OnnxDenominationBackend   -> ML, INACTIVE until real weights exist
    2. HeuristicTemplateMatchBackend -> heuristic color-histogram matching (ACTIVE)

To go live with a trained model, drop weights at
settings.DENOMINATION_MODEL_PATH — no code changes required, the ONNX
backend will automatically activate and take priority over the heuristic.
To add a TFLite backend instead (e.g. for mobile deployment), see
app/models/denomination_classifier/tflite_backend.py and register it here.
"""

from typing import List

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.constants import UNKNOWN_DENOMINATION
from app.models.denomination_classifier.base import DenominationClassifierBackend
from app.models.denomination_classifier.heuristic_backend import (
    HeuristicTemplateMatchBackend,
)
from app.models.denomination_classifier.onnx_backend import OnnxDenominationBackend
from app.schemas.response_schemas import DenominationResult

logger = get_logger(__name__)


class DenominationService:
    def __init__(self):
        # Ordered by preference: first available + confident-enough backend wins.
        self.backends: List[DenominationClassifierBackend] = [
            OnnxDenominationBackend(settings.DENOMINATION_MODEL_PATH),
            HeuristicTemplateMatchBackend(settings.REFERENCE_NOTES_DIR),
        ]

        for backend in self.backends:
            backend.load()

        active = [b.name for b in self.backends if b.is_available()]
        logger.info("Denomination backends active: %s", active or ["<none>"])

    def classify(self, aligned_note_image) -> DenominationResult:
        for backend in self.backends:
            if not backend.is_available():
                continue

            prediction = backend.predict(aligned_note_image)
            if prediction is None:
                continue

            label, confidence = prediction

            if confidence < settings.DENOMINATION_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "%s predicted '%s' at confidence %.2f — below threshold %.2f, "
                    "trying next backend.",
                    backend.name, label, confidence, settings.DENOMINATION_CONFIDENCE_THRESHOLD,
                )
                continue

            return DenominationResult(
                predicted_value=label,
                confidence=round(confidence, 3),
                method=backend.name,
            )

        # No backend produced a sufficiently confident prediction.
        return DenominationResult(
            predicted_value=UNKNOWN_DENOMINATION,
            confidence=0.0,
            method="fallback_unknown",
        )
