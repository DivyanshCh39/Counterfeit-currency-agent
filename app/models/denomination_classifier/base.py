"""
Abstract interface for denomination classification backends.

DenominationService (app/services/denomination_service.py) talks only to
this interface, never to a concrete backend directly. This is what makes
it possible to swap the heuristic baseline for a trained ONNX or TFLite
model later WITHOUT touching the service layer, the pipeline, the API
schema, or the router — only a new backend class needs to be written and
registered in DenominationService.__init__().

Every backend must return either:
    (label: str, confidence: float in [0, 1])
or:
    None   -> backend could not produce any prediction at all
              (e.g. model not loaded, image unusable)

A low-but-present confidence is still a valid return value — deciding
whether that confidence is "good enough" is DenominationService's job
(via settings.DENOMINATION_CONFIDENCE_THRESHOLD), not the backend's.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class DenominationClassifierBackend(ABC):
    """Common contract for all denomination classification strategies."""

    #: Human-readable identifier stored in DenominationResult.method
    #: (e.g. "ml_classifier_onnx", "heuristic_template_match").
    name: str = "unnamed_backend"

    def load(self) -> None:
        """
        Optional hook for backends that need to load weights/resources.
        Default no-op — override in backends that need it (e.g. ONNX/TFLite).
        Must never raise; on failure, is_available() should simply return False.
        """
        return None

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is currently usable (weights loaded, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, aligned_note_image: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Args:
            aligned_note_image: perspective-corrected BGR note image
                (output of PreprocessingService / AlignmentService).

        Returns:
            (denomination_label, confidence) or None.
        """
        raise NotImplementedError
