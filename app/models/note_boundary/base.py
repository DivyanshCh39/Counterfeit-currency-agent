"""
Abstract interface for note-boundary detection backends.

Mirrors app/models/denomination_classifier/base.py's design exactly, for
the same reason: NoteBoundaryService (app/services/note_boundary_service.py)
talks only to this interface, never to a concrete backend directly, so the
existing OpenCV contour heuristic can keep running as the always-available
fallback while an optional trained segmentation model is added alongside
it — without PreprocessingService, the pipeline, the API schema, or the
router ever needing to change.

Every backend must return either:
    corner_points: np.ndarray of shape (4, 2), float32
        -> the SAME format app/utils/geometry_utils.find_largest_quadrilateral()
           already returns, so four_point_warp() (unchanged) can consume
           the result from ANY backend identically.
or:
    None -> backend could not localize a note boundary at all
            (model not loaded, no contour found, mask empty, etc.)

Scope note: this interface is for note LOCALIZATION/ALIGNMENT support
only — it has nothing to do with, and must never be extended toward,
genuine/counterfeit classification. See training/train_note_boundary_model.py
docstring for the same constraint on the training side.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class NoteBoundaryBackend(ABC):
    """Common contract for all note-boundary localization strategies."""

    #: Human-readable identifier, surfaced in PreprocessingResult debug info
    #: (e.g. "ml_segmenter_onnx", "heuristic_contour").
    name: str = "unnamed_backend"

    def load(self) -> None:
        """
        Optional hook for backends that need to load weights/resources.
        Default no-op — override in backends that need it (e.g. ONNX).
        Must never raise; on failure, is_available() should simply return False.
        """
        return None

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is currently usable (weights loaded, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            image: resized (pre-alignment) BGR image, i.e. the SAME input
                PreprocessingService.detect_note_contour() already receives
                today (settings.PREPROCESS_RESIZE_WIDTH working width).

        Returns:
            (4, 2) float32 array of corner points (any order — the existing
            four_point_warp()/order_points() already handles reordering),
            or None if no note boundary could be localized.
        """
        raise NotImplementedError
