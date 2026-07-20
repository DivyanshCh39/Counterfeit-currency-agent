"""
Note-boundary localization service.

Design: an ordered list of interchangeable NoteBoundaryBackend
implementations (see app/models/note_boundary/base.py) — the exact same
pattern app/services/denomination_service.py already uses for swapping in
a trained model ahead of a heuristic. The service tries each backend in
order and returns the first non-None result.

Backend order today:
    1. NoteBoundaryOnnxBackend -> ML segmentation model, INACTIVE until
       real weights exist at settings.NOTE_BOUNDARY_MODEL_PATH
    2. HeuristicContourBackend -> existing OpenCV edge/contour detector
       (ACTIVE — this is the exact same v1 heuristic PreprocessingService
       used directly before this revision; it always stays available, so
       note localization never stops working even with zero trained
       weights on disk)

To go live with a trained segmentation model, run
training/prepare_segmentation_data.py + training/train_note_boundary_model.py
and drop the resulting weights at settings.NOTE_BOUNDARY_MODEL_PATH — no
other code changes required, the ONNX backend activates automatically and
takes priority over the heuristic, exactly like DenominationService does
for its own ONNX backend.

SCOPE: this service only ever returns note-boundary corner points for
perspective alignment. It has no connection to, and must never be
extended toward, counterfeit/genuine decisioning.
"""

from typing import List, Optional

import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.models.note_boundary.base import NoteBoundaryBackend
from app.models.note_boundary.heuristic_backend import HeuristicContourBackend
from app.models.note_boundary.onnx_backend import NoteBoundaryOnnxBackend

logger = get_logger(__name__)


class NoteBoundaryService:
    def __init__(self):
        # Ordered by preference: first available backend that returns a
        # result wins. HeuristicContourBackend.is_available() is always
        # True, so this list always yields SOME attempt even with no ML
        # weights present — there is no "no backend available" case here,
        # unlike DenominationService (which has a genuine unknown/fallback
        # state because guessing a denomination heuristically can be
        # actively wrong; guessing note boundary geometry heuristically is
        # just today's existing, already-shipped behavior).
        self.backends: List[NoteBoundaryBackend] = [
            NoteBoundaryOnnxBackend(settings.NOTE_BOUNDARY_MODEL_PATH),
            HeuristicContourBackend(),
        ]

        for backend in self.backends:
            backend.load()

        active = [b.name for b in self.backends if b.is_available()]
        logger.info("Note-boundary backends active: %s", active or ["<none>"])

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        for backend in self.backends:
            if not backend.is_available():
                continue

            corner_points = backend.detect(image)
            if corner_points is None:
                logger.debug(
                    "%s found no note boundary — trying next backend.", backend.name
                )
                continue

            logger.debug("Note boundary localized by backend '%s'.", backend.name)
            return corner_points

        return None
