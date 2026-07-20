"""
Heuristic backend: thin wrapper around the existing OpenCV contour
detector (app/utils/geometry_utils.find_largest_quadrilateral).

This is NOT a new detection algorithm — it is the exact same v1 heuristic
PreprocessingService already used directly before this revision, now
exposed through the NoteBoundaryBackend interface so it can sit in the
same ordered backend list as the optional ML segmenter. It has no
load-time dependency (no weights, nothing that can fail to initialize),
so is_available() always returns True — this is what guarantees note
localization keeps working even when no trained segmentation model has
ever been placed on disk.
"""

from typing import Optional

import numpy as np

from app.config.settings import settings
from app.models.note_boundary.base import NoteBoundaryBackend
from app.utils.geometry_utils import find_largest_quadrilateral


class HeuristicContourBackend(NoteBoundaryBackend):
    name = "heuristic_contour"

    def is_available(self) -> bool:
        return True  # pure OpenCV, no weights/resources to load

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        return find_largest_quadrilateral(
            image, min_area_ratio=settings.MIN_NOTE_CONTOUR_AREA_RATIO
        )
