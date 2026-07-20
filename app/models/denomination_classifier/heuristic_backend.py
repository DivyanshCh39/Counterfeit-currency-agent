"""
HEURISTIC baseline backend: color-histogram template matching against
reference note images in data/reference_notes/.

This is the backend that is actually active by default in the prototype,
since no trained classifier weights exist yet. It is intentionally crude:
- No shape/texture/print-detail features, only HSV color histograms.
- No learned decision boundary — just "nearest histogram wins".
- Accuracy depends entirely on how visually distinct your reference note
  images are in overall color, and will degrade on note designs that
  share similar color palettes across denominations.

It exists purely to keep the pipeline runnable end-to-end before a real
CNN/TFLite/ONNX model is trained. See README for the upgrade path.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.models.denomination_classifier.base import DenominationClassifierBackend

logger = get_logger(__name__)


class HeuristicTemplateMatchBackend(DenominationClassifierBackend):
    name = "heuristic_template_match"

    def __init__(self, reference_notes_dir: Path):
        self.reference_notes_dir = reference_notes_dir
        self._reference_histograms: dict = {}  # label -> histogram, cached on load()

    def load(self) -> None:
        """Precompute and cache reference histograms so predict() is cheap."""
        self._reference_histograms.clear()

        if not self.reference_notes_dir.exists():
            logger.warning(
                "Reference notes directory missing: %s — heuristic denomination "
                "backend will be unavailable.",
                self.reference_notes_dir,
            )
            return

        for ref_path in self.reference_notes_dir.glob("*.jpg"):
            ref_image = cv2.imread(str(ref_path))
            if ref_image is None:
                logger.warning("Could not read reference image %s", ref_path)
                continue
            label = ref_path.stem  # e.g. "500.jpg" -> "500"
            self._reference_histograms[label] = self._compute_histogram(ref_image)

        logger.info(
            "Heuristic denomination backend loaded %d reference templates.",
            len(self._reference_histograms),
        )

    def is_available(self) -> bool:
        return len(self._reference_histograms) > 0

    def predict(self, aligned_note_image: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.is_available():
            return None

        query_hist = self._compute_histogram(aligned_note_image)

        best_label: Optional[str] = None
        best_score = -1.0

        for label, ref_hist in self._reference_histograms.items():
            score = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        # HISTCMP_CORREL is in [-1, 1]; rescale to a [0, 1] pseudo-confidence.
        # This is NOT a calibrated probability — it's a similarity proxy.
        confidence = max(0.0, min(1.0, (best_score + 1) / 2))
        return best_label, confidence

    @staticmethod
    def _compute_histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist
