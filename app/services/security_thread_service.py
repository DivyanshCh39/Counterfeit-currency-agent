"""
Security thread verification module.

Genuine banknote security threads are typically a narrow embedded strip
running through the note (often vertically in the aligned/warped view)
with distinct optical properties: higher local contrast than surrounding
paper, a strong directional (vertical) edge signature, and continuity
along nearly the full height of the note. This prototype approximates
"is a thread-like feature present and continuous" using three independent,
hand-specified heuristics — NOT a trained thread-detection/verification
model (no labeled genuine vs. counterfeit thread dataset is available for
this prototype).

Sub-scores (each normalized to [0, 1]):
    1. region_contrast     — grayscale intensity std-dev within the ROI
                               (a thread reads as a locally higher-contrast
                               band vs. the surrounding paper texture)
    2. vertical_feature     — Sobel vertical-edge column-energy peak-to-mean
                               ratio (a thread produces one dominant vertical
                               edge column; plain paper does not)
    3. band_continuity      — fraction of rows along the ROI's height that
                               contain a detected edge (a genuine thread runs
                               close to continuously top-to-bottom; a forged
                               or absent thread tends to be patchy/discontinuous)

These are combined into a single weighted `thread_score`. Each sub-score
below its threshold adds a human-readable entry to `suspicious_flags`.

>>> TODO (replace with trained model): swap this entire module for a
>>> trained thread-verification model once real data exists — e.g. a
>>> segmentation model that localizes the thread precisely (rather than
>>> assuming a fixed ROI band) and/or a similarity model (e.g. Siamese
>>> network) comparing the extracted thread strip against genuine
>>> reference thread crops. The three heuristics below are reasonable
>>> geometric/statistical proxies but cannot verify thread *content*
>>> (embedded text/holography), only its coarse presence and continuity.
"""

from typing import List

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class SecurityThreadService:
    def __init__(self):
        self.contrast_threshold = settings.SECURITY_THREAD_CONTRAST_THRESHOLD
        self.vertical_peak_threshold = settings.SECURITY_THREAD_VERTICAL_PEAK_THRESHOLD
        self.band_continuity_threshold = settings.SECURITY_THREAD_BAND_CONTINUITY_THRESHOLD

        self.w_contrast = settings.SECURITY_THREAD_WEIGHT_CONTRAST
        self.w_vertical = settings.SECURITY_THREAD_WEIGHT_VERTICAL_FEATURE
        self.w_band_continuity = settings.SECURITY_THREAD_WEIGHT_BAND_CONTINUITY

        self.present_threshold = settings.SECURITY_THREAD_PRESENT_THRESHOLD

    def check(self, roi_image: np.ndarray) -> dict:
        """
        Args:
            roi_image: cropped security-thread region (BGR), expected to
                be a tall, narrow vertical strip (see roi_config.py).

        Returns dict with:
            thread_score, present, continuity_score (alias of thread_score,
            kept for API/schema backward compatibility), sub_scores,
            suspicious_flags (list[str])
        """
        if roi_image is None or roi_image.size == 0:
            return self._empty_result(["Security thread ROI is empty or unreadable."])

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        raw_contrast = self._compute_region_contrast(gray)
        raw_vertical = self._compute_vertical_feature_strength(gray)
        raw_band_continuity = self._compute_band_continuity(gray)

        norm_contrast = self._normalize(raw_contrast, self.contrast_threshold)
        norm_vertical = self._normalize(raw_vertical, self.vertical_peak_threshold)
        norm_band_continuity = self._normalize(raw_band_continuity, self.band_continuity_threshold)

        thread_score = (
            norm_contrast * self.w_contrast
            + norm_vertical * self.w_vertical
            + norm_band_continuity * self.w_band_continuity
        )
        thread_score = round(min(1.0, max(0.0, thread_score)), 3)
        present = thread_score >= self.present_threshold

        suspicious_flags: List[str] = []
        if norm_contrast < 0.5:
            suspicious_flags.append(
                f"Low local contrast in security thread region (std-dev={raw_contrast:.1f}, "
                f"expected>={self.contrast_threshold:.1f}) — expected band may be missing "
                f"or washed out."
            )
        if norm_vertical < 0.5:
            suspicious_flags.append(
                f"Weak vertical feature signature (peak-to-mean ratio={raw_vertical:.2f}, "
                f"expected>={self.vertical_peak_threshold:.2f}) — no clear thread-like "
                f"vertical line detected."
            )
        if norm_band_continuity < 0.5:
            suspicious_flags.append(
                f"Discontinuous band along expected thread path (continuity={raw_band_continuity:.2f}, "
                f"expected>={self.band_continuity_threshold:.2f}) — thread may be broken, "
                f"partially printed, or absent."
            )
        if not present and not suspicious_flags:
            suspicious_flags.append(
                "Composite security thread score below presence threshold."
            )

        return {
            "thread_score": thread_score,
            "continuity_score": thread_score,  # backward-compatible alias
            "present": present,
            "sub_scores": {
                "region_contrast": {"raw": round(raw_contrast, 3), "normalized": round(norm_contrast, 3)},
                "vertical_feature": {"raw": round(raw_vertical, 3), "normalized": round(norm_vertical, 3)},
                "band_continuity": {"raw": round(raw_band_continuity, 3), "normalized": round(norm_band_continuity, 3)},
            },
            "suspicious_flags": suspicious_flags,
        }

    # ------------------------------------------------------------------
    # Individual heuristic scoring functions
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_region_contrast(gray: np.ndarray) -> float:
        """Grayscale intensity std-dev — higher implies a distinct band vs. flat paper."""
        return float(gray.std())

    @staticmethod
    def _compute_vertical_feature_strength(gray: np.ndarray) -> float:
        """
        Sobel vertical-edge detector (highlights vertical lines), then
        compares the strongest column's average gradient energy to the
        mean across all columns. A genuine thread produces one dominant
        column of strong vertical edges; uniform paper does not.
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        column_energy = np.mean(np.abs(sobel_x), axis=0)  # one value per column

        mean_energy = float(column_energy.mean()) + 1e-6
        max_energy = float(column_energy.max())
        return max_energy / mean_energy

    @staticmethod
    def _compute_band_continuity(gray: np.ndarray) -> float:
        """
        Fraction of rows (top-to-bottom) that contain at least one Canny
        edge pixel. A continuous thread produces edges in nearly every row
        along the ROI's height; a broken/absent thread leaves large
        edge-free gaps.
        """
        edges = cv2.Canny(gray, 50, 150)
        row_has_edge = np.count_nonzero(edges, axis=1) > 0
        return float(np.count_nonzero(row_has_edge)) / max(1, edges.shape[0])

    @staticmethod
    def _normalize(raw_value: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0
        return min(1.0, max(0.0, raw_value / threshold))

    @staticmethod
    def _empty_result(reasons: List[str]) -> dict:
        return {
            "thread_score": 0.0,
            "continuity_score": 0.0,
            "present": False,
            "sub_scores": {},
            "suspicious_flags": reasons,
        }
