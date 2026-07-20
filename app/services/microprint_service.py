"""
Microprint region analysis module.

Genuine currency microprint is extremely fine, high-frequency detail that
degrades badly under photocopying/reprinting/screen-capture reproduction.
This prototype approximates "is this still crisp micro-detail, or a blurred
smear" using four independent, hand-specified image-quality heuristics —
NOT a learned texture/authenticity classifier (no labeled genuine vs.
counterfeit microprint dataset is available for this prototype).

Sub-scores (each normalized to [0, 1]):
    1. sharpness         — Laplacian variance (classic blur metric)
    2. edge_density       — fraction of Canny edge pixels (fine strokes -> many edges)
    3. frequency_detail    — high-frequency energy ratio in the 2D FFT spectrum
                              (fine print concentrates energy at high spatial frequency;
                              blur/smoothing concentrates it at low frequency)
    4. patch_texture       — mean local intensity std-dev over small patches
                              (flat/smeared reproductions have low local variance
                              even if overall contrast looks fine)

These are combined into a single weighted `clarity_score`. Each sub-score
that falls below its threshold adds a human-readable entry to
`suspicious_flags`.

>>> TODO (replace with trained model): swap this entire module for a
>>> patch-based CNN classifier trained on paired genuine/counterfeit
>>> microprint crops once such a dataset exists. The four heuristics below
>>> are reasonable, well-understood image-quality proxies, but they cannot
>>> distinguish "blurry photo of a genuine note" from "sharp photo of a
>>> counterfeit with fuzzy microprint" as reliably as a model trained on
>>> real examples of both.
"""

from typing import List

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class MicroprintService:
    def __init__(self):
        self.sharpness_threshold = settings.MICROPRINT_SHARPNESS_THRESHOLD
        self.edge_density_threshold = settings.MICROPRINT_EDGE_DENSITY_THRESHOLD
        self.frequency_detail_threshold = settings.MICROPRINT_FREQUENCY_DETAIL_THRESHOLD
        self.patch_texture_threshold = settings.MICROPRINT_PATCH_TEXTURE_THRESHOLD

        self.w_sharpness = settings.MICROPRINT_WEIGHT_SHARPNESS
        self.w_edge_density = settings.MICROPRINT_WEIGHT_EDGE_DENSITY
        self.w_frequency_detail = settings.MICROPRINT_WEIGHT_FREQUENCY_DETAIL
        self.w_patch_texture = settings.MICROPRINT_WEIGHT_PATCH_TEXTURE

        self.pass_threshold = settings.MICROPRINT_PASS_THRESHOLD

    def score(self, roi_image: np.ndarray) -> dict:
        """
        Args:
            roi_image: cropped microprint region (BGR).

        Returns dict with:
            clarity_score, passed, sub_scores (dict of 4 raw+normalized
            values), suspicious_flags (list[str])
        """
        if roi_image is None or roi_image.size == 0:
            return self._empty_result(["Microprint ROI is empty or unreadable."])

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        raw_sharpness = self._compute_sharpness(gray)
        raw_edge_density = self._compute_edge_density(gray)
        raw_frequency_detail = self._compute_frequency_detail(gray)
        raw_patch_texture = self._compute_patch_texture(gray)

        norm_sharpness = self._normalize(raw_sharpness, self.sharpness_threshold)
        norm_edge_density = self._normalize(raw_edge_density, self.edge_density_threshold)
        norm_frequency_detail = self._normalize(
            raw_frequency_detail, self.frequency_detail_threshold
        )
        norm_patch_texture = self._normalize(raw_patch_texture, self.patch_texture_threshold)

        clarity_score = (
            norm_sharpness * self.w_sharpness
            + norm_edge_density * self.w_edge_density
            + norm_frequency_detail * self.w_frequency_detail
            + norm_patch_texture * self.w_patch_texture
        )
        clarity_score = round(min(1.0, max(0.0, clarity_score)), 3)
        passed = clarity_score >= self.pass_threshold

        suspicious_flags: List[str] = []
        if norm_sharpness < 0.5:
            suspicious_flags.append(
                f"Low sharpness in microprint region (Laplacian variance={raw_sharpness:.1f}, "
                f"expected>={self.sharpness_threshold:.1f})."
            )
        if norm_edge_density < 0.5:
            suspicious_flags.append(
                f"Low edge density in microprint region ({raw_edge_density:.3f}, "
                f"expected>={self.edge_density_threshold:.3f}) — fine strokes may be "
                f"smudged or missing."
            )
        if norm_frequency_detail < 0.5:
            suspicious_flags.append(
                f"Low high-frequency detail ({raw_frequency_detail:.3f}, "
                f"expected>={self.frequency_detail_threshold:.3f}) — possible print "
                f"blur/smoothing typical of low-quality reproduction."
            )
        if norm_patch_texture < 0.5:
            suspicious_flags.append(
                f"Low local texture variance ({raw_patch_texture:.1f}, "
                f"expected>={self.patch_texture_threshold:.1f}) — region may be a flat "
                f"smear rather than fine microprint."
            )
        if not passed and not suspicious_flags:
            suspicious_flags.append(
                "Composite microprint clarity score below pass threshold."
            )

        return {
            "clarity_score": clarity_score,
            "passed": passed,
            "sub_scores": {
                "sharpness": {"raw": round(raw_sharpness, 3), "normalized": round(norm_sharpness, 3)},
                "edge_density": {"raw": round(raw_edge_density, 3), "normalized": round(norm_edge_density, 3)},
                "frequency_detail": {"raw": round(raw_frequency_detail, 3), "normalized": round(norm_frequency_detail, 3)},
                "patch_texture": {"raw": round(raw_patch_texture, 3), "normalized": round(norm_patch_texture, 3)},
            },
            "suspicious_flags": suspicious_flags,
        }

    # ------------------------------------------------------------------
    # Individual heuristic scoring functions
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sharpness(gray: np.ndarray) -> float:
        """Laplacian-variance blur metric. Higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_edge_density(gray: np.ndarray) -> float:
        """Fraction of pixels detected as edges by Canny. Higher = more fine detail."""
        edges = cv2.Canny(gray, 100, 200)
        return float(np.count_nonzero(edges)) / edges.size

    @staticmethod
    def _compute_frequency_detail(gray: np.ndarray) -> float:
        """
        Ratio of high-frequency energy to total energy in the 2D FFT
        magnitude spectrum. Fine microprint concentrates energy at high
        spatial frequency; blur/smoothing concentrates it near the
        zero-frequency (DC) center.
        """
        f = np.fft.fft2(gray.astype("float32"))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4  # inner quarter-radius treated as "low frequency"

        y_idx, x_idx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        high_freq_mask = dist_from_center > radius

        total_energy = magnitude.sum() + 1e-6
        high_freq_energy = magnitude[high_freq_mask].sum()
        return float(high_freq_energy / total_energy)

    @staticmethod
    def _compute_patch_texture(gray: np.ndarray, patch_size: int = 8) -> float:
        """
        Mean local intensity standard deviation over non-overlapping
        patches. Flat/smeared regions (e.g. from print smudging or a
        low-resolution reproduction) have low local variance even when
        overall image contrast looks normal.
        """
        h, w = gray.shape
        patch_stds = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = gray[y : y + patch_size, x : x + patch_size]
                patch_stds.append(float(patch.std()))

        if not patch_stds:
            return 0.0
        return float(np.mean(patch_stds))

    @staticmethod
    def _normalize(raw_value: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0
        return min(1.0, max(0.0, raw_value / threshold))

    @staticmethod
    def _empty_result(reasons: List[str]) -> dict:
        return {
            "clarity_score": 0.0,
            "passed": False,
            "sub_scores": {},
            "suspicious_flags": reasons,
        }
