"""
Image preprocessing module.

Consolidates every step that happens BEFORE denomination/ROI/authenticity
analysis:
    1. load image (from raw bytes)
    2. validate supported file format
    3. blur detection (Laplacian variance)
    4. brightness detection (mean grayscale intensity)
    5. resize while preserving aspect ratio
    6. note contour detection
    7. perspective transform / alignment
    8. (optional) save intermediate outputs for debugging

This module is intentionally self-contained so it can be reused by the
main pipeline (app/services/pipeline_service.py), by standalone scripts,
or by tests — it only depends on utils/config/core, not on other services.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.exceptions import (
    AlignmentFailedError,
    InvalidImageError,
)
from app.utils.file_utils import ensure_dir, generate_unique_filename, save_image
from app.utils.geometry_utils import four_point_warp
from app.utils.image_utils import (
    compute_brightness,
    compute_sharpness,
    get_image_dimensions,
    read_image_from_bytes,
    resize_keep_aspect,
    validate_image_format,
)
from app.services.note_boundary_service import NoteBoundaryService

logger = get_logger(__name__)


@dataclass
class QualityReport:
    sharpness_score: float
    brightness_score: float
    width: int
    height: int
    is_acceptable: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    """
    Full output of the preprocessing stage. Downstream services
    (denomination/ROI/etc.) should operate on `aligned_image` when
    `detected` and `aligned_image is not None`; otherwise the caller
    should treat this as an early-exit ("unclear") case.
    """

    original_image: np.ndarray
    resized_image: np.ndarray
    quality: QualityReport
    detected: bool
    corner_points: Optional[np.ndarray]
    aligned_image: Optional[np.ndarray]
    debug_paths: dict


class PreprocessingService:
    def __init__(self, debug_enabled: Optional[bool] = None):
        self.debug_enabled = (
            debug_enabled if debug_enabled is not None else settings.DEBUG_SAVE_INTERMEDIATE
        )
        self.debug_dir: Path = settings.DEBUG_OUTPUT_DIR
        self.note_boundary_service = NoteBoundaryService()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            return read_image_from_bytes(image_bytes)
        except ValueError as exc:
            raise InvalidImageError(str(exc)) from exc

    # ------------------------------------------------------------------
    # 2. Validate format
    # ------------------------------------------------------------------
    def validate_format(self, filename: str) -> None:
        if not validate_image_format(filename, settings.ALLOWED_IMAGE_EXTENSIONS):
            raise InvalidImageError(
                f"Unsupported file format for '{filename}'. "
                f"Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}"
            )

    # ------------------------------------------------------------------
    # 3 & 4. Blur + brightness -> quality report
    # ------------------------------------------------------------------
    def assess_quality(self, image: np.ndarray) -> QualityReport:
        width, height = get_image_dimensions(image)
        sharpness = compute_sharpness(image)
        brightness = compute_brightness(image)

        reasons: List[str] = []

        if width < settings.MIN_IMAGE_WIDTH or height < settings.MIN_IMAGE_HEIGHT:
            reasons.append(
                f"Resolution too low ({width}x{height}); "
                f"minimum is {settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT}."
            )

        if sharpness < settings.BLUR_LAPLACIAN_VAR_THRESHOLD:
            reasons.append(
                f"Image too blurry (sharpness={sharpness:.1f}, "
                f"minimum={settings.BLUR_LAPLACIAN_VAR_THRESHOLD})."
            )

        if brightness < settings.BRIGHTNESS_MIN_THRESHOLD:
            reasons.append(
                f"Image too dark (brightness={brightness:.1f}, "
                f"minimum={settings.BRIGHTNESS_MIN_THRESHOLD})."
            )
        elif brightness > settings.BRIGHTNESS_MAX_THRESHOLD:
            reasons.append(
                f"Image overexposed (brightness={brightness:.1f}, "
                f"maximum={settings.BRIGHTNESS_MAX_THRESHOLD})."
            )

        return QualityReport(
            sharpness_score=round(sharpness, 2),
            brightness_score=round(brightness, 2),
            width=width,
            height=height,
            is_acceptable=(len(reasons) == 0),
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # 5. Resize
    # ------------------------------------------------------------------
    def resize(self, image: np.ndarray, target_width: int = None) -> np.ndarray:
        width = target_width or settings.PREPROCESS_RESIZE_WIDTH
        return resize_keep_aspect(image, target_width=width)

    # ------------------------------------------------------------------
    # 6. Note contour detection
    # ------------------------------------------------------------------
    def detect_note_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Delegates to NoteBoundaryService, which tries an optional trained
        segmentation model first (if weights are present at
        settings.NOTE_BOUNDARY_MODEL_PATH) and always falls back to the
        original v1 OpenCV edge/contour heuristic otherwise — see
        app/services/note_boundary_service.py. Returns a 4x2 array of
        corner points, or None if no note-like boundary was found by
        either backend.
        """
        return self.note_boundary_service.detect(image)

    # ------------------------------------------------------------------
    # 7. Perspective transform / alignment
    # ------------------------------------------------------------------
    def align_note(self, image: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
        try:
            warped = four_point_warp(image, corner_points)
        except Exception as exc:  # noqa: BLE001
            raise AlignmentFailedError(str(exc)) from exc

        if warped is None or warped.size == 0:
            raise AlignmentFailedError("Perspective warp produced an empty image.")
        return warped

    # ------------------------------------------------------------------
    # 8. Debug saving
    # ------------------------------------------------------------------
    def _save_debug_step(
        self, image: np.ndarray, step_name: str, debug_tag: str
    ) -> Optional[str]:
        if not self.debug_enabled or image is None or image.size == 0:
            return None

        step_dir = self.debug_dir / debug_tag
        ensure_dir(step_dir)
        filename = f"{step_name}.jpg"
        output_path = save_image(image, step_dir, filename)
        logger.debug("Saved debug step '%s' to %s", step_name, output_path)
        return str(output_path)

    def _draw_contour_debug(
        self, image: np.ndarray, corner_points: Optional[np.ndarray]
    ) -> np.ndarray:
        overlay = image.copy()
        if corner_points is not None:
            pts = corner_points.astype(int).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        return overlay

    # ------------------------------------------------------------------
    # Orchestration entrypoint
    # ------------------------------------------------------------------
    def run(self, image_bytes: bytes, filename: str = "upload.jpg") -> PreprocessingResult:
        """
        Runs the full preprocessing sequence and returns a PreprocessingResult.
        Raises InvalidImageError if the file can't be loaded/validated.
        Does NOT raise on poor quality / failed detection — callers should
        inspect `.quality.is_acceptable` and `.detected` / `.aligned_image`
        to decide whether to short-circuit to an "unclear" verdict.
        """
        debug_tag = generate_unique_filename("").rstrip(".")
        debug_paths: dict = {}

        self.validate_format(filename)
        original_image = self.load_image(image_bytes)
        debug_paths["original"] = self._save_debug_step(original_image, "01_original", debug_tag)

        resized_image = self.resize(original_image)
        debug_paths["resized"] = self._save_debug_step(resized_image, "02_resized", debug_tag)

        quality = self.assess_quality(resized_image)

        if not quality.is_acceptable:
            logger.info("Preprocessing quality gate failed: %s", quality.reasons)
            return PreprocessingResult(
                original_image=original_image,
                resized_image=resized_image,
                quality=quality,
                detected=False,
                corner_points=None,
                aligned_image=None,
                debug_paths=debug_paths,
            )

        corner_points = self.detect_note_contour(resized_image)
        contour_debug = self._draw_contour_debug(resized_image, corner_points)
        debug_paths["contour"] = self._save_debug_step(contour_debug, "03_contour", debug_tag)

        if corner_points is None:
            logger.info("Preprocessing: no note contour detected.")
            return PreprocessingResult(
                original_image=original_image,
                resized_image=resized_image,
                quality=quality,
                detected=False,
                corner_points=None,
                aligned_image=None,
                debug_paths=debug_paths,
            )

        try:
            aligned_image = self.align_note(resized_image, corner_points)
        except AlignmentFailedError as exc:
            logger.warning("Alignment failed: %s", exc)
            return PreprocessingResult(
                original_image=original_image,
                resized_image=resized_image,
                quality=quality,
                detected=True,
                corner_points=corner_points,
                aligned_image=None,
                debug_paths=debug_paths,
            )

        debug_paths["aligned"] = self._save_debug_step(aligned_image, "04_aligned", debug_tag)

        return PreprocessingResult(
            original_image=original_image,
            resized_image=resized_image,
            quality=quality,
            detected=True,
            corner_points=corner_points,
            aligned_image=aligned_image,
            debug_paths=debug_paths,
        )
