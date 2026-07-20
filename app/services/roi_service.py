"""
ROI extraction module.

Crops serial number / microprint / security thread regions from an
ALIGNED note image, using denomination-aware fractional coordinates
defined in app/config/roi_config.py (never hardcoded here).

Design notes:
- This module owns ROI *cropping + metadata*, nothing else. It doesn't
  know or care how the crops get analyzed downstream (OCR, clarity
  scoring, etc.) — see ocr_service.py / microprint_service.py /
  security_thread_service.py for that.
- Per-denomination templates are entirely config-driven (roi_config.py).
  Adding a new denomination's calibrated coordinates never requires
  touching this file.
- Optionally saves each crop to disk for debugging/inspection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from app.config.logging_config import get_logger
from app.config.roi_config import get_roi_template
from app.config.settings import settings
from app.core.exceptions import ROIExtractionError
from app.schemas.response_schemas import BoundingBox
from app.utils.file_utils import ensure_dir, generate_unique_filename, save_image

logger = get_logger(__name__)

RegionName = str  # "serial_number" | "microprint" | "security_thread"


@dataclass
class ROIExtractionResult:
    """
    Full metadata + pixel data for a single extracted region, handed to
    downstream analysis services (OCR, microprint scoring, thread check)
    and also reportable to the API layer for transparency/debugging.
    """

    region_name: RegionName
    crop: np.ndarray
    bbox: BoundingBox
    denomination_used: str
    template_matched: bool  # False if the DEFAULT template had to be used
    saved_path: Optional[str] = None


class ROIService:
    def __init__(self, save_crops: Optional[bool] = None, output_dir: Optional[Path] = None):
        self.save_crops = save_crops if save_crops is not None else settings.ROI_SAVE_CROPS
        self.output_dir: Path = output_dir or settings.ROI_OUTPUT_DIR

    def extract_rois(
        self, aligned_image: np.ndarray, denomination: str, debug_tag: Optional[str] = None
    ) -> Dict[RegionName, ROIExtractionResult]:
        """
        Args:
            aligned_image: perspective-corrected BGR note image.
            denomination: label from DenominationService (may be "unknown"
                or any value with no dedicated template — falls back to
                the DEFAULT template automatically).
            debug_tag: optional folder name to group this request's saved
                crops under (e.g. a request/session id). Auto-generated if
                omitted.

        Returns:
            dict keyed by region name -> ROIExtractionResult.

        Raises:
            ROIExtractionError if a region's computed crop is empty/invalid.
        """
        template, template_matched = get_roi_template(denomination)
        h, w = aligned_image.shape[:2]

        if not template_matched:
            logger.info(
                "No dedicated ROI template for denomination '%s' — using DEFAULT.",
                denomination,
            )

        tag = debug_tag or generate_unique_filename("").rstrip(".")

        results: Dict[RegionName, ROIExtractionResult] = {}

        for region_name, (fx_min, fy_min, fx_max, fy_max) in template.items():
            x_min = int(fx_min * w)
            y_min = int(fy_min * h)
            x_max = int(fx_max * w)
            y_max = int(fy_max * h)

            if x_max <= x_min or y_max <= y_min:
                raise ROIExtractionError(f"Invalid ROI bounds for region '{region_name}'")

            crop = aligned_image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                raise ROIExtractionError(f"Empty crop for region '{region_name}'")

            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

            saved_path = None
            if self.save_crops:
                saved_path = self._save_crop(crop, region_name, tag)

            results[region_name] = ROIExtractionResult(
                region_name=region_name,
                crop=crop,
                bbox=bbox,
                denomination_used=denomination,
                template_matched=template_matched,
                saved_path=saved_path,
            )

        return results

    def _save_crop(self, crop: np.ndarray, region_name: str, tag: str) -> str:
        crop_dir = self.output_dir / tag
        ensure_dir(crop_dir)
        filename = f"{region_name}.jpg"
        output_path = save_image(crop, crop_dir, filename)
        logger.debug("Saved ROI crop '%s' to %s", region_name, output_path)
        return str(output_path)
