"""
Denomination-aware ROI configuration.

Each template maps region name -> fractional bounding box
(x_min, y_min, x_max, y_max), expressed as fractions of the ALIGNED note's
width/height (i.e. after perspective correction, so (0,0) is the top-left
corner of the note and (1,1) is the bottom-right corner regardless of the
original photo's resolution).

The "500" and "100" templates below were measured directly against real
note photos (pixel-by-pixel visual calibration, not guessed) — see the
project history for the calibration images. DEFAULT and any other
denomination template still use the pre-calibration placeholder estimates
and should be recalibrated the same way once real reference images for
those denominations are available. Use tools/roi_calibrator.html for that.

This module is the single source of truth for ROI geometry. ROIService
(app/services/roi_service.py) only reads from here — it contains no
hardcoded coordinates itself, so adding/adjusting a denomination template
never requires touching service code.

Note: "denomination_numeral" and "promise_clause" are OPTIONAL regions —
not in REQUIRED_REGIONS — so templates that omit them (e.g. custom
templates registered elsewhere, or older code) remain valid; ROIService
and pipeline_service.py simply skip those extra checks when the region
isn't defined for a given denomination.
"""

from typing import Dict, Tuple

FractionalBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), each in [0, 1]

# Every template must define at least these regions (core MVP checks).
REQUIRED_REGIONS: Tuple[str, ...] = ("serial_number", "microprint", "security_thread")

# Optional regions — supported by ROIService/pipeline_service when present,
# but not required for a template to be considered valid.
OPTIONAL_REGIONS: Tuple[str, ...] = ("denomination_numeral", "promise_clause")

# Used when a denomination has no dedicated template yet.
DEFAULT_TEMPLATE_KEY = "DEFAULT"

ROI_TEMPLATES: Dict[str, Dict[str, FractionalBox]] = {
    # Calibrated against a real ₹500 note photo (see module docstring).
    "500": {
        "serial_number": (0.56, 0.83, 0.92, 0.96),
        "microprint": (0.19, 0.38, 0.33, 0.75),
        "security_thread": (0.53, 0.05, 0.60, 0.90),
        "denomination_numeral": (0.66, 0.64, 0.93, 0.82),
        "promise_clause": (0.585, 0.20, 0.78, 0.66),
    },
    # Calibrated against a real ₹100 note photo (aligned reference image,
    # same pixel-by-pixel method used for "500"). Security thread location
    # was confirmed via column-wise dark-pixel analysis of the aligned
    # image, since the dashed thread isn't reliably visible to the eye at
    # this compression level.
    "100": {
        "serial_number": (0.585, 0.84, 0.995, 0.985),
        "microprint": (0.25, 0.35, 0.40, 0.75),
        "security_thread": (0.58, 0.02, 0.64, 0.98),
        "denomination_numeral": (0.74, 0.58, 1.0, 0.83),
        "promise_clause": (0.60, 0.20, 0.80, 0.70),
    },
    # DEFAULT fallback template used for any denomination not explicitly
    # listed above (including "unknown" from DenominationService). Mirrors
    # the calibrated "500" template as the best available starting point.
    DEFAULT_TEMPLATE_KEY: {
        "serial_number": (0.56, 0.83, 0.92, 0.96),
        "microprint": (0.19, 0.38, 0.33, 0.75),
        "security_thread": (0.53, 0.05, 0.60, 0.90),
        "denomination_numeral": (0.68, 0.66, 0.87, 0.81),
        "promise_clause": (0.61, 0.22, 0.83, 0.65),
    },
}


def get_roi_template(denomination: str) -> Tuple[Dict[str, FractionalBox], bool]:
    """
    Returns (template, is_denomination_specific).

    is_denomination_specific is False whenever the DEFAULT template had to
    be used — callers (ROIService) surface this in ROI metadata so
    downstream consumers/logs know the crop coordinates were not
    calibrated for this specific note design.
    """
    if denomination in ROI_TEMPLATES:
        return ROI_TEMPLATES[denomination], True
    return ROI_TEMPLATES[DEFAULT_TEMPLATE_KEY], False


def register_roi_template(denomination: str, template: Dict[str, FractionalBox]) -> None:
    """
    Adds or overrides a per-denomination template at runtime — e.g. from a
    future calibration script/notebook that measures real note geometry
    and wants to persist it without a code change/redeploy.

    Raises ValueError if the template is missing any required region.
    """
    missing = set(REQUIRED_REGIONS) - set(template.keys())
    if missing:
        raise ValueError(
            f"ROI template for '{denomination}' is missing required regions: {sorted(missing)}"
        )
    ROI_TEMPLATES[denomination] = template


def list_configured_denominations() -> Tuple[str, ...]:
    """Denominations with a dedicated (non-DEFAULT) template."""
    return tuple(k for k in ROI_TEMPLATES if k != DEFAULT_TEMPLATE_KEY)
