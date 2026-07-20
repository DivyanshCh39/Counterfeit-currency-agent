"""
Tests for ROIService and the denomination-aware ROI configuration
(app/config/roi_config.py).
"""

import numpy as np
import pytest

from app.config.roi_config import REQUIRED_REGIONS, get_roi_template, register_roi_template
from app.core.exceptions import ROIExtractionError
from app.services.roi_service import ROIService


def _dummy_aligned_note(width=1000, height=500) -> np.ndarray:
    return np.random.randint(0, 255, (height, width, 3), dtype="uint8")


def test_get_roi_template_falls_back_to_default_for_unknown_denomination():
    template, matched = get_roi_template("unknown")
    assert matched is False
    # DEFAULT now also carries optional regions (denomination_numeral,
    # promise_clause) alongside the 3 required ones — check required is a
    # subset, not exact equality.
    assert set(REQUIRED_REGIONS).issubset(template.keys())


def test_get_roi_template_returns_specific_template_when_configured():
    template, matched = get_roi_template("500")
    assert matched is True
    assert set(REQUIRED_REGIONS).issubset(template.keys())


def test_register_roi_template_rejects_incomplete_template():
    with pytest.raises(ValueError):
        register_roi_template("999", {"serial_number": (0.1, 0.1, 0.2, 0.2)})


def test_register_roi_template_accepts_complete_template_and_is_used():
    custom_template = {
        "serial_number": (0.1, 0.1, 0.3, 0.2),
        "microprint": (0.4, 0.4, 0.5, 0.5),
        "security_thread": (0.6, 0.0, 0.65, 1.0),
    }
    register_roi_template("999", custom_template)

    template, matched = get_roi_template("999")
    assert matched is True
    assert template == custom_template


def test_extract_rois_returns_metadata_for_all_regions_without_saving():
    service = ROIService(save_crops=False)
    image = _dummy_aligned_note()

    results = service.extract_rois(image, "500")

    assert set(REQUIRED_REGIONS).issubset(results.keys())
    for region_name, result in results.items():
        assert result.region_name == region_name
        assert result.crop.size > 0
        assert result.denomination_used == "500"
        assert result.template_matched is True
        assert result.saved_path is None  # saving disabled


def test_extract_rois_falls_back_and_flags_default_template():
    service = ROIService(save_crops=False)
    image = _dummy_aligned_note()

    results = service.extract_rois(image, "unknown")

    for result in results.values():
        assert result.template_matched is False


def test_extract_rois_saves_crops_when_enabled(tmp_path):
    service = ROIService(save_crops=True, output_dir=tmp_path)
    image = _dummy_aligned_note()

    results = service.extract_rois(image, "500", debug_tag="test_request")

    for region_name, result in results.items():
        assert result.saved_path is not None
        saved_file = tmp_path / "test_request" / f"{region_name}.jpg"
        assert saved_file.exists()


def test_extract_rois_raises_on_invalid_bounds():
    register_roi_template(
        "BAD",
        {
            "serial_number": (0.5, 0.5, 0.5, 0.9),  # x_max == x_min -> invalid
            "microprint": (0.1, 0.1, 0.2, 0.2),
            "security_thread": (0.3, 0.3, 0.4, 0.4),
        },
    )
    service = ROIService(save_crops=False)
    image = _dummy_aligned_note()

    with pytest.raises(ROIExtractionError):
        service.extract_rois(image, "BAD")
