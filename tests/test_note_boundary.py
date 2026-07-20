"""
Tests for the note-boundary localization module:
    app/utils/geometry_utils.py (mask_to_quad)
    app/models/note_boundary/*
    app/services/note_boundary_service.py
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.note_boundary.heuristic_backend import HeuristicContourBackend
from app.models.note_boundary.onnx_backend import NoteBoundaryOnnxBackend
from app.services.note_boundary_service import NoteBoundaryService
from app.utils.geometry_utils import mask_to_quad


def _synthetic_note_image(width=800, height=400) -> np.ndarray:
    """A clean, high-contrast rectangle on a dark background — the same
    kind of image the existing find_largest_quadrilateral heuristic tests
    rely on being trivially detectable."""
    img = np.zeros((height, width, 3), dtype="uint8")
    cv2.rectangle(img, (150, 80), (650, 320), (220, 210, 200), -1)
    return img


# ----------------------------------------------------------------------
# mask_to_quad (geometry_utils.py)
# ----------------------------------------------------------------------

def test_mask_to_quad_returns_none_for_empty_mask():
    empty_mask = np.zeros((400, 800), dtype="uint8")
    assert mask_to_quad(empty_mask) is None


def test_mask_to_quad_returns_none_for_tiny_blob_below_area_ratio():
    mask = np.zeros((400, 800), dtype="uint8")
    mask[0:5, 0:5] = 255  # far below the default 0.15 min_area_ratio
    assert mask_to_quad(mask, min_area_ratio=0.15) is None


def test_mask_to_quad_extracts_correct_quad_from_rectangular_mask():
    mask = np.zeros((400, 800), dtype="uint8")
    mask[80:320, 150:650] = 255
    quad = mask_to_quad(mask, min_area_ratio=0.15)

    assert quad is not None
    assert quad.shape == (4, 2)
    xs, ys = quad[:, 0], quad[:, 1]
    assert xs.min() == pytest.approx(150, abs=2)
    assert xs.max() == pytest.approx(649, abs=2)
    assert ys.min() == pytest.approx(80, abs=2)
    assert ys.max() == pytest.approx(319, abs=2)


# ----------------------------------------------------------------------
# HeuristicContourBackend — must behave identically to the pre-existing
# find_largest_quadrilateral() it wraps.
# ----------------------------------------------------------------------

def test_heuristic_backend_is_always_available():
    assert HeuristicContourBackend().is_available() is True


def test_heuristic_backend_detects_synthetic_note():
    backend = HeuristicContourBackend()
    quad = backend.detect(_synthetic_note_image())
    assert quad is not None
    assert quad.shape == (4, 2)


def test_heuristic_backend_returns_none_on_blank_image():
    blank = np.full((400, 800, 3), 128, dtype="uint8")
    backend = HeuristicContourBackend()
    assert backend.detect(blank) is None


# ----------------------------------------------------------------------
# NoteBoundaryOnnxBackend — must stay inactive without real weights, and
# never raise.
# ----------------------------------------------------------------------

def test_onnx_backend_reports_unavailable_without_weights(tmp_path):
    missing_path = tmp_path / "does_not_exist.onnx"
    backend = NoteBoundaryOnnxBackend(missing_path)
    backend.load()
    assert backend.is_available() is False


def test_onnx_backend_detect_returns_none_when_unavailable():
    backend = NoteBoundaryOnnxBackend(Path("/nonexistent/weights.onnx"))
    backend.load()
    assert backend.detect(_synthetic_note_image()) is None


# ----------------------------------------------------------------------
# NoteBoundaryService — ordering/fallback behavior.
# ----------------------------------------------------------------------

def test_service_falls_back_to_heuristic_when_onnx_weights_absent():
    service = NoteBoundaryService()
    active_names = [b.name for b in service.backends if b.is_available()]

    assert "heuristic_contour" in active_names
    # No weights placed at settings.NOTE_BOUNDARY_MODEL_PATH in this test
    # environment -> ONNX backend must not report itself available.
    assert "ml_segmenter_onnx" not in active_names


def test_service_detects_synthetic_note_via_fallback_chain():
    service = NoteBoundaryService()
    quad = service.detect(_synthetic_note_image())
    assert quad is not None
    assert quad.shape == (4, 2)


def test_service_returns_none_when_no_backend_finds_a_boundary():
    service = NoteBoundaryService()
    blank = np.full((400, 800, 3), 128, dtype="uint8")
    assert service.detect(blank) is None
