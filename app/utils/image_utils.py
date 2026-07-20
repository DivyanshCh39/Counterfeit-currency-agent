"""
Low-level image helper functions shared across services.
"""

from typing import Tuple

import cv2
import numpy as np

from app.config.logging_config import get_logger

logger = get_logger(__name__)


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image array."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes — invalid or corrupted file.")
    return image


def compute_sharpness(image: np.ndarray) -> float:
    """
    Laplacian-variance blur metric.
    Higher = sharper. This is a HEURISTIC proxy for image quality, not a
    learned quality model.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """
    Mean grayscale pixel intensity (0-255) as a HEURISTIC brightness proxy.
    Very low values indicate an underexposed/dark image; very high values
    indicate an overexposed/washed-out image — both hurt downstream OCR
    and ROI analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def validate_image_format(filename: str, allowed_extensions: tuple) -> bool:
    """
    Extension-based format validation. Kept here (in addition to
    file_utils.is_allowed_extension) so the preprocessing module can be
    used standalone without importing file_utils.
    """
    from pathlib import Path

    return Path(filename).suffix.lower() in allowed_extensions


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Returns (width, height)."""
    h, w = image.shape[:2]
    return w, h


def resize_keep_aspect(image: np.ndarray, target_width: int = 800) -> np.ndarray:
    """Resize image to a target width, preserving aspect ratio."""
    h, w = image.shape[:2]
    if w == 0:
        return image
    scale = target_width / float(w)
    new_dims = (target_width, int(h * scale))
    return cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
