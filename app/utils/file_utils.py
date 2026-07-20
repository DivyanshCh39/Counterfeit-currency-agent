"""
Filesystem helper utilities — saving uploads, annotated outputs, etc.
"""

import uuid
from pathlib import Path

import cv2
import numpy as np


def generate_unique_filename(extension: str = ".jpg") -> str:
    return f"{uuid.uuid4().hex}{extension}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(image: np.ndarray, output_dir: Path, filename: str) -> Path:
    ensure_dir(output_dir)
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), image)
    return output_path


def save_bytes(data: bytes, output_dir: Path, filename: str) -> Path:
    """Persists raw file bytes as-is (no decode/re-encode), used for the
    /upload endpoint so the original uploaded file is preserved exactly."""
    ensure_dir(output_dir)
    output_path = output_dir / filename
    output_path.write_bytes(data)
    return output_path


def is_allowed_extension(filename: str, allowed_extensions: tuple) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions
