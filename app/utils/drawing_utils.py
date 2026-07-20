"""
Drawing/annotation helpers to visualize pipeline results on the output image.
"""

from typing import Tuple

import cv2
import numpy as np

COLOR_OK = (0, 200, 0)         # green (BGR)
COLOR_WARNING = (0, 165, 255)  # orange
COLOR_FAIL = (0, 0, 255)       # red


def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = COLOR_OK,
    thickness: int = 2,
) -> np.ndarray:
    """box = (x_min, y_min, x_max, y_max)"""
    x_min, y_min, x_max, y_max = box
    annotated = image.copy()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)

    if label:
        text_y = max(y_min - 10, 15)
        cv2.putText(
            annotated,
            label,
            (x_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_verdict_banner(image: np.ndarray, verdict: str, score: float) -> np.ndarray:
    """Adds a top banner showing the final verdict and score (score is on a [0, 100] scale)."""
    color_map = {
        "likely genuine": COLOR_OK,
        "suspicious": COLOR_FAIL,
        "unclear": COLOR_WARNING,
    }
    color = color_map.get(verdict, COLOR_WARNING)

    annotated = image.copy()
    h, w = annotated.shape[:2]
    banner_height = 40
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), color, -1)
    annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)

    text = f"{verdict.upper()}  (score: {score:.1f}/100)"
    cv2.putText(
        annotated,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
