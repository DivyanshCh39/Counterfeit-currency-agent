"""
Geometry helpers for contour handling and perspective warping.
Used by app/services/preprocessing_service.py for note detection and
alignment.
"""

from typing import Optional

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points as: top-left, top-right, bottom-right, bottom-left.
    Required before computing a perspective transform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to obtain a top-down view of the note
    given 4 corner points.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def find_largest_quadrilateral(
    image: np.ndarray, min_area_ratio: float = 0.15
) -> Optional[np.ndarray]:
    """
    HEURISTIC note detector: finds the largest 4-point contour in the image,
    assumed to be the currency note against a contrasting background.
    Returns None if no suitable contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area / float(image_area) < min_area_ratio:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

        # fallback: use the minimum-area bounding rectangle of the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype("float32")

    return None


def mask_to_quad(
    binary_mask: np.ndarray, min_area_ratio: float = 0.15
) -> Optional[np.ndarray]:
    """
    Converts a binary segmentation mask (note=nonzero, background=0, already
    resized to the ORIGINAL image's resolution) into the same (4, 2) float32
    quad-point format find_largest_quadrilateral() returns above, so both
    can feed the same downstream four_point_warp() unchanged.

    Used by app/models/note_boundary/onnx_backend.py to bridge a learned
    segmentation model's pixel-mask output back to a note boundary — kept
    here (rather than inline in the backend) so the deterministic
    mask->quad geometry, and its fallback behavior, stays identical to and
    alongside the equivalent edge-contour->quad logic above, instead of
    being duplicated.

    Returns None if the mask is empty or the largest blob is too small
    relative to the image (min_area_ratio, same meaning/default as
    find_largest_quadrilateral's parameter of the same name).
    """
    mask_u8 = (binary_mask > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask_area = mask_u8.shape[0] * mask_u8.shape[1]
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) / float(mask_area) < min_area_ratio:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")

    # Segmentation masks are rarely a clean quadrilateral after approxPolyDP
    # (soft/uneven edges are more common than with a Canny-edge contour) —
    # the minimum-area bounding rectangle is a robust fallback here, same
    # as find_largest_quadrilateral's own fallback for the same situation.
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return box.astype("float32")
