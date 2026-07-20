"""
Abstract interface for OCR backends used to read the serial number ROI.

Mirrors app/models/denomination_classifier/base.py: OCRService talks only
to this interface, so a future custom-trained serial-recognition model
(e.g. a CRNN fine-tuned on real note serial fonts) can be dropped in as a
new backend without touching OCRService, the pipeline, or the API schema.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class OCRBackend(ABC):
    #: Human-readable identifier surfaced in logs / debugging.
    name: str = "unnamed_ocr_backend"

    def load(self) -> None:
        """Optional hook for backends that need to initialize resources."""
        return None

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, image: np.ndarray) -> List[Tuple[str, float, float]]:
        """
        Returns a list of (text_fragment, confidence, x_position) tuples
        detected in the image. Empty list if nothing was detected. Never
        raises for "no text found" — only for genuine backend failures.

        x_position is the fragment's leftmost x-coordinate in the input
        image (pixels). It exists purely so callers can sort fragments
        into left-to-right reading order before joining them — some OCR
        backends (e.g. EasyOCR) can return multi-fragment detections in an
        order that doesn't match reading order, especially when a serial
        number is visually split by a printed security mark. Backends that
        only ever return a single combined fragment (e.g. Tesseract in
        single-line mode) may report 0.0 here since ordering is moot.
        """
        raise NotImplementedError
