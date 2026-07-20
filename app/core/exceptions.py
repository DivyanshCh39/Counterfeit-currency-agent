"""
Custom exception types for clearer error handling across the pipeline.
"""


class CurrencyAgentError(Exception):
    """Base exception for all pipeline-related errors."""


class InvalidImageError(CurrencyAgentError):
    """Raised when the uploaded file is not a valid/readable image."""


class ImageQualityTooLowError(CurrencyAgentError):
    """Raised when image quality gate fails (too blurry / too small)."""


class NoteNotDetectedError(CurrencyAgentError):
    """Raised when no currency note contour could be detected in the image."""


class AlignmentFailedError(CurrencyAgentError):
    """Raised when perspective correction fails to produce a valid warp."""


class DenominationClassificationError(CurrencyAgentError):
    """Raised when the denomination classifier fails or is inconclusive."""


class ROIExtractionError(CurrencyAgentError):
    """Raised when a required ROI cannot be cropped from the aligned note."""


class OCRProcessingError(CurrencyAgentError):
    """Raised when OCR fails to run (not when it simply finds no text)."""
