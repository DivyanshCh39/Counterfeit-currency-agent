"""
Pydantic models for API responses.
This is the contract the frontend / mobile / POS clients will consume.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ImageQualityResult(BaseModel):
    is_acceptable: bool
    sharpness_score: float
    brightness_score: float = 0.0
    width: int
    height: int
    reason: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    detected: bool
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0


class DenominationResult(BaseModel):
    predicted_value: Optional[str] = None
    confidence: float = 0.0
    method: str = Field(
        default="heuristic_template_match",
        description="'ml_classifier' or 'heuristic_template_match'",
    )


class SerialNumberResult(BaseModel):
    serial_number: Optional[str] = Field(
        default=None, description="Cleaned, extracted serial number text."
    )
    normalized_text: Optional[str] = Field(
        default=None,
        description="Serial text after correcting commonly confused characters "
        "(O/0, I/1, etc.), only populated when the raw text failed the pattern check.",
    )
    confidence: float = Field(
        default=0.0, description="Raw OCR backend confidence, in [0, 1]."
    )
    quality_score: float = Field(
        default=0.0,
        description="Composite [0,1] score combining the 4 validation checks "
        "and OCR confidence — see SerialConsistencyService.",
    )
    format_valid: bool = False
    validation_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual check results: length_valid, characters_valid, "
        "spacing_valid, pattern_valid.",
    )
    suspicious_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for any failed validation check.",
    )
    region: Optional[BoundingBox] = None


class MicroprintResult(BaseModel):
    clarity_score: float = 0.0
    passed: bool = False
    suspicious_flags: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for any failed sub-score "
        "(sharpness, edge density, frequency detail, patch texture).",
    )
    region: Optional[BoundingBox] = None


class SecurityThreadResult(BaseModel):
    present: bool = False
    continuity_score: float = 0.0
    suspicious_flags: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations for any failed sub-score "
        "(region contrast, vertical feature strength, band continuity).",
    )
    region: Optional[BoundingBox] = None


class DenominationNumeralResult(BaseModel):
    extracted_text: Optional[str] = Field(
        default=None, description="Digits OCR'd from the large printed numeral, if any."
    )
    matches_predicted_denomination: Optional[bool] = Field(
        default=None,
        description="None if OCR found no readable numeral (inconclusive, not a "
        "mismatch); True/False once a numeral was actually read.",
    )
    confidence: float = 0.0
    suspicious_reasons: List[str] = Field(default_factory=list)
    region: Optional[BoundingBox] = None


class PromiseClauseResult(BaseModel):
    extracted_text: Optional[str] = None
    matched_keywords: List[str] = Field(default_factory=list)
    keyword_match_ratio: float = 0.0
    text_present: bool = False
    suspicious_reasons: List[str] = Field(default_factory=list)
    region: Optional[BoundingBox] = None


class CheckBreakdown(BaseModel):
    serial_number: SerialNumberResult
    microprint: MicroprintResult
    security_thread: SecurityThreadResult
    denomination_numeral: Optional[DenominationNumeralResult] = Field(
        default=None,
        description="Present only when the denomination's ROI template defines "
        "a 'denomination_numeral' region (see app/config/roi_config.py).",
    )
    promise_clause: Optional[PromiseClauseResult] = Field(
        default=None,
        description="Present only when the denomination's ROI template defines "
        "a 'promise_clause' region (see app/config/roi_config.py).",
    )


class UploadResponse(BaseModel):
    file_id: str = Field(
        description="Pass this to POST /analyze/{file_id} to run analysis "
        "on this stored file without re-uploading."
    )
    filename: str = Field(description="Original filename as sent by the client.")
    size_bytes: int
    content_type: Optional[str] = None
    stored_path: str = Field(description="Server-side filesystem path (for debugging).")


class AnalyzeResponse(BaseModel):
    verdict: str = Field(
        description="'likely genuine' | 'suspicious' | 'unclear'"
    )
    overall_score: float = Field(
        description="Weighted composite score in range [0, 100]"
    )
    feature_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature scores (each [0, 100]) that fed the composite score: "
        "image_quality, denomination_confidence, serial_quality, "
        "microprint_clarity, security_thread — plus numeral_consistency and/or "
        "promise_clause when this denomination's ROI template defines those "
        "optional regions (see app/config/roi_config.py). Everything in this "
        "dict actually influenced overall_score/verdict; contrast with the "
        "per-check `suspicious_reasons` fields and the top-level `notes` list, "
        "which are explanatory/diagnostic only and do not affect the score. "
        "Empty if the pipeline exited before the decision stage (e.g. no note "
        "detected).",
    )
    explanations: List[str] = Field(
        default_factory=list,
        description="Human-readable explanation of each feature score and "
        "why the composite score produced this verdict — output of "
        "DecisionService, distinct from the granular per-module warnings in `notes`.",
    )
    score_overridden: bool = Field(
        default=False,
        description="True when a hard-trigger rule forced overall_score/verdict "
        "independently of the weighted composite calculation — currently only "
        "set when a denomination-numeral mismatch was confirmed (printed numeral "
        "OCR disagreed with the classified denomination). When True, "
        "overall_score has been capped and verdict forced to 'suspicious' "
        "regardless of how the other signals scored; see `explanations` for "
        "the specific reason.",
    )
    denomination: DenominationResult
    image_quality: ImageQualityResult
    detection: DetectionResult
    checks: Optional[CheckBreakdown] = None
    annotated_image_path: Optional[str] = Field(
        default=None, description="Server-side filesystem path (for debugging)."
    )
    annotated_image_url: Optional[str] = Field(
        default=None,
        description="Web-accessible URL for the annotated output image "
        "(e.g. '/static/results/<id>.jpg'), for direct use in a frontend <img> tag.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Human-readable explanations, warnings, or disclaimers "
        "(e.g. flags placeholder/heuristic logic used, or granular "
        "suspicious reasons from individual checks).",
    )


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
