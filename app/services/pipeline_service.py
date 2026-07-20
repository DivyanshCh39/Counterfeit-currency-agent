"""
Pipeline orchestrator.
Chains all pipeline stages together and produces the final AnalyzeResponse.
Handles early exits (e.g. poor image quality -> "unclear" without running
downstream stages).

NOTE: image loading, format validation, blur/brightness quality checks,
note contour detection, and perspective alignment are now delegated to
app/services/preprocessing_service.py (see PreprocessingService).
"""

from typing import List

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.constants import VERDICT_UNCLEAR
from app.core.exceptions import InvalidImageError, ROIExtractionError
from app.schemas.response_schemas import (
    AnalyzeResponse,
    BoundingBox,
    CheckBreakdown,
    DenominationNumeralResult,
    DenominationResult,
    DetectionResult,
    ImageQualityResult,
    MicroprintResult,
    PromiseClauseResult,
    SecurityThreadResult,
    SerialNumberResult,
)
from app.services.decision_service import DecisionService
from app.services.denomination_service import DenominationService
from app.services.denomination_text_service import DenominationTextService
from app.services.microprint_service import MicroprintService
from app.services.ocr_service import OCRService
from app.services.preprocessing_service import PreprocessingResult, PreprocessingService
from app.services.roi_service import ROIService
from app.services.security_thread_service import SecurityThreadService
from app.utils.drawing_utils import COLOR_FAIL, COLOR_OK, draw_bounding_box, draw_verdict_banner
from app.utils.file_utils import generate_unique_filename, save_image

logger = get_logger(__name__)


class PipelineService:
    def __init__(self):
        self.preprocessing_service = PreprocessingService()
        self.denomination_service = DenominationService()
        self.roi_service = ROIService()
        self.ocr_service = OCRService()
        self.denomination_text_service = DenominationTextService(self.ocr_service.backend)
        self.microprint_service = MicroprintService()
        self.security_thread_service = SecurityThreadService()
        self.decision_service = DecisionService()

    def run(self, image_bytes: bytes, filename: str = "upload.jpg") -> AnalyzeResponse:
        """
        Entry point. Takes RAW image bytes (not a decoded array) so that
        PreprocessingService owns loading/validation/quality/alignment
        end-to-end. Raises InvalidImageError for unreadable/unsupported
        files (the router turns this into an HTTP 400).
        """
        notes: List[str] = [
            "PROTOTYPE DISCLAIMER: several checks use heuristic, non-learned "
            "logic due to lack of labeled counterfeit training data. Not for "
            "legal-grade authentication."
        ]

        # --- Stage 0: preprocessing (load, validate, quality, detect, align) ---
        try:
            pre: PreprocessingResult = self.preprocessing_service.run(image_bytes, filename)
        except InvalidImageError as exc:
            # Re-raise — router is responsible for mapping this to HTTP 400.
            raise exc

        quality_result = self._to_quality_schema(pre)

        if not pre.quality.is_acceptable:
            return AnalyzeResponse(
                verdict=VERDICT_UNCLEAR,
                overall_score=0.0,
                denomination=DenominationResult(),
                image_quality=quality_result,
                detection=DetectionResult(detected=False),
                notes=notes + pre.quality.reasons,
            )

        detection_result = self._to_detection_schema(pre)

        if not pre.detected:
            return self._early_exit(
                quality_result, "No currency note detected in image.", notes,
                detection_result=detection_result,
            )

        if pre.aligned_image is None:
            return self._early_exit(
                quality_result,
                "Note detected but perspective alignment failed.",
                notes,
                detection_result=detection_result,
            )

        aligned = pre.aligned_image

        # --- Stage 1: denomination classification ---
        denomination_result = self.denomination_service.classify(aligned)
        denom_label = denomination_result.predicted_value or "DEFAULT"

        # --- Stage 2: ROI extraction ---
        try:
            rois = self.roi_service.extract_rois(aligned, denom_label)
        except ROIExtractionError as exc:
            return self._early_exit(
                quality_result,
                str(exc),
                notes,
                detection_result=detection_result,
                denomination_result=denomination_result,
            )

        # --- Stage 3: serial number OCR + validation ---
        serial_roi = rois["serial_number"]
        ocr_result = self.ocr_service.read_serial_number(serial_roi.crop, denom_label)
        serial_result = SerialNumberResult(
            serial_number=ocr_result["serial_number"],
            normalized_text=ocr_result["normalized_text"],
            confidence=ocr_result["confidence"],
            quality_score=ocr_result["quality_score"],
            format_valid=ocr_result["format_valid"],
            validation_flags=ocr_result["validation_flags"],
            suspicious_reasons=ocr_result["suspicious_reasons"],
            region=serial_roi.bbox,
        )
        notes.extend(f"Serial number check: {reason}" for reason in serial_result.suspicious_reasons)

        # --- Stage 4: microprint clarity ---
        microprint_roi = rois["microprint"]
        microprint_raw = self.microprint_service.score(microprint_roi.crop)
        microprint_result = MicroprintResult(
            clarity_score=microprint_raw["clarity_score"],
            passed=microprint_raw["passed"],
            suspicious_flags=microprint_raw["suspicious_flags"],
            region=microprint_roi.bbox,
        )
        notes.extend(f"Microprint check: {flag}" for flag in microprint_result.suspicious_flags)

        # --- Stage 5: security thread ---
        thread_roi = rois["security_thread"]
        thread_raw = self.security_thread_service.check(thread_roi.crop)
        thread_result = SecurityThreadResult(
            present=thread_raw["present"],
            continuity_score=thread_raw["continuity_score"],
            suspicious_flags=thread_raw["suspicious_flags"],
            region=thread_roi.bbox,
        )
        notes.extend(f"Security thread check: {flag}" for flag in thread_result.suspicious_flags)

        # --- Stage 5.5: optional denomination numeral + promise-clause checks ---
        # Only run if this denomination's ROI template defines these regions
        # (see app/config/roi_config.py OPTIONAL_REGIONS).
        numeral_result = None
        promise_result = None

        numeral_roi = rois.get("denomination_numeral")
        if numeral_roi is not None:
            numeral_raw = self.denomination_text_service.read_denomination_numeral(
                numeral_roi.crop, denomination_result.predicted_value
            )
            numeral_result = DenominationNumeralResult(
                extracted_text=numeral_raw["extracted_text"],
                matches_predicted_denomination=numeral_raw["matches_predicted_denomination"],
                confidence=numeral_raw["confidence"],
                suspicious_reasons=numeral_raw["suspicious_reasons"],
                region=numeral_roi.bbox,
            )
            notes.extend(
                f"Denomination numeral check: {reason}"
                for reason in numeral_result.suspicious_reasons
            )

        promise_roi = rois.get("promise_clause")
        if promise_roi is not None:
            promise_raw = self.denomination_text_service.verify_promise_clause(promise_roi.crop)
            promise_result = PromiseClauseResult(
                extracted_text=promise_raw["extracted_text"],
                matched_keywords=promise_raw["matched_keywords"],
                keyword_match_ratio=promise_raw["keyword_match_ratio"],
                text_present=promise_raw["text_present"],
                suspicious_reasons=promise_raw["suspicious_reasons"],
                region=promise_roi.bbox,
            )
            notes.extend(
                f"Promise-clause check: {reason}" for reason in promise_result.suspicious_reasons
            )

        # --- Stage 6: final decision ---
        # Map the tri-state numeral result to the explicit state string
        # DecisionService expects. None here means "region wasn't checked
        # for this denomination" — distinct from "inconclusive" (checked,
        # but OCR found nothing legible), which is handled below.
        numeral_match_state = None
        if numeral_result is not None:
            if numeral_result.matches_predicted_denomination is True:
                numeral_match_state = "match"
            elif numeral_result.matches_predicted_denomination is False:
                numeral_match_state = "mismatch"
            else:
                numeral_match_state = "inconclusive"

        # keyword_match_ratio is already a continuous [0,1] score computed
        # by DenominationTextService — reused as-is, no OCR/extraction
        # logic touched.
        promise_clause_score = promise_result.keyword_match_ratio if promise_result is not None else None

        image_quality_score = self.decision_service.compute_image_quality_score(
            sharpness_score=pre.quality.sharpness_score,
            brightness_score=pre.quality.brightness_score,
        )
        decision = self.decision_service.decide(
            image_quality_score=image_quality_score,
            denomination_confidence=denomination_result.confidence,
            serial_quality_score=serial_result.quality_score,
            microprint_clarity_score=microprint_result.clarity_score,
            thread_continuity_score=thread_result.continuity_score,
            numeral_match_state=numeral_match_state,
            promise_clause_score=promise_clause_score,
        )

        # --- Stage 7: annotate output image ---
        annotated_path = self._annotate_and_save(
            aligned, decision["verdict"], decision["overall_score"],
            serial_roi.bbox, microprint_roi.bbox, thread_roi.bbox,
            serial_result.format_valid, microprint_result.passed, thread_result.present,
            numeral_result=numeral_result, promise_result=promise_result,
        )

        return AnalyzeResponse(
            verdict=decision["verdict"],
            overall_score=decision["overall_score"],
            feature_scores=decision["per_feature_scores"],
            explanations=decision["explanations"],
            score_overridden=decision["score_overridden"],
            denomination=denomination_result,
            image_quality=quality_result,
            detection=detection_result,
            checks=CheckBreakdown(
                serial_number=serial_result,
                microprint=microprint_result,
                security_thread=thread_result,
                denomination_numeral=numeral_result,
                promise_clause=promise_result,
            ),
            annotated_image_path=str(annotated_path),
            notes=notes,
        )

    def _to_quality_schema(self, pre: PreprocessingResult) -> ImageQualityResult:
        q = pre.quality
        return ImageQualityResult(
            is_acceptable=q.is_acceptable,
            sharpness_score=q.sharpness_score,
            brightness_score=q.brightness_score,
            width=q.width,
            height=q.height,
            reason=(q.reasons[0] if q.reasons else None),
            reasons=q.reasons,
        )

    def _to_detection_schema(self, pre: PreprocessingResult) -> DetectionResult:
        if not pre.detected or pre.corner_points is None:
            return DetectionResult(detected=False, confidence=0.0)

        x_coords = pre.corner_points[:, 0]
        y_coords = pre.corner_points[:, 1]
        bbox = BoundingBox(
            x_min=int(x_coords.min()),
            y_min=int(y_coords.min()),
            x_max=int(x_coords.max()),
            y_max=int(y_coords.max()),
        )
        # NOTE: confidence is a placeholder constant since contour detection
        # is heuristic (non-learned) — there is no true probability here.
        return DetectionResult(detected=True, bounding_box=bbox, confidence=0.6)

    def _early_exit(
        self,
        quality_result: ImageQualityResult,
        reason: str,
        notes: List[str],
        detection_result: DetectionResult = None,
        denomination_result: DenominationResult = None,
    ) -> AnalyzeResponse:
        logger.info("Pipeline early exit: %s", reason)
        return AnalyzeResponse(
            verdict=VERDICT_UNCLEAR,
            overall_score=0.0,
            denomination=denomination_result or DenominationResult(),
            image_quality=quality_result,
            detection=detection_result or DetectionResult(detected=False),
            notes=notes + [reason],
        )

    def _annotate_and_save(
        self,
        aligned_image,
        verdict: str,
        score: float,
        serial_bbox,
        microprint_bbox,
        thread_bbox,
        serial_ok: bool,
        microprint_ok: bool,
        thread_ok: bool,
        numeral_result=None,
        promise_result=None,
    ):
        annotated = aligned_image.copy()
        annotated = draw_bounding_box(
            annotated,
            (serial_bbox.x_min, serial_bbox.y_min, serial_bbox.x_max, serial_bbox.y_max),
            "Serial No.",
            COLOR_OK if serial_ok else COLOR_FAIL,
        )
        annotated = draw_bounding_box(
            annotated,
            (
                microprint_bbox.x_min,
                microprint_bbox.y_min,
                microprint_bbox.x_max,
                microprint_bbox.y_max,
            ),
            "Microprint",
            COLOR_OK if microprint_ok else COLOR_FAIL,
        )
        annotated = draw_bounding_box(
            annotated,
            (thread_bbox.x_min, thread_bbox.y_min, thread_bbox.x_max, thread_bbox.y_max),
            "Security Thread",
            COLOR_OK if thread_ok else COLOR_FAIL,
        )

        if numeral_result is not None and numeral_result.region is not None:
            b = numeral_result.region
            numeral_ok = numeral_result.matches_predicted_denomination is not False
            annotated = draw_bounding_box(
                annotated, (b.x_min, b.y_min, b.x_max, b.y_max),
                "Denom. No.", COLOR_OK if numeral_ok else COLOR_FAIL,
            )

        if promise_result is not None and promise_result.region is not None:
            b = promise_result.region
            annotated = draw_bounding_box(
                annotated, (b.x_min, b.y_min, b.x_max, b.y_max),
                "Promise Clause", COLOR_OK if promise_result.text_present else COLOR_FAIL,
            )

        annotated = draw_verdict_banner(annotated, verdict, score)

        filename = generate_unique_filename(".jpg")
        output_path = save_image(annotated, settings.SAMPLE_UPLOADS_DIR / "results", filename)
        return output_path
