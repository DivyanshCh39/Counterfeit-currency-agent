"""
Denomination numeral OCR + promise-clause text verification.

Two independent, lightweight checks that reuse the SAME already-loaded
OCR backend as OCRService (passed in via constructor) — no duplicate
model loading.

HEURISTIC (v1):
    - denomination numeral check: OCR the large printed numeral, compare
      (as digits only) against the denomination DenominationService
      predicted. A mismatch is a meaningful signal (the printed numeral
      should always agree with the note's actual denomination); OCR
      simply failing to read anything is NOT treated as suspicious on its
      own (that's usually just image quality), only an explicit mismatch is.
    - promise-clause check: OCR the small legal-text block and look for
      expected boilerplate keywords. This wording is standard across all
      denominations, so failure here is a weaker/softer signal than the
      numeral mismatch — mostly useful for flagging a blank/obscured/
      altered area rather than confirming authenticity.

Neither check currently feeds into DecisionService's weighted score (see
pipeline_service.py) — they're surfaced as informational checks + notes
for this iteration. A natural next step once labeled data exists would be
folding a numeral-mismatch penalty into the weighted score.
"""

import re
from typing import List, Optional

import numpy as np

from app.config.logging_config import get_logger
from app.config.text_verification_config import (
    EXPECTED_PROMISE_CLAUSE_KEYWORDS,
    PROMISE_CLAUSE_MIN_KEYWORD_MATCH_RATIO,
)
from app.models.ocr.base import OCRBackend

logger = get_logger(__name__)


class DenominationTextService:
    def __init__(self, ocr_backend: OCRBackend):
        self.ocr_backend = ocr_backend

    # ------------------------------------------------------------------
    # 1. Large denomination numeral
    # ------------------------------------------------------------------
    def read_denomination_numeral(
        self, roi_image: np.ndarray, predicted_denomination: Optional[str]
    ) -> dict:
        """
        Returns dict: {extracted_text, matches_predicted_denomination,
        confidence, suspicious_reasons}
        `matches_predicted_denomination` is None (not False) when OCR
        found no readable numeral at all — that's inconclusive, not a
        mismatch.
        """
        suspicious_reasons: List[str] = []

        if roi_image is None or roi_image.size == 0 or not self.ocr_backend.is_available():
            return {
                "extracted_text": None,
                "matches_predicted_denomination": None,
                "confidence": 0.0,
                "suspicious_reasons": [],
            }

        try:
            fragments = self.ocr_backend.read_text(roi_image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Denomination numeral OCR failed: %s", exc)
            return {
                "extracted_text": None,
                "matches_predicted_denomination": None,
                "confidence": 0.0,
                "suspicious_reasons": [],
            }

        if not fragments:
            return {
                "extracted_text": None,
                "matches_predicted_denomination": None,
                "confidence": 0.0,
                "suspicious_reasons": [],
            }

        # Same class of bug as OCRService.read_serial_number: taking only
        # the single highest-confidence fragment silently drops the rest
        # of the numeral if EasyOCR splits it (e.g. the "₹" symbol
        # detected separately from the digits). Sort left-to-right by
        # x_position and join everything — this is a single line of text,
        # so a simple x-sort gives correct reading order.
        ordered_fragments = sorted(fragments, key=lambda f: f[2])
        best_text = " ".join(text for text, _confidence, _x in ordered_fragments).strip()
        best_confidence = sum(conf for _text, conf, _x in ordered_fragments) / len(ordered_fragments)
        digits_only = re.sub(r"[^0-9]", "", best_text)

        matches: Optional[bool] = None
        if digits_only and predicted_denomination and predicted_denomination.isdigit():
            matches = digits_only == predicted_denomination
            if not matches:
                suspicious_reasons.append(
                    f"Printed numeral '{digits_only}' does not match the classified "
                    f"denomination '{predicted_denomination}'."
                )

        return {
            "extracted_text": digits_only or None,
            "matches_predicted_denomination": matches,
            "confidence": round(float(best_confidence), 3),
            "suspicious_reasons": suspicious_reasons,
        }

    # ------------------------------------------------------------------
    # 2. Promise-clause / boilerplate legal text
    # ------------------------------------------------------------------
    def verify_promise_clause(self, roi_image: np.ndarray) -> dict:
        """
        Returns dict: {extracted_text, matched_keywords, keyword_match_ratio,
        text_present, suspicious_reasons}
        """
        suspicious_reasons: List[str] = []

        if roi_image is None or roi_image.size == 0 or not self.ocr_backend.is_available():
            return {
                "extracted_text": None,
                "matched_keywords": [],
                "keyword_match_ratio": 0.0,
                "text_present": False,
                "suspicious_reasons": [],
            }

        try:
            fragments = self.ocr_backend.read_text(roi_image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Promise-clause OCR failed: %s", exc)
            return {
                "extracted_text": None,
                "matched_keywords": [],
                "keyword_match_ratio": 0.0,
                "text_present": False,
                "suspicious_reasons": [],
            }

        # NOTE: deliberately NOT sorted by x_position here (unlike the
        # single-line serial number / numeral reads) — this ROI is a
        # multi-line paragraph, and EasyOCR's natural detection order is
        # already top-to-bottom. Sorting by x alone would interleave lines.
        combined_text = " ".join(text for text, _confidence, _x in fragments).upper()

        matched_keywords = [
            kw for kw in EXPECTED_PROMISE_CLAUSE_KEYWORDS if kw in combined_text
        ]
        match_ratio = (
            len(matched_keywords) / len(EXPECTED_PROMISE_CLAUSE_KEYWORDS)
            if EXPECTED_PROMISE_CLAUSE_KEYWORDS
            else 0.0
        )
        text_present = match_ratio >= PROMISE_CLAUSE_MIN_KEYWORD_MATCH_RATIO

        if not text_present:
            suspicious_reasons.append(
                f"Expected legal text (promise-to-pay clause) not clearly detected "
                f"({len(matched_keywords)}/{len(EXPECTED_PROMISE_CLAUSE_KEYWORDS)} "
                f"keywords found) — may be low image quality, or the area is "
                f"blank/altered."
            )

        return {
            "extracted_text": combined_text or None,
            "matched_keywords": matched_keywords,
            "keyword_match_ratio": round(match_ratio, 3),
            "text_present": text_present,
            "suspicious_reasons": suspicious_reasons,
        }
