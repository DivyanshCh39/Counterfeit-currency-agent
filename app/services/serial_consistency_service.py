"""
Serial number cleaning + validation module.

Takes raw OCR output for the serial number ROI and:
1. cleans it (strips whitespace/punctuation noise, uppercases)
2. validates it against four independent checks:
   - length          (SERIAL_LENGTH_RANGES)
   - allowed characters (ALLOWED_CHARACTERS_PATTERN)
   - spacing         (flags irregular/fragmented OCR output)
   - pattern         (full denomination-aware regex, SERIAL_PATTERNS)
3. produces a composite confidence/quality score
4. produces a human-readable list of suspicious_reasons for anything that failed

HEURISTIC (v1): all four checks are hand-specified rules, not learned or
checksum-based — genuine currency serial numbers follow issuing-authority-
specific generation algorithms this prototype does not model. See README
limitations.
"""

import re
from typing import Dict, List, Optional

from app.config.logging_config import get_logger
from app.config.serial_config import (
    ALLOWED_CHARACTERS_PATTERN,
    CHARACTER_CONFUSION_MAP,
    get_expected_length_range,
    get_serial_pattern,
)

logger = get_logger(__name__)

# Any whitespace/control character run longer than this inside the raw OCR
# fragment is treated as "irregular spacing" (e.g. OCR split one serial
# number into disjoint pieces).
_MAX_ALLOWED_INTERNAL_WHITESPACE_RUN = 1


class SerialConsistencyService:
    def evaluate(
        self, raw_ocr_text: Optional[str], ocr_confidence: float, denomination: str
    ) -> dict:
        """
        Args:
            raw_ocr_text: unprocessed OCR output for the serial number ROI
                (may be None/empty if OCR found nothing).
            ocr_confidence: confidence reported by the OCR backend, in [0, 1].
            denomination: label from DenominationService, used to select
                the expected length/pattern for this note.

        Returns dict with:
            serial_number, normalized_text, confidence, quality_score,
            validation_flags (dict of 4 booleans + overall),
            suspicious_reasons (list[str])
        """
        suspicious_reasons: List[str] = []

        if not raw_ocr_text or not raw_ocr_text.strip():
            suspicious_reasons.append("No text detected in serial number region.")
            return self._build_result(
                serial_number=None,
                normalized_text=None,
                confidence=0.0,
                validation_flags=self._empty_flags(),
                suspicious_reasons=suspicious_reasons,
            )

        # --- Check 1: spacing (evaluated on the RAW pre-clean text) ---
        spacing_valid = self._check_spacing(raw_ocr_text)
        if not spacing_valid:
            suspicious_reasons.append(
                "Irregular spacing/fragmentation detected in OCR output "
                "(possible broken or unreliable read)."
            )

        cleaned = self._clean(raw_ocr_text)

        # --- Check 2: length ---
        (min_len, max_len), length_matched_denom = get_expected_length_range(denomination)
        length_valid = min_len <= len(cleaned) <= max_len
        if not length_valid:
            suspicious_reasons.append(
                f"Serial number length ({len(cleaned)}) outside expected range "
                f"[{min_len}-{max_len}] for denomination '{denomination}'."
            )

        # --- Check 3: allowed characters ---
        characters_valid = bool(re.match(ALLOWED_CHARACTERS_PATTERN, cleaned))
        if not characters_valid:
            suspicious_reasons.append(
                "Cleaned serial number contains disallowed characters."
            )

        # --- Check 4: full pattern (structure) ---
        pattern, pattern_matched_denom = get_serial_pattern(denomination)
        pattern_valid = bool(re.match(pattern, cleaned))

        # Second, lower-trust attempt: normalize commonly OCR-confused
        # characters and re-check the pattern.
        normalized_text = None
        normalized_valid = False
        if not pattern_valid:
            normalized_text = self._normalize_confusable_chars(cleaned)
            if normalized_text != cleaned:
                normalized_valid = bool(re.match(pattern, normalized_text))

        if not (pattern_valid or normalized_valid):
            suspicious_reasons.append(
                f"Does not match expected serial format for denomination '{denomination}'."
            )
        elif normalized_valid and not pattern_valid:
            suspicious_reasons.append(
                "Serial number only matches expected format after correcting "
                "likely OCR character misreads (lower confidence)."
            )

        if not pattern_matched_denom:
            suspicious_reasons.append(
                f"No dedicated serial format configured for denomination "
                f"'{denomination}' — validated against the DEFAULT pattern only."
            )

        validation_flags = {
            "length_valid": length_valid,
            "characters_valid": characters_valid,
            "spacing_valid": spacing_valid,
            "pattern_valid": pattern_valid or normalized_valid,
        }

        used_normalization = normalized_valid and not pattern_valid
        quality_score = self._compute_quality_score(
            validation_flags, ocr_confidence, used_normalization
        )

        return self._build_result(
            serial_number=cleaned,
            normalized_text=normalized_text,
            confidence=ocr_confidence,
            validation_flags=validation_flags,
            suspicious_reasons=suspicious_reasons,
            quality_score=quality_score,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    @staticmethod
    def _check_spacing(raw_text: str) -> bool:
        """
        Flags irregular internal whitespace: tabs, newlines, or more than
        one consecutive space, or leading/trailing whitespace beyond a
        simple trim. Most serial number formats are a single contiguous
        token, so any of this usually signals a broken/fragmented OCR read.
        """
        if "\t" in raw_text or "\n" in raw_text:
            return False
        if re.search(r" {2,}", raw_text):
            return False
        return True

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", text).upper()

    @staticmethod
    def _normalize_confusable_chars(text: str) -> str:
        return "".join(CHARACTER_CONFUSION_MAP.get(ch, ch) for ch in text)

    # ------------------------------------------------------------------
    # Scoring + result assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_quality_score(
        flags: Dict[str, bool], ocr_confidence: float, used_normalization: bool = False
    ) -> float:
        # PLACEHOLDER weighting — not derived from real labeled data.
        checks_passed = sum(1 for v in flags.values() if v)
        total_checks = len(flags)
        rule_score = checks_passed / total_checks if total_checks else 0.0

        score = 0.6 * rule_score + 0.4 * ocr_confidence
        if used_normalization:
            # pattern only matched after correcting likely OCR misreads -> less trust
            score *= 0.8
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def _empty_flags() -> Dict[str, bool]:
        return {
            "length_valid": False,
            "characters_valid": False,
            "spacing_valid": False,
            "pattern_valid": False,
        }

    @staticmethod
    def _build_result(
        serial_number: Optional[str],
        normalized_text: Optional[str],
        confidence: float,
        validation_flags: Dict[str, bool],
        suspicious_reasons: List[str],
        quality_score: float = 0.0,
    ) -> dict:
        return {
            "serial_number": serial_number,
            "normalized_text": normalized_text,
            "confidence": round(float(confidence), 3),
            "quality_score": quality_score,
            "validation_flags": validation_flags,
            "format_valid": validation_flags["pattern_valid"],
            "suspicious_reasons": suspicious_reasons,
        }
