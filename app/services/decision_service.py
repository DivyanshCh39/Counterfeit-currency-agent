"""
Final decision engine.

Combines up to 7 independent signals into one composite score and verdict:
    1. image_quality           — how comfortably the photo cleared the
                                   sharpness/brightness minimums (not a
                                   pass/fail gate — that already happened
                                   earlier in PreprocessingService)
    2. denomination_confidence  — how confident DenominationService was
                                   about which note this is
    3. serial_quality           — composite score from SerialConsistencyService
    4. microprint_clarity       — composite score from MicroprintService
    5. security_thread          — composite score from SecurityThreadService
    6. numeral_consistency      — OPTIONAL. Does the OCR'd large printed
                                   numeral agree with the classified
                                   denomination? Only present when this
                                   denomination's ROI template defines a
                                   'denomination_numeral' region.
    7. promise_clause           — OPTIONAL. How much of the expected
                                   promise-to-pay legal text was detected?
                                   Only present when this denomination's
                                   ROI template defines a 'promise_clause'
                                   region.

RULE-BASED weighted sum (v1) — explicitly not ML, since no labeled
genuine/counterfeit dataset is available for this prototype. Weights and
thresholds are all in app/config/settings.py, tunable without code changes.
See README limitations.

Inputs to decide() are on a [0, 1] scale (matching every upstream service's
own scoring convention). Outputs (overall_score, per_feature_scores) are
rescaled to [0, 100] for the API/UI, since that reads more naturally as a
"score out of 100" for end users. Verdict thresholds in settings.py are
expressed on that same 0-100 output scale.

NOTE on denomination_confidence as an authenticity signal: it is NOT a
fraud signal by itself (an unusual photo angle can lower it just as
easily as a forged note can) — it is included at a deliberately low
weight, mainly to catch cases where the note doesn't visually resemble
any known reference well enough to trust the ROI template that was used
for the other checks.

--- HYBRID SCORING FOR numeral_consistency / promise_clause (this revision) ---
These two signals were previously computed upstream and surfaced only as
human-readable "suspicious_reasons" text — they had NO effect on
overall_score or verdict. That gap is closed here, using a deliberate
HYBRID approach rather than a single weighting scheme, because the two
signals differ in how much a single reading should be trusted:

  * numeral_consistency is WEIGHTED (settings.WEIGHT_NUMERAL_CONSISTENCY)
    *and* backed by a HARD TRIGGER: if the printed numeral was actually
    read and it definitively disagrees with the classified denomination,
    that is a strong, specific signal (e.g. classifier says '500' but the
    note itself prints '100') that should not be quietly averaged away by
    a handful of high heuristic-quality scores elsewhere. When this fires,
    overall_score is capped at settings.NUMERAL_MISMATCH_SCORE_CAP and the
    verdict is forced to "suspicious" — see `numeral_match_state` below.
    An INCONCLUSIVE read (OCR found nothing legible) is NOT treated as a
    mismatch — it contributes a neutral, mildly-favorable score
    (settings.NUMERAL_INCONCLUSIVE_SCORE) rather than being punished,
    since a failed read is usually a photo-angle/lighting issue.

  * promise_clause is WEIGHTED ONLY (settings.WEIGHT_PROMISE_CLAUSE), no
    hard trigger — missing/garbled legal-tender text is a moderate signal
    that is far more sensitive to ordinary OCR failure than a numeral
    mismatch is, so it nudges the composite score rather than forcing a
    verdict on its own.

Both signals are OPTIONAL inputs. When a denomination's ROI template
doesn't define these regions, pass None (the default) and the composite
score is computed exactly as in the original 5-signal formula — the
weights for present signals are renormalized by their own sum, so omitting
these two never silently deflates the score for denominations that don't
have them.
"""

from typing import Dict, List, Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.constants import (
    VERDICT_LIKELY_GENUINE,
    VERDICT_SUSPICIOUS,
    VERDICT_UNCLEAR,
)

logger = get_logger(__name__)

# Valid values for the `numeral_match_state` argument to decide().
NUMERAL_STATE_MATCH = "match"
NUMERAL_STATE_MISMATCH = "mismatch"
NUMERAL_STATE_INCONCLUSIVE = "inconclusive"
_VALID_NUMERAL_STATES = (NUMERAL_STATE_MATCH, NUMERAL_STATE_MISMATCH, NUMERAL_STATE_INCONCLUSIVE, None)


class DecisionService:
    def __init__(self):
        self.w_image_quality = settings.WEIGHT_IMAGE_QUALITY
        self.w_denomination = settings.WEIGHT_DENOMINATION_CONFIDENCE
        self.w_serial = settings.WEIGHT_SERIAL_CHECK
        self.w_microprint = settings.WEIGHT_MICROPRINT_CHECK
        self.w_thread = settings.WEIGHT_SECURITY_THREAD_CHECK
        self.w_numeral_consistency = settings.WEIGHT_NUMERAL_CONSISTENCY
        self.w_promise_clause = settings.WEIGHT_PROMISE_CLAUSE

        self.genuine_threshold = settings.GENUINE_SCORE_THRESHOLD
        self.suspicious_threshold = settings.SUSPICIOUS_SCORE_THRESHOLD

        self.high_band = settings.DECISION_FEATURE_HIGH_THRESHOLD
        self.low_band = settings.DECISION_FEATURE_LOW_THRESHOLD

        self.numeral_inconclusive_score = settings.NUMERAL_INCONCLUSIVE_SCORE
        self.numeral_mismatch_score_cap = settings.NUMERAL_MISMATCH_SCORE_CAP

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def decide(
        self,
        image_quality_score: float,
        denomination_confidence: float,
        serial_quality_score: float,
        microprint_clarity_score: float,
        thread_continuity_score: float,
        numeral_match_state: Optional[str] = None,
        promise_clause_score: Optional[float] = None,
    ) -> dict:
        """
        Args:
            image_quality_score, denomination_confidence,
            serial_quality_score, microprint_clarity_score,
            thread_continuity_score: all on a [0, 1] scale (unchanged
                contract with upstream services).
            numeral_match_state: one of "match" / "mismatch" /
                "inconclusive" / None. None means this denomination's ROI
                template doesn't define a denomination_numeral region, so
                the check didn't run at all — NOT the same as
                "inconclusive" (which means it ran but OCR found nothing
                legible). Callers should map
                DenominationNumeralResult.matches_predicted_denomination
                (True/False/None) to ("match"/"mismatch"/"inconclusive")
                themselves, and pass plain None only when numeral_result
                itself is None (region not checked for this note).
            promise_clause_score: PromiseClauseResult.keyword_match_ratio
                ([0, 1]), or None if this denomination's ROI template
                doesn't define a promise_clause region.

        Returns dict with:
            verdict, overall_score (0-100), per_feature_scores (dict, each
            0-100 — includes "numeral_consistency"/"promise_clause" keys
            only when those inputs were provided), explanations (list[str]),
            score_overridden (bool — True if the numeral-mismatch hard
            trigger fired and forced the score/verdict below).
        """
        if numeral_match_state not in _VALID_NUMERAL_STATES:
            raise ValueError(
                f"numeral_match_state must be one of {_VALID_NUMERAL_STATES}, "
                f"got {numeral_match_state!r}"
            )

        raw_0to1 = {
            "image_quality": image_quality_score,
            "denomination_confidence": denomination_confidence,
            "serial_quality": serial_quality_score,
            "microprint_clarity": microprint_clarity_score,
            "security_thread": thread_continuity_score,
        }

        # Convert the 5 core signals to the reported 0-100 scale — all
        # further math, thresholds, and explanation text operate here.
        per_feature_scores: Dict[str, float] = {
            key: round(min(1.0, max(0.0, value)) * 100, 1) for key, value in raw_0to1.items()
        }

        # (score_0to100, weight) pairs actually being combined this call.
        # Starts with the 5 core signals, which alone sum to weight 1.0 —
        # exactly reproducing the pre-existing formula when neither
        # optional signal is supplied.
        weighted_pairs: List[tuple] = [
            (per_feature_scores["image_quality"], self.w_image_quality),
            (per_feature_scores["denomination_confidence"], self.w_denomination),
            (per_feature_scores["serial_quality"], self.w_serial),
            (per_feature_scores["microprint_clarity"], self.w_microprint),
            (per_feature_scores["security_thread"], self.w_thread),
        ]

        numeral_mismatch_confirmed = False
        if numeral_match_state is not None:
            if numeral_match_state == NUMERAL_STATE_MATCH:
                numeral_score_0to1 = 1.0
            elif numeral_match_state == NUMERAL_STATE_MISMATCH:
                numeral_score_0to1 = 0.0
                numeral_mismatch_confirmed = True
            else:  # inconclusive
                numeral_score_0to1 = self.numeral_inconclusive_score

            per_feature_scores["numeral_consistency"] = round(
                min(1.0, max(0.0, numeral_score_0to1)) * 100, 1
            )
            weighted_pairs.append(
                (per_feature_scores["numeral_consistency"], self.w_numeral_consistency)
            )

        if promise_clause_score is not None:
            per_feature_scores["promise_clause"] = round(
                min(1.0, max(0.0, promise_clause_score)) * 100, 1
            )
            weighted_pairs.append((per_feature_scores["promise_clause"], self.w_promise_clause))

        total_weight = sum(weight for _score, weight in weighted_pairs)
        composite_score = (
            sum(score * weight for score, weight in weighted_pairs) / total_weight
            if total_weight > 0
            else 0.0
        )
        composite_score = round(min(100.0, max(0.0, composite_score)), 1)

        verdict = self._score_to_verdict(composite_score)

        # --- Hard trigger: confirmed numeral mismatch overrides the
        # --- weighted result regardless of how high other signals scored.
        score_overridden = False
        if numeral_mismatch_confirmed and composite_score > self.numeral_mismatch_score_cap:
            composite_score = self.numeral_mismatch_score_cap
            score_overridden = True
        if numeral_mismatch_confirmed and verdict != VERDICT_SUSPICIOUS:
            verdict = VERDICT_SUSPICIOUS
            score_overridden = True

        explanations = self._build_explanations(
            per_feature_scores, composite_score, verdict,
            numeral_match_state=numeral_match_state,
            numeral_mismatch_confirmed=numeral_mismatch_confirmed,
            score_overridden=score_overridden,
        )

        logger.info(
            "Decision: verdict=%s overall_score=%.1f features=%s score_overridden=%s",
            verdict, composite_score, per_feature_scores, score_overridden,
        )

        return {
            "verdict": verdict,
            "overall_score": composite_score,
            "per_feature_scores": per_feature_scores,
            "explanations": explanations,
            "score_overridden": score_overridden,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _score_to_verdict(self, composite_score: float) -> str:
        if composite_score >= self.genuine_threshold:
            return VERDICT_LIKELY_GENUINE
        if composite_score >= self.suspicious_threshold:
            return VERDICT_UNCLEAR
        return VERDICT_SUSPICIOUS

    def _build_explanations(
        self,
        features: Dict[str, float],
        composite_score: float,
        verdict: str,
        numeral_match_state: Optional[str],
        numeral_mismatch_confirmed: bool,
        score_overridden: bool,
    ) -> List[str]:
        explanations: List[str] = []

        # Standard continuous-score bands apply to every feature EXCEPT
        # numeral_consistency, which gets its own explicit wording below —
        # its "inconclusive" placeholder score (settings.NUMERAL_INCONCLUSIVE_SCORE,
        # e.g. 0.70 -> 70.0) would otherwise fall in the same high_band as
        # a genuine "match" and print misleading "matches with high
        # confidence" text for what is actually an unread numeral.
        templates = {
            "image_quality": (
                "Image quality is strong ({score:.1f}/100), supporting reliable analysis.",
                "Image quality is adequate ({score:.1f}/100) but not ideal.",
                "Image quality is weak ({score:.1f}/100), which reduces confidence in every "
                "downstream check.",
            ),
            "denomination_confidence": (
                "Denomination was identified with high confidence ({score:.1f}/100).",
                "Denomination was identified with moderate confidence ({score:.1f}/100).",
                "Denomination could not be confidently identified ({score:.1f}/100); the "
                "ROI positions used for other checks may be uncalibrated for this note.",
            ),
            "serial_quality": (
                "Serial number passed validation checks with a high quality score "
                "({score:.1f}/100).",
                "Serial number partially passed validation checks ({score:.1f}/100).",
                "Serial number failed validation checks ({score:.1f}/100).",
            ),
            "microprint_clarity": (
                "Microprint region shows strong clarity ({score:.1f}/100), consistent with "
                "genuine fine-detail printing.",
                "Microprint region clarity is borderline ({score:.1f}/100).",
                "Microprint region shows low clarity ({score:.1f}/100), consistent with "
                "blurred or low-quality reproduction.",
            ),
            "security_thread": (
                "Security thread signature is strong ({score:.1f}/100) — a continuous, "
                "high-contrast thread-like feature was detected.",
                "Security thread signature is borderline ({score:.1f}/100).",
                "Security thread signature is weak ({score:.1f}/100) — no clear continuous "
                "thread-like feature was detected.",
            ),
            "promise_clause": (
                "Promise-to-pay legal text was clearly detected ({score:.1f}/100 keyword "
                "match).",
                "Promise-to-pay legal text was partially detected ({score:.1f}/100 keyword "
                "match).",
                "Promise-to-pay legal text was largely undetected ({score:.1f}/100 keyword "
                "match).",
            ),
        }

        for feature_name, score in features.items():
            if feature_name == "numeral_consistency":
                continue  # handled explicitly below
            high_text, mid_text, low_text = templates[feature_name]
            if score >= self.high_band:
                explanations.append(high_text.format(score=score))
            elif score >= self.low_band:
                explanations.append(mid_text.format(score=score))
            else:
                explanations.append(low_text.format(score=score))

        if "numeral_consistency" in features:
            score = features["numeral_consistency"]
            if numeral_match_state == NUMERAL_STATE_MATCH:
                explanations.append(
                    f"Printed denomination numeral matches the classified denomination "
                    f"({score:.1f}/100)."
                )
            elif numeral_match_state == NUMERAL_STATE_MISMATCH:
                explanations.append(
                    f"Printed denomination numeral does NOT match the classified "
                    f"denomination ({score:.1f}/100) — a strong indicator of alteration "
                    f"or misprint."
                )
            else:  # inconclusive
                explanations.append(
                    f"Denomination numeral could not be read clearly ({score:.1f}/100); "
                    f"treated as inconclusive rather than suspicious."
                )

        explanations.append(f"Composite weighted score: {composite_score:.1f}/100.")

        if score_overridden:
            explanations.append(
                f"OVERRIDE APPLIED: a confirmed denomination-numeral mismatch capped the "
                f"score at {self.numeral_mismatch_score_cap:.0f}/100 and forced the verdict "
                f"to '{VERDICT_SUSPICIOUS}', regardless of the weighted composite result."
            )
        elif verdict == VERDICT_LIKELY_GENUINE:
            explanations.append(
                f"Score meets the 'likely genuine' threshold (>= {self.genuine_threshold:.0f}/100)."
            )
        elif verdict == VERDICT_SUSPICIOUS:
            explanations.append(
                f"Score falls below the 'suspicious' threshold (< {self.suspicious_threshold:.0f}/100)."
            )
        else:
            explanations.append(
                f"Score falls between the suspicious and genuine thresholds "
                f"[{self.suspicious_threshold:.0f}, {self.genuine_threshold:.0f}) — flagged "
                f"'unclear' for manual review rather than a confident call either way."
            )

        return explanations

    # ------------------------------------------------------------------
    # Image quality scoring helper (used by pipeline_service before calling decide())
    # ------------------------------------------------------------------
    @staticmethod
    def compute_image_quality_score(sharpness_score: float, brightness_score: float) -> float:
        """
        Converts the pass/fail quality gate metrics (already validated by
        PreprocessingService) into a continuous [0,1] "quality headroom"
        score: images that just barely cleared the minimum thresholds
        score low; images comfortably above them score high.
        """
        sharpness_ceiling = (
            settings.BLUR_LAPLACIAN_VAR_THRESHOLD
            * settings.IMAGE_QUALITY_SHARPNESS_HEADROOM_MULTIPLIER
        )
        sharpness_ratio = min(1.0, sharpness_score / sharpness_ceiling) if sharpness_ceiling > 0 else 0.0

        ideal_center = (settings.BRIGHTNESS_MIN_THRESHOLD + settings.BRIGHTNESS_MAX_THRESHOLD) / 2
        half_range = (settings.BRIGHTNESS_MAX_THRESHOLD - settings.BRIGHTNESS_MIN_THRESHOLD) / 2
        brightness_deviation = abs(brightness_score - ideal_center)
        brightness_ratio = (
            1.0 - min(1.0, brightness_deviation / half_range) if half_range > 0 else 0.0
        )

        return round(min(1.0, max(0.0, 0.5 * sharpness_ratio + 0.5 * brightness_ratio)), 3)
