"""
Tests for DecisionService (app/services/decision_service.py).
"""

import pytest

from app.config.settings import settings
from app.core.constants import VERDICT_LIKELY_GENUINE, VERDICT_SUSPICIOUS, VERDICT_UNCLEAR
from app.services.decision_service import DecisionService


def test_decide_returns_all_expected_keys():
    service = DecisionService()
    result = service.decide(
        image_quality_score=0.8,
        denomination_confidence=0.8,
        serial_quality_score=0.8,
        microprint_clarity_score=0.8,
        thread_continuity_score=0.8,
    )

    assert set(result.keys()) == {
        "verdict", "overall_score", "per_feature_scores", "explanations", "score_overridden",
    }
    assert set(result["per_feature_scores"].keys()) == {
        "image_quality", "denomination_confidence", "serial_quality",
        "microprint_clarity", "security_thread",
    }
    assert len(result["explanations"]) > 0


def test_high_scores_across_the_board_yield_likely_genuine():
    service = DecisionService()
    result = service.decide(
        image_quality_score=0.9,
        denomination_confidence=0.9,
        serial_quality_score=0.9,
        microprint_clarity_score=0.9,
        thread_continuity_score=0.9,
    )
    assert result["verdict"] == VERDICT_LIKELY_GENUINE
    assert result["overall_score"] >= settings.GENUINE_SCORE_THRESHOLD


def test_low_scores_across_the_board_yield_suspicious():
    service = DecisionService()
    result = service.decide(
        image_quality_score=0.1,
        denomination_confidence=0.1,
        serial_quality_score=0.1,
        microprint_clarity_score=0.1,
        thread_continuity_score=0.1,
    )
    assert result["verdict"] == VERDICT_SUSPICIOUS
    assert result["overall_score"] < settings.SUSPICIOUS_SCORE_THRESHOLD


def test_mid_range_scores_yield_unclear():
    service = DecisionService()
    result = service.decide(
        image_quality_score=0.55,
        denomination_confidence=0.55,
        serial_quality_score=0.55,
        microprint_clarity_score=0.55,
        thread_continuity_score=0.55,
    )
    assert result["verdict"] == VERDICT_UNCLEAR


def test_overall_score_matches_manual_weighted_sum():
    service = DecisionService()
    inputs = dict(
        image_quality_score=0.6,
        denomination_confidence=0.7,
        serial_quality_score=0.8,
        microprint_clarity_score=0.5,
        thread_continuity_score=0.9,
    )
    result = service.decide(**inputs)

    expected = 100 * (
        inputs["image_quality_score"] * settings.WEIGHT_IMAGE_QUALITY
        + inputs["denomination_confidence"] * settings.WEIGHT_DENOMINATION_CONFIDENCE
        + inputs["serial_quality_score"] * settings.WEIGHT_SERIAL_CHECK
        + inputs["microprint_clarity_score"] * settings.WEIGHT_MICROPRINT_CHECK
        + inputs["thread_continuity_score"] * settings.WEIGHT_SECURITY_THREAD_CHECK
    )
    assert abs(result["overall_score"] - expected) < 0.5


def test_scores_are_clamped_to_valid_range():
    service = DecisionService()
    result = service.decide(
        image_quality_score=1.5,   # out of range, should clamp to 1.0 -> 100.0
        denomination_confidence=-0.5,  # out of range, should clamp to 0.0
        serial_quality_score=0.5,
        microprint_clarity_score=0.5,
        thread_continuity_score=0.5,
    )
    assert result["per_feature_scores"]["image_quality"] == 100.0
    assert result["per_feature_scores"]["denomination_confidence"] == 0.0


def test_compute_image_quality_score_rewards_headroom_above_minimums():
    barely_passing = DecisionService.compute_image_quality_score(
        sharpness_score=settings.BLUR_LAPLACIAN_VAR_THRESHOLD * 1.01,
        brightness_score=settings.BRIGHTNESS_MIN_THRESHOLD + 1,
    )
    comfortably_passing = DecisionService.compute_image_quality_score(
        sharpness_score=settings.BLUR_LAPLACIAN_VAR_THRESHOLD * 3,
        brightness_score=(settings.BRIGHTNESS_MIN_THRESHOLD + settings.BRIGHTNESS_MAX_THRESHOLD) / 2,
    )
    assert comfortably_passing > barely_passing
    assert 0.0 <= barely_passing <= 1.0
    assert 0.0 <= comfortably_passing <= 1.0


# ----------------------------------------------------------------------
# New in this revision: numeral_consistency / promise_clause hybrid logic
# ----------------------------------------------------------------------

def _high_base_inputs():
    return dict(
        image_quality_score=0.9,
        denomination_confidence=0.9,
        serial_quality_score=0.9,
        microprint_clarity_score=0.9,
        thread_continuity_score=0.9,
    )


def test_numeral_mismatch_forces_suspicious_even_with_high_scores_elsewhere():
    """Regression test for the exact bug from the audit: a confirmed
    numeral mismatch must override an otherwise-'likely genuine' result."""
    service = DecisionService()
    result = service.decide(
        **_high_base_inputs(),
        numeral_match_state="mismatch",
        promise_clause_score=0.9,
    )
    assert result["verdict"] == VERDICT_SUSPICIOUS
    assert result["overall_score"] <= settings.NUMERAL_MISMATCH_SCORE_CAP
    assert result["score_overridden"] is True
    assert any("OVERRIDE APPLIED" in e for e in result["explanations"])


def test_numeral_match_does_not_override_and_keeps_high_verdict():
    service = DecisionService()
    result = service.decide(
        **_high_base_inputs(),
        numeral_match_state="match",
        promise_clause_score=0.9,
    )
    assert result["verdict"] == VERDICT_LIKELY_GENUINE
    assert result["score_overridden"] is False
    assert result["per_feature_scores"]["numeral_consistency"] == 100.0


def test_numeral_inconclusive_is_not_punished_like_a_mismatch():
    service = DecisionService()
    result = service.decide(**_high_base_inputs(), numeral_match_state="inconclusive")
    assert result["score_overridden"] is False
    assert result["verdict"] == VERDICT_LIKELY_GENUINE
    assert result["per_feature_scores"]["numeral_consistency"] == round(
        settings.NUMERAL_INCONCLUSIVE_SCORE * 100, 1
    )


def test_numeral_and_promise_absent_reproduces_original_5_signal_formula():
    """When a denomination's ROI template has neither optional region,
    behavior must be byte-identical to the pre-existing 5-signal formula
    (this is the backward-compatibility guarantee for denominations
    without a denomination_numeral/promise_clause ROI)."""
    service = DecisionService()
    inputs = dict(
        image_quality_score=0.6, denomination_confidence=0.7,
        serial_quality_score=0.8, microprint_clarity_score=0.5,
        thread_continuity_score=0.9,
    )
    result = service.decide(**inputs)  # numeral_match_state=None, promise_clause_score=None

    expected = 100 * (
        inputs["image_quality_score"] * settings.WEIGHT_IMAGE_QUALITY
        + inputs["denomination_confidence"] * settings.WEIGHT_DENOMINATION_CONFIDENCE
        + inputs["serial_quality_score"] * settings.WEIGHT_SERIAL_CHECK
        + inputs["microprint_clarity_score"] * settings.WEIGHT_MICROPRINT_CHECK
        + inputs["thread_continuity_score"] * settings.WEIGHT_SECURITY_THREAD_CHECK
    )
    assert abs(result["overall_score"] - expected) < 0.5
    assert "numeral_consistency" not in result["per_feature_scores"]
    assert "promise_clause" not in result["per_feature_scores"]
    assert result["score_overridden"] is False


def test_weak_promise_clause_alone_does_not_hard_trigger_but_lowers_score():
    service = DecisionService()
    high = service.decide(**_high_base_inputs(), numeral_match_state="match", promise_clause_score=0.95)
    low = service.decide(**_high_base_inputs(), numeral_match_state="match", promise_clause_score=0.05)

    assert low["overall_score"] < high["overall_score"]
    assert low["score_overridden"] is False  # weighted-only signal, never hard-triggers


def test_invalid_numeral_match_state_raises():
    service = DecisionService()
    with pytest.raises(ValueError):
        service.decide(**_high_base_inputs(), numeral_match_state="not-a-real-state")
