"""
Denomination-aware serial number format configuration.

Mirrors the design of app/config/roi_config.py: SerialConsistencyService
never hardcodes a regex — it always asks this module, so adding a real
per-denomination serial format later is a config-only change.

HEURISTIC / PLACEHOLDER: patterns below are illustrative, not the real
issuing-authority serial number grammar. Replace with actual formats
before relying on this for anything beyond a demo.
"""

from typing import Dict, Tuple

DEFAULT_PATTERN_KEY = "DEFAULT"

# denomination -> regex matched against the CLEANED (A-Z0-9 only, uppercased) OCR text
SERIAL_PATTERNS: Dict[str, str] = {
    "500": r"^[A-Z]{2}[0-9]{6}$",
    "100": r"^[A-Z]{1}[0-9]{6}$",
    # DEFAULT fallback pattern used for any denomination without a
    # dedicated format (including "unknown" from DenominationService).
    DEFAULT_PATTERN_KEY: r"^[A-Z0-9]{6,10}$",
}

# denomination -> (min_length, max_length) of the CLEANED serial text.
# Checked independently of SERIAL_PATTERNS so a length failure can be
# reported as its own, more specific, suspicious reason.
SERIAL_LENGTH_RANGES: Dict[str, Tuple[int, int]] = {
    "500": (8, 8),
    "100": (7, 7),
    DEFAULT_PATTERN_KEY: (6, 10),
}

# Character set permitted in a cleaned serial number. Applied independently
# of the full pattern so "wrong charset" and "wrong structure" can be
# distinguished in suspicious_reasons.
ALLOWED_CHARACTERS_PATTERN = r"^[A-Z0-9]+$"

# Commonly OCR-confused character substitutions (letter -> digit it's
# frequently misread as, or vice versa). Used to attempt a second,
# lower-trust match when the raw OCR text fails the format check outright.
CHARACTER_CONFUSION_MAP: Dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
}


def get_serial_pattern(denomination: str) -> Tuple[str, bool]:
    """
    Returns (regex_pattern, is_denomination_specific).
    Falls back to the DEFAULT pattern if denomination has no dedicated entry.
    """
    if denomination in SERIAL_PATTERNS:
        return SERIAL_PATTERNS[denomination], True
    return SERIAL_PATTERNS[DEFAULT_PATTERN_KEY], False


def get_expected_length_range(denomination: str) -> Tuple[Tuple[int, int], bool]:
    """Returns ((min_len, max_len), is_denomination_specific)."""
    if denomination in SERIAL_LENGTH_RANGES:
        return SERIAL_LENGTH_RANGES[denomination], True
    return SERIAL_LENGTH_RANGES[DEFAULT_PATTERN_KEY], False


def register_serial_pattern(denomination: str, pattern: str) -> None:
    """Add/override a per-denomination serial format at runtime."""
    SERIAL_PATTERNS[denomination] = pattern
