"""
Configuration for the two new text-verification checks:
    1. denomination_numeral — OCR the large printed numeral (e.g. "500")
       and cross-check it against DenominationService's predicted value.
    2. promise_clause       — OCR the small boilerplate legal text (the
       "I promise to pay the bearer..." clause, Governor signature line,
       etc.) and check for expected keywords.

HEURISTIC (v1): both are keyword/substring checks, not a learned text
authenticity model. The promise-clause wording is standard boilerplate
across all Indian currency notes, so the expected keyword list is
denomination-agnostic — it does not need a per-denomination config.
"""

from typing import List

# Case-insensitive keywords expected somewhere in the promise-clause OCR
# text. This wording is the same across all denominations, so this list
# is intentionally NOT denomination-specific.
EXPECTED_PROMISE_CLAUSE_KEYWORDS: List[str] = [
    "PROMISE",
    "PAY",
    "BEARER",
    "GOVERNOR",
]

# Minimum fraction of the keyword list that must be found for the clause
# to be considered "present" (tolerant of partial OCR misreads — this is
# NOT a hard pass/fail gate on its own, just a signal).
PROMISE_CLAUSE_MIN_KEYWORD_MATCH_RATIO: float = 0.5
