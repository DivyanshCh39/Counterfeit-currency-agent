"""
Static constants used across the pipeline.

Note: denomination-aware ROI coordinates used to live here but have moved
to app/config/roi_config.py, since they are tunable configuration rather
than fixed constants.
"""

# Supported denominations for this prototype (extend as needed)
SUPPORTED_DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000"]

# Returned by DenominationService when no backend produces a
# sufficiently confident prediction.
UNKNOWN_DENOMINATION = "unknown"

VERDICT_LIKELY_GENUINE = "likely genuine"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_UNCLEAR = "unclear"
