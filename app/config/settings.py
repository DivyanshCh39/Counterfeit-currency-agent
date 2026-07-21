"""
Central application settings.
All tunable thresholds live here so heuristic logic can be adjusted
without touching service code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # --- App metadata ---
    APP_NAME: str = "Counterfeit Currency Identification Agent"
    APP_VERSION: str = "0.1.0-prototype"
    DEBUG: bool = True

    # --- Paths ---
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    REFERENCE_NOTES_DIR: Path = DATA_DIR / "reference_notes"
    SAMPLE_UPLOADS_DIR: Path = DATA_DIR / "sample_uploads"
    WEIGHTS_DIR: Path = BASE_DIR / "app" / "models" / "weights"

    # --- Upload constraints ---
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png")
    UPLOAD_STORAGE_DIR: Path = DATA_DIR / "sample_uploads" / "incoming"

    # --- Image quality thresholds (HEURISTIC - tune during testing) ---
    BLUR_LAPLACIAN_VAR_THRESHOLD: float = 100.0  # below this -> "unclear"
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 200

    # --- Brightness thresholds (HEURISTIC - mean grayscale pixel intensity 0-255) ---
    BRIGHTNESS_MIN_THRESHOLD: float = 40.0   # below this -> "too dark"
    BRIGHTNESS_MAX_THRESHOLD: float = 235.0  # above this -> "overexposed"

    # --- Preprocessing ---
    PREPROCESS_RESIZE_WIDTH: int = 800  # working width used pre-detection

    # --- Debug output (intermediate preprocessing steps) ---
    DEBUG_SAVE_INTERMEDIATE: bool = True
    DEBUG_OUTPUT_DIR: Path = DATA_DIR / "sample_uploads" / "debug"

    # --- Detection thresholds (HEURISTIC) ---
    MIN_NOTE_CONTOUR_AREA_RATIO: float = 0.15  # note area / image area

    # --- ROI extraction ---
    ROI_SAVE_CROPS: bool = True
    ROI_OUTPUT_DIR: Path = DATA_DIR / "sample_uploads" / "roi_crops"

    # --- Denomination classifier ---
    DENOMINATION_MODEL_PATH: Path = WEIGHTS_DIR / "denomination_classifier.onnx"

    # --- Note-boundary segmentation (optional — see app/models/note_boundary/) ---
    # Same activation pattern as DENOMINATION_MODEL_PATH: absent by default,
    # the heuristic OpenCV contour detector (app/utils/geometry_utils.find_largest_quadrilateral)
    # remains fully active either way. Dropping a real .onnx file here
    # activates NoteBoundaryOnnxBackend automatically — no code changes.
    NOTE_BOUNDARY_MODEL_PATH: Path = WEIGHTS_DIR / "note_boundary_segmenter.onnx"
    # Probability above which a pixel is considered part of the note, when
    # binarizing the segmentation model's raw sigmoid output.
    NOTE_BOUNDARY_MASK_THRESHOLD: float = 0.5
    DENOMINATION_CONFIDENCE_THRESHOLD: float = 0.6

    # --- OCR ---
    OCR_ENGINE: str = "easyocr"  # options: "easyocr", "tesseract"
    # Serial number format patterns now live in app/config/serial_config.py
    # (denomination-aware, with a DEFAULT fallback) instead of a single
    # global regex here.

    # --- Microprint scoring (HEURISTIC — see app/services/microprint_service.py) ---
    # Sub-feature thresholds: each raw metric is normalized to [0,1] by
    # dividing by its threshold (capped at 1.0), then combined by weight.
    MICROPRINT_SHARPNESS_THRESHOLD: float = 150.0        # Laplacian variance
    MICROPRINT_EDGE_DENSITY_THRESHOLD: float = 0.08       # fraction of edge pixels (Canny)
    MICROPRINT_FREQUENCY_DETAIL_THRESHOLD: float = 0.12   # high-frequency FFT energy ratio
    MICROPRINT_PATCH_TEXTURE_THRESHOLD: float = 10.0      # mean local-patch intensity std dev

    MICROPRINT_WEIGHT_SHARPNESS: float = 0.30
    MICROPRINT_WEIGHT_EDGE_DENSITY: float = 0.25
    MICROPRINT_WEIGHT_FREQUENCY_DETAIL: float = 0.25
    MICROPRINT_WEIGHT_PATCH_TEXTURE: float = 0.20
    MICROPRINT_PASS_THRESHOLD: float = 0.5  # composite clarity_score >= this -> passed

    # --- Security thread scoring (HEURISTIC — see app/services/security_thread_service.py) ---
    SECURITY_THREAD_CONTRAST_THRESHOLD: float = 18.0        # grayscale intensity std dev
    SECURITY_THREAD_VERTICAL_PEAK_THRESHOLD: float = 3.0     # Sobel column-energy peak-to-mean ratio
    SECURITY_THREAD_BAND_CONTINUITY_THRESHOLD: float = 0.4   # fraction of rows with a detected edge

    SECURITY_THREAD_WEIGHT_CONTRAST: float = 0.30
    SECURITY_THREAD_WEIGHT_VERTICAL_FEATURE: float = 0.35
    SECURITY_THREAD_WEIGHT_BAND_CONTINUITY: float = 0.35
    SECURITY_THREAD_PRESENT_THRESHOLD: float = 0.45  # composite thread_score >= this -> present

    # --- Decision engine (see app/services/decision_service.py) ---
    # Weights (HEURISTIC) applied to each per-feature score that feeds the
    # composite verdict score. The original 5 weights below still sum to
    # 1.0 on their own (unchanged from v1) — WEIGHT_NUMERAL_CONSISTENCY and
    # WEIGHT_PROMISE_CLAUSE are additional weight mass that only applies
    # when those two optional checks actually ran for a given denomination
    # (see roi_config.py OPTIONAL_REGIONS). DecisionService renormalizes by
    # the sum of whichever weights are actually active for a given note, so
    # a denomination lacking these two ROIs behaves exactly as before.
    WEIGHT_IMAGE_QUALITY: float = 0.10
    WEIGHT_DENOMINATION_CONFIDENCE: float = 0.10
    WEIGHT_SERIAL_CHECK: float = 0.25
    WEIGHT_MICROPRINT_CHECK: float = 0.30
    WEIGHT_SECURITY_THREAD_CHECK: float = 0.25

    # Denomination-numeral cross-check gets the larger of the two new
    # weights: a confirmed mismatch between the OCR'd printed numeral and
    # the classified denomination is a strong, specific forgery/misprint
    # signal (see NUMERAL_MISMATCH_SCORE_CAP below for the hard-trigger
    # half of this — the weight alone is not considered sufficient for
    # a confirmed mismatch).
    WEIGHT_NUMERAL_CONSISTENCY: float = 0.20
    # Promise-clause text presence is a moderate, softer signal — missing
    # or garbled legal-tender text is suspicious but far more sensitive to
    # ordinary OCR/lighting failure than a numeral mismatch is, so it stays
    # weighted-only (no hard trigger).
    WEIGHT_PROMISE_CLAUSE: float = 0.10

    # Score used for numeral_consistency when OCR found no readable numeral
    # at all (inconclusive — NOT the same as a confirmed mismatch). Kept
    # deliberately neutral/mildly favorable rather than punitive, since a
    # missing read is usually a photo-angle/lighting problem, not evidence
    # of anything. On the same [0,1] scale as every other raw sub-score.
    NUMERAL_INCONCLUSIVE_SCORE: float = 0.70

    # Hard-trigger: when the printed numeral is read AND definitively does
    # NOT match the classified denomination, overall_score is capped at
    # this value (0-100 scale) and the verdict is forced to "suspicious"
    # regardless of how high every other signal scored. Set below
    # SUSPICIOUS_SCORE_THRESHOLD so the displayed score and verdict never
    # visibly contradict each other after the override.
    NUMERAL_MISMATCH_SCORE_CAP: float = 40.0

    # Multiplier used to rescale raw sharpness (which has no natural upper
    # bound) into a [0,1] "headroom above the minimum" score: a sharpness
    # of (threshold * multiplier) or higher maps to a full 1.0.
    IMAGE_QUALITY_SHARPNESS_HEADROOM_MULTIPLIER: float = 2.0

    # Score bands used only for generating human-readable explanations
    # (do NOT affect the composite score itself). On the same 0-100 scale
    # as the final output (see DecisionService).
    DECISION_FEATURE_HIGH_THRESHOLD: float = 70.0
    DECISION_FEATURE_LOW_THRESHOLD: float = 40.0

    # --- Verdict thresholds ---
    # NOTE: overall_score and feature_scores are reported to the API/UI on
    # a 0-100 scale (see DecisionService.decide()), so these thresholds are
    # on that same scale.
    GENUINE_SCORE_THRESHOLD: float = 75.0
    SUSPICIOUS_SCORE_THRESHOLD: float = 45.0
    # score below SUSPICIOUS_SCORE_THRESHOLD but image quality was fine -> "suspicious"
    # score above GENUINE_SCORE_THRESHOLD -> "likely genuine"
    # anything else, or failed quality gate -> "unclear"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()