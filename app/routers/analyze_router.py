"""
Analysis endpoints.

Two ways to trigger analysis:
    POST /analyze              — upload + analyze in one call (simplest,
                                   used by the web UI)
    POST /analyze/{file_id}    — analyze a file previously stored via
                                   POST /upload (see upload_router.py)

Response JSON includes: verdict, overall_score, feature_scores,
explanations, denomination, image_quality, detection, checks (serial
number + validation reasons, microprint, security thread), and both a
filesystem path and a web-accessible URL for the annotated output image.

NOTE: image loading/format validation is handled inside PipelineService ->
PreprocessingService (app/services/preprocessing_service.py).
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.core.exceptions import InvalidImageError
from app.schemas.response_schemas import AnalyzeResponse, ErrorResponse
from app.services.pipeline_service import PipelineService

logger = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analyze"])

# Single shared pipeline instance (stateless services, safe to reuse).
pipeline_service = PipelineService()


def _attach_annotated_image_url(result: AnalyzeResponse) -> AnalyzeResponse:
    """Converts the filesystem annotated_image_path into a URL the static
    mount in app/main.py can actually serve to a browser."""
    if result.annotated_image_path:
        result.annotated_image_url = f"/static/results/{Path(result.annotated_image_path).name}"
    return result


@router.post(
    "",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze_currency_note(file: UploadFile = File(...)):
    """Upload and analyze a note image in a single request."""
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: "
            f"{settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    try:
        result = pipeline_service.run(file_bytes, file.filename or "upload.jpg")
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline failure")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc

    return _attach_annotated_image_url(result)


@router.post(
    "/{file_id}",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_uploaded_file(file_id: str):
    """Analyze a file previously stored via POST /upload."""
    upload_root = settings.UPLOAD_STORAGE_DIR.resolve()
    file_path = (settings.UPLOAD_STORAGE_DIR / file_id).resolve()

    # Prevent path traversal via a crafted file_id.
    if upload_root not in file_path.parents:
        raise HTTPException(status_code=400, detail="Invalid file_id.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"No uploaded file found for file_id '{file_id}'."
        )

    file_bytes = file_path.read_bytes()

    try:
        result = pipeline_service.run(file_bytes, file_path.name)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline failure")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc

    return _attach_annotated_image_url(result)
