"""
Upload endpoint.

Stores a raw note image on disk and returns a file_id that can later be
passed to POST /analyze/{file_id}. This decouples "receive the photo"
from "run the (potentially slower) analysis pipeline" — useful for
integrations like counting machines or POS terminals that may want to
capture first and trigger analysis separately/asynchronously.

For the simple web UI (and most one-shot use cases), POST /analyze
directly with a file is simpler and remains fully supported — this
endpoint is an addition, not a replacement.

Storage location: settings.UPLOAD_STORAGE_DIR
    (default: data/sample_uploads/incoming/)
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.schemas.response_schemas import ErrorResponse, UploadResponse
from app.utils.file_utils import generate_unique_filename, is_allowed_extension, save_bytes

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_note_image(file: UploadFile = File(...)):
    if not is_allowed_extension(file.filename, settings.ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: "
            f"{settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    extension = "." + file.filename.rsplit(".", 1)[-1].lower()
    file_id = generate_unique_filename(extension)
    stored_path = save_bytes(file_bytes, settings.UPLOAD_STORAGE_DIR, file_id)

    logger.info("Stored upload '%s' as file_id=%s", file.filename, file_id)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=len(file_bytes),
        content_type=file.content_type,
        stored_path=str(stored_path),
    )
