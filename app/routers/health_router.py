"""
Health check endpoint — useful for deployment readiness probes
(mobile backend, POS terminal integration, counting machine service checks).
"""

from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health_check():
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
