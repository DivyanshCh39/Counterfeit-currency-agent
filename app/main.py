"""
FastAPI application entrypoint.
Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open the demo UI at:
    http://localhost:8000/ui
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config.logging_config import configure_logging, get_logger
from app.config.settings import settings
from app.routers import analyze_router, health_router, upload_router

configure_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s v%s starting up.", settings.APP_NAME, settings.APP_VERSION)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Prototype API for flagging currency note images as "
        "'likely genuine', 'suspicious', or 'unclear'. "
        "ACADEMIC / DEMONSTRATION USE ONLY — not for legal-grade authentication."
    ),
    lifespan=lifespan,
)

# CORS — permissive for local prototype use across web/mobile clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Storage locations ---
# Original uploaded files (POST /upload):      settings.UPLOAD_STORAGE_DIR
# Annotated output images (POST /analyze[...]): settings.SAMPLE_UPLOADS_DIR / "results"
settings.UPLOAD_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(settings.SAMPLE_UPLOADS_DIR / "results").mkdir(parents=True, exist_ok=True)

# Serve annotated result images so the web UI / API clients can load them
# directly via <img src="/static/results/<id>.jpg">.
app.mount(
    "/static/results",
    StaticFiles(directory=str(settings.SAMPLE_UPLOADS_DIR / "results")),
    name="results",
)

app.include_router(health_router.router)
app.include_router(upload_router.router)
app.include_router(analyze_router.router)

_WEB_UI_INDEX = Path(__file__).resolve().parent.parent / "ui" / "web" / "index.html"


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": settings.APP_NAME,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
    }


@app.get("/ui", include_in_schema=False)
def serve_web_ui():
    """Minimal HTML/JS demo frontend — see ui/web/index.html."""
    return FileResponse(str(_WEB_UI_INDEX))