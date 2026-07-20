"""
End-to-end smoke test for PipelineService.
Requires a real sample image at data/sample_uploads/sample_note.jpg to run
meaningfully; otherwise it is skipped.
"""

from pathlib import Path

import pytest

from app.services.pipeline_service import PipelineService

SAMPLE_PATH = Path("data/sample_uploads/sample_note.jpg")


def test_pipeline_runs_end_to_end_without_crashing():
    if not SAMPLE_PATH.exists():
        pytest.skip(f"No sample image found at {SAMPLE_PATH}")

    image_bytes = SAMPLE_PATH.read_bytes()

    pipeline = PipelineService()
    result = pipeline.run(image_bytes, SAMPLE_PATH.name)

    assert result.verdict in ("likely genuine", "suspicious", "unclear")
    assert 0.0 <= result.overall_score <= 1.0
