"""
Pydantic models for incoming request data.
Note: the image itself arrives as multipart/form-data (UploadFile) in the
router, not as JSON — this schema covers accompanying request metadata.
"""

from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequestMeta(BaseModel):
    """Optional metadata that can accompany an image upload."""

    declared_denomination: Optional[str] = Field(
        default=None,
        description="If the client already knows the denomination, "
        "it can be passed as a hint to skip/validate classification.",
    )
    device_source: Optional[str] = Field(
        default="unknown",
        description="Origin of the capture, e.g. 'mobile', 'pos_terminal', "
        "'counting_machine', 'web_ui'.",
    )
