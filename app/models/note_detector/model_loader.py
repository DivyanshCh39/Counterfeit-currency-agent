"""
PLACEHOLDER for a trained note detector (e.g., YOLO/ONNX).

In this prototype, note detection is done purely heuristically via
app/utils/geometry_utils.find_largest_quadrilateral() (called from
app/services/preprocessing_service.py), so this loader currently returns
None and is not wired into the pipeline.

To upgrade later:
1. Train/export a YOLOv8 or similar detector to ONNX.
2. Place weights in app/models/weights/note_detector.onnx
3. Implement load_model() + predict() below.
4. Swap PreprocessingService.detect_note_contour() to call this instead
   of the heuristic (see app/services/preprocessing_service.py).
"""

from typing import Optional

from app.config.logging_config import get_logger

logger = get_logger(__name__)


class NoteDetectorModel:
    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self.model = None

    def load(self) -> None:
        if not self.weights_path:
            logger.warning(
                "NoteDetectorModel.load() called with no weights_path — "
                "prototype relies on heuristic detection instead."
            )
            return
        # TODO: load ONNX/TorchScript model here
        raise NotImplementedError("Trained note detector not yet integrated.")

    def predict(self, image):
        if self.model is None:
            raise NotImplementedError(
                "No trained note detector loaded. Use PreprocessingService's "
                "heuristic detection instead."
            )
        # TODO: run inference and return bounding box + confidence
        raise NotImplementedError
