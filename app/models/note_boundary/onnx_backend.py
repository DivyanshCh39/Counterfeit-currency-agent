"""
ML backend: trained note-boundary SEGMENTATION model exported to ONNX.

INACTIVE BY DEFAULT in this prototype — no trained weights exist yet.
is_available() returns False until a real .onnx file is placed at
settings.NOTE_BOUNDARY_MODEL_PATH, at which point NoteBoundaryService
will automatically prefer this backend over the OpenCV heuristic one
(see NoteBoundaryService backend ordering) — exactly the same activation
pattern as app/models/denomination_classifier/onnx_backend.py.

WHY A SEGMENTATION MODEL (not a 4-point-regression or plain bbox model):
the Roboflow export's annotations are majority variable-length polygon
outlines of the note (see project audit — ~58% of label lines are
polygons, not 5-value boxes). A segmentation mask is the natural,
lossless training target those polygons rasterize into directly (see
training/prepare_segmentation_data.py), and a segmentation model degrades
more gracefully than a direct corner-point regressor when the note is
partially occluded (e.g. a thumb covering one corner) — it can still
mark the visible note pixels even if one corner is genuinely obscured,
whereas a 4-point regressor has no visible ground truth for a hidden
corner to learn from. The mask -> quad bridge (app/utils/geometry_utils.mask_to_quad)
is a deterministic, unlearned step, so this backend's OWN output contract
is identical to every other backend's: a (4, 2) corner-point array, or None.

SCOPE: this model localizes the note boundary ONLY — it is not trained on,
and must never be extended toward, genuine/counterfeit labels (there are
none in this dataset; see project audit). It contributes nothing to
DecisionService; it only feeds four_point_warp(), exactly like the
heuristic backend it sits alongside.

To activate:
1. Run training/prepare_segmentation_data.py to rasterize the Roboflow
   polygon labels into binary masks.
2. Run training/train_note_boundary_model.py to train and export ONNX.
3. That's it — the .onnx file lands at settings.NOTE_BOUNDARY_MODEL_PATH
   by default; restart the API and this backend activates automatically.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.models.note_boundary.base import NoteBoundaryBackend
from app.utils.geometry_utils import mask_to_quad

logger = get_logger(__name__)


class NoteBoundaryOnnxBackend(NoteBoundaryBackend):
    name = "ml_segmenter_onnx"

    INPUT_SIZE: Tuple[int, int] = (256, 256)  # TODO: match training config

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self._session = None

    def load(self) -> None:
        if not self.weights_path.exists():
            logger.info(
                "Note-boundary ONNX segmentation model not found at %s — "
                "backend will remain inactive (heuristic contour detector "
                "stays in charge of alignment).",
                self.weights_path,
            )
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.weights_path))
            logger.info("Note-boundary ONNX segmentation model loaded from %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load note-boundary ONNX model: %s", exc)
            self._session = None

    def is_available(self) -> bool:
        return self._session is not None

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_available():
            return None

        try:
            original_h, original_w = image.shape[:2]
            input_tensor = self._preprocess(image)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            # Expected output: (1, 1, H, W) single-channel sigmoid probability
            # map at INPUT_SIZE resolution — see train_note_boundary_model.py.
            prob_mask = outputs[0][0, 0]
            full_res_mask = cv2.resize(
                prob_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (full_res_mask >= settings.NOTE_BOUNDARY_MASK_THRESHOLD).astype("uint8")

            return mask_to_quad(
                binary_mask, min_area_ratio=settings.MIN_NOTE_CONTOUR_AREA_RATIO
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Note-boundary ONNX inference failed: %s", exc)
            return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Must exactly mirror the training-time preprocessing pipeline — see
        training/train_note_boundary_model.py. Same NCHW / RGB / [0,1]
        convention as app/models/denomination_classifier/onnx_backend.py,
        for consistency across every ONNX backend in this project.
        """
        resized = cv2.resize(image, self.INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(chw, axis=0)  # CHW -> NCHW
        return batched
