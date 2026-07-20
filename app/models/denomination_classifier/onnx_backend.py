"""
ML backend: trained denomination classifier exported to ONNX.

INACTIVE BY DEFAULT in this prototype — no trained weights exist yet.
is_available() returns False until a real .onnx file is placed at
settings.DENOMINATION_MODEL_PATH, at which point DenominationService
will automatically prefer this backend over the heuristic one (see
DenominationService backend ordering).

To activate:
1. Train a small image classifier (transfer learning on e.g. MobileNetV2/
   EfficientNet-Lite is plenty for a 7-class denomination problem) on
   labeled genuine-note images, one class per denomination.
2. Export to ONNX (`torch.onnx.export` or `tf2onnx`, depending on framework).
3. Place the .onnx file at app/models/weights/denomination_classifier.onnx
   (or update settings.DENOMINATION_MODEL_PATH).
4. Update _preprocess() below to match the exact preprocessing used during
   training (input size, channel order, normalization).
5. Update the input/output tensor names in predict() to match your export.
No other file in the project needs to change.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.models.denomination_classifier.base import DenominationClassifierBackend
from app.models.denomination_classifier.labels import DENOMINATION_LABELS

logger = get_logger(__name__)


class OnnxDenominationBackend(DenominationClassifierBackend):
    name = "ml_classifier_onnx"

    INPUT_SIZE: Tuple[int, int] = (224, 224)  # TODO: match training config

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self._session = None

    def load(self) -> None:
        if not self.weights_path.exists():
            logger.info(
                "ONNX denomination classifier weights not found at %s — "
                "backend will remain inactive (prototype falls back to heuristic).",
                self.weights_path,
            )
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.weights_path))
            logger.info("ONNX denomination classifier loaded from %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load ONNX denomination classifier: %s", exc)
            self._session = None

    def is_available(self) -> bool:
        return self._session is not None

    def predict(self, aligned_note_image: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.is_available():
            return None

        try:
            input_tensor = self._preprocess(aligned_note_image)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})
            probabilities = outputs[0][0]

            class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[class_idx])
            label = DENOMINATION_LABELS.get(class_idx, "UNKNOWN")
            return label, confidence
        except Exception as exc:  # noqa: BLE001
            logger.error("ONNX denomination inference failed: %s", exc)
            return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Must exactly mirror the training-time preprocessing pipeline —
        see training/train_denomination_classifier.py.

        Output layout is NCHW (batch, channels, height, width), matching
        the convention PyTorch/torchvision models use when exported to
        ONNX. If you train with a different framework (e.g. TensorFlow,
        which conventionally exports NHWC), transpose accordingly here.
        """
        resized = cv2.resize(image, self.INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(chw, axis=0)  # CHW -> NCHW
        return batched
