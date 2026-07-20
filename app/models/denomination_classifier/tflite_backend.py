"""
ML backend: trained denomination classifier exported to TensorFlow Lite.

NOT ACTIVE in this prototype and NOT registered in DenominationService by
default — provided as a template for the mobile/POS/counting-machine
deployment target mentioned in the project brief, where TFLite is often
preferred over ONNX for on-device inference (smaller runtime, better
mobile framework support).

To activate:
1. Train/export a classifier to .tflite (e.g. via TensorFlow's
   `tf.lite.TFLiteConverter`).
2. Place the file at app/models/weights/denomination_classifier.tflite.
3. Add a DENOMINATION_MODEL_PATH_TFLITE setting, or repurpose the existing
   DENOMINATION_MODEL_PATH.
4. In DenominationService.__init__, swap/add:
       TFLiteDenominationBackend(settings.DENOMINATION_MODEL_PATH_TFLITE)
   into the `self.backends` list — no other file needs to change, since
   this class implements the same DenominationClassifierBackend interface
   as OnnxDenominationBackend.
5. `pip install tflite-runtime` (or full tensorflow) on the target device.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config.logging_config import get_logger
from app.models.denomination_classifier.base import DenominationClassifierBackend
from app.models.denomination_classifier.labels import DENOMINATION_LABELS

logger = get_logger(__name__)


class TFLiteDenominationBackend(DenominationClassifierBackend):
    name = "ml_classifier_tflite"

    INPUT_SIZE: Tuple[int, int] = (224, 224)  # TODO: match training config

    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self._interpreter = None
        self._input_details = None
        self._output_details = None

    def load(self) -> None:
        if not self.weights_path.exists():
            logger.info(
                "TFLite denomination classifier weights not found at %s — "
                "backend will remain inactive.",
                self.weights_path,
            )
            return
        try:
            try:
                import tflite_runtime.interpreter as tflite  # lightweight, preferred on-device
            except ImportError:
                import tensorflow.lite as tflite  # fallback if full TF is installed

            self._interpreter = tflite.Interpreter(model_path=str(self.weights_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info("TFLite denomination classifier loaded from %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load TFLite denomination classifier: %s", exc)
            self._interpreter = None

    def is_available(self) -> bool:
        return self._interpreter is not None

    def predict(self, aligned_note_image: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.is_available():
            return None

        try:
            input_tensor = self._preprocess(aligned_note_image)
            self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
            self._interpreter.invoke()
            probabilities = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

            class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[class_idx])
            label = DENOMINATION_LABELS.get(class_idx, "UNKNOWN")
            return label, confidence
        except Exception as exc:  # noqa: BLE001
            logger.error("TFLite denomination inference failed: %s", exc)
            return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """TODO: must exactly mirror the training-time preprocessing pipeline."""
        resized = cv2.resize(image, self.INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched
