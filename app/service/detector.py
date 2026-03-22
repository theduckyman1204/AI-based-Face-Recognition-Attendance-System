from typing import List

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


class FaceDetector:
    """Lightweight wrapper around YOLO for face detection."""

    def __init__(self, weights_path: str, confidence: float = 0.4):
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for detection. Install with `pip install ultralytics`."
            )
        self.model = YOLO(weights_path)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> List[dict]:
        """Run detection and return bounding boxes with confidences."""
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        detections: List[dict] = []
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            for box, score in zip(res.boxes.xyxy, res.boxes.conf):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(score),
                    }
                )
        return detections

    @staticmethod
    def crop_face(image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop a detected face region; clamps coords to image bounds."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        return image[y1:y2, x1:x2]
