from __future__ import annotations

from datetime import datetime
import random
from functools import lru_cache
from typing import List, Optional, Sequence

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.entities import Appearance, Person
from app.ropository.appearances import search_appearances, store_appearances
from app.ropository.persons import upsert_person
from app.service.detector import FaceDetector
from app.service.recognizer import ArcFaceRecognizer
from app.utils.timezone_utils import now_vn_naive


class FacePipeline:
    """Coordinates detection, recognition, and persistence."""

    def __init__(self):
        self.detector = FaceDetector(settings.yolo_weights, confidence=settings.detection_confidence)
        self.recognizer = ArcFaceRecognizer(
            model_path=settings.arcface_model_path,
            threshold=settings.recognition_threshold,
            fallback_model_path=settings.arcface_fallback_model_path,
        )

    def _decode_image(self, data: bytes) -> np.ndarray:
        frame = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image bytes")
        return image

    def analyze_image(self, data: bytes, source_name: Optional[str], db: Session) -> List[dict]:
        """Process a still image and persist the results."""
        image = self._decode_image(data)
        self.recognizer.rebuild_index(db.query(Person).all())
        detections = self.detector.detect(image)

        now = now_vn_naive()
        results: List[dict] = []
        for det in detections:
            face_crop = self.detector.crop_face(image, det["bbox"])
            name, score, embedding = self.recognizer.recognize(face_crop)
            results.append(
                {
                    "name": name,
                    "confidence": score,
                    "appeared_at": now,
                    "source_type": "image",
                    "source_name": source_name,
                    "bbox": det["bbox"],
                    "embedding": embedding,
                }
            )

        self._persist_results(db, results)
        return results

    def analyze_video(
        self,
        path: str,
        source_name: Optional[str],
        db: Session,
        sample_rate: int = 5,
    ) -> List[dict]:
        """
        Process a video file on disk.

        `sample_rate` defines the frame interval (in frames) to run detection on to keep the workload light.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        self.recognizer.rebuild_index(db.query(Person).all())

        candidate_frames: List[dict] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % sample_rate != 0:
                idx += 1
                continue

            detections = self.detector.detect(frame)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
            if not detections:
                idx += 1
                continue

            # Lưu lại frame có detection, ưu tiên bbox có confidence cao nhất
            best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
            candidate_frames.append(
                {
                    "frame": frame.copy(),
                    "bbox": best_det["bbox"],
                    "timestamp": timestamp,
                    "appeared_at": now_vn_naive(),
                }
            )
            idx += 1

        cap.release()

        if not candidate_frames:
            raise ValueError("Không tìm thấy khuôn mặt nào trong video.")

        sample_count = min(5, len(candidate_frames))
        sampled_frames = random.sample(candidate_frames, sample_count)

        recognitions: List[dict] = []
        for item in sampled_frames:
            face_crop = self.detector.crop_face(item["frame"], item["bbox"])
            name, score, embedding = self.recognizer.recognize(face_crop)
            recognitions.append(
                {
                    "name": name,
                    "confidence": score,
                    "appeared_at": item["appeared_at"],
                    "source_type": "video",
                    "source_name": source_name,
                    "bbox": item["bbox"],
                    "embedding": embedding,
                }
            )

        # Voting theo tên; tie-break bằng trung bình confidence cao hơn
        vote_map: dict[str, List[dict]] = {}
        for rec in recognitions:
            vote_map.setdefault(rec["name"], []).append(rec)

        def _vote_key(entry: tuple[str, List[dict]]) -> tuple[int, float]:
            _, items = entry
            mean_conf = float(np.mean([it["confidence"] for it in items])) if items else -1.0
            return len(items), mean_conf

        winning_name, winning_items = max(vote_map.items(), key=_vote_key)
        winning_conf = float(np.mean([it["confidence"] for it in winning_items]))
        best_sample = max(winning_items, key=lambda it: it["confidence"])

        results = [
            {
                "name": winning_name,
                "confidence": winning_conf,
                "appeared_at": best_sample.get("appeared_at"),
                "source_type": "video",
                "source_name": source_name,
                "bbox": best_sample.get("bbox"),
                "embedding": best_sample.get("embedding"),
            }
        ]

        self._persist_results(db, results)
        return results

    def search_between(self, db: Session, start_time: datetime, end_time: datetime) -> Sequence[Appearance]:
        return search_appearances(db, start_time, end_time)

    def _persist_results(self, db: Session, results: List[dict]) -> None:
        payload = []
        for res in results:
            payload.append(
                {
                    "name": res["name"],
                    "appeared_at": res.get("appeared_at"),
                }
            )
        if payload:
            store_appearances(db, payload)

    # New: enroll a face to persons table with embedding
    def enroll_person(self, data: bytes, name: str, db: Session) -> dict:
        image = self._decode_image(data)
        detections = self.detector.detect(image)
        if not detections:
            raise ValueError("Không tìm thấy khuôn mặt nào trong ảnh. Vui lòng thử lại.")

        # Chọn bbox có độ tin cậy cao nhất
        best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
        face_crop = self.detector.crop_face(image, best_det["bbox"])
        embedding = self.recognizer.embed(face_crop)
        embedding_text = ArcFaceRecognizer.serialize_embedding(embedding)

        person = upsert_person(db, name=name, embedding_text=embedding_text)
        return {
            "id": person.id,
            "name": person.name,
            "samples": 1,
        }

    def enroll_person_from_video(self, path: str, name: str, db: Session, sample_rate: int = 5) -> dict:
        """
        Đăng ký khuôn mặt từ video:
        - Lấy frame mỗi `sample_rate` khung.
        - Chạy face detection, lấy bbox có độ tin cậy cao nhất.
        - Trích xuất embedding ArcFace, tính trung bình rồi lưu vào bảng persons.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate phải lớn hơn 0")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {path}")

        embeddings: list[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % sample_rate != 0:
                idx += 1
                continue

            detections = self.detector.detect(frame)
            if not detections:
                idx += 1
                continue

            best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
            face_crop = self.detector.crop_face(frame, best_det["bbox"])
            embeddings.append(self.recognizer.embed(face_crop))
            idx += 1

        cap.release()

        if not embeddings:
            raise ValueError("Không tìm thấy khuôn mặt nào trong video.")

        mean_embedding = np.mean(embeddings, axis=0).astype("float32")
        mean_embedding = ArcFaceRecognizer.normalize_embedding(mean_embedding)
        embedding_text = ArcFaceRecognizer.serialize_embedding(mean_embedding)

        person = upsert_person(db, name=name, embedding_text=embedding_text)
        return {"id": person.id, "name": person.name, "samples": len(embeddings)}


@lru_cache(maxsize=1)
def get_pipeline() -> FacePipeline:
    """Singleton pipeline to avoid reloading models per request."""
    return FacePipeline()
