from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch

from app.models.entities import Person
from app.models.arcface_mobilenet_v3 import ArcFace

logger = logging.getLogger(__name__)


class ArcFaceRecognizer:
    """
    ArcFace inference wrapper that loads a checkpoint trained with MobileNetV3 backbone.

    `model_path` trỏ tới một checkpoint dạng state_dict (ví dụ: `app/models/arcface_mobilenet_v3_e24.pth`).
    Checkpoint này chứa các weight cho module ArcFace:
        - backbone.*      : MobileNetV3 feature extractor
        - embedding.*     : Linear(960 -> feature_dim)
        - arcface.*       : ArcMarginProduct(feature_dim -> num_classes)

    Ở chế độ eval, ArcFace.forward(x) trả về embedding đã L2-normalize,
    dùng được trực tiếp để so sánh cosine trong nhận diện khuôn mặt.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        device: str | None = None,
        fallback_model_path: str | None = None,
    ):
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(model_path, fallback_model_path)
        self._known: Dict[str, np.ndarray] = {}

    def _load_model(self, model_path: str, fallback_model_path: str | None = None):
        """
        Load ArcFace model từ một checkpoint state_dict.

        Nếu checkpoint không phải dict (ai đó lỡ torch.save(model)), sẽ dùng trực tiếp.
        Nếu là dict (OrderedDict), sẽ:
          - Suy ra feature_dim từ shape của embedding.weight
          - Suy ra num_classes từ shape của arcface.weight
          - Khởi tạo ArcFace(num_classes, feature_dim)
          - Nạp state_dict (bỏ prefix 'module.' nếu có)
        """

        def _build_from_checkpoint(path: str):
            logger.info("Loading ArcFace state_dict from: %s", path)
            ckpt = torch.load(path, map_location=self.device)

            # Trường hợp hiếm: ai đó đã torch.save(model) -> dùng luôn
            if not isinstance(ckpt, dict):
                model = ckpt
                model.to(self.device)
                model.eval()
                return model

            # Suy ra kích thước từ weight
            try:
                # embedding.weight: [feature_dim, 960]
                feature_dim = ckpt["embedding.weight"].shape[0]
                # arcface.weight: [num_classes, feature_dim]
                num_classes = ckpt["arcface.weight"].shape[0]
            except KeyError as e:
                raise RuntimeError(
                    f"Checkpoint tại {path} không có cấu trúc mong đợi, thiếu key: {e}"
                )

            model = ArcFace(
                num_classes=num_classes,
                feature_dim=feature_dim,
                backbone_weight=None,  # backbone weight đã nằm trong state_dict
            )

            # Nếu được train bằng DataParallel, key thường có prefix 'module.'
            clean_state = {}
            for k, v in ckpt.items():
                new_k = k[len("module."):] if k.startswith("module.") else k
                clean_state[new_k] = v

            model.load_state_dict(clean_state, strict=True)
            model.to(self.device)
            model.eval()
            return model

        try:
            return _build_from_checkpoint(model_path)
        except Exception as exc:  # noqa: BLE001
            if fallback_model_path and fallback_model_path != model_path and os.path.exists(fallback_model_path):
                logger.warning(
                    "Failed to load ArcFace model %s (%s). Falling back to %s",
                    model_path,
                    exc,
                    fallback_model_path,
                )
                try:
                    return _build_from_checkpoint(fallback_model_path)
                except Exception as exc_fallback:  # noqa: BLE001
                    raise RuntimeError(
                        f"Cannot load ArcFace checkpoint at {model_path} "
                        f"and fallback {fallback_model_path}: {exc_fallback}"
                    )
            raise RuntimeError(f"Cannot load ArcFace checkpoint at {model_path}: {exc}")

    # ------------------------------------------------------------------ #
    #   QUẢN LÝ EMBEDDING ĐÃ LƯU (INDEX)
    # ------------------------------------------------------------------ #

    def rebuild_index(self, persons: Iterable[Person]) -> None:
        """Load embeddings từ bảng persons trong DB vào memory."""
        self._known.clear()
        for person in persons:
            if not person.embedding:
                continue
            embedding = self._deserialize_embedding(person.embedding)
            if embedding is None:
                continue
            self._known[person.name] = embedding

    # ------------------------------------------------------------------ #
    #   NHẬN DIỆN
    # ------------------------------------------------------------------ #

    def recognize(self, face_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Trả về (name, score, embedding).
        Name mặc định là 'unknown' nếu không ai vượt ngưỡng threshold.
        """
        embedding = self._embed(face_image)
        if not self._known:
            return "unknown", 0.0, embedding

        best_name = "unknown"
        best_score = -1.0
        for name, known_embedding in self._known.items():
            score = float(np.dot(embedding, known_embedding))
            if score > best_score:
                best_name = name
                best_score = score

        if best_score < self.threshold:
            best_name = "unknown"
        return best_name, best_score, embedding

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        """Trả về embedding đã chuẩn hóa (dùng cho enroll)."""
        return self._embed(face_image)

    def _embed(self, face_image: np.ndarray) -> np.ndarray:
        """
        Tính embedding bằng ArcFace.

        Pipeline:
        - Resize mặt về 112x112
        - BGR -> RGB
        - Scale về [0, 1], rồi chuẩn hóa về [-1, 1]
        - Forward qua model ArcFace (ở eval mode)
        - L2-normalize embedding về norm ≈ 1.0
        """
        resized = cv2.resize(face_image, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - 0.5) / 0.5  # normalize về [-1, 1]
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)  # ArcFace ở eval sẽ trả embedding
        if isinstance(output, (tuple, list)):
            output = output[0]

        embedding = output.squeeze(0).detach().cpu().numpy().astype("float32")
        norm = np.linalg.norm(embedding) + 1e-9
        return embedding / norm

    # ------------------------------------------------------------------ #
    #   SERIALIZE / DESERIALIZE EMBEDDING
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deserialize_embedding(raw: str) -> np.ndarray | None:
        try:
            vector = np.fromstring(raw, sep=",", dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0:
                return None
            return vector / norm
        except Exception:
            return None

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> str:
        return ",".join([f"{v:.8f}" for v in embedding.tolist()])

    @staticmethod
    def normalize_embedding(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector) + 1e-9
        return vector / norm
