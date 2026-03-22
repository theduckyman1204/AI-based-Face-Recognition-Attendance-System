import os
import sys
from typing import Optional

import numpy as np

from app.core.config import settings

# Backward-compat for older DeepSORT/YOLO files that still reference deprecated NumPy aliases.
if not hasattr(np, "float"):  # pragma: no cover
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):  # pragma: no cover
    np.int = int  # type: ignore[attr-defined]


class DeepSortWrapper:
    """Thin wrapper to use cloned deep_sort_pytorch repo without polluting globals."""

    def __init__(
        self,
        repo_path: str = settings.deep_sort_repo_path,
        checkpoint: str = settings.deep_sort_checkpoint,
        use_cuda: bool = settings.deep_sort_use_cuda,
    ):
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self.use_cuda = use_cuda
        self._tracker = self._load()

    def _load(self):
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"DeepSORT checkpoint not found at {self.checkpoint}. "
                "Download ckpt.t7 into deep_sort/deep/checkpoint."
            )
        # Temporarily inject repo path to import deep_sort_pytorch
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
        parent = os.path.dirname(self.repo_path)
        if parent and parent not in sys.path:
            sys.path.append(parent)
        if not os.path.isdir(self.repo_path):
            raise ModuleNotFoundError(
                f"DeepSORT repo path missing: {self.repo_path}. "
                "Clone YOLOv8-Object-Detection-with-DeepSORT-Tracking and set DEEP_SORT_REPO_PATH."
            )
        try:
            from deep_sort_pytorch.deep_sort import DeepSort  # type: ignore
            from deep_sort_pytorch.utils.parser import get_config  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise ModuleNotFoundError(
                f"Cannot import deep_sort_pytorch from {self.repo_path}. "
                f"sys.path={sys.path}. "
                "Ensure the repo is cloned and DEEP_SORT_REPO_PATH points to its deep_sort_pytorch folder."
            ) from exc

        cfg = get_config()
        cfg.merge_from_file(os.path.join(self.repo_path, "configs", "deep_sort.yaml"))
        tracker = DeepSort(
            model_path=self.checkpoint,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=self.use_cuda,
        )
        return tracker

    def update(self, boxes_xyxy: np.ndarray, confidences: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker and return outputs [x1, y1, x2, y2, track_id, cls_id].
        """
        if boxes_xyxy.size == 0:
            self._tracker.increment_ages()
            return np.empty((0, 6))

        bbox_xywh = self._xyxy_to_xywh(boxes_xyxy)
        classes = np.zeros(len(confidences), dtype=int)  # single-class faces
        outputs = self._tracker.update(bbox_xywh, confidences, classes, frame)
        return outputs if outputs is not None else np.empty((0, 6))

    @staticmethod
    def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
        converted = np.zeros_like(boxes)
        converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # center x
        converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # center y
        converted[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        converted[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
        return converted
