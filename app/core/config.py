from pathlib import Path

from pydantic import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent  # /app/app
REPO_ROOT = BASE_DIR.parent  # /app


class Settings(BaseSettings):
    """Application settings loaded from environment variables or `.env`."""

    app_name: str = "Face Recognition API"
    mysql_url: str = "mysql+pymysql://faceuser:facepass@mysql:3306/face_recog"
    yolo_weights: str = str(BASE_DIR / "models" / "bestv8.pt")
    arcface_model_path: str = str(BASE_DIR / "models" / "arcface_mobilenet_v3_e24.pth")
    arcface_fallback_model_path: str | None = str(BASE_DIR / "models" / "arcface_mobilenet_v3_e33.pth")
    detection_confidence: float = 0.4
    recognition_threshold: float = 0.5
    deep_sort_repo_path: str = str(REPO_ROOT / "YOLOv8-Object-Detection-with-DeepSORT-Tracking" / "deep_sort_pytorch")
    deep_sort_checkpoint: str = str(
        REPO_ROOT
        / "YOLOv8-Object-Detection-with-DeepSORT-Tracking"
        / "deep_sort_pytorch"
        / "deep_sort"
        / "deep"
        / "checkpoint"
        / "ckpt.t7"
    )
    deep_sort_use_cuda: bool = True
    db_connect_retries: int = 30
    db_connect_interval: float = 2.0

    class Config:
        # Load env vars from the repository's app/.env when running locally
        env_file = "app/.env"
        env_file_encoding = "utf-8"


settings = Settings()
