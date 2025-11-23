# Face Recognition API (Docker only)

FastAPI service for YOLO face detection + ArcFace recognition with DeepSORT tracking, backed by MySQL. Includes phpMyAdmin and a Streamlit UI, all run via Docker Compose.

## Prerequisites
- Place model files before building:
  - YOLO weights: `app/models/yolov12n.pt` (or your own).
  - ArcFace TorchScript checkpoint: `app/models/arcface_mobilenet_v3_e33.pth` (update `ARC_FACE_MODEL_PATH` in `app/.env` if different).
  - DeepSORT ReID checkpoint: `YOLOv8-Object-Detection-with-DeepSORT-Tracking/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7` (update `DEEP_SORT_CHECKPOINT` as needed).
- Ensure `app/.env` exists (defaults provided; DB URL is overridden by Compose).
- Python deps pinned: `pydantic<2` and `fastapi<0.110` (FastAPI versions before 0.110 use Pydantic v1).

## Run with Docker Compose
1. Build and start the stack (API, MySQL, phpMyAdmin, Streamlit):  
   `docker-compose up --build`
2. Access services:  
   - API (FastAPI docs): http://localhost:8001  
   - phpMyAdmin: http://localhost:8080 (user `faceuser`, pass `facepass`)  
   - Streamlit UI: http://localhost:8501  
   - MySQL mapped host port: `3307` (service still listens on 3306 internally)

## API (served by the API container)
- `POST /images/detect` — multipart `file` (image), optional `source_name`; runs detection/recognition and stores appearances.
- `POST /videos/detect` — multipart `file` (video), optional `source_name`, `sample_rate`; uses DeepSORT to avoid duplicate inserts per person.
- `GET /videos/appearances/events?start_time=...&end_time=...` — all appearance events in window.
- `GET /videos/appearances/names?start_time=...&end_time=...` — unique names in window.

## Notes
- DeepSORT expects the cloned repo at `YOLOv8-Object-Detection-with-DeepSORT-Tracking/deep_sort_pytorch`.
- Embeddings are stored on the `persons` table; replace with a vector store if needed.
- For production, swap `Base.metadata.create_all` for migrations and tighten CORS/security.
