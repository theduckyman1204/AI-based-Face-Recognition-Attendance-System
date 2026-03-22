from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.MySQL import get_db
from app.schemas.detection import DetectionBox, DetectionResponse, RecognitionResult
from app.service.pipeline import get_pipeline
from app.utils.timezone_utils import to_vn_time

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/detect", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file containing faces"),
    source_name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    pipeline = get_pipeline()
    data = await file.read()
    try:
        results = pipeline.analyze_image(data, source_name, db)
    except Exception as exc:  # noqa: BLE001 - bubble to client
        raise HTTPException(status_code=400, detail=str(exc))

    payload = []
    for item in results:
        bbox = item.get("bbox")
        payload.append(
            RecognitionResult(
                name=item["name"],
                confidence=item.get("confidence"),
                appeared_at=to_vn_time(item["appeared_at"]),
                bbox=DetectionBox(
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    confidence=item.get("confidence", 0.0),
                )
                if bbox
                else None,
            )
        )
    return DetectionResponse(results=payload)
