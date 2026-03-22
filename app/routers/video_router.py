import os
import tempfile
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.MySQL import get_db
from app.schemas.detection import AppearanceSearchResponse, DetectionBox, DetectionResponse, RecognitionResult
from app.service.pipeline import get_pipeline
from app.utils.timezone_utils import to_vn_time

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/detect", response_model=DetectionResponse)
async def detect_video(
    file: UploadFile = File(..., description="Video file containing faces"),
    db: Session = Depends(get_db),
):
    pipeline = get_pipeline()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1]) as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        results = pipeline.analyze_video(temp_path, None, db, sample_rate=5)
    except Exception as exc:  # noqa: BLE001 - bubble to client
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    payload: List[RecognitionResult] = []
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


@router.get("/appearances/events", response_model=AppearanceSearchResponse)
def search_appearances(
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db),
):
    """Return all appearance events between two timestamps."""
    pipeline = get_pipeline()
    events = pipeline.search_between(db, start_time, end_time)

    results: List[RecognitionResult] = []
    for ev in events:
        bbox_payload = None
        results.append(
            RecognitionResult(
                name=ev.name,
                confidence=None,
                appeared_at=to_vn_time(ev.appeared_at),
                bbox=bbox_payload,
            )
        )

    names = sorted({item.name for item in results})
    return AppearanceSearchResponse(names=names, events=results)


@router.get("/appearances/names", response_model=AppearanceSearchResponse)
def search_unique_names(
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db),
):
    """
    Return only the unique person names that appeared between start and end time.
    Events list will be empty to keep the response light.
    """
    pipeline = get_pipeline()
    events = pipeline.search_between(db, start_time, end_time)
    names = sorted({ev.name for ev in events})
    return AppearanceSearchResponse(names=names, events=[])
