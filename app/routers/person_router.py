from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.MySQL import get_db
from app.schemas.person import PersonEnrollResponse
from app.service.pipeline import get_pipeline

router = APIRouter(prefix="/persons", tags=["persons"])


@router.post("/enroll", response_model=PersonEnrollResponse)
async def enroll_person(
    name: str = Form(..., description="Tên định danh cho khuôn mặt này"),
    file: UploadFile = File(..., description="Ảnh khuôn mặt"),
    db: Session = Depends(get_db),
):
    pipeline = get_pipeline()
    data = await file.read()
    try:
        person = pipeline.enroll_person(data, name, db)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    return PersonEnrollResponse(id=person["id"], name=person["name"], samples=person.get("samples"), message="Enroll thành công")


@router.post("/enroll-video", response_model=PersonEnrollResponse)
async def enroll_person_video(
    name: str = Form(..., description="Tên định danh cho khuôn mặt này"),
    file: UploadFile = File(..., description="Video chứa khuôn mặt"),
    db: Session = Depends(get_db),
):
    pipeline = get_pipeline()
    try:
        # Lưu tạm video để OpenCV đọc
        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1]) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())

        person = pipeline.enroll_person_from_video(temp_path, name, db, sample_rate=5)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return PersonEnrollResponse(
        id=person["id"], name=person["name"], samples=person.get("samples"), message="Enroll video thành công"
    )
