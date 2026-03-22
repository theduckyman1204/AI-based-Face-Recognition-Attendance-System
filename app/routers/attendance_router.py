from __future__ import annotations

from datetime import date, datetime, time as dtime
from typing import Callable, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.MySQL import get_db
from app.ropository.appearances import get_or_create_person, store_appearances
from app.ropository.attendance_stats import query_daily_attendance, recompute_attendance_stats
from app.schemas.attendance import (
    AttendanceEvent,
    AttendanceResponse,
    AttendanceStatItem,
    AttendanceStatsResponse,
    AttendanceStatUpdateIn,
    ManualAppearanceIn,
)
from app.utils.timezone_utils import VN_TZ, from_vn_time, now_vn, now_vn_naive, to_vn_time
from app.models.entities import AttendanceStat

router = APIRouter(prefix="/attendance", tags=["attendance"])


def _default_date_vn() -> date:
    return datetime.now(VN_TZ).date()


def _filter_people(entries: Dict[int, dict], predicate: Callable[[dtime, dtime | None], bool]) -> list[AttendanceEvent]:
    people: list[AttendanceEvent] = []
    for data in entries.values():
        arrived_at = data.get("arrived_at")
        left_at = data.get("left_at")
        if not arrived_at:
            continue
        arrive_time = arrived_at.astimezone(VN_TZ).time()
        leave_time = left_at.astimezone(VN_TZ).time() if left_at else None
        if predicate(arrive_time, leave_time):
            people.append(
                AttendanceEvent(
                    name=data["person"].name,
                    arrived_at=arrived_at,
                    left_at=left_at,
                )
            )
    return people


@router.get("/on-time-arrivals", response_model=AttendanceResponse)
def on_time_arrivals(
    target_date: date = Query(None, description="Ngày cần tra cứu (UTC+7). Nếu bỏ trống sẽ dùng hôm nay."),
    db: Session = Depends(get_db),
):
    target = target_date or _default_date_vn()
    entries = query_daily_attendance(db, target)
    people = _filter_people(entries, lambda arrive, _leave: arrive < dtime(hour=8))
    return AttendanceResponse(date=target, people=people)


@router.get("/late-arrivals", response_model=AttendanceResponse)
def late_arrivals(
    target_date: date = Query(None, description="Ngày cần tra cứu (UTC+7). Nếu bỏ trống sẽ dùng hôm nay."),
    db: Session = Depends(get_db),
):
    target = target_date or _default_date_vn()
    entries = query_daily_attendance(db, target)
    people = _filter_people(
        entries, lambda arrive, _leave: arrive >= dtime(hour=8) and arrive < dtime(hour=12)
    )
    return AttendanceResponse(date=target, people=people)


@router.get("/early-leaves", response_model=AttendanceResponse)
def early_leaves(
    target_date: date = Query(None, description="Ngày cần tra cứu (UTC+7). Nếu bỏ trống sẽ dùng hôm nay."),
    db: Session = Depends(get_db),
):
    target = target_date or _default_date_vn()
    entries = query_daily_attendance(db, target)
    people = _filter_people(
        entries, lambda _arrive, leave: leave is not None and leave >= dtime(hour=12) and leave < dtime(hour=17)
    )
    return AttendanceResponse(date=target, people=people)


@router.get("/on-time-leaves", response_model=AttendanceResponse)
def on_time_leaves(
    target_date: date = Query(None, description="Ngày cần tra cứu (UTC+7). Nếu bỏ trống sẽ dùng hôm nay."),
    db: Session = Depends(get_db),
):
    target = target_date or _default_date_vn()
    entries = query_daily_attendance(db, target)
    people = _filter_people(entries, lambda _arrive, leave: leave is not None and leave >= dtime(hour=17))
    return AttendanceResponse(date=target, people=people)


@router.post("/appearances", response_model=AttendanceEvent)
def add_manual_appearance(payload: ManualAppearanceIn, db: Session = Depends(get_db)):
    appeared_at_vn = payload.appeared_at or now_vn()
    try:
        created = store_appearances(
            db,
            [
                {
                    "name": payload.name,
                    "appeared_at": from_vn_time(appeared_at_vn),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

    record = created[0]
    return AttendanceEvent(
        name=record.name,
        arrived_at=to_vn_time(record.appeared_at),
        left_at=None,
    )


@router.get("/stats", response_model=AttendanceStatsResponse)
def attendance_stats(db: Session = Depends(get_db)):
    stats = recompute_attendance_stats(db)
    items: list[AttendanceStatItem] = []
    for stat in stats:
        person_name = stat.person.name if stat.person else "unknown"
        items.append(
            AttendanceStatItem(
                name=person_name,
                on_time_days=stat.on_time_days,
                late_days=stat.late_days,
                early_leave_days=stat.early_leave_days,
                updated_at=to_vn_time(stat.updated_at),
            )
        )
    return AttendanceStatsResponse(stats=items)


@router.put("/stats", response_model=AttendanceStatItem)
def update_attendance_stat(payload: AttendanceStatUpdateIn, db: Session = Depends(get_db)):
    person = get_or_create_person(db, payload.name)
    stat = db.query(AttendanceStat).filter(AttendanceStat.person_id == person.id).first()
    if not stat:
        stat = AttendanceStat(person_id=person.id)
        db.add(stat)

    stat.on_time_days = payload.on_time_days
    stat.late_days = payload.late_days
    stat.early_leave_days = payload.early_leave_days
    stat.updated_at = now_vn_naive()

    db.commit()
    db.refresh(stat)

    return AttendanceStatItem(
        name=person.name,
        on_time_days=stat.on_time_days,
        late_days=stat.late_days,
        early_leave_days=stat.early_leave_days,
        updated_at=to_vn_time(stat.updated_at),
    )
