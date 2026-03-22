from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, Iterable, Tuple

from sqlalchemy.orm import Session

from app.models.entities import Appearance, AttendanceStat, Person
from app.utils.timezone_utils import VN_TZ, to_vn_time, vn_day_bounds


def _collect_daily_entries(appearances: Iterable[Appearance]) -> Dict[Tuple[int, date], dict]:
    """
    Group appearances by person/day (local VN) to find first and last timestamps.
    """
    per_day: Dict[Tuple[int, date], dict] = {}
    for app in appearances:
        vn_ts = to_vn_time(app.appeared_at)
        key = (app.person_id, vn_ts.date())
        entry = per_day.get(key)
        if not entry:
            per_day[key] = {
                "person": app.person,
                "arrived_at": vn_ts,
                "left_at": vn_ts,
            }
            continue

        if vn_ts < entry["arrived_at"]:
            entry["arrived_at"] = vn_ts
        if vn_ts > entry["left_at"]:
            entry["left_at"] = vn_ts
    return per_day


def recompute_attendance_stats(db: Session) -> Iterable[AttendanceStat]:
    """
    Recompute attendance stats for all persons based on all appearance records.
    """
    appearances = db.query(Appearance).join(Person).all()
    per_day = _collect_daily_entries(appearances)

    aggregates: Dict[int, dict] = defaultdict(lambda: {"on_time": 0, "late": 0, "early_leave": 0, "person": None})

    for (_, _day), entry in per_day.items():
        person = entry["person"]
        arrive_dt = to_vn_time(entry["arrived_at"])
        leave_dt = to_vn_time(entry["left_at"])

        # Nếu lệch ngày (do chênh tz hoặc ghi nhận về muộn), quy về cùng ngày đến để tính ca làm.
        if leave_dt.date() != arrive_dt.date() and abs((leave_dt - arrive_dt)) <= timedelta(hours=24):
            leave_dt = datetime.combine(arrive_dt.date(), leave_dt.timetz())

        arrive = arrive_dt.time()
        leave = leave_dt.time()
        agg = aggregates[person.id]
        agg["person"] = person

        # Đi làm đúng giờ nếu đến trước 08:00 và về sau hoặc bằng 17:00.
        if arrive < dtime(hour=8) and leave >= dtime(hour=17):
            agg["on_time"] += 1
        elif arrive >= dtime(hour=8) and arrive < dtime(hour=12):
            agg["late"] += 1

        if leave >= dtime(hour=12) and leave < dtime(hour=17):
            agg["early_leave"] += 1

    results: list[AttendanceStat] = []
    for person_id, data in aggregates.items():
        stat = db.query(AttendanceStat).filter(AttendanceStat.person_id == person_id).first()
        if not stat:
            stat = AttendanceStat(person_id=person_id, person=data["person"])
            db.add(stat)
        stat.on_time_days = data["on_time"]
        stat.late_days = data["late"]
        stat.early_leave_days = data["early_leave"]
        stat.updated_at = to_vn_time(datetime.utcnow()).replace(tzinfo=None)
        results.append(stat)

    db.commit()
    for item in results:
        db.refresh(item)
    return results


def query_daily_attendance(db: Session, target_date: date) -> Dict[int, dict]:
    """
    Return per-person arrival/leave info for a single day (VN timezone).
    """
    start_utc, end_utc = vn_day_bounds(target_date)
    appearances = (
        db.query(Appearance)
        .join(Person)
        .filter(Appearance.appeared_at >= start_utc)
        .filter(Appearance.appeared_at <= end_utc)
        .all()
    )
    return _collect_daily_entries(appearances)
