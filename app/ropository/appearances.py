from datetime import datetime
from typing import Iterable, List, Sequence

from sqlalchemy.orm import Session

from app.models.entities import Appearance, Person
from app.utils.timezone_utils import now_vn_naive, to_vn_time, vn_day_bounds
from app.ropository.attendance_stats import recompute_attendance_stats


def get_or_create_person(db: Session, name: str) -> Person:
    person = db.query(Person).filter(Person.name == name).first()
    if person:
        return person
    person = Person(name=name)
    db.add(person)
    db.commit()
    db.refresh(person)
    return person


def store_appearances(
    db: Session,
    records: Iterable[dict],
) -> List[Appearance]:
    created: List[Appearance] = []
    for record in records:
        person = get_or_create_person(db, record.get("name", "unknown"))
        appeared_at = record.get("appeared_at", now_vn_naive())

        # Giới hạn tối đa 2 lần xuất hiện/ngày (giờ VN) và cập nhật mốc đến/về.
        vn_dt = to_vn_time(appeared_at)
        start_utc, end_utc = vn_day_bounds(vn_dt.date())
        day_appearances = (
            db.query(Appearance)
            .filter(Appearance.person_id == person.id)
            .filter(Appearance.appeared_at >= start_utc)
            .filter(Appearance.appeared_at <= end_utc)
            .order_by(Appearance.appeared_at)
            .all()
        )

        if len(day_appearances) == 0:
            appearance = Appearance(person_id=person.id, name=person.name, appeared_at=appeared_at)
            db.add(appearance)
            created.append(appearance)
        elif len(day_appearances) == 1:
            first = day_appearances[0]
            if appeared_at < first.appeared_at:
                first.appeared_at = appeared_at
                created.append(first)
            else:
                appearance = Appearance(person_id=person.id, name=person.name, appeared_at=appeared_at)
                db.add(appearance)
                created.append(appearance)
        else:  # đã có 2 mốc -> giữ mốc sớm nhất và muộn nhất
            earliest, latest = day_appearances[0], day_appearances[1]
            if appeared_at < earliest.appeared_at:
                earliest.appeared_at = appeared_at
                created.append(earliest)
            elif appeared_at > latest.appeared_at:
                latest.appeared_at = appeared_at
                created.append(latest)
            # nếu nằm giữa thì bỏ qua để không vượt quá 2 lần/ngày

    db.commit()
    for appearance in created:
        db.refresh(appearance)

    # Cập nhật thống kê đi muộn/về sớm dựa trên toàn bộ appearances hiện có.
    recompute_attendance_stats(db)
    return created


def search_appearances(
    db: Session,
    start_time: datetime,
    end_time: datetime,
) -> Sequence[Appearance]:
    return (
        db.query(Appearance)
        .join(Person)
        .filter(Appearance.appeared_at >= start_time)
        .filter(Appearance.appeared_at <= end_time)
        .order_by(Appearance.appeared_at)
        .all()
    )
