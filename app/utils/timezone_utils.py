from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Tuple

VN_TZ = timezone(timedelta(hours=7))


def now_vn() -> datetime:
    """Return current time in Vietnam timezone (tz-aware)."""
    return datetime.now(VN_TZ)


def now_vn_naive() -> datetime:
    """Return current time in Vietnam timezone but without tzinfo (store as local time)."""
    return now_vn().replace(tzinfo=None)


def to_vn_time(value: datetime) -> datetime:
    """
    Convert a datetime to Vietnam timezone-aware datetime.
    - Nếu datetime không có tzinfo, coi đó là giờ Việt Nam hiện tại.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=VN_TZ)
    return value.astimezone(VN_TZ)


def from_vn_time(value: datetime) -> datetime:
    """
    Convert a Vietnam (or timezone-aware) datetime to naive Vietnam time for storage.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=VN_TZ)
    return value.astimezone(VN_TZ).replace(tzinfo=None)


def vn_day_bounds(target: date) -> Tuple[datetime, datetime]:
    """
    Return UTC naive start/end datetime that correspond to a local Vietnam day.
    """
    start_local = datetime.combine(target, dtime.min, tzinfo=VN_TZ)
    end_local = datetime.combine(target, dtime.max, tzinfo=VN_TZ)
    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc
