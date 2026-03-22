from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AttendanceEvent(BaseModel):
    name: str
    arrived_at: Optional[datetime] = Field(None, description="Thời gian xuất hiện đầu tiên (UTC+7)")
    left_at: Optional[datetime] = Field(None, description="Thời gian xuất hiện cuối cùng (UTC+7)")


class AttendanceResponse(BaseModel):
    date: date
    timezone: str = "UTC+7"
    people: List[AttendanceEvent]


class ManualAppearanceIn(BaseModel):
    name: str = Field(..., description="Tên người")
    appeared_at: Optional[datetime] = Field(None, description="Thời gian xuất hiện (UTC+7). Nếu bỏ trống sẽ dùng thời gian hiện tại.")


class AttendanceStatItem(BaseModel):
    name: str
    on_time_days: int
    late_days: int
    early_leave_days: int
    updated_at: datetime


class AttendanceStatsResponse(BaseModel):
    timezone: str = "UTC+7"
    stats: List[AttendanceStatItem]


class AttendanceStatUpdateIn(BaseModel):
    name: str
    on_time_days: int = 0
    late_days: int = 0
    early_leave_days: int = 0
