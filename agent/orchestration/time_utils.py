from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

US_EASTERN_TZ = ZoneInfo("America/New_York")


def now_us_eastern() -> datetime:
    return datetime.now(US_EASTERN_TZ)


def now_us_eastern_iso(*, timespec: str = "seconds") -> str:
    return now_us_eastern().isoformat(timespec=timespec)


def to_us_eastern(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(US_EASTERN_TZ)
