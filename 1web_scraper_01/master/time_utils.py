from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

US_EASTERN_TZ = ZoneInfo("America/New_York")


def now_us_eastern_iso(*, timespec: str = "seconds") -> str:
    return datetime.now(US_EASTERN_TZ).isoformat(timespec=timespec)
