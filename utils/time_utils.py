from datetime import datetime


def format_seconds(total: int) -> str:
    """Convert a raw second count into MM:SS string. e.g. 90 → '1:30'"""
    minutes, seconds = divmod(total, 60)
    return f"{minutes}:{seconds:02d}"


def now_iso() -> str:
    """Return current local time as an ISO 8601 string for SQLite storage."""
    return datetime.now().isoformat(timespec="seconds")
