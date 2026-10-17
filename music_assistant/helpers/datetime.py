"""Helpers for date and time."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from zoneinfo import available_timezones

LOCAL_TIMEZONE = dt.datetime.now(dt.UTC).astimezone().tzinfo


def utc() -> dt.datetime:
    """Get current UTC datetime."""
    return dt.datetime.now(dt.UTC)


def utc_timestamp() -> float:
    """Return UTC timestamp in seconds as float."""
    return utc().timestamp()


def now() -> dt.datetime:
    """Get current datetime in local timezone."""
    return dt.datetime.now(LOCAL_TIMEZONE)


def now_timestamp() -> float:
    """Return current datetime as timestamp in local timezone."""
    return now().timestamp()


def future_timestamp(**kwargs: float) -> float:
    """Return current timestamp + timedelta."""
    return (now() + dt.timedelta(**kwargs)).timestamp()


def from_utc_timestamp(timestamp: float) -> dt.datetime:
    """Return datetime from UTC timestamp."""
    return dt.datetime.fromtimestamp(timestamp, dt.UTC)


def iso_from_utc_timestamp(timestamp: float) -> str:
    """Return ISO 8601 datetime string from UTC timestamp."""
    return from_utc_timestamp(timestamp).isoformat()


def from_iso_string(iso_datetime: str) -> dt.datetime:
    """Return datetime from ISO datetime string."""
    return dt.datetime.fromisoformat(iso_datetime)


def host_timezone_name() -> str:
    """
    Return the host's IANA timezone name (e.g. "Europe/Amsterdam"), falling back to "UTC".

    Unlike ``LOCAL_TIMEZONE`` (a fixed-offset tzinfo, e.g. "CEST"), this resolves an actual
    IANA zone name, checked in order: the ``TZ`` environment variable, then the
    ``/etc/localtime`` symlink target. Never raises.
    """
    available = available_timezones()
    tz_env = os.environ.get("TZ", "").strip()
    if tz_env in available:
        return tz_env
    try:
        link_target = str(Path("/etc/localtime").readlink())
    except OSError:
        link_target = ""
    _, _, candidate = link_target.partition("zoneinfo/")
    if candidate in available:
        return candidate
    return "UTC"


def local_clock_time_to_utc(hour: int, minute: int = 0) -> tuple[int, int]:
    """
    Convert a server-local wall clock time to UTC hour/minute.

    This uses the server's current local timezone offset.
    """
    local_timezone = LOCAL_TIMEZONE or dt.UTC
    local_datetime = dt.datetime.now(local_timezone).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    utc_datetime = local_datetime.astimezone(dt.UTC)
    return utc_datetime.hour, utc_datetime.minute
