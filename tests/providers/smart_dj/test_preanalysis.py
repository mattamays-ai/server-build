"""Tests for Smart DJ pre-analysis on arrival (0.0.28 lane B).

The pre-analysis worker must reuse the sweep's machinery: same hard timeout
via ``_sweep_attempt``, same concurrency limiter size, same deferral while a
player streams, same failure circuit breaker, same bounded queue semantics.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

from music_assistant_models.enums import MediaType, PlaybackState

from music_assistant.providers.smart_dj import provider as provider_module
from music_assistant.providers.smart_dj.provider import SmartDJProvider


def _player(state: PlaybackState) -> Any:
    """Fake a player whose state object carries .playback_state."""
    return SimpleNamespace(state=SimpleNamespace(playback_state=state))


def _event(data: Any) -> Any:
    """Fake a MassEvent payload (the real event is a frozen dataclass)."""
    return cast("Any", SimpleNamespace(data=data))


def _track(item_id: str = "t0") -> Any:
    """Fake a newly added library track (library item + provider mapping)."""
    mapping = SimpleNamespace(
        item_id=f"p-{item_id}",
        provider_instance="prov1",
        available=True,
    )
    return SimpleNamespace(
        item_id=item_id,
        name=f"track {item_id}",
        uri=f"library://track/{item_id}",
        provider="library",
        provider_mappings=[mapping],
        media_type=MediaType.TRACK,
    )


def _pre_prov(
    players: list[Any] | None = None,
    enabled: bool = True,
    silent: Any = None,
) -> SmartDJProvider:
    """Build a bare provider wired for pre-analysis tests (no real MA)."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = logging.getLogger("smart_dj_preanalysis_test")
    prov._library_scan_task = None
    prov._library_scan_cancel = asyncio.Event()
    prov._library_scan_state = {}
    prov._preanalysis_queue = deque()
    prov._preanalysis_pending_uris = set()
    prov._preanalysis_task = None
    prov._preanalysis_deferring = False
    prov.config = cast("Any", SimpleNamespace(get_value=lambda _key, _default=True: enabled))

    async def _noop() -> None:
        return None

    async def _default_silent(_item_id: str, _provider_id: str, **_kwargs: Any) -> bool:
        return True

    def _create_task(coro: Any, *_args: Any, **_kwargs: Any) -> Any:
        return asyncio.create_task(coro)

    prov.mass = cast(
        "Any",
        SimpleNamespace(
            players=players if players is not None else [],
            create_task=_create_task,
            config=SimpleNamespace(get_raw_core_config_value=lambda *_a, **_k: 2),
        ),
    )
    setattr(prov, "_save_cache", _noop)  # noqa: B010 - test wiring on a bare instance
    setattr(  # noqa: B010
        prov, "_analyze_media_silent", silent if silent is not None else _default_silent
    )
    return prov


async def _drained(prov: SmartDJProvider, timeout: float = 5.0) -> None:
    """Wait until the pre-analysis worker finished draining the queue."""
    async with asyncio.timeout(timeout):
        while prov._preanalysis_task is not None and not prov._preanalysis_task.done():
            await asyncio.sleep(0.01)


async def test_new_track_is_analyzed_on_arrival() -> None:
    """A MEDIA_ITEM_ADDED event for a track is analyzed in the background."""
    calls: list[str] = []

    async def _silent(item_id: str, _provider_id: str, **_kwargs: Any) -> bool:
        calls.append(item_id)
        return True

    prov = _pre_prov(silent=_silent)
    await prov._on_media_added(_event(_track("x1")))
    assert len(prov._preanalysis_queue) == 1
    await _drained(prov)
    assert calls == ["p-x1"]
    assert prov._preanalysis_queue == deque()
    assert prov._preanalysis_task is None


async def test_non_track_events_are_ignored() -> None:
    """Albums/artists/playlists never enter the pre-analysis queue."""
    prov = _pre_prov()
    album = SimpleNamespace(media_type=MediaType.ALBUM, uri="library://album/a0")
    await prov._on_media_added(_event(album))
    assert len(prov._preanalysis_queue) == 0
    assert prov._preanalysis_task is None


async def test_toggle_off_ignores_arrivals() -> None:
    """With the toggle OFF, arrivals are dropped before any queueing."""
    prov = _pre_prov(enabled=False)
    await prov._on_media_added(_event(_track("x1")))
    assert len(prov._preanalysis_queue) == 0
    assert prov._preanalysis_task is None


async def test_toggle_reads_live_with_explicit_default_on() -> None:
    """The reader defaults ON for a never-stored key (no bool(None) trap)."""

    def _values_missing(_key: str, default: Any = None) -> Any:
        return default

    prov = _pre_prov()
    prov.config = cast("Any", SimpleNamespace(get_value=_values_missing))
    assert prov._preanalysis_enabled() is True
    prov.config = cast("Any", SimpleNamespace(get_value=lambda _k, _d=False: False))
    assert prov._preanalysis_enabled() is False


async def test_writer_and_reader_are_paired() -> None:
    """The settings toggle ships writer (ConfigEntry) AND reader together."""
    assert provider_module.CONF_PREANALYSIS_ENABLED == "preanalysis_enabled"
    writer = inspect.getsource(SmartDJProvider.get_config_entries)
    assert "CONF_PREANALYSIS_ENABLED" in writer
    assert "default_value=True" in writer
    reader = inspect.getsource(SmartDJProvider._preanalysis_enabled)
    assert "CONF_PREANALYSIS_ENABLED" in reader


async def test_queue_cap_drops_oldest_with_log(caplog: Any) -> None:
    """A full queue drops the OLDEST pending track and logs a WARNING."""
    real_cap = provider_module.PREANALYSIS_QUEUE_MAX
    provider_module.PREANALYSIS_QUEUE_MAX = 2
    try:
        with caplog.at_level(logging.WARNING, logger="smart_dj_preanalysis_test"):
            prov = _pre_prov()
            await prov._on_media_added(_event(_track("t0")))
            await prov._on_media_added(_event(_track("t1")))
            await prov._on_media_added(_event(_track("t2")))
        assert [t.item_id for t in prov._preanalysis_queue] == ["t1", "t2"]
        assert "t0" not in prov._preanalysis_pending_uris
        assert any("dropping oldest" in r.message for r in caplog.records)
    finally:
        provider_module.PREANALYSIS_QUEUE_MAX = real_cap
        await _drained(prov)


async def test_duplicate_track_is_not_queued_twice() -> None:
    """The same uri pending twice is a no-op the second time."""
    prov = _pre_prov()
    track = _track("dup")
    await prov._on_media_added(_event(track))
    await prov._on_media_added(_event(track))
    assert len(prov._preanalysis_queue) == 1


async def test_defers_while_playing_then_proceeds() -> None:
    """No dispatch while a player is PLAYING; analysis resumes when idle."""
    calls: list[str] = []

    async def _silent(item_id: str, _provider_id: str, **_kwargs: Any) -> bool:
        calls.append(item_id)
        return True

    prov = _pre_prov(players=[_player(PlaybackState.PLAYING)], silent=_silent)
    real_poll = provider_module.LIBRARY_SCAN_POLL_SECONDS
    provider_module.LIBRARY_SCAN_POLL_SECONDS = 0.01
    try:
        await prov._on_media_added(_event(_track("x1")))
        await asyncio.sleep(0.05)
        live_deferring = prov._preanalysis_deferring
        assert live_deferring is True
        assert calls == []
        cast("Any", prov.mass).players = [_player(PlaybackState.IDLE)]
        await _drained(prov)
    finally:
        provider_module.LIBRARY_SCAN_POLL_SECONDS = real_poll
    final_deferring = prov._preanalysis_deferring
    assert final_deferring is False
    assert calls == ["p-x1"]


async def test_failure_breaker_aborts_and_drains(caplog: Any) -> None:
    """After LIBRARY_SCAN_MAX_FAILURES consecutive failures the queue drains."""
    calls = 0

    async def _broken(_item_id: str, _provider_id: str, **_kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider exploded")

    with caplog.at_level(logging.WARNING, logger="smart_dj_preanalysis_test"):
        prov = _pre_prov(silent=_broken)
        for i in range(60):
            await prov._on_media_added(_event(_track(f"t{i}")))
        await _drained(prov)
    assert calls <= provider_module.LIBRARY_SCAN_MAX_FAILURES + 2  # concurrent in-flight
    assert len(prov._preanalysis_queue) == 0
    assert any("consecutive failures" in r.message for r in caplog.records)


async def test_timeout_counts_as_failure() -> None:
    """The sweep's hard per-track timeout is an honest pre-analysis failure."""
    calls = 0

    async def _hung(_item_id: str, _provider_id: str, **_kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        await asyncio.sleep(300)
        return True

    prov = _pre_prov(silent=_hung)
    real_timeout = provider_module.ANALYSIS_ATTEMPT_TIMEOUT
    provider_module.ANALYSIS_ATTEMPT_TIMEOUT = 0.05  # type: ignore[assignment]
    try:
        await prov._on_media_added(_event(_track("hung")))
        await _drained(prov)
    finally:
        provider_module.ANALYSIS_ATTEMPT_TIMEOUT = real_timeout
    assert calls == 1
    assert len(prov._preanalysis_queue) == 0
