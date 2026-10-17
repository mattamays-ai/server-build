"""Tests for the library-wide local analysis sweep and its safeguards."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
from music_assistant_models.enums import PlaybackState

from music_assistant.providers.smart_dj import provider as provider_module
from music_assistant.providers.smart_dj.provider import SmartDJProvider


def _player(state: PlaybackState) -> Any:
    """Fake a player whose state object carries .playback_state."""
    return SimpleNamespace(state=SimpleNamespace(playback_state=state))


def _sweep_prov(
    players: list[Any] | None = None,
    tracks: int = 3,
    silent: Any = None,
) -> SmartDJProvider:
    """Build a bare provider wired for sweep tests (no real MA underneath)."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov._library_scan_task = None
    prov._library_scan_cancel = asyncio.Event()
    prov._library_scan_state = {}

    async def _library_items(**_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(item_id=f"t{i}", provider="prov1") for i in range(tracks)]

    async def _noop() -> None:
        return None

    async def _default_silent(_item_id: str, _provider_id: str) -> bool:
        return True

    prov.mass = cast(
        "Any",
        SimpleNamespace(
            players=players if players is not None else [],
            music=SimpleNamespace(tracks=SimpleNamespace(library_items=_library_items)),
            streams=SimpleNamespace(audio_analysis=SimpleNamespace(providers=[object()])),
            config=SimpleNamespace(get_raw_core_config_value=lambda *_a, **_k: 2),
        ),
    )
    setattr(prov, "_save_cache", _noop)  # noqa: B010 - test wiring on a bare instance
    setattr(  # noqa: B010
        prov, "_analyze_media_silent", silent if silent is not None else _default_silent
    )
    return prov


async def _finished(prov: SmartDJProvider, timeout: float = 5.0) -> dict[str, Any]:
    async with asyncio.timeout(timeout):
        while prov._library_scan_task is not None and not prov._library_scan_task.done():
            await asyncio.sleep(0.01)
    return dict(prov._library_scan_state)


async def test_bad_command_raises() -> None:
    """Only start/status/cancel are valid commands."""
    prov = _sweep_prov()
    with pytest.raises(RuntimeError, match="start, status, cancel"):
        await prov.analyze_library_local(command="explode")


async def test_status_before_any_sweep_is_empty() -> None:
    """Status with no sweep ever started is an empty dict, not an error."""
    prov = _sweep_prov()
    assert await prov.analyze_library_local(command="status") == {}


async def test_sweep_counts_analyzed_and_is_idempotent() -> None:
    """A clean sweep stores counts; a second start while running is a no-op."""
    prov = _sweep_prov(tracks=2)
    state = await prov.analyze_library_local(command="start")
    assert state["running"] is True
    assert state["total"] == 2
    # idempotent start: same live state, no second worker
    again = await prov.analyze_library_local(command="start")
    assert again["started_at"] == state["started_at"]
    final = await _finished(prov)
    assert final["running"] is False
    assert final["analyzed"] == 2
    assert final["skipped"] == 0
    assert final["failed"] == 0


async def test_sweep_skips_already_analyzed() -> None:
    """False from the silent analyzer counts as skipped, not failed."""

    async def _already(_item_id: str, _provider_id: str) -> bool:
        return False

    prov = _sweep_prov(tracks=2, silent=_already)
    await prov.analyze_library_local(command="start")
    final = await _finished(prov)
    assert final["analyzed"] == 0
    assert final["skipped"] == 2
    assert final["failed"] == 0


async def test_sweep_defers_to_active_playback() -> None:
    """No dispatch while a player is PLAYING; the sweep resumes when idle."""
    prov = _sweep_prov(players=[_player(PlaybackState.PLAYING)], tracks=1)
    monkeypatch_poll = provider_module.LIBRARY_SCAN_POLL_SECONDS
    provider_module.LIBRARY_SCAN_POLL_SECONDS = 0.01
    try:
        await prov.analyze_library_local(command="start", pause_while_playing=True)
        await asyncio.sleep(0.05)
        live = dict(prov._library_scan_state)
        assert live["deferring"] is True
        assert live["analyzed"] == 0
        # playback stops -> the sweep proceeds
        prov.mass.players = [_player(PlaybackState.IDLE)]
        final = await _finished(prov)
    finally:
        provider_module.LIBRARY_SCAN_POLL_SECONDS = monkeypatch_poll
    assert final["deferring"] is False
    assert final["analyzed"] == 1


async def test_sweep_plays_through_when_pause_disabled() -> None:
    """pause_while_playing=False dispatches even with a player playing."""
    prov = _sweep_prov(players=[_player(PlaybackState.PLAYING)], tracks=1)
    await prov.analyze_library_local(command="start", pause_while_playing=False)
    final = await _finished(prov)
    assert final["analyzed"] == 1


async def test_cancel_stops_the_sweep() -> None:
    """Cancel marks the state cancelled and the worker stops cleanly."""

    async def _slow(_item_id: str, _provider_id: str) -> bool:
        await asyncio.sleep(30)
        return True

    prov = _sweep_prov(tracks=1, silent=_slow)
    await prov.analyze_library_local(command="start")
    await asyncio.sleep(0.05)
    state = await prov.analyze_library_local(command="cancel")
    assert state["cancelled"] is True  # set immediately when the worker is interrupted
    final = await _finished(prov, timeout=2.0)
    assert final["running"] is False


async def test_failure_circuit_breaker_aborts() -> None:
    """After LIBRARY_SCAN_MAX_FAILURES consecutive failures the sweep aborts."""
    calls = 0

    async def _broken(_item_id: str, _provider_id: str) -> bool:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider exploded")

    prov = _sweep_prov(tracks=60, silent=_broken)
    await prov.analyze_library_local(command="start")
    final = await _finished(prov)
    assert final["running"] is False
    assert calls <= provider_module.LIBRARY_SCAN_MAX_FAILURES + 8  # concurrent in-flight
    assert final["failed"] >= provider_module.LIBRARY_SCAN_MAX_FAILURES
    assert any("aborted" in e for e in final["errors"])


async def test_registration_scope_is_library_write() -> None:
    """The sweep writes the analysis DB: it must never be guest-reachable."""
    source = inspect.getsource(SmartDJProvider.loaded_in_mass)
    assert (
        '("smart_dj/analyze_library_local", self.analyze_library_local, Scope.LIBRARY_WRITE)'
        in source
    )
