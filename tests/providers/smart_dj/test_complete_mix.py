"""
Regression tests for smart_dj/complete_mix (0.0.28 follow-up).

The command crashed at its LAST key: _transition_report walks every element
with .get(), but complete_mix fed it live media objects for the appended
tracks while only the seed was a snapshot dict. The queue had already been
mutated and the preroll already fired by then, so the FE rendered
"Complete mix failed" while the music had in fact changed - a
silent-success-reported-as-failure.

These tests pin both halves: the real path must not raise, and the row
shapes must stay dicts (the negative control proves the old shape still
fails, so a future refactor cannot quietly reintroduce it).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import music_assistant.providers.smart_dj.provider as prov_module
from music_assistant.providers.smart_dj.engine import DJControls
from music_assistant.providers.smart_dj.provider import SmartDJProvider, _transition_report


def _track(name: str) -> Any:
    """Return a minimal Track-shaped media object with no .get method."""
    return SimpleNamespace(name=name, artist_str="Artist", item_id=name.lower())


def _bare_provider() -> Any:
    """Return a bare provider whose queue and radio fetch are stubbed."""
    prov = SmartDJProvider.__new__(SmartDJProvider)
    prov.logger = __import__("logging").getLogger("smart-dj-test")
    prov.mass = SimpleNamespace(  # type: ignore[assignment]
        player_queues=SimpleNamespace(
            get=_fake_queue_get,
            load=_noop_load,
        ),
        music=SimpleNamespace(tracks=SimpleNamespace(get=_track_get)),
        get_provider_instances=lambda _d: [SimpleNamespace(instance_id="spotify--x")],
        create_task=_noop_task,
    )
    prov._read_full_queue = lambda _qid: []  # type: ignore[method-assign, assignment, misc]
    return prov


def _fake_queue_get(_qid: str) -> Any:
    """Return a non-crossfade queue stand-in."""
    return SimpleNamespace(crossfade_enabled=False)


async def _noop_load(*_args: Any, **_kwargs: Any) -> None:
    """Stand in for player_queues.load (the real one mutates the queue)."""
    return


async def _track_get(item_id: str, _provider: str) -> Any:
    """Resolve a radio id to a media object, like mass.music.tracks.get."""
    return _track(f"Radio {item_id}")


def _noop_task(coro: Any) -> None:
    """Swallow the fire-and-forget preroll task."""
    coro.close()


def test_complete_mix_survives_track_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The real command returns transition notes instead of raising AttributeError.

    This is the regression guard: before the fix this call raised
    "'Track' object has no attribute 'get'" from _transition_report, after the
    queue had already been mutated.
    """
    provider = _bare_provider()
    seed = {"queue_item_id": "q9", "name": "Seed Song", "item_id": "seed1", "provider": "spotify"}

    async def fake_snapshot(_qid: str) -> list[dict[str, Any]]:
        return [seed]

    async def fake_radio(_request: Any, _seed: Any, _count: int) -> list[dict[str, Any]]:
        return [{"id": "r1"}, {"id": "r2"}]

    monkeypatch.setattr(provider, "_queue_snapshot", fake_snapshot)
    monkeypatch.setattr(provider, "_parse_controls", lambda *_a, **_k: DJControls())
    monkeypatch.setattr(prov_module, "_fetch_radio_tracks", fake_radio)
    monkeypatch.setattr(
        prov_module,
        "build_queue_item",
        lambda _qid, track: SimpleNamespace(queue_item_id=track.name),
    )

    result = asyncio.run(provider.complete_mix("queue-1", count=2))

    assert result["seed"] == "Seed Song"
    assert result["added"] == ["Radio r1", "Radio r2"]
    # one boundary per adjacent pair: seed->r1, r1->r2
    assert len(result["transition_notes"]) == 2
    assert [note["to"] for note in result["transition_notes"]] == ["Radio r1", "Radio r2"]


def test_transition_report_rejects_media_objects() -> None:
    """
    Negative control: media objects still fail, so the guard has teeth.

    If a future refactor makes this pass, the fix above stopped being load-
    bearing and the flattening in complete_mix can be removed - or, worse,
    something is silently accepting both shapes and the contract drifted.
    """
    seed = {"name": "Seed"}
    with pytest.raises(AttributeError):
        _transition_report([seed, _track("Radio r1"), _track("Radio r2")], DJControls())


def test_transition_report_accepts_flattened_rows() -> None:
    """The flattened row shape complete_mix now builds is valid input."""
    rows: list[dict[str, Any]] = [
        {"name": "Seed"},
        {"name": "Radio r1", "artist": "Artist", "queue_item_id": "Radio r1", "analysis": None},
        {"name": "Radio r2", "artist": "Artist", "queue_item_id": "Radio r2", "analysis": None},
    ]
    notes = _transition_report(rows, DJControls())
    assert len(notes) == 2
    assert [note["to"] for note in notes] == ["Radio r1", "Radio r2"]
