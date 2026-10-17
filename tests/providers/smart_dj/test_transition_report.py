"""Tests for 0.0.25 batch 2: _transition_report + save_sorted_playlist."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import music_assistant.providers.smart_dj.provider as prov_module
from music_assistant.providers.smart_dj.engine import DJControls
from music_assistant.providers.smart_dj.provider import _transition_report

if TYPE_CHECKING:
    import pytest


def test_report_user_pin_style() -> None:
    """An explicit user style pins every boundary with source=user."""
    control = DJControls(transition_style="echo_out")
    ordered = [
        {"name": "A", "bpm": 120.0, "camelot": "8A", "instrumental": True},
        {"name": "B", "bpm": 126.0, "camelot": "8B", "instrumental": False},
    ]
    notes = _transition_report(ordered, control)
    assert len(notes) == 1
    note = notes[0]
    assert note["bpm_delta_pct"] == 5.0
    assert note["style"] == "echo_out"
    assert note["source"] == "user"
    assert note["both_vocal"] is False  # one side is instrumental


def test_report_genre_profile_wins_over_auto() -> None:
    """With style=auto, the track's genre profile fills the pin."""
    control = DJControls()
    ordered = [
        {"name": "A", "bpm": 120.0, "genre": "House"},
        {"name": "B", "bpm": 121.0, "genre": "House"},
    ]
    note = _transition_report(ordered, control)[0]
    assert note["style"] == "full_blend"
    assert note["strictness"] == "normal"
    assert note["source"] == "genre"


def test_report_engine_default_when_no_profile() -> None:
    """With style=auto and no genre match, the planner decides per pair."""
    control = DJControls()
    ordered = [
        {"name": "A", "bpm": 100.0, "genre": "Klezmer"},
        {"name": "B", "bpm": 110.0, "genre": "Klezmer"},
    ]
    note = _transition_report(ordered, control)[0]
    assert note["style"] == "auto"
    assert note["source"] == "engine"
    assert note["bpm_delta_pct"] == 10.0


def test_report_handles_missing_analysis() -> None:
    """Unanalyzed tracks produce None facts, never a crash."""
    control = DJControls()
    notes = _transition_report([{"name": "A"}, {"name": "B"}], control)
    assert notes[0]["bpm_delta_pct"] is None
    assert notes[0]["key_distance"] is None
    assert notes[0]["both_vocal"] is True  # unknown counts as vocal (safe side)


def test_report_vocal_collision_flagged() -> None:
    """Two vocal tracks flag the collision the renderer will have to dodge."""
    control = DJControls()
    ordered = [
        {"name": "A", "bpm": 120.0, "instrumental": False},
        {"name": "B", "bpm": 120.0, "instrumental": False},
    ]
    assert _transition_report(ordered, control)[0]["both_vocal"] is True


def test_report_zero_bpm_guard() -> None:
    """A zero bpm never divides by zero."""
    control = DJControls()
    ordered = [{"name": "A", "bpm": 0.0}, {"name": "B", "bpm": 120.0}]
    assert _transition_report(ordered, control)[0]["bpm_delta_pct"] is None


def test_report_handles_nested_analysis_rows() -> None:
    """
    Queue/playlist snapshot rows carry nested analysis; report merges it.

    Regression guard: the first version read bpm flat off the row, so real
    rank_queue/complete_mix boundaries reported bpm_delta_pct=None.
    """
    control = DJControls()
    ordered = [
        {
            "name": "A",
            "queue_item_id": "1",
            "analysis": {"bpm": 120.0, "camelot": "8A", "instrumental": True},
        },
        {
            "name": "B",
            "queue_item_id": "2",
            "analysis": {"bpm": 126.0, "camelot": "8B", "instrumental": False},
        },
    ]
    note = _transition_report(ordered, control)[0]
    assert note["bpm_delta_pct"] == 5.0
    assert note["key_distance"] is not None
    assert note["to"] == "B"


def _analysis_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Three analyzed rows plus their media stand-ins, out of bpm order."""
    rows: list[dict[str, Any]] = [
        {
            "queue_item_id": "mix-0",
            "name": "A",
            "item_id": "a",
            "analysis": {"bpm": 120.0, "energy": 0.5, "camelot": "8A", "instrumental": True},
        },
        {
            "queue_item_id": "mix-1",
            "name": "B",
            "item_id": "b",
            "analysis": {"bpm": 128.0, "energy": 0.7, "camelot": "8B", "instrumental": False},
        },
        {
            "queue_item_id": "mix-2",
            "name": "C",
            "item_id": "c",
            "analysis": {"bpm": 124.0, "energy": 0.6, "camelot": "9A", "instrumental": False},
        },
    ]
    media = {
        row["queue_item_id"]: SimpleNamespace(uri=f"uri-{row['queue_item_id']}") for row in rows
    }
    return rows, media


def test_save_sorted_playlist_writes_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """save_sorted_playlist removes all positions and re-adds in sorted order."""
    provider = prov_module.SmartDJProvider.__new__(prov_module.SmartDJProvider)
    rows, media = _analysis_rows()
    calls: dict[str, Any] = {}

    async def fake_analysis_rows(*_args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return rows, media

    class FakePlaylists:
        async def tracks(self, item_id: str, provider_arg: str) -> AsyncIterator[Any]:
            self.seen_item_id = item_id
            yield SimpleNamespace(item_id="a", provider="p")
            yield SimpleNamespace(item_id="b", provider="p")

        async def _handle_remove_playlist_tracks(
            self, pid: str, positions: tuple[int, ...], user_id: str | None
        ) -> None:
            calls["removed"] = positions

        async def _handle_add_playlist_tracks(
            self, pid: str, uris: list[str], user_id: str | None
        ) -> None:
            calls["added_uris"] = uris

    provider.mass = SimpleNamespace(music=SimpleNamespace(playlists=FakePlaylists()))  # type: ignore[assignment]
    monkeypatch.setattr(prov_module, "_playlist_analysis_rows", fake_analysis_rows)
    monkeypatch.setattr(provider, "_parse_controls", lambda *_args, **_kwargs: DJControls())

    result = asyncio.run(provider.save_sorted_playlist("spotify", "pl-1"))

    assert result["saved"] is True
    assert result["track_count"] == 3
    assert calls["removed"] == (0, 1, 2)
    assert sorted(calls["added_uris"]) == ["uri-mix-0", "uri-mix-1", "uri-mix-2"]
    assert len(result["levels"]) == 3
    assert all(level["energy"] is not None for level in result["levels"])
    assert len(result["transition_notes"]) == 2
